"""Pure tensor/metadata checks; no actor forwards, model or simulator calls."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest

from src.gps.bidirectional.interpolation_assay import (
    DEFAULT, CHECKPOINTS, load_config, work_bound, interpolate_actor,
    assess_panels, assess_raw, choose_alpha, choose_raw_alpha, _safe_source, verify_inputs)


def test_fixed_work_bound_counts_both_modes_all_candidates_and_fresh_final():
    bound = work_bound()
    assert bound["qualification_controls"] == 8800
    assert bound["final_controls"] == 12800
    assert bound["total_controls"] == bound["actor_rows"] == 21600
    assert bound["model_controls"] == bound["adam_steps"] == 0


def test_only_fixed_ladder_and_source_arm(tmp_path):
    assert load_config() == DEFAULT
    copied = load_config(); copied["alphas"].reverse()
    assert load_config() == DEFAULT
    path = tmp_path/"config.json"; path.write_text(json.dumps(copied))
    with pytest.raises(ValueError): load_config(path)
    with pytest.raises(ValueError): work_bound(copied)
    assert [v[0] for v in CHECKPOINTS.values()] == [
        "precision_precision_parent", "precision_precision_child_raw", "precision_precision_child_guided"]


@pytest.mark.parametrize("alpha", [0., .0625, .125, .25, .5, 1.])
def test_all_parameters_interpolated_buffers_and_parent_rng_preserved_without_forward(alpha):
    import torch
    from src.gps.bidirectional.policy import FamilyGaussianActor, policy_hash
    parent = FamilyGaussianActor(hidden_dim=4, seed=11, initial_std=.01)
    child = FamilyGaussianActor.from_checkpoint(parent.policy_state())
    with torch.no_grad():
        for index, parameter in enumerate(child.parameters()):
            parameter.add_(.01*(index+1))
    child._generator.manual_seed(987)
    before = (policy_hash(parent), policy_hash(child))
    result = interpolate_actor(parent, child, alpha)
    assert (policy_hash(parent), policy_hash(child)) == before
    assert torch.equal(result._generator.get_state(), parent._generator.get_state())
    for name, parameter in result.named_parameters():
        a = dict(parent.named_parameters())[name]; b = dict(child.named_parameters())[name]
        expected = a if alpha == 0 else b if alpha == 1 else (a.double()+alpha*(b.double()-a.double())).to(a.dtype)
        assert torch.equal(parameter, expected)
    assert not torch.equal(dict(child.named_parameters())["log_std"], parent.log_std)
    for name, buffer in result.named_buffers():
        assert torch.equal(buffer, dict(parent.named_buffers())[name])
    if alpha == 0: assert policy_hash(result) == policy_hash(parent)
    if alpha == 1: assert policy_hash(result) == policy_hash(child)


def test_architecture_and_buffer_changes_fail_before_forward():
    import torch
    from src.gps.bidirectional.policy import FamilyGaussianActor
    parent = FamilyGaussianActor(hidden_dim=4)
    with pytest.raises(ValueError): interpolate_actor(parent, FamilyGaussianActor(hidden_dim=5), .5)
    child = FamilyGaussianActor.from_checkpoint(parent.policy_state())
    with torch.no_grad(): child.action_scale.add_(.01)
    with pytest.raises(ValueError, match="Buffer changed"): interpolate_actor(parent, child, .5)
    child = FamilyGaussianActor.from_checkpoint(parent.policy_state())
    child.log_std.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"): interpolate_actor(parent, child, .5)
    for value in (True, float("nan"), -.01, 1.01):
        with pytest.raises(ValueError): interpolate_actor(parent, parent, value)


def _rows(cost, *, captures=10, deterministic=True):
    return [SimpleNamespace(context_id=f"root:{i}", cost=cost, capture=i<captures,
                            complete=True, deterministic=deterministic) for i in range(10)]


def _panels(*, guided=1., parent=1.01, raw=1.01, captures=10, deterministic=True):
    return {"guided": _rows(guided, captures=captures, deterministic=deterministic),
            "parent": _rows(parent, deterministic=deterministic),
            "raw": _rows(raw, deterministic=deterministic)}


def test_qualification_requires_own_complete_improvement_both_modes_and_both_references():
    mean = _panels(); stochastic = _panels(deterministic=False)
    assert assess_panels(mean, stochastic)["passed"]
    assert not assess_panels(mean, _panels(guided=1.02, deterministic=False))["passed"]
    assert not assess_panels(_panels(raw=1.), stochastic)["passed"]
    assert not assess_panels(_panels(captures=9), stochastic)["passed"]
    broken = _panels(); broken["raw"][0].complete = False
    with pytest.raises(ValueError): assess_panels(broken, stochastic)
    broken = _panels(); broken["guided"][0].context_id = "unpaired"
    with pytest.raises(ValueError): assess_panels(broken, stochastic)
    with pytest.raises(ValueError): assess_panels(mean, mean)


def test_select_qualification_only_smallest_cost_tie_to_smaller_alpha():
    rows = [dict(alpha=a, criteria=dict(passed=False), guided_mean_complete_cost=1.) for a in DEFAULT["alphas"]]
    assert choose_alpha(rows) is None
    rows[1]["criteria"]["passed"] = rows[3]["criteria"]["passed"] = True
    assert choose_alpha(rows) == .125
    rows[3]["guided_mean_complete_cost"] = .999
    assert choose_alpha(rows) == .5
    with pytest.raises(ValueError): choose_alpha(rows[:-1])


def test_best_raw_is_independently_selected_and_cannot_be_evaded_by_same_alpha_control():
    mean = _panels(raw=1.005); stochastic = _panels(raw=1.005, deterministic=False)
    assert assess_raw(mean, stochastic)["passed"]
    mean["best_raw"] = _rows(.9995)
    stochastic["best_raw"] = _rows(.9995, deterministic=False)
    assert not assess_panels(mean, stochastic)["passed"]  # Guided beats same-alpha raw but loses best-raw.
    stochastic.pop("best_raw")
    with pytest.raises(ValueError): assess_panels(mean, stochastic)
    rows = [dict(alpha=a, raw_criteria=dict(passed=False), raw_mean_complete_cost=1.) for a in DEFAULT["alphas"]]
    assert choose_raw_alpha(rows) is None  # Parent fallback, independently executed in final.
    rows[0]["raw_criteria"]["passed"] = rows[4]["raw_criteria"]["passed"] = True
    assert choose_raw_alpha(rows) == .0625
    rows[4]["raw_mean_complete_cost"] = .999
    assert choose_raw_alpha(rows) == 1.
    with pytest.raises(ValueError): choose_raw_alpha(rows[:-1])


def test_input_verification_rejects_hash_drift_without_loading_checkpoint(tmp_path):
    import hashlib
    inputs = tmp_path/"inputs"; inputs.mkdir()
    raw = inputs/"metadata.json"; raw.write_text("closed metadata")
    value = dict(checkpoints={r: dict(sha256=d) for r, (_, d) in CHECKPOINTS.items()},
        files=[dict(file="metadata.json", sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), bytes=raw.stat().st_size)])
    (tmp_path/"input_provenance.json").write_text(json.dumps(value))
    assert verify_inputs(tmp_path) == value
    raw.write_text("drift")
    with pytest.raises(ValueError, match="changed"): verify_inputs(tmp_path)
    with pytest.raises(ValueError): _safe_source(inputs, "../metadata.json")
    link = inputs/"link"; link.symlink_to(raw)
    with pytest.raises(ValueError): _safe_source(inputs, "link")
