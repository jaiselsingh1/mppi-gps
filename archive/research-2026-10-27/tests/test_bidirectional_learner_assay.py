"""Pure metadata/tensor-state tests; no actor forwards or simulator calls."""
from copy import deepcopy
from types import SimpleNamespace
import json
import math
import pytest

from src.gps.bidirectional.learner_assay import (
    DEFAULT,load_config,work_bound,configure_arm,comparison_passes,_safe_source)


def test_precision_caps_count_both_modes_children_and_qualification():
    bound=work_bound()
    assert bound["total_controls"]==22800
    assert bound["executed_controls"]==22800
    assert bound["model_controls"]==0
    assert bound["actor_rows"]==77200
    assert bound["adam_steps"]==112
    assert bound["diagnostic_backward_passes"]==4


def test_only_one_declared_assay_configuration(tmp_path):
    value=deepcopy(DEFAULT);value["precision_initial_std"]=.02
    path=tmp_path/"changed.json";path.write_text(json.dumps(value))
    with pytest.raises(ValueError):load_config(path)
    with pytest.raises(ValueError):work_bound(value)
    assert load_config()==DEFAULT
    first=load_config();first["arms"].append("third")
    assert load_config()==DEFAULT


@pytest.mark.parametrize("arm",["original","precision"])
def test_arm_preserves_mean_optimizer_and_rng_without_forward(arm):
    import torch
    from src.gps.bidirectional.policy import FamilyGaussianActor,policy_hash
    from src.gps.bidirectional.learning import PolicyLearner,LearningConfig
    from src.gps.bidirectional.feedback import _digest
    parent=PolicyLearner(FamilyGaussianActor(8,4,2,[-1.],[1.]),LearningConfig(return_scale=25.),seed=7)
    # Existing Adam moments are part of the intervention contract, even for log_std.
    for parameter in parent.actor.parameters():
        parent.optimizer.state[parameter]={"step":torch.tensor(3.),
            "exp_avg":torch.full_like(parameter,.01),"exp_avg_sq":torch.full_like(parameter,.02)}
    before=parent.state_dict();digest=_digest(before)
    learner,metadata=configure_arm(parent,arm);after=learner.state_dict()
    assert _digest(parent.state_dict())==digest
    for key in ("optimizer","numpy_rng","torch_rng","updates"):
        assert _digest(after[key])==_digest(before[key])
    for key,value in before["policy"]["actor"].items():
        if key!="log_std":assert torch.equal(value,after["policy"]["actor"][key])
    assert learner.actor.log_std.requires_grad
    assert metadata["changed_parameters"]==([] if arm=="original" else ["log_std"])
    if arm=="precision":
        assert torch.allclose(learner.actor.log_std,torch.full_like(learner.actor.log_std,math.log(.01)))
        assert policy_hash(learner.actor)!=policy_hash(parent.actor)
    else:assert _digest(after)==digest


def _rows(cost,captures=10):
    return [SimpleNamespace(context_id=f"root:{i}",cost=cost,capture=i<captures) for i in range(10)]


def _panels(guided=1.,parent=1.01,raw=1.01,captures=10):
    return {"guided":_rows(guided,captures),"parent":_rows(parent),"raw":_rows(raw)}


def test_success_requires_both_modes_both_comparisons_and_capture():
    assert comparison_passes(_panels(),_panels())[0]
    assert not comparison_passes(_panels(),_panels(guided=1.02))[0]
    assert not comparison_passes(_panels(raw=1.),_panels())[0]
    assert not comparison_passes(_panels(captures=9),_panels())[0]  # capture regression even at90%
    with pytest.raises(ValueError):
        bad=_panels();bad["raw"][0].context_id="foreign";comparison_passes(bad,_panels())


def test_unsafe_or_missing_ancestor_rejected(tmp_path):
    with pytest.raises(ValueError):_safe_source(tmp_path,"../foreign")
    with pytest.raises(ValueError):_safe_source(tmp_path,"missing")
    target=tmp_path/"valid";target.write_text("exact")
    link=tmp_path/"link";link.symlink_to(target)
    with pytest.raises(ValueError):_safe_source(tmp_path,"link")
    assert _safe_source(tmp_path,"valid")==target
