"""S1.2-V validity block only: unchanged parents, reference controls and takeovers.

No fitting, reference labelling corpus or subsequent experiment is dispatched.
Imports remain free of policy construction and simulator imports until the
saved-only input/context gate has passed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .consequence_execution import ConsequenceMeter
from .consequence_native import load_npz
from .durability import durable_json
from .provenance import fingerprint, sha
from .visitation_native import NativeVisitation, same_context

METER_CONTRACT = "walker-visitation-v-only-meter-v1; lifecycle2048/frequent512"


def work_limits() -> dict[str, int]:
    return dict(
        native_controls=224000,
        physics_substeps=896000,
        neural_rows=224000,
        prediction_calls=224000,
        backward_rows=0,
        optimizer_steps=0,
        resets=32,
        restores=192,
        external_mj_forward_calls=256,
        snapshot_reads=448224,
        checkpoint_loads=4,
        policy_constructions=4,
        environment_constructions=1,
    )


class VisitationMeter(ConsequenceMeter):
    """Versioned V counters with the existing durable 2048/512-byte writer."""

    contract = METER_CONTRACT

    def __init__(self, out: Path, **kwargs: Any):
        supplied = kwargs.pop("limits", None)
        if supplied is not None and supplied != work_limits():
            raise ValueError("S1.2-V work ceilings must match the frozen V-only census")
        super().__init__(out, limits=work_limits(), **kwargs)

    def record(self, event: str, **fields: Any) -> str:
        from .runtime import BudgetExhausted

        if getattr(self, "_sequence", 0) >= 1_200_000:
            raise BudgetExhausted("S1.2-V journal record ceiling reached")
        return super().record(event, **fields)

    def save_arrays(self, tag: str, **arrays: Any) -> dict[str, Any]:
        from .runtime import BudgetExhausted

        if self._array_sequence >= 300:
            raise BudgetExhausted("S1.2-V array group ceiling reached")
        return super().save_arrays(tag, **arrays)


def _need(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def static_context_gate(prepared: Path, catalog: dict, config: dict) -> dict:
    """Compare all saved reset contexts; performs no inference, construction or physics."""
    validity = config["validity"]
    seeds = validity["positive_control_reference_reset_seeds"]
    _need(len(seeds) == len(set(seeds)) == 32, "S1.2-V needs all32 declared reference resets")
    _need(set(catalog["canonical_contexts"]) == {str(s) for s in seeds}, "Canonical seeds differ")
    _need(set(catalog["parent_initial_contexts"]) == {str(s) for s in seeds}, "Parent seeds differ")
    comparisons = []
    for seed in seeds:
        canonical_descriptor = catalog["canonical_contexts"][str(seed)]
        canonical = load_npz(prepared / canonical_descriptor["file"])
        _need(
            set(canonical)
            == {"state", "warmstart", "integration", "observation", "spec", "dimensions"},
            "Canonical context layout differs",
        )
        contexts = catalog["parent_initial_contexts"][str(seed)]
        _need(set(contexts) == {"0", "1", "2"}, "Every seed needs allthree retained parents")
        for stream in range(3):
            descriptor = contexts[str(stream)]
            context = load_npz(prepared / descriptor["file"])
            _need(set(context) == set(canonical), "Parent context layout differs")
            same_context(context, canonical, context["observation"])
            comparisons.append(
                dict(
                    seed=seed,
                    stream=stream,
                    file=descriptor["file"],
                    canonical_file=canonical_descriptor["file"],
                )
            )
    anchors = catalog["anchors"]
    _need(anchors == validity["anchors"] and len(anchors) == 96, "Frozen anchor roster differs")
    seen = set()
    for anchor in anchors:
        identity = (
            anchor["stream"],
            anchor["panel"],
            anchor["source_case"],
            anchor["anchor_ordinal"],
        )
        _need(identity not in seen, "Duplicate V anchor")
        seen.add(identity)
        _need(anchor["panel"] in {"primary", "confirmation"}, "Unexpected V panel")
        _need(anchor["stream"] in (0, 1, 2), "Unexpected V stream")
        _need(anchor["anchor_ordinal"] in (0, 1, 2, 3), "Unexpected V anchor ordinal")
        _need(anchor["source_tag"] in catalog["episodes"], "Missing complete source episode")
        _need(
            (anchor["control"] == 0) == (anchor["anchor_ordinal"] == 0),
            "Noninitial anchor must follow at least one actual parent action",
        )
    for panel in ("primary", "confirmation"):
        for stream in range(3):
            selected = [a for a in anchors if a["panel"] == panel and a["stream"] == stream]
            _need(
                len(selected) == 16 and len({a["source_case"] for a in selected}) == 4,
                "V source allocation differs",
            )
            for case in {a["source_case"] for a in selected}:
                _need(
                    {a["anchor_ordinal"] for a in selected if a["source_case"] == case}
                    == {0, 1, 2, 3},
                    "V source lacks one of its four frozen anchors",
                )
    return dict(
        schema="walker-visitation-static-context-gate-v1",
        passed=True,
        comparisons=comparisons,
        contexts=96,
        canonical_seeds=32,
        network_forwards=0,
        native_controls=0,
        policy_constructions=0,
    )


def _load_parent(run: Path, descriptor: dict, normalization: dict, meter: Any):
    """One checkpoint decode, full continuation binding, then one existing constructor."""
    import torch

    from .consequence_learning import _actor_hash
    from .policy import Policy

    path = run / "prepared" / descriptor["file"]
    _need(
        path.stat().st_size == descriptor["bytes"] and sha(path) == descriptor["sha256"],
        "Parent checkpoint whole-file binding differs",
    )
    payload = meter.measure(
        "checkpoint_loads",
        1,
        lambda: torch.load(path, map_location="cpu", weights_only=True),
        purpose="retained_parent_single_decode",
    )
    _need(
        set(payload)
        == {
            "schema",
            "seed",
            "obs_dim",
            "policy_hash",
            "state_dict",
            "private_rng_state",
            "training",
            "optimizer_state",
            "metadata",
        },
        "Parent checkpoint keys differ",
    )
    _need(
        payload["schema"] == "walker-tanh-policy-v1" and payload["obs_dim"] == 17,
        "Parent interface differs",
    )
    state, incoming = payload["state_dict"], payload["optimizer_state"]
    continuation = descriptor["continuation"]
    _need(
        set(incoming)
        == {
            "schema",
            "adam",
            "rng",
            "updates",
            "fit_updates",
            "policy_hash",
            "parent_policy_hash",
            "anchor_policy_hash",
            "anchor",
            "parameter_names",
            "fit_config",
            "fit_data",
        }
        and incoming["schema"] == "walker-adam-proximal-fit-v1",
        "Incomplete incoming Adam/RNG package",
    )
    _need(
        _actor_hash(state)
        == payload["policy_hash"]
        == descriptor["actor_hash"]
        == continuation["actor_hash"],
        "Parent tensor hash differs before construction",
    )
    _need(
        fingerprint(incoming) == continuation["optimizer_sha256"],
        "Complete incoming Adam/RNG fingerprint differs before construction",
    )
    _need(
        incoming["updates"] == continuation["incoming_updates"] == 2000
        and incoming["fit_updates"] == continuation["incoming_fit_updates"] == 2000,
        "Incoming update counters differ",
    )
    _need(
        fingerprint(payload["private_rng_state"]) == continuation["private_rng_sha256"]
        and fingerprint(incoming["rng"]) == continuation["pcg_sha256"],
        "Incoming private RNG differs",
    )
    global_rng = np.load(
        run / "prepared" / descriptor["global_cpu_sidecar"]["file"], allow_pickle=False
    )
    _need(
        fingerprint(global_rng) == continuation["global_rng_sha256"], "Global RNG sidecar differs"
    )
    for key in ("mean", "std"):
        actual = state["observation_" + key].detach().cpu().numpy()
        expected = np.asarray(normalization[key], dtype=np.float32)
        _need(
            actual.dtype == expected.dtype
            and actual.shape == (17,)
            and actual.tobytes() == expected.tobytes(),
            "Parent frozen normalization differs",
        )
    global_before = fingerprint(torch.get_rng_state())

    def construct():
        actor = Policy(
            seed=payload["seed"],
            device="cpu",
            normalization=normalization,
            obs_dim=17,
            hidden_dim=128,
        )
        actor.load_state_dict(state, strict=True)
        actor._generator.set_state(payload["private_rng_state"].cpu())
        actor.train(payload["training"])
        return actor

    actor = meter.measure(
        "policy_constructions", 1, construct, purpose="retained_parent_checked_payload"
    )
    _need(
        actor.hash() == descriptor["actor_hash"]
        and fingerprint(torch.get_rng_state()) == global_before,
        "Parent constructor changed actor/global RNG",
    )
    return (
        actor,
        incoming,
        dict(
            file=str(path.relative_to(run)),
            bytes=descriptor["bytes"],
            sha256=descriptor["sha256"],
            actor_hash=actor.hash(),
            optimizer_sha256=fingerprint(incoming),
            private_rng_sha256=fingerprint(actor._generator.get_state()),
            pcg_sha256=fingerprint(incoming["rng"]),
            continuation_verified_before_constructor=True,
            runtime_global_rng_sha256=global_before,
        ),
    )


def _load_reference(run: Path, catalog: dict, config: dict, meter: Any):
    import torch

    from .diagnostic_policies import LegacySACActor

    global_before = fingerprint(torch.get_rng_state())
    descriptor = catalog["reference"]
    path = run / "prepared" / descriptor["file"]
    _need(
        path.stat().st_size == descriptor["bytes"]
        and sha(path) == descriptor["sha256"] == config["reference"]["checkpoint_sha256"],
        "Reference checkpoint binding differs",
    )
    meter.authorize(dict(checkpoint_loads=1, policy_constructions=1))
    actor = meter.measure(
        "policy_constructions",
        1,
        lambda: LegacySACActor.from_checkpoint(path, meter=meter),
        purpose="frozen_reference_actor_only",
    )
    _need(actor.hash() == config["reference"]["actor_hash"], "Reference actor hash differs")
    _need(
        fingerprint(torch.get_rng_state()) == global_before,
        "Reference construction changed global RNG",
    )
    return actor, dict(
        file=str(path.relative_to(run)),
        bytes=descriptor["bytes"],
        sha256=descriptor["sha256"],
        actor_hash=actor.hash(),
        rule=LegacySACActor.execution_rule,
    )


def reference_screen(rows: list[dict]) -> dict:
    """Use only the72 noninitial comparisons; require the same two improving streams."""
    panels, regressions = {}, []
    for panel in ("primary", "confirmation"):
        streams, qualifying = {}, []
        for stream in range(3):
            counts = {}
            for stage, label in (("parent_replay", "parent"), ("reference_takeover", "reference")):
                selected = [
                    r
                    for r in rows
                    if r["stage"] == stage
                    and r["panel"] == panel
                    and r["stream"] == stream
                    and r["anchor_ordinal"] in (1, 2, 3)
                    and r["t"] > 0
                ]
                _need(len(selected) == 12, "Screen needs all12 noninitial anchors per stream/panel")
                identities = {(r["source_tag"], r["anchor_ordinal"], r["t"]) for r in selected}
                _need(len(identities) == 12, "Duplicate noninitial outcome")
                counts[label] = dict(
                    walking=sum(int(r["walking"]) for r in selected),
                    survived=sum(int(r["survived"]) for r in selected),
                )
                if label == "parent":
                    parent_identities = identities
                else:
                    _need(
                        identities == parent_identities, "Reference/parent anchors are not paired"
                    )
            gains = {
                key: counts["reference"][key] - counts["parent"][key]
                for key in ("walking", "survived")
            }
            if gains["walking"] >= 2:
                qualifying.append(stream)
            for key, gain in gains.items():
                if gain < 0:
                    regressions.append(dict(panel=panel, stream=stream, outcome=key, gain=gain))
            streams[str(stream)] = dict(**counts, gains=gains)
        panels[panel] = dict(streams=streams, qualifying_streams=qualifying)
    common = sorted(
        set(panels["primary"]["qualifying_streams"])
        & set(panels["confirmation"]["qualifying_streams"])
    )
    return dict(
        schema="walker-visitation-reference-screen-v1",
        passed=len(common) >= 2 and not regressions,
        panels=panels,
        same_qualifying_streams=common,
        regressions=regressions,
        noninitial_anchors=72,
        secondary_initial_anchors=24,
        permits="result_review_only",
        learning_launched=False,
    )


def runV(run: Path, config: dict, meter: Any, *, verified_catalog: dict | None = None) -> dict:
    """Run only the frozen validity roster, stopping before takeovers on failed prerequisites."""
    run = Path(run)
    worker = run / "worker"
    worker.mkdir(parents=True, exist_ok=True)
    if verified_catalog is None:
        from .visitation_inputs import verify_prepared

        catalog = verify_prepared(run / "prepared")
    else:
        # The bounded owner may verify inputs in its lightweight pre-Torch
        # phase, retaining that exact returned catalog rather than paying a
        # redundant hash pass under the smaller preparation RSS allowance.
        catalog = verified_catalog
    static = static_context_gate(run / "prepared", catalog, config)
    static["prepared_verification_accounting"] = catalog.get("_verification_accounting", {})
    durable_json(worker / "static_gate.json", static)
    normalization = load_npz(run / "prepared" / catalog["normalization"]["file"])
    actors, incoming, registry = {}, {}, {}
    for stream in range(3):
        alias = f"s{stream}_parent"
        actors[alias], incoming[alias], registry[alias] = _load_parent(
            run, catalog["parents"][str(stream)], normalization, meter
        )
    durable_json(worker / "parent_actor_registry.json", registry)
    native, rows = None, []
    result = dict(
        schema="walker-visitation-validity-result-v1",
        status="INCOMPLETE",
        prepared_verification_accounting=catalog.get("_verification_accounting", {}),
        phase_status=dict(
            static="passed",
            parent_replay="not_started",
            reference_positive_control="not_started",
            takeovers="not_started",
        ),
        learning_launched=False,
        followup_launched=False,
    )
    with (worker / "outcomes.jsonl").open("x") as stream:

        def retain(value: dict) -> None:
            encoded = json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
            meter.reserve_external(len(encoded.encode()))
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            rows.append(value)

        def anchor_branch(anchor: dict, actor: Any, alias: str, stage: str) -> dict:
            tag = f"{stage}_s{anchor['stream']}_{anchor['panel']}_{anchor['source_case']:02d}_{anchor['anchor_ordinal']}"
            value = native.closed_loop(
                tag,
                actor,
                actor_alias=alias,
                checkpoint_sha256=registry[alias]["sha256"],
                actor_hash=registry[alias]["actor_hash"],
                stage=stage,
                source=native.episode(anchor["source_tag"]),
                source_tag=anchor["source_tag"],
                t=anchor["control"],
                expected_replay=stage == "parent_replay",
            )
            value.update(
                stream=anchor["stream"],
                panel=anchor["panel"],
                source_case=anchor["source_case"],
                anchor_ordinal=anchor["anchor_ordinal"],
            )
            return value

        try:
            native = NativeVisitation(run, config, meter)
            result["phase_status"]["parent_replay"] = "in_progress"
            for anchor in catalog["anchors"]:
                alias = f"s{anchor['stream']}_parent"
                retain(anchor_branch(anchor, actors[alias], alias, "parent_replay"))
            expected_controls = sum(
                catalog["episodes"][a["source_tag"]]["controls"] - a["control"]
                for a in catalog["anchors"]
            )
            _need(
                expected_controls == config["expected_parent_replay_controls"] == 11426,
                "Frozen retained-parent source census differs",
            )
            _need(
                sum(r["controls"] for r in rows) == expected_controls,
                "Retained-parent replay control census differs",
            )
            result["phase_status"]["parent_replay"] = "passed"
            replay_gate = dict(
                schema="walker-visitation-parent-replay-gate-v1",
                passed=True,
                episodes=96,
                controls=expected_controls,
                rtol=0,
                atol=1e-10,
                saved_falls_included=True,
                actions_byte_exact=True,
            )
            durable_json(worker / "parent_replay_gate.json", replay_gate)
            actors["reference"], registry["reference"] = _load_reference(
                run, catalog, config, meter
            )
            durable_json(worker / "actor_registry.json", registry)
            seeds = config["validity"]["positive_control_reference_reset_seeds"]
            own = []
            result["phase_status"]["reference_positive_control"] = "in_progress"
            for ordinal, seed in enumerate(seeds):
                panel = "primary" if ordinal < 16 else "confirmation"
                value = native.closed_loop(
                    f"reference_reset_{panel}_{ordinal % 16:02d}",
                    actors["reference"],
                    actor_alias="reference",
                    checkpoint_sha256=registry["reference"]["sha256"],
                    actor_hash=registry["reference"]["actor_hash"],
                    stage="reference_reset",
                    seed=seed,
                    canonical=native.canonical(seed),
                )
                value.update(
                    stream=None, panel=panel, source_case=ordinal % 16, anchor_ordinal=None
                )
                retain(value)
                own.append(value)
            positive = dict(
                schema="walker-visitation-positive-control-gate-v1",
                panels={
                    panel: dict(
                        episodes=16,
                        walking=sum(int(r["walking"]) for r in own if r["panel"] == panel),
                        survived=sum(int(r["survived"]) for r in own if r["panel"] == panel),
                    )
                    for panel in ("primary", "confirmation")
                },
            )
            positive["passed"] = all(p["walking"] >= 15 for p in positive["panels"].values())
            durable_json(worker / "reference_positive_control_gate.json", positive)
            if not positive["passed"]:
                result["status"] = "STOPPED_REFERENCE_POSITIVE_CONTROL_FAILED"
                result["phase_status"]["reference_positive_control"] = "negative"
                result["stopped_before"] = "reference_takeovers"
                return result
            result["phase_status"]["reference_positive_control"] = "passed"
            result["phase_status"]["takeovers"] = "in_progress"
            for anchor in catalog["anchors"]:
                retain(
                    anchor_branch(anchor, actors["reference"], "reference", "reference_takeover")
                )
            screen = reference_screen(rows)
            durable_json(worker / "reference_usefulness_screen.json", screen)
            result.update(status="COMPLETED_V_FIXED_ROSTER", reference_screen=screen)
            result["phase_status"]["takeovers"] = "complete"
            return result
        except BaseException as error:
            result["status"] = "STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE"
            result["error"] = dict(type=type(error).__name__, message=str(error))
            result["stopped_during"] = next(
                (key for key, value in result["phase_status"].items() if value == "in_progress"),
                "construction",
            )
            raise
        finally:
            closure_error = None
            try:
                if native is not None:
                    native.env.close()
                for alias, state in incoming.items():
                    _need(
                        actors[alias].hash() == registry[alias]["actor_hash"]
                        and fingerprint(state) == registry[alias]["optimizer_sha256"]
                        and fingerprint(actors[alias]._generator.get_state())
                        == registry[alias]["private_rng_sha256"],
                        "V execution changed retained parent/continuation",
                    )
                    if "runtime_global_rng_sha256" in registry[alias]:
                        import torch

                        _need(
                            fingerprint(torch.get_rng_state())
                            == registry[alias]["runtime_global_rng_sha256"],
                            "V execution changed global Torch RNG",
                        )
                if "reference" in actors:
                    _need(
                        actors["reference"].hash() == registry["reference"]["actor_hash"],
                        "V execution changed the frozen reference",
                    )
            except BaseException as error:
                closure_error = error
                result["closure_error"] = dict(type=type(error).__name__, message=str(error))
                result["status"] = "STOPPED_EXECUTION_CLOSURE_FAILURE"
            if not (worker / "actor_registry.json").exists():
                durable_json(worker / "actor_registry.json", registry)
            durable_json(
                worker / "outcomes.json", dict(schema="walker-visitation-outcomes-v1", rows=rows)
            )
            durable_json(worker / "phase_status.json", result)
            if closure_error is not None:
                raise closure_error
