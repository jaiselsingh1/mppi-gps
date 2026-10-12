"""Descriptive autopsy of a Walker2d MPPI fall from an exactly replayable trace.

Each decision in a window before the fall is restored exactly and re-planned
(the re-planned action must equal the recorded one). From that decision the
script records what the K sampled rollouts predicted, scores the executed
plan by an explicit rollout, and runs two independent probes from the same
state and nominal: more samples (K probe) and a longer horizon (H probe).
Probes use their own RNGs and never affect the replay. The H probe is only
reported: open-loop plans that long also predict falls during ordinary
walking, so its base rate is measured on a control trace instead.

"Survives" means no unhealthy state within the probe's horizon, not beyond
it, and a predicted fall is *a* fall within H, not necessarily the later fall
along the executed trajectory. The classification is a heuristic summary of
evidence and may be "mixed" or "unresolved"; the per-decision rows are the
primary output. Short-horizon causes are never attributed: every sampled
trajectory falling does not identify why.
"""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import load_episode, restore_decision


def _falls(env: Walker2d, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-trajectory fall flag and first-fall step (1-based; 0 if none)."""
    unhealthy = ~env._is_healthy(states[..., 2], states[..., 3])
    fell = unhealthy.any(axis=1)
    return fell, np.where(fell, np.argmax(unhealthy, axis=1) + 1, 0)


def _score_plan(env: Walker2d, state, warmstart, plan) -> dict:
    states, costs, _ = env.batch_rollout(state, plan[None], initial_warmstart=warmstart)
    fell, steps = _falls(env, states)
    return {"cost": float(costs[0]), "falls": bool(fell[0]), "steps_to_fall": int(steps[0])}


def _sample_summary(env: Walker2d, data: dict) -> dict:
    fell, _ = _falls(env, data["states"])
    costs, weights = data["costs"], data["weights"]
    best = int(np.argmin(costs))
    return {
        "fraction_samples_fall": float(fell.mean()),
        "weight_on_falling_samples": float(weights[fell].sum()),
        "best_sample_cost": float(costs[best]),
        "best_sample_falls": bool(fell[best]),
        "best_nonfalling_cost": float(costs[~fell].min()) if (~fell).any() else None,
        "best_falling_cost": float(costs[fell].min()) if fell.any() else None,
        "weighted_sample_cost": float(weights @ costs),
    }


def _probe(env, cfg, nominal, state, warmstart, seed) -> dict:
    probe = MPPI(env, cfg, seed=seed)
    snapshot = probe.snapshot()
    snapshot["U"] = nominal
    snapshot["initial_plan"] = False
    probe.restore(snapshot)
    probe.plan_step(state, initial_warmstart=warmstart)
    return {**_sample_summary(env, probe.get_rollout_data()),
            "plan": _score_plan(env, state, warmstart, probe.last_plan)}


def _h_probe_control(env, cfg, h_cfg, trace, stride, seed) -> dict:
    """H-probe fall rates at ordinary decisions of another replayable episode."""
    recorded = load_episode(trace, 0)
    controller = MPPI(env, cfg)
    rows = []
    for step in range(100, len(recorded["actions"]), stride):
        state = restore_decision(env, controller, recorded, step)
        warmstart = env.get_warmstart()
        action, _ = controller.plan_step(state, initial_warmstart=warmstart)
        if not np.array_equal(action, recorded["actions"][step]):
            raise RuntimeError(f"Control replay diverged at step {step}.")
        nominal = recorded["pre_plan_nominals"][step]
        padded = np.concatenate([nominal, np.repeat(nominal[-1:], h_cfg.H - cfg.H, axis=0)])
        probe = _probe(env, h_cfg, padded, state, warmstart, seed + step)
        rows.append({
            "step": step,
            "default_fraction_samples_fall": _sample_summary(env, controller.get_rollout_data())["fraction_samples_fall"],
            "h_probe_fraction_samples_fall": probe["fraction_samples_fall"],
            "h_probe_plan_falls": probe["plan"]["falls"],
        })
    return {
        "trace": str(trace.resolve()), "reset_seed": recorded["reset_seed"],
        "planner_seed": recorded["planner_seed"], "rows": rows,
        "mean_h_probe_fraction_samples_fall": float(np.mean([r["h_probe_fraction_samples_fall"] for r in rows])),
        "h_probe_plan_fall_rate": float(np.mean([r["h_probe_plan_falls"] for r in rows])),
        "mean_default_fraction_samples_fall": float(np.mean([r["default_fraction_samples_fall"] for r in rows])),
    }


def _classify(rows: list[dict]) -> dict:
    """Heuristic evidence categories; the rows remain the primary record."""
    visible = [row for row in rows if row["plan"]["falls"]]
    evidence = {}
    sampling = [row["step"] for row in rows
                if row["samples"]["fraction_samples_fall"] == 1.0
                and row["probe_k"]["fraction_samples_fall"] < 1.0]
    if sampling:
        evidence["sampling"] = (
            f"every default sample predicts a fall within H but the K probe finds a surviving one at steps {sampling}"
        )
    averaging = [row["step"] for row in rows
                 if row["plan"]["falls"] and not row["samples"]["best_sample_falls"]]
    if averaging:
        evidence["averaging"] = (
            f"the executed plan predicts a fall within H while the best sample does not at steps {averaging}"
        )
    objective = [row["step"] for row in rows
                 if row["samples"]["best_sample_falls"]
                 and row["samples"]["best_nonfalling_cost"] is not None]
    if objective:
        evidence["objective"] = (
            f"the lowest-cost sample predicts a fall within H although surviving samples exist at steps {objective}"
        )

    causal = [key for key in ("sampling", "averaging", "objective") if key in evidence]
    label = causal[0] if len(causal) == 1 else ("mixed" if causal else "unresolved")
    sealed = None
    for row in reversed(rows):
        if row["samples"]["fraction_samples_fall"] < 1.0 or row["probe_k"]["fraction_samples_fall"] < 1.0:
            break
        sealed = row["step"]
    return {"classification": label, "evidence": evidence,
            "first_step_plan_predicts_a_fall": visible[0]["step"] if visible else None,
            "no_tested_sampler_survives_from_step": sealed}


def main(
    trace: Path,
    out_dir: Path,
    episode: int = 0,
    window: int = 80,
    probe_samples: int = 1024,
    probe_horizon: int = 96,
    probe_seed: int = 900_000,
    config: Path = _REPO_ROOT / "configs/walker2d_best.json",
    task_config: Path = _REPO_ROOT / "configs/walker2d_task.json",
    control_trace: Path | None = None,
    control_stride: int = 50,
) -> None:
    """Autopsy the fall that ends an episode of a replayable runner trace.

    Args:
        trace: Trace written by run_walker2d with decision snapshots.
        out_dir: Fresh output directory for rows.jsonl and summary.json.
        window: Number of decisions before the falling action to examine.
        control_trace: Optional replayable trace of ordinary walking; the H probe
            runs at every control_stride-th decision from step 100 to give its
            base fall rate away from any fall.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"walker_fall_autopsy: error: {out_dir} is not empty.")
    recorded = load_episode(trace, episode)
    cfg = MPPIConfig(**json.loads(config.read_text()))
    env = Walker2d(**json.loads(task_config.read_text()))
    rows = []
    try:
        final_state = recorded["states"][-1]
        if env._is_healthy(final_state[2], final_state[3]):
            raise SystemExit("walker_fall_autopsy: error: the episode does not end in a fall.")
        falling_step = len(recorded["actions"]) - 1
        controller = MPPI(env, cfg)
        k_cfg = replace(cfg, K=probe_samples)
        h_cfg = replace(cfg, H=probe_horizon)
        for step in range(max(0, falling_step - window), falling_step + 1):
            state = restore_decision(env, controller, recorded, step)
            warmstart = env.get_warmstart()
            nominal = recorded["pre_plan_nominals"][step]
            action, info = controller.plan_step(state, initial_warmstart=warmstart)
            if not np.array_equal(action, recorded["actions"][step]):
                raise RuntimeError(f"Replay diverged from the recorded action at step {step}.")
            padded = np.concatenate([nominal, np.repeat(nominal[-1:], probe_horizon - cfg.H, axis=0)])
            rows.append({
                "step": step,
                "steps_before_fall": falling_step - step,
                "n_eff": info["n_eff"],
                "samples": _sample_summary(env, controller.get_rollout_data()),
                "plan": _score_plan(env, state, warmstart, controller.last_plan),
                "probe_k": _probe(env, k_cfg, nominal, state, warmstart, probe_seed + step),
                "probe_h": _probe(env, h_cfg, padded, state, warmstart, probe_seed + step),
            })
        control = None
        if control_trace is not None:
            control = _h_probe_control(env, cfg, h_cfg, control_trace, control_stride, probe_seed)
    finally:
        env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "rows.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    summary = {
        "trace": str(trace.resolve()), "episode": episode,
        "reset_seed": recorded["reset_seed"], "planner_seed": recorded["planner_seed"],
        "falling_action_index": falling_step, "episode_steps": len(recorded["actions"]),
        "window": window, "mppi_config": asdict(cfg),
        "probes": {"samples": probe_samples, "horizon": probe_horizon, "seed_base": probe_seed},
        "replay_exact": True,
        **_classify(rows),
        "h_probe_note": (
            "H-probe results are not used as evidence: compare their fall rates "
            "with h_probe_control, measured at ordinary decisions."
        ),
        "h_probe_control": control,
        "caveat": (
            "Heuristic evidence over one fall. 'Survives' means within the probe horizon only; "
            "a predicted fall is a fall within H, not necessarily the executed one; "
            "short-horizon causes are not attributed."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("classification", "first_step_plan_predicts_a_fall",
                       "no_tested_sampler_survives_from_step", "evidence")}, indent=2))


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
