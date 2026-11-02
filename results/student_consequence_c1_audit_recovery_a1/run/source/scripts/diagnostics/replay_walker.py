"""Replay a recorded Walker2d episode with the current code and compare exactly.

Re-runs one reset/planner seed pair through the standard runner, compares the
new trace with a reference trace field by field, and checks that decisions
restored from the new trace reproduce their continuations. The reference file
is only read.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.runners import run_walker2d as runner
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import compare_fields, load_episode, restore_decision

_COMPARED = ("states", "warmstarts", "actions", "costs", "contacts", "contact_normal_forces")


def _restore_check(episode: dict, config: Path, task_config: Path, step: int, length: int) -> dict:
    env = Walker2d(**json.loads(task_config.read_text()))
    try:
        controller = MPPI(env, MPPIConfig(**json.loads(config.read_text())))
        restore_decision(env, controller, episode, step)
        actions, states = [], [env.get_state()]
        for _ in range(step, min(step + length, len(episode["actions"]))):
            action, _ = controller.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
            env.step(action)
            actions.append(action)
            states.append(env.get_state())
    finally:
        env.close()
    end = step + len(actions)
    report = compare_fields(
        {"actions": episode["actions"][step:end], "states": episode["states"][step : end + 1]},
        {"actions": actions, "states": states},
        ("actions", "states"),
    )
    return {
        "step": step,
        "steps_replayed": len(actions),
        "exact": all(field["exact"] for field in report.values()),
        "fields": report,
    }


def main(
    reference: Path,
    reference_episode: int,
    out_dir: Path,
    steps: int = 1000,
    config: Path = _REPO_ROOT / "configs/walker2d_best.json",
    task_config: Path = _REPO_ROOT / "configs/walker2d_task.json",
    restore_length: int = 20,
) -> None:
    """Replay reference_episode of a reference trace and write replay_report.json.

    Exits with status 1 after writing the report if any comparison is inexact.

    Args:
        reference: Runner trace (.npz) to reproduce; never modified.
        reference_episode: Episode index inside the reference trace.
        out_dir: Fresh output directory for the new trace and report.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"replay_walker: error: {out_dir} is not empty.")
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = load_episode(reference, reference_episode)
    traces = out_dir / "traces.npz"
    runner.main(
        config=config,
        task_config=task_config,
        episodes=1,
        steps=steps,
        seed=expected["reset_seed"],
        planner_seed=expected["planner_seed"],
        traces_output=traces,
        metrics_output=out_dir / "metrics.json",
        log_every=0,
    )
    replay = load_episode(traces, 0)

    comparison = compare_fields(expected, replay, _COMPARED)
    restore_steps = sorted(
        {0, len(replay["actions"]) // 2, max(0, len(replay["actions"]) - restore_length)}
    )
    restores = [
        _restore_check(replay, config, task_config, step, restore_length) for step in restore_steps
    ]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=_REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
    )
    report = {
        "reference": {
            "path": str(reference.resolve()),
            "episode": reference_episode,
            "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
            "reset_seed": expected["reset_seed"],
            "planner_seed": expected["planner_seed"],
            "steps": len(expected["actions"]),
        },
        "replay": {
            "trace": str(traces.resolve()),
            "steps": len(replay["actions"]),
            "git_commit": commit,
            "git_dirty": dirty,
        },
        "reference_trace_exact": all(field["exact"] for field in comparison.values()),
        "fields": comparison,
        "restore_checks": restores,
    }
    (out_dir / "replay_report.json").write_text(json.dumps(report, indent=2) + "\n")
    restore_exact = all(check["exact"] for check in restores)
    print(
        f"reference_trace_exact={report['reference_trace_exact']} "
        f"restore_exact={restore_exact} "
        f"steps={len(replay['actions'])}/{len(expected['actions'])}"
    )
    if not (report["reference_trace_exact"] and restore_exact):
        # The report is already saved; a mismatch must still fail the command.
        raise SystemExit(1)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
