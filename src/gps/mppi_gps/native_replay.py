"""Replay native Walker traces using library entry points and exact comparisons."""

import hashlib
import json
import subprocess
from functools import partial
from importlib import import_module
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUNTIME_IMPORTS = {
    "runner": ("src.gps.mppi_gps.native_runner", None),
    "Walker2d": ("src.envs.walker2d", "Walker2d"),
    "MPPI": ("src.mppi.mppi", "MPPI"),
    "MPPIConfig": ("src.utils.config", "MPPIConfig"),
    "compare_fields": ("src.utils.trace_replay", "compare_fields"),
    "load_episode": ("src.utils.trace_replay", "load_episode"),
    "restore_decision": ("src.utils.trace_replay", "restore_decision"),
}


def _dependency(name, overrides=None):
    if overrides is not None and name in overrides:
        return overrides[name]
    if name in globals():
        return globals()[name]
    module_name, attribute = _RUNTIME_IMPORTS[name]
    module = import_module(module_name)
    return module if attribute is None else getattr(module, attribute)


def __getattr__(name):
    if name in _RUNTIME_IMPORTS:
        return _dependency(name)
    raise AttributeError(name)


def _compatibility_binding(name, overrides):
    value = globals()[name]
    if overrides is not None and name in overrides and overrides[name] is not value:
        return overrides[name]
    return (
        partial(value, _overrides=overrides)
        if name == "_restore_check" and overrides is not None
        else value
    )


_COMPARED = ("states", "warmstarts", "actions", "costs", "contacts", "contact_normal_forces")


def _restore_check(
    episode: dict, config: Path, task_config: Path, step: int, length: int, *, _overrides=None
) -> dict:
    Walker2d = _dependency("Walker2d", _overrides)
    MPPI = _dependency("MPPI", _overrides)
    MPPIConfig = _dependency("MPPIConfig", _overrides)
    restore_decision = _dependency("restore_decision", _overrides)
    compare_fields = _dependency("compare_fields", _overrides)
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
    return _run(reference, reference_episode, out_dir, steps, config, task_config, restore_length)


def _run(
    reference: Path,
    reference_episode: int,
    out_dir: Path,
    steps: int = 1000,
    config: Path = _REPO_ROOT / "configs/walker2d_best.json",
    task_config: Path = _REPO_ROOT / "configs/walker2d_task.json",
    restore_length: int = 20,
    *,
    _overrides=None,
) -> None:
    """Replay reference_episode of a reference trace and write replay_report.json.

    Exits with status 1 after writing the report if any comparison is inexact.

    Args:
        reference: Runner trace (.npz) to reproduce; never modified.
        reference_episode: Episode index inside the reference trace.
        out_dir: Fresh output directory for the new trace and report.
    """
    runner = _dependency("runner", _overrides)
    _restore_check = _compatibility_binding("_restore_check", _overrides)
    _REPO_ROOT = _compatibility_binding("_REPO_ROOT", _overrides)
    _COMPARED = _compatibility_binding("_COMPARED", _overrides)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"replay_walker: error: {out_dir} is not empty.")
    out_dir.mkdir(parents=True, exist_ok=True)
    load_episode = _dependency("load_episode", _overrides)
    compare_fields = _dependency("compare_fields", _overrides)
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


def cli(argv=None) -> int:
    """Parse the historical replay options; help needs no simulator import."""
    import tyro

    tyro.cli(main, args=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
