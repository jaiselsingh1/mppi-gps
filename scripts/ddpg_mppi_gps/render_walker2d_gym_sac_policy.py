from __future__ import annotations

import json
import sys
from pathlib import Path

import tyro

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ddpg_mppi_gps.train_walker2d_drqv2 import resolve_device, resolve_run_dir
from scripts.ddpg_mppi_gps.train_walker2d_gym_sac_baseline import (
    evaluate,
    load_actor_from_checkpoint,
)


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> tuple[Path, str]:
    if checkpoint == "best":
        return run_dir / "checkpoint_best.pt", "best"
    if checkpoint == "latest":
        return run_dir / "checkpoint_latest.pt", "latest"
    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path, path.stem


def main(
    run_name: str = "walker2d_gym_sac_baseline",
    runs_root: str = "runs",
    checkpoint: str = "best",
    out: str | None = None,
    steps: int = 1000,
    seed: int = 0,
    device: str = "auto",
    height: int = 540,
    width: int = 960,
    camera: str | None = "track",
    fps: int | None = None,
    render_every: int = 1,
) -> None:
    run_dir = resolve_run_dir(runs_root, run_name)
    checkpoint_path, checkpoint_tag = resolve_checkpoint(run_dir, checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint_path}")
    torch_device = resolve_device(device)
    actor, config = load_actor_from_checkpoint(checkpoint_path, torch_device)
    if camera in {"", "none", "None"}:
        camera = None

    output_path = Path(out).expanduser() if out is not None else run_dir / f"policy_{checkpoint_tag}_seed{seed}.mp4"
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()

    stats = evaluate(
        actor=actor,
        env_id=str(config.get("env_id", "Walker2d-v5")),
        device=torch_device,
        episodes=1,
        steps=steps,
        seed=seed,
        render_video_path=output_path,
        height=height,
        width=width,
        camera=camera,
        fps=fps,
        render_every=render_every,
    )
    record = {
        "type": "render",
        "checkpoint": str(checkpoint_path),
        "checkpoint_tag": checkpoint_tag,
        **stats,
    }
    metrics_path = run_dir / f"render_policy_{checkpoint_tag}_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(
        f"reward={stats['eval_mean_reward']:.1f} "
        f"len={stats['eval_mean_episode_len']:.0f} -> {output_path}"
    )


if __name__ == "__main__":
    tyro.cli(main)
