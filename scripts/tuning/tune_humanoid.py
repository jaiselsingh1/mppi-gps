"""Small humanoid MPPI sweeps.

The first group keeps the MPPI update and dm_control-style humanoid reward
unchanged. Later groups are explicit ablations for extra stand shaping and
actuator clipping diagnostics.
"""

from __future__ import annotations

import json
import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.envs.humanoid import Humanoid
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


RESULTS_PATH = Path("runs/humanoid_sweep.json")


@dataclass
class HumanoidTrial:
    name: str
    target_speed: float = 0.0
    steps: int = 200
    seeds: int = 3
    K: int = 512
    H: int = 48
    lam: float = 1.0
    noise_sigma: float = 0.25
    noise_std: list[float] | None = None
    clip_actions: bool = False
    terminal_stand_weight: float = 48.0
    reward_weight: float = 1.0
    stand_weight: float = 0.0
    task_weight: float = 0.0
    lateral_weight: float = 0.0
    lateral_vel_weight: float = 0.0
    root_angvel_weight: float = 0.0
    posture_weight: float = 0.0
    qvel_weight: float = 0.0
    ctrl_weight: float = 0.0
    use_pd_nominal: bool = False
    pd_kp: float = 20.0
    pd_kd: float = 5.0
    tags: list[str] = field(default_factory=list)


def grouped_noise_std(
    env: Humanoid,
    *,
    abdomen: float,
    hip: float,
    knee: float,
    ankle: float,
    arm: float,
) -> list[float]:
    std = []
    for i in range(env.action_dim):
        name = env.model.actuator(i).name
        if name.startswith("abdomen"):
            std.append(abdomen)
        elif "hip" in name:
            std.append(hip)
        elif "knee" in name:
            std.append(knee)
        elif "ankle" in name:
            std.append(ankle)
        elif "shoulder" in name or "elbow" in name:
            std.append(arm)
        else:
            std.append(hip)
    return std


def evaluate_seed(trial: HumanoidTrial, seed: int) -> dict[str, float]:
    np.random.seed(seed)
    env = Humanoid(
        target_speed=trial.target_speed,
        terminal_stand_weight=trial.terminal_stand_weight,
        reward_weight=trial.reward_weight,
        stand_weight=trial.stand_weight,
        task_weight=trial.task_weight,
        lateral_weight=trial.lateral_weight,
        lateral_vel_weight=trial.lateral_vel_weight,
        root_angvel_weight=trial.root_angvel_weight,
        posture_weight=trial.posture_weight,
        qvel_weight=trial.qvel_weight,
        ctrl_weight=trial.ctrl_weight,
    )
    cfg = MPPIConfig(
        K=trial.K,
        H=trial.H,
        lam=trial.lam,
        noise_sigma=trial.noise_sigma,
        noise_std=trial.noise_std,
        clip_actions=trial.clip_actions,
    )
    controller = MPPI(env, cfg)
    env.reset()
    state = env.get_state()

    costs = []
    n_eff = []
    head_heights = []
    upright = []
    stand_rewards = []
    task_rewards = []
    small_controls = []
    forward_vx = []
    abs_actions = []
    x0 = env.task_metrics()["x_pos"]
    survived = trial.steps

    for t in range(trial.steps):
        nominal = None
        if trial.use_pd_nominal:
            nominal = np.tile(
                env.pose_pd_action(kp=trial.pd_kp, kd=trial.pd_kd),
                (trial.H, 1),
            )
        action, info = controller.plan_step(state, nominal=nominal)
        _, cost, _, _ = env.step(action)
        state = env.get_state()
        metrics = env.task_metrics()
        c = env.running_cost_components(state, action, env.data.sensordata.copy())

        costs.append(float(cost))
        n_eff.append(float(info["n_eff"]))
        head_heights.append(float(metrics["head_height"]))
        upright.append(float(metrics["upright"]))
        stand_rewards.append(float(c.standing_reward))
        task_rewards.append(float(c.task_reward))
        small_controls.append(float(c.small_control))
        forward_vx.append(float(metrics["forward_vx"]))
        abs_actions.append(float(np.max(np.abs(action))))

        if not metrics["healthy"]:
            survived = t + 1
            break

    final_metrics = env.task_metrics()
    env.close()

    return {
        "seed": seed,
        "survived_steps": float(survived),
        "healthy_final": float(final_metrics["healthy"]),
        "cost_sum": float(np.sum(costs)),
        "cost_mean": float(np.mean(costs)),
        "n_eff_mean": float(np.mean(n_eff)),
        "n_eff_min": float(np.min(n_eff)),
        "x_delta": float(final_metrics["x_pos"] - x0),
        "vx_mean": float(np.mean(forward_vx)),
        "vx_final": float(final_metrics["forward_vx"]),
        "head_min": float(np.min(head_heights)),
        "head_final": float(final_metrics["head_height"]),
        "upright_min": float(np.min(upright)),
        "upright_final": float(final_metrics["upright"]),
        "stand_mean": float(np.mean(stand_rewards)),
        "stand_min": float(np.min(stand_rewards)),
        "task_mean": float(np.mean(task_rewards)),
        "small_control_mean": float(np.mean(small_controls)),
        "action_absmax": float(np.max(abs_actions)),
    }


def summarize(trial: HumanoidTrial, runs: list[dict[str, float]]) -> dict[str, object]:
    keys = [k for k in runs[0].keys() if k != "seed"]
    summary = {}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=float)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))
    return {
        "trial": asdict(trial),
        "runs": runs,
        "summary": summary,
    }


def make_trials() -> list[HumanoidTrial]:
    probe_env = Humanoid()
    leg_noise = grouped_noise_std(
        probe_env,
        abdomen=0.14,
        hip=0.24,
        knee=0.28,
        ankle=0.18,
        arm=0.08,
    )
    quiet_leg_noise = grouped_noise_std(
        probe_env,
        abdomen=0.10,
        hip=0.18,
        knee=0.22,
        ankle=0.14,
        arm=0.05,
    )
    probe_env.close()

    return [
        HumanoidTrial(
            name="stand_strict_baseline",
            target_speed=0.0,
            steps=200,
            lam=1.0,
            noise_sigma=0.25,
            tags=["strict_mppi", "dm_control_reward"],
        ),
        HumanoidTrial(
            name="stand_strict_hotter",
            target_speed=0.0,
            steps=200,
            lam=5.0,
            noise_sigma=0.25,
            tags=["strict_mppi", "dm_control_reward"],
        ),
        HumanoidTrial(
            name="stand_strict_low_noise",
            target_speed=0.0,
            steps=200,
            lam=1.0,
            noise_sigma=0.15,
            tags=["strict_mppi", "dm_control_reward"],
        ),
        HumanoidTrial(
            name="stand_strict_big_k",
            target_speed=0.0,
            steps=200,
            K=1024,
            lam=1.0,
            noise_sigma=0.20,
            tags=["strict_mppi", "dm_control_reward"],
        ),
        HumanoidTrial(
            name="stand_strict_terminal96",
            target_speed=0.0,
            steps=200,
            lam=1.0,
            noise_sigma=0.25,
            terminal_stand_weight=96.0,
            tags=["strict_mppi", "dm_control_reward", "terminal_weight"],
        ),
        HumanoidTrial(
            name="stand_strict_long_horizon",
            target_speed=0.0,
            steps=200,
            H=72,
            lam=1.0,
            noise_sigma=0.20,
            terminal_stand_weight=96.0,
            tags=["strict_mppi", "dm_control_reward", "terminal_weight"],
        ),
        HumanoidTrial(
            name="stand_strict_diag_quiet",
            target_speed=0.0,
            steps=200,
            lam=2.0,
            noise_sigma=0.25,
            noise_std=quiet_leg_noise,
            tags=["strict_mppi", "dm_control_reward", "diag_covariance"],
        ),
        HumanoidTrial(
            name="stand_clip_baseline",
            target_speed=0.0,
            steps=200,
            lam=1.0,
            noise_sigma=0.25,
            clip_actions=True,
            tags=["clip_ablation", "dm_control_reward"],
        ),
        HumanoidTrial(
            name="stand_shaping",
            target_speed=0.0,
            steps=200,
            lam=1.0,
            noise_sigma=0.20,
            terminal_stand_weight=96.0,
            stand_weight=2.0,
            root_angvel_weight=0.01,
            posture_weight=0.001,
            qvel_weight=0.0002,
            ctrl_weight=0.001,
            tags=["cost_ablation"],
        ),
        HumanoidTrial(
            name="stand_pd_nominal",
            target_speed=0.0,
            steps=200,
            lam=1.0,
            noise_sigma=0.20,
            use_pd_nominal=True,
            pd_kp=20.0,
            pd_kd=5.0,
            tags=["nominal_ablation", "dm_control_reward"],
        ),
        HumanoidTrial(
            name="walk_strict_baseline",
            target_speed=1.0,
            steps=200,
            lam=1.0,
            noise_sigma=0.25,
            tags=["strict_mppi", "dm_control_reward"],
        ),
        HumanoidTrial(
            name="walk_strict_terminal96",
            target_speed=1.0,
            steps=200,
            lam=1.0,
            noise_sigma=0.25,
            terminal_stand_weight=96.0,
            tags=["strict_mppi", "dm_control_reward", "terminal_weight"],
        ),
        HumanoidTrial(
            name="walk_075_strict_terminal96",
            target_speed=0.75,
            steps=200,
            lam=1.0,
            noise_sigma=0.25,
            terminal_stand_weight=96.0,
            tags=["strict_mppi", "dm_control_reward", "terminal_weight"],
        ),
        HumanoidTrial(
            name="walk_075_strict_big_k",
            target_speed=0.75,
            steps=200,
            K=1024,
            lam=1.0,
            noise_sigma=0.20,
            terminal_stand_weight=96.0,
            tags=["strict_mppi", "dm_control_reward", "terminal_weight"],
        ),
        HumanoidTrial(
            name="walk_clip_terminal96",
            target_speed=1.0,
            steps=200,
            lam=1.0,
            noise_sigma=0.25,
            clip_actions=True,
            terminal_stand_weight=96.0,
            tags=["clip_ablation", "dm_control_reward", "terminal_weight"],
        ),
        HumanoidTrial(
            name="walk_075_clip_terminal96",
            target_speed=0.75,
            steps=200,
            lam=1.0,
            noise_sigma=0.25,
            clip_actions=True,
            terminal_stand_weight=96.0,
            tags=["clip_ablation", "dm_control_reward", "terminal_weight"],
        ),
        HumanoidTrial(
            name="walk_strict_hotter",
            target_speed=1.0,
            steps=200,
            lam=5.0,
            noise_sigma=0.25,
            tags=["strict_mppi", "dm_control_reward"],
        ),
        HumanoidTrial(
            name="walk_075_strict_long_horizon",
            target_speed=0.75,
            steps=200,
            H=72,
            lam=1.0,
            noise_sigma=0.20,
            terminal_stand_weight=96.0,
            tags=["strict_mppi", "dm_control_reward", "terminal_weight"],
        ),
        HumanoidTrial(
            name="walk_05_strict_long_horizon",
            target_speed=0.5,
            steps=200,
            H=72,
            lam=1.0,
            noise_sigma=0.20,
            terminal_stand_weight=96.0,
            tags=["strict_mppi", "dm_control_reward", "terminal_weight"],
        ),
        HumanoidTrial(
            name="walk_strict_diag_leg",
            target_speed=1.0,
            steps=200,
            lam=2.0,
            noise_sigma=0.25,
            noise_std=leg_noise,
            tags=["strict_mppi", "dm_control_reward", "diag_covariance"],
        ),
        HumanoidTrial(
            name="walk_slow_strict_diag_leg",
            target_speed=0.5,
            steps=200,
            lam=2.0,
            noise_sigma=0.25,
            noise_std=leg_noise,
            tags=["strict_mppi", "dm_control_reward", "diag_covariance"],
        ),
        HumanoidTrial(
            name="walk_stand_shaping",
            target_speed=1.0,
            steps=200,
            lam=2.0,
            noise_sigma=0.25,
            noise_std=leg_noise,
            terminal_stand_weight=96.0,
            stand_weight=2.0,
            root_angvel_weight=0.01,
            posture_weight=0.001,
            tags=["cost_ablation", "diag_covariance"],
        ),
        HumanoidTrial(
            name="walk_075_stand_shaping",
            target_speed=0.75,
            steps=200,
            lam=1.0,
            noise_sigma=0.20,
            terminal_stand_weight=96.0,
            stand_weight=2.0,
            root_angvel_weight=0.01,
            posture_weight=0.001,
            qvel_weight=0.0002,
            ctrl_weight=0.001,
            tags=["cost_ablation"],
        ),
        HumanoidTrial(
            name="walk_075_pd_nominal",
            target_speed=0.75,
            steps=200,
            lam=1.0,
            noise_sigma=0.20,
            terminal_stand_weight=96.0,
            use_pd_nominal=True,
            pd_kp=20.0,
            pd_kd=5.0,
            tags=["nominal_ablation", "dm_control_reward"],
        ),
        HumanoidTrial(
            name="walk_clip_diag_leg",
            target_speed=1.0,
            steps=200,
            lam=2.0,
            noise_sigma=0.25,
            noise_std=leg_noise,
            clip_actions=True,
            tags=["clip_ablation", "diag_covariance"],
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials = make_trials()
    if args.names is not None:
        names = set(args.names)
        trials = [trial for trial in trials if trial.name in names]
    if args.tags is not None:
        tags = set(args.tags)
        trials = [
            trial for trial in trials
            if any(tag in tags for tag in trial.tags)
        ]
    if args.steps is not None:
        for trial in trials:
            trial.steps = args.steps
    if args.seeds is not None:
        for trial in trials:
            trial.seeds = args.seeds

    results = []
    for trial in trials:
        print(f"\n=== {trial.name} ===", flush=True)
        runs = []
        for seed in range(trial.seeds):
            run = evaluate_seed(trial, seed)
            runs.append(run)
            print(
                "seed={seed} survived={survived_steps:.0f} healthy={healthy_final:.0f} "
                "cost={cost_mean:.3f} x={x_delta:.3f} vx={vx_mean:.3f} "
                "head_min={head_min:.3f} stand_min={stand_min:.3f} "
                "n_eff={n_eff_mean:.1f} amax={action_absmax:.2f}".format(**run),
                flush=True,
            )
        result = summarize(trial, runs)
        results.append(result)
        s = result["summary"]
        print(
            "summary survived={survived_steps_mean:.1f} "
            "healthy_rate={healthy_final_mean:.2f} cost={cost_mean_mean:.3f} "
            "x={x_delta_mean:.3f} vx={vx_mean_mean:.3f} "
            "head_min={head_min_mean:.3f} stand_min={stand_min_mean:.3f} "
            "n_eff={n_eff_mean_mean:.1f} amax={action_absmax_mean:.2f}".format(**s),
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
