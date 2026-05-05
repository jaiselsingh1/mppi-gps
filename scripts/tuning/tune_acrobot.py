import json
from pathlib import Path

import numpy as np
import optuna
from typing import NamedTuple
import functools

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

BEST_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "acrobot_best.json"
BEST_METRICS_PATH = Path(__file__).resolve().parents[2] / "configs" / "acrobot_best_metrics.json"
CANDIDATE_PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "acrobot_candidate.json"
CANDIDATE_METRICS_PATH = Path(__file__).resolve().parents[2] / "configs" / "acrobot_candidate_metrics.json"
STUDY_DB_PATH = Path(__file__).resolve().parents[2] / "logs" / "acrobot_mppi_tuning.db"
STUDY_NAME = "acrobot_mppi_capture_v2"


class FixedConfig(NamedTuple):
    N_TRIALS = 120
    N_STARTUP_TRIALS = 24
    EVAL_STEPS = 1000
    N_SEEDS = 10
    K_CHOICES = (256, 512)
    HOLD_STEPS = 25
    MIN_HIT_SUCCESS_RATE = 0.8
    MIN_HOLD_SUCCESS_RATE = 0.6
    MIN_N_EFF = 8.0

    # Objective weights. A trial that cannot hit and hold the task should not
    # win just because its smooth cost is low.
    COST_WEIGHT = 1.0
    HIT_FAILURE_PENALTY = 20_000.0
    HOLD_FAILURE_PENALTY = 50_000.0
    TIME_TO_HIT_WEIGHT = 500.0
    FINAL_DIST_WEIGHT = 100.0
    FINAL_QVEL_WEIGHT = 10.0
    N_EFF_FAILURE_PENALTY = 25.0


def _score_from_episode_stats(
    costs: list[float],
    hit_successes: list[bool],
    hold_successes: list[bool],
    times_to_hit: list[int],
    final_tip_dists: list[float],
    final_qvel_norms: list[float],
    n_eff_means: list[float],
    config: FixedConfig,
) -> tuple[float, dict]:
    mean_cost_per_step = float(np.mean(costs) / config.EVAL_STEPS)
    hit_success_rate = float(np.mean(hit_successes))
    hold_success_rate = float(np.mean(hold_successes))
    mean_time_to_hit = float(np.mean(times_to_hit))
    mean_final_tip_dist = float(np.mean(final_tip_dists))
    mean_final_qvel_norm = float(np.mean(final_qvel_norms))
    mean_n_eff = float(np.mean(n_eff_means))
    n_eff_shortfall = max(0.0, (config.MIN_N_EFF - mean_n_eff) / config.MIN_N_EFF)

    score = (
        config.COST_WEIGHT * mean_cost_per_step
        + config.HIT_FAILURE_PENALTY * (1.0 - hit_success_rate)
        + config.HOLD_FAILURE_PENALTY * (1.0 - hold_success_rate)
        + config.TIME_TO_HIT_WEIGHT * (mean_time_to_hit / config.EVAL_STEPS)
        + config.FINAL_DIST_WEIGHT * mean_final_tip_dist
        + config.FINAL_QVEL_WEIGHT * mean_final_qvel_norm
        + config.N_EFF_FAILURE_PENALTY * n_eff_shortfall
    )
    metrics = {
        "score": float(score),
        "mean_cost_per_step": mean_cost_per_step,
        "hit_success_rate": hit_success_rate,
        "hold_success_rate": hold_success_rate,
        "mean_time_to_hit": mean_time_to_hit,
        "mean_final_tip_dist": mean_final_tip_dist,
        "mean_final_qvel_norm": mean_final_qvel_norm,
        "mean_n_eff": mean_n_eff,
        "n_eval_episodes": len(costs),
        "episode_hit_successes": [bool(x) for x in hit_successes],
        "episode_hold_successes": [bool(x) for x in hold_successes],
        "episode_times_to_hit": [int(x) for x in times_to_hit],
    }
    return float(score), metrics


def _passes_promotion_gate(metrics: dict, config: FixedConfig) -> bool:
    return (
        metrics.get("hit_success_rate", 0.0) >= config.MIN_HIT_SUCCESS_RATE
        and metrics.get("hold_success_rate", 0.0) >= config.MIN_HOLD_SUCCESS_RATE
    )

def objective(trial: optuna.Trial, config: FixedConfig) -> float:
    # MPPI hyperparameters
    K = trial.suggest_categorical("K", config.K_CHOICES)
    H = trial.suggest_int("H", 80, 512, log=True)
    noise_sigma = trial.suggest_float("noise_sigma", 0.08, 2.0, log=True)
    lam = trial.suggest_float("lam", 1.0, 10_000.0, log=True)
    adaptive_lam = trial.suggest_categorical("adaptive_lam", [False, True])
    n_eff_threshold = trial.suggest_categorical("n_eff_threshold", [8.0, 16.0, 32.0, 64.0, 128.0])

    cfg = MPPIConfig(
        K=K,
        H=H,
        lam=lam,
        noise_sigma=noise_sigma,
        adaptive_lam=adaptive_lam,
        n_eff_threshold=n_eff_threshold,
    )

    ep_costs: list[float] = []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []
    n_eff_means: list[float] = []

    env = Acrobot()
    controller = MPPI(env, cfg)
    for seed in range(config.N_SEEDS):
        np.random.seed(seed)
        env.reset()
        controller.reset()

        state = env.get_state()
        ep_cost = 0.0
        first_success_t: int | None = None
        hold_count = 0
        max_hold_count = 0
        ep_n_eff: list[float] = []

        for t in range(config.EVAL_STEPS):
            action, info = controller.plan_step(state)
            obs, cost, done, _ = env.step(action)
            state = env.get_state()
            ep_cost += cost
            ep_n_eff.append(info["n_eff"])

            metrics = env.task_metrics()
            if metrics["success"]:
                if first_success_t is None:
                    first_success_t = t
                hold_count += 1
            else:
                hold_count = 0
            max_hold_count = max(max_hold_count, hold_count)

            if done:
                break

        final_metrics = env.task_metrics()
        ep_costs.append(float(ep_cost))
        hit_successes.append(first_success_t is not None)
        hold_successes.append(max_hold_count >= config.HOLD_STEPS)
        times_to_hit.append(first_success_t if first_success_t is not None else config.EVAL_STEPS)
        final_tip_dists.append(final_metrics["tip_dist"])
        final_qvel_norms.append(final_metrics["qvel_norm"])
        n_eff_means.append(float(np.mean(ep_n_eff)))

        partial_score, _ = _score_from_episode_stats(
            ep_costs,
            hit_successes,
            hold_successes,
            times_to_hit,
            final_tip_dists,
            final_qvel_norms,
            n_eff_means,
            config,
        )

        trial.report(partial_score, step=seed)
        if trial.should_prune():
            env.close()
            raise optuna.TrialPruned()

    env.close()

    score, metrics = _score_from_episode_stats(
        ep_costs,
        hit_successes,
        hold_successes,
        times_to_hit,
        final_tip_dists,
        final_qvel_norms,
        n_eff_means,
        config,
    )
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
    return score


def main():
    config = FixedConfig()
    STUDY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB_PATH}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=config.N_STARTUP_TRIALS,
            multivariate=True,
            seed=0,
        ),
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(
                n_warmup_steps=5,
                n_startup_trials=config.N_STARTUP_TRIALS,
            ),
            patience=2,
        ),
    )
    config_objective = functools.partial(objective, config=config)
    study.optimize(config_objective, n_trials=config.N_TRIALS, show_progress_bar=True)

    print("\n=== Best trial ===")
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)
    print("Best metrics:", study.best_trial.user_attrs)

    # Always save the best candidate from this study for inspection.
    best = dict(study.best_params)
    CANDIDATE_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    CANDIDATE_METRICS_PATH.write_text(json.dumps(study.best_trial.user_attrs, indent=2))
    print(f"Saved candidate params to {CANDIDATE_PARAMS_PATH}")
    print(f"Saved candidate metrics to {CANDIDATE_METRICS_PATH}")

    if not _passes_promotion_gate(study.best_trial.user_attrs, config):
        print(
            "Candidate did not pass promotion gate; not overwriting "
            f"{BEST_PARAMS_PATH}. Required hit >= {config.MIN_HIT_SUCCESS_RATE:.2f}, "
            f"hold >= {config.MIN_HOLD_SUCCESS_RATE:.2f}."
        )
        return

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    BEST_METRICS_PATH.write_text(json.dumps(study.best_trial.user_attrs, indent=2))
    print(f"Promoted params to {BEST_PARAMS_PATH}")
    print(f"Promoted metrics to {BEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
