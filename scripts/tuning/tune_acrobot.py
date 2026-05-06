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
STUDY_NAME = "acrobot_mppi_seeded_fast_search_v2"


class FixedConfig(NamedTuple):
    SEARCH_N_TRIALS = 32
    SEARCH_N_STARTUP_TRIALS = 8
    SEARCH_EVAL_STEPS = 750
    SEARCH_SEEDS = (0, 2, 4, 6, 8)
    VALIDATION_TOP_N = 8
    VALIDATION_EVAL_STEPS = 1000
    VALIDATION_SEEDS = tuple(range(10))
    VALIDATION_ANCHOR_PARAMS = (
        {"K": 256, "H": 256, "noise_sigma": 0.15, "lam": 0.001},
        {"K": 512, "H": 192, "noise_sigma": 0.10, "lam": 0.0003},
        {"K": 256, "H": 363, "noise_sigma": 0.028294654890294296, "lam": 0.0016039011379167672},
    )
    K_CHOICES = (256, 512)
    H_MIN = 128
    H_MAX = 384
    NOISE_SIGMA_MIN = 0.02
    NOISE_SIGMA_MAX = 0.30
    LAM_MIN = 0.0003
    LAM_MAX = 0.02
    HOLD_STEPS = 25
    MIN_HIT_SUCCESS_RATE = 0.8
    MIN_HOLD_SUCCESS_RATE = 0.6

    # Objective weights. A trial that cannot hit and hold the task should not
    # win just because its smooth cost is low. n_eff is logged as a diagnostic
    # only; fixed-temperature MPPI keeps lambda exactly as configured.
    COST_WEIGHT = 1.0
    HIT_FAILURE_PENALTY = 20_000.0
    HOLD_FAILURE_PENALTY = 50_000.0
    TIME_TO_HIT_WEIGHT = 500.0
    FINAL_DIST_WEIGHT = 100.0
    FINAL_QVEL_WEIGHT = 10.0


def _score_from_episode_stats(
    costs: list[float],
    hit_successes: list[bool],
    hold_successes: list[bool],
    times_to_hit: list[int],
    final_tip_dists: list[float],
    final_qvel_norms: list[float],
    n_eff_means: list[float],
    eval_steps: int,
    config: FixedConfig,
) -> tuple[float, dict]:
    mean_cost_per_step = float(np.mean(costs) / eval_steps)
    hit_success_rate = float(np.mean(hit_successes))
    hold_success_rate = float(np.mean(hold_successes))
    mean_time_to_hit = float(np.mean(times_to_hit))
    mean_final_tip_dist = float(np.mean(final_tip_dists))
    mean_final_qvel_norm = float(np.mean(final_qvel_norms))
    mean_n_eff = float(np.mean(n_eff_means))

    score = (
        config.COST_WEIGHT * mean_cost_per_step
        + config.HIT_FAILURE_PENALTY * (1.0 - hit_success_rate)
        + config.HOLD_FAILURE_PENALTY * (1.0 - hold_success_rate)
        + config.TIME_TO_HIT_WEIGHT * (mean_time_to_hit / eval_steps)
        + config.FINAL_DIST_WEIGHT * mean_final_tip_dist
        + config.FINAL_QVEL_WEIGHT * mean_final_qvel_norm
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
        "eval_steps": int(eval_steps),
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


def _param_key(params: dict) -> tuple:
    return (
        int(params["K"]),
        int(params["H"]),
        round(float(params["noise_sigma"]), 12),
        round(float(params["lam"]), 12),
    )


def _params_to_cfg(params: dict) -> MPPIConfig:
    return MPPIConfig(
        K=int(params["K"]),
        H=int(params["H"]),
        lam=float(params["lam"]),
        noise_sigma=float(params["noise_sigma"]),
    )


def _evaluate_mppi_config(
    cfg: MPPIConfig,
    eval_steps: int,
    seeds: tuple[int, ...],
    config: FixedConfig,
    trial: optuna.Trial | None = None,
) -> tuple[float, dict]:
    ep_costs: list[float] = []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []
    n_eff_means: list[float] = []

    env = Acrobot()
    controller = MPPI(env, cfg)
    for report_step, seed in enumerate(seeds):
        np.random.seed(seed)
        env.reset()
        controller.reset()

        state = env.get_state()
        ep_cost = 0.0
        first_success_t: int | None = None
        hold_count = 0
        max_hold_count = 0
        ep_n_eff: list[float] = []

        for t in range(eval_steps):
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
        times_to_hit.append(first_success_t if first_success_t is not None else eval_steps)
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
            eval_steps,
            config,
        )

        if trial is not None:
            trial.report(partial_score, step=report_step)
        if trial is not None and trial.should_prune():
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
        eval_steps,
        config,
    )
    metrics["params"] = {
        "K": cfg.K,
        "H": cfg.H,
        "noise_sigma": cfg.noise_sigma,
        "lam": cfg.lam,
    }
    metrics["seeds"] = [int(seed) for seed in seeds]
    return score, metrics


def objective(trial: optuna.Trial, config: FixedConfig) -> float:
    params = {
        "K": trial.suggest_categorical("K", config.K_CHOICES),
        "H": trial.suggest_int("H", config.H_MIN, config.H_MAX, log=True),
        "noise_sigma": trial.suggest_float(
            "noise_sigma",
            config.NOISE_SIGMA_MIN,
            config.NOISE_SIGMA_MAX,
            log=True,
        ),
        "lam": trial.suggest_float("lam", config.LAM_MIN, config.LAM_MAX, log=True),
    }
    score, metrics = _evaluate_mppi_config(
        _params_to_cfg(params),
        eval_steps=config.SEARCH_EVAL_STEPS,
        seeds=config.SEARCH_SEEDS,
        config=config,
        trial=trial,
    )
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
    return score


def _unique_top_trials(study: optuna.Study, limit: int) -> list[optuna.trial.FrozenTrial]:
    complete = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    complete.sort(key=lambda t: float(t.value))

    seen: set[tuple] = set()
    selected: list[optuna.trial.FrozenTrial] = []
    for trial in complete:
        key = _param_key(trial.params)
        if key in seen:
            continue
        seen.add(key)
        selected.append(trial)
        if len(selected) >= limit:
            break
    return selected


def _validation_candidates(
    study: optuna.Study,
    limit: int,
    anchor_params: tuple[dict, ...],
) -> list[tuple[dict, optuna.trial.FrozenTrial | None, str]]:
    """Top search trials plus fixed hand/baseline anchors.

    The anchors keep the final validation honest: even if a known-good manual
    config is not ranked in the cheap search phase, it still gets compared
    against the proposed Optuna candidates under the full seed protocol.
    """
    selected: list[tuple[dict, optuna.trial.FrozenTrial | None, str]] = []
    seen: set[tuple] = set()

    def add(params: dict, trial: optuna.trial.FrozenTrial | None, source: str) -> None:
        key = _param_key(params)
        if key in seen:
            return
        seen.add(key)
        selected.append((dict(params), trial, source))

    for trial in _unique_top_trials(study, limit):
        add(trial.params, trial, f"search_trial={trial.number}")
    for params in anchor_params:
        add(params, None, "anchor")

    return selected


def main():
    config = FixedConfig()
    STUDY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB_PATH}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=config.SEARCH_N_STARTUP_TRIALS,
            multivariate=True,
            seed=0,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.SEARCH_N_STARTUP_TRIALS,
            n_warmup_steps=1,
        ),
    )
    study.enqueue_trial(
        {"K": 256, "H": 256, "noise_sigma": 0.15, "lam": 0.001},
        skip_if_exists=True,
    )
    study.enqueue_trial(
        {"K": 512, "H": 192, "noise_sigma": 0.10, "lam": 0.0003},
        skip_if_exists=True,
    )
    study.enqueue_trial(
        {
            "K": 256,
            "H": 363,
            "noise_sigma": 0.028294654890294296,
            "lam": 0.0016039011379167672,
        },
        skip_if_exists=True,
    )
    config_objective = functools.partial(objective, config=config)
    study.optimize(config_objective, n_trials=config.SEARCH_N_TRIALS, show_progress_bar=True)

    print("\n=== Best fast-search trial ===")
    print("Best search params:", study.best_params)
    print("Best search score:", study.best_value)
    print("Best search metrics:", study.best_trial.user_attrs)

    validation_results: list[tuple[float, dict, optuna.trial.FrozenTrial | None]] = []
    candidates = _validation_candidates(
        study,
        limit=config.VALIDATION_TOP_N,
        anchor_params=config.VALIDATION_ANCHOR_PARAMS,
    )
    print(f"\n=== Final validation ({len(candidates)} candidates) ===")
    for rank, (params, trial, source) in enumerate(candidates, start=1):
        cfg = _params_to_cfg(params)
        score, metrics = _evaluate_mppi_config(
            cfg,
            eval_steps=config.VALIDATION_EVAL_STEPS,
            seeds=config.VALIDATION_SEEDS,
            config=config,
        )
        metrics["search_trial_number"] = int(trial.number) if trial is not None else None
        metrics["search_score"] = float(trial.value) if trial is not None else None
        metrics["validation_source"] = source
        validation_results.append((score, metrics, trial))
        print(
            f"candidate {rank}: validation_score={score:.3f} "
            f"{source} params={metrics['params']} "
            f"hit={metrics['hit_success_rate']:.2f} "
            f"hold={metrics['hold_success_rate']:.2f}"
        )

    if not validation_results:
        raise RuntimeError("No completed fast-search trials to validate.")

    validation_results.sort(key=lambda item: item[0])
    best_validation_score, best_validation_metrics, _ = validation_results[0]
    best = dict(best_validation_metrics["params"])

    print("\n=== Best validated candidate ===")
    print("Best params:", best)
    print("Best validation score:", best_validation_score)
    print("Best validation metrics:", best_validation_metrics)

    # Always save the best validated candidate for inspection.
    CANDIDATE_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    CANDIDATE_METRICS_PATH.write_text(json.dumps(best_validation_metrics, indent=2))
    print(f"Saved candidate params to {CANDIDATE_PARAMS_PATH}")
    print(f"Saved candidate metrics to {CANDIDATE_METRICS_PATH}")

    if not _passes_promotion_gate(best_validation_metrics, config):
        print(
            "Candidate did not pass promotion gate; not overwriting "
            f"{BEST_PARAMS_PATH}. Required hit >= {config.MIN_HIT_SUCCESS_RATE:.2f}, "
            f"hold >= {config.MIN_HOLD_SUCCESS_RATE:.2f}."
        )
        return

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PARAMS_PATH.write_text(json.dumps(best, indent=2))
    BEST_METRICS_PATH.write_text(json.dumps(best_validation_metrics, indent=2))
    print(f"Promoted params to {BEST_PARAMS_PATH}")
    print(f"Promoted metrics to {BEST_METRICS_PATH}")


if __name__ == "__main__":
    main()
