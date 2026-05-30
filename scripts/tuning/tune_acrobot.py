from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

ROOT = Path(__file__).resolve().parents[2]
BEST_PARAMS_PATH = ROOT / "configs" / "acrobot_best.json"
BEST_METRICS_PATH = ROOT / "configs" / "acrobot_best_metrics.json"

SEARCH_SPACES = {
    "cost": {
        "K": [256, 512],
        "H": {"low": 96, "high": 256, "step": 16},
        "noise_sigma": {"low": 0.05, "high": 0.8, "log": True},
        "lam": {"low": 3e-4, "high": 50.0, "log": True},
    },
    "reliability": {
        "K": [512, 1024],
        "H": {"low": 128, "high": 320, "step": 32},
        "noise_sigma": {"low": 0.05, "high": 0.8, "log": True},
        "lam": {"low": 3e-4, "high": 5.0, "log": True},
    },
}

RELIABILITY_GATE = {
    "hit_success_rate": 0.7,
    "hold_success_rate": 0.7,
}

KNOWN_CANDIDATES: list[dict[str, float | int]] = [
    {"K": 512, "H": 144, "noise_sigma": 0.18553280373949055, "lam": 0.01437687465331295},
    {"K": 512, "H": 192, "noise_sigma": 0.3, "lam": 0.15},
    {"K": 512, "H": 192, "noise_sigma": 0.1, "lam": 0.0003},
    {"K": 512, "H": 192, "noise_sigma": 0.3, "lam": 0.5},
    {"K": 512, "H": 192, "noise_sigma": 0.3, "lam": 1.0},
    {"K": 512, "H": 160, "noise_sigma": 0.0740854367242068, "lam": 0.01203646278944494},
    {"K": 512, "H": 144, "noise_sigma": 0.16839468869030633, "lam": 0.0004834143007523171},
    {"K": 512, "H": 160, "noise_sigma": 0.10587623642502389, "lam": 0.024993752698899238},
    {"K": 512, "H": 160, "noise_sigma": 0.23036837164650636, "lam": 0.0028117940626934345},
    {"K": 512, "H": 144, "noise_sigma": 0.09042519778817845, "lam": 0.01571342212648393},
    {"K": 512, "H": 144, "noise_sigma": 0.17864074394437565, "lam": 0.0049819999789946934},
    {"K": 512, "H": 144, "noise_sigma": 0.21735838623108658, "lam": 0.0016163781357284455},
]


@dataclass(frozen=True)
class EvalSettings:
    seeds: tuple[int, ...]
    steps: int
    hold_steps: int


@dataclass(frozen=True)
class RunSettings:
    study_name: str
    storage: str
    run_dir: str
    n_trials: int
    search: EvalSettings
    validation: EvalSettings
    validate_top_n: int
    sampler_seed: int
    n_startup_trials: int
    write_best: bool
    mode: str


def parse_seed_list(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("seed list must contain at least one integer")
    return seeds


def ensure_storage_parent(storage: str) -> None:
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return
    db_path = Path(storage[len(prefix):])
    if str(db_path) == ":memory:":
        return
    db_path.parent.mkdir(parents=True, exist_ok=True)


def make_config(params: dict[str, Any]) -> MPPIConfig:
    return MPPIConfig(
        K=int(params["K"]),
        H=int(params["H"]),
        noise_sigma=float(params["noise_sigma"]),
        lam=float(params["lam"]),
        use_is_correction=False,
    )


def sample_params(trial: optuna.Trial, mode: str) -> dict[str, Any]:
    search_space = SEARCH_SPACES[mode]
    return {
        "K": trial.suggest_categorical("K", search_space["K"]),
        "H": trial.suggest_int(
            "H",
            search_space["H"]["low"],
            search_space["H"]["high"],
            step=search_space["H"]["step"],
        ),
        "noise_sigma": trial.suggest_float(
            "noise_sigma",
            search_space["noise_sigma"]["low"],
            search_space["noise_sigma"]["high"],
            log=search_space["noise_sigma"]["log"],
        ),
        "lam": trial.suggest_float(
            "lam",
            search_space["lam"]["low"],
            search_space["lam"]["high"],
            log=search_space["lam"]["log"],
        ),
    }


def finite_mean(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.mean())


def finite_min(values: list[float]) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(arr.min())


def evaluate_config(
    params: dict[str, Any],
    settings: EvalSettings,
    trial: optuna.Trial | None = None,
) -> dict[str, Any]:
    env = Acrobot()
    controller = MPPI(env, make_config(params))

    episode_costs: list[float] = []
    episode_lengths: list[int] = []
    hit_successes: list[bool] = []
    hold_successes: list[bool] = []
    final_successes: list[bool] = []
    final_hold_successes: list[bool] = []
    times_to_hit: list[int] = []
    final_tip_dists: list[float] = []
    final_qvel_norms: list[float] = []
    n_eff_values: list[float] = []
    action_values: list[np.ndarray] = []
    action_jumps: list[float] = []

    try:
        for seed_idx, seed in enumerate(settings.seeds):
            np.random.seed(seed)
            env.reset()
            controller.reset()
            state = env.get_state()
            ep_cost = 0.0
            first_success_t: int | None = None
            hold_count = 0
            max_hold_count = 0
            prev_action: np.ndarray | None = None
            steps_taken = 0

            for t in range(settings.steps):
                action, info = controller.plan_step(state)
                _, cost, done, _ = env.step(action)
                state = env.get_state()
                ep_cost += float(cost)
                steps_taken += 1
                n_eff_values.append(float(info["n_eff"]))

                action_arr = np.asarray(action, dtype=float).reshape(-1)
                action_values.append(action_arr)
                if prev_action is not None:
                    action_jumps.append(float(np.linalg.norm(action_arr - prev_action)))
                prev_action = action_arr

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
            episode_costs.append(ep_cost)
            episode_lengths.append(steps_taken)
            hit_successes.append(first_success_t is not None)
            hold_successes.append(max_hold_count >= settings.hold_steps)
            final_successes.append(bool(final_metrics["success"]))
            final_hold_successes.append(hold_count >= settings.hold_steps)
            times_to_hit.append(first_success_t if first_success_t is not None else settings.steps)
            final_tip_dists.append(float(final_metrics["tip_dist"]))
            final_qvel_norms.append(float(final_metrics["qvel_norm"]))

            if trial is not None:
                total_steps = max(1, int(np.sum(episode_lengths)))
                trial.report(float(np.sum(episode_costs) / total_steps), step=seed_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    finally:
        env.close()

    total_steps = max(1, int(np.sum(episode_lengths)))
    actions = np.asarray(action_values, dtype=float) if action_values else np.zeros((0, 1))
    action_abs = np.abs(actions) if actions.size else np.asarray([], dtype=float)

    return {
        "params": normalized_params(params),
        "seeds": list(settings.seeds),
        "steps": int(settings.steps),
        "hold_steps": int(settings.hold_steps),
        "mean_cost_per_step": float(np.sum(episode_costs) / total_steps),
        "mean_episode_cost": float(np.mean(episode_costs)),
        "std_episode_cost": float(np.std(episode_costs)),
        "episode_costs": [float(x) for x in episode_costs],
        "episode_lengths": [int(x) for x in episode_lengths],
        "hit_success_rate": float(np.mean(hit_successes)),
        "hold_success_rate": float(np.mean(hold_successes)),
        "final_success_rate": float(np.mean(final_successes)),
        "final_hold_success_rate": float(np.mean(final_hold_successes)),
        "mean_time_to_hit": float(np.mean(times_to_hit)),
        "mean_final_tip_dist": float(np.mean(final_tip_dists)),
        "mean_final_qvel_norm": float(np.mean(final_qvel_norms)),
        "mean_n_eff": finite_mean(n_eff_values),
        "min_n_eff": finite_min(n_eff_values),
        "mean_action_jump": finite_mean(action_jumps),
        "max_action_jump": finite_min([-x for x in action_jumps]) * -1.0 if action_jumps else None,
        "mean_abs_action": float(action_abs.mean()) if action_abs.size else None,
        "max_abs_action": float(action_abs.max()) if action_abs.size else None,
    }


def normalized_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "K": int(params["K"]),
        "H": int(params["H"]),
        "noise_sigma": float(params["noise_sigma"]),
        "lam": float(params["lam"]),
        "use_is_correction": False,
    }


def reliability_objective_score(metrics: dict[str, Any]) -> float:
    return float(
        metrics["mean_cost_per_step"]
        + 1000.0 * (1.0 - metrics["hold_success_rate"])
        + 750.0 * (1.0 - metrics["hit_success_rate"])
        + 500.0 * (1.0 - metrics["final_hold_success_rate"])
        + 0.01 * metrics["mean_time_to_hit"]
    )


def objective_score(metrics: dict[str, Any], mode: str) -> float:
    if mode == "reliability":
        return reliability_objective_score(metrics)
    return float(metrics["mean_cost_per_step"])


def objective_factory(search_settings: EvalSettings, mode: str):
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, mode)
        metrics = evaluate_config(params, search_settings, trial=trial)
        score = objective_score(metrics, mode)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("objective_score", score)
        trial.set_user_attr("hit_success_rate", metrics["hit_success_rate"])
        trial.set_user_attr("hold_success_rate", metrics["hold_success_rate"])
        trial.set_user_attr("final_hold_success_rate", metrics["final_hold_success_rate"])
        trial.set_user_attr("mean_n_eff", metrics["mean_n_eff"])
        return score

    return objective


def enqueue_known_candidates(study: optuna.Study) -> None:
    if len(study.trials) > 0:
        return
    for candidate in KNOWN_CANDIDATES:
        study.enqueue_trial(candidate)


def trial_record(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": float(trial.value) if trial.value is not None else None,
        "params": normalized_params(trial.params) if trial.params else {},
        "user_attrs": trial.user_attrs,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def validate_top_trials(
    study: optuna.Study,
    settings: EvalSettings,
    top_n: int,
) -> list[dict[str, Any]]:
    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    complete_trials.sort(key=lambda t: float(t.value))
    validations = []
    for trial in complete_trials[:top_n]:
        metrics = evaluate_config(trial.params, settings)
        validations.append({
            "trial_number": int(trial.number),
            "search_value": float(trial.value),
            "search_metrics": trial.user_attrs.get("metrics"),
            "params": normalized_params(trial.params),
            "validation_metrics": metrics,
        })
        print(
            "validated trial={trial} search={search:.6f} validation={validation:.6f} "
            "hit={hit:.2f} hold={hold:.2f} final_hold={final_hold:.2f}".format(
                trial=trial.number,
                search=float(trial.value),
                validation=metrics["mean_cost_per_step"],
                hit=metrics["hit_success_rate"],
                hold=metrics["hold_success_rate"],
                final_hold=metrics["final_hold_success_rate"],
            ),
            flush=True,
        )
    return validations


def validation_sort_key(item: dict[str, Any], mode: str) -> tuple:
    metrics = item["validation_metrics"]
    if mode == "reliability":
        return (
            -metrics["hold_success_rate"],
            -metrics["hit_success_rate"],
            -metrics["final_hold_success_rate"],
            metrics["mean_cost_per_step"],
            metrics["mean_time_to_hit"],
        )
    return (metrics["mean_cost_per_step"],)


def selected_validation(validations: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not validations:
        raise RuntimeError("No completed Optuna trials were available for validation.")
    return min(validations, key=lambda item: validation_sort_key(item, mode))


def passes_reliability_gate(metrics: dict[str, Any]) -> bool:
    return (
        metrics["hit_success_rate"] >= RELIABILITY_GATE["hit_success_rate"]
        and metrics["hold_success_rate"] >= RELIABILITY_GATE["hold_success_rate"]
    )


def build_summary(
    settings: RunSettings,
    study: optuna.Study,
    validations: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = selected_validation(validations, settings.mode)
    selected_metrics = selected["validation_metrics"]
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_search = min(complete, key=lambda t: float(t.value)) if complete else None
    return {
        "study_name": settings.study_name,
        "storage": settings.storage,
        "run_dir": settings.run_dir,
        "objective": "reliability-first score over search seeds" if settings.mode == "reliability" else "minimize mean_cost_per_step over search seeds",
        "selection_rule": "hold_success_rate, hit_success_rate, final_hold_success_rate, cost, time_to_hit" if settings.mode == "reliability" else "mean_cost_per_step",
        "search_space": SEARCH_SPACES[settings.mode],
        "known_candidates": [normalized_params(c) for c in KNOWN_CANDIDATES],
        "reliability_gate": RELIABILITY_GATE,
        "passed_reliability_gate": passes_reliability_gate(selected_metrics),
        "settings": asdict(settings),
        "n_trials_requested": settings.n_trials,
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(complete),
        "best_search_trial": trial_record(best_search) if best_search is not None else None,
        "selected_trial_number": selected["trial_number"],
        "selected_params": selected["params"],
        "selected_validation_metrics": selected_metrics,
        "top_candidate_validations": validations,
    }


def write_best_configs(summary: dict[str, Any]) -> None:
    selected_params = summary["selected_params"]
    write_json(BEST_PARAMS_PATH, selected_params)
    write_json(BEST_METRICS_PATH, summary)
    print(f"wrote {BEST_PARAMS_PATH}", flush=True)
    print(f"wrote {BEST_METRICS_PATH}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retune Acrobot MPPI with Optuna.")
    parser.add_argument("--study-name", default="acrobot_mppi_bounded_retune_20260524")
    parser.add_argument("--storage", default="sqlite:///logs/acrobot_mppi_bounded_retune_20260524.db")
    parser.add_argument("--run-dir", default="runs/acrobot_mppi_bounded_retune_20260524")
    parser.add_argument("--n-trials", type=int, default=48)
    parser.add_argument("--search-seeds", type=parse_seed_list, default=parse_seed_list("0,1,2,3,4"))
    parser.add_argument("--validation-seeds", type=parse_seed_list, default=parse_seed_list("0,1,2,3,4,5,6,7,8,9"))
    parser.add_argument("--search-steps", type=int, default=600)
    parser.add_argument("--validation-steps", type=int, default=600)
    parser.add_argument("--hold-steps", type=int, default=25)
    parser.add_argument("--validate-top-n", type=int, default=8)
    parser.add_argument("--sampler-seed", type=int, default=0)
    parser.add_argument("--n-startup-trials", type=int, default=8)
    parser.add_argument("--write-best", action="store_true")
    parser.add_argument("--mode", choices=("cost", "reliability"), default="cost")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive")
    if args.search_steps <= 0 or args.validation_steps <= 0:
        raise ValueError("step counts must be positive")
    if args.validate_top_n <= 0:
        raise ValueError("--validate-top-n must be positive")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_storage_parent(args.storage)

    settings = RunSettings(
        study_name=args.study_name,
        storage=args.storage,
        run_dir=args.run_dir,
        n_trials=args.n_trials,
        search=EvalSettings(args.search_seeds, args.search_steps, args.hold_steps),
        validation=EvalSettings(args.validation_seeds, args.validation_steps, args.hold_steps),
        validate_top_n=args.validate_top_n,
        sampler_seed=args.sampler_seed,
        n_startup_trials=args.n_startup_trials,
        write_best=args.write_best,
        mode=args.mode,
    )

    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(1, min(args.n_startup_trials, args.n_trials)),
        n_warmup_steps=2,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    enqueue_known_candidates(study)

    write_json(run_dir / "run_settings.json", asdict(settings))
    print("starting study", args.study_name, flush=True)
    print("settings", json.dumps(asdict(settings), sort_keys=True), flush=True)

    study.optimize(
        objective_factory(settings.search, args.mode),
        n_trials=args.n_trials,
        show_progress_bar=not args.no_progress_bar,
    )

    records = [trial_record(trial) for trial in study.trials]
    write_jsonl(run_dir / "trials.jsonl", records)

    validations = validate_top_trials(study, settings.validation, args.validate_top_n)
    write_json(run_dir / "validation_results.json", validations)

    summary = build_summary(settings, study, validations)
    write_json(run_dir / "study_summary.json", summary)

    selected = summary["selected_validation_metrics"]
    print("\n=== Selected Acrobot MPPI config ===", flush=True)
    print(json.dumps(summary["selected_params"], indent=2, sort_keys=True), flush=True)
    print(
        "validation mean_cost_per_step={cost:.6f} hit={hit:.2f} hold={hold:.2f} final_hold={final_hold:.2f}".format(
            cost=selected["mean_cost_per_step"],
            hit=selected["hit_success_rate"],
            hold=selected["hold_success_rate"],
            final_hold=selected["final_hold_success_rate"],
        ),
        flush=True,
    )

    if args.write_best:
        if args.mode == "reliability" and not summary["passed_reliability_gate"]:
            print(
                "--write-best set, but selected validation did not pass reliability gate; configs left unchanged",
                flush=True,
            )
        else:
            write_best_configs(summary)
    else:
        print("--write-best not set; configs left unchanged", flush=True)


if __name__ == "__main__":
    main()
