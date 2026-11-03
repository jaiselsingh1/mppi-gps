"""Frozen J1 scalar measurements and prospective prediction; no simulator or policy imports.

Scores consume already preserved arrays. Fitting an explanatory/prediction model
is development-only diagnostic arithmetic, never an update of the torque actor.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

VERSION = "walker-similarity-j1-metrics-v1"
SCORE_IDS = ("A", "B", "C", "D", "E")
OCCUPANCY_SEED = 2026129801
BOOTSTRAP_SEED = 2026129802


def _array(value: Any, *, columns: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Metric arrays must be finite")
    if columns is not None and (result.ndim != 2 or result.shape[1] != columns):
        raise ValueError(f"Expected a two-dimensional array with {columns} columns")
    return result


def _pair(left: Any, right: Any, columns: int = 6) -> tuple[np.ndarray, np.ndarray]:
    left, right = _array(left, columns=columns), _array(right, columns=columns)
    if left.shape != right.shape or not len(left):
        raise ValueError("Paired metric arrays need identical nonempty shapes")
    return left, right


def action_mse(actions: Any, predictions: Any) -> float:
    actions, predictions = _pair(actions, predictions)
    return float(np.mean((actions - predictions) ** 2, dtype=np.float64))


def smoothed_action_mse(actions: Any, predictions: Any, window: int = 5) -> float:
    actions, predictions = _pair(actions, predictions)
    if type(window) is not int or not 1 <= window <= len(actions):
        raise ValueError("Smoothing needs a positive window within the actual prefix")
    # Explicit means preserve the declared causal box arithmetic and valid support.
    left = np.stack(
        [np.mean(actions[t - window + 1 : t + 1], axis=0) for t in range(window - 1, len(actions))]
    )
    right = np.stack(
        [
            np.mean(predictions[t - window + 1 : t + 1], axis=0)
            for t in range(window - 1, len(predictions))
        ]
    )
    filtered = left - right
    return float(np.mean(filtered**2, dtype=np.float64))


def fourier_action_mse(actions: Any, predictions: Any) -> float:
    actions, predictions = _pair(actions, predictions)
    transformed = np.fft.fft(actions - predictions, axis=0, norm="ortho")
    return float(np.mean(np.abs(transformed) ** 2))


def deterministic_action_law_mse(samples: Any, deterministic: Any) -> float:
    """Action-dimension-normalized W2² to a point mass: exactly expected MSE."""
    values = _array(samples, columns=6)
    point = _array(deterministic)
    if not len(values) or point.shape != (6,):
        raise ValueError("A nonempty action law and one six-dimensional point are required")
    return float(np.mean((values - point) ** 2))


def normalized_student_features(observations: Any, normalization: dict) -> np.ndarray:
    observations = _array(observations, columns=17).astype(np.float32)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    if (
        mean.shape != (17,)
        or std.shape != (17,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("The original finite 17-input normalization is required")
    result = ((observations - mean) / std).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Normalized student features must be finite")
    return result


def fit_state_explainer(
    observations: Any,
    actions: Any,
    normalization: dict,
    *,
    split: str = "development",
    group_ids: Sequence[Any] = (),
) -> dict:
    if split != "development":
        raise ValueError("Explanatory fits are development-only")
    features = normalized_student_features(observations, normalization)
    actions = _array(actions, columns=6)
    if len(features) != len(actions) or not len(features):
        raise ValueError("Explanatory rows must be paired and nonempty")
    design = np.column_stack((np.ones(len(features)), features))
    penalty = np.eye(18) * 1e-3
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ actions)
    singular = np.linalg.svd(design, compute_uv=False)
    return dict(
        version=VERSION,
        fit_split=split,
        development_groups=sorted(set(map(str, group_ids))),
        ridge=1e-3,
        coefficients=coefficients.tolist(),
        normalization={k: np.asarray(normalization[k]).tolist() for k in ("mean", "std")},
        rank=int(np.linalg.matrix_rank(design)),
        singular_values=singular.tolist(),
        feature_min=features.min(axis=0).tolist(),
        feature_max=features.max(axis=0).tolist(),
        rows=len(features),
        residual_mse=float(np.mean((actions - design @ coefficients) ** 2)),
    )


def state_explained_score(
    observations: Any, actions: Any, predictions: Any, explainer: dict
) -> dict:
    if explainer.get("fit_split") != "development" or explainer.get("ridge") != 1e-3:
        raise ValueError("The frozen development explanatory map is required")
    actions, predictions = _pair(actions, predictions)
    features = normalized_student_features(observations, explainer["normalization"])
    if len(features) != len(actions):
        raise ValueError("Explanatory/teacher/policy rows differ")
    fitted = np.column_stack((np.ones(len(features)), features)) @ _array(explainer["coefficients"])
    explained = action_mse(fitted, predictions)
    residual = action_mse(actions, fitted)
    outside = np.any(
        (features < _array(explainer["feature_min"]))
        | (features > _array(explainer["feature_max"])),
        axis=1,
    )
    return dict(
        score=explained + 2 * residual,
        explained_mse=explained,
        residual_mse=residual,
        residual_weight=2,
        outside_development_box_fraction=float(outside.mean()),
        explanation="Residual includes misspecification and support shift; it is not identified noise.",
    )


def _trace(trace: dict, horizon: int) -> tuple[np.ndarray, np.ndarray, int]:
    states = _array(trace["states"], columns=19)
    done = np.asarray(trace["done"])
    if (
        done.ndim != 1
        or done.dtype.kind != "b"
        or len(states) != len(done) + 1
        or not 0 < len(done) <= horizon
    ):
        raise ValueError("Actual N+1 states and N boolean terminal flags are required")
    if np.any(done[:-1]):
        raise ValueError("A trace cannot contain fabricated controls after a fall")
    return states, done, len(done)


def motion_metrics(teacher: dict, student: dict, horizon: int = 1000) -> dict:
    if type(horizon) is not int or horizon < 1:
        raise ValueError("A positive fixed horizon is required")
    ts, td, tn = _trace(teacher, horizon)
    ss, sd, sn = _trace(student, horizon)
    common = min(tn, sn)
    if not np.allclose(
        ts[: common + 1, 0] - ts[0, 0], ss[: common + 1, 0] - ss[0, 0], rtol=0, atol=1e-10
    ):
        raise ValueError("Motion errors require synchronized actual timing")
    t, s = ts[1 : common + 1], ss[1 : common + 1]
    progress = (t[:, 1] - ts[0, 1]) - (s[:, 1] - ss[0, 1])
    height = t[:, 2] - s[:, 2]
    angles = (t[:, 3:10] - s[:, 3:10] + np.pi) % (2 * np.pi) - np.pi
    velocities = t[:, 10:19] - s[:, 10:19]
    components = np.column_stack(
        (
            progress**2,
            (height / 0.25) ** 2,
            np.mean((angles / 0.5) ** 2, axis=1),
            np.mean((velocities / 3) ** 2, axis=1),
        )
    )
    missing = (horizon - common) / horizon
    error = float(
        np.sqrt(np.sum(np.minimum(components, 9), dtype=np.float64) / (4 * horizon) + 4 * missing)
    )
    full = tn == sn == horizon and not td.any() and not sd.any()
    joint_rms = float(np.sqrt(np.mean(angles**2)))
    vx_rms = float(np.sqrt(np.mean(velocities[:, 0] ** 2)))
    height_rms = float(np.sqrt(np.mean(height**2)))
    progress_final = float(abs(progress[-1])) if full else None
    reproduction = bool(
        full and joint_rms <= 0.25 and vx_rms <= 0.3 and height_rms <= 0.1 and progress_final <= 0.5
    )
    return dict(
        error=error,
        reproduced=reproduction,
        full_window=bool(full),
        common_controls=common,
        teacher_controls=tn,
        student_controls=sn,
        missing_fraction=missing,
        missing_penalty_inside_sqrt=4 * missing,
        joint_angle_rms=joint_rms,
        actuated_angle_rms=float(np.sqrt(np.mean(angles[:, 1:] ** 2))),
        forward_velocity_rms=vx_rms,
        height_rms=height_rms,
        final_relative_progress_error=progress_final,
        uncapped_component_means=np.mean(components, axis=0).tolist(),
        capped_component_means=np.mean(np.minimum(components, 9), axis=0).tolist(),
        errors_scope="actual synchronized common prefix; missing fraction is explicit, not fabricated states",
    )


def fit_occupancy_normalizer(
    pairs: Any,
    *,
    split: str = "development",
    group_ids: Sequence[Any] = (),
    seed: int = OCCUPANCY_SEED,
) -> dict:
    if split != "development" or seed != OCCUPANCY_SEED:
        raise ValueError("Occupancy scales/directions are frozen development-only")
    pairs = _array(pairs, columns=23)
    if not len(pairs):
        raise ValueError("Development occupancy rows are missing")
    directions = np.random.Generator(np.random.PCG64(seed)).normal(size=(32, 23))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return dict(
        version=VERSION,
        fit_split=split,
        development_groups=sorted(set(map(str, group_ids))),
        mean=pairs.mean(axis=0).tolist(),
        std=np.maximum(pairs.std(axis=0), 0.05).tolist(),
        directions=directions.tolist(),
        seed=seed,
        std_floor=0.05,
        rows=len(pairs),
    )


def sliced_occupancy_distance(
    teacher_observations: Any,
    teacher_actions: Any,
    student_observations: Any,
    student_actions: Any,
    normalizer: dict,
    horizon: int = 1000,
) -> dict:
    if normalizer.get("fit_split") != "development" or normalizer.get("seed") != OCCUPANCY_SEED:
        raise ValueError("Frozen development occupancy preprocessing is required")
    to, so = _array(teacher_observations, columns=17), _array(student_observations, columns=17)
    ta, sa = _array(teacher_actions, columns=6), _array(student_actions, columns=6)
    if len(to) != len(ta) or len(so) != len(sa) or not 0 < len(so) <= horizon or len(to) != horizon:
        raise ValueError("Occupancy uses actual paired rows and a complete teacher")
    mean, std = _array(normalizer["mean"]), _array(normalizer["std"])
    directions = _array(normalizer["directions"], columns=23)
    if (
        mean.shape != (23,)
        or std.shape != (23,)
        or np.any(std < 0.05)
        or directions.shape != (32, 23)
        or not np.allclose(np.linalg.norm(directions, axis=1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("Occupancy preprocessing layout differs")
    q = (np.arange(1000) + 0.5) / 1000
    left = ((np.column_stack((to, ta)) - mean) / std) @ directions.T
    right = ((np.column_stack((so, sa)) - mean) / std) @ directions.T
    lq, rq = (
        np.quantile(left, q, axis=0, method="linear"),
        np.quantile(right, q, axis=0, method="linear"),
    )
    distance = float(np.mean((lq - rq) ** 2))
    missing = (horizon - len(so)) / horizon
    return dict(
        score=distance + 4 * missing,
        sliced_w2_squared=distance,
        missing_fraction=missing,
        cemetery_mass_penalty=4 * missing,
        directions=32,
        quantiles=1000,
        order_invariant=True,
    )


def score_case(
    teacher: dict,
    incoming: dict,
    predictions_on_teacher: Any,
    explainer: dict,
    occupancy_normalizer: dict,
) -> dict:
    if len(teacher["actions"]) != 1000:
        raise ValueError("A J1 corpus must contain 1000 unique executed controls")
    c = state_explained_score(
        teacher["observations"], teacher["actions"], predictions_on_teacher, explainer
    )
    d = motion_metrics(teacher, incoming)
    e = sliced_occupancy_distance(
        teacher["observations"],
        teacher["actions"],
        incoming["observations"],
        incoming["actions"],
        occupancy_normalizer,
    )
    return dict(
        version=VERSION,
        scores=dict(
            A=action_mse(teacher["actions"], predictions_on_teacher),
            B=smoothed_action_mse(teacher["actions"], predictions_on_teacher),
            C=c["score"],
            D=d["error"],
            E=e["score"],
        ),
        components=dict(C=c, D=d, E=e),
        corpus_aggregation="one complete trajectory per case; otherwise trajectory-equal mean required",
    )


def fit_local_feedback(
    delta_observations: Any,
    delta_actions: Any,
    *,
    held_observations: Any | None = None,
    held_actions: Any | None = None,
    contact_changed: Any | None = None,
) -> dict:
    x, y = _array(delta_observations, columns=17), _array(delta_actions, columns=6)
    if len(x) != len(y) or not len(x):
        raise ValueError("Gain perturbations must be nonempty and paired")
    coefficients, _, rank, singular = np.linalg.lstsq(x, y, rcond=1e-10)
    result = dict(
        gain=coefficients.T.tolist(),
        rank=int(rank),
        singular_values=singular.tolist(),
        condition_number=float(singular[0] / singular[-1]) if rank == 17 else None,
        uniquely_identified=bool(rank == 17),
        training_rmse=float(np.sqrt(np.mean((y - x @ coefficients) ** 2))),
        contact_regime_changes=None
        if contact_changed is None
        else int(np.asarray(contact_changed, dtype=bool).sum()),
        optimal_gain_claim=False,
    )
    if (held_observations is None) != (held_actions is None):
        raise ValueError("Held perturbation inputs and outputs must be paired")
    if held_observations is not None:
        hx, hy = _array(held_observations, columns=17), _array(held_actions, columns=6)
        if not len(hx) or len(hx) != len(hy):
            raise ValueError("Held perturbations must be nonempty and paired")
        result["held_rmse"] = float(np.sqrt(np.mean((hy - hx @ coefficients) ** 2)))
    return result


def stride_segments(contact: Any, *, initial_contact: bool | None = None) -> dict:
    values = np.asarray(contact)
    if values.ndim != 1 or values.dtype.kind != "b" or not len(values):
        raise ValueError("Raw boolean contacts are required; missing sensors are not absence")
    onsets = (np.flatnonzero(values[1:] & ~values[:-1]) + 1).tolist()
    if initial_contact is False and values[0]:
        onsets.insert(0, 0)
    boundaries = [0, *[x for x in onsets if x > 0], len(values)]
    intervals = [
        dict(start=a, stop=b, complete=bool(a in onsets and b in onsets))
        for a, b in zip(boundaries[:-1], boundaries[1:])
        if b > a
    ]
    return dict(
        onsets=onsets,
        intervals=intervals,
        debounce=False,
        initial_contact_known=initial_contact is not None,
        nonperiodic_allowed=True,
    )


def _records(records: Sequence[dict], split: str, source: str) -> list[dict]:
    rows = [
        r
        for r in records
        if r.get("split") == split and r.get("source") == source and r.get("eligible", True)
    ]
    keys = [(str(r["group"]), str(r["learner"]), r["source"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicated learning cases cannot count as independent observations")
    for r in rows:
        if (
            set(r["scores"]) != set(SCORE_IDS)
            or not np.isfinite(list(r["scores"].values())).all()
            or min(r["scores"].values()) < 0
        ):
            raise ValueError("Every case requires the five finite pre-update distances")
        trials, successes = r.get("walking_trials", 16), r["walking_successes"]
        if (
            type(trials) is not int
            or trials != 16
            or type(successes) is not int
            or not 0 <= successes <= trials
        ):
            raise ValueError("J1 requires all 16 standalone outcomes per eligible case")
        if not np.isfinite(r["post_motion"]) or not 0 <= r["post_motion"] <= 3 + 1e-12:
            raise ValueError("Post-update motion error is outside the frozen range")
    return rows


def _expit(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -value))


def fit_prediction_models(records: Sequence[dict], *, source: str = "mppi") -> dict:
    rows = _records(records, "development", source)
    if not rows:
        raise ValueError("No eligible development learning cases")
    y = np.array([r["post_motion"] for r in rows])
    successes = np.array([r["walking_successes"] for r in rows], dtype=np.float64)
    n = np.full(len(rows), 16.0)
    boundary = bool(successes.sum() in (0, n.sum()))
    degenerate = bool(np.ptp(successes / n) == 0)
    models = {}
    for candidate in SCORE_IDS:
        score = np.array([r["scores"][candidate] for r in rows])
        mean, scale = float(score.mean()), float(score.std())
        scale = scale if scale > 0 else 1.0
        x = np.column_stack((np.ones(len(rows)), (score - mean) / scale))
        ridge = np.diag([0.0, 1.0])
        linear = np.linalg.solve(x.T @ x + ridge, x.T @ np.log1p(y))
        p0 = (successes.sum() + 0.5) / (n.sum() + 1) if boundary else successes.sum() / n.sum()
        logistic = np.array([np.log(p0 / (1 - p0)), 0.0])
        converged, iterations = degenerate, 0
        if not degenerate:
            for iteration in range(100):
                p = _expit(x @ logistic)
                gradient = x.T @ (n * p - successes) + ridge @ logistic
                hessian = x.T @ ((n * p * (1 - p))[:, None] * x) + ridge
                step = np.linalg.solve(hessian, gradient)
                logistic -= step
                iterations = iteration + 1
                if np.max(np.abs(step)) <= 1e-10:
                    converged = True
                    break
            if not converged or not np.isfinite(logistic).all():
                raise ValueError("Frozen binomial Newton solver did not converge")
        models[candidate] = dict(
            score_mean=mean,
            score_std=scale,
            score_degenerate=bool(score.std() == 0),
            motion_coefficients=linear.tolist(),
            walking_coefficients=logistic.tolist(),
            walking_degenerate=degenerate,
            walking_constant_probability=float(p0) if degenerate else None,
            newton_iterations=iterations,
            newton_converged=converged,
        )
    return dict(
        version=VERSION,
        fit_split="development",
        fit_source=source,
        models=models,
        development_groups=sorted({str(r["group"]) for r in rows}),
        development_roster_complete=bool(
            len({str(r["group"]) for r in rows}) == 8
            and all(
                sum(str(r["group"]) == group for r in rows) == 2
                for group in {str(r["group"]) for r in rows}
            )
        ),
        case_ids=[r.get("case_id", f"{r['group']}/{r['learner']}/{source}") for r in rows],
        motion_sd=float(y.std()),
        walking_degenerate=degenerate,
        ridge=1,
        prediction_capacity="intercept plus one standardized scalar for every candidate",
        walking_loss="binomial likelihood; all0/all1 uses (successes+.5)/(trials+1), slope0, no discrimination claim",
    )


def predict_case_scores(scores: dict, models: dict) -> dict:
    if models.get("fit_split") != "development" or set(scores) != set(SCORE_IDS):
        raise ValueError("Sealed development models and all five scores are required")
    result = {}
    for candidate in SCORE_IDS:
        model = models["models"][candidate]
        score = float(scores[candidate])
        if not np.isfinite(score) or score < 0:
            raise ValueError("Predicted scores must be finite nonnegative pre-update values")
        x = np.array([1.0, (score - model["score_mean"]) / model["score_std"]])
        log_motion = float(x @ _array(model["motion_coefficients"]))
        result[candidate] = dict(
            motion=float(np.expm1(np.clip(log_motion, 0, np.log(4)))),
            walking_probability=float(_expit(x @ _array(model["walking_coefficients"]))),
        )
    return result


def _auc(binary: np.ndarray, prediction: np.ndarray) -> float | None:
    positive, negative = prediction[binary], prediction[~binary]
    if not len(positive) or not len(negative):
        return None
    differences = positive[:, None] - negative[None, :]
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def evaluate_predictions(
    records: Sequence[dict],
    models: dict,
    *,
    source: str = "mppi",
    seed: int = BOOTSTRAP_SEED,
    draws: int = 9999,
) -> dict:
    if seed != BOOTSTRAP_SEED or draws != 9999:
        raise ValueError("Protected comparisons use the frozen 9999 group resamples and seed")
    rows = _records(records, "protected", source)
    if not rows:
        return dict(
            version=VERSION,
            objective_passed=False,
            reason="No eligible protected learning cases",
            source=source,
        )
    predictions = [predict_case_scores(r["scores"], models) for r in rows]
    motion = np.array([r["post_motion"] for r in rows])
    rates = np.array([r["walking_successes"] / 16 for r in rows])
    pm = np.array([[p[c]["motion"] for c in SCORE_IDS] for p in predictions])
    pw = np.array([[p[c]["walking_probability"] for c in SCORE_IDS] for p in predictions])
    motion_loss = (pm - motion[:, None]) ** 2
    # Proper per-episode Bernoulli Brier, including the common within-case variance.
    walking_loss = pw**2 - 2 * pw * rates[:, None] + rates[:, None]
    groups = sorted({str(r["group"]) for r in rows})
    if set(groups) & set(models["development_groups"]):
        raise ValueError("Protected and development source groups must be disjoint")
    group_m = np.stack(
        [motion_loss[[str(r["group"]) == g for r in rows]].mean(axis=0) for g in groups]
    )
    group_w = np.stack(
        [walking_loss[[str(r["group"]) == g for r in rows]].mean(axis=0) for g in groups]
    )
    rmse, brier = np.sqrt(group_m.mean(axis=0)), group_w.mean(axis=0)
    motion_gain = 1 - rmse[1:] / rmse[0] if rmse[0] > 0 else np.zeros(4)
    walking_gain = brier[0] - brier[1:]
    point = np.concatenate((motion_gain, walking_gain))
    indices = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(groups), size=(draws, len(groups)), dtype=np.int64
    )
    bm, bw = np.sqrt(group_m[indices].mean(axis=1)), group_w[indices].mean(axis=1)
    valid_denominator = bm[:, 0] > 0
    boot_motion = np.divide(
        bm[:, 1:], bm[:, :1], out=np.ones((draws, 4)), where=valid_denominator[:, None]
    )
    boot = np.column_stack((1 - boot_motion, bw[:, :1] - bw[:, 1:]))
    stderr = boot.std(axis=0, ddof=1)
    nondegenerate = stderr > 1e-14
    standardized = np.divide(boot - point, stderr, out=np.zeros_like(boot), where=nondegenerate)
    critical = float(max(0, np.quantile(np.max(standardized, axis=1), 0.95, method="linear")))
    lower = point - critical * stderr
    # Zero-variance uncertainty is not promoted to a positive confidence certificate.
    lower[~nondegenerate] = np.minimum(lower[~nondegenerate], 0)
    positive_groups = {str(r["group"]) for r in rows if r["walking_successes"] > 0}
    positive_streams = {
        (str(r["group"]), str(r["learner"])) for r in rows if r["walking_successes"] > 0
    }
    roster = len(groups) == 8 and all(sum(str(r["group"]) == g for r in rows) == 2 for g in groups)
    motion_identified = bool(
        roster
        and models["development_roster_complete"]
        and models["motion_sd"] >= 0.1
        and motion.std() >= 0.1
    )
    walking_identified = bool(
        roster
        and models["development_roster_complete"]
        and not models["walking_degenerate"]
        and sum(rates > 0) >= 4
        and sum(rates == 0) >= 4
        and len(positive_groups) >= 3
        and len(positive_streams) >= 3
    )
    binary_available = all(type(r.get("reproduced")) is bool for r in rows)
    binary = np.array([r["reproduced"] for r in rows]) if binary_available else None
    table = []
    for j, candidate in enumerate(SCORE_IDS):
        row = dict(
            candidate=candidate,
            motion_rmse=float(rmse[j]),
            walking_brier=float(brier[j]),
            reproduction_auc=_auc(binary, -pm[:, j]) if binary_available else None,
        )
        if j:
            relative_w = float(walking_gain[j - 1] / brier[0]) if brier[0] > 0 else 0.0
            row.update(
                motion_relative_improvement=float(motion_gain[j - 1]),
                walking_absolute_improvement=float(walking_gain[j - 1]),
                walking_relative_improvement=relative_w,
                motion_simultaneous_lower=float(lower[j - 1]),
                walking_simultaneous_lower=float(lower[j + 3]),
                passed=bool(
                    source == "mppi"
                    and motion_identified
                    and walking_identified
                    and rmse[0] > 0
                    and brier[0] > 0
                    and motion_gain[j - 1] >= 0.1
                    and walking_gain[j - 1] >= 0.02
                    and relative_w >= 0.1
                    and lower[j - 1] > 0
                    and lower[j + 3] > 0
                ),
            )
        else:
            row["passed"] = False
        table.append(row)
    return dict(
        version=VERSION,
        source=source,
        objective_passed=any(r["passed"] for r in table),
        motion_identified=motion_identified,
        walking_identified=walking_identified,
        eligible_cases=len(rows),
        eligible_groups=groups,
        positive_case_count=int(sum(rates > 0)),
        zero_case_count=int(sum(rates == 0)),
        motion_sd=float(motion.std()),
        score_table=table,
        binary_reproduction_identified=bool(binary_available and binary.any() and not binary.all()),
        bootstrap=dict(
            seed=seed,
            draws=draws,
            group_order=groups,
            group_indices=indices.tolist(),
            group_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
            simultaneous_contrasts=8,
            critical_value=critical,
            standard_errors=stderr.tolist(),
            degenerate_contrasts=(~nondegenerate).tolist(),
            baseline_zero_motion_resamples=int(sum(~valid_denominator)),
            scope="Paired source-group bootstrap retains both learner streams; finite-benchmark uncertainty, not coverage guarantee",
        ),
        predictions=predictions,
    )


def score_definitions() -> dict:
    return dict(
        version=VERSION,
        lower_is_closer=True,
        candidates={
            "A": "mean six-torque squared residual on teacher states",
            "B": "fixed five-control causal box residual, valid controls only",
            "C": "development affine explained MSE plus twice residual MSE",
            "D": "same-start synchronized motion with explicit missing fraction",
            "E": "development-standardized joint occupancy SW2 plus explicit missing fraction",
        },
        rejected_candidates={
            "orthonormal_fourier_mse": "Parseval alias of A",
            "action_law_to_point_w2": "Expected MSE alias",
            "explained_only": "Ignores contradictory same-observation supervision",
            "conditional_mean_lambda1": "With the true conditional mean, orthogonal decomposition equals expected MSE",
        },
        score_parameters=dict(
            smoothing_controls=5,
            explanatory_ridge=1e-3,
            residual_weight=2,
            occupancy_directions=32,
            quantiles=1000,
            occupancy_seed=OCCUPANCY_SEED,
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_draws=9999,
        ),
    )


def calibration_suite() -> dict:
    """Known-array calibrations only; none are learned Walker competence evidence."""
    rng = np.random.Generator(np.random.PCG64(2026129803))
    a, b = rng.normal(size=(31, 6)), rng.normal(size=(31, 6))
    ordinary = action_mse(a, b)
    fourier_error = abs(fourier_action_mse(a, b) - ordinary)
    point = rng.normal(size=6)
    wasserstein_error = abs(
        deterministic_action_law_mse(a, point) - action_mse(a, np.tile(point, (len(a), 1)))
    )
    zero_obs = np.zeros((1000, 17))
    zero_actions = np.zeros((1000, 6))
    normalization = dict(mean=np.zeros(17).tolist(), std=np.ones(17).tolist())
    explainer = fit_state_explainer(zero_obs, zero_actions, normalization, group_ids=["synthetic"])
    states = np.zeros((1001, 19))
    states[:, 0] = np.arange(1001) * 0.008
    states[:, 1] = states[:, 0] * 1.5
    states[:, 2] = 1.1
    states[:, 10] = 1.5
    trace = dict(
        states=states, done=np.zeros(1000, dtype=bool), observations=zero_obs, actions=zero_actions
    )
    occupancy = fit_occupancy_normalizer(
        np.column_stack((zero_obs, zero_actions)), group_ids=["synthetic"]
    )
    copy = score_case(trace, trace, zero_actions, explainer, occupancy)["scores"]
    contradictory = np.tile(np.array([-1.0, 1.0]), 500)[:, None] * np.ones((1, 6))
    contradictory_explainer = fit_state_explainer(zero_obs, contradictory, normalization)
    contradiction = state_explained_score(
        zero_obs, contradictory, zero_actions, contradictory_explainer
    )
    gain = rng.normal(size=(6, 17))
    perturbations = np.vstack((np.eye(17), -np.eye(17)))
    held = rng.normal(size=(8, 17))
    feedback = fit_local_feedback(
        perturbations, perturbations @ gain.T, held_observations=held, held_actions=held @ gain.T
    )
    changed = dict(trace)
    changed["states"] = states.copy()
    changed["states"][1:, 4] = 0.5
    coordination = motion_metrics(trace, changed)["error"]
    observations, actions = rng.normal(size=(1000, 17)), rng.normal(size=(1000, 6))
    scale = fit_occupancy_normalizer(np.column_stack((observations, actions)))
    permutation = rng.permutation(1000)
    shuffled_pairs = sliced_occupancy_distance(
        observations, actions, observations[permutation], actions[permutation], scale
    )["score"]
    contacts = np.tile([False, False, True, True, True, False], 10)
    modified_contacts = contacts.copy()
    modified_contacts[1] = True
    modified_contacts[3] = False
    segments = stride_segments(contacts)
    altered_segments = stride_segments(modified_contacts)
    cases = []
    for split, prefix in (("development", "dev"), ("protected", "final")):
        for group in range(8):
            for learner in range(2):
                score = group + 0.2 * learner
                cases.append(
                    dict(
                        split=split,
                        group=f"{prefix}{group}",
                        learner=learner,
                        source="mppi",
                        scores={candidate: score for candidate in SCORE_IDS},
                        post_motion=float(np.expm1(0.1 + 0.09 * score)),
                        walking_successes=0 if group < 4 else 8,
                        walking_trials=16,
                        reproduced=group < 4,
                    )
                )
    good = fit_prediction_models(cases)
    shuffled_cases = [dict(row) for row in cases]
    dev = [row for row in shuffled_cases if row["split"] == "development"]
    labels = [row["post_motion"] for row in dev]
    for row, index in zip(dev, rng.permutation(len(dev)), strict=True):
        row["post_motion"] = labels[index]
    shuffled = fit_prediction_models(shuffled_cases)
    protected = [row for row in cases if row["split"] == "protected"]
    good_error = np.mean(
        [
            (predict_case_scores(row["scores"], good)["A"]["motion"] - row["post_motion"]) ** 2
            for row in protected
        ]
    )
    shuffled_error = np.mean(
        [
            (predict_case_scores(row["scores"], shuffled)["A"]["motion"] - row["post_motion"]) ** 2
            for row in protected
        ]
    )
    nonlinear_obs = rng.normal(size=(1000, 17))
    nonlinear_actions = np.repeat(np.sin(3 * nonlinear_obs[:, :1]), 6, axis=1)
    nonlinear_map = fit_state_explainer(nonlinear_obs, nonlinear_actions, normalization)
    nonlinear_copy = state_explained_score(
        nonlinear_obs, nonlinear_actions, nonlinear_actions, nonlinear_map
    )
    lambda1_error = abs(
        contradiction["explained_mse"]
        + contradiction["residual_mse"]
        - action_mse(contradictory, zero_actions)
    )
    checks = dict(
        exact_copy=all(value == 0 for value in copy.values()),
        fourier_alias=fourier_error <= 1e-12,
        deterministic_wasserstein_alias=wasserstein_error <= 1e-12,
        contradiction_retained=bool(
            contradiction["explained_mse"] == 0
            and contradiction["residual_mse"] == 1
            and contradiction["score"] == 2
        ),
        known_affine_gain=bool(
            np.max(np.abs(np.asarray(feedback["gain"]) - gain)) <= 1e-12
            and feedback["held_rmse"] <= 1e-12
        ),
        coordination_changes_motion=bool(coordination > 0),
        joint_pair_order_invariance=bool(shuffled_pairs <= 1e-24),
        raw_contact_spikes_and_dropouts_visible=segments["onsets"] != altered_segments["onsets"],
        shuffled_predictive_labels_worse=bool(shuffled_error > good_error),
        C_is_not_identity_metric=bool(nonlinear_copy["score"] > 0),
        conditional_mean_lambda1_alias=bool(lambda1_error <= 1e-12),
    )
    return dict(
        version=VERSION,
        passed=all(checks.values()),
        checks=checks,
        exact_copy_scores=copy,
        fourier_max_error=fourier_error,
        deterministic_wasserstein_max_error=wasserstein_error,
        contradictory_observation_components=contradiction,
        affine_gain_max_error=float(np.max(np.abs(np.asarray(feedback["gain"]) - gain))),
        held_affine_rmse=feedback["held_rmse"],
        coordination_motion_error=coordination,
        time_shuffled_joint_occupancy_score=shuffled_pairs,
        raw_contacts=segments,
        contacts_with_spikes_and_dropouts=altered_segments,
        synthetic_prediction_mse=float(good_error),
        shuffled_label_prediction_mse=float(shuffled_error),
        nonlinear_exact_copy_C=nonlinear_copy,
        nonlinear_exact_copy_action_mse=action_mse(nonlinear_actions, nonlinear_actions),
        C_identity_limitation="C is a model-dependent proxy, not an identity metric; exact nonlinear copies can have positive explanatory residual.",
        conditional_mean_lambda1_identity_error=lambda1_error,
        scope="Synthetic array/controller arithmetic only; no Walker execution or fresh-student learning competence claim",
    )
