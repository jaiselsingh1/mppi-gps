"""Diagnostic B (Step 23): what a state-only policy learns from MPPI's labels.

Uses Step 22's raw labels (72 per anchor: 9 warm starts x 8 seeds); no new
simulation.
- Student against E[label | state]: the squared distance between a saved
  student's action and an anchor's mean label, minus the mean label's own
  sampling variance, estimates the student's distance to E[label | state].
  As a share of Step 22's state component, 0 is the conditional mean and 1 is
  no better than one constant action.
- Modes: each anchor's labels projected on their first principal axis and fit
  by one- and two-component Gaussian mixtures (EM). Bimodal means BIC favours
  two components by more than 10, both weights are at least 0.2 and Ashman's D
  exceeds 2.
"""
import json
from pathlib import Path

import numpy as np
import tyro

from scripts.diagnostics.label_variance_split import LABELS, _components, _resamples, _with_interval

ACTUATORS = ("right thigh", "right leg", "right foot", "left thigh", "left leg", "left foot")
_STUDENT = "runs/2026-10-23_step15_teacher_seed0/policy_032.pt"
_VAR_FLOOR = 1e-4


def student_share(actions: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """actions (N, nu) against labels (N, W, R, nu): distance to E[label | state]."""
    W = labels.shape[1]
    sampling = (labels.mean(axis=2).var(axis=1, ddof=1) / W).mean()
    distance = float(((actions - labels.mean(axis=(1, 2))) ** 2).mean() - sampling)
    state = _components(labels)["state"]
    return {"distance": distance, "state": state, "share": distance / state}


def _normal(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def two_modes(x: np.ndarray, iterations: int = 300) -> dict[str, float]:
    """One- against two-component Gaussian mixture on a 1-D sample."""
    n = len(x)
    log_one = float(np.sum(np.log(_normal(x, x.mean(), max(x.var(), _VAR_FLOOR)))))
    low = x <= np.median(x)
    weight = np.array([0.5, 0.5])
    mean = np.array([x[low].mean(), x[~low].mean()])
    var = np.maximum([x[low].var(), x[~low].var()], _VAR_FLOOR)
    for _ in range(iterations):
        density = weight[:, None] * np.stack([_normal(x, m, v) for m, v in zip(mean, var)])
        responsibility = density / density.sum(axis=0)
        mass = responsibility.sum(axis=1)
        weight, mean = mass / n, responsibility @ x / mass
        var = np.maximum((responsibility * (x - mean[:, None]) ** 2).sum(axis=1) / mass, _VAR_FLOOR)
    density = weight[:, None] * np.stack([_normal(x, m, v) for m, v in zip(mean, var)])
    log_two = float(np.sum(np.log(density.sum(axis=0))))
    delta_bic = (-2.0 * log_one + 2.0 * np.log(n)) - (-2.0 * log_two + 5.0 * np.log(n))
    ashman = float(np.sqrt(2.0) * abs(mean[0] - mean[1]) / np.sqrt(var.sum()))
    return {"delta_bic": float(delta_bic), "ashman_d": ashman, "min_weight": float(weight.min()),
            "bimodal": bool(delta_bic > 10.0 and weight.min() >= 0.2 and ashman > 2.0)}


def principal_modes(labels: np.ndarray) -> dict[str, np.ndarray]:
    """labels (N, W, R, nu): per anchor, modes along the first principal axis."""
    flat = labels.reshape(len(labels), -1, labels.shape[-1])
    rows = {"bimodal": [], "delta_bic": [], "axis_share": [], "loadings": []}
    for cloud in flat:
        centred = cloud - cloud.mean(axis=0)
        _, singular, axes = np.linalg.svd(centred, full_matrices=False)
        modes = two_modes(centred @ axes[0])
        rows["bimodal"].append(modes["bimodal"])
        rows["delta_bic"].append(modes["delta_bic"])
        rows["axis_share"].append(singular[0] ** 2 / np.sum(singular ** 2))
        rows["loadings"].append(np.abs(axes[0]))
    return {key: np.asarray(value) for key, value in rows.items()}


def main(raw: Path, out_dir: Path, student: str = _STUDENT, bootstrap: int = 2000, seed: int = 7_900_000) -> None:
    import torch

    from src.rl.sac import load_policy

    data = np.load(raw)
    repo = Path(__file__).resolve().parents[2]
    payload = torch.load(repo / student, map_location="cpu", weights_only=False)
    policy = load_policy(payload.get("policy", payload))
    out_dir.mkdir(parents=True, exist_ok=False)
    result = {"raw": str(raw), "student": student, "populations": {}}
    for population in ("teacher", "student"):
        if f"{population}_executed" not in data.files:
            continue
        episodes = data[f"{population}_episode"]
        actions = np.asarray([policy.act(o) for o in data[f"{population}_obs"]])
        resamples = list(_resamples(episodes, bootstrap, seed))
        summary = {}
        for label in LABELS:
            labels = data[f"{population}_{label}"]
            modes = principal_modes(labels)
            share = student_share(actions, labels)
            summary[label] = {
                "student": _with_interval(share, [student_share(actions[i], labels[i]) for i in resamples]),
                "bimodal_fraction": _with_interval({"value": float(modes["bimodal"].mean())},
                                                   [{"value": float(modes["bimodal"][i].mean())}
                                                    for i in resamples])["value"],
                "delta_bic_median": float(np.median(modes["delta_bic"])),
                "axis_share_mean": float(modes["axis_share"].mean()),
                "loadings_mean": dict(zip(ACTUATORS, np.round(modes["loadings"].mean(axis=0), 3).tolist())),
            }
        result["populations"][population] = summary
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    for population, summary in result["populations"].items():
        for label, parts in summary.items():
            print(population, label, "student share", round(parts["student"]["share"]["mean"], 3),
                  parts["student"]["share"]["ci95"], "bimodal", round(parts["bimodal_fraction"]["mean"], 3),
                  parts["bimodal_fraction"]["ci95"])


if __name__ == "__main__":
    tyro.cli(main)
