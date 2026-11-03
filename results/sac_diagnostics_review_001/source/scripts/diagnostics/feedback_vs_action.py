"""Feedback vs action: does a clone's feedback error predict walking better than its action error?

SAC walks through large added action noise, yet clones that recover 98% of
SAC's torque mostly fall. The hypothesis: what matters is the clone's feedback,
meaning how its action changes with the state, not its action error at visited
states. Refits the 39 students of jitter_tolerance (checked against that run's
recorded held-out recovery), compares each with SAC on held-out SAC states, and
writes only to --out. No physics happens here; walking comes from that run.

Design, fixed before the first run:

- Action error: held-out action MSE against SAC (1 - recovered share, as there).
- Feedback error: the Jacobian of action with respect to the 17-value
  observation, in z-scored observation units, at 2,000 held-out SAC states;
  ||J_student - J_SAC||_F / ||J_SAC||_F, averaged over states.
- Reading: "feedback predicts better" if |Spearman(feedback error, walking)| is
  at least 0.15 above |Spearman(action error, walking)| across the 39 students,
  or if, at signal share 0.9, feedback error separates the white-jitter students
  (53% walking) from the slow-jitter ones (0%) while their action errors overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.diagnostics.jitter_tolerance import TRAIN, TYPES, UPDATES, jitter
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts, spearman
from scripts.diagnostics.smooth_teacher_students import RECIPE
from src.gps.mppi_gps import policy
from src.gps.mppi_gps.diagnostic_policies import LegacySACActor

ROWS = 2000


def jacobian(forward, obs: np.ndarray) -> np.ndarray:
    x = torch.as_tensor(obs, dtype=torch.float32).requires_grad_(True)
    y = forward(x)
    rows = [torch.autograd.grad(y[:, k].sum(), x, retain_graph=True)[0] for k in range(6)]
    return torch.stack(rows, dim=1).detach().numpy()  # (N, 6, 17)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tolerance", type=Path, default=Path("runs/2026-11-02_jitter_tolerance_001/summary.json")
    )
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    recorded = json.loads(args.tolerance.read_text())["conditions"]

    rng = np.random.default_rng(20261102)  # the tolerance run's stream, same draw order
    sac = load_sac_rollouts(K1)
    train = [sac[k] for k in TRAIN]
    held = [r for k, r in enumerate(sac) if k not in TRAIN]
    obs = np.concatenate([r["observations"] for r in train]).astype(np.float32)
    clean = np.concatenate([r["actions"] for r in train])
    held_obs = np.concatenate([r["observations"] for r in held]).astype(np.float32)
    held_clean = np.concatenate([r["actions"] for r in held])
    signal_var = clean.var(0)
    normalization = dict(mean=obs.mean(0), std=np.maximum(obs.std(0), 1e-6))
    sd = normalization["std"]
    sources = {kind: jitter(kind, len(train), len(train[0]["actions"]), rng) for kind in TYPES}

    pick = np.random.default_rng(0).choice(len(held_obs), ROWS, replace=False)
    probe = held_obs[pick]
    actor = LegacySACActor.from_checkpoint(K1 / "inputs" / "checkpoints" / "reference.pt")

    def sac_forward(x):
        mean, _ = actor(x)
        return torch.tanh(mean) * actor.action_scale + actor.action_bias

    j_sac = jacobian(sac_forward, probe) * sd  # z-scored observation units
    sac_norm = np.linalg.norm(j_sac, axis=(1, 2))

    students = []
    for condition in recorded:
        kind, level = condition["jitter"], condition["signal_share"]
        source = sources["MPPI" if kind == "clean" else kind]
        scale = np.sqrt(signal_var * (1.0 - level) / level) if level < 1.0 else 0.0
        labels = np.clip(clean + scale * np.concatenate(source), -1.0, 1.0).astype(np.float32)
        for entry in condition["students"]:
            seed = entry["seed"]
            student, _, _ = policy.fit_policy(
                policy.Policy(seed, "cpu", normalization, hidden_dim=128),
                obs,
                labels,
                updates=UPDATES,
                seed=seed,
                **RECIPE,
            )
            student.eval()
            mse = float(np.mean((student.predict(held_obs) - held_clean) ** 2))
            explained = 1.0 - mse / held_clean.var(0).mean()
            if abs(explained - entry["held_clean_explained"]) > 1e-6:
                raise ValueError(f"{kind} {level} seed {seed} does not reproduce")
            j = jacobian(student, probe) * sd
            gap = np.linalg.norm(j - j_sac, axis=(1, 2)) / sac_norm
            students.append(
                dict(
                    jitter=kind,
                    signal_share=level,
                    seed=seed,
                    walks=entry["standing_walks"],
                    action_mse=mse,
                    feedback_error=float(np.mean(gap)),
                )
            )
            print(students[-1], flush=True)

    walks = np.array([s["walks"] for s in students], dtype=float)
    rho_action = spearman(np.array([s["action_mse"] for s in students]), walks)
    rho_feedback = spearman(np.array([s["feedback_error"] for s in students]), walks)
    at = {
        kind: [s for s in students if s["jitter"] == kind and s["signal_share"] == 0.9]
        for kind in ("white", "slow")
    }
    action_overlap = max(s["action_mse"] for s in at["white"]) >= min(
        s["action_mse"] for s in at["slow"]
    )
    feedback_separates = max(s["feedback_error"] for s in at["white"]) < min(
        s["feedback_error"] for s in at["slow"]
    )
    summary = dict(
        students=students,
        spearman_action_error_vs_walking=rho_action,
        spearman_feedback_error_vs_walking=rho_feedback,
        at_signal_share_0_9=dict(
            white=[(s["action_mse"], s["feedback_error"], s["walks"]) for s in at["white"]],
            slow=[(s["action_mse"], s["feedback_error"], s["walks"]) for s in at["slow"]],
            action_errors_overlap=bool(action_overlap),
            feedback_separates=bool(feedback_separates),
        ),
        reading=(
            "feedback predicts better"
            if abs(rho_feedback) >= abs(rho_action) + 0.15
            or (action_overlap and feedback_separates)
            else "no advantage for feedback"
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "students"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
