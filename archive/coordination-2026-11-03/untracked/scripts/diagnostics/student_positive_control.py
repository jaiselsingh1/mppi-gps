"""Positive control: can this cloning pipeline copy a policy that clearly walks?

Clones SAC's own saved rollouts and smooth-MPPI walks into two networks, then
checks walking from mid-stride states and from standing. Reads the K1 run and the
noise-cutoff run and writes only to --out.

Design, fixed before the first run:

- Data, 9,000 rows each: SAC rollouts 0-8 (held out: 9-15) and the 9 smooth-MPPI
  training walks (held out: reset seed 5).
- Networks: the S1 student (17 -> 128 tanh -> 128 tanh -> 6 tanh, inputs z-scored
  on the training rows, repo fit_policy recipe) and SAC's own actor shape
  (17 -> 256 ReLU -> 256 ReLU -> 6, then tanh, raw inputs as SAC sees them),
  trained by the same Adam lr 3e-4, batch 256, gradient clip 0.5 on action MSE.
  10,000 updates, seeds 0-2.
- Measures: train and held-out action MSE; "keeps walking" from mid-stride
  starts (controls 100-900 of held-out walks, 500 controls, rule as in
  student_takeover); K1 walking from S1's 32 standing starts.
- Calibration: replaying a held-out walk's own actions from each start must
  reproduce its saved states for 25 controls.
- Readings: the pipeline "can clone a walker" if SAC-data students in either
  network keep walking from >= 50% of held-out mid-stride starts. "Network shape
  matters" if, on SAC data, the two networks differ by >= 30 points. MPPI's data
  is "the obstacle" if the pipeline can clone a walker but MPPI-data students in
  the same network keep walking from < 10%.
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
from torch import nn

from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts
from scripts.diagnostics.smooth_teacher_students import (
    ARMS,
    DATA_SEEDS,
    EVAL_SEEDS,
    RECIPE,
    SOURCE,
    STEPS,
)
from scripts.diagnostics.student_failure import HELD_OUT_SEED
from scripts.diagnostics.student_takeover import STARTS, WALK_MAE, drive, restore
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps import policy
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.utils.trace_replay import load_episode

SEEDS = (0, 1, 2)
UPDATES = 10_000
TRAIN_SAC = range(9)


class SacShape(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(17, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 6)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))

    def predict(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self(torch.as_tensor(np.asarray(obs, dtype=np.float32))).numpy()


def fit_sac_shape(obs: np.ndarray, actions: np.ndarray, seed: int) -> SacShape:
    model = SacShape(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=RECIPE["lr"])
    rng = np.random.default_rng(seed)
    x, y = torch.as_tensor(obs), torch.as_tensor(actions)
    for _ in range(UPDATES):
        batch = torch.as_tensor(rng.integers(0, len(x), RECIPE["batch_size"]))
        loss = torch.mean((model(x[batch]) - y[batch]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    return model.eval()


def fit_s1(obs: np.ndarray, actions: np.ndarray, seed: int) -> policy.Policy:
    normalization = dict(mean=obs.mean(0), std=np.maximum(obs.std(0), 1e-6))
    parent = policy.Policy(seed, "cpu", normalization, hidden_dim=128)
    child, _, _ = policy.fit_policy(parent, obs, actions, updates=UPDATES, seed=seed, **RECIPE)
    return child.eval()


def standing(env: Walker2d, student, seed: int) -> bool:
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        env.reset()
    finally:
        np.random.set_state(saved)
    r = drive(env, lambda o, c: student.predict(o.astype(np.float32)).astype(np.float64), STEPS)
    vx = r["states"][1:, 10]
    return bool(
        r["controls"] == STEPS and not np.isnan(vx).any() and np.mean(np.abs(vx - 1.5)) <= WALK_MAE
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)

    sac = load_sac_rollouts(K1)
    for r in sac:
        r["actions"] = r["actions"].astype(np.float64)
    mppi = [load_episode(SOURCE / ARMS["smooth"] / "traces.npz", k) for k in range(10)]
    for e in mppi:
        e["observations"] = observation(e["states"][:-1])
    sources = {
        "SAC": (
            [sac[k] for k in TRAIN_SAC],
            [r for k, r in enumerate(sac) if k not in TRAIN_SAC],
        ),
        "MPPI smooth": (
            [e for e in mppi if e["reset_seed"] in DATA_SEEDS],
            [e for e in mppi if e["reset_seed"] == HELD_OUT_SEED],
        ),
    }
    env = Walker2d(**NATIVE_TASK)

    calibration = {}
    for name, (_, held) in sources.items():
        dev = 0.0
        for walk in held:
            for t in STARTS:
                restore(env, walk, t)
                r = drive(env, lambda o, c, w=walk, t=t: w["actions"][t + c], 25)
                dev = max(
                    dev, float(np.max(np.abs(r["states"][:, 1:] - walk["states"][t : t + 26, 1:])))
                )
        calibration[name] = dev

    cells = []
    for name, (train, held) in sources.items():
        obs = np.concatenate([w["observations"] for w in train]).astype(np.float32)
        actions = np.concatenate([w["actions"] for w in train]).astype(np.float32)
        held_obs = np.concatenate([w["observations"] for w in held]).astype(np.float32)
        held_actions = np.concatenate([w["actions"] for w in held]).astype(np.float32)
        for network, fit in (("S1 128 tanh", fit_s1), ("SAC 256 ReLU", fit_sac_shape)):
            for seed in SEEDS:
                student = fit(obs, actions, seed)
                act = lambda o, c, s=student: s.predict(o.astype(np.float32)).astype(np.float64)  # noqa: E731
                takeovers = []
                for walk in held:
                    for t in STARTS:
                        restore(env, walk, t)
                        takeovers.append(drive(env, act, 500)["walks"])
                cells.append(
                    dict(
                        data=name,
                        network=network,
                        seed=seed,
                        train_mse=float(np.mean((student.predict(obs) - actions) ** 2)),
                        held_mse=float(np.mean((student.predict(held_obs) - held_actions) ** 2)),
                        held_variance=float(held_actions.var(0).mean()),
                        mid_stride_keeps_walking=float(np.mean(takeovers)),
                        mid_stride_starts=len(takeovers),
                        standing_walks=int(sum(standing(env, student, s) for s in EVAL_SEEDS)),
                    )
                )
                print(cells[-1], flush=True)

    table = {}
    for c in cells:
        key = f"{c['data']} / {c['network']}"
        entry = table.setdefault(key, dict(mid_stride=[], standing=[], train_mse=[], held_mse=[]))
        entry["mid_stride"].append(c["mid_stride_keeps_walking"])
        entry["standing"].append(c["standing_walks"])
        entry["train_mse"].append(c["train_mse"])
        entry["held_mse"].append(c["held_mse"])
    mean = {k: float(np.mean(v["mid_stride"])) for k, v in table.items()}
    sac_best = max(mean["SAC / S1 128 tanh"], mean["SAC / SAC 256 ReLU"])
    readings = dict(
        pipeline_can_clone_a_walker=sac_best >= 0.5,
        network_shape_matters=abs(mean["SAC / S1 128 tanh"] - mean["SAC / SAC 256 ReLU"]) >= 0.3,
        mppi_data_is_the_obstacle=[
            net
            for net in ("S1 128 tanh", "SAC 256 ReLU")
            if mean[f"SAC / {net}"] >= 0.5 and mean[f"MPPI smooth / {net}"] < 0.1
        ],
    )
    summary = dict(
        calibration_replay_max_state_dev=calibration,
        table=table,
        mid_stride_mean=mean,
        readings=readings,
        cells=cells,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
