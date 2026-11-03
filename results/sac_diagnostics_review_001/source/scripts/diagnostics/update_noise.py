"""Update noise: does cutting MPPI's leftover update noise let it track a walking policy?

Match-only MPPI could not track the walking SAC clone, even when handed the
clone's exact plan every step. The hypothesis: MPPI executes a weighted average
of noisy samples with only about 8-10 effective samples, so every update leaves
noise in the plan. This tests changes to the update in the same match-only task
and writes only to --out.

Design, fixed before the first run:

- Task and target as match_only_mppi: zero task cost, torque matching to the
  walking SAC clone at w = 1.5, standard MPPI otherwise.
- Variants: "baseline" (K 256, lambda 3, sigma 0.3); "nominal sample" (one of
  the 256 samples is the plan itself, unperturbed); "more samples" (K 1024);
  "hotter" (lambda 30); "narrower" (sigma 0.1).
- Modes: "seeded", where the plan is replaced by the clone's own 48-step
  sequence before every decision (a diagnostic of the update alone, not a
  method); and "own plan", MPPI's normal warm start from a zero initial plan.
- Episodes: reset seeds 0-2, up to 1,000 controls.
- Measures: K1 walking, controls survived, executed torque gap to the clone,
  mean effective sample count per decision.
- Readings: "update noise causes the drift" if in seeded mode some variant walks
  >= 2/3 with torque gap <= 0.01 while baseline walks 0/3. "A fixed MPPI tracks
  the clone on its own" if in own-plan mode some variant walks >= 2/3.
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

import mujoco
import numpy as np

from scripts.diagnostics.coupled_mppi import CONFIG
from scripts.diagnostics.match_only_mppi import MatchOnlyPlanning, matching_only, outcome, reset
from scripts.diagnostics.sac_continuation import K1, load_sac_rollouts
from scripts.diagnostics.student_positive_control import fit_s1
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

WEIGHT = 1.5
SEEDS = (0, 1, 2)
STEPS = 1000
FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS
VARIANTS = {
    "baseline": {},
    "nominal sample": {},
    "more samples": dict(K=1024),
    "hotter": dict(lam=30.0),
    "narrower": dict(noise_sigma=0.1),
}


class NominalSampleMPPI(MPPI):
    """MPPI whose first sample is always the unperturbed plan."""

    def _sample_noise(self) -> np.ndarray:
        noise = super()._sample_noise()
        noise[0] = 0.0
        return noise


def clone_sequence(env, data, clone, state, warm, horizon) -> np.ndarray:
    mujoco.mj_setState(env.model, data, state, FULL)
    data.qacc_warmstart[:] = warm
    sequence = np.empty((horizon, 6))
    for t in range(horizon):
        obs = np.concatenate([data.qpos[1:], np.clip(data.qvel, -10.0, 10.0)])
        sequence[t] = clone.predict(obs.astype(np.float32))
        data.ctrl[:] = sequence[t]
        for _ in range(env._frame_skip):
            mujoco.mj_step(env.model, data)
    return sequence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), action="append", help="default: all")
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    run = Walker2d(**NATIVE_TASK)
    planning = MatchOnlyPlanning(**NATIVE_TASK)
    data = mujoco.MjData(run.model)
    sac = load_sac_rollouts(K1)
    clone = fit_s1(
        np.concatenate([r["observations"] for r in sac[:9]]).astype(np.float32),
        np.concatenate([r["actions"] for r in sac[:9]]).astype(np.float32),
        0,
    )
    base = json.loads(CONFIG.read_text())
    seeds = SEEDS[:1] if args.quick else SEEDS
    steps = 40 if args.quick else STEPS

    rows = []
    for name in args.variant or VARIANTS:
        kind = NominalSampleMPPI if name == "nominal sample" else MPPI
        controller = kind(planning, MPPIConfig(**dict(base, **VARIANTS[name])))
        for mode in ("seeded", "own plan"):
            for seed in seeds:
                reset(run, seed)
                controller.reset(seed=10_000 + seed)
                states, actions, effective, cost, done = [], [], [], 0.0, False
                for _ in range(steps):
                    state, warm = run.get_state(), run.get_warmstart()
                    nominal = (
                        clone_sequence(run, data, clone, state, warm, controller.H)
                        if mode == "seeded"
                        else None
                    )
                    action, info = controller.plan_step(
                        state,
                        nominal=nominal,
                        prior_cost=matching_only(run, clone, state, WEIGHT),
                        initial_warmstart=warm,
                    )
                    states.append(state)
                    actions.append(action)
                    effective.append(info["n_eff"])
                    _, c, done, _ = run.step(action)
                    cost += c
                    if done:
                        break
                rows.append(
                    dict(
                        variant=name,
                        mode=mode,
                        seed=seed,
                        mean_n_eff=float(np.mean(effective)),
                        **outcome(run, states, actions, cost, done, steps, clone),
                    )
                )
                print(rows[-1], flush=True)

    cells = {}
    for r in rows:
        cell = cells.setdefault(
            f"{r['variant']} / {r['mode']}", dict(walks=0, runs=0, gap=[], controls=[], n_eff=[])
        )
        cell["walks"] += r["walks"]
        cell["runs"] += 1
        cell["gap"].append(r["torque_gap"])
        cell["controls"].append(r["controls"])
        cell["n_eff"].append(r["mean_n_eff"])
    for cell in cells.values():
        cell.update(
            gap=float(np.mean(cell["gap"])),
            controls=float(np.mean(cell["controls"])),
            n_eff=float(np.mean(cell["n_eff"])),
        )
    seeded_fixed = [
        k
        for k, c in cells.items()
        if k.endswith("/ seeded")
        and not k.startswith("baseline")
        and c["walks"] >= 2
        and c["gap"] <= 0.01
    ]
    own_fixed = [
        k
        for k, c in cells.items()
        if k.endswith("/ own plan") and not k.startswith("baseline") and c["walks"] >= 2
    ]
    baseline = cells.get("baseline / seeded")
    summary = dict(
        cells=cells,
        rows=rows,
        readings=dict(
            update_noise_causes_the_drift=bool(
                seeded_fixed and baseline and baseline["walks"] == 0
            ),
            variants_fixing_seeded=seeded_fixed,
            fixed_mppi_tracks_on_its_own=bool(own_fixed),
            variants_fixing_own_plan=own_fixed,
        ),
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
