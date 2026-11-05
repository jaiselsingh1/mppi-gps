"""Position control for Walker MPPI: does commanding target joint angles give learnable actions?

Kendall, 2026-11-04: "Maybe some position control instead of torque would also smooth things out."
MPPI's torques are mostly noise relative to the state (10-nearest-state R^2 about 4%), while its
motion is largely set by the state (62-66%), so most of the loop's machinery exists to work around
torque-level noise. Writes only to --out.

Design, fixed before the first run:

- Model: assets/walker2d.xml with each motor replaced by a PD actuator inside MuJoCo (<general>,
  gaintype fixed, biastype affine): control c in [-1, 1] sets the target angle theta = s c (s = 150
  degrees for thighs and knees, 45 for feet; c = 0 is the standing pose; targets past a joint limit
  push into it), torque = kp (theta - q) - kv qdot, limited to +-100 as now. kv = 0.1 kp.
- Task and cost: NATIVE_TASK unchanged (controls stay in [-1, 1], so the loop, the fit and the
  policy's tanh output work unchanged; the control cost now falls on targets).
- MPPI: configs/walker2d_best.json with noise_sigma set per run (target space), all else unchanged.
- Episodes: reset seeds 0..--episodes-1 (the torque bank's are 0-9), planner seeds 10,000 + seed,
  --controls controls each (1,000 = 8 s).
- Measured per setting: walks (WALKER.admissible), mean cost against the torque bank's, survivors'
  speed, stride regularity (stride_cycle.Section with the torque bank's levels: change and spread),
  and action predictability (action_spectrum's 10-nearest-state R^2 across walks, controls 100+),
  also computed for the torque bank's walks.
- Saved: each episode's states (before each control, start first) and actions, a position bank for
  the cloning test.
- Readings: "position MPPI walks" if it walks in >= 8 of 10 episodes; "its actions are more
  learnable" if its action R^2 is >= 3x the torque bank's.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

SCALE = {"thigh": np.deg2rad(150.0), "leg": np.deg2rad(150.0), "foot": np.deg2rad(45.0)}


def position_xml(kp, kv, out):
    """The Walker model with PD position actuators; returns its path."""
    tree = ET.parse(ROOT / "assets" / "walker2d.xml")
    actuator = tree.getroot().find("actuator")
    motors = list(actuator.findall("motor"))
    for m in motors:
        joint = m.get("joint")
        s = SCALE[joint.split("_")[0]]
        actuator.remove(m)
        ET.SubElement(
            actuator,
            "general",
            joint=joint,
            ctrllimited="true",
            ctrlrange="-1 1",
            forcelimited="true",
            forcerange="-100 100",
            gaintype="fixed",
            gainprm=f"{kp * s}",
            biastype="affine",
            biasprm=f"0 {-kp} {-kv}",
        )
    path = out / f"walker2d_position_kp{kp:g}.xml"
    tree.write(path)
    return path


def predictability(walks):
    """action_spectrum's 10-nearest-state R^2 of the actions across walks (controls 100+)."""
    from scripts.diagnostics.action_spectrum import FIRST, neighbour_index, state_dependence
    from scripts.diagnostics.noise_cutoff import observation

    obs = [observation(s[:-1] if len(s) > len(a) else s)[FIRST:] for s, a in walks]
    pooled = np.concatenate(obs)
    mu, sd = pooled.mean(0), np.maximum(pooled.std(0), 1e-8)
    index = neighbour_index([(o - mu) / sd for o in obs])
    y = np.concatenate([a[FIRST:] for _, a in walks])
    return float(state_dependence(y, index, np.random.default_rng(0))["r2"])


def main() -> int:
    from scripts.diagnostics.coupled_mppi import CONFIG
    from scripts.diagnostics.loop_task import WALKER
    from scripts.diagnostics.policy_picks import load_bank
    from scripts.diagnostics.stride_cycle import Section
    from scripts.diagnostics.trajectory_pi import step_costs
    from src.envs.walker2d import Walker2d
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kp", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--kv-ratio", type=float, default=0.1, help="kv = this x kp")
    parser.add_argument(
        "--set", action="append", default=[], help="MPPI config override, key=value (e.g. H=96)"
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--controls", type=int, default=1000)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    xml = position_xml(args.kp, args.kv_ratio * args.kp, args.out)
    env = Walker2d(**NATIVE_TASK, model_path=str(xml), nthread=args.threads)
    config = dict(json.loads(CONFIG.read_text()), noise_sigma=args.sigma)
    for item in args.set:
        key, value = item.split("=", 1)
        config[key] = type(config[key])(float(value)) if key in config else float(value)
    torque_bank = load_bank(args.bank)
    section = Section([e["states"] for e in torque_bank])
    episodes = []
    for seed in range(args.episodes):
        np.random.seed(seed)
        env.reset()
        controller = MPPI(env, MPPIConfig(**config))
        controller.reset(seed=10_000 + seed)
        states, actions = [env.get_state()], []
        for _ in range(args.controls):
            action, _ = controller.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
            env.step(action)
            actions.append(np.asarray(action, dtype=np.float64))
            states.append(env.get_state())
        states, actions = np.array(states), np.array(actions)
        walks = bool(WALKER.admissible(states[None, 1:])[0])
        cost = float(step_costs(env, states[None, 1:], actions[None]).sum())
        np.savez_compressed(
            args.out / f"episode_{seed}.npz", states=states, actions=actions, walks=walks, cost=cost
        )
        healthy = env._is_healthy(states[1:, 2], states[1:, 3])
        up = int(np.argmin(healthy)) if not healthy.all() else len(healthy)
        episodes.append(
            dict(
                seed=seed,
                walks=walks,
                cost=cost,
                speed=float(states[1 : up + 1, 10].mean()),
                survived_s=up * 0.008,
            )
        )
        print(episodes[-1], round(time.perf_counter() - started), "s", flush=True)
    walking = [i for i, e in enumerate(episodes) if e["walks"]]
    pos_walks = []
    for i in walking:
        with np.load(args.out / f"episode_{i}.npz") as d:
            pos_walks.append((d["states"][:-1], d["actions"]))
    torque_walks = [(e["states"], e["actions"]) for e in torque_bank]
    summary = dict(
        kp=args.kp,
        kv=args.kv_ratio * args.kp,
        config=config,
        sigma=args.sigma,
        walks=len(walking),
        episodes=episodes,
        mean_cost=float(np.mean([e["cost"] for e in episodes])),
        torque_bank_mean_cost=float(
            np.mean(
                [
                    e["cost"]
                    for e in json.loads((args.bank / "summary.json").read_text())["episodes"]
                    if e["walks"]
                ]
            )
        ),
        stride=section.regularity([s for s, _ in pos_walks]) if len(pos_walks) >= 3 else None,
        torque_stride=section.regularity([s for s, _ in torque_walks]),
        action_r2=predictability(pos_walks) if len(pos_walks) >= 3 else None,
        torque_action_r2=predictability(torque_walks),
        seconds=time.perf_counter() - started,
    )
    summary["readings"] = dict(
        position_mppi_walks=summary["walks"] >= 8,
        actions_more_learnable=None
        if summary["action_r2"] is None
        else bool(summary["action_r2"] >= 3 * summary["torque_action_r2"]),
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
