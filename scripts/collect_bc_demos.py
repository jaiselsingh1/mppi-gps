"""Collect (state, action) trajectories from MPPI for pure SL/BC."""

import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass

from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


@dataclass
class BcConfig:
    M: int = 1000                                          # number of trajectories
    T: int = 1000                                        # steps per trajectory
    save_path: Path = Path("data/acrobot_bc.h5")


def main():
    bc_cfg   = BcConfig()
    mppi_cfg = MPPIConfig.load("acrobot")
    M, T     = bc_cfg.M, bc_cfg.T

    env = Acrobot()
    controller = MPPI(env, cfg=mppi_cfg)

    nq, nv  = env.model.nq, env.model.nv
    obs_dim = nq + nv
    act_dim = env.action_dim

    # preallocate so the file shape is fixed and the in-RAM footprint is bounded
    states  = np.zeros((M, T, obs_dim), dtype=np.float32)
    actions = np.zeros((M, T, act_dim), dtype=np.float32)
    costs   = np.zeros((M, T),          dtype=np.float32)

    for i in range(M):
        env.reset()
        controller.reset()                               # clear MPPI warm-start buffer

        for t in range(T):
            obs   = env._get_obs()
            state = env.get_state()
            action, info = controller.plan_step(state)
            _, cost, _, _ = env.step(action)

            states[i, t]  = obs
            actions[i, t] = action
            costs[i, t]   = cost

        print(f"trajectory {i:3d}/{M}  total_cost={costs[i].sum():.2f}  "
              f"final_cost_min={info['cost_min']:.2f}")

    bc_cfg.save_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(bc_cfg.save_path, "w") as f:
        f.create_dataset("states",  data=states)
        f.create_dataset("actions", data=actions)
        f.create_dataset("costs",   data=costs)
        f.attrs["M"]       = M
        f.attrs["T"]       = T
        f.attrs["obs_dim"] = obs_dim
        f.attrs["act_dim"] = act_dim

    print(f"\nsaved {M} trajectories of length {T} to {bc_cfg.save_path}")
    print(f"mean total cost across trajectories: {costs.sum(axis=1).mean():.2f}")
    env.close()


if __name__ == "__main__":
    main()
