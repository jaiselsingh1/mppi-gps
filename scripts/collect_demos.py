"""Collect demos for SL"""

import numpy as np 
import h5py 
from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from pathlib import Path

# settings for data collection 
num_conditions = 5 
episode_len = 200
save_path = Path("data/acrobot_demos.h5")

cfg = MPPIConfig(
    K = 2048, 
    H = 100, 
    lam = 716.7028104220127,
    noise_sigma = 0.641222666442091,
    adaptive_lam = False,
)

def random_initial_state(env: Acrobot, 
                         rng: np.random.Generator):
    env.reset()
    env.data.qpos[:] += rng.normal(0, 0.1, size = env.model.nq)
    env.data.qvel[:] += rng.normal(0, 0.1, size = env.model.nv)
    return env.get_state()

def main():
    env = Acrobot()
    controller = MPPI(env, cfg = cfg)
    rng = np.random.default_rng(42)

    nq, nv = env.model.nq, env.model.nv 
    obs_dim = nq + nv # for acrobot this is 4 

    with h5py.File(save_path, "w") as f:
        for i in range(num_conditions):
            state = random_initial_state(env, rng)
            controller.reset()
            grp = f.create_group(f"condition_{i}")

            for t in range(episode_len):
                action, info = controller.plan_step(state)
                obs, cost, done, _ = env.step(action)
                state = env.get_state()

                data = controller.get_rollout_data()

                # extract obs from the full physics state 
                obs_rollout = data["states"][:, :, 1 : 1 + obs_dim]

                step_grp = grp.create_group(f"step{t}")
                step_grp.create_dataset("obs", data=obs_rollout)           # (K, H, 4)
                step_grp.create_dataset("actions", data=data["actions"])    # (K, H, 1)
                step_grp.create_dataset("weights", data=data["weights"])    # (K,)
                step_grp.create_dataset("costs", data=data["costs"])        # (K,)

            print(f"condition {i}: collected {episode_len} steps, "f"final cost_min={info['cost_min']:.2f}")
                
        print(f"\nSaved to {save_path}")
        env.close()

if __name__ == "__main__":
    main()
