from mujoco_playground import registry
from mujoco_playground.config import dm_control_suite_params as p
envs = registry.dm_control_suite.ALL_ENVS
print(len(envs), envs)
for name in ["AcrobotSwingup", "AcrobotSwingupSparse", "FingerTurnHard", "FingerTurnEasy", "BallInCup",
             "CartpoleSwingupSparse", "PendulumSwingup", "ReacherHard", "HopperHop", "HumanoidStand"]:
    if name not in envs:
        print(name, "MISSING"); continue
    c = dict(p.brax_ppo_config(name)); cfg = registry.get_default_config(name)
    print("%-24s steps %5.0fM repeat %d ep %d ctrl_dt %s sim_dt %s" % (name, c["num_timesteps"] / 1e6, c["action_repeat"], c["episode_length"], cfg.ctrl_dt, cfg.sim_dt))
