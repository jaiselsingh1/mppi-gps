"""The loop's Walker task as an RL environment (rss_checks baselines; runs in .venv-rl).

Same model (assets/walker2d.xml), control step (4 x 0.002 s), observation (qpos[1:], qvel clipped at
+-10), reset (init qpos/qvel + U(-5e-3, 5e-3)), healthy rule (0.8 < z < 2, |pitch| < 1) and cost as
src/envs/walker2d.Walker2d with NATIVE_TASK:
  c_t = 2 |1.5 - vx| + 0.5 vz^2 + 0.1 pitch^2 + 5 [unhealthy] + 0.001 |u|^2   (on the state after u)
Reward r_t = 5 - c_t - 20 [falls at t], episode ends at the fall (terminated) or after 1,000 controls
(truncated). Then sum_t r_t = 5 H - (MPPI's absorbing-cost objective: running cost through the fall,
5 per remaining control, 20 once), i.e. the exact objective MPPI optimizes, up to a constant.
judge(): the loop's walking criterion on its 32 standing starts (policy_survival.standing_starts,
exported to standing_starts.npz): survive 1,000 controls and mean |vx - 1.5| <= 0.3 after control 100.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ROOT = Path.home() / "mppi-gps-claude-sac"
XML = str(ROOT / "assets/walker2d.xml")
STARTS = ROOT / "rss_checks/rl/standing_starts.npz"


def cost_terms(qpos, qvel, u, healthy):
    return (2.0 * abs(1.5 - qvel[0]) + 0.5 * qvel[1] ** 2 + 0.1 * qpos[2] ** 2
            + 5.0 * (not healthy) + 1e-3 * float(np.sum(np.square(u))))  # fmt: skip


def healthy_of(qpos):
    return bool(0.8 < qpos[1] < 2.0 and -1.0 < qpos[2] < 1.0)


def obs_of(data):
    return np.concatenate([data.qpos[1:], np.clip(data.qvel, -10.0, 10.0)]).astype(np.float32)


class OurWalker(gym.Env):
    def __init__(self, max_steps=1000):
        self.model = mujoco.MjModel.from_xml_path(XML)
        self.data = mujoco.MjData(self.model)
        self.init_qpos = self.data.qpos.copy()
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (17,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        r = self.np_random
        self.data.qpos[:] = self.init_qpos + r.uniform(-5e-3, 5e-3, self.model.nq)
        self.data.qvel[:] = r.uniform(-5e-3, 5e-3, self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return obs_of(self.data), {}

    def set_state(self, x):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_setState(self.model, self.data, x, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return obs_of(self.data)

    def step(self, a):
        u = np.clip(np.asarray(a, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = u
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        h = healthy_of(self.data.qpos)
        c = cost_terms(self.data.qpos, self.data.qvel, u, h)
        r = 5.0 - c - (20.0 if not h else 0.0)
        self.t += 1
        return obs_of(self.data), float(r), (not h), self.t >= self.max_steps, dict(vx=float(self.data.qvel[0]))


def judge(act, n_starts=32):
    """act: (obs batch) -> actions. Returns standing walks (of 32), survivals, median seconds up."""
    starts = np.load(STARTS)["states"][:n_starts]
    model = mujoco.MjModel.from_xml_path(XML)
    datas = [mujoco.MjData(model) for _ in starts]
    for d, x in zip(datas, starts):
        mujoco.mj_setState(model, d, x, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, d)
    alive = np.ones(len(starts), bool)
    lasted = np.full(len(starts), 1000)
    speed = np.zeros((1000, len(starts)))
    for t in range(1000):
        o = np.stack([obs_of(d) for d in datas])
        u = np.clip(np.asarray(act(o), dtype=np.float64), -1.0, 1.0)
        for i, d in enumerate(datas):
            if not alive[i]:
                continue
            d.ctrl[:] = u[i]
            for _ in range(4):
                mujoco.mj_step(model, d)
            speed[t, i] = d.qvel[0]
            if not healthy_of(d.qpos):
                alive[i] = False
                lasted[i] = t + 1
    err = np.abs(speed[100:] - 1.5).mean(0)
    walks = (lasted == 1000) & (err <= 0.3)
    return dict(standing_walks=int(walks.sum()), survived=int((lasted == 1000).sum()),
                median_s=float(np.median(lasted)) * 0.008)  # fmt: skip
