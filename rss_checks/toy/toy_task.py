"""Stock dm_control reacher-easy and point_mass-easy with MuJoCo + numpy only (PREREG_gate.md).
Models, resets and reference episodes come from dm_control 1.0.38 (export_stock.py). State x = [time, qpos(2), qvel(2)]."""
from pathlib import Path
import numpy as np, mujoco
from mujoco import rollout

T = 1000
_SCALE = np.sqrt(-2.0 * np.log(0.1))


class Toy:
    def __init__(self, name, nthread=4, root=None):
        root = Path(root or Path(__file__).resolve().parent); self.name = name
        self.m = mujoco.MjModel.from_xml_path(str(root / f"{name}.xml")); Z = np.load(root / f"{name}_stock.npz")
        self.Z = Z; self.bank_q, self.bank_v, self.bank_tg = Z["qpos"], Z["qvel"], Z["target"]
        self.ts = float(Z["target_size"]); self.fs = float(Z["finger_size"])
        self.nobs = 6 if name == "reacher" else 4
        self.roll = rollout.Rollout(nthread=nthread); self.datas = [mujoco.MjData(self.m) for _ in range(nthread)]
        assert mujoco.mj_stateSize(self.m, mujoco.mjtState.mjSTATE_FULLPHYSICS) == 5

    def starts(self, idx):
        idx = np.asarray(idx)
        return np.concatenate([np.zeros((len(idx), 1)), self.bank_q[idx], self.bank_v[idx]], 1), self.bank_tg[idx].copy()

    def step(self, x, u):
        """x (W, 5), u (W, 2) -> next x (W, 5)."""
        s, _ = self.roll.rollout(self.m, self.datas, x, u[:, None, :])
        return s[:, 0]

    def open_loop(self, x, U):
        """x (W, 5), U (W, H, 2) -> states after each control (W, H, 5)."""
        s, _ = self.roll.rollout(self.m, self.datas, x, U)
        return s

    def finger(self, x):
        q1, q2 = x[..., 1], x[..., 1] + x[..., 2]
        return np.stack([0.12 * np.cos(q1) + 0.12 * np.cos(q2), 0.12 * np.sin(q1) + 0.12 * np.sin(q2)], -1)

    def obs(self, x, tg):
        if self.name == "reacher":
            return np.concatenate([x[..., 1:3], tg - self.finger(x), x[..., 3:5]], -1)
        return x[..., 1:5]

    def reward(self, x, u, tg):
        """Stock reward from the state after the control and the applied control."""
        if self.name == "reacher":
            return (np.linalg.norm(tg - self.finger(x), axis=-1) <= self.ts + self.fs).astype(float)
        d = np.linalg.norm(x[..., 1:3], axis=-1)
        near = np.where(d <= self.ts, 1.0, np.exp(-0.5 * ((d - self.ts) / self.ts * _SCALE) ** 2))
        small = (np.where(np.abs(u) < 1, 1 - u ** 2, 0.0).mean(-1) + 4) / 5
        return near * small

    def parity(self):
        Z = self.Z; worst = dict(state=0.0, obs=0.0, reward=0.0, obs0=0.0)
        for e in range(len(Z["ref_U"])):
            x = np.r_[0.0, Z["ref_q0"][e], Z["ref_v0"][e]][None]; tg = Z["ref_tg"][e][None]
            worst["obs0"] = max(worst["obs0"], np.abs(self.obs(x, tg)[0] - Z["ref_O0"][e]).max())
            for t in range(T):
                u = Z["ref_U"][e][t][None]; x = self.step(x, u)
                worst["state"] = max(worst["state"], np.abs(x[0, 1:3] - Z["ref_Q"][e][t]).max(), np.abs(x[0, 3:5] - Z["ref_V"][e][t]).max())
                worst["obs"] = max(worst["obs"], np.abs(self.obs(x, tg)[0] - Z["ref_O"][e][t]).max())
                worst["reward"] = max(worst["reward"], abs(float(self.reward(x, u, tg)[0]) - float(Z["ref_R"][e][t])))
        return {k: float(v) for k, v in worst.items()}
