"""Batched MPPI-GPS E-step on the GPU (MuJoCo Warp), for Walker.

Computes what `e_step(planner, env, x, nominal, warmstart,
terminal=WalkerPolicyTail(env, policy, T))` computes, for many states at once:
the policy's closed-loop nominal, K noisy open-loop rollouts of H controls,
the policy's closed-loop tail after each, the n_eff-targeted softmin and the
label. Every rollout is a GPU world. Each control step's physics runs as one
captured CUDA graph, with the policy evaluated in torch between launches.
Costs come from the env's own `rollout_cost` on the copied-back trajectories,
so the objective is the CPU one; only the physics is float32 (and, on GPU,
not bitwise repeatable).
"""
from dataclasses import dataclass

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.math import effective_sample_size, softmin_weights, temperature_for_neff_fraction


class _Worlds:
    """`nworld` copies of the model; one captured graph per control step."""

    def __init__(self, env: Walker2d, nworld: int) -> None:
        self.nworld = nworld
        self.model = mjw.put_model(env.model)
        self.data = mjw.make_data(env.model, nworld=nworld)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.ctrl = wp.to_torch(self.data.ctrl)
        self.warmstart = wp.to_torch(self.data.qacc_warmstart)
        self.time = wp.to_torch(self.data.time)
        with wp.ScopedCapture() as capture:
            for _ in range(env._frame_skip):
                mjw.step(self.model, self.data)
        self.graph = capture.graph

    def load(self, qpos: torch.Tensor, qvel: torch.Tensor, warmstart: torch.Tensor) -> None:
        """Set the first len(qpos) worlds; pad the rest by repetition."""
        index = torch.arange(self.nworld, device=qpos.device) % len(qpos)
        self.qpos.copy_(qpos[index])
        self.qvel.copy_(qvel[index])
        self.warmstart.copy_(warmstart[index])
        self.time.zero_()

    def obs(self, n: int) -> torch.Tensor:
        """Walker2d.rollout_states_to_obs for the first n worlds."""
        return torch.cat([self.qpos[:n, 1:], self.qvel[:n].clamp(-10.0, 10.0)], dim=1)

    def rollout(self, steps: int, n: int, actions: torch.Tensor | None = None, policy=None):
        """Open loop (actions (n, steps, nu)) or closed loop (policy(obs) -> ctrl).

        Returns qpos (n, steps, nq), qvel (n, steps, nv) after each control
        step, and the controls applied (n, steps, nu).
        """
        qpos, qvel, ctrl = [], [], []
        for t in range(steps):
            u = actions[:, t] if actions is not None else policy(self.obs(n))
            self.ctrl[:n].copy_(u)
            torch.cuda.synchronize()
            wp.capture_launch(self.graph)
            wp.synchronize()
            qpos.append(self.qpos[:n].clone())
            qvel.append(self.qvel[:n].clone())
            ctrl.append(u)
        return torch.stack(qpos, 1), torch.stack(qvel, 1), torch.stack(ctrl, 1)


def _full_states(qpos: torch.Tensor, qvel: torch.Tensor) -> np.ndarray:
    """[time, qpos, qvel] float64, the layout env.rollout_cost indexes."""
    q, v = qpos.double().cpu().numpy(), qvel.double().cpu().numpy()
    return np.concatenate([np.zeros(q.shape[:2] + (1,)), q, v], axis=-1)


@dataclass
class GPULabels:
    labels: np.ndarray
    lam: np.ndarray
    n_eff: np.ndarray
    sample_mean_cost: np.ndarray
    expected_cost: np.ndarray
    nominal_cost: np.ndarray
    plan_cost: np.ndarray


class WalkerGPUEStep:
    def __init__(
        self,
        env: Walker2d,
        actor: torch.nn.Module,
        cfg: MPPIConfig,
        tail_steps: int,
        states_per_batch: int = 64,
    ) -> None:
        if cfg.lam_target_neff_fraction is None:
            raise ValueError("The E-step needs an n_eff-targeted temperature.")
        self.env, self.actor, self.cfg = env, actor, cfg
        self.tail_steps, self.batch = tail_steps, states_per_batch
        self.samples = _Worlds(env, states_per_batch * cfg.K)
        self.singles = _Worlds(env, 2 * states_per_batch)
        low, high = env.action_bounds
        self.low_np, self.high_np = low, high
        self.low = torch.as_tensor(low, dtype=torch.float32, device="cuda")
        self.high = torch.as_tensor(high, dtype=torch.float32, device="cuda")

    @torch.no_grad()
    def policy(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.actor(obs)
        action = torch.tanh(mean) * self.actor.action_scale + self.actor.action_bias
        return torch.clamp(action, self.low, self.high)

    def _cost(self, worlds: _Worlds, n: int, actions: np.ndarray) -> np.ndarray:
        """H-step cost of open-loop `actions`, plus the policy tail (absorbing if fallen)."""
        env = self.env
        u = torch.as_tensor(actions, dtype=torch.float32, device="cuda")
        hq, hv, _ = worlds.rollout(self.cfg.H, n, actions=u)
        states = _full_states(hq, hv)
        sensors = np.zeros(states.shape[:2] + (0,))
        cost = env.rollout_cost(states, actions, sensors)
        if not self.tail_steps:
            return cost
        fell = ~np.all(env._is_healthy(states[..., 2], states[..., 3]), axis=1)
        tq, tv, tu = worlds.rollout(self.tail_steps, n, policy=self.policy)
        tail_states = _full_states(tq, tv)
        tail = env.rollout_cost(tail_states, tu.double().cpu().numpy(), np.zeros(tail_states.shape[:2] + (0,)))
        tail[fell] = env._w_unhealthy * self.tail_steps
        return cost + tail

    def __call__(
        self,
        states: np.ndarray,
        warmstarts: np.ndarray,
        noise_seeds: list[int] | None = None,
        noise_streams: list[tuple[int, int]] | None = None,
    ) -> GPULabels:
        """Label `states`. Noise for state i is the first draw of an MPPI
        planner seeded noise_seeds[i], or, with noise_streams[i] = (seed, j),
        the j-th draw of one planner per seed: the stream the CPU training path
        uses for the j-th labelled state of an episode."""
        env, cfg, k, h = self.env, self.cfg, self.cfg.K, self.cfg.H
        nq, nv = env.model.nq, env.model.nv
        rows = {name: [] for name in GPULabels.__dataclass_fields__}
        planners: dict[int, list] = {}

        def noise(i: int) -> np.ndarray:
            if noise_streams is None:
                return MPPI(env, cfg, seed=noise_seeds[i])._sample_noise()
            seed, draw = noise_streams[i]
            planner = planners.setdefault(seed, [MPPI(env, cfg, seed=seed), 0])
            if draw != planner[1]:
                raise ValueError(f"noise stream {seed} expected draw {planner[1]}, got {draw}.")
            planner[1] += 1
            return planner[0]._sample_noise()

        for start in range(0, len(states), self.batch):
            stop = min(start + self.batch, len(states))
            n = stop - start
            gpu = lambda a: torch.as_tensor(a, dtype=torch.float32, device="cuda")
            q0, v0 = gpu(states[start:stop, 1:1 + nq]), gpu(states[start:stop, 1 + nq:1 + nq + nv])
            w0 = gpu(warmstarts[start:stop])
            self.singles.load(q0, v0, w0)
            _, _, nominal = self.singles.rollout(h, n, policy=self.policy)
            nominal = nominal.double().cpu().numpy()
            draws = np.stack([noise(i) for i in range(start, stop)])
            sampled = np.clip(nominal[:, None] + draws, self.low_np, self.high_np)
            self.samples.load(q0.repeat_interleave(k, 0), v0.repeat_interleave(k, 0), w0.repeat_interleave(k, 0))
            scores = self._cost(self.samples, n * k, sampled.reshape(n * k, h, -1)).reshape(n, k)
            plans = np.empty_like(nominal)
            for i in range(n):
                lam = temperature_for_neff_fraction(scores[i], cfg.lam_target_neff_fraction)
                weights = softmin_weights(scores[i], lam)
                eps = sampled[i] - nominal[i]
                plans[i] = np.clip(nominal[i] + np.einsum("k,kha->ha", weights, eps), self.low_np, self.high_np)
                finite = np.isfinite(scores[i])
                rows["labels"].append(plans[i, 0].copy())
                rows["lam"].append(lam)
                rows["n_eff"].append(effective_sample_size(weights))
                rows["sample_mean_cost"].append(np.mean(scores[i][finite]))
                rows["expected_cost"].append(np.sum(weights[finite] * scores[i][finite]))
            self.singles.load(torch.cat([q0, q0]), torch.cat([v0, v0]), torch.cat([w0, w0]))
            both = self._cost(self.singles, 2 * n, np.concatenate([nominal, plans]))
            rows["nominal_cost"].extend(both[:n])
            rows["plan_cost"].extend(both[n:])
        return GPULabels(**{name: np.asarray(values) for name, values in rows.items()})
