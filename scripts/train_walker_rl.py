"""TD3 trainer for Walker2d — produces the independent 'trained policy'.

Purpose (decision log #16): study MPPI<->policy compatibility from the other
direction — start from an RL-trained policy with a proper alternating gait
and measure how the merge/mixing machinery shares control with MPPI —
instead of bootstrapping the policy from MPPI demonstrations (which
collapsed the gait's left/right mode structure into a peg-leg shuffle).

Reward is the verified gymnasium Walker2d-v5 form: vx + healthy_bonus -
1e-3*||a||^2, terminating on unhealthy. The actor IS a DeterministicPolicy,
so checkpoints load directly into the GPS machinery (same featurize /
normalization path, stats left at identity).

Usage: PYTHONPATH=. python scripts/train_walker_rl.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

from src.envs.walker2d import Walker2d
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import PolicyConfig


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()

        def mlp():
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1, self.q2 = mlp(), mlp()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class Replay:
    def __init__(self, obs_dim, act_dim, size=1_000_000):
        self.obs = np.empty((size, obs_dim), dtype=np.float32)
        self.act = np.empty((size, act_dim), dtype=np.float32)
        self.rew = np.empty(size, dtype=np.float32)
        self.nobs = np.empty((size, obs_dim), dtype=np.float32)
        self.done = np.empty(size, dtype=np.float32)
        self.size, self.ptr, self.cap = 0, 0, size

    def add(self, o, a, r, no, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = o, a, r
        self.nobs[i], self.done[i] = no, d
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        t = lambda x: torch.as_tensor(x[idx])
        return t(self.obs), t(self.act), t(self.rew), t(self.nobs), t(self.done)


def main(
    total_steps: int = 1_000_000,
    start_steps: int = 10_000,
    expl_noise: float = 0.1,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_delay: int = 2,
    target_noise: float = 0.2,
    noise_clip: float = 0.5,
    eval_every: int = 20_000,
    out_dir: str = "runs/walker_td3",
    seed: int = 0,
    match_task_cost: bool = False,
    resume: bool = False,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env, eval_env = Walker2d(), Walker2d()
    obs_dim, act_dim = 17, 6

    actor = DeterministicPolicy(obs_dim, act_dim, PolicyConfig())
    actor_targ = DeterministicPolicy(obs_dim, act_dim, PolicyConfig())
    actor_targ.load_state_dict(actor.state_dict())
    critic, critic_targ = Critic(obs_dim, act_dim), Critic(obs_dim, act_dim)
    critic_targ.load_state_dict(critic.state_dict())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    replay = Replay(obs_dim, act_dim)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    start_step = 0
    state_path = out / "train_state.pt"
    if resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(st["actor"])
        actor_targ.load_state_dict(st["actor_targ"])
        critic.load_state_dict(st["critic"])
        critic_targ.load_state_dict(st["critic_targ"])
        actor_opt.load_state_dict(st["actor_opt"])
        critic_opt.load_state_dict(st["critic_opt"])
        rp = np.load(out / "replay.npz")
        n = int(rp["size"])
        replay.obs[:n], replay.act[:n], replay.rew[:n] = rp["obs"], rp["act"], rp["rew"]
        replay.nobs[:n], replay.done[:n] = rp["nobs"], rp["done"]
        replay.size, replay.ptr = n, int(rp["ptr"])
        start_step = int(st["step"])
        print(f"resumed from step {start_step} (replay size {n})", flush=True)

    def reward_fn(env, action):
        if match_task_cost:
            # same optimum as the MPPI task cost (peak exactly at the 1.5
            # m/s target) but with a monotone below-target slope: the
            # symmetric |1.5-vx| version plateaued in a timid 1.2 m/s gait
            # (weak gradient vs the alive bonus), while the v5 max-vx
            # reward trains reliably but optimizes a different objective
            # (gps10: certificate correctly distrusted it, BC destroyed it)
            vx = float(env.data.qvel[0])
            angle = float(env.data.qpos[2])
            vel_term = min(vx, 1.5) - 0.5 * max(vx - 1.5, 0.0)
            return 1.0 + vel_term - 0.1 * angle**2 - 1e-3 * float(np.sum(action**2))
        # gymnasium Walker2d-v5: forward + alive - ctrl (alive only while
        # healthy; episodes terminate on unhealthy so the bonus is constant)
        return float(env.data.qvel[0]) + 1.0 - 1e-3 * float(np.sum(action**2))

    obs = env.reset()
    ep_len, t0 = 0, time.time()
    for step in range(start_step, total_steps):
        if step < start_steps:
            a = np.random.uniform(-1, 1, act_dim)
        else:
            with torch.no_grad():
                a = actor.forward(
                    torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                ).squeeze(0).numpy()
            a = np.clip(a + expl_noise * np.random.randn(act_dim), -1, 1)
        _, _, done, _ = env.step(a)
        nobs = env._get_obs()
        r = reward_fn(env, a)
        ep_len += 1
        timeout = ep_len >= 1000
        replay.add(obs, a, r, nobs, float(done and not timeout))
        obs = nobs
        if done or timeout:
            obs = env.reset()
            ep_len = 0

        if step >= start_steps:
            o, ac, r_, no, d = replay.sample(batch_size)
            with torch.no_grad():
                noise = (target_noise * torch.randn_like(ac)).clamp(-noise_clip, noise_clip)
                na = (actor_targ.forward(no) + noise).clamp(-1, 1)
                tq1, tq2 = critic_targ(no, na)
                target = r_.unsqueeze(-1) + gamma * (1 - d.unsqueeze(-1)) * torch.min(tq1, tq2)
            q1, q2 = critic(o, ac)
            closs = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            critic_opt.zero_grad()
            closs.backward()
            critic_opt.step()
            if step % policy_delay == 0:
                aloss = -critic(o, actor.forward(o))[0].mean()
                actor_opt.zero_grad()
                aloss.backward()
                actor_opt.step()
                with torch.no_grad():
                    for p, tp in zip(actor.parameters(), actor_targ.parameters()):
                        tp.mul_(1 - tau).add_(tau * p)
                    for p, tp in zip(critic.parameters(), critic_targ.parameters()):
                        tp.mul_(1 - tau).add_(tau * p)

        if (step + 1) % eval_every == 0:
            rets, lens, vxs = [], [], []
            for es in range(3):
                np.random.seed(1000 + es)
                eval_env.reset()
                R, L = 0.0, 0
                for _ in range(1000):
                    with torch.no_grad():
                        ea = actor.forward(
                            torch.as_tensor(eval_env._get_obs(), dtype=torch.float32).unsqueeze(0)
                        ).squeeze(0).numpy()
                    _, _, ed, _ = eval_env.step(ea)
                    R += reward_fn(eval_env, ea)
                    L += 1
                    vxs.append(float(eval_env.data.qvel[0]))
                    if ed:
                        break
                rets.append(R)
                lens.append(L)
            sps = (step + 1) / (time.time() - t0)
            print(f"RL step={step+1} eval_return={np.mean(rets):.0f} "
                  f"eval_len={np.mean(lens):.0f}/1000 vx={np.mean(vxs):.2f} "
                  f"({sps:.0f} steps/s)", flush=True)
            torch.save(actor.state_dict(), out / "actor_latest.pt")
            torch.save(actor.state_dict(), out / f"actor_{step+1:07d}.pt")
            # container restarts kill long runs (3 observed); full state for
            # --resume, replay saved uncompressed for write speed
            torch.save({
                "actor": actor.state_dict(), "actor_targ": actor_targ.state_dict(),
                "critic": critic.state_dict(), "critic_targ": critic_targ.state_dict(),
                "actor_opt": actor_opt.state_dict(), "critic_opt": critic_opt.state_dict(),
                "step": step + 1,
            }, state_path)
            n = replay.size
            np.savez(out / "replay.npz", obs=replay.obs[:n], act=replay.act[:n],
                     rew=replay.rew[:n], nobs=replay.nobs[:n], done=replay.done[:n],
                     size=n, ptr=replay.ptr)


if __name__ == "__main__":
    tyro.cli(main)
