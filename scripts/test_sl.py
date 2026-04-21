"""Pure SL behavior cloning with MSE loss on MPPI executed trajectories.

Loads (M, T) executed (state, action) pairs from collect_bc_demos.py and fits
the policy mean by MSE. No weights, no NLL — just regression.
"""

import numpy as np
import h5py
import torch
import mujoco
import mediapy
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

from src.policy.gaussian_policy import GaussianPolicy, HistoryGaussianPolicy
from src.policy.deterministic_policy import DeterministicPolicy
from src.utils.config import PolicyConfig, MPPIConfig
from src.utils.eval import evaluate_policy
from src.envs.acrobot import Acrobot
from src.mppi.mppi import MPPI
import torch.nn.functional as F


@dataclass
class BCConfig:
    demo_path:   Path        = Path("data/acrobot_bc.h5")
    runs_root:   Path        = Path("runs")
    run_name:    str | None  = None   # auto-derived in main() from use_history if unset

    obs_dim:     int   = 6
    act_dim:     int   = 1

    batch_size:  int   = 128
    num_epochs:  int   = 200
    val_frac:    float = 0.2       # fraction of trajectories held out
    n_eval_eps:  int   = 10
    eval_ep_len: int   = 500

    ema_alpha:   float = 0.5       # causal EMA on action targets; 0.0 disables

    seed:        int   = 0


# data
def load_demos(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (M, T, obs_dim), (M, T, act_dim) — preserves trajectory structure
    so we can split train/val by trajectory before flattening."""
    with h5py.File(path, "r") as f:
        states  = f["states"][:].astype(np.float32)
        actions = f["actions"][:].astype(np.float32)
    return states, actions


def smooth_actions_ema(actions: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA along the time axis of a (M, T, act_dim) array.

    y[t] = alpha * y[t-1] + (1 - alpha) * x[t],  y[0] = x[0]

    Rationale: MPPI's per-step action output carries noise roughly as large
    as the signal (see /tmp/diag.py). Training MSE can't go below the
    irreducible per-step variance; smoothing the targets strips that noise
    so the plateau falls. Only the training targets are smoothed — the
    stored states still reflect the unsmoothed environment transitions.
    """
    if alpha <= 0.0:
        return actions
    out = np.empty_like(actions)
    out[:, 0] = actions[:, 0]
    for t in range(1, actions.shape[1]):
        out[:, t] = alpha * out[:, t - 1] + (1.0 - alpha) * actions[:, t]
    return out

def make_windowed_dataset(
        states:  np.ndarray,  # (M, T, obs_dim)
        actions: np.ndarray,  # (M, T, act_dim)
        K: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every (trajectory i, step t) produce one training example:
    obs_hist[n]      = states[i,  t-K+1 .. t]     (zero-padded at front for t < K-1)
    prev_act_hist[n] = actions[i, t-K   .. t-1]   (zero-padded at front for t < K)
    target[n]        = actions[i, t]
    Shapes: (M*T, K, obs_dim), (M*T, K, act_dim), (M*T, act_dim).
    """
    M, T, obs_dim = states.shape 
    act_dim = actions.shape[-1]

    # pad the obs by K-1 for the t that's smaller than K-1 
    obs_pad = np.zeros((M, K-1, obs_dim), dtype = states.dtype)
    states_padded = np.concatenate([obs_pad, states], axis=1)

    # pad actions by K at the front → slice [t : t+K] covers original steps t-K..t-1
    act_pad        = np.zeros((M, K, act_dim), dtype=actions.dtype)
    actions_padded = np.concatenate([act_pad, actions], axis=1)     # (M, T+K, act_dim)

    # time_idx[t] = [t, t+1, ..., t+K-1]  → shape (T, K)
    time_idx = np.arange(T)[:, None] + np.arange(K)[None, :]

    # states_padded[:, time_idx, :]  →  (M, T, K, obs_dim)
    obs_hist      = states_padded[:, time_idx, :].reshape(M * T, K, obs_dim)
    prev_act_hist = actions_padded[:, time_idx, :].reshape(M * T, K, act_dim)
    targets       = actions.reshape(M * T, act_dim)

    return obs_hist, prev_act_hist, targets


    
def split_and_flatten(states: np.ndarray,
                      actions: np.ndarray,
                      val_frac: float,
                      rng: np.random.Generator):
    """Split trajectories (not transitions) into train/val, then flatten each
    half to (N, obs_dim) / (N, act_dim)."""
    M = states.shape[0]
    perm = rng.permutation(M)
    n_val = max(1, int(M * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    def flatten(idx):
        s = states[idx].reshape(-1, states.shape[-1])
        a = actions[idx].reshape(-1, actions.shape[-1])
        return s, a

    tr_s, tr_a = flatten(train_idx)
    va_s, va_a = flatten(val_idx)
    return (tr_s, tr_a), (va_s, va_a), len(train_idx), len(val_idx)


# training
def train_step_mse(policy: DeterministicPolicy,
                   obs: np.ndarray,
                   actions: np.ndarray) -> float:
    """One Adam step on the MSE between policy output and target action."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    act_t = torch.as_tensor(actions, dtype=torch.float32)

    mu = policy.forward(obs_t)
    loss = F.mse_loss(mu, act_t)

    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_mse(policy: DeterministicPolicy,
             obs: np.ndarray,
             actions: np.ndarray,
             batch: int = 16384) -> float:
    """Mean MSE over a dataset, computed in chunks to bound memory."""
    total, n = 0.0, 0
    for s in range(0, len(obs), batch):
        o = torch.as_tensor(obs[s:s + batch], dtype=torch.float32)
        a = torch.as_tensor(actions[s:s + batch], dtype=torch.float32)
        mu = policy.forward(o)
        total += F.mse_loss(mu, a, reduction = "sum")
        n     += a.numel()
    return total / max(n, 1)


def train_step_mse_history(policy: HistoryGaussianPolicy,
                           obs_hist: np.ndarray,
                           prev_act_hist: np.ndarray,
                           targets: np.ndarray) -> float:
    oh  = torch.as_tensor(obs_hist,      dtype=torch.float32)
    ph  = torch.as_tensor(prev_act_hist, dtype=torch.float32)
    tgt = torch.as_tensor(targets,       dtype=torch.float32)

    mu, _ = policy.forward(oh, ph)
    loss  = F.mse_loss(mu, tgt)

    policy.optimizer.zero_grad()
    loss.backward()
    policy.optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_mse_history(policy: HistoryGaussianPolicy,
                     obs_hist: np.ndarray,
                     prev_act_hist: np.ndarray,
                     targets: np.ndarray,
                     batch: int = 16384) -> float:
    total, n = 0.0, 0
    for s in range(0, len(obs_hist), batch):
        oh  = torch.as_tensor(obs_hist[s:s + batch],      dtype=torch.float32)
        ph  = torch.as_tensor(prev_act_hist[s:s + batch], dtype=torch.float32)
        tgt = torch.as_tensor(targets[s:s + batch],       dtype=torch.float32)
        mu, _ = policy.forward(oh, ph)
        total += F.mse_loss(mu, tgt, reduction="sum")
        n     += tgt.numel()
    return total / max(n, 1)


def evaluate_policy_history(policy: HistoryGaussianPolicy,
                            env: Acrobot,
                            n_episodes: int,
                            episode_len: int,
                            seed: int,
                            render: bool = False) -> dict:
    """Closed-loop rollout with a ring buffer for the (obs, prev_action) window.

    obs_buf[-1] is always the current obs; act_buf[-1] is the most recent
    prev action (zero-padded for t < K).
    """
    K           = policy.K
    raw_obs_dim = env._get_obs().shape[0]   # (4,) for acrobot — raw, pre-featurization
    act_dim     = policy.act_dim

    returns: list[float] = []
    frames:  list[np.ndarray] = []
    renderer = mujoco.Renderer(env.model, height=480, width=640) if render else None

    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()

        obs_buf = np.zeros((K, raw_obs_dim), dtype=np.float32)
        act_buf = np.zeros((K, act_dim),     dtype=np.float32)

        ep_cost = 0.0
        for t in range(episode_len):
            # slide current obs into the window
            obs_buf = np.roll(obs_buf, -1, axis=0)
            obs_buf[-1] = env._get_obs()

            obs_hist      = torch.as_tensor(obs_buf, dtype=torch.float32).unsqueeze(0)  # (1, K, obs_dim)
            prev_act_hist = torch.as_tensor(act_buf, dtype=torch.float32).unsqueeze(0)  # (1, K, act_dim)
            with torch.no_grad():
                mu, _ = policy.forward(obs_hist, prev_act_hist)
            action = mu.squeeze(0).numpy()

            _, cost, done, _ = env.step(action)
            ep_cost += cost

            # record this action as the most recent prev action for next step
            act_buf = np.roll(act_buf, -1, axis=0)
            act_buf[-1] = action

            if renderer is not None and ep == 0:
                renderer.update_scene(env.data)
                frames.append(renderer.render().copy())

            if done:
                break
        returns.append(ep_cost)

    arr = np.array(returns)
    return {
        "mean_cost": float(arr.mean()),
        "std_cost":  float(arr.std()),
        "per_ep":    arr.tolist(),
        "frames":    frames,
    }


def evaluate_mppi(env: Acrobot,
                  controller: MPPI,
                  n_episodes: int,
                  episode_len: int,
                  seed: int) -> dict:
    """Same eval protocol as evaluate_policy but stepping with MPPI.

    Identical seed schedule (seed + ep) → episode i starts from the *exact*
    same initial condition as episode i of evaluate_policy, so the per-episode
    gap is apples-to-apples.
    """
    returns: list[float] = []
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        env.reset()
        controller.reset()

        ep_cost = 0.0
        for t in range(episode_len):
            state = env.get_state()
            action, _ = controller.plan_step(state)
            _, cost, done, _ = env.step(action)
            ep_cost += cost
            if done:
                break
        returns.append(ep_cost)

    arr = np.array(returns)
    return {
        "mean_cost": float(arr.mean()),
        "std_cost":  float(arr.std()),
        "per_ep":    arr.tolist(),
    }


def main(cfg: BCConfig = BCConfig()) -> None:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    policy_cfg  = PolicyConfig()
    use_history = policy_cfg.use_history
    K           = policy_cfg.history_len

    run_name = cfg.run_name or ("bc_history" if use_history else "bc_mlp")
    run_dir  = cfg.runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = run_dir / "checkpoint.pt"
    loss_plot  = run_dir / "loss.png"
    video_path = run_dir / "policy.mp4"
    print(f"run_dir: {run_dir}")

    states, actions = load_demos(cfg.demo_path)
    M_all, T, _ = states.shape
    print(f"loaded {M_all} trajectories of length {T}")

    # split by trajectory first so windows don't leak across train/val
    perm  = rng.permutation(M_all)
    n_val = max(1, int(M_all * cfg.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    n_tr, n_va = len(train_idx), len(val_idx)
    print(f"train trajs: {n_tr}  val trajs: {n_va}")

    if use_history:
        tr_obs, tr_pact, tr_tgt = make_windowed_dataset(
            states[train_idx], actions[train_idx], K,
        )
        va_obs, va_pact, va_tgt = make_windowed_dataset(
            states[val_idx], actions[val_idx], K,
        )
        policy = HistoryGaussianPolicy(cfg.obs_dim, cfg.act_dim, policy_cfg)
        print(f"train windows: {len(tr_obs):,}   val windows: {len(va_obs):,}")
    else:
        tr_s = states[train_idx].reshape(-1, states.shape[-1])
        tr_a = actions[train_idx].reshape(-1, actions.shape[-1])
        va_s = states[val_idx].reshape(-1, states.shape[-1])
        va_a = actions[val_idx].reshape(-1, actions.shape[-1])
        policy = DeterministicPolicy(cfg.obs_dim, cfg.act_dim, policy_cfg)
        print(f"train samples: {len(tr_s):,}   val samples: {len(va_s):,}")

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val = float("inf")

    N = len(tr_obs) if use_history else len(tr_s)
    idx = np.arange(N)

    for epoch in range(cfg.num_epochs):
        rng.shuffle(idx)
        running, n_batches = 0.0, 0
        for start in range(0, N, cfg.batch_size):
            b = idx[start:start + cfg.batch_size]
            if use_history:
                running += train_step_mse_history(
                    policy, tr_obs[b], tr_pact[b], tr_tgt[b],
                )
            else:
                running += train_step_mse(policy, tr_s[b], tr_a[b])
            n_batches += 1

        train_losses.append(running / n_batches)
        if use_history:
            val_losses.append(eval_mse_history(policy, va_obs, va_pact, va_tgt))
        else:
            val_losses.append(eval_mse(policy, va_s, va_a))

        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            torch.save(policy.state_dict(), ckpt_path)
            tag = "  ↳ new best"
        else:
            tag = ""

        print(f"epoch {epoch:3d}  "
              f"train_mse={train_losses[-1]:.5f}  "
              f"val_mse={val_losses[-1]:.5f}{tag}")

    # loss curve
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses,   label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_plot, dpi=120)
    print(f"saved loss curve to {loss_plot}")

    # reload best-val checkpoint before env eval
    policy.load_state_dict(torch.load(ckpt_path))
    policy.eval()
    print(f"reloaded best-val checkpoint (val_mse={best_val:.5f})")

    # multi-seed env eval
    env = Acrobot()
    eval_fn = evaluate_policy_history if use_history else evaluate_policy
    stats = eval_fn(
        policy, env,
        n_episodes=cfg.n_eval_eps,
        episode_len=cfg.eval_ep_len,
        seed=cfg.seed,
        render=True,
    )

    # MPPI baseline on the exact same initial conditions
    mppi = MPPI(env, cfg=MPPIConfig.load("acrobot"))
    mppi_stats = evaluate_mppi(
        env, mppi,
        n_episodes=cfg.n_eval_eps,
        episode_len=cfg.eval_ep_len,
        seed=cfg.seed,
    )

    print()
    print(f"BC  policy:    {stats['mean_cost']:8.2f} ± {stats['std_cost']:.2f}")
    print(f"MPPI baseline: {mppi_stats['mean_cost']:8.2f} ± {mppi_stats['std_cost']:.2f}")
    print(f"gap (BC - MPPI): {stats['mean_cost'] - mppi_stats['mean_cost']:+8.2f}")
    print()
    print("per-episode (BC vs MPPI):")
    for ep, (b, m) in enumerate(zip(stats["per_ep"], mppi_stats["per_ep"])):
        print(f"  ep {ep:2d}  BC={b:8.2f}  MPPI={m:8.2f}  gap={b - m:+8.2f}")

    if stats["frames"]:
        mediapy.write_video(str(video_path), stats["frames"], fps=30)
        print(f"saved rollout video to {video_path}")

    env.close()


if __name__ == "__main__":
    main()
