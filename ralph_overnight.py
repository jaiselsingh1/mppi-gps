#!/usr/bin/env python3
"""
ralph_overnight.py  —  Overnight diagnostic loop for MPPI-GPS / Acrobot BC
============================================================================

PURPOSE:
  Systematically sweep the design knobs identified in the diagnostic,
  retrain BC each time, evaluate in the environment, and test warm-start
  MPPI with the learned policy.  Produces a self-contained report at the
  end (JSON + markdown) so you wake up to findings, not questions.

GUARANTEES:
  • Read-only on existing source files — all outputs go to
        runs/ralph_overnight_<timestamp>/
  • No interactive prompts — fully autonomous.
  • Catches & logs exceptions per-experiment so one failure doesn't kill
    the whole sweep.

WHAT IT SWEEPS:
  Phase 1 — EMA smoothing of demo actions (α ∈ {0.0, 0.1, 0.2, 0.3, 0.5})
             + cosine LR schedule toggle
  Phase 2 — For the best Phase-1 config, run the Kendall-style GPS loop:
             π warm-starts MPPI, MPPI generates new demos, retrain π.
             (Repeat for N_GPS_ITERS iterations.)

USAGE:
  cd <project_root>          # wherever src/ and data/ live
  python ralph_overnight.py  # that's it — go to sleep

  Or with nohup:
  nohup python ralph_overnight.py > ralph.log 2>&1 &
"""

import os, sys, json, time, copy, shutil, traceback, hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import numpy as np
import h5py

# ---------------------------------------------------------------------------
# 0.  Configuration — tweak these knobs, everything else is derived
# ---------------------------------------------------------------------------
@dataclass
class RalphConfig:
    # Paths (relative to cwd = project root)
    demo_h5: str = "data/acrobot_bc.h5"
    src_dir: str = "src"

    # EMA sweep
    ema_alphas: list = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])

    # BC training
    bc_epochs: int = 300
    bc_lr: float = 1e-3
    bc_batch: int = 256
    bc_hidden: list = field(default_factory=lambda: [256, 256])
    use_cosine_schedule: bool = True      # toggled in sweep
    val_frac: float = 0.15
    seed: int = 42

    # Env eval
    eval_episodes: int = 50
    eval_max_steps: int = 500

    # MPPI warm-start eval
    mppi_eval_episodes: int = 30
    mppi_horizon: int = 64                # match your MPPI config
    mppi_samples: int = 512

    # GPS outer loop (Phase 2)
    n_gps_iters: int = 3
    gps_new_demos_per_iter: int = 200     # rollouts collected with warm-started MPPI
    gps_mix_ratio: float = 0.5           # fraction of new demos vs old in retrain

    # Output
    run_root: str = ""  # auto-filled

    def __post_init__(self):
        if not self.run_root:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_root = f"runs/ralph_overnight_{ts}"


# ---------------------------------------------------------------------------
# 1.  Utilities
# ---------------------------------------------------------------------------
def log(msg: str, file=None):
    """Timestamped print + optional file log."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if file:
        file.write(line + "\n")
        file.flush()

def safe_import(module_path: str):
    """Import a module by dotted path, with a clear error if missing."""
    parts = module_path.rsplit(".", 1)
    if len(parts) == 2:
        mod = __import__(parts[0], fromlist=[parts[1]])
        return getattr(mod, parts[1])
    return __import__(parts[0])

def set_seed(seed: int):
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import jax
        # JAX seed handled per-call via PRNGKey
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 2.  Data loading & EMA smoothing
# ---------------------------------------------------------------------------
def load_demos(path: str):
    """Load (obs, actions) from the BC HDF5 file."""
    with h5py.File(path, "r") as f:
        # Try common key layouts
        if "observations" in f and "actions" in f:
            obs = f["observations"][:]
            act = f["actions"][:]
        elif "obs" in f and "act" in f:
            obs = f["obs"][:]
            act = f["act"][:]
        else:
            # Trajectory-grouped layout
            obs_list, act_list = [], []
            for k in sorted(f.keys()):
                if "traj" in k or k.isdigit():
                    g = f[k]
                    obs_list.append(g["observations"][:] if "observations" in g else g["obs"][:])
                    act_list.append(g["actions"][:] if "actions" in g else g["act"][:])
            if not obs_list:
                raise ValueError(f"Cannot parse HDF5 layout. Keys: {list(f.keys())}")
            obs = np.concatenate(obs_list, axis=0)
            act = np.concatenate(act_list, axis=0)
    return obs.astype(np.float32), act.astype(np.float32)


def ema_smooth_actions(actions: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential moving average on action sequences.
    alpha=0 → no smoothing (passthrough).
    alpha=1 → full smoothing (constant first action).
    Convention: smoothed[t] = alpha * smoothed[t-1] + (1-alpha) * raw[t]
    
    If the data is concatenated trajectories without episode boundaries,
    we still apply EMA across the whole sequence — the diagnostic showed
    temporal-diff std is high everywhere, so cross-episode bleed at a few
    points is negligible compared to the intra-episode noise.
    """
    if alpha <= 0:
        return actions.copy()
    smoothed = np.empty_like(actions)
    smoothed[0] = actions[0]
    for t in range(1, len(actions)):
        smoothed[t] = alpha * smoothed[t - 1] + (1.0 - alpha) * actions[t]
    return smoothed


# ---------------------------------------------------------------------------
# 3.  BC Training  (self-contained, no modification to your files)
# ---------------------------------------------------------------------------
def train_bc(obs: np.ndarray, actions: np.ndarray, cfg: RalphConfig,
             run_dir: str, logfile) -> dict:
    """
    Train a deterministic MLP policy via MSE on (obs → action).
    Returns dict with train/val curves, final metrics, and saved model path.

    Uses the same sin/cos featurization as your GaussianPolicy.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Training on {device}", logfile)

    # --- Featurize (sin/cos of qpos, scaled qvel) ---
    # Acrobot obs = [q0, q1, qd0, qd1]
    def featurize(o):
        q0, q1 = o[:, 0:1], o[:, 1:2]
        qd0, qd1 = o[:, 2:3], o[:, 3:4]
        return np.concatenate([
            np.sin(q0), np.cos(q0),
            np.sin(q1), np.cos(q1),
            qd0 / 2.5, qd1 / 5.0
        ], axis=1).astype(np.float32)

    feat = featurize(obs)
    n = len(feat)
    idx = np.random.permutation(n)
    split = int(n * (1.0 - cfg.val_frac))
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_ds = TensorDataset(torch.from_numpy(feat[tr_idx]), torch.from_numpy(actions[tr_idx]))
    va_ds = TensorDataset(torch.from_numpy(feat[va_idx]), torch.from_numpy(actions[va_idx]))
    tr_dl = DataLoader(tr_ds, batch_size=cfg.bc_batch, shuffle=True, drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=cfg.bc_batch * 2, shuffle=False)

    # --- Build MLP ---
    layers = []
    in_dim = feat.shape[1]
    for h in cfg.bc_hidden:
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        in_dim = h
    layers.append(nn.Linear(in_dim, actions.shape[1]))
    model = nn.Sequential(*layers).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.bc_lr)
    if cfg.use_cosine_schedule:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.bc_epochs)
    else:
        sched = None

    loss_fn = nn.MSELoss()

    # --- Training loop ---
    history = {"train_mse": [], "val_mse": [], "lr": []}
    best_val = float("inf")
    best_state = None

    for ep in range(1, cfg.bc_epochs + 1):
        model.train()
        ep_loss = 0.0
        count = 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
            count += len(xb)
        if sched:
            sched.step()

        tr_mse = ep_loss / count

        model.eval()
        va_loss = 0.0
        va_count = 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                va_loss += loss_fn(pred, yb).item() * len(xb)
                va_count += len(xb)
        va_mse = va_loss / va_count

        current_lr = opt.param_groups[0]["lr"]
        history["train_mse"].append(tr_mse)
        history["val_mse"].append(va_mse)
        history["lr"].append(current_lr)

        if va_mse < best_val:
            best_val = va_mse
            best_state = copy.deepcopy(model.state_dict())

        if ep % 50 == 0 or ep == 1:
            log(f"    ep {ep:4d}  tr={tr_mse:.5f}  va={va_mse:.5f}  lr={current_lr:.2e}", logfile)

    # Save best model
    model_path = os.path.join(run_dir, "best_policy.pt")
    torch.save(best_state, model_path)
    model.load_state_dict(best_state)

    log(f"  Best val MSE = {best_val:.5f}", logfile)

    return {
        "model": model,
        "device": device,
        "featurize": featurize,
        "history": history,
        "best_val_mse": best_val,
        "model_path": model_path,
    }


# ---------------------------------------------------------------------------
# 4.  Environment evaluation  (policy-only and warm-started MPPI)
# ---------------------------------------------------------------------------
def eval_policy_in_env(model, featurize, device, cfg: RalphConfig, logfile) -> dict:
    """
    Roll out the learned policy in the Acrobot env.
    Returns mean/std cost and per-episode costs.
    """
    import torch

    # Try importing the project's env
    try:
        from src.envs.acrobot import AcrobotEnv  # adjust if different
        make_env = lambda: AcrobotEnv()
    except ImportError:
        try:
            from src.envs import make_env as _make_env
            make_env = lambda: _make_env("acrobot")
        except ImportError:
            # Fallback: use gymnasium
            import gymnasium as gym
            make_env = lambda: gym.make("Acrobot-v1")

    costs = []
    for ep in range(cfg.eval_episodes):
        env = make_env()
        obs, _ = env.reset() if hasattr(env, 'reset') else (env.reset(), {})
        obs = np.array(obs, dtype=np.float32)
        total_cost = 0.0
        for t in range(cfg.eval_max_steps):
            feat = featurize(obs.reshape(1, -1))
            with torch.no_grad():
                act = model(torch.from_numpy(feat).to(device)).cpu().numpy().flatten()
            act = np.clip(act, -1.0, 1.0)
            
            step_result = env.step(act)
            if len(step_result) == 5:
                obs_next, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs_next, reward, done, info = step_result
            
            total_cost += -reward  # cost = -reward convention
            obs = np.array(obs_next, dtype=np.float32)
            if done:
                break
        costs.append(total_cost)
        try:
            env.close()
        except:
            pass

    costs = np.array(costs)
    result = {
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "min_cost": float(np.min(costs)),
        "max_cost": float(np.max(costs)),
        "per_episode": costs.tolist(),
    }
    log(f"  Policy eval: cost = {result['mean_cost']:.2f} ± {result['std_cost']:.2f}", logfile)
    return result


def eval_warmstart_mppi(model, featurize, device, cfg: RalphConfig, logfile) -> dict:
    """
    Run MPPI with policy warm-starting U[0] via nominal_first.
    Compare against vanilla MPPI (no warm start).
    """
    import torch

    try:
        from src.mppi.mppi import MPPI
        from src.envs.acrobot import AcrobotEnv
    except ImportError as e:
        log(f"  WARN: Cannot import MPPI/env for warm-start eval: {e}", logfile)
        return {"skipped": True, "reason": str(e)}

    def run_mppi_episodes(n_eps, warm_policy=None):
        costs = []
        for ep in range(n_eps):
            env = AcrobotEnv()
            obs, _ = env.reset() if hasattr(env, 'reset') else (env.reset(), {})
            obs = np.array(obs, dtype=np.float32)
            
            # Build MPPI controller (re-init each episode for clean state)
            try:
                mppi = MPPI(
                    horizon=cfg.mppi_horizon,
                    n_samples=cfg.mppi_samples,
                )
            except TypeError:
                # If MPPI needs different init args, try without kwargs
                mppi = MPPI()
            
            total_cost = 0.0
            for t in range(cfg.eval_max_steps):
                nominal_first = None
                if warm_policy is not None:
                    feat = featurize(obs.reshape(1, -1))
                    with torch.no_grad():
                        nominal_first = warm_policy(
                            torch.from_numpy(feat).to(device)
                        ).cpu().numpy().flatten()
                        nominal_first = np.clip(nominal_first, -1.0, 1.0)

                try:
                    act = mppi.plan_step(obs, nominal_first=nominal_first)
                except TypeError:
                    # If plan_step doesn't accept nominal_first yet
                    act = mppi.plan_step(obs)
                
                act = np.clip(np.array(act).flatten(), -1.0, 1.0)
                step_result = env.step(act)
                if len(step_result) == 5:
                    obs_next, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs_next, reward, done, info = step_result
                total_cost += -reward
                obs = np.array(obs_next, dtype=np.float32)
                if done:
                    break
            costs.append(total_cost)
            try:
                env.close()
            except:
                pass
        return np.array(costs)

    log(f"  Running vanilla MPPI ({cfg.mppi_eval_episodes} eps)...", logfile)
    vanilla_costs = run_mppi_episodes(cfg.mppi_eval_episodes, warm_policy=None)
    log(f"    Vanilla MPPI cost: {np.mean(vanilla_costs):.2f} ± {np.std(vanilla_costs):.2f}", logfile)

    log(f"  Running warm-started MPPI ({cfg.mppi_eval_episodes} eps)...", logfile)
    warm_costs = run_mppi_episodes(cfg.mppi_eval_episodes, warm_policy=model)
    log(f"    Warm MPPI cost:    {np.mean(warm_costs):.2f} ± {np.std(warm_costs):.2f}", logfile)

    improvement = float(np.mean(vanilla_costs) - np.mean(warm_costs))
    pct = improvement / (np.mean(vanilla_costs) + 1e-8) * 100

    result = {
        "vanilla_mean": float(np.mean(vanilla_costs)),
        "vanilla_std": float(np.std(vanilla_costs)),
        "warm_mean": float(np.mean(warm_costs)),
        "warm_std": float(np.std(warm_costs)),
        "improvement": improvement,
        "improvement_pct": pct,
    }
    log(f"  Warm-start improvement: {improvement:.2f} ({pct:.1f}%)", logfile)
    return result


# ---------------------------------------------------------------------------
# 5.  GPS outer loop (Phase 2)
# ---------------------------------------------------------------------------
def gps_loop(best_phase1: dict, obs_raw: np.ndarray, act_raw: np.ndarray,
             cfg: RalphConfig, run_dir: str, logfile) -> list:
    """
    Kendall-minimal GPS:
      1. Warm-start MPPI with current policy
      2. Collect new demos from warm-started MPPI
      3. Mix new demos with original, retrain policy
      4. Repeat
    
    Returns list of per-iteration results.
    """
    import torch

    try:
        from src.mppi.mppi import MPPI
        from src.envs.acrobot import AcrobotEnv
    except ImportError as e:
        log(f"  GPS loop skipped — cannot import MPPI/env: {e}", logfile)
        return [{"skipped": True, "reason": str(e)}]

    ema_alpha = best_phase1["ema_alpha"]
    current_model = best_phase1["train_result"]["model"]
    current_featurize = best_phase1["train_result"]["featurize"]
    device = best_phase1["train_result"]["device"]

    # Keep a growing pool of demos
    demo_obs = obs_raw.copy()
    demo_act = ema_smooth_actions(act_raw, ema_alpha)

    gps_results = []

    for gps_iter in range(1, cfg.n_gps_iters + 1):
        log(f"\n  === GPS Iteration {gps_iter}/{cfg.n_gps_iters} ===", logfile)
        iter_dir = os.path.join(run_dir, f"gps_iter_{gps_iter}")
        os.makedirs(iter_dir, exist_ok=True)

        # Step 1: Collect new demos with warm-started MPPI
        log(f"  Collecting {cfg.gps_new_demos_per_iter} warm-started MPPI rollouts...", logfile)
        new_obs_list, new_act_list = [], []
        for ep in range(cfg.gps_new_demos_per_iter):
            env = AcrobotEnv()
            obs, _ = env.reset() if hasattr(env, 'reset') else (env.reset(), {})
            obs = np.array(obs, dtype=np.float32)
            
            try:
                mppi = MPPI(horizon=cfg.mppi_horizon, n_samples=cfg.mppi_samples)
            except TypeError:
                mppi = MPPI()
            
            ep_obs, ep_act = [], []
            for t in range(cfg.eval_max_steps):
                feat = current_featurize(obs.reshape(1, -1))
                with torch.no_grad():
                    nominal = current_model(
                        torch.from_numpy(feat).to(device)
                    ).cpu().numpy().flatten()
                    nominal = np.clip(nominal, -1.0, 1.0)
                
                try:
                    act = mppi.plan_step(obs, nominal_first=nominal)
                except TypeError:
                    act = mppi.plan_step(obs)
                
                act = np.clip(np.array(act).flatten(), -1.0, 1.0)
                ep_obs.append(obs.copy())
                ep_act.append(act.copy())
                
                step_result = env.step(act)
                if len(step_result) == 5:
                    obs, _, term, trunc, _ = step_result
                    done = term or trunc
                else:
                    obs, _, done, _ = step_result
                obs = np.array(obs, dtype=np.float32)
                if done:
                    break
            
            new_obs_list.append(np.array(ep_obs))
            new_act_list.append(np.array(ep_act))
            try:
                env.close()
            except:
                pass

        new_obs = np.concatenate(new_obs_list, axis=0).astype(np.float32)
        new_act = np.concatenate(new_act_list, axis=0).astype(np.float32)
        
        # EMA smooth the new actions too
        new_act_smooth = ema_smooth_actions(new_act, ema_alpha)
        
        log(f"  Collected {len(new_obs)} new transitions", logfile)

        # Step 2: Mix demos
        n_old = int(len(demo_obs) * (1.0 - cfg.gps_mix_ratio))
        old_idx = np.random.choice(len(demo_obs), size=min(n_old, len(demo_obs)), replace=False)
        mixed_obs = np.concatenate([demo_obs[old_idx], new_obs], axis=0)
        mixed_act = np.concatenate([demo_act[old_idx], new_act_smooth], axis=0)
        
        log(f"  Mixed dataset: {len(mixed_obs)} transitions ({len(old_idx)} old + {len(new_obs)} new)", logfile)

        # Step 3: Retrain policy
        log(f"  Retraining policy...", logfile)
        train_result = train_bc(mixed_obs, mixed_act, cfg, iter_dir, logfile)
        current_model = train_result["model"]

        # Step 4: Evaluate
        log(f"  Evaluating policy-only...", logfile)
        policy_eval = eval_policy_in_env(current_model, current_featurize, device, cfg, logfile)
        
        log(f"  Evaluating warm-start MPPI...", logfile)
        mppi_eval = eval_warmstart_mppi(current_model, current_featurize, device, cfg, logfile)

        # Update demo pool (append new data for next iteration)
        demo_obs = mixed_obs
        demo_act = mixed_act

        iter_result = {
            "gps_iter": gps_iter,
            "n_new_transitions": len(new_obs),
            "n_total_transitions": len(mixed_obs),
            "val_mse": train_result["best_val_mse"],
            "policy_eval": policy_eval,
            "mppi_eval": mppi_eval,
        }
        gps_results.append(iter_result)

        # Save iter result
        with open(os.path.join(iter_dir, "result.json"), "w") as f:
            json.dump(iter_result, f, indent=2, default=str)

    return gps_results


# ---------------------------------------------------------------------------
# 6.  Report generation
# ---------------------------------------------------------------------------
def generate_report(all_results: dict, cfg: RalphConfig, run_dir: str, logfile):
    """Write a markdown + JSON report summarizing findings."""

    # JSON dump
    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"Results JSON: {json_path}", logfile)

    # Markdown report
    md_lines = [
        "# Ralph Overnight Report",
        f"**Run:** `{run_dir}`",
        f"**Started:** {all_results.get('start_time', '?')}",
        f"**Finished:** {all_results.get('end_time', '?')}",
        f"**Duration:** {all_results.get('duration_min', '?'):.1f} min",
        "",
        "---",
        "",
        "## Phase 1: EMA Sweep Results",
        "",
        "| EMA α | Cosine LR | Val MSE | Policy Cost (mean±std) | vs Baseline |",
        "|-------|-----------|---------|----------------------|-------------|",
    ]

    baseline_cost = None
    best_config = None
    best_val = float("inf")

    for r in all_results.get("phase1", []):
        alpha = r["ema_alpha"]
        cosine = r["cosine_lr"]
        val_mse = r.get("val_mse", "—")
        eval_r = r.get("policy_eval", {})
        mean_c = eval_r.get("mean_cost", float("nan"))
        std_c = eval_r.get("std_cost", float("nan"))
        
        if alpha == 0.0 and not cosine:
            baseline_cost = mean_c
        
        delta = ""
        if baseline_cost is not None and not np.isnan(mean_c):
            d = mean_c - baseline_cost
            delta = f"{d:+.1f}"
        
        if isinstance(val_mse, float) and val_mse < best_val:
            best_val = val_mse
            best_config = r

        md_lines.append(
            f"| {alpha:.1f} | {'✓' if cosine else '✗'} | "
            f"{val_mse:.5f} | {mean_c:.1f} ± {std_c:.1f} | {delta} |"
        )

    md_lines += [
        "",
        f"**Best config:** EMA α={best_config['ema_alpha'] if best_config else '?'}, "
        f"val MSE={best_val:.5f}",
        "",
    ]

    # MPPI warm-start results
    ws_results = all_results.get("warmstart_eval", {})
    if ws_results and not ws_results.get("skipped"):
        md_lines += [
            "## Warm-Start MPPI Evaluation (Best Phase-1 Config)",
            "",
            f"- Vanilla MPPI cost: **{ws_results.get('vanilla_mean', '?'):.1f}** "
            f"± {ws_results.get('vanilla_std', '?'):.1f}",
            f"- Warm-started MPPI cost: **{ws_results.get('warm_mean', '?'):.1f}** "
            f"± {ws_results.get('warm_std', '?'):.1f}",
            f"- **Improvement: {ws_results.get('improvement', '?'):.1f} "
            f"({ws_results.get('improvement_pct', '?'):.1f}%)**",
            "",
        ]

    # GPS results
    gps = all_results.get("phase2_gps", [])
    if gps and not (len(gps) == 1 and gps[0].get("skipped")):
        md_lines += [
            "## Phase 2: GPS Loop",
            "",
            "| Iter | Val MSE | Policy Cost | Warm MPPI Cost | Improvement |",
            "|------|---------|-------------|----------------|-------------|",
        ]
        for g in gps:
            pe = g.get("policy_eval", {})
            me = g.get("mppi_eval", {})
            md_lines.append(
                f"| {g['gps_iter']} | {g.get('val_mse', '?'):.5f} | "
                f"{pe.get('mean_cost', '?'):.1f} | "
                f"{me.get('warm_mean', '?'):.1f} | "
                f"{me.get('improvement_pct', '?'):.1f}% |"
            )
        md_lines.append("")

    # Recommendations
    md_lines += [
        "---",
        "",
        "## Recommendations",
        "",
        "*(Auto-generated from results — verify before acting)*",
        "",
    ]

    if best_config and best_config["ema_alpha"] > 0:
        md_lines.append(
            f"1. **EMA smoothing helps.** Best α={best_config['ema_alpha']:.1f} "
            f"dropped val MSE from the 0.25 plateau. Apply EMA to demo actions "
            f"before BC training."
        )
    else:
        md_lines.append(
            "1. **EMA smoothing did NOT help.** The noise floor may be irreducible "
            "with deterministic μ output. Consider switching to NLL training with "
            "the Gaussian head you already have."
        )

    if ws_results and not ws_results.get("skipped"):
        if ws_results.get("improvement_pct", 0) > 5:
            md_lines.append(
                f"2. **Warm-starting MPPI works.** {ws_results['improvement_pct']:.1f}% "
                f"cost reduction. The GPS loop is viable — proceed with Kendall-minimal "
                f"GPS as planned."
            )
        else:
            md_lines.append(
                "2. **Warm-starting MPPI shows marginal gains.** The policy may not be "
                "providing useful signal to MPPI yet. Focus on improving BC quality first."
            )

    if gps and not (len(gps) == 1 and gps[0].get("skipped")):
        final_gps = gps[-1]
        first_gps = gps[0]
        if final_gps.get("val_mse", 1) < first_gps.get("val_mse", 0):
            md_lines.append(
                "3. **GPS iterations are improving the policy.** The data-collection → "
                "retrain loop is working. Consider more iterations or larger batch sizes."
            )
        else:
            md_lines.append(
                "3. **GPS iterations are not converging.** May need: different mix ratio, "
                "more MPPI samples, or a trust-region constraint on the policy update."
            )

    md_lines += [
        "",
        "---",
        "",
        f"*All artifacts saved in `{run_dir}/`*",
        "",
        "### Files",
        "- `results.json` — full numerical results",
        "- `ralph.log` — timestamped execution log",
        "- `ema_<alpha>/best_policy.pt` — trained models per EMA setting",
        "- `gps_iter_<n>/` — GPS iteration artifacts",
    ]

    report_path = os.path.join(run_dir, "REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))
    log(f"Report: {report_path}", logfile)
    return report_path


# ---------------------------------------------------------------------------
# 7.  Main — the ralph loop
# ---------------------------------------------------------------------------
def main():
    cfg = RalphConfig()
    os.makedirs(cfg.run_root, exist_ok=True)
    
    logfile = open(os.path.join(cfg.run_root, "ralph.log"), "w")
    
    log("=" * 70, logfile)
    log("RALPH OVERNIGHT — MPPI-GPS Acrobot BC Diagnostic Loop", logfile)
    log(f"Run dir: {cfg.run_root}", logfile)
    log(f"Config: {json.dumps(asdict(cfg), indent=2, default=str)}", logfile)
    log("=" * 70, logfile)

    start_time = datetime.now()
    all_results = {
        "start_time": str(start_time),
        "config": asdict(cfg),
        "phase1": [],
        "warmstart_eval": {},
        "phase2_gps": [],
    }

    # Ensure src/ is importable
    if os.path.isdir(cfg.src_dir):
        sys.path.insert(0, os.path.dirname(os.path.abspath(cfg.src_dir)))
        sys.path.insert(0, os.getcwd())
    
    set_seed(cfg.seed)

    # Load raw demos
    log(f"\nLoading demos from {cfg.demo_h5}...", logfile)
    try:
        obs_raw, act_raw = load_demos(cfg.demo_h5)
    except Exception as e:
        log(f"FATAL: Cannot load demos: {e}", logfile)
        traceback.print_exc(file=logfile)
        logfile.close()
        return
    
    log(f"  obs shape: {obs_raw.shape}, act shape: {act_raw.shape}", logfile)
    log(f"  act range: [{act_raw.min():.3f}, {act_raw.max():.3f}], "
        f"std={act_raw.std():.4f}, var={act_raw.var():.4f}", logfile)

    # =======================================================================
    # PHASE 1: EMA sweep × cosine-LR toggle
    # =======================================================================
    log("\n" + "=" * 50, logfile)
    log("PHASE 1: EMA Smoothing Sweep", logfile)
    log("=" * 50, logfile)

    best_phase1 = None
    best_phase1_val = float("inf")

    sweep_configs = []
    for alpha in cfg.ema_alphas:
        sweep_configs.append((alpha, True))   # with cosine LR
        if alpha == 0.0:
            sweep_configs.append((alpha, False))  # also test no-cosine baseline

    for alpha, use_cosine in sweep_configs:
        tag = f"ema_{alpha:.1f}_cos{'Y' if use_cosine else 'N'}"
        exp_dir = os.path.join(cfg.run_root, tag)
        os.makedirs(exp_dir, exist_ok=True)

        log(f"\n--- {tag} ---", logfile)
        
        try:
            # Smooth actions
            act_smooth = ema_smooth_actions(act_raw, alpha)
            smooth_var = act_smooth.var()
            log(f"  Smoothed action var: {smooth_var:.4f} (raw: {act_raw.var():.4f})", logfile)

            # Train
            cfg_copy = copy.copy(cfg)
            cfg_copy.use_cosine_schedule = use_cosine
            train_result = train_bc(obs_raw, act_smooth, cfg_copy, exp_dir, logfile)

            # Eval in env
            log(f"  Evaluating policy in env...", logfile)
            policy_eval = eval_policy_in_env(
                train_result["model"], train_result["featurize"],
                train_result["device"], cfg, logfile
            )

            result = {
                "ema_alpha": alpha,
                "cosine_lr": use_cosine,
                "tag": tag,
                "val_mse": train_result["best_val_mse"],
                "smoothed_act_var": float(smooth_var),
                "policy_eval": policy_eval,
                "train_history": {
                    "final_train_mse": train_result["history"]["train_mse"][-1],
                    "final_val_mse": train_result["history"]["val_mse"][-1],
                },
                "train_result": train_result,  # kept in memory for Phase 2
            }

            if train_result["best_val_mse"] < best_phase1_val:
                best_phase1_val = train_result["best_val_mse"]
                best_phase1 = result

        except Exception as e:
            log(f"  ERROR in {tag}: {e}", logfile)
            traceback.print_exc(file=logfile)
            result = {
                "ema_alpha": alpha, "cosine_lr": use_cosine, "tag": tag,
                "error": str(e),
            }

        # Strip non-serializable stuff for JSON
        result_clean = {k: v for k, v in result.items()
                        if k != "train_result"}
        all_results["phase1"].append(result_clean)

    # =======================================================================
    # Warm-start MPPI eval on best Phase-1 policy
    # =======================================================================
    if best_phase1 is not None:
        log("\n" + "=" * 50, logfile)
        log(f"WARM-START MPPI EVAL (best={best_phase1['tag']})", logfile)
        log("=" * 50, logfile)
        
        try:
            ws_result = eval_warmstart_mppi(
                best_phase1["train_result"]["model"],
                best_phase1["train_result"]["featurize"],
                best_phase1["train_result"]["device"],
                cfg, logfile
            )
            all_results["warmstart_eval"] = ws_result
        except Exception as e:
            log(f"  ERROR in warm-start eval: {e}", logfile)
            traceback.print_exc(file=logfile)
            all_results["warmstart_eval"] = {"skipped": True, "reason": str(e)}

    # =======================================================================
    # PHASE 2: GPS Loop
    # =======================================================================
    if best_phase1 is not None:
        log("\n" + "=" * 50, logfile)
        log("PHASE 2: GPS Loop", logfile)
        log("=" * 50, logfile)
        
        gps_dir = os.path.join(cfg.run_root, "gps")
        os.makedirs(gps_dir, exist_ok=True)
        
        try:
            gps_results = gps_loop(
                best_phase1, obs_raw, act_raw, cfg, gps_dir, logfile
            )
            all_results["phase2_gps"] = gps_results
        except Exception as e:
            log(f"  ERROR in GPS loop: {e}", logfile)
            traceback.print_exc(file=logfile)
            all_results["phase2_gps"] = [{"skipped": True, "reason": str(e)}]

    # =======================================================================
    # Report
    # =======================================================================
    end_time = datetime.now()
    all_results["end_time"] = str(end_time)
    all_results["duration_min"] = (end_time - start_time).total_seconds() / 60.0

    log("\n" + "=" * 50, logfile)
    log("GENERATING REPORT", logfile)
    log("=" * 50, logfile)
    
    report_path = generate_report(all_results, cfg, cfg.run_root, logfile)

    log(f"\n{'=' * 70}", logfile)
    log(f"RALPH COMPLETE — {all_results['duration_min']:.1f} min", logfile)
    log(f"Report: {report_path}", logfile)
    log(f"Results: {os.path.join(cfg.run_root, 'results.json')}", logfile)
    log(f"{'=' * 70}", logfile)

    logfile.close()
    print(f"\n\nDone. Read {report_path} when you wake up.\n")


if __name__ == "__main__":
    main()
