"""Policy drift: how fast does each version of the controller leave a trajectory, and in which coordinates?

For one similarity_loop run and round, drives each saved trajectory from its
start four ways and plots the sd-scaled state gap to the trajectory over time
(log scale), one panel per trajectory:

  open loop       u = u_t                         (no feedback at all)
  policy on walk  u = pi(x_bar_t) - K_t dx         (policy actions, walk's feedback)
  walk + policy   u = u_t + pi(x) - pi(x_bar_t)    (walk actions, policy's feedback)
  policy          u = pi(x)

If "walk + policy" drifts as fast as "open loop", the policy adds no useful
feedback. A second figure shows which observed coordinates have drifted most
when the policy's gap first passes 1 sd (mean over trajectories). The policy
is loaded from policy_<round>.pt when the run saved it, else rebuilt as in
policy_decomposition. Writes figures and traces.npz to --out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.diagnostics.feedback_rollouts import linearize, lqr, step
from scripts.diagnostics.feedback_tube import capped
from scripts.diagnostics.noise_cutoff import observation
from scripts.diagnostics.plot_similarity_loop import GRID, INK, MUTED, SERIES, SURFACE
from scripts.diagnostics.policy_decomposition import rebuild
from scripts.diagnostics.similarity_loop import SOURCES, R
from src.envs.walker2d import Walker2d
from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK
from src.gps.mppi_gps.policy import Policy

ARMS = ("open loop", "policy on walk", "walk + policy", "policy")
NAMES = (
    "height", "torso", "thigh", "leg", "foot", "thigh L", "leg L", "foot L",
    "v fwd", "v up", "v torso", "v thigh", "v leg", "v foot", "v thigh L", "v leg L", "v foot L",
)  # fmt: skip


def load(path):
    """A saved policy: a Policy checkpoint or a bare state dict (controller_loop), 2 or 3 hidden layers."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    weights = sorted(k for k in state if k.startswith("network.") and k.endswith(".weight"))
    if len(weights) == 3 and "state_dict" in payload:
        return Policy.from_checkpoint(path)[0]
    width = int(state["network.0.weight"].shape[0])
    p = Policy(
        0,
        "cpu",
        dict(mean=state["observation_mean"].numpy(), std=state["observation_std"].numpy()),
        hidden_dim=width,
    )
    layers = []
    for k in sorted(weights, key=lambda k: int(k.split(".")[1])):
        out, inp = state[k].shape
        layers += [torch.nn.Linear(inp, out), torch.nn.Tanh()]
    p.network = torch.nn.Sequential(*layers)
    p.load_state_dict(state, strict=True)
    return p


def trace(env, p, start, xbar, ubar, K, scale, arm):
    x, seen = start.copy()[None], []
    on_path = (
        None
        if arm == "open loop"
        else p.predict(observation(xbar).astype(np.float32)).astype(np.float64)
    )
    for t in range(len(ubar)):
        dx = (x[:, 1:] - xbar[t, 1:]) / scale
        if arm == "open loop":
            u = ubar[t][None]
        elif arm == "policy on walk":
            u = on_path[t] - dx @ K[t].T
        else:
            here = p.predict(observation(x).astype(np.float32)).astype(np.float64)
            u = here if arm == "policy" else ubar[t] + here - on_path[t]
        x = step(env, x, np.clip(u, -1.0, 1.0))
        seen.append(x[0])
        if not env._is_healthy(x[0, 2], x[0, 3]):
            break
    return np.array(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--cap", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    saved = args.run / f"policy_{args.round}.pt"
    p = load(saved) if saved.exists() else rebuild(env, args.bank, args.run, args.round)
    p.eval()
    gaps, firsts = {}, []
    with np.load(args.run / f"round_{args.round}.npz") as z:
        for s in SOURCES:
            with np.load(args.bank / f"episode_{s}.npz") as d:
                start, scale = d["states"][0], d["states"][:, 1:].std(0) + 1e-6
            st, u = z[f"states_{s}"], z[f"controls_{s}"]
            xbar = np.concatenate([start[None], st[:-1]])
            K = capped(lqr(*linearize(env, xbar, u, scale), R), args.cap)
            for arm in ARMS:
                seen = trace(env, p, start, xbar, u, K, scale, arm)
                per = np.abs((seen[:, 2:] - st[: len(seen), 2:]) / scale[1:])
                gaps[(s, arm)] = np.sqrt((per**2).mean(1))
                if arm == "policy":
                    over = np.flatnonzero(gaps[(s, arm)] > 1.0)
                    firsts.append(per[over[0]] if len(over) else per[-1])
            print(s, {a: len(gaps[(s, a)]) for a in ARMS}, flush=True)
    np.savez_compressed(
        args.out / "traces.npz", **{f"{s}_{a.replace(' ', '_')}": g for (s, a), g in gaps.items()}
    )

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.size": 9,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
    for ax, s in zip(axes.flat, SOURCES):
        for color, arm in zip(SERIES, ARMS):
            g = np.maximum(gaps[(s, arm)], 1e-16)
            ax.plot(np.arange(1, len(g) + 1) / 125, g, color=color, lw=1.6, label=arm)
            if len(g) < 1000:  # fell
                ax.plot(len(g) / 125, g[-1], "x", color=color, ms=7, mew=2)
        ax.set_yscale("log")
        ax.set_ylim(1e-14, 1e2)
        ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
        ax.set_title(f"walk {s}", loc="left")
        ax.grid(True, color=GRID, lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("seconds")
    for ax in axes[:, 0]:
        ax.set_ylabel("state gap to the walk (sd units)")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(
        f"How fast each controller leaves the walk (round {args.round}; x = fell)", y=0.995, x=0.01,
        ha="left", fontsize=11,
    )  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(args.out / "drift.png", dpi=150)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    mean = np.mean(firsts, axis=0)
    ax.bar(np.arange(len(NAMES)), mean, color=SERIES[3], width=0.7)
    ax.set_xticks(np.arange(len(NAMES)), NAMES, rotation=45, ha="right")
    ax.set_ylabel("drift (sd units)")
    ax.set_title("Where the policy has drifted when its gap first passes 1 sd", loc="left")
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "drift_coordinates.png", dpi=150)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
