# MPPI loop

A policy learns to control a MuJoCo robot from trajectories that MPPI (sampling-based MPC) makes,
with MPPI and the policy tied together through **where MPPI's rollouts go**: a similarity term in
MPPI's rollout cost pulls its rollouts toward a reference set of states. Before the policy can do
the task the reference is mostly one shared walk, so MPPI's many valid solutions agree on one
learnable family; as the policy improves the reference shifts to the states the policy itself
reaches. Nothing from the policy seeds or samples MPPI's plans.

This branch holds only the method, its baseline and the probes behind the current findings. The
full research history (diagnostics, earlier loops, results) is on `sac-continuation` and `main`.

## How a round works

1. **Reference**: last round's medoid walk (weight 1 − s) and the deployed policy's own visited
   states (weight s), where s is the policy's task score.
2. **MPPI walks**: one walk per training start, with the similarity in MPPI's rollout cost and
   small noise on the executed actions; MPPI's planned actions are the labels.
3. **Fit**: the policy fits MPPI's actions and LQR-gain slopes, mixed with its own by s.
4. **Swap-in**: the deployed policy changes only when the retrained one scores at least as well.
5. **Judge** the deployed policy from fresh standing starts and mid-walk takeovers.

## Code (reading order)

| file | what it holds |
|---|---|
| `src/mppi_loop/loop.py` | the rounds, MPPI walk worker, DAgger baseline worker, CLI |
| `src/mppi_loop/similarity.py` | the state-set similarity cost, medoid walk |
| `src/mppi_loop/fit.py` | policy fit (actions + slopes), mixed targets, policy–MPPI gap |
| `src/mppi_loop/dynamics.py` | batched stepping, linearization, LQR gains, per-step costs |
| `src/mppi_loop/policy.py` | policy network |
| `src/mppi_loop/spec.py` | Walker observation and success rule |
| `src/mppi_loop/measures.py` | gait section and walk spread (logging only) |
| `src/mppi_loop/tasks/` | `walker.py` (in use), `hopper.py` (template; its MPPI teacher does not hop yet) |
| `src/mppi/mppi.py`, `src/envs/` | MPPI and the MuJoCo environments |
| `experiments/` | `refit_probe`, `walkable_probe`, `robustness_probe` (the 2026-11-06 findings) |

## Running

```bash
python scripts/run_loop.py --weight 1 --seed 1 --out runs/w1_seed1             # the method
python scripts/run_loop.py --weight 0 --seed 1 --out runs/w0_seed1             # no similarity
python scripts/run_loop.py --weight 0 --dagger --seed 1 --out runs/dagger_seed1  # DAgger baseline
```

Add `--resume` to continue an interrupted run exactly; `--quick` is a short plumbing check. Each
round writes `rows.json` (scores, judge, spread, gap), the round's walks and both policies.
Walker needs the sample bank folder `runs/2026-11-03_sample_bank_001` (`--bank`), kept on
robopil5090 (not in git).

## Status (Walker2d, 2026-11-06)

- With the similarity, 3 of 5 seeds learn to walk from all 32 standing starts within 15 rounds
  (rounds 9, 12, 14; scores 0.90–0.98 by round 26); without it, 0 of 4. A regular stride
  emerges on its own. The DAgger baseline is running.
- Walking needs nearly all of a round's MPPI walks to agree; one walk that does not fit blocks
  it (`experiments/walkable_probe.py`).
- Open: GPS-style action coupling as the closest baseline; a second, harder environment;
  wall-clock against RL; robustness (walkers are fragile to small perturbations and cannot get
  up from the ground); the Walker task still takes its starts, cost normalizer and starting
  policy from a SAC-made sample bank.
