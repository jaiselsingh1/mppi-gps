# 2026-11-02 to 11-06: probes, the controller loop, and the MPPI loop (Claude)

**Code:** tag `archive/sac-continuation` (all scripts in `scripts/diagnostics/`; main loop files
`controller_loop.py`, `similarity_loop.py`, `mppi_loop.py`) and tag `archive/mppi-loop`, which
is the package now on main. Commit messages `aa54790`, `c13d14c`, `5439320` and `9a142f0`
describe each step. 44 diagnostics, figures and the algorithm box that were never committed are
in `/Users/jaisel/Documents/workbench/mppi-gps-archive-20261107/mppi-loop/`. Runs are in the
main checkout's `runs/2026-11-*` and on robopil5090 in `~/mppi-gps-claude-sac/runs/`.

## Probes, 11-02 and 11-03 (no student walked in any of them)

- A student copies a teacher only if the state sets the teacher's action: SAC's action is about
  99% state-determined and its clones walk; adding 10% state-independent jitter makes clones fall.
- MPPI's action is about 4% state-determined (27% with smoother noise); its motion is cleaner
  (62–66%). Only about 8–10 of 256 rollouts get real weight, so each update keeps noise.
- What predicts whether a student walks is its disagreement with a good teacher on the states
  the student itself reaches (identical data and offline error: 0/32 vs 25/32).
- Tried inside MPPI, none made its actions learnable: torque or posture matching, executing the
  single best rollout, terminal costs for policy confidence / value / stability, mirrored
  sampling noise, larger effort cost, averaging only one side of the samples, reweighting
  recorded samples toward the policy, a weak pull toward the policy over 3 rounds. Terminal
  costs could steer *where* MPPI goes, not how consistent its actions are. Noise along flat cost
  directions was measured and ruled out as the cause.

## Controller loop, 11-04 and 11-05

MPPI walks and the policy are compared as local controllers; all walks share one start-up and
one repeating stride; a refit is kept only if the task score does not drop.
- 11-04: a policy walking 8 s from **99/100** fresh standing starts, settling into one stride.
- 11-05: keeping a learner training while the deployed copy waits for a better score, and
  mixing targets toward the deployed policy by its score: MPPI and policy converged in **6/6**
  seeds (gap 0.0093). Plain cloning and DAgger baselines added.

## MPPI loop, 11-06 (now on main)

MPPI drives walks each round with a similarity term in its rollout cost that pulls rollout states
toward a reference set (last round's shared walk and the policy's own states, weighted by the
policy's score); the policy learns MPPI's actions.
- With the similarity: 3/5 seeds walk from all 32 starts within 15 rounds; without: 0/4.
- Walking needs nearly all of a round's MPPI walks to agree (`experiments/walkable_probe.py`).
- Walkers are fragile to small pushes and cannot get up; Hopper's MPPI teacher does not hop yet.
- Review the same day: the loop had become too complicated and Walker-specific (medoid walk,
  LQR-gain slopes, fall checks, gates). Next step: the minimal version (MPPI + one similarity
  term + plain copying) on a simple stock task, adding pieces back only by side-by-side test.
