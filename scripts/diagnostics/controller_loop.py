"""Controller loop: trajectories and policy converge as controllers (action on the path and reaction off it).

The audit of similarity_loop (2026-11-04) found that matching actions along each
trajectory does not make them converge: the trajectory's stabilizer (tracking
LQR) and the policy react differently just off the path (about 2.6x the on-path
disagreement), so moving or blending a trajectory toward the policy did not
shrink the gap, the searches stopped accepting moves, and the remaining
convergence was the policy refitting frozen trajectories. Kendall: the policy
is done when MPPI and the policy converge to each other, with a limit on how
much the policy affects MPPI. This makes each trajectory a local controller
u = u_bar_t - K_t (x - x_bar_t) / scale and lets both halves of it move toward
the policy. Writes only to --out.

Design, fixed before the first run:

- Gains: K(gamma) = (1 - gamma) K_LQR - gamma J, where K_LQR are the tracking
  LQR gains along the trajectory (R = 1, singular values capped at 3) and J
  the policy's derivative with respect to the sd-scaled state at x_bar. Each
  round, per trajectory, gamma is the largest of 1, 0.75, 0.5, 0.25, 0 whose
  closed loop is about as robust as LQR's: of 32 closed-loop rollouts with the
  search's starting feedforward noise (low-pass, sd 0.03), the share that do
  the task within the limit is at least the gamma = 0 share minus 0.1.
  Changing the gains does not change the trajectory itself, only how it reacts.
- Search: similarity_loop.search with K(gamma) as the stabilizer (recomputed
  when the nominal moves) and toward-policy proposals; the similarity weight
  as in the change note below.
  The limit: the trajectory does the task and costs <= 1.05x its start cost
  (the walker: MPPI's walk). 8 iterations per round.
- Fit: the policy continues for --fit-updates updates of |pi(x_bar) - u|^2 +
  3 |J + K(gamma)|^2 (slope_fit.fit_rows) on all trajectories, fresh Adam,
  seed = round. With gamma = 1 the slope target is the policy's own slope, so
  nothing is imposed; with gamma < 1 it is pulled toward a stabilizing one.
- Converged: gamma = 1 on every trajectory and the action gap (per-action MSE
  between trajectories and the guiding policy, before refitting) near 0; then
  the policy's own reactions stabilize trajectories its actions reproduce.
- Point mass (--task point): point_mass_check's setup (6 irregular start
  trajectories, 32 held-out starts, the optimal LQR reference); 30 rounds, 200
  updates per round. Expected: gamma -> 1 quickly (its own reactions are
  stable), gap -> 0, held-out cost near the optimal reference.
- Walker (--task walker): the bank's walks 0, 1, 2, 3, 4, 6 re-executed in
  closed loop, a 3 x 256 start policy fit lightly (500 updates) on the bank's
  walking episodes; 30 rounds, 500 updates per round; every 5 rounds the policy
  alone from S1's 32 standing starts and from 51 mid-walk states (bank walks
  0-2, controls 100-900), 8 s each: survival, and walking (survives and after
  the first 100 controls mean |speed - 1.5| <= 0.3, S1's standing rule).
- Measures per round: gap before and after the refit, gamma per trajectory,
  reflex agreement 1 - sum |J + K_LQR|^2 / sum |K_LQR|^2, cost against the
  start, whether every trajectory does the task within the limit, how many
  searches moved their trajectory.
- Readings: "converges as controllers" if by the last round mean gamma >= 0.9
  and the gap before refitting fell >= 90% from round 1; point mass "matches
  the known answer" if its held-out cost is within 1.1x the reference; walker
  "the policy walks" if it walks the full 8 s from >= 16 of 32 standing starts.

Changed after the first point-mass run (gamma 1 and gap 0 from round 1, but the
held-out cost stayed at the start policy's 1.13x optimal for 30 rounds: with
the similarity dominating, the trajectories simply became the policy and
stopped improving the task, Kendall's "mppi shouldn't copy policy"): the
search objective is task cost + alpha_k D, with each trajectory's alpha_k set
by dual ascent toward a small disagreement eps ("close to the policy but
better", the trust region of guided policy search): after each round's search,
alpha_k doubles if the trajectory's gap to the guiding policy (per-action MSE)
is above eps and halves if below eps / 2. Start alpha_k = task cost / D (equal
weight). eps = 0.001 (point mass) and 0.002 (walker). The limit is unchanged.
"Converges as controllers" now means mean gamma >= 0.9 and the gap before
refitting within eps by the last round (with a trust region the gap settles
near eps rather than at 0).

Fixed before the walker runs (found in the quick check, where no 120-control
rollout can count as walking): the robustness bar is max(LQR share - 0.1, half
the LQR share), and gamma stays 0 when the LQR share is 0, so a fragile
trajectory is never handed to the policy's reactions untested.

Added at walker round 6 (the eps 0.002 and 0.005 runs were identical, the gap
being far above both, while trajectory costs reached 1.042 of the 1.05 limit):
--budget sets the limit; the eps 0.005 run is replaced by eps 0.002 with
--budget 1.15, to measure how much the limit holds the trajectories back.

Added after the start-up diagnosis (startup_diagnosis, 2026-11-04: the six MPPI
start-ups disagree from the first action, the policy averages them and falls
at about 1.45 s from standing; training on walk 1's start-up alone raised
median survival from standing from 1.7 to 4.75 s, walk 2's did not help, so
start-ups must also work from nearby standing states): --starts M gives each
trajectory M nearby starts per round (Walker: fresh reset draws, seeds 900,000
+ 1,000 r + 10 k + j, disjoint from the bank's and S1's evaluation seeds; point
mass: the start plus N(0, 0.02^2) per position and velocity). In the search a
sample counts only if its controller, run from each nearby start for --window
controls (250 = 2 s), stays healthy and ends within 1 sd (observed
coordinates) of its own path from the trajectory's start; a robust candidate
replaces a fragile nominal. The final controller's rollouts from the nearby
starts that pass are added to the fit (with the same gains as slope targets),
as guided policy search trains on its local controllers across initial
conditions. Also measured: per trajectory, the share of nearby starts the final
controller brings back (start_robustness).

Added after the nearby-starts walker run (survival from standing doubled to
2.1-3.5 s in rounds 15-25 but decayed to 1.6 s by round 30; each walk kept its
own start-up, so the policy still averaged six; training on one start-up
alone had raised it from 1.7 to 4.75 s): --shared-startup W gives all
trajectories one start-up controller for their first W controls. A carrier
trajectory is chosen before round 1: the one whose first-W controller (capped
tracking LQR) brings the most of the other training starts back within 1 sd of
its path at W. Each round the carrier is improved with the other training
starts (and --starts nearby ones) as extra starts over W controls, so its
start-up must work from all of them. Every other trajectory then runs the
carrier's start-up controller from its own start for W controls and continues
with its own walk from wherever that leaves it: its previous continuation, or
the carrier's if its own no longer walks from there, whichever walks within
the limit at lower cost, improved like any trajectory (budget minus the
start-up's cost). If neither walks, the trajectory keeps its previous self for
that round and is counted (shared = False). The fit uses each start-up part's
gains in the carrier's state scale. Point mass --cluster s: all starts near
(1, 0) at rest, perturbed by N(0, s^2), the analog of standing starts.

Fixed after the first clustered point-mass run with --shared-startup (the
carrier never moved in 30 rounds and its start-up brought back only 14-57% of
the other starts): gamma was chosen by robustness from the carrier's own start
only, where the policy's own reactions pass (gamma = 1), but those reactions
steer toward the policy's path, not back to the carrier's, so no candidate met
the search's extra-start requirement. With extra starts, gamma must now also
bring them back about as well as the tracking gains do (bar max(LQR share -
0.1, half of it)), the same test the search applies.

Added after round 15 of the walker --shared-startup run (the policy alone
walked 8 s from 78 of 100 new standing starts but from 10 of 51 mid-walk
states; every trajectory starts from standing, so nearly all fit data is the
steady walking cycle). Kendall, 2026-11-04: "init new traj from somewhere along
previous trajectories helps to have data diversity" and "learning trajs from
states the policy visits is another way to share info between them".
--visit-every V (walker only): at the start of rounds 1, 1 + V, ..., the guiding
policy runs alone for 8 s from 12 fresh standing draws (seeds 800,000 + 1,000 r
+ j, disjoint from the others) and from 2 random states along each trajectory
(controls 100+). From each run that does not walk, one state: 0.5 s before it
fell (MPPI is local and cannot rescue a fall already under way), or, if it
stayed up off speed, the start of its worst second. Alternating between the two
kinds of origin in random order, MPPI (configs/walker2d_best.json) drives from
each picked state for --visit-length controls (500 = 4 s), at most 2 x
--visit-new tries until --visit-new of them walk; each is re-executed in closed
loop (tracking LQR, the bank's pooled state scale) like the bank walks and kept
if it walks. Its limit is --budget times its own MPPI cost. Kept trajectories
join the set: improved every round like the free trajectories (gamma, search
toward the policy, their own alpha; no shared start-up, since they start in
motion) and fit with the others; at most --visit-max are kept (oldest dropped).
Measured: the six trajectories as before; the visited set's size, picks, gamma,
gap and cost; and, for every run, the policy alone from 51 mid-walk states of
walks 7-9, which the loop never trains on (held-out mid-walk).

Added after testing every saved policy of the shared start-up run (2026-11-04):
walking from 100 new standing starts flipped from round to round (rounds
21-30: 69, 62, 46, 1, 94, 76, 96, 98, 41, 98), and the policy's actions moved
about 0.03 rms per round from round 6 to 30 without shrinking, even in rounds
where the trajectories moved 0.01. Two refits of the same policy on the same
data differing only in the minibatch seed ended 0.026 apart (actions-only
check): the movement is the refit's own minibatch noise at a constant step
size, not the trajectories. With the step size falling linearly to 0 over the
fit, the two seeds ended 0.004 apart and fit the targets slightly better.
--settle: every refit uses that falling step size (same 2,000 updates), so when
the trajectories stop moving the policy can too. Also measured each round:
policy_moved, the rms change of the policy's actions along the trajectories
from the refit; --judge-every sets how often the policy alone is tested (1 for
these runs, since single rounds proved unrepresentative). Readings fixed before
the runs: "the policy settles" if mean policy_moved over the last 10 rounds is
<= 0.01 (the shared start-up run: about 0.028); "walks stably" if the policy
walks from >= 16 of S1's 32 standing starts in >= 9 of the last 10 rounds.

Added after an audit of the shared start-up run's log (2026-11-04), which found
three mistakes in how the loop applied the ideas, not in the ideas: (1) the
pull ignored whether the policy can do the task: in rounds 1-6 the gap fell
from 0.115 to 0.0026 while the trajectories' cost rose from 1.07 to 1.47x
MPPI's and the policy still fell at 1.3 s from every standing start; (2) alpha
had no effective cap: on 3 of 6 walks it doubled every round to 2-4e8 by round
30, pinning them at the cost limit while they chased a jittering policy;
(3) 2,000 updates per round is about 65 passes over the round's data, and after
round 5 the refit did not reduce the gap on net (rounds 10-30: 0.00235 before,
0.00233 after). Kendall, the same day: "Policy might be still training too much
on the data" and "More interesting to figure out how to get a limit cycle";
and his standing rule that task performance comes before policy influence.
--gate: before its search, each trajectory's pull weight is alpha_k = c_k A_k.
c_k (competence) is the share of 8 evenly spaced states along the trajectory
(its start included) from which the guiding policy alone does the task for 250
controls (walker: stays up and averages >= 0.75 m/s forward, half the target;
point mass: ends at the goal). A_k is the dual variable as before (doubles
above eps, halves below eps / 2), updated only when c_k > 0, and capped so that
at the target gap the pull term is at most twice the trajectory's task cost:
A_k <= 2 C_k / (eps x controls / 2). With c_k = 0 the search improves the task
alone and the policy copies (Kendall: the policy copies MPPI first; MPPI does
not copy a policy that cannot do the task). Competence is measured with policy
rollouts in the simulator, not a learned value function (which Kendall preferred
on 2026-10-27 for "can the policy cope here"); rollouts are exact here.
Runs: --gate --settle --judge-every 1, --fit-updates 200 (about 6 passes) and
500 (about 16), 40 rounds, the shared start-up as before, no visited states.
Also measured each round: the stride regularity of the policy alone from S1's
standing starts (stride_cycle.Section; stride-to-stride change 0.27 for the SAC
oracle, 1.12 for MPPI's walks, 0.90 for the shared start-up run's round-30
policy). Readings fixed before the runs, besides those above: "holds a stride"
if the mean stride change over the last 10 rounds is <= 0.4; "does not copy a
falling policy" if the trajectories' mean cost stays <= 1.2x MPPI's in every
round whose mean competence is < 0.25.

Added for Kendall's "figure out how to get a limit cycle" (2026-11-04). The
policy alone had no repeating stride (stride change 0.90; SAC oracle 0.27;
MPPI's walks 1.12), and the loop asked it to match six irregular walks step by
step. A feasibility check (stride_cycle): MPPI's most repeatable stride (walk
2, 0.86 s) locks into an exact cycle under periodic tracking feedback (residual
0.000 with gains capped at 10; about 0.01 with blips of 0.09 at cap 3), at 1.31x
MPPI's cost per control, and about 40-50% of the states MPPI's walks pass
through stay up when switched onto it. --cycle (walker, with --shared-startup):
before round 1 the most repeatable stride after the start-up among the training
walks (smallest change between consecutive section crossings) is locked into a
cycle: its controller (feedforward plus periodic tracking LQR, capped at 3) runs
stride after stride from its own start until it returns to the same state (30
strides), twice, the second time with gains along the first result. Every walk
is then the shared start-up, a continuation (the carrier's round-0 walk after
the start-up, else the walk's own, tracked from where the start-up leaves it
with capped LQR; one that joins the cycle is preferred), and, from the place where the
continuation passes closest to the cycle (within 1 sd; up to 5 places 25
controls apart, the first that keeps walking to the end), the cycle repeated to
the end. Each round: the carrier is improved as with --shared-startup (limit
--budget x its MPPI cost, or its current cost if the cycle puts it above); the
cycle is improved like a trajectory (task, the gated pull, limit --budget x
MPPI's cost per control x its length) and locked again with gains (1 - gamma)
periodic LQR + gamma x the policy's reaction, kept only if it still walks
within the limit; every walk is rebuilt; the policy is refit on all walks (the
cycle parts with the cycle's gains as slope targets). Continuations are MPPI's
and are not optimized. Measured: walks built and joined to the cycle, the
cycle's residual, cost per control against MPPI's, gamma, competence and gap.
In this mode "does not copy a falling policy" uses the cycle's cost against its
locked start. Run: --cycle --gate --settle --judge-every 1 --fit-updates 500, 40
rounds, compared with the gated fit-500 run without --cycle. Reading as above:
the policy holds a stride if its mean stride change over the last 10 rounds is
<= 0.4.
Changed after the first --cycle run's round 1 (only 1 of 6 walks could be built:
each walk's own continuation, recorded from a different state, could not be
tracked from where the shared start-up leaves it): walks try the carrier's
continuation first, as described above. That run was stopped and is kept as
_stopped. Run 002 measured the cycle's gap on the search's cycle rather than the
one used; fixed (gap and pull weight on the cycle actually used), stopped. Run
003 joined walks at the closest pass, which after round 1's start-up change was
control 973 of 1000: "joined" walks were almost entirely MPPI continuation, at
0.5-0.9 sd from the cycle. Changed: the join is the earliest near pass (local
minima of distance to the cycle within 0.5 sd, in time order); with the round-0
and round-1 start-ups every walk joins at control 222 and is within 0.01 sd of
the cycle over its last 300 controls. Also logged: each walk's join control
and the share of the walks on the cycle (on_cycle).
Changed after run 004 (rounds 1-4: walks gap 0.10 -> 0.03, the policy's gap on the
cycle 0.076 -> 0.009, its competence on the cycle 0 -> 0.75; round 5: the cycle's
gamma reached 1, walks could no longer join under those gains, fell back to
joining at control 973, and the refit lost the cycle, gap 0.009 -> 0.145): the
cycle's repeats use the largest gamma (at most the cycle's own) under which as
many walks join by control 500 as under the tracking gains alone, less one (the
start-up's rule for its extra starts); the gamma test on the cycle's own start
alone does not test joining. Also, in every --gate run from here on, the pull
term itself is capped at PULL_CAP times the task cost at the current gap (alpha
<= PULL_CAP C / max(D, D at eps)), not only at the target gap: in the gated
fit-500 run, competence 0.1 at a gap 10x eps let the pull raise the
trajectories' cost from 0.97x to 1.2x MPPI's while the policy still fell.
Changed after run 005 (the join rule held: at round 5, cycle gamma 1 but the
repeats at 0.75, all walks on the cycle, gap 0.012; but no re-lock succeeded
after round 3, so the cycle stopped moving toward the policy): locking uses the
tracking gains (gamma 0) instead of the cycle's gamma (a check on round 7's
cycle: 12 of 12 perturbed strides lock at gamma 0.5 and 0.75, 7 of 12 at
gamma 1), a lock must also settle (mean change over its last 5 strides <= 0.1
sd), and the reason a re-lock is refused is logged (lock_result).

Added after a dashboard of all runs (plot_controller_loop, 2026-11-04 ~19:00):
the only run that walked from standing is the shared start-up run, whose pull
was on from round 1 (gap 0.0036 by round 5 at 1.41x MPPI's cost; standing 15/32
at round 10, 31/32 at round 30); every --gate run sits at 0/32 from standing
(the fit-500 one through round 25, its competence <= 0.19, so it is close to
cloning). The gate may have switched off what worked: pulling the trajectories
toward what the policy can reproduce, within the task limit, is the guided
policy search mechanism (Mordatch 2015 applies it from the start), and those
trajectories still walked. The comparison is confounded (that run trained 4x
more per round, at a constant step, with no cap). --pull-always (with --gate):
competence is taken as 1, so the pull is on from round 1, with the cap on the
pull term (at most PULL_CAP x task at the current gap) kept. Runs, 40 rounds,
otherwise as the gated fit-500 run and cycle run 006 (--settle --judge-every 1
--fit-updates 500, shared start-up 188, 4 nearby starts): --pull-always without
and with --cycle. Readings fixed before the runs: "the early pull helps" if the
pull-always run walks from >= 16 of 32 standing starts in at least 5 of rounds
21-40 while its gated counterpart does in none (judged without the cycle, and
separately with it).

Added from Kendall (2026-11-04): "Weight the policy by its score. The worse it
does at the task, the less mppi needs to copy it (and the more it should copy
mppi)", and the user: "the policy shouldn't update its params if it doesn't get
better at the task". The gated runs had the first half but trained lightly
while the policy was bad, the opposite of the second; the only run that walked
from standing trained 2,000 full-step updates per round from round 1.
--score (with --gate): a trajectory's pull weight is its score, the trajectory's
task cost per control over the policy's own cost per control run alone from the
same start (at least 250 controls; falling keeps costing), at most 1, in place
of competence. --copy-low N: each round's refit takes --fit-updates updates at
policy score 0 falling linearly to N at score 1, the policy score being MPPI's
mean walk cost over the policy's own mean 8 s cost from 32 fixed standing starts
(the 6 training starts and 26 draws, seeds 700,000+, disjoint from every test
set), at most 1. --accept-if-better: after each refit the policy score is
measured again and the refit is kept only if it did not drop; otherwise the
policy keeps its old parameters (the trajectories' round stands). Logged:
policy_score, policy_accepted, fit_updates_used. Runs: --gate --score
--copy-low 200 --fit-updates 2000 --accept-if-better --settle --judge-every 1,
40 rounds, without and with --cycle. Readings fixed before the runs: the policy
walks stably (>= 16/32 from standing in >= 9 of the last 10 rounds) and the
policy score never drops (by construction) while the walking counts may.

Changed after cycle run 006 (converged on its cycle at round 9, gap 0.0011,
competence 1.0; then the pull dragged the cycle to 1.32x MPPI's cost per control,
only 3 of 6 walks could join it at round 12 and none at round 13, where the run
crashed on a missing key for a round without walks): a re-locked cycle is
accepted only if as many walks still join it early under the tracking gains as
joined the old one, less one ("walks stop joining"); every assembled walk must
cost at most --budget times its MPPI walk (Kendall's limit at the walk, which
the cycle runs had not enforced: the pull-always cycle run reached 1.71x), and
the carrier's limit is the plain limit again; a round without walks is logged,
not a crash.

Added after the pull bench (pull_bench, 2026-11-04: from a policy that had copied
MPPI, the carry-on pull at a 1.3x allowance was the only clear gain after one
update, task score 0.139/0.112 against cloning's 0.090/0.089 and survival 3.4 s
against 1.3 s; action matching, the loop's pull, was about cloning) and the
score-rule cycle run (the only loop policy that walks: 99/100 fresh standing
starts with a repeating stride, from round 7). --pull: action (as before),
carry-on (pulls.carry_on_pull: every 100 controls the policy alone runs 1 s from
the trajectory's state and the cost is its distance to the trajectory; the
cycle is continued by itself; weighted so that at the start of each search it
equals the action pull it replaces, so the gate, cap and dual are unchanged), or
none (the trajectories improve the task alone: cloning, Kendall's bar). Runs, 40
rounds, the walking run's settings on the fixed code (--gate --score --copy-low
200 --fit-updates 2000 --accept-if-better --settle --cycle --judge-every 1,
shared start-up 188, 4 nearby starts): --pull action, carry-on, none. Readings
fixed before the runs: "the pull beats cloning" if a pull run walks from >= 16
of 32 standing starts in >= 9 of its last 10 rounds while the none run does not,
or if its final policy score exceeds the none run's by >= 0.1; "carry-on beats
action" by the same tests between those two.

Changed after a relaunch of those three runs (robopil5090 rebooted at their
rounds 21/14/21) diverged from round 1: with 10 simulator and 6 math threads
instead of 8 and 4, the first refit already differed (policy score 0.0917 vs
0.0890) and the outcomes flipped (first launch: action walks from round 7, none
from round 9, carry-on not by round 14; relaunch: carry-on walks from round 6,
action and none not by round 9). Results depend on the thread counts, and
whether a run finds a walking policy swings on differences that small, so one
draw per setting cannot rank the pulls. --seed S shifts every random draw of
the loop (search and gamma samplers by 1,000,000 S, refit minibatches by
1,000 S, the starting policy's initialization and fit by S; the data, the
evaluation sets and the acceptance starts are unchanged); S = 0 is the code
path of every earlier run. Runs: --pull action, carry-on, none x --seed 1, 2,
3, otherwise as above, 30 rounds, all with --threads 4 and 2 math threads
(fixed, so each run repeats exactly). Readings fixed before the runs, per pull
over the 3 seeds: how many seeds walk from >= 16 of 32 standing starts in >= 9
of rounds 21-30; the first round each seed walks from >= 16; the mean final
policy score. "A pull beats cloning" if it has more walking seeds than none, or
the same number and a mean final score >= 0.1 higher.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.diagnostics.coupled_mppi import CONFIG
from scripts.diagnostics.feedback_fit import slopes
from scripts.diagnostics.feedback_rollouts import closed_loop, linearize, lqr
from scripts.diagnostics.feedback_tube import capped
from scripts.diagnostics.loop_task import WALKER
from scripts.diagnostics.pulls import carry_on_pull
from scripts.diagnostics.similarity_loop import R, discrepancy, search
from scripts.diagnostics.slope_fit import fit_rows
from scripts.diagnostics.stride_cycle import (
    Section,
    best_stride,
    close_cycle,
    periodic_gains,
    tiled,
)
from scripts.diagnostics.trajectory_pi import step_costs
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

CAP = 3.0
GAMMAS = (1.0, 0.75, 0.5, 0.25)
ROBUST_DRAWS, ROBUST_SD, ROBUST_SLACK = 32, 0.03, 0.1
EPS = {"point": 0.001, "walker": 0.002}
BUDGET = 1.05
SLOPE = 3.0
ITERATIONS = 8
VISIT_STANDING, VISIT_ALONG = 12, 2  # policy runs from standing draws; per trajectory, along it
VISIT_LEAD = 62  # controls before a fall (0.5 s) to start a visited trajectory from
GATE_STATES, GATE_HORIZON = 8, 250  # competence: states along a trajectory, controls each
PULL_CAP = 2.0  # pull term at the target gap at most this many times the task cost
CYCLE_LOCK = 30  # strides run to lock a stride into a cycle
JOIN_TRIES, JOIN_NEAR, JOIN_MAX = 5, 0.5, 1.0  # join places tried; preferred, farthest (sd)
JOIN_EARLY = 500  # a walk "joins early" if it is on the cycle by this control
PULL = {"kind": "action"}  # --pull: what the trajectories are pulled toward (set by main)
SEED = {"s": 0}  # --seed: shifts every random draw of the loop (0: as before; set by main)


def parts(env, policy, xbar, u, scale, spec):
    """Capped tracking-LQR gains and the policy's reaction -J along a trajectory."""
    K_lqr = capped(lqr(*linearize(env, xbar, u, scale), R, spec.free), CAP)
    return K_lqr, -slopes(policy, xbar, scale, spec)


def robustness(env, spec, start, xbar, u, K, scale, budget, noise):
    """Share of closed-loop rollouts with feedforward noise that do the task within the limit."""
    s, c = closed_loop(env, start, xbar, u, K, scale, np.clip(u[None] + noise, -1.0, 1.0))
    return float((spec.admissible(s) & (step_costs(env, s, c).sum(1) <= budget)).mean())


def brought_back(env, spec, xbar, u, K, scale, extra, window, reference):
    """Share of extra starts the controller keeps healthy and brings within 1 sd of the path."""
    n = min(window, len(u))
    passed = 0
    for x0 in extra:
        se, _ = closed_loop(env, x0, xbar[:n], u[:n], K[:n], scale, u[None, :n])
        d = (se[0, -1, spec.observed] - reference[spec.observed]) / scale[spec.observed - 1]
        passed += bool(env._is_healthy(se[0, :, 2], se[0, :, 3]).all()) and bool(
            np.sqrt((d**2).mean()) <= 1.0
        )
    return passed / len(extra)


def choose_gamma(
    env,
    spec,
    start,
    xbar,
    u,
    scale,
    budget,
    K_lqr,
    reaction,
    seed,
    extra=None,
    window=250,
    states=None,
):
    sampler = MPPI(
        env,
        MPPIConfig(
            **dict(json.loads(CONFIG.read_text()), H=len(u), K=ROBUST_DRAWS, noise_sigma=1.0)
        ),
    )
    sampler.reset(seed=70_000 + seed)
    noise = sampler._sample_noise() * ROBUST_SD
    base = robustness(env, spec, start, xbar, u, K_lqr, scale, budget, noise)
    if base == 0.0:
        return 0.0  # nothing to compare against: keep the tracking stabilizer
    floor = max(base - ROBUST_SLACK, 0.5 * base)
    if extra is not None:
        reference = states[min(window, len(u)) - 1]
        base_extra = brought_back(env, spec, xbar, u, K_lqr, scale, extra, window, reference)
        floor_extra = max(base_extra - ROBUST_SLACK, 0.5 * base_extra)
    for gamma in GAMMAS:
        K = (1.0 - gamma) * K_lqr + gamma * reaction
        if robustness(env, spec, start, xbar, u, K, scale, budget, noise) < floor:
            continue
        if extra is not None and (
            brought_back(env, spec, xbar, u, K, scale, extra, window, reference) < floor_extra
        ):
            continue
        return gamma
    return 0.0


def gap(policy, start, states, controls, spec):
    """Per-action MSE between a trajectory's actions and the policy along its own states."""
    return float(
        2.0
        * discrepancy(policy, start, states[None], controls[None], spec=spec).sum()
        / controls.size
    )


def dual_step(alphas, k, gap_value, eps, competence, cap):
    """The pull weight's update: doubles above eps, halves below eps / 2; with the gate (cap not
    None) only when the policy has some competence, and capped."""
    if cap is not None and not competence > 0:
        return
    if gap_value > eps:
        alphas[k] = min(alphas[k] * 2.0, 1e12)
    elif gap_value < eps / 2:
        alphas[k] = max(alphas[k] / 2.0, 1e-12)
    if cap is not None:
        alphas[k] = min(alphas[k], cap)


def improve(
    env,
    spec,
    start,
    u,
    states,
    scale,
    policy,
    budget,
    r,
    k,
    alphas,
    eps,
    extra,
    window,
    gate=None,
    dual=True,
    periodic=False,
):
    """One trajectory's step: choose gamma, search with K(gamma), update its alpha.

    Returns its new controls, states and gains, a row of measures, and the rollouts from the
    extra starts that its controller brings back (for the fit)."""
    xbar = np.concatenate([start[None], states[:-1]])
    K_lqr, reaction = parts(env, policy, xbar, u, scale, spec)
    gamma = choose_gamma(
        env,
        spec,
        start,
        xbar,
        u,
        scale,
        budget,
        K_lqr,
        reaction,
        1000 * r + k + 1_000_000 * SEED["s"],
        extra,
        window,
        states,
    )
    task = float(step_costs(env, states[None], u[None]).sum())
    D = float(discrepancy(policy, start, states[None], u[None], spec=spec).sum())
    if alphas[k] is None:
        alphas[k] = task / max(D, 1e-9 * task)
    alpha, competence, cap = alphas[k], None, None
    if gate is not None:  # pull only as far as the policy can do the task along this trajectory
        competence = gate(policy, xbar, u, task)
        cap = PULL_CAP * task / max(D, eps * u.size / 2.0)  # pull term <= PULL_CAP x task
        alphas[k] = min(alphas[k], cap)
        alpha = competence * alphas[k]

    def stabilizer(xb, uu):
        a, b = parts(env, policy, xb, uu, scale, spec)
        return (1.0 - gamma) * a + gamma * b

    similarity = None
    if PULL["kind"] == "none":
        alpha = 0.0  # cloning: the trajectories improve the task alone
    elif PULL["kind"] == "carry-on" and alpha > 0:
        # toward states the policy can carry on from (pull_bench's winner), weighted so that at the
        # start of the search it equals the action pull it replaces (same gate, cap and dual)
        base = carry_on_pull(env, spec, policy, start, scale, periodic)
        D_c = float(base(states[None], u[None]).sum())
        weight = alpha * D / max(D_c, 0.05)  # floor: a policy that already carries on

        def similarity(s_, c_, f=base, w=weight):
            return w * f(s_, c_)

    u2, s2, K2, _, log = search(
        env,
        start,
        xbar,
        u,
        scale,
        policy,
        alpha,
        budget,
        ITERATIONS,
        1000 * r + k + 1_000_000 * SEED["s"],
        None,
        True,
        None,
        True,
        spec,
        stabilizer,
        extra_starts=extra,
        extra_length=window,
        similarity=similarity,
    )
    extra_rows, robustness_share = [], None
    if extra is not None:
        n = min(window, len(u2))
        xbar2 = np.concatenate([start[None], s2[:-1]])
        passed = 0
        for x0 in extra:
            se, ce = closed_loop(env, x0, xbar2[:n], u2[:n], K2[:n], scale, u2[None, :n])
            healthy = bool(env._is_healthy(se[0, :, 2], se[0, :, 3]).all())
            d = (se[0, -1, spec.observed] - s2[n - 1, spec.observed]) / scale[spec.observed - 1]
            if healthy and np.sqrt((d**2).mean()) <= 1.0:
                passed += 1
                extra_rows.append((np.concatenate([x0[None], se[0, :-1]]), ce[0], K2[:n], scale))
        robustness_share = passed / len(extra)
    row = dict(
        gamma=gamma,
        alpha=alpha,
        moved=bool(np.abs(u2 - u).max() > 0),
        cost=float(step_costs(env, s2[None], u2[None]).sum()),
        does_task=bool(spec.admissible(s2[None])[0]),
        reflex_agreement_before=float(1.0 - ((K_lqr - reaction) ** 2).sum() / (K_lqr**2).sum()),
        gap_before_fit=gap(policy, start, s2, u2, spec),
        accepted=sum(e.get("beta_taken") is not None for e in log),
        start_robustness=robustness_share,
        competence=competence,
    )
    if dual:
        dual_step(alphas, k, row["gap_before_fit"], eps, competence, cap)
    return u2, s2, K2, row, extra_rows


def refit(
    policy, spec, starts, found, gains, row_scales, extra_rows, r, fit_updates, rows, settle=False
):
    """Continue the policy on all trajectories (and the extra-start rollouts) with slope targets."""
    xb = [np.concatenate([s[None], st[:-1]]) for s, (_, st) in zip(starts, found)]
    acts = [u for u, _ in found]
    ks = list(gains)
    scs = list(row_scales)
    for x_e, u_e, k_e, sc in extra_rows:
        xb.append(x_e)
        acts.append(u_e)
        ks.append(k_e)
        scs.append(np.broadcast_to(sc, (len(u_e), len(sc))))
    policy = fit_rows(
        policy,
        np.concatenate(xb),
        np.concatenate(acts),
        np.concatenate(ks),
        np.concatenate(scs),
        SLOPE,
        updates=fit_updates,
        seed=r + 1000 * SEED["s"],
        spec=spec,
        lr_decay=settle,
    )
    for row, start, (u, st) in zip(rows, starts, found):
        row["gap_after_fit"] = gap(policy, start, st, u, spec)
    return policy


def controller_round(
    env,
    spec,
    starts,
    current,
    scales,
    policy,
    budgets,
    r,
    fit_updates,
    alphas,
    eps,
    nearby=None,
    window=250,
    fit_more=(),
    settle=False,
    gate=None,
):
    """One round: per trajectory choose gamma, search with K(gamma), then refit the policy.

    fit_more: further (x_bar, controls, gains, scale) rows for the refit (visited trajectories)."""
    found, gains, rows, extra_rows = [], [], [], []
    for k, (start, (u, states)) in enumerate(zip(starts, current)):
        extra = nearby(k, r) if nearby is not None else None
        u2, s2, K2, row, extra_k = improve(
            env,
            spec,
            start,
            u,
            states,
            scales[k],
            policy,
            budgets[k],
            r,
            k,
            alphas,
            eps,
            extra,
            window,
            gate=gate,
        )
        found.append((u2, s2))
        gains.append(K2)
        rows.append(row)
        extra_rows += extra_k
    row_scales = [np.broadcast_to(sc, (len(u), len(sc))) for sc, (u, _) in zip(scales, found)]
    extra_rows += list(fit_more)
    policy = refit(
        policy, spec, starts, found, gains, row_scales, extra_rows, r, fit_updates, rows, settle
    )
    return found, policy, rows


def choose_carrier(env, spec, starts, current, scales, window):
    """The trajectory whose start-up controller brings the most other training starts back."""
    scores = []
    for k, (u, states) in enumerate(current):
        xbar = np.concatenate([starts[k][None], states[:-1]])[:window]
        K = capped(lqr(*linearize(env, xbar, u[:window], scales[k]), R, spec.free), CAP)
        passed = 0
        for j, x0 in enumerate(starts):
            if j == k:
                continue
            se, _ = closed_loop(env, x0, xbar, u[:window], K, scales[k], u[None, :window])
            d = (se[0, -1, spec.observed] - states[window - 1, spec.observed]) / scales[k][
                spec.observed - 1
            ]
            passed += bool(env._is_healthy(se[0, :, 2], se[0, :, 3]).all()) and bool(
                np.sqrt((d**2).mean()) <= 1.0
            )
        scores.append(passed)
    return int(np.argmax(scores)), scores


def shared_round(
    env,
    spec,
    starts,
    current,
    scales,
    policy,
    budgets,
    r,
    fit_updates,
    alphas,
    eps,
    nearby,
    window,
    carrier,
    fit_more=(),
    settle=False,
    gate=None,
):
    """One round with one start-up controller shared by every trajectory (the carrier's)."""
    n, c, W = len(starts), carrier, window
    others = np.array([starts[k] for k in range(n) if k != c])
    extra = others if nearby is None else np.concatenate([others, nearby(c, r)])
    found, gains, rows, row_scales = [None] * n, [None] * n, [None] * n, [None] * n
    u_c, s_c, K_c, row_c, extra_rows = improve(
        env,
        spec,
        starts[c],
        *current[c],
        scales[c],
        policy,
        budgets[c],
        r,
        c,
        alphas,
        eps,
        extra,
        W,
        gate=gate,
    )
    row_c["shared"] = True
    found[c], gains[c], rows[c] = (u_c, s_c), K_c, row_c
    row_scales[c] = np.broadcast_to(scales[c], (len(u_c), len(scales[c])))
    xbar_c = np.concatenate([starts[c][None], s_c[:-1]])
    for k in range(n):
        if k == c:
            continue
        sk, uk = closed_loop(env, starts[k], xbar_c[:W], u_c[:W], K_c[:W], scales[c], u_c[None, :W])
        x_w = sk[0, W - 1]
        startup_cost = float(step_costs(env, sk, uk).sum())
        prev_u, prev_s = current[k]
        options = []
        if env._is_healthy(sk[0, :, 2], sk[0, :, 3]).all():
            for seg_u, seg_s in ((prev_u[W:], prev_s[W:]), (u_c[W:], s_c[W:])):
                xb = np.concatenate([x_w[None], seg_s[:-1]])
                K = capped(lqr(*linearize(env, xb, seg_u, scales[k]), R, spec.free), CAP)
                st, ct = closed_loop(env, x_w, xb, seg_u, K, scales[k], seg_u[None])
                cost = float(step_costs(env, st, ct).sum())
                if spec.admissible(st)[0] and cost <= budgets[k] - startup_cost:
                    options.append((cost, ct[0], st[0]))
        if not options:
            u2, s2, K2, row, _ = improve(
                env,
                spec,
                starts[k],
                prev_u,
                prev_s,
                scales[k],
                policy,
                budgets[k],
                r,
                k,
                alphas,
                eps,
                None,
                W,
                gate=gate,
            )
            row["shared"] = False
            found[k], gains[k], rows[k] = (u2, s2), K2, row
            row_scales[k] = np.broadcast_to(scales[k], (len(u2), len(scales[k])))
            continue
        _, seg_u, seg_s = min(options, key=lambda o: o[0])
        u2s, s2s, K2s, row, _ = improve(
            env,
            spec,
            x_w,
            seg_u,
            seg_s,
            scales[k],
            policy,
            budgets[k] - startup_cost,
            r,
            k,
            alphas,
            eps,
            None,
            W,
            gate=gate,
        )
        u_full = np.concatenate([uk[0], u2s])
        s_full = np.concatenate([sk[0], s2s])
        row.update(
            shared=True,
            moved=bool(np.abs(u_full - prev_u).max() > 0),
            cost=float(step_costs(env, s_full[None], u_full[None]).sum()),
            does_task=bool(spec.admissible(s_full[None])[0]),
            gap_before_fit=gap(policy, starts[k], s_full, u_full, spec),
        )
        found[k], gains[k], rows[k] = (u_full, s_full), np.concatenate([K_c[:W], K2s]), row
        row_scales[k] = np.concatenate(
            [
                np.broadcast_to(scales[c], (W, len(scales[c]))),
                np.broadcast_to(scales[k], (len(u2s), len(scales[k]))),
            ]
        )
    extra_rows += list(fit_more)
    policy = refit(
        policy, spec, starts, found, gains, row_scales, extra_rows, r, fit_updates, rows, settle
    )
    return found, policy, rows


def cycle_gains(env, spec, policy, xbar, u, scale, gamma):
    """Feedback for the cycle repeated forever: (1 - gamma) periodic tracking LQR (capped) plus
    gamma times the policy's own reaction, along one stride."""
    K = periodic_gains(env, xbar, u, scale, CAP)
    return (1.0 - gamma) * K + gamma * (-slopes(policy, xbar, scale, spec)) if gamma > 0 else K


def locked(env, spec, xbar, u, scale, gains):
    """Lock a stride into a cycle: run its controller stride after stride from its own start until
    it returns to the same state (twice: the second time with gains recomputed along the first
    result). The cycle, or why not: "fell", "does not walk", "does not settle" (mean change of
    the start state over the last 5 strides above 0.1 sd)."""
    cycle = None
    for _ in range(2):
        got = close_cycle(env, xbar, u, gains(xbar, u), scale, CYCLE_LOCK)
        if got is None:
            return "fell"
        start, states, controls, residuals = got
        if not spec.admissible(states[None])[0]:
            return "does not walk"
        if np.mean(residuals[-5:]) > 0.1:
            return "does not settle"
        xbar, u = np.concatenate([start[None], states[:-1]]), controls
        cycle = dict(start=start, s=states, u=controls, xbar=xbar, residuals=residuals)
    cycle["K"] = gains(cycle["xbar"], cycle["u"])
    cycle["advance"] = float(cycle["s"][-1, 1] - cycle["start"][1])
    cycle["residual"] = float(np.mean(cycle["residuals"][-5:]))
    cycle["cost"] = float(step_costs(env, cycle["s"][None], cycle["u"][None]).sum())
    return cycle


def cycle_walk(env, spec, x0, startup, transient, cycle, cycle_scale, budget=np.inf):
    """Shared start-up, then a continuation, then the cycle repeated. The join is the earliest
    pass of the continuation near the cycle: its local minima of distance to the cycle within
    JOIN_NEAR sd, earliest first, then (if none of those keeps walking) the closest places within
    JOIN_MAX sd; up to JOIN_TRIES places each, 25 controls apart; the first that keeps walking to
    the end. Without such a place, the start-up and the continuation alone if they walk; else
    None. A walk counts only if it walks and costs at most `budget` (Kendall's limit)."""
    sx, su, sK, s_scale = startup
    tx, tu, tK, t_scale = transient
    s1, u1 = closed_loop(env, x0, sx, su, sK, s_scale, su[None])
    if not env._is_healthy(s1[0, :, 2], s1[0, :, 3]).all():
        return None
    x_w = s1[0, -1]
    s2, u2 = closed_loop(env, x_w, tx, tu, tK, t_scale, tu[None])
    before = np.concatenate([x_w[None], s2[0, :-1]])
    d = np.sqrt(
        (((before[:, None, 2:] - cycle["xbar"][None, :, 2:]) / cycle_scale[1:]) ** 2).mean(-1)
    )
    phase, dist = d.argmin(1), d.min(1)
    lows = [
        t
        for t in range(len(dist))
        if dist[t] <= JOIN_NEAR and dist[t] == dist[max(0, t - 10) : t + 11].min()
    ]  # passes near the cycle, roughly one per stride, in time order
    early, close = lows[:JOIN_TRIES], [int(t) for t in np.argsort(dist) if dist[t] <= JOIN_MAX]
    tried = []
    for t in early + close:
        if len(tried) == 2 * JOIN_TRIES:
            break
        if t in tried or (t not in early and any(abs(t - q) < 25 for q in tried)):
            continue
        tried.append(t)
        ref, ru, rK = tiled(
            cycle["xbar"], cycle["u"], cycle["K"], cycle["advance"], int(phase[t]), len(tu) - t,
            before[t],
        )  # fmt: skip
        s3, u3 = closed_loop(env, before[t], ref, ru, rK, cycle_scale, ru[None])
        states = np.concatenate([s1[0], s2[0, :t], s3[0]])
        controls = np.concatenate([u1[0], u2[0, :t], u3[0]])
        if (
            spec.admissible(states[None])[0]
            and step_costs(env, states[None], controls[None]).sum() <= budget
        ):
            n_s, n_t = len(su), int(t)
            return dict(
                s=states,
                u=np.concatenate([u1[0], u2[0, :t], u3[0]]),
                K=np.concatenate([sK, tK[:t], rK]),
                scale=np.concatenate(
                    [
                        np.broadcast_to(s_scale, (n_s, len(s_scale))),
                        np.broadcast_to(t_scale, (n_t, len(t_scale))),
                        np.broadcast_to(cycle_scale, (len(ru), len(cycle_scale))),
                    ]
                ),
                join=n_s + n_t,
                join_distance=float(dist[t]),
            )
    states = np.concatenate([s1[0], s2[0]])
    controls = np.concatenate([u1[0], u2[0]])
    if (
        spec.admissible(states[None])[0]
        and step_costs(env, states[None], controls[None]).sum() <= budget
    ):
        return dict(
            s=states,
            u=np.concatenate([u1[0], u2[0]]),
            K=np.concatenate([sK, tK]),
            scale=np.concatenate(
                [
                    np.broadcast_to(s_scale, (len(su), len(s_scale))),
                    np.broadcast_to(t_scale, (len(tu), len(t_scale))),
                ]
            ),
            join=None,
            join_distance=float(dist.min()),
        )
    return None


def cycle_round(
    env,
    spec,
    starts,
    current,
    scales,
    policy,
    budgets,
    r,
    fit_updates,
    alphas,
    eps,
    nearby,
    window,
    carrier,
    cycle,
    transients,
    cycle_scale,
    cycle_budget,
    settle=False,
    gate=None,
):
    """One round with one start-up and one cycle shared by every walk.

    The carrier is improved as in shared_round (its start-up must work from every training start);
    the cycle is improved like any trajectory (task, pull toward the policy as far as the gate
    allows, the limit) and then locked again; every walk is rebuilt from the start-up, its own
    continuation and the cycle; the policy is refit on all of them."""
    n, c, W = len(starts), carrier, window
    others = np.array([starts[k] for k in range(n) if k != c])
    extra = others if nearby is None else np.concatenate([others, nearby(c, r)])
    carrier_budget = budgets[c]
    u_c, s_c, K_c, row_c, extra_rows = improve(
        env, spec, starts[c], *current[c], scales[c], policy, carrier_budget, r, c, alphas, eps,
        extra, W, gate=gate,
    )  # fmt: skip
    xbar_c = np.concatenate([starts[c][None], s_c[:-1]])
    startup = (xbar_c[:W], u_c[:W], K_c[:W], scales[c])
    u2, s2, _, row_y, _ = improve(
        env, spec, cycle["start"], cycle["u"], cycle["s"], cycle_scale, policy, cycle_budget, r, n,
        alphas, eps, None, len(cycle["u"]), gate=gate, dual=False, periodic=True,
    )  # fmt: skip
    gamma = row_y["gamma"]
    # locking finds the cycle nearest the searched stride; it uses the tracking gains (the most
    # stabilizing); the gains the walks repeat it with are chosen below by the join rule
    relocked = locked(
        env,
        spec,
        np.concatenate([cycle["start"][None], s2[:-1]]),
        u2,
        cycle_scale,
        lambda xb, uu: periodic_gains(env, xb, uu, cycle_scale, CAP),
    )
    if isinstance(relocked, dict) and relocked["cost"] > cycle_budget:
        relocked = "over the limit"
    accepted = isinstance(relocked, dict)
    lock_result = "relocked" if accepted else relocked

    def build(K, cyc):
        """Every walk with cycle cyc repeated under gains K: [(walk or None, continuation)]."""
        with_K, out = dict(cyc, K=K), []
        for k in range(n):
            # the carrier's continuation first (the start-up brings every start near the
            # carrier's path), then the walk's own; a walk that joins the cycle is preferred
            walk, used = None, None
            for name, option in (("carrier", transients[c]), ("own", transients[k]))[
                : 1 if k == c else 2
            ]:
                got = cycle_walk(
                    env, spec, starts[k], startup, option, with_K, cycle_scale, budgets[k]
                )
                if got is not None and (
                    walk is None or (walk["join"] is None and got["join"] is not None)
                ):
                    walk, used = got, name
                if walk is not None and walk["join"] is not None:
                    break
            out.append((walk, used))
        return out

    def early(built_walks):
        return sum(
            w is not None and w["join"] is not None and w["join"] <= JOIN_EARLY
            for w, _ in built_walks
        )

    def tracking_of(cyc):
        return cycle_gains(env, spec, policy, cyc["xbar"], cyc["u"], cycle_scale, 0.0)

    # the cycle moves only if the walks can still join it (as many early joins under the tracking
    # gains as before, less one): run 006 was dragged until no walk could join (round 13)
    kept_walks = build(tracking_of(cycle), cycle)
    if accepted:
        moved_walks = build(tracking_of(relocked), relocked)
        if early(moved_walks) < early(kept_walks) - 1:
            accepted, lock_result = False, "walks stop joining"
    if accepted:
        cycle = relocked
    else:  # keep the cycle, with this round's gains
        cycle = dict(
            cycle,
            K=cycle_gains(env, spec, policy, cycle["xbar"], cycle["u"], cycle_scale, gamma),
        )
    # gap and pull-weight update on the cycle actually used (re-locked or kept), not the search's
    row_y.update(
        relocked=accepted,
        lock_result=lock_result,
        residual=cycle["residual"],
        cost_per_control=cycle["cost"] / len(cycle["u"]),
        gap_searched=row_y["gap_before_fit"],
        gap_before_fit=gap(policy, cycle["start"], cycle["s"], cycle["u"], spec),
    )
    D_y = max(row_y["gap_before_fit"], eps) * cycle["u"].size / 2.0
    cap = None if gate is None else PULL_CAP * cycle["cost"] / D_y
    dual_step(alphas, n, row_y["gap_before_fit"], eps, row_y["competence"], cap)

    # the cycle's repeats use the largest gamma (at most the cycle's own) under which walks still
    # join early about as well as under the tracking gains alone (the start-up's rule for extra
    # starts): the gamma test on the cycle's own start does not test joining
    tracking = tracking_of(cycle)
    walks_k = moved_walks if accepted else kept_walks
    tile_gamma, tile_K = 0.0, tracking
    base = early(walks_k)
    for g in (x for x in GAMMAS if x <= gamma):
        K_g = cycle_gains(env, spec, policy, cycle["xbar"], cycle["u"], cycle_scale, g)
        candidate = build(K_g, cycle)
        if early(candidate) >= max(base - 1, 0):
            walks_k, tile_gamma, tile_K = candidate, g, K_g
            break
    cycle = dict(cycle, K=tile_K)
    row_y.update(tile_gamma=tile_gamma, early_joins=early(walks_k), early_joins_tracking=base)
    found, rows, built = list(current), [], []
    for k, (walk, used) in enumerate(walks_k):
        row = dict(
            walks=walk is not None,
            continuation=used,
            joined=None if walk is None else walk["join"],
            join_distance=None if walk is None else walk["join_distance"],
        )
        if walk is not None:
            found[k] = (walk["u"], walk["s"])
            row.update(
                cost=float(step_costs(env, walk["s"][None], walk["u"][None]).sum()),
                gap_before_fit=gap(policy, starts[k], walk["s"], walk["u"], spec),
            )
            built.append((k, walk, row))
        rows.append(row)
    if not built:  # nothing walks this round: keep the policy
        row_y["gap_after_fit"] = None
        return found, policy, rows, row_c, row_y, cycle
    policy = refit(
        policy,
        spec,
        [starts[k] for k, _, _ in built],
        [(w["u"], w["s"]) for _, w, _ in built],
        [w["K"] for _, w, _ in built],
        [w["scale"] for _, w, _ in built],
        extra_rows,
        r,
        fit_updates,
        [row for _, _, row in built],
        settle,
    )
    row_y["gap_after_fit"] = gap(policy, cycle["start"], cycle["s"], cycle["u"], spec)
    return found, policy, rows, row_c, row_y, cycle


def summarize(r, rows, budgets, c0):
    return dict(
        round=r,
        gamma=[x["gamma"] for x in rows],
        mean_gamma=float(np.mean([x["gamma"] for x in rows])),
        gap_before_fit=float(np.mean([x["gap_before_fit"] for x in rows])),
        gap_after_fit=float(np.mean([x["gap_after_fit"] for x in rows])),
        reflex_agreement=float(np.mean([x["reflex_agreement_before"] for x in rows])),
        cost_ratio=float(np.mean([x["cost"] / c for x, c in zip(rows, c0)])),
        all_within_limit=all(x["does_task"] and x["cost"] <= b for x, b in zip(rows, budgets)),
        moved=sum(x["moved"] for x in rows),
        alpha=[x["alpha"] for x in rows],
        start_robustness=None
        if all(x["start_robustness"] is None for x in rows)
        else float(
            np.mean([x["start_robustness"] for x in rows if x["start_robustness"] is not None])
        ),
        shared=sum(x.get("shared", False) for x in rows),
        mean_competence=None
        if all(x.get("competence") is None for x in rows)
        else float(np.mean([x["competence"] for x in rows])),
    )


def point_task(args, out):
    from scripts.diagnostics.point_mass_check import (
        HELD_OUT,
        LENGTH,
        POINT,
        TRAIN,
        XML,
        OptimalFeedback,
        PointMass,
        done,
        drive,
        new_policy,
        starts,
    )

    (out / "point_mass.xml").write_text(XML)
    env = PointMass(out / "point_mass.xml", nthread=args.threads)
    rng = np.random.default_rng(0)
    train, held = starts(rng, TRAIN), starts(rng, HELD_OUT)
    if args.cluster is not None:
        train, held = np.zeros((TRAIN, 5)), np.zeros((HELD_OUT, 5))
        for x in (train, held):
            x[:, 1] = 1.0
            x[:, 1:5] += rng.normal(0.0, args.cluster, (len(x), 4))
    sampler = MPPI(env, MPPIConfig(**dict(json.loads(CONFIG.read_text()), H=LENGTH, K=TRAIN)))
    sampler.reset(seed=0)
    noise = sampler._sample_noise() / 0.3 * 0.4
    noise[:, 250:] = 0.0
    states, controls = drive(env, None, train, noise=noise)
    c0 = step_costs(env, states, controls).sum(1)
    current = [(controls[k], states[k]) for k in range(TRAIN)]
    xbars = [np.concatenate([train[k][None], states[k][:-1]]) for k in range(TRAIN)]
    scales = [x[:, 1:].std(0) + 1e-6 for x in xbars]
    policy = fit_rows(
        new_policy(xbars),
        np.concatenate(xbars),
        controls.reshape(-1, 2),
        np.zeros((TRAIN * LENGTH, 2, 4)),
        np.ones((TRAIN * LENGTH, 4)),
        0.0,
        updates=500,
        seed=getattr(args, "seed", 0),
        spec=POINT,
    )
    A, B = linearize(env, np.zeros((1, 5)), np.zeros((1, 2)), np.ones(4))
    A, B, P = A[0], B[0], np.diag([1.0, 1.0, 0.1, 0.1])
    for _ in range(5000):
        K_opt = np.linalg.solve(0.02 * np.eye(2) + B.T @ P @ B, B.T @ P @ A)
        P = np.diag([1.0, 1.0, 0.1, 0.1]) + A.T @ P @ (A - B @ K_opt)
    s_best, u_best = drive(env, OptimalFeedback(K_opt), held)
    held_best = step_costs(env, s_best, u_best).sum(1)

    def judge(p):
        b, ub = drive(env, p, held)
        return dict(
            held_out_done=float(done(b).mean()),
            held_out_cost_over_optimal=float(np.median(step_costs(env, b, ub).sum(1) / held_best)),
        )

    return (
        env,
        POINT,
        list(train),
        current,
        scales,
        policy,
        args.budget * c0,
        c0,
        judge,
        lambda r: True,
        None,
        lambda p, xbar, u=None, task=None: float(
            done(
                drive(
                    env,
                    p,
                    xbar[np.linspace(0, len(xbar) - 1, GATE_STATES).astype(int)],
                    GATE_HORIZON,
                )[0]
            ).mean()
        ),
        None,
        None,
    )


def walker_task(args, out):
    from scripts.diagnostics.coupled_mppi import WALK_MAE
    from scripts.diagnostics.feedback_rollouts import step
    from scripts.diagnostics.noise_cutoff import observation
    from scripts.diagnostics.policy_picks import load_bank
    from scripts.diagnostics.policy_survival import standing_starts
    from scripts.diagnostics.similarity_loop import SOURCES, TAKEOVERS
    from src.envs.walker2d import Walker2d
    from src.gps.mppi_gps import policy as policy_module
    from src.gps.mppi_gps.diagnostic_policies import NATIVE_TASK

    env = Walker2d(**NATIVE_TASK, nthread=args.threads)
    length = 120 if args.quick else 1000
    sources = SOURCES[:2] if args.quick else SOURCES
    starts, current, scales, c0 = [], [], [], []
    for s in sources:
        with np.load(args.bank / f"episode_{s}.npz") as d:
            S, U = d["states"][:length], d["actions"][:length].astype(np.float64)
            scale = d["states"][:, 1:].std(0) + 1e-6
        K = lqr(*linearize(env, S, U, scale), R)
        st, ctl = closed_loop(env, S[0], S, U, K, scale, U[None])
        starts.append(S[0])
        current.append((ctl[0], st[0]))
        scales.append(scale)
        c0.append(float(step_costs(env, st, ctl).sum()))
    c0 = np.array(c0)
    bank = load_bank(args.bank)
    states0 = np.concatenate([e["states"] for e in bank])
    obs0 = observation(states0)
    policy = policy_module.Policy(
        0, "cpu", dict(mean=obs0.mean(0), std=np.maximum(obs0.std(0), 1e-6)), hidden_dim=256
    )
    torch.manual_seed(getattr(args, "seed", 0))
    policy.network = torch.nn.Sequential(
        torch.nn.Linear(17, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(),
        torch.nn.Linear(256, 256), torch.nn.Tanh(), torch.nn.Linear(256, 6), torch.nn.Tanh(),
    )  # fmt: skip
    policy = fit_rows(
        policy,
        states0,
        np.concatenate([e["actions"] for e in bank]),
        np.zeros((len(states0), 6, 18)),
        np.ones((len(states0), 18)),
        0.0,
        updates=50 if args.quick else 500,
        seed=getattr(args, "seed", 0),
    )
    stand = standing_starts(env)
    mid = np.array([bank[w]["states"][t] for w in range(3) for t in TAKEOVERS])
    held_mid = []
    for e in (7, 8, 9):  # walks the loop never trains on (SOURCES are 0-4 and 6; 5 fell)
        with np.load(args.bank / f"episode_{e}.npz") as d:
            held_mid += [d["states"][t] for t in TAKEOVERS]
    held_mid = np.array(held_mid)
    pooled_scale = states0[:, 1:].std(0) + 1e-6
    section = Section([e["states"] for e in bank])

    def travel(p, x):
        """The policy alone for 1,000 controls: states before each control, controls survived,
        whether it walks (survives and, after the first 100 controls, mean |forward speed - 1.5|
        <= 0.3, S1's standing rule), forward speeds."""
        alive, lasted, speed, seen = np.ones(len(x), bool), np.full(len(x), 1000), [], []
        for t in range(1000):
            seen.append(x)
            u = p.predict(observation(x).astype(np.float32)).astype(np.float64)
            x = step(env, x, np.clip(u, -1.0, 1.0))
            fell = alive & ~env._is_healthy(x[:, 2], x[:, 3])
            lasted[fell], alive = t + 1, alive & ~fell
            speed.append(x[:, 10])
        speed = np.array(speed)
        error = np.abs(speed[100:] - 1.5).mean(0)
        return np.stack(seen, 1), lasted, (lasted == 1000) & (error <= WALK_MAE), speed

    def alone(p, x):
        return travel(p, x)[1:3]

    def judge(p):
        if args.quick:
            return {}
        seen, a, wa, _ = travel(p, stand.copy())
        (b, wb), (c, wc) = (alone(p, x.copy()) for x in (mid, held_mid))
        cycle = section.regularity([seen[i] for i in np.flatnonzero(a == 1000)]) or {}
        return dict(
            standing_median_s=float(np.median(a)) * 0.008,
            standing_walks=int(wa.sum()),
            midwalk_median_s=float(np.median(b)) * 0.008,
            midwalk_walks=int(wb.sum()),
            heldout_midwalk_median_s=float(np.median(c)) * 0.008,
            heldout_midwalk_walks=int(wc.sum()),
            stride_runs=cycle.get("runs", 0),
            stride_change=cycle.get("change"),
            stride_spread=cycle.get("spread"),
            stride_period_sd_s=cycle.get("period_sd_s"),
        )

    def competence(p, xbar, u=None, task=None):
        """Share of 8 states along x_bar from which the policy alone stays up 2 s at >= 0.75 m/s."""
        x = xbar[np.linspace(0, len(xbar) - 1, GATE_STATES).astype(int)].copy()
        alive, speed = np.ones(len(x), bool), []
        for _ in range(GATE_HORIZON):
            u = p.predict(observation(x).astype(np.float32)).astype(np.float64)
            x = step(env, x, np.clip(u, -1.0, 1.0))
            alive &= env._is_healthy(x[:, 2], x[:, 3])
            speed.append(x[:, 10])
        return float((alive & (np.mean(speed, 0) >= 0.5 * 1.5)).mean())

    def own_cost(p, x, length):
        """Task cost of the policy alone from states x for `length` controls (falling keeps costing)."""
        seen, used = [], []
        for _ in range(length):
            u = np.clip(p.predict(observation(x).astype(np.float32)).astype(np.float64), -1, 1)
            x = step(env, x, u)
            seen.append(x)
            used.append(u)
        return step_costs(env, np.stack(seen, 1), np.stack(used, 1)).sum(1)

    def score(p, xbar, u, task):
        """Kendall's score for one trajectory: its cost per control over the policy's own cost per
        control from the same start (at least 250 controls), at most 1."""
        H = max(len(u), GATE_HORIZON)
        return float(min(1.0, (task / len(u)) / (own_cost(p, xbar[:1].copy(), H)[0] / H)))

    accept_starts = [starts[k] for k in range(len(starts))]
    for j in range(32 - len(starts)):
        saved = np.random.get_state()
        try:
            np.random.seed(700_000 + j)  # disjoint from training, nearby, S1 and test seeds
            env.reset()
        finally:
            np.random.set_state(saved)
        accept_starts.append(env.get_state())
    accept_starts = np.array(accept_starts)

    def task_score(p):
        """The policy's task score: MPPI's mean walk cost over the policy's own mean cost for 8 s
        from 32 fixed standing starts (the 6 training starts and 26 draws), at most 1."""
        return float(min(1.0, np.mean(c0) / own_cost(p, accept_starts.copy(), 1000).mean()))

    def mppi_from(x0, length, seed):
        """MPPI driving from x0, re-executed in closed loop like the bank walks; None unless it walks."""
        controller = MPPI(env, MPPIConfig(**json.loads(CONFIG.read_text())))
        controller.reset(seed=seed)
        env.reset(state=x0)
        S, U = [], []
        for _ in range(length):
            x = env.get_state()
            action, _ = controller.plan_step(x, initial_warmstart=env.get_warmstart())
            S.append(x)
            U.append(np.asarray(action, dtype=np.float64))
            env.step(action)
            after = env.get_state()
            if not env._is_healthy(after[2:3], after[3:4]).all():
                return None
        S, U = np.array(S), np.array(U)
        K = lqr(*linearize(env, S, U, pooled_scale), R)
        st, ctl = closed_loop(env, S[0], S, U, K, pooled_scale, U[None])
        if not WALKER.admissible(st)[0]:
            return None
        cost = float(step_costs(env, st, ctl).sum())
        return dict(start=S[0], u=ctl[0], s=st[0], scale=pooled_scale, c0=cost)

    def visit(p, current, r, wanted, length):
        """New trajectories from states the policy visits when it runs alone (Kendall, 2026-11-04)."""
        drawn = []
        for j in range(VISIT_STANDING):
            saved = np.random.get_state()
            try:
                np.random.seed(800_000 + 1000 * r + j)
                env.reset()
            finally:
                np.random.set_state(saved)
            drawn.append(env.get_state())
        rng = np.random.default_rng([r, 80])
        lo = 0 if args.quick else 100
        along = [s[t] for _, s in current for t in rng.integers(lo, len(s), VISIT_ALONG)]
        origins = np.concatenate([np.array(drawn), np.array(along)])
        seen, lasted, ok, speed = travel(p, origins.copy())
        picks = {True: [], False: []}  # by origin: standing draw or along a trajectory
        for i in np.flatnonzero(~ok):
            if lasted[i] < 1000:
                t, why = max(int(lasted[i]) - 1 - VISIT_LEAD, 0), "fall"
            else:
                error = np.abs(speed[100:, i] - 1.5)
                t, why = (
                    100 + int(np.argmax(np.convolve(error, np.ones(125) / 125, "valid"))),
                    "slow",
                )
            picks[bool(i < VISIT_STANDING)].append((seen[i, t], why, bool(i < VISIT_STANDING), t))
        for group in picks.values():
            rng.shuffle(group)
        order = [q for pair in itertools.zip_longest(picks[True], picks[False]) for q in pair if q]
        made, tried = [], []
        for x0, why, standing, t in order:
            if len(made) == wanted or len(tried) == 2 * wanted:
                break
            got = mppi_from(x0, length, 60_000 + 100 * r + len(tried))
            tried.append(dict(why=why, from_standing=standing, at_control=t, kept=got is not None))
            if got is not None:
                got.update(why=why, from_standing=standing, at_control=t)
                made.append(got)
        stats = dict(
            policy_runs=len(origins),
            runs_walking=int(ok.sum()),
            runs_fell=int((lasted < 1000).sum()),
            tried=tried,
            kept=len(made),
        )
        return made, stats

    every = lambda r: r % args.judge_every == 0 or r == args.rounds  # noqa: E731
    return (
        env,
        WALKER,
        starts,
        current,
        scales,
        policy,
        args.budget * c0,
        c0,
        judge,
        every,
        visit,
        competence,
        score,
        task_score,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=("point", "walker"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--fit-updates", type=int, default=None)
    parser.add_argument("--eps", type=float, default=None, help="trust-region gap (per-action MSE)")
    parser.add_argument("--budget", type=float, default=BUDGET, help="cost limit vs start")
    parser.add_argument("--bank", type=Path, default=Path("runs/2026-11-03_sample_bank_001"))
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--starts", type=int, default=0, help="nearby starts per trajectory")
    parser.add_argument("--window", type=int, default=250, help="controls checked from them")
    parser.add_argument(
        "--shared-startup", type=int, default=0, help="controls of one shared start-up"
    )
    parser.add_argument(
        "--cluster", type=float, default=None, help="point mass: starts near (1, 0)"
    )
    parser.add_argument(
        "--visit-every", type=int, default=0, help="rounds between visited-state trajectories"
    )
    parser.add_argument("--visit-new", type=int, default=2, help="new ones each time")
    parser.add_argument("--visit-max", type=int, default=8, help="most kept (oldest dropped)")
    parser.add_argument("--visit-length", type=int, default=500, help="controls each")
    parser.add_argument("--settle", action="store_true", help="refit step size falls linearly to 0")
    parser.add_argument("--judge-every", type=int, default=5, help="rounds between walker tests")
    parser.add_argument(
        "--gate", action="store_true", help="pull scaled by the policy's competence, capped"
    )
    parser.add_argument(
        "--cycle", action="store_true", help="walks share one locked stride after the start-up"
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="with --gate: pull weighted by the policy's task score on each trajectory (Kendall)",
    )
    parser.add_argument(
        "--copy-low",
        type=int,
        default=None,
        help="refit updates fall from --fit-updates at score 0 to this at score 1 (Kendall)",
    )
    parser.add_argument(
        "--accept-if-better",
        action="store_true",
        help="keep a refit only if the policy's task score does not drop",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="shifts every random draw of the loop (0: as before)"
    )
    parser.add_argument(
        "--pull",
        choices=("action", "carry-on", "none"),
        default="action",
        help="what the trajectories are pulled toward",
    )
    parser.add_argument(
        "--pull-always",
        action="store_true",
        help="with --gate: competence taken as 1 (pull from round 1), the cap kept",
    )
    parser.add_argument("--quick", action="store_true", help="plumbing check only")
    args = parser.parse_args()
    if args.visit_every > 0 and args.task != "walker":
        parser.error("--visit-every is walker only")
    if args.cycle and (args.task != "walker" or args.shared_startup <= 0 or args.visit_every > 0):
        parser.error("--cycle needs --task walker and --shared-startup, without --visit-every")
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=False)
    setup = point_task if args.task == "point" else walker_task
    (
        env,
        spec,
        starts,
        current,
        scales,
        policy,
        budgets,
        c0,
        judge,
        every,
        visit,
        competence,
        score,
        task_score,
    ) = setup(args, args.out)
    if (args.score or args.copy_low is not None or args.accept_if_better) and task_score is None:
        parser.error("--score, --copy-low and --accept-if-better are walker only")
    if args.score and not args.gate:
        parser.error("--score modifies --gate")
    gate = (score if args.score else competence) if args.gate else None
    PULL["kind"] = args.pull
    SEED["s"] = args.seed
    if args.pull_always:
        if not args.gate:
            parser.error("--pull-always modifies --gate")
        gate = lambda p, xbar, u=None, task=None: 1.0  # noqa: E731
    visit_length = min(args.visit_length, 120) if args.quick else args.visit_length
    visits, born = [], 0  # visited-state trajectories, oldest first
    fit_updates = args.fit_updates or (200 if args.task == "point" else 500)
    if args.quick:
        fit_updates = min(fit_updates, 30)
    alphas, eps = [None] * len(starts), args.eps or EPS[args.task]

    def nearby(k, r):
        if args.task == "point":
            rng = np.random.default_rng([r, k])
            x = np.repeat(starts[k][None], args.starts, axis=0)
            x[:, 1:5] += rng.normal(0.0, 0.02, (args.starts, 4))
            return x
        drawn = []
        for j in range(args.starts):
            saved = np.random.get_state()
            try:
                np.random.seed(900_000 + 1000 * r + 10 * k + j)
                env.reset()
            finally:
                np.random.set_state(saved)
            drawn.append(env.get_state())
        return np.array(drawn)

    carrier, carrier_scores = None, None
    if args.shared_startup > 0:
        carrier, carrier_scores = choose_carrier(
            env, spec, starts, current, scales, args.shared_startup
        )
    cycle, cycle_info = None, {}
    if args.cycle:
        W = args.shared_startup
        xbars0 = [np.concatenate([x[None], st[:-1]]) for x, (_, st) in zip(starts, current)]
        cycle_scale = np.concatenate(xbars0)[:, 1:].std(0) + 1e-6
        w, a, b, change = best_stride(
            [(xb, u) for xb, (u, _) in zip(xbars0, current)], Section(xbars0), first=W
        )
        cycle = locked(
            env,
            spec,
            xbars0[w][a:b],
            current[w][0][a:b],
            cycle_scale,
            lambda xb, uu: periodic_gains(env, xb, uu, cycle_scale, CAP),
        )
        if not isinstance(cycle, dict):
            raise SystemExit(f"the most repeatable stride does not lock into a cycle: {cycle}")
        mppi_per_control = float(np.mean(c0) / len(current[0][0]))
        cycle_budget = args.budget * mppi_per_control * len(cycle["u"])
        cycle_cost0 = cycle["cost"]
        transients = []
        for k, xb in enumerate(xbars0):
            u = current[k][0]
            K = capped(lqr(*linearize(env, xb, u, scales[k]), R, spec.free), CAP)
            transients.append((xb[W:], u[W:], K[W:], scales[k]))
        alphas.append(None)  # the cycle's pull weight (index len(starts))
        cycle_info = dict(
            cycle_source=dict(walk=int(w), controls=[int(a), int(b)], section_change=change),
            cycle_residual=cycle["residual"],
            cycle_cost_per_control_vs_mppi=cycle["cost"] / len(cycle["u"]) / mppi_per_control,
        )
    policy_score = None
    if args.copy_low is not None or args.accept_if_better:
        policy_score = task_score(policy)
        cycle_info["policy_score"] = policy_score
    rows = [
        dict(round=0, carrier=carrier, carrier_scores=carrier_scores, **cycle_info, **judge(policy))
    ]
    print(rows[-1], flush=True)
    for r in range(1, args.rounds + 1):
        visit_stats = None
        fit_r = fit_updates
        if args.copy_low is not None:  # copy MPPI harder the worse the policy does (Kendall)
            fit_r = int(round(fit_updates * (1.0 - policy_score) + args.copy_low * policy_score))
        if args.visit_every > 0 and (r - 1) % args.visit_every == 0:
            made, visit_stats = visit(policy, current, r, args.visit_new, visit_length)
            for m in made:
                m.update(id=born, born=r, budget=args.budget * m["c0"])
                alphas.append(None)  # index len(starts) + id, used only by this trajectory
                visits.append(m)
                born += 1
            visits = visits[-args.visit_max :] if args.visit_max > 0 else []
        guiding = copy.deepcopy(policy)  # the round's guiding policy, for policy_moved
        visit_rows, fit_more = [], []
        for v in visits:  # same guiding policy as the round's own trajectories
            u2, s2, K2, row, _ = improve(
                env,
                spec,
                v["start"],
                v["u"],
                v["s"],
                v["scale"],
                policy,
                v["budget"],
                r,
                len(starts) + v["id"],
                alphas,
                eps,
                None,
                args.window,
                gate=gate,
            )
            v["u"], v["s"] = u2, s2
            row.update(id=v["id"], born=v["born"], why=v["why"])
            visit_rows.append(row)
            fit_more.append((np.concatenate([v["start"][None], s2[:-1]]), u2, K2, v["scale"]))
        if cycle is not None:
            current, policy, per, row_c, row_y, cycle = cycle_round(
                env,
                spec,
                starts,
                current,
                scales,
                policy,
                budgets,
                r,
                fit_r,
                alphas,
                eps,
                nearby if args.starts > 0 else None,
                args.shared_startup,
                carrier,
                cycle,
                transients,
                cycle_scale,
                cycle_budget,
                args.settle,
                gate,
            )
        elif carrier is None:
            current, policy, per = controller_round(
                env,
                spec,
                starts,
                current,
                scales,
                policy,
                budgets,
                r,
                fit_r,
                alphas,
                eps,
                nearby if args.starts > 0 else None,
                args.window,
                fit_more,
                args.settle,
                gate,
            )
        else:
            current, policy, per = shared_round(
                env,
                spec,
                starts,
                current,
                scales,
                policy,
                budgets,
                r,
                fit_r,
                alphas,
                eps,
                nearby if args.starts > 0 else None,
                args.shared_startup,
                carrier,
                fit_more,
                args.settle,
                gate,
            )
        accepted_policy = None
        if args.accept_if_better:  # no update unless the policy gets better at the task
            new_score = task_score(policy)
            accepted_policy = bool(new_score >= policy_score)
            if accepted_policy:
                policy_score = new_score
            else:
                policy = guiding
        elif policy_score is not None:
            policy_score = task_score(policy)
        if cycle is not None:
            built = [x for x in per if x["walks"]]
            summary_r = dict(
                round=r,
                walks=len(built),
                joined=sum(x["joined"] is not None for x in per),
                join_distance=[x["join_distance"] for x in per],
                join_control=[x["joined"] for x in per],
                on_cycle=float(
                    np.mean([(1000 - x["joined"]) / 1000 if x["joined"] else 0.0 for x in per])
                ),
                gap_before_fit=float(np.mean([x["gap_before_fit"] for x in built]))
                if built
                else None,
                gap_after_fit=float(np.mean([x["gap_after_fit"] for x in built]))
                if built
                else None,
                walk_cost_ratio=float(
                    np.mean([x["cost"] / c0[k] for k, x in enumerate(per) if x["walks"]])
                )
                if built
                else None,
                mean_gamma=row_y["gamma"],
                mean_competence=row_y["competence"],
                cost_ratio=cycle["cost"] / cycle_cost0,  # the cycle against its locked start
                cycle=dict(
                    {
                        k: row_y[k]
                        for k in (
                            "gamma",
                            "alpha",
                            "competence",
                            "moved",
                            "relocked",
                            "residual",
                            "gap_before_fit",
                            "gap_after_fit",
                            "lock_result",
                            "tile_gamma",
                            "early_joins",
                            "early_joins_tracking",
                        )
                    },
                    cost_per_control_vs_mppi=row_y["cost_per_control"] / mppi_per_control,
                ),  # fmt: skip
                carrier=dict(
                    {k: row_c[k] for k in ("gamma", "moved", "start_robustness", "competence")}
                ),
            )
            rows.append(dict(**summary_r, **(judge(policy) if every(r) else {})))
        else:
            rows.append(
                dict(**summarize(r, per, budgets, c0), **(judge(policy) if every(r) else {}))
            )
        on_path = spec.observation(
            np.concatenate(
                [np.concatenate([x[None], s[:-1]]) for x, (_, s) in zip(starts, current)]
            )
        ).astype(np.float32)
        rows[-1]["policy_moved"] = float(
            np.sqrt(((guiding.predict(on_path) - policy.predict(on_path)) ** 2).mean())
        )
        if policy_score is not None:
            rows[-1].update(
                policy_score=policy_score, policy_accepted=accepted_policy, fit_updates_used=fit_r
            )
        rows[-1]["trajectories"] = per
        if args.visit_every > 0:
            for v, row in zip(visits, visit_rows):
                row["gap_after_fit"] = gap(policy, v["start"], v["s"], v["u"], spec)
            rows[-1]["visited"] = dict(
                count=len(visits),
                new=visit_stats,
                mean_gamma=float(np.mean([x["gamma"] for x in visit_rows])) if visits else None,
                gap_before_fit=float(np.mean([x["gap_before_fit"] for x in visit_rows]))
                if visits
                else None,
                gap_after_fit=float(np.mean([x["gap_after_fit"] for x in visit_rows]))
                if visits
                else None,
                cost_ratio=float(np.mean([x["cost"] / v["c0"] for x, v in zip(visit_rows, visits)]))
                if visits
                else None,
                all_within_limit=all(
                    x["does_task"] and x["cost"] <= v["budget"] for x, v in zip(visit_rows, visits)
                ),
            )
            rows[-1]["visited_trajectories"] = visit_rows
        print(
            {k: v for k, v in rows[-1].items() if not k.endswith("trajectories")},
            round(time.perf_counter() - started),
            "s",
            flush=True,
        )
        torch.save(policy.state_dict(), args.out / f"policy_{r}.pt")
        np.savez_compressed(
            args.out / f"round_{r}.npz",
            **{f"controls_{k}": u for k, (u, _) in enumerate(current)},
            **{f"states_{k}": s for k, (_, s) in enumerate(current)},
            **{f"visited_controls_{v['id']}": v["u"] for v in visits},
            **{f"visited_states_{v['id']}": v["s"] for v in visits},
            **{f"visited_start_{v['id']}": v["start"] for v in visits},
            **(
                {}
                if cycle is None
                else dict(
                    cycle_controls=cycle["u"], cycle_states=cycle["s"], cycle_start=cycle["start"]
                )
            ),
        )
    last = rows[-1]
    readings = dict(
        converges_as_controllers=bool(last["mean_gamma"] >= 0.9 and last["gap_before_fit"] <= eps)
    )
    if "held_out_cost_over_optimal" in last:
        readings["matches_known_answer"] = bool(last["held_out_cost_over_optimal"] <= 1.1)
    if "standing_walks" in last:
        readings["policy_walks"] = bool(last["standing_walks"] >= 16)
    tail = rows[-10:]
    if len(tail) == 10 and all("policy_moved" in x for x in tail):
        readings["policy_settles"] = bool(np.mean([x["policy_moved"] for x in tail]) <= 0.01)
    if len(tail) == 10 and all("standing_walks" in x for x in tail):
        readings["walks_stably"] = bool(sum(x["standing_walks"] >= 16 for x in tail) >= 9)
    changes = [x["stride_change"] for x in tail if x.get("stride_change") is not None]
    if len(tail) == 10 and len(changes) >= 5:
        readings["holds_a_stride"] = bool(np.mean(changes) <= 0.4)
    if args.gate:
        readings["does_not_copy_a_falling_policy"] = all(
            x["cost_ratio"] <= 1.2 for x in rows[1:] if x["mean_competence"] < 0.25
        )
    summary = dict(
        task=args.task,
        fit_updates=fit_updates,
        eps=eps,
        budget=args.budget,
        starts=args.starts,
        window=args.window,
        shared_startup=args.shared_startup,
        cluster=args.cluster,
        visit_every=args.visit_every,
        visit_new=args.visit_new,
        visit_max=args.visit_max,
        visit_length=visit_length,
        settle=args.settle,
        judge_every=args.judge_every,
        gate=args.gate,
        cycle=args.cycle,
        pull_always=args.pull_always,
        score=args.score,
        copy_low=args.copy_low,
        accept_if_better=args.accept_if_better,
        pull=args.pull,
        seed=args.seed,
        rows=rows,
        readings=readings,
        seconds=time.perf_counter() - started,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(readings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
