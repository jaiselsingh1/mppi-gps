# Acrobot independent-discovery diagnosis

This is a post-result development note dated 2026-10-27. It changes no task,
reward, controller, qualification criterion or historical run. It uses source
inspection and arithmetic on saved arrays only: **0 new simulator controls, 0
actor forwards, 0 fits, 0 tests, 0 remote calls**. A proposed experiment below
has not been implemented, frozen or executed. The complete remote qualification
archive was not yet available to this review; its independent audit remains the
proper source for final all-bank claims.

## What failed

The locally supplied qualification summary reports two complete ordinary-root
executions at each H=48,96,144,288, with K=512 native Gaussian candidates, 32
additional smooth proposals, 16 memory/fallback proposals and one paid mean
verification per decision. Every condition has 0/2 local approaches and 0/2
captures. Mean complete costs are respectively 1520.164, 1544.730, 1546.171 and
1578.882. The reported total is 452,719,936 completed/requested physical control
slots in 19,200 completed queries; no query errors or unresolved controls are
reported. This is failure to acquire useful independent experience on this
panel, before actor learning or earned guidance. Longer lookahead did not
establish the required contrast. Two roots do not estimate broad reliability.

The two reset seeds, 110000 and 110001, are shared across horizons. The task is
800 controls at .01 s, initially near q=(pi,0), with small angular and velocity
perturbations. Local success requires tip distance <=.5 by control600. Actual
ending success requires the final100 consecutive controls within .2 distance
and joint-speed norm <=.5. All these conditions remain necessary. The terminal
1000 failure charge is actual only at T; the endpoint cost of a truncated
model query is explicitly a heuristic, not a complete continuation target.

## What the available actual H48 trajectory establishes

The preview file is a saved 801-state/800-action complete executed trajectory,
not a hypothetical plan. Its SHA matches its supplied provenance. Its measured
minimum tip distance is 1.822795, and final distance is 1.988601. The shoulder
stays in [2.351286,3.967439] radians around the hanging value pi, while the elbow
ranges [-1.836919,3.458218]. Action RMS is .803768, maximum absolute action1,
and 33.125% of controls have absolute action >=.98. There are73 action sign
switches and mean absolute adjacent action change .285639. It does not fail
merely because commands are absent or numerically tiny. These action statistics
alone do not prove that noise frequency caused failure.

Its stage-cost component sums are target468.136122, velocity1.546706, excess
velocity.186323, control0; terminal50.617325 plus actual failure1000 gives
complete cost1520.486475. Thus the complete failure penalty, rather than a
missing ending charge, is present. The planner did not approach the target and
then merely lose final capture on this trajectory.

Saved model parameters from the separately executed reference ledger allow a
read-only rigid-body energy calculation on these states. With the recorded
mass/inertia/COM parameters and the usual kinetic-plus-gravitational energy,
the preview rises from19.660257 J to a maximum/final34.237362 J; upright rest is
58.86 J. This is arithmetic on recorded q and qvel, not an engine query or an
energy-based teacher. It describes this H48 execution only.

## Concrete source-based hypotheses, and their limits

1. **Independent proposal coverage of coordinated swing-up may be weak.** The
   XML has an unactuated shoulder, one elbow motor with gear2 and native control
   bounds [-1,1], unit-mass one-metre links, damping.05 and unchanged .01 s RK4.
   Most current proposals are per-control IID Gaussian perturbations with
   sigma=.185533 around the planner's own retained nominal. Only32 trajectories
   are broad uniform independent proposals, interpolated through knots spaced
   10 controls (.1 s). Empty memory contributes another16 nominal-centred
   Gaussian proposals; the qualification episodes start with empty memory.
   There is no separate long-block or constant-sign exploration channel. Broad
   low-frequency coordinated energy transfer is therefore a plausible coverage
   gap. The preview's large actions show that increasing amplitude alone is
   not already justified. No actor seeds or restricts any of these proposals.

2. **The existing objective offers a folded-arm basin.** Pure geometric
   arithmetic gives target running cost1 at q=(pi,0), but .5 at q=(pi,pi): the
   folded tip is at height2 and still distance2 from the target. At zero speed,
   the corresponding endpoint heuristics are100 and50. This does not make the
   folded state optimal under complete control; it explains why folding can
   improve a limited plan while missing actual ending success. The preview's
   final distance near2 is consistent with that basin, without proving that
   the sampled bank contained a preferable swing-up that was misranked.
   Changing rewards to remove this basin would change the research task and
   is not proposed here.

3. **Optimization/plan retention remains a separate question.** This search
   scores all proposals, verifies its Gibbs mean, retains the lower-cost
   champion (bank member or mean), and shifts its own plan after each control.
   Legacy MPPI adopted the weighted mean. The small fixed temperature .014377
   can concentrate weights, but current bank effective sample sizes and score
   gaps have not been read in this review. A receding controller can also
   repeatedly improve a hypothetical suffix without executing it. Neither
   softmin collapse nor repeated postponement can be inferred from the one
   actual trajectory or the config alone.

4. **The physics/feasibility question is unresolved, not answered by a named
   reference.** CPU/Warp post-action state-boundary qualification was separately
   repaired and passed before this screen; these results do not establish a
   new integration bug. The fixed energy/LQR reference has an arithmetic-valid
   existing audit but itself achieved0/18 local and capture successes. Its
   maximum energies approach58.735--58.760 J while minimum tip distances remain
   1.649--1.995, illustrating that total energy alone does not certify correct
   shoulder/elbow coordination or capture. This failed reference is not a
   physical impossibility proof, a successful capacity witness, or a source
   of training/proposal actions.

## Historical “best Acrobot” does not qualify this task

The stored selected configuration has H144/K512 and the same temperature and
noise scale. Its metrics report .3 hit/hold/final-hold success over ten seeds,
600 steps and a25-step hold. The old tuning runner uses the environment's
uniform-random two-angle reset with zero velocity, not these near-hanging
starts. Its reported hold uses position success; it does not require the
current final100-control slow capture or include the new1000 ending penalty.
Its mean final tip distance1.1683 and speed norm1.9454 are also not current
competence evidence. No qualifying near-hanging successful MPPI trace was
found in the inspected local artifacts. Stored aggregate tuning metrics alone
cannot supply one. The proposed work must preserve, rather than reinterpret,
this distinction.

## First next action: classify the existing banks, without new queries

After the closed archive is available and hash-bound to its independent audit,
compute the following for every recorded bank and verified mean, separated by
horizon, root and elapsed phase. These fields already exist in raw ModelResult
arrays: pre/post full states, actions, stage components, endpoint heuristic,
hold counts, distances, total scores and query references. Join decisions to
their saved independent plan/nominal/RNG and actual next executed state.

- Count any model paths entering local distance.5, and any truly terminal
  paths satisfying capture. Do not call a near-target nonterminal path a
  successful complete episode. Compare their scores and ranks to the selected
  plan, including stage versus endpoint terms and effective sample size.
- Record selected source (Gaussian, knot, fallback/memory, verified mean),
  shoulder excursion and nearest target approach. If no useful paths were
  sampled, first classify distribution/coverage failure; a critic cannot
  manufacture a successful continuation.
- Where a selected hypothetical plan forecasts an approach, compare its
  predicted suffix and its subsequent replanned nominal/actions. A model/live
  first-step disagreement is a prediction/restore concern; repeated loss of
  a forecast approach under correct first-step prediction is a control or
  commitment concern. Neither is automatically a distribution failure.

This is the smallest evidence that can decide whether a new proposal assay is
justified. It uses no new actor predictions and must not select a checkpoint or
an action for deployment after viewing outcomes.

## One finite follow-up if coverage is the diagnosed bottleneck

Propose a **new, explicitly developmental, two-arm independent-search assay**,
not another broad horizon grid or an actor-training campaign. Freeze before
execution, use fresh reset seeds110100/110101 and private planner seeds30000/
30001 in both arms, H144, all800 actual controls, and the unchanged task,
physics, costs, temperature, Gaussian sigma, warm-start/retention and scoring.
Keep560 bank candidates and one verified mean per decision in both arms.

The control is exactly the frozen production search. The treatment retains
its512 native Gaussian and16 empty-memory fallback slots; replace only the32
short-knot exploration slots with32 independent piecewise-constant uniform
control-space proposals, eight each at block lengths20,40,80,144 controls
(.2,.4,.8,1.44 s), padded/truncated to the actual query horizon. They are
random independent proposals, with no actor, analytic reference, teacher or
state-dependent energy-controller inputs. This tests a specific temporal
proposal family at matched simulation count. Since block versus interpolated
uniform proposals also differ in their temporal RMS distribution, a positive
result would support that proposal intervention as a whole, not isolate a
pure correlation-time effect. Preserve all samples and RNG draws so the
unchanged channels can be audited; do not resample failures or add proposals.

The maximum for these four complete executions, including all560 candidate
slots, mean verification and live controls with the terminal window truncated,
is exactly **235,407,776 simulated/executed control slots**:
4 * (561 * sum(r=1..800,min(144,r)) + 800). Any additional qualification,
verification or failed attempt must have a separate declared reserve and be
charged; no assumed GPU throughput converts this count into a free experiment.
Use the existing qualified backend signature and durable query archive; record
wall time and raw storage too. No actor inference or fitting is required.

Report actual local approach, full capture and full cost for each root, bank
coverage and source of successful acquired experience. A useful development
lead would be2/2 treatment captures with model/live consistency, while retaining
all control results; 0/2 would retain the discovery negative. This small panel
cannot qualify the standalone pipeline or establish efficiency. In particular,
if the control remains0/2 local, treatment capture would repair discovery but
still would not establish the intended local-success/ending-failure contrast.
The original >=.9 local, >=.9 long-capture, <=.5 short-capture and fresh
confirmation/matched-compute requirements remain unchanged. Any later horizon
qualification is a separately frozen, paid experiment. Do not proceed to actor
learning merely because some hypothetical plan or one chosen root looks good.

## Provenance bindings

Paths below are relative to the repository root. This note reads the frozen
packet006 search/workflow, rather than silently attributing later selector
instrumentation to the completed controller. Task, original environment,
MuJoCo wrapper and XML bytes match current sources; current search differs in
later guided-selector evidence logging, outside the independent search path.

| File | SHA256 |
| --- | --- |
| runs/2026-10-27_bidirectional_qualification_summary_001/qualification.json | 3966d1429352b1e75ab0d2d352c9cb9d677703bee38b3d2e25322df95f67311e |
| runs/2026-10-27_bidirectional_qualification_preview_001/provenance.json | dd165a5dee98d48482f9f2b46b0383f72dd0cada9a9e19882c222bc3cb274a04 |
| runs/2026-10-27_bidirectional_qualification_preview_001/dev_h48_0000.npz | e6b3eab364de0d4641eaac7669e329e323c89f1308c653242ea12b2c4df6f311 |
| runs/2026-10-27_bidirectional_reference_001/reference.json | a646ba36b0dc3466d773109edafb2c8e5f1c2da546e55dab3abff9495fc537f3 |
| runs/2026-10-27_bidirectional_reference_001/audit_001.json | 709a3f17cc8a8275300d53e8fca36f093709470a6bb8ea2ca7877823e3f68a0c |
| runs/2026-10-27_bidirectional_reference_001/evidence/analytic_reference/ledger.jsonl | 2e8469d59d88c4901503eea643563e7249711e1e6ee18112391247efe6d830f2 |
| configs/acrobot_best_metrics.json | ae4422554bce3aca6aaa53cc59a077d7af5054e9183df2b2a0b42dbdbd8ac634 |
| scripts/tuning/tune_acrobot.py | 85b7e0868c8134aa4884ec28578ccbb8e5b2d95c48965ab37e25b876511dbeb2 |
| configs/bidirectional_acrobot_qualification_dev001.json | 38f5cb67984c46ce31040e4d2e7f6aa517a8d5a6a7a0d7b0b7c71939a80d9f58 |
| runs/2026-10-27_bidirectional_remote_packet_006/source/src/gps/bidirectional/search.py | 38b029ac1eead9659bba012ad02b8746a15f5bd4a01e6217ae2ef28a0643b9a9 |
| runs/2026-10-27_bidirectional_remote_packet_006/source/src/gps/bidirectional/task.py | 3072e90465fe33e7cf63fad76f7a89760beee60c152386b15d664104208a8c23 |
| runs/2026-10-27_bidirectional_remote_packet_006/source/src/gps/bidirectional/workflow.py | 806fde4f132eda3cb26406a2b764f4320ba80636db85897f380c206550eaccc0 |
| runs/2026-10-27_bidirectional_remote_packet_006/source/src/gps/bidirectional/reference.py | 42c8497a68f4bb322e1b2307325769fcdfcae4cd97486301a04e575f8a826ca3 |
| assets/acrobot.xml | 324c8a6bf3bfea0b7faf72395dd3ac7a75cc9c2047159d1f1cd212aeee203cbe |
| src/envs/acrobot.py | 9c92efaf5a76de52f96b38daedbc52b127e73615aa86225c5a0f7edf52dd145e |

## Addendum: fixed saved-bank analysis completed, then a new version authorized

This section was added after the original note and after the predeclared bank
analysis completed. The completed qualification arithmetic audit is PASS:
`runs/2026-10-27_bidirectional_qualification_summary_001/audit_001.json`, SHA256
`8f2420598b27970e2ba8fd6a21b22b0e641cafe4767ea5fae46d55a852770534`.
Its wall time is942.0703224159988 s. This is arithmetic verification of saved
evidence; it does not change the failed scientific qualification.

The fixed sample is the eight horizon/root executions at elapsed
0,100,200,300,400,500,600,700: **64 banks of560 candidates plus64 separately
verified means**, not all6400 decisions. Sample specification and analyzer were
declared before these bank outcomes were read. No candidate or verified mean
in that sample predicts a local approach by the true deadline; no sampled
true-ending opportunity predicts capture. The smallest distance anywhere in
the sampled candidate states is1.5489531520294986, at H288, still outside local
distance.5. Minimum predicted distances by horizon are H48:1.810342398,
H96:1.861971460, H144:1.914528634, H288:1.548953152. The maximum saved selected
first-step model/live state discrepancy is7.363145826388973e-7. Thus this sample
does not show a useful approach being present but discarded by the scorer;
it supports investigating proposal coverage before actor fitting. It does not
prove no useful plan existed in the unsampled banks or in control space.

The result is
`runs/2026-10-27_bidirectional_acrobot_bank_diagnosis_001/output_001/bank_diagnosis.json`,
SHA256`891d889602488c0fb26c02a599ced09105d36043d4828735bea7adcb304dc5e6`;
`candidates.csv` retains all35840 candidate rows plus64 mean rows. Its bound
source/specification/provenance and consumed-file digests are in that result
and its output manifest. Analysis wall time is7.319022235999 s, with0 new
physical controls, actor inferences or fits.

On this evidence, the parent authorized the narrow new independent block
proposal version described above. The new `SearchConfig` option
`independent_block_lengths=(20,40,80,144)` changes only the32 broad slots,
cycling the four lengths eight times, with explicit per-row provenance
`independent_blocks:<length>`. Empty/default lengths retain the original knot
proposals. Both versions first draw the identical baseline uniform knot-value
matrix; block rows use only its first ceil(h/block_length) entries while all
unused draws remain consumed. Blocks must be integer lengths at least the
knot spacing. Consequently native Gaussian and memory/fallback arrays and
private RNG endpoints remain byte-identical when sampled from the same planner
state. There are no actor, critic, witness or reference inputs. JSON lists are
canonicalized to immutable tuples for configuration/snapshot comparison.

The new source and synthetic proposal tests are prepared for accountable test
execution and freezing by the parent; this diagnosis did not run them or launch
the four proposed physics executions. Their fixed235,407,776-control upper
bound, fresh roots110100/110101, planner seeds30000/30001, H144, cold memories,
unchanged task/ending criteria and development-only interpretation remain as
declared. The old qualification, source packet, audit and negative results
remain unchanged.

## Addendum: native conventions, energy and phase, 2026-10-27

This additive review inspected no temporal-assay or Ending-v3 outcome. It ran
only a new standalone NumPy/JSON analysis of existing source bytes and saved
reference/bank arrays. Its [script](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_acrobot_static_diagnosis_001/analyze.py)
and [receipt](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_acrobot_static_diagnosis_001/analysis_001.json)
record **0 new simulator controls, actor forwards, fits, tests or remote calls**.
The timed analysis took **0.384051125 s** at
`2026-10-27T06:02:03.542997+00:00`; this is the script's analysis time, not total
human or previous development time. It binds all18 consumed episode NPZs,
the reference ledger/result, the fixed-bank result, and current/frozen sources.
The original document prefix had SHA256
`40b5da3ddbed268f2ddf0e4ec546f36ece6c0f79764232aa470162a17e91d1c4`;
the frozen packet007 document remains untouched.

### No new angle, action-scale or reset-to-zero bug was found

The XML places the shoulder at world height2, with two unit-length links and
positive-y shoulder/relative-elbow axes. Thus **q=(0,0) is upright**, and
q=(pi,0) hangs down. For q=(q0,q1), the exact tip position is

```
x = sin(q0) + sin(q0+q1)
y = 0
z = 2 + cos(q0) + cos(q0+q1).
```

The current task uses this POST-action full-state geometry for primary cost,
distance and capture. The old PRE/POST sensor-boundary defect is a separately
preserved, repaired historical finding. The reference's positive-cos potential,
wrapped angles and zero-angle upright LQR convention agree with this XML.
Textbook Acrobot formulas using zero at hanging rest require a coordinate
conversion; their literal signs should not be substituted here.

The wrapper sends native actions directly to `data.ctrl`. The motor has gear2
and bounds[-1,1], giving elbow torque **tau=2u**, maximum2 N m; there is no
additional software action scaling. Each control integrates one .01 s RK4
step. The XML disables constraints, so the drawn floor supplies no contact
obstacle. These observations follow the source and MuJoCo's documented motor
and joint-transmission semantics. [MuJoCo XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html#actuator-motor)

`IndependentSearch.__init__` initializes its native plan to zero once.
`_sample` retains that plan exactly as Gaussian row0. `propose` compares the
best bank row with a separately queried mean, retains the winner, shifts it,
and repeats the final action into the tail. It does **not** reset the nominal
to zero every decision. The full snapshot also retains q, joint velocity,
elapsed time, hold count and warmstart. A finite planning horizon therefore
does not erase physical energy or make multicycle swing-up impossible.

### What the cost does and does not discourage

At the POST boundary, with d the target distance and v the two joint speeds,

```
r = exp(log(.1) * (max(d-.2, 0)/2)^2)
c_target = 1 - clip(z/4, 0, 1) + .05*x^2
c_stage = c_target + .15*r^2*((v0/2.5)^2+(v1/5)^2)
          + .05*sum((max(abs(v)-8, 0)/8)^2).
c_endpoint = 100*c_target + 25*r^2*((v0/2.5)^2+(v1/5)^2).
```

The action-cost coefficient is zero. Stage costs are summed per control,
without multiplying by dt. Near ideal hanging rest, r is about.000245 and
r² about6e-8: the target-gated speed penalty is negligible there. Speeds over8
have a soft penalty, not a hard physical bound. A claim that the cost simply
forbids the velocity needed for pumping is unsupported.

For small deviations delta0=q0-pi and delta1=q1, direct Taylor expansion gives
`c_target = 1-.05*delta0^2-.05*delta0*delta1-.075*delta1^2+O(||delta||^4)`.
The quadratic is negative definite: the exact hanging geometry is a local
maximum of the static target term. This does not eliminate dynamic or folded
limited-horizon traps, but it rules out calling hanging rest a static local
minimum of that term.

Every model query includes the endpoint heuristic. Actual failure1000 is
included only when the query reaches the true episode endT800. With H144 that
first becomes visible at elapsed656; with H288 at elapsed512. Actual capture
still requires the final100 consecutive controls at distance<=.2 and speed
norm<=.5. A truncated endpoint is not a complete continuation target.

### Energy is available in the model; correct configuration remains unproved

Using only the saved physical parameter packet, define
`a=m1*c1+m2*l1=1.5`, `b=m2*c2=.5`, and

```
E(q,v) = .5*v.T*M(q1)*v
         + g*((m1+m2)*pivot_height + a*cos(q0) + b*cos(q0+q1)).
```

Ideal upright/hanging rest energies are58.86/19.62 J, a39.24 J difference.
For this native torque and the recorded viscous damping, the continuous
mechanical identity is `dE/dt = 2*u*v1-.05*(v0^2+v1^2)`.
Consequently ideal hanging-rest to upright-rest motion needs at least19.62 rad
of **total elbow travel**, ignoring damping. This is a necessary work bound,
not net elbow displacement, a minimum-time calculation or a reachability
proof. Actual perturbed starts change the required energy; discrete RK4
trajectories need not satisfy the differential identity exactly.

For additional scale, diagonalizing the hanging-rest undamped linear system
with the saved mass matrix gives periods **.913415902 s and2.356877965 s**.
These local analytic periods motivate examining temporal proposal structure;
they do not specify a nonlinear swing-up frequency or require MPPI to see a
whole period at once. H144 covers1.44 s, and the treated broad blocks last
.2,.4,.8 or1.44 s. Receding actions could still accumulate energy over8 s.

The failed reference applies direct bounded native torque
`clip(.1*(E_target-E)*v1 + .2*cos(2*pi*t), -1, 1)` outside its upright switch.
The elbow-velocity factor has the correct direct torque-power sign here.
This is a disclosed simple energy heuristic, **not a validated implementation
of Spong's partial-feedback-linearization swing-up design**.
[Spong, 1995, original paper](https://www.diag.uniroma1.it/~oriolo/ur_LL/material/Spong_SwingUp.pdf)

All14,400 saved reference action decisions used `energy_shaping`; none used
`upright_lqr`. Independent energy arithmetic gives maximum58.760223357 J.
At that maximum, episode`reference_015`, elapsed702, wrapped angles have
norm3.294089369 rad and speed norm12.074875315 rad/s: the system has nearly
the upright-rest total energy while folded and moving rapidly. All18 complete
episodes still failed local approach and capture. Energy regulation alone
does not specify the desired configuration, phase or slow capture. The
large LQR gains and saturation also do not certify its bounded capture basin;
because this run never entered the LQR branch, that issue is not an observed
cause of these failures.

### The remaining proposal hypothesis is narrower than “no history”

Broad knot/block candidates are regenerated relative to each planning window.
The retained nominal carries a finite action sequence, but tail filling holds
the last action; there is no persistent oscillation phase/coefficient object
that extrapolates a waveform after the retained suffix is exhausted. This is
a concrete representation hypothesis, not evidence that a phase proposal
would solve the task. The actor does not restrict these channels.

The fixed64-bank evidence contains35,840 bank candidates and64 verified means.
Its minimum candidate/mean distances are1.548953152/1.668797983, outside local
distance.5. Gibbs effective-sample-size quartiles are1.000000006,2.502935248,
20.424057149;44 selected plans came from bank rows and20 from verified means.
Thus phase cancellation by averaging is a possible phenomenon to check, not
an explanation established for every failed decision. In this fixed sample
there is no local approach for the selector to recover. The sample does not
establish exhaustive control-space failure or eight-second infeasibility.

The next unestablished link remains a complete independent native-control
feasibility witness under the same task and resets. The failed energy
reference and old, differently qualified tuning scores do not supply it.
No ordinary Acrobot actor-learning, actor-to-planner feedback or efficiency
claim follows from this diagnosis. A separate
[phase-proposal draft](/Users/jaisel/Documents/mppi-gps/docs/bidirectional_phase_proposal_protocol_DRAFT.md)
is conditional and performs no new experiment.

### Addendum source identity

All six current source files below were byte-identical to packet007 at this
analysis. The receipt additionally records every consumed raw-input SHA256.

| Input | SHA256 |
| --- | --- |
| assets/acrobot.xml | 324c8a6bf3bfea0b7faf72395dd3ac7a75cc9c2047159d1f1cd212aeee203cbe |
| src/envs/acrobot.py | 9c92efaf5a76de52f96b38daedbc52b127e73615aa86225c5a0f7edf52dd145e |
| src/envs/mujoco_env.py | ebc5833d2cf0dd7242e2e4674cc1099dda964cfd4ad8bcc3f59c1c319d61e502 |
| src/gps/bidirectional/task.py | 3072e90465fe33e7cf63fad76f7a89760beee60c152386b15d664104208a8c23 |
| src/gps/bidirectional/search.py | 35825ca22e0823fcfaa15b12370dc78e31c5f4e065a0a0f8a1f0600a93f7e8cf |
| src/gps/bidirectional/reference.py | 42c8497a68f4bb322e1b2307325769fcdfcae4cd97486301a04e575f8a826ca3 |
| static analysis script | 1e3e5390c7c01a81af3d0aa5bf567580315ae2816f1b6d09d5c7aa091392fd04 |
| analysis_001.json | e69943957cbd4a27ceb22c91dd3924f32fa9f928d2ee04c31c555954f2e0cd05 |
