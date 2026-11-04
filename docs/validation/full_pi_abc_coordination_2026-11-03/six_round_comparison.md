# Six-round comparison versus held P1.1

**Recommendation: keep P1.1 held.** The completed six-round experiment already
tests the same full-horizon fixed-reconstruction mechanism with exceptionally
strong overlap. Its acquisition block never changed a teacher. A new twenty-round
run of this geometry is not presently justified merely by having more rounds,
fresh seeds or available capacity. This is a research-allocation judgment, not
a proof that its result is predetermined: material implementation and protocol
differences remain below. The current coordination request supersedes automatic
launch on a future capacity pass.

This review reads the other branch/worktree **without changing it**. It launches
no physics, fitting, policy inference or experiment. S-* evidence denotes
`codex/full-pi-alternating-bounded`, frozen scientific commit
`627d4329fad537e32b4662e2cebaba3896d58d17`; P-* denotes our
`codex/full-trajectory-pi` P1.1, frozen source
`0570f3c5b48c07a5ae88161e994f6cbe29fc9271`. Exact file bindings appear below.

## Recipe overlap and differences

| Ingredient | Completed six-round experiment | Held P1.1 | Implication / evidence |
| --- | --- | --- | --- |
| Mechanism | C+alpha D; D=half summed pre-action torque residual; alpha 0.12 versus 0 | Same | Already tests repeated alternation, not just one fit. S-CFG/S-SRC/P-CFG/P-OPT |
| Parents and original controls | Three W2 update2000 packages; two H4 independent tapes | Exact same three checkpoint and two original-recording SHA256s | Parent/teacher identity joins pass. S-IN/P-CFG |
| Physical core and task | Walker H1000/K256; target 1.5; original cost | Same | Task dictionaries match excluding thread count. Core MPPI, fitter, both environment files, math and XML are byte-identical in both frozen exports. |
| Search geometry | Sigma 0.3, order 2 low-pass 23.4375 Hz/125 Hz, ESS 0.05, 100 passes, geometric 1→0.1, sample 0 incumbent | Same | The failed acquisition geometry is directly relevant. S-CFG/P-CFG |
| State advancement | Fixed complete initial context, no shift or live execution during solve | Same | Both are full-trajectory PI, not MPC. S-SRC/P-OPT |
| Outer learning |6 rounds × 250 updates = 1,500 new per lineage, total 3,500 |20 rounds × 100 = 2,000 new, total 4,000 | Different policy-feedback cadence and endpoint; twenty rounds are not a continuation of the saved six. |
| Learner | Existing 17/128/128/6 tanh torque policy, frozen normalization, batch 256/LR 3e-4/eta 0/clip 0.5, Adam/private-PCG continuation | Same mathematical learner and private continuation; nine exact inherited stream files | No architecture or representation contrast. Global envelopes differ: 4100000000+stream versus inherited 2841000000+stream. |
| Action implementation / runtime | Existing Torch Policy.predict throughout; macOS ARM Python3.11.15/Torch2.10.0 | Canonical grouping-invariant NumPy FP32 actor for scoring/execution, Torch fit; Linux x86 Python3.11.16/Torch2.10.0+cpu | Not numerical identity. P1.1 retains worker-native states/C 376.5714/392.5842; six-round teachers use Mac C 372.7087/392.2749. |
| Planner randomness | Seed 4110000000+100*stream+slot, PCG carried across rounds | Fresh seed 2801000000+1e6*stream+1000*round+10*slot, plus prototype 20000 offset | Same sampling family, different realized banks/history. |
| Returned nominal | Best walking/cost-admitted scored nominal by J; raw best also recorded | Best raw scored nominal by J, then accept or retain previous tape | Logical distinction; inactive in all 72 observed solves because raw best was the incoming tape and none was rejected. |
| Final weighted mean | Saved explicitly unscored/ineligible | Forecast for diagnostics, still ineligible | Extra measurement/compute, not an alternative update. |
| Student-state queries | Common task-only H1000/I25 at 0/100, original slot 0 template, fresh prescribed query RNG | Same anchors/horizon/template/budget, but each arm uses its own fixed alpha and post-fit actor | Different diagnostic operator; P1.1 direct policy pull can mechanically shrink raw gap. Both require walking qualification. |
| Evaluations and controls | Same 16 primary reset identities reused every round; 16 disjoint final confirmation; round 0 parent |16 fresh paired resets each round; final disjoint 16; frozen parents reevaluated on every panel | Stronger prospective reset separation/parent pairing in P1.1; does not fix acquisition collapse. |
| Gap metrics and windows | Action MSE; early 1–2 versus late 5–6 | Teacher mean pathwise L2 RMS and query first-action L2; early 1–5 versus late 16–20 | Thresholds are not numerically interchangeable; neither small error nor longer survival establishes walking. |
| Evidence / alpha0 inference | Population D computed in both acquisition arms; sparse full banks 0/50/99, query 0/12/24 | Alpha0 population D missing, nominal D measured; acquisition 0/49/99, no full query banks | Different diagnostic costs/coverage, same alpha0 optimization score C. |

## What the saved result establishes

The coordinator completed all three streams, with no quality-dependent stop.
S-AUD passed 1,944,260 checks; S-JOIN passed 698 additional continuation/context
checks. I separately joined the three original parent hashes and two teacher
hashes and reduced the saved stream/summary records, without evaluating a policy.

- All 72 acquisition solves returned unchanged input tapes, with zero objective
  gain; raw best was also the incumbent. This is not retention caused by rejecting
  otherwise improved teaching data with an agreement proxy.
- The saved-prefix census checks 1,843,200 acquisition candidates: only the 72
  initial incumbent samples reached H1000. All 7,128 later nominals failed full
  survival. Sparse saved states and per-candidate length records support distinct
  audit scopes; unpreserved candidate physics was not exhaustively rerun.
- Updated actors participated in later acquisition; the agreement term changed
  weights and first-update displacements. Nevertheless teachers, fitting arrays,
  policy/optimizer identities and evaluation outcomes were identical across arms
  at every paired endpoint. The trajectory block was ineffective under this
  allocation, leaving repeated cloning of the unchanged two tapes.
- Walking was 0/16 in every round/stream/arm primary panel and every confirmation.
  Reconstruction fell from about 0.205 to 0.133–0.135. Streams 1/2 survival temporarily
  reached 16/16 and 9/16 at round 5, then 1/16 and 0/16 at round 6; all confirmation
  survival was 0/16. Retention improvement did not become competent walking.
- 82/84 student-state anchors were available; all 82 selected query trajectories
  fell. Competent-query coverage was zero, so qualified gap closure is unresolved,
  not zero disagreement and not a successful recovery expert.

These banks are correlated and nested within **two source trajectories**, not
1.8 million independent task replications. The result identifies failure to
obtain useful acquisition movement under its frozen recipe; it does not isolate
the similarity term as the unique cause or prove full-trajectory PI impossible.

Science consumed 2,368,006,000 model controls, 316,190 native controls, 341,280,156
forward rows and 9,000 Adam updates, with no unknown partial operations in the
closed audit. Scientific counters passed their finite ceilings. The published
validation explicitly cannot certify exact all-in engineering cost or its
engineering-wall ceiling; supplementary readers and publication reducers retain
their failures/errata. This qualification does not erase the completed scientific
comparison. Full raw evidence is local-only; the publication manifest does not
assert a remote backup. Audit physics was limited replay, not every hypothetical
candidate. S-AUD/S-JOIN/S-PUBLISH state these scopes.

## Consequence for the coordinated plan

P1.1 has **no alternating-loop result**: origin replay and native revalidation
passed, but P0/learning did not launch. Its different fit cadence, numerical
backend, random banks, diagnostics and fresh panels mean the six-round result
cannot be relabelled as a P1.1 execution or a theorem about round 20. Nonetheless,
those differences do not presently provide a discriminating reason to spend
another 240 acquisition solves while leaving the same observed feasibility
bottleneck untouched. Preserve the prepared P1.1 package and held status while
coordination resolves what new information a next version would target.

Pre-update teachability remains open. The six-round post-fit reconstruction and
unsuccessful-query diagnostics did not validate a pre-update score that predicts
walking after a fixed learning update. Smaller D, identical fitted policies or
retention alone cannot fill that gap. Neither branch has established the broader
equal-compute cloning/DAgger, CIO convergence/ablation or second-body criteria.
This review selects no new intervention and authorizes no follow-up experiment.

## Exact evidence bindings

Paths are retained as read-only local locations; S-* files remain owned by the
other branch. SHA256s bind what was inspected, not a new independent reproduction.

| ID | Absolute file | Bytes | SHA256 |
| --- | --- | ---: | --- |
|S-CFG|`/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/results/full_pi_alternating_001/run/config.json`|3143|`789078961866fbf07420ae0d3d3da07545df45d36298097b9ca27fb1db18958f`|
|S-SRC|`/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/runs/2026-11-03_full_pi_alternating_bounded_001/source/src/gps/mppi_gps/full_pi_alternating.py`|30970|`62a1432e0506e41e832e56cbc9e589d97ae1f15c55b69020531ed3a58793a378`|
|S-IN|`/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/results/full_pi_alternating_001/run/input_manifest.json`|1240|`363401e8e2dfdc7d1ab16f824ea60b65e3ff3dd78064849020c948ae2b580dfd`|
|S-AUD|`/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/results/full_pi_alternating_001/audit_001/audit.json`|959|`5cbf604e51ef1f1980e114a0de493936b0933d97b0a607b602011ba3842a0675`|
|S-JOIN|`/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/results/full_pi_alternating_001/join_audit_001/audit.json`|551|`9f37d82a5f9c8d04e183393d03426a5763e2bbdef1a29efb8e12142136a2d39d`|
|S-SUM|`/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/results/full_pi_alternating_001/report_001/summary.json`|28365|`0e0518e8753c86a710d851d5688220ededb86487c5d7e7acb66a1847cb473e50`|
|S-DETAIL|`/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/results/full_pi_alternating_001/trace_report_001/supplement.json`|145722|`ce7c2d9504bdb2e038aead13b360ea82e341b9a8403f684c6a33896b6af87842`|
|S-PREFIX|`/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/results/full_pi_alternating_001/accounting/prefix_audit_receipt.json`|439|`8c52ab6c626b0dc5e55daac666c50324f042138383157459a23c507d15a00dcf`|
|S-PUBLISH|`/Users/jaisel/Documents/workbench/mppi-full-pi-alternating-bounded/results/full_pi_alternating_001/validation_receipt.json`|1036|`cc576d127aab0c32e259d99b0843ff1da456c0e1fd3244f91735168b9d7d65f0`|
|P-CFG|`/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi/configs/studies/trajectory_pi_p1_1.json`|80484|`4c26199f9c0bdf18d1057200feb010c5c8d7f417eda48224ee43e58940690089`|
|P-OPT|`/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi/runs/2026-11-03_trajectory_pi_p1_1_001/source/src/gps/mppi_gps/trajectory_pi_optimizer.py`|15562|`243aca969d1ac208f99915fa7cace835a4405251db60cb06240df3d56949206f`|
|P-PREP|`/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi/results/trajectory_pi_p1_1_001/frozen/prepared_manifest.json`|11818|`cf00c63ba713af0cdd8211c7f8a22f88862ff4923d724b916c90da3399d58384`|
|P-RV|`/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi/results/trajectory_pi_p1_1_001/native_validation/result.json`|11154|`119b7866ea19b604a605afff1bf1605609ab08783747b01b52d04624900c5e9b`|
|P-HOLD|`/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi/results/trajectory_pi_p1_1_001/prelaunch_admission.json`|1117|`89358838cd7e2f0dde818c77d80d05cc1a12e654cec5550368a9df93c2c86097`|

Shared frozen dependency SHA256s, independently compared byte-for-byte:

| File | SHA256 in both exports |
| --- | --- |
|`src/mppi/mppi.py`|`0368614152155a63cf41919fcbe5a8c1d2caee79f3144e106c806f3922cfec99`|
|`src/gps/mppi_gps/policy.py`|`e12860aaf55cc602517462e29f6d24163391254ec36589df42fdb681560f8abc`|
|`src/envs/walker2d.py`|`1ade23d8fd8f6162cbbf3fd4d86c161726ebbbd5fdd0058c64df31f1b5f9b5c6`|
|`src/envs/mujoco_env.py`|`1abf27853b56288d7371d92db1d993a6437a194a50446b34c04c7f6620dd662a`|
|`src/utils/math.py`|`fb50fb821540a52f727b906e7d6b05ffd231ce9764dffeb49f3599cd431038bd`|
|`assets/walker2d.xml`|`4f33bf5069a0bf65056871ab4579f09c62696e7f46c3af6936efedacc572d590`|
