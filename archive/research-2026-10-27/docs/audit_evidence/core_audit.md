# Independent core audit — 2026-10-23

Scope: code, runtime arrays, exact simulator calls, external reference implementation and tests. Initial checkout `01eb594ccba3780f29179f14db3caf9755c5d6dc`; no theory/objective/log narratives were used to establish the findings below. Server live training checkout was not edited. Server experiments ran in `/home/jaisel/mppi-gps-audit-core-20261023`. Five concurrent training processes were active, so timing is a profile under contention, not a throughput benchmark.

## Findings and fixes

### P1 — CPU and GPU policy tails used different solver-state semantics (fixed locally)

`WalkerPolicyTail.__call__` calls MuJoCo's batch rollout separately at every tail control boundary with `mjSTATE_FULLPHYSICS`, which excludes solver warmstart, and no `initial_warmstart`; each tail control begins with zero solver warmstart. `_Worlds.rollout` carried the previous control's solver warmstart, including from the planning horizon into the tail. Therefore the advertised identical CPU objective/physics path was false even before considering float32 and GPU nondeterminism.

A regression (`tests/test_gpu_tail_semantics.py`) runs the actual GPU `_Worlds.rollout` and `_cost` methods on CPU tensor views with an exact MuJoCo graph-launch substitute, eliminating precision/hardware confounds. One solver iteration makes initial solver state observably relevant: before the fix GPU-orchestrated cost was -19.3546640848, CPU reference -19.2090091072, difference 0.145655. Red evidence: `gpu_tail_semantics_red.txt`. The fix adds an optional per-control warmstart reset to `_Worlds.rollout` and enables it only for the policy tail. The regression passes to 1e-12 after the fix (`gpu_tail_semantics_green.txt`). This preserves the existing CPU definition. It does not claim that cold-start CPU tails are the best numerical approximation of continuous physics.

Old GPU results remain results of the old solver-state convention. They cannot be silently relabeled as fixed-version experiments. The live training checkout was not changed; new claims need new runs on an identified revision.

### P1 — At the actual training horizon, residual CPU/GPU discrepancy changes predictive-check decisions (unresolved numerical validity)

The existing test (K64,H8,tail10, random actor, three states) passes before and after the fix. A stronger audit uses the real checkpoint `2026-10-23_step14_check_seed0/policy_058.pt`, K256,H32,tail100, reset123456, states at t=0,20,60, and identical planner noise seeds 811000–811002. It includes the partial final batch (batch size2).

| State | Original CPU–GPU label L2 | Fixed CPU–GPU label L2 | Fixed maximum action-coordinate difference | Fixed GPU repeat L2 |
|---|---:|---:|---:|---:|
| t0 | .021574 | .022784 | .015859 | .026265 |
| t20 | .069670 | .042704 | .026545 | .020514 |
| t60 | .036858 | .022640 | .012902 | .019840 |

Every fixed maximum-coordinate difference exceeds the short test's 0.01 tolerance. At t20 the CPU rejects the label because plan cost194.886 > nominal192.577; both original and fixed GPU runs accept it (fixed plan190.709 < GPU nominal). GPU self-repeat differences are also nonzero. Three states cannot establish the rate of these flips, and similar repeat-noise magnitude does not prove that CPU/GPU bias is harmless. A larger prespecified state/checkpoint suite must measure label discrepancies and predictive-check flip rates before claiming interchangeable E-steps. Do not replace this with an arbitrary relaxed tolerance.

Raw original/fixed arrays, checkpoint hash, all CPU/GPU diagnostics and exact input states: `gpu_differential_profile.json`, `gpu_fixed/gpu_differential_profile.json`, `gpu_audit_inputs.npz`. Reusable driver: `scripts/benchmarks/audit_gpu_estep.py`.

### P2 — Softmin failed on temperatures its own budget solver returns (fixed)

`temperature_for_budget` returns infinity for full trust and zero for a constant-score infeasible budget. `softmin_weights` formerly produced NaNs at zero, and at infinity when any score was nonfinite. It also returned bad probabilities rather than rejecting negative/NaN temperatures, and raised an incidental empty-reduction error on an entirely invalid batch.

Six failing regression cases were written before the change (`math_boundary_red.txt`). The fix implements the zero-temperature limit (uniform over finite minima), the infinite-temperature limit (uniform over finite scores), and explicit rejection of invalid temperature/all-invalid scores. `math_boundary_green.txt` and `tests/test_math_properties.py` guard the fix. These boundary paths were not demonstrated to occur in the published/running experiment configurations; this is a numerical API defect, not evidence that all previous results are invalid.

### P2 — ESS trust is a conditional empirical bound, not an unconditional policy guarantee

For achieved ESS E, `KL(w || uniform_K) <= log(K/E)` is valid; 500 deterministic Hypothesis examples cover ordering, normalization, sample expected-cost improvement, shift/scale equivariance, monotonic ESS, feasible ESS/budget roots and the entropy inequality. With scores `[0,0,4,inf]`, target rho=.9 is unreachable: achieved ESS approaches3 of4, and KL is larger than `log(1/.9)`. The solver returns a bracket end as documented. The advertised target is therefore not itself a certificate when invalid trajectories prevent achieving it. Equal minima can also prevent reaching a low target. The finite-sample weights' KL is not automatically the state-conditional policy KL or a guarantee for the deterministic mean plan or fitted neural policy.

Guard: `test_unreachable_ess_targets_report_saturation_not_a_kl_guarantee` in `tests/test_math_hypothesis.py`. No algorithm change made to turn infeasible requests into a different method.

## Independent label reconstruction

`tests/test_core_independent_audit.py::trace_one_label` derives the point-mass dynamics from XML constants m=.3, damping1, gear.1 and dt.02, independently of the environment's finite-difference linearization. It rebuilds nominal controls, 128 clipped noisy action sequences (H12, sigma.7, seed112358), every state, quadratic running/terminal score (independent DARE), a Brent-root ESS temperature (not the repository bisection), Gibbs weights, weighted plan, first-action label, ESS, clip fraction and all EStepResult diagnostics.

Recorded values: lambda2.19422440407, ESS38.4, nominal24.3087048391, sample mean33.3988204900, weighted expectation30.1302393937, mean-plan23.0944618794, clipping.1494140625, first action[-.06117121930,.07766128975]. Maximum state error2.78e-17, score error1.71e-13, weight error3.75e-16. Full trace: `independent_label_trace.json`.

This verifies the finite-horizon E-step pipeline on a tractable model. The evaluation audit extends this exact label through `tests/test_mstep_independent_audit.py`: an independent NumPy neural forward/backward pass and first Adam update are compared with `_fit`, then serialization/reload is checked. That is a narrow end-to-end label → fit → deployment calculation, not independent verification of every Walker training label or an end-to-end learning claim.

## External MPPI differential

Executed installed upstream [`UM-ARM-Lab/pytorch_mppi`](https://github.com/UM-ARM-Lab/pytorch_mppi), version0.9.1, SHA256 of implementation `c9160a7e3b5bed9818c72c64581a8beb898c4627aa0651a8ec534826044df89a` (manifest in label JSON). With identical pre-generated noise, bounded actions, quadratic dynamics/cost, and a zero nominal, the unmodified external algorithm matches actions, all weights and the mean plan within1e-10. At a nonzero nominal the algorithms intentionally differ: upstream includes the Williams action-prior importance term, while ours prohibits `use_is_correction`. A second explicit adapter disables that external prior term and verifies the shared clipped Gibbs-update kernel at nonzero nominal. This is **not** evidence that our policy-centered method is globally identical to Williams MPPI. Raw test: `independent_label_and_reference.txt` (3 passed, including analytic trace).

## Coverage and tests

Latest full run: **525 passed, 2 skipped, 3 xfailed**, 21.06s. Requested modules combined: **81% statement+branch coverage** (full file/line results in `full_suite_coverage.txt`, machine-readable `coverage.json`). GPU coverage in that local run is only the CPU-backed orchestration regression; actual CUDA tests are separately recorded. Existing strict expected failures remain expected failures, not corrected behavior. Original subprocess failure in sandbox was rerun with IPC access; it was not counted as a product defect.

Audit dependencies were installed only in `/private/tmp/mppi-gps-audit-tools`; project runtime dependencies resolve before that directory. Bootstrap command:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 COVERAGE_FILE=/private/tmp/mppi-gps-core-audit.coverage .venv/bin/python -c "import site,runpy; site.addsitedir('/private/tmp/mppi-gps-audit-tools'); runpy.run_module('pytest',run_name='__main__')" -p hypothesis.extra.pytestplugin -p pytest_cov --cov=src.gps --cov=src.mppi --cov=src.utils.math --cov=scripts.training.train_walker_mdgps --cov-branch --cov-report=json:docs/audit_evidence/coverage.json --cov-report=term-missing -q
```

The property-based suite executes 500 generated examples plus one saturation counterexample. It skips explicitly if optional Hypothesis is absent; a normal minimal environment cannot be credited with executing those examples. External-reference tests likewise skip without `pytorch-mppi`.

## GPU profile

Original profile:594 `cudaDeviceSynchronize` calls consume82.76% self-CPU time (4.563s);592 graph launches;1826 async memory copies. These operation counts show repeated synchronization and host/device transfers in the current implementation. The original profiled context took22.69s and the fixed context18.59s including profiler overhead under changing contention; their difference is **not a measured speedup**. Torch-reported peak21,667,840bytes excludes Warp allocations and cannot be reported as total GPU memory. Tables are in `gpu_profile_table.txt` and `gpu_fixed/gpu_profile_table.txt`. Large raw Chrome traces remain on server at `/tmp/mppi-gps-gpu-audit-20261023/gpu_profile_trace.json` and `/tmp/mppi-gps-gpu-audit-fixed-20261023/gpu_profile_trace.json` (~542MB each), rather than bloating the git workspace.

## Mutation campaign

Completed **3393 generated mutants: 2390 killed, 648 survived, 354 uncovered, 1 segfault, 0 timeouts and 0 pending**. The one crashing mutant terminated its isolated process with signal 11 and is reported separately, not counted as killed. It replaces the valid MuJoCo data pool with None (`mutation_segfault.diff`); this injected fault is not present in the original source. The kill fraction using **all generated mutants** is 2390/3393 = 70.44%. Restricting the denominator to 3039 exercised mutants gives 78.64%, but that excludes 354 uncovered mutants and must never be reported without that qualification. These figures are not adjusted for equivalent mutants.

| File | Generated | Killed | Survived | Uncovered | Segfault |
|---|---:|---:|---:|---:|---:|
| src/gps/averaging.py | 60 | 39 | 21 | 0 | 0 |
| src/gps/gpu_estep.py | 425 | 64 | 24 | 337 | 0 |
| src/gps/mdgps.py | 148 | 135 | 13 | 0 | 0 |
| src/gps/policy_tail.py | 94 | 74 | 19 | 0 | 1 |
| src/mppi/gate.py | 316 | 218 | 97 | 1 | 0 |
| src/mppi/gate_v2.py | 394 | 279 | 115 | 0 | 0 |
| src/mppi/mppi.py | 788 | 505 | 267 | 16 | 0 |
| src/utils/math.py | 152 | 120 | 32 | 0 | 0 |
| scripts/training/train_walker_mdgps.py | 1016 | 956 | 60 | 0 | 0 |

Machine-readable full mutant IDs/statuses and per-file counts are `mutation_results.json` and `mutation_summary.json`. Source and test snapshot hashes are `mutation_subject_manifest.json` and `mutation_test_snapshot.json`; config is `mutation_config.toml`. No generated mutant was left pending. The 354 uncovered mutants were generated but not executed; 337 are in the GPU module, whose CUDA paths were unavailable in this Mac campaign. The remaining 17 uncovered mutants are in MPPI and gate code.

Initial mutmut3.8 baseline failures and the `src` package incompatibility are documented in `mutation_src_layout_blocker.txt`. Mutmut assumes `src` is a source root rather than the actual Python package used by this repository. A temporary copy renames only the package/import prefix to `audit_subject` so mutations can be associated with executed functions; source semantics and baseline test behavior are otherwise preserved. No such rename was applied to the repository. All requested source files were selected. Raw campaign output: `mutation_run.txt`.

### Representative survivor triage

The mutants below are injected faults, **not proof that the original source contains those faults**. Their survival establishes a test gap unless the transformation is equivalent on the stated contract. Exact diffs: `mutation_survivor_examples.diff`.

- `mdgps.e_step__mutmut_20` deletes the warmstart keyword passed into `planner.plan_step`: genuine gap. Existing exact nominal replay tests do not independently assert that the E-step passes the same solver state into sampled trajectories. Add a solver-sensitive sample-score/label differential for nonzero warmstart.
- GPU `_cost__mutmut_39` removes the per-world reduction axis; `_47` changes torso-angle state index3 to4; `_76` divides instead of multiplies the absorbing penalty by tail length: genuine gaps. A mixed batch of healthy and fallen samples with a recovered endpoint is needed. The short real-CUDA comparison and the controlled warmstart regression do not cover this.
- Trainer `_collect__mutmut_45` replaces the DAgger planner configuration with `None`: genuine gap in that branch's direct integration tests. The MDGPS-heavy suite cannot certify the DAgger baseline path merely from its line coverage.
- GPU `_cost__mutmut_17` removes explicit sampled actions. It survived the initial CPU-backed-world test because its substitute used the same constant policy as the actions; the strengthened regression now uses the actual `_Worlds.rollout`, and this particular injected fault raises an error. The full mutation campaign uses the earlier test snapshot, so its original score is retained rather than retroactively improved.
- `math.softmin_weights__mutmut_23` changes all finite unnormalized weights1 to2 at infinite temperature: mathematically equivalent after normalization; a surviving mutant is correct here.
- `mdgps.e_step__mutmut_8` always passes `initial_warmstart=None` instead of omitting that keyword when None: equivalent for the supported simulator API.
- Several sensor-array mutations are equivalent for Walker, which has no sensors and ignores that argument. Error-message alterations are not mathematical failures.
- Trainer `_collect__mutmut_19` uppercases the config path; it survives on the Mac's case-insensitive filesystem. This is platform-specific coverage weakness, not a valid Linux-compatible path. It illustrates why local mutation results cannot substitute for server tests.

The final M-step independent reconstruction and strengthened actual `_Worlds` regression were added after the mutation campaign snapshot. They are in the final coverage suite, but no retroactive mutation-score credit is taken for them.
