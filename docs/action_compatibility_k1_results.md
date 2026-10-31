# K1: MPPI trajectories versus untrained, cloned and qualified-reference policies

**The requested comparison is complete and independently audited.** All five existing cloning lineages match every successful MPPI teacher trajectory better than every untrained control. Nevertheless, none of the three non-reference policies in fresh standalone evaluation walks. The independent SAC reference walks 16/16 primary and 16/16 confirmation while disagreeing with MPPI much more than either clones or untrained controls.

This is an action-compatibility diagnostic. It does not establish that diverse trajectories are inherently unlearnable, that similarity measures teachability, or that the 128-tanh architecture is insufficient. No new policy training or coupling experiment occurred.

[Protocol](action_compatibility_k1_protocol.md) · [exact evidence capsule](../results/action_compatibility_k1/README.md) · [full trajectory matrix](../results/action_compatibility_k1/report/trajectory_policy_matrix.csv) · [generated report](../results/action_compatibility_k1/report/report.md) · [source curves](../results/action_compatibility_k1/report/source_control_curves.png) · [independent audit](../results/action_compatibility_k1/audit/receipt.json).

## The central comparison

The frozen score is **action MSE, lower means closer**:

\[
E(\pi,D)=\frac{\sum_t\|\pi(o_t)-u_t\|^2}{6N}.
\]

Every policy is evaluated on the same saved physical observations within a dataset, using its own frozen feature/normalization interface. The compact table averages trajectories equally; the untrained column averages the three predeclared random initializations. The representative clone is the preselected H4 independent-data child. The full matrix retains all nine policies and every trajectory, source, actuator and time bin.

| Dataset | Trajectories / executed rows | Mean of 3 untrained controls | Representative clone | Qualified SAC reference |
|---|---:|---:|---:|---:|
| Successful MPPI teachers |14 /14,000 |0.286433 |0.176143 |0.644604 |
| Qualified-reference primary rollouts |16 /16,000 |0.293185 |0.339820 |0.000000 |
| Executed random-action rollouts |4 /70 |0.377675 |0.492388 |0.879562 |

The reference ordering is self-log0 <MPPI 0.6446 < random 0.8796 under the unchanged metric. Its zero self-score reuses its deterministic logged predictions and is largely true by construction. This is not an independent network replay or a validation of teaching quality. The three dataset groups visit different state distributions; the four random episodes terminate after 8, 26, 22 and 14 controls. Their tiny, early-failure dataset is not a matched-distribution causal control for data quality.

On the **same MPPI observations**, a walking reference is farther from the recorded torques than even the untrained policies, while failing clones are much closer. On reference observations, untrained policies are closer than the failing clones. These facts distinguish agreement from competence; they do not make MSE an invalid measure of agreement. No score was redesigned to obtain a preferred ranking, and successful demonstrations were not rejected for disagreement.

## Fresh standalone competence

All four primary policies use the same 16 fresh native reset contexts, without planner prefix, tail or fallback. Walking requires full 1,000-control survival and velocity MAE ≤0.3 m/s around 1.5 m/s. Complete cost charges native fall terminal20 once and 5 per missing control.

| Policy | Walking | Full survival | Mean controls | Velocity MAE | Complete cost |
|---|---:|---:|---:|---:|---:|
| Independent SAC reference |16/16 |16/16 |1000.0000 |0.227554 |521.968779 |
| Incoming H4 policy |0/16 |0/16 |241.4375 |0.845919 |4276.989846 |
| H4 independent-data child |0/16 |0/16 |105.3750 |1.851854 |4879.018233 |
| Untrained_0 |0/16 |0/16 |84.1875 |2.811312 |5114.418200 |

Reference confirmation on 16 disjoint resets: **16/16 walking and survival**, mean velocity MAE 0.235726 and complete cost 536.940916. It therefore passes the frozen ≥15/16-on-both-panels **provisional** qualification. This is a finite development result, not optimality, a population guarantee or training-seed robustness.

The only candidate was the independent SAC ladder's documented final 1M checkpoint, selected by provenance before new scoring. Its17-observation identity-normalized,256-ReLU actor executes deterministic tanh(mean) torques. The task/XML/native CPU interfaces matched; its discounted SAC training reward/fall tail differs from this finite evaluation. Its architecture and learning procedure differ from the 128-tanh clone. **Its competence does not prove the current student class or cloning procedure is sufficient.** No other checkpoint was searched after observing outcomes.

Four conditional Uniform[-1,1]^6 episodes were collected only after confirmation; all fall, with mean 17.5 controls and complete cost5046.297110. All attempts and terminal states remain in raw evidence; post-fall observations were never padded.

## What reconstruction reveals

**1. Do clones reconstruct MPPI better than untrained controls?** Yes: incoming H4, its three children and L1 clipped each beat each of the three untrained controls on **all 14 individual teacher trajectories**, including trajectory-held-out sources. They also beat their frozen within-trajectory permuted labels on all 14. This is genuine observation-conditioned action structure, not merely comparing whole paths or low output magnitude.

It is still incomplete reconstruction: fitted teacher MSE is roughly 0.10–0.17, per-scalar torque RMS error about 0.32–0.42. Own-H4-arm first 100-control MSE is 0.112/0.145/0.128 for independent/high/backoff children. Errors begin early and span actuators. The data do not justify a claim that almost-perfect cloning is being destroyed by only tiny errors.

**2. Does improvement generalize, or concentrate on fitted data?** Both partial generalization and a substantial exposure association appear:

| Comparison | MSE |
|---|---:|
| L1 on fitted later-independent trajectories |0.122041 |
| L1 on held guided trajectories sharing those reset contexts |0.234133 |
| Incoming H4 on fitted guided history |0.157112 |
| Incoming H4 on held later-independent trajectories |0.244307 |
| Independent/high/backoff children on their own H4 acquisitions |0.171506 /0.163248 /0.158160 |
| The same children on other H4 acquisition arms |approximately 0.223–0.240 |

All held sources still beat untrained controls, so “no generalization” is wrong. H4 children improve their own new-source errors by approximately 31–35%, while original-discovery errors worsen about 5%. Their retained 6096-row aggregate stays near0.126, masking opposing source changes. Their historical and new standalone outcomes still fail. Different exposure strata visit different states, and held trajectories can share training reset ancestry; these are not clean estimates of independent-reset generalization or the causal effect of mixing.

Clones have mean absolute teacher-observation torque around 0.29–0.32 versus teacher 0.44 and almost no≥0.95 saturation. The reference uses approximately 0.66 mean absolute torque and 23% saturation. Attenuation, different strategies and off-distribution responses are plausible contributors to score differences, not established failure causes.

## Ambiguity beyond whole-path diversity

**3. Is there observation-conditioned ambiguity?** There is limited **pooled, source-conditioned target multiplicity**, not demonstrated contradictory supervision causing current failure.

The label-blind geometry identifies10 exact incoming-H4 normalized-float32 input collision groups, containing 24 of14,000 teacher rows. Four groups have torque differences above 1e-6, with maximum absolute difference0.229932. At these points full 19-state, available warm start/integration, independent nominal/RNG and bank match; the guidance objective/rho differs. All branches subsequently walk. These are concrete alternative successful teacher decisions at a common input, beyond merely different whole trajectories.

**No actual cloning lineage fitted multiple teacher rows from any of these collision groups.** They cross v2 independent/guided or H4 acquisition branches; actual children fitted one branch. The pooled empirical deterministic reconstruction floor **1.111659×10⁻⁶ is not a training-loss floor for any actual child** and is far smaller than observed 0.11–0.24 errors. The96 auxiliary rows are outside this teacher-only geometry; unobserved or auxiliary ambiguity is not ruled out.

The fixed distance-matched assay finds **no excess between-trajectory disagreement** in its selected comparisons:

| Comparison | Matched between−within action MSE | Matched anchors | Distance residual mean / maximum |
|---|---:|---:|---:|
| Same reset ancestry |−0.03201 |652/1400 overall;652/1000 with a same-ancestry peer |0.00512 /0.019945 |
| Different reset ancestry |−0.02871 |1315/1400 |0.00206 /0.019682 |

Pairs satisfy the frozen 0.02 standardized-RMS caliper but remain dependent descriptive samples. There are no matched pairs below 0.1 standardized RMS; same-ancestry matches start at 0.25. Scalar distance matching does not control contact regime, displacement direction or all context. This neither proves compatibility nor identifies a hidden planner variable. **Insufficient evidence** remains to distinguish conflict, legitimate alternatives, contacts and sampling variability generally. No new same-state MPPI query was made.

## Reference disagreement and unresolved causes

**4. Does a qualified independently trained reference exist?** Yes, under the disclosed native torque interface and 32 new evaluation resets. This establishes attainable walking by that existing256-ReLU SAC actor, not optimality or 128-tanh sufficiency.

**5. What does its disagreement reveal?** Walking can use an action mapping far from MPPI's mapping at MPPI-visited states. Present-policy proximity is not a universal competence ranking and does not establish teachability. Different action magnitude, gait and state distribution can contribute. Qualification from fresh native resets does not establish recovery from those particular MPPI-visited states. Agreement with self-generated deterministic logs is a plumbing identity. No claim follows that MPPI's successful data are bad or that one particular reference action is the unique correct label.

**6. Which explanations remain consistent?** Incomplete fitting and source-specific transfer remain compatible with limited capacity, optimization/proximity/initialization effects, coverage and execution sensitivity. Lower on-data MSE coexists with failure, but residuals are not tiny. Sparse cross-corpus collisions and the matched assay do not establish that diverse paths caused cloning failure. The study performs no new update, so it does not prospectively measure what any corpus teaches after a matched update or test a new guidance mechanism. These torque-policy findings are not a failed CIO–GPS reproduction.

## Accounting, validity and missing evidence

| Work | Stage A | Stage B | Combined | Frozen ceiling |
|---|---:|---:|---:|---:|
| New policy forward rows |120,746 |167,442 |288,188 |1m per stage / 2m total |
| Live controls |0 |38,966 |38,966 |84k reserved / 100k authorized |
| Model-rollout controls |0 |0 |0 |0 |
| Native physics substeps |0 |155,864 |155,864 |336k maximum roster |
| Episodes |0 |84 |84 |84 |
| Backward rows /Adam steps |0 /0 |0 /0 |0 /0 |0 /0 |

All requested experimental work completed; no failed/unknown-partial query or retry occurred. Random episodes failed the task and were retained. Six thousand saved guide predictions and exact-input memoization reduced additional inference. All final actor tensors match frozen input checkpoints; normalization and optimizer packages were unchanged.

Stage wall times were1.009 s and 109.524 s; worker110.674 s; external launch111.760 s. Independent audit: **1,313,287 checks,279 bound files,1113 hashed arrays**,6.300 s, zero new inference/physics/training. Audit counts are assertions, not independent trials. It recomputed metrics, source alignment, geometry, native outcomes, conditionals, work census and serialized tensors; it did not replay networks, optimizer arithmetic or expensive historical audits. **63 focused synthetic/CLI/report checks passed** before launch, and 65 passed after the editorial report revision. No real roster inference or physics ran before sealing; these checks performed no training updates.

[Experimental and engineering accounting](../results/action_compatibility_k1/accounting/engineering_accounting.json) and [final publication validation and overhead](validation/diagnostic_k1_2026-10-31/README.md) separate research counters, test/report/audit overhead, storage copies and unavailable overhead. Engineering wall time before the first explicit timestamp, global synthetic-fixture arithmetic counts and model/tool billing are unknown, not zero. No efficiency advantage is claimed.

Full new evidence is **local-only** at `/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_action_compatibility_k1_001/`: sealed inputs/checkpoints, full new standalone states/integrations, ledgers and raw prediction matrices. [File manifest](../results/action_compatibility_k1/external/full_run_file_manifest.json) gives exact sizes/hashes. Original discovery terminal/warmstart/integration arrays remain unavailable locally, though audited pre-action rows and saved planner context are present. Historical full model banks are not reconstructed. Bulk evidence remains outside Git and the vault.

Protocol/source checkpoint: commit `25b3d62`; config SHA256 `167a45e2f947dfaa415267ad96c3f6eca37c06a0c810f32c5be626556bf57f43`; input-manifest SHA256 `5303607bba11f3c26f51646ba02da489bb537e64da83ab1ab6724ed7b26a2f99`; geometry SHA256 `7dc0a9f7e62964073fb21d4ab1c0786de764dd6f21b3eb6d357137325f9c89ab`. The original worker/configuration/results remain unchanged. Editorial report revisions preserve the first render and change presentation/caveats, not scores or the experiment.

## One next experiment to recommend, not launch

Recommend the already discussed **128 versus 256 tanh width contrast**, conditional on a source-bound clean executed-data benchmark. Use fresh paired seeds, identical corpus/features/normalization rule, paired minibatches, optimizer settings and fixed update/primary endpoint; assess reconstruction and standalone walking separately. Both arms must rerun; historical L1 is not a matched control. Do not simultaneously change activations, targets, horizon, proximity or coupling.

The prediction is that a specific width intervention reduces substantial within-fitted-trajectory residuals and improves execution. Better fitting without walking would leave coverage/execution sensitivity unresolved; no fitting gain would leave optimization/target structure unresolved. This is a test of one intervention, **not a conclusion that capacity caused the failure**. K1 supplies no strong reason to prioritize a generic trajectory-diversity repair. This recommendation is secondary to the completed compatibility comparison and authorizes no follow-up training or simulation.
