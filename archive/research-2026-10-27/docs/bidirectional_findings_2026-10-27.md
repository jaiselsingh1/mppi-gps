# Bidirectional MPPI–policy findings, 27 October 2026

**Latest focused work:** [action-label-free return curricula](bidirectional_return_curriculum_findings.md) completed three separately frozen two-round schedules. All raw audits PASSed. Fresh standalone actors learned Ending capture64/64 without teacher action labels, but the strongest supported actor supplied no better candidate than strong independent search across2,400 complete forecasts. No handover executed and no matched next-student benefit was established. These newer negatives and costs are additive; the historical account below remains unchanged.

**Version note:** the initial account below preserves the v0–v2 development
snapshot. The latest v3 addendum at the end reports a stronger controller
result, followed by a failed matched-child learning test. All overall pilot
verdicts remain negative; the earlier33/64 result is no longer the strongest
Ending controller observation.

The implementation has demonstrated part of the proposed relationship on the cheap Ending task: independent search acquired successful complete experience; a learned neural actor eventually executed the ending itself; and earned policy agreement changed subsequent replanning and improved some complete executed outcomes. **Every overall pilot verdict remains negative.** The strongest guided controller captured **33/64** fresh endings versus **0/64** for independent search with the same trajectory memory, below the declared 90% criterion. Learning from that guided experience in a subsequent generation was not tested. The real Acrobot prerequisite screen also failed, so no full-task ordinary-reset Acrobot learning campaign or Walker run of this new loop followed.

These are development mechanism tests. Each execution has source/config hashes, but the original v0 configuration was a candidate development configuration, not a completed frozen scientific comparison. The later v1/v2 changes were explicit new versions made after observing earlier failures. Their outcomes cannot be pooled as independent replications or presented as a single predeclared success. A matched three-seed comparison, learned-value MPC control, broad standalone-control result, novelty claim, and all-cost efficiency claim are absent.

The executable mechanism uses a 128×128 family-conditioned stochastic action policy. Successful **executed** independent MPPI trajectories supply auxiliary action supervision and reset contexts. The actor's long-term targets separately come from its own complete stochastic continuations: raw cost-to-go, including the actual ending charge, with a leave-one-out baseline over four draws from the same starting context. A clipped policy-gradient objective and the auxiliary objective propose updates. Actor-only deterministic and stochastic qualification panels test proposed behavior; they are distinct controllers. No critic supplies imagined successful endings.

The reverse coupling preserves the independent planner's proposals, private RNG, nominal sequence, and fixed trajectory memory. A separate selector adds supported action-agreement costs; a new weighted mean receives its own model query. Complete actor suffix handoffs establish eligibility, and paired **executed whole replanning** episodes qualify influence. Unsupported/version-invalid/revoked influence is zero. The Ending adapter has one ending family; it does not test multimodal gait separation. Its radius-3 support spans essentially the bounded observation domain from endpoint handoffs, so intermediate-state generalization remains an assumption, constrained by fresh controller tests rather than a formal certificate. [Frozen v2 implementation](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v2_001/source/src/gps/bidirectional/ending_pilot.py), [configuration](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v2_001/config.json), [query ledger](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v2_001/evidence/pilot/ledger.jsonl), [global work ledger](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v2_001/work_ledger.jsonl).

| Execution | Observed result | Paid scalar controls / queries | Actor rows / Adam steps | Runner wall |
|---|---|---:|---:|---:|
| [v0 first execution](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_001/pilot_result.json) | Outcome-construction error after the first actor episode; retained incomplete | 44,425 / 625 | 400,025 / 2,000 | 3.467360 s |
| [v0 corrected execution](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_002/pilot_result.json) | 8/8 teachers successful; actor 0/16 after two rejected MC proposals | 50,688 / 6,888 | 425,728 / 2,040 | 9.651508 s |
| [v1 provisional training](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v1_001/pilot_result.json) | Actor 16/16; agreement weight 0.1 failed controller qualification | 200,592 / 9,896 | 501,136 / 2,040 | 19.130739 s |
| [v2 candidate qualification](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v2_001/pilot_result.json) | Actor 16/16; selected weight 10; fresh guided 33/64; overall FAIL | 1,108,030 / 26,846 | 952,336 / 2,040 | 84.325877 s |

The four executions cost **1,403,735 scalar controls, 2,279,225 actor rows, and 8,120 Adam steps**, with six separately metered diagnostic backward passes and 116.575483 s runner wall. Standard backward calls equal Adam calls here. Global and attached learner meters observe the same work and are not added. These are the listed Ending executions, not an exhaustive project-wide total. Qualification, rejected proposals, verification, evaluation, and the first failed execution remain charged; none becomes free because its result is negative.

V0 repeatedly discarded a proposal that failed immediate deployment qualification. V1 instead retained unpromoted updates in a separate training chain while keeping deployment qualification separate. Its second proposal and final actor captured all 16 actor-qualification endings; the final mean complete cost was **1.957809**, versus **20.922160** for v0. Both PG and auxiliary supervision were active, so this does not isolate which gradient component caused recovery. V2 repeated the same seeded actor training; it is not an independent confirmation of actor learning.

V2's coefficient qualification measured 0/16 captures at 0.1, 1/16 at 1, and 11/16 at 10 on their qualification panels. Fresh evaluation of the selected coefficient produced 33/64 captures, **9.010106** mean complete-cost improvement, and **1,438 changed controls**. Revocation restored the baseline's actions and cost in the recorded sentinel. Thus the executed-controller effect is measured, while reliable transfer failed. The 31 remaining failures also matter: a successful actor does not automatically make action agreement adequate long-term planner guidance. No claim attributes the gain specifically to early preparation, proves a curriculum benefit, or shows that a hypothetical plan improvement equals executed improvement.

A later [Ending saved-bank diagnosis](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_ending_bank_diagnosis_001/output_001/diagnosis.json) narrows this failure. Every one of the **256 guided gate-reaching banks** contained a good-lane prefix, yet **104 selected prefixes forecast the bad lane**. Of these, **87** already had a bad-lane best augmented sample: finite-prefix cost plus soft action agreement ranked it first. In the other **17**, a good-lane best augmented sample lost to a bad-lane synthesized mean with lower augmented score. That mean was actually queried, so this is a measured scoring/synthesis limitation, not a missing mean verification. Across the **31 failed roots, 20** had an earlier selected good-lane forecast that was subsequently lost under replanning. No committed-suffix counterfactual was executed; forecast reversal does not prove that commitment would repair control. All near misses remain failures under the original threshold. [Detailed interpretation](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_ending_bank_diagnosis_001/interpretation_001.json).

This analysis used saved predictions and queried prefixes, not new actor trajectories or heldout training labels. It examined 76,032 candidate prefixes and **304,128 analytic state/control cost rows**, charged separately as postprocessing rather than simulator or neural work. The diagnosis/interpretation wall times were **5.798874 / 0.150283 s**, with zero additional model queries, actor inference rows, or optimizer calls; a [source-bound counter receipt](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_work_index_001/revision_004/postprocessing_ending_001.json) preserves those explicit zeros separately from analytic row counts. Its pattern differs from Acrobot's sampled proposal-coverage failure: Ending had useful queried prefixes available, but soft scoring, synthesis and subsequent replanning did not preserve enough of their long-term intent.

[Qualification curves, PNG](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_001/qualification_curves_001.png) · [PDF](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_001/qualification_curves_001.pdf) · [raw figure inputs/hashes](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_001/figure_evidence_001.json). Curves use recorded complete actor costs, not invented rewards. Candidate points are separately tested proposals, not necessarily deployed policies; stage root sets differ. Figure processing used zero simulator controls, actor rows, or optimizer steps; recorded processing time is 0.929230 s excluding library import/startup.

Independent NumPy/stdlib audits check raw transitions, cost components, ending flags, query/episode linkage, checkpoint identities, complete own-return/LOO summaries, work totals, and frozen source hashes. The corrected v0 audit is [PASS, 8.194434 s](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_002/audit_004.json); v1 is [PASS, 11.719494 s](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v1_001/audit_004.json); v2 is [PASS, 34.822391 s](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v2_001/audit_003.json). PASS means arithmetic validity, not thesis success. The first v0 execution is [PENDING](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_001/audit_002.json): its complete paid 25-step raw actor episode lacks a closed Outcome. Earlier failed audit outputs remain unchanged. Audits perform no actor forward or simulator call; neural outputs/log densities are source/checkpoint/chronology bound rather than independently recomputed.

For real Acrobot, the completed development screen tested ordinary resets at **H=48, 96, 144, 288**, two episodes each. Every horizon produced **0/2 local competence and 0/2 capture**. All **452,719,936** requested controls completed across **19,200 queries**, with zero unknown work. Archive, stage, and launch-to-end wall were **5,153.882476 / 5,153.890901 / 5,155.762932 s**, respectively. The desired locally competent/ending-failing horizon contrast was therefore not obtained. The [closed summary](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_qualification_summary_001/qualification.json), [copied raw query ledger](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_qualification_summary_001/query_ledger.jsonl), and [remote identities/clocks](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_qualification_summary_001/remote_provenance.json) preserve this negative. The complete remote [independent raw audit passed](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_qualification_summary_001/audit_001.json), taking **942.070322 s** and zero additional controls/inference/fits. This is a discovery/control prerequisite failure, not evidence against or for an unrun global learner.

A fixed posthoc diagnosis read **64 already-paid banks**, each with all 560 proposals and its separately verified mean: **35,904 hypothetical candidate rows**. No sampled bank or mean approached the local threshold; the closest predicted tip distance was **1.548953** at H288, versus the 0.5 threshold. This supports a coverage failure within those sampled proposals, rather than demonstrating rejection of an already successful sampled candidate. It does not prove that every unexamined proposal failed, certify a new executed continuation, or identify which untested search change would help. [Bound specification and bank diagnosis](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_acrobot_bank_diagnosis_001/output_001/bank_diagnosis.json). This analysis took 7.319022 s, with zero new controls/inference/fits. [Eight actual trajectory descriptions](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_qualification_trajectory_diagnosis_001/trajectory_diagnosis.json), [trajectory curves](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_qualification_trajectory_diagnosis_001/actual_trajectories_by_horizon.png), and [H288 playback](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_qualification_video_002/episode.gif) are saved-state descriptions, with no new controller execution.

Approximately 74 GB of authoritative full raw evidence remains on the remote host. An optional local tar copy stopped after **1,848,995,840 bytes** and an approximate **208.872646 s** transfer interval; the partial tar is retained and explicitly [marked NOT_EVIDENCE](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_qualification_transfer_001/copy_status.json). It is not substituted for the complete remote archive or the separately SHA-bound copied ledger/traces.

Backend qualification separately paid **1,033,024 model controls / 80 queries**, with no actor inference or fitting, in **14.093486 s**; its [arithmetic audit](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_backend_003/audit_002.json) passed. Three operational smokes each paid **7,930 controls / 1,050 actor rows / 10 optimizer steps** in 3.883265, 3.980080, and 3.609477 s stage wall; the continuation smoke paid **5,564 / 564 / 4** in 2.474331 s. Where recorded, smoke003/resume launch-to-end wall was 5.162472/3.838895 s. They verify plumbing and retained negative outcomes under small conditions. [Resume comparison](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_resume_comparison_001/report_001.md) found exact final actor, optimizer, private RNG, and retained-outcome agreement with uninterrupted execution. These checks do not establish ordinary full-task competence.

The later [smoke006 audit](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_operational_smoke_006/audit_001.json) and [resume002 audit](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_operational_resume_002/audit_001.json) also passed. They charged **15,154 / 10,208 controls**, **1,050 / 564 actor rows**, and **10 / 4 optimizer calls**; stage wall was **4.732258 / 3.739631 s** and launch-to-end **6.432084 / 5.100309 s**. Their separate [exact continuation comparison](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_resume_comparison_002/comparison_001.json) is MATCH. Software suite005 recorded [240 passes](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_test_suite_005/test_result.json) in 3.142787 s stage/5.029609 s launch wall, charging 44,778 scalar controls, 19,867 actor rows and 128 Adam calls. Suite004's **236 passes/4 failures** remain retained and charged. These are software/mechanism checks, not learned-control success.

The newest closed [suite007 recorded 264 passes](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_test_suite_007/test_result.json), in **2.666915 s stage / 4.154812 s launch-to-end** wall. It charged another **44,778 scalar controls, 19,867 actor rows, and 128 Adam calls**. Suite006's **262 passes/2 failures** and its same metered work remain preserved; later passing tests do not erase failed verification costs or change prior scientific scores.

The [recorded own-work index, revision004](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_work_index_001/revision_004/README.md) includes failed runs, all seven closed metered test suites, qualifications, verification, evaluations, and separate postprocessing clocks. It distinguishes physics frames, physics controls, scalar controls, actor rows, optimizer calls, and the analytic verification rows; it deduplicates copied evidence and binds the query roster to input ledger hashes. Earlier unmetered development/verification neural or optimizer work remains **unknown**, so even its recorded exact-unit totals are lower bounds for the whole development effort. No all-cost efficiency claim follows.

At this update, the physical **remote007 temporal-proposal discovery assay is active**, with **no results included here**. Its [fixed prelaunch protocol](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_remote_packet_007/protocol_prelaunch.json) caps four ordinary-reset episodes at **235,407,776 controls** and specifies zero actor learning/inference. It compares short knots with independent broad temporal blocks; task, reward, reset panel and other proposal channels remain fixed within that assay. The cap is not completed work, and the active assay is excluded from completed totals until actual ledger evidence is available. It is a separate discovery diagnosis, not a repair claim, a horizon-gap qualification, or a completed global-learning experiment. Existing negatives remain unchanged.

| Causal link | Verdict supported here |
|---|---|
| Independent acquisition | Ending successes acquired; real Acrobot prerequisite failed |
| Valid long-term credit | Complete actor-owned returns/LOO summaries audited; no prediction critic used |
| Global standalone behavior | Ending actor competent on 16 fresh roots in v1/v2; no Acrobot/Walker result |
| Guidance changes decisions | Measured in v2 while independent proposals remain protected |
| Improved executed endings | Partial v2 gain, but declared 90% criterion failed |
| Next-generation feedback, exploration, efficiency | Untested; no supporting claim |

This remains an actor-containing **GPS-inspired mechanism test**, not completed canonical GPS or a completed standalone-control pipeline. Alternating trajectory optimization and policy fitting, long-horizon learned MPC guidance, and bidirectional planner–learner feedback are established prior art; the [primary-source map](/Users/jaisel/Documents/mppi-gps/docs/bidirectional_prior_art.md) explains the closest relationships. The present evidence adds an executable, falsifiable ownership/trust/accounting protocol and a partial cheap-task mechanism observation. It establishes neither a new RL algorithm nor superiority over strong learned-value MPC. Existing runs and negative verdicts remain intact.

## Addendum: closed v3 controller gain, matched-child learning failure

The new [v3 result](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v3_001/pilot_result.json)
is **FAIL: `failed_matched_child_improvement`**. It is one development training
seed, not a three-seed scientific comparison or an independent repeat of the
earlier seeded actor-training chain. Its source/configuration were frozen as
a new explicit version; v0–v2 negatives remain unchanged. At this rendering,
the full independent arithmetic audit is **pending**. The figure builder has
separately checked frozen source/config/ledger/checkpoint/episode hashes,
complete raw cost sums, flags and query linkage; it does not replay physics
or recompute neural outputs.

V3 replaces the failed soft agreement selector in this Ending diagnostic with
an explicit complete-continuation test: simulate each of five candidate first
actions, then finish with the deterministic frozen current actor and compare
complete costs. The first-action roster includes actor-first, the independent
champion and recorded independent-bank representatives. The independent
planner still owns its proposals, nominal, RNG and trajectory memory. Every
actor continuation is additional paid model and neural work. This is a
classical rollout/policy-improvement component, not evidence that action
agreement itself was repaired or that the original finite-horizon scorer
learned long-term information. No critic or privileged witness was used.
[Frozen implementation](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v3_001/source/src/gps/bidirectional/ending_pilot.py),
[configuration](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v3_001/config.json).

After auxiliary bootstrap and two actor-owned complete-return update
proposals, deterministic qualification went0/16,0/16,16/16; a separate final
qualification again obtained16/16. These are different fresh reset panels,
not a common-root learning curve. The rejected first candidate remained in
the training chain; only the second became the deployed actor. Both PG and
auxiliary supervision were active, so the result does not isolate PG's causal
contribution. Actor-only stochastic qualification is shown separately.

| Fresh64-root executed controller panel | Mean complete cost | Captures |
| --- | ---: | ---: |
| Independent short search with memory | 20.169813028 | 0/64 |
| Frozen actor alone | 1.955603146 | 64/64 |
| Complete-continuation guided replanning | 1.840621718 | 64/64 |

Guided control reduced complete cost by18.329191310 versus independent search
and.114981427 versus actor alone, with lower costs in all64 paired roots and
1,551 changed executed controls. The same actor already captured all64;
guidance improved cost rather than capture relative to actor-only execution.
Separate16-root qualification,16-root confirmation and8-root acquisition
panels precede this executed-controller panel; they are not extra training
seeds. Revocation restored the raw baseline in the recorded sentinel.

The subsequent matched children start from the same parent, optimizer and
private RNG, use identical actor-owned MC experience/update work and100 common
auxiliary rows, and differ in100 new auxiliary rows from the raw/guided
acquisition arm, with the declared fallback rule. Fresh child validation is
the test of feedback into learning, not the preceding controller evaluation:

| Fresh64-root actor-only child validation | Deterministic cost / capture | Stochastic cost / capture |
| --- | ---: | ---: |
| Unchanged parent | 1.957268929 /64/64 | 7.918168584 /44/64 |
| Raw-experience child | 1.965054048 /64/64 | 7.925836782 /44/64 |
| Guided-experience child | 1.966077584 /64/64 | 7.926767839 /44/64 |

The guided-experience child is worse than the parent and raw-experience child
by.008808655 and.001023536 in deterministic mean cost, respectively; it loses
all64 deterministic pairs. Stochastic means also worsen by.008599255 and
.000931057. There are three tiny stochastic pairwise improvements against
the raw child, none above the declared.001 gain margin; a margin-based
`positive_pairs=0` is not a claim that every raw difference has the same sign.
The preserved [paired differences](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/paired_differences.csv)
make this distinction explicit. All three stochastic actors capture44/64,
so no capture gain rescues this gate. Better controller experience did not
produce the required better child in this tested update. This does not prove
that such experience can never help another learner.

The run stopped at that unsupported link: **no second feedback round and no
end-only pipeline holdout were consumed**. Paid work was **1,191,547 scalar
controls in190,987 completed queries,603,868 actor rows and2,080 Adam steps**,
with2,080 standard backward passes and four separately observed diagnostic
backward passes; **254.549137 s** runner wall. There were zero MuJoCo/Warp
controls, rejected queries or unknown controls. Global and attached learner
meters overlap and are not added. The2,370,686 scalar-control cap is not spent
work. No all-cost efficiency, curriculum-causality, broad reliability,
Acrobot/Walker competence or completed bidirectional pipeline claim follows.

The publication-style descriptive figures use all1,074 closed saved episode
packets and sum25,290 saved raw cost rows, without new simulation, neural
inference, fitting or tests:

- [Actor checkpoint qualifications](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/actor_checkpoint_qualifications.png)
  ([PDF](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/actor_checkpoint_qualifications.pdf)).
- [Complete executed controller comparison](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/executed_controller_comparison.png)
  ([PDF](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/executed_controller_comparison.pdf)).
- [Matched-child failure](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/matched_child_validation.png)
  ([PDF](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/matched_child_validation.pdf)).
- [Actual state/action/cumulative-cost curves over all64 roots](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/actual_control_trajectories.png)
  ([PDF](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/actual_control_trajectories.pdf)).

The [reproducible standalone script](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/build_figures_002.py)
and [figure receipt](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/output_002/figure_evidence.json)
bind all consumed raw/source inputs. Final rendering took2.940929 s; the
retained first rendering took12.912941 s, including initial font-cache setup.
The first rendering's overstrong stochastic subtitle is corrected only in
the new output002, with originals preserved. Pairwise CSV postprocessing took
.003779 s. These repeated analytic reads are postprocessing costs, not new
experience. Source manifest SHA256 is
`8438b460143acdce6418ef5bf6d9b3e47b7ea38c4209ec35993542fa8f5ce94b`;
result SHA256 is
`209eebca4b558a4dc30a6a25e2149cb4228fe08a16fee9f5359a5fbf707a9f00`.

### V3 audit closure after figure rendering

The [independent v3 arithmetic audit](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_learned_ending_v3_001/audit_001.json)
subsequently closed **PASS**, with no errors or pending items, in
**547.352469 s**. It checked all190,987 queries and1,191,547 scalar transition
rows, plus25,290 episode transition rows separately; the repeated checks are
analytic verification work, not extra experience. It verified2,600 completion
decisions,154,797 completion-model controls and2,517 changed controls across
all audited replanning panels. The fresh64-root comparison's1,551 changes are
a subset, not an alternative total. The audit added zero physics controls,
actor rows or fitting. PASS establishes arithmetic/source/linkage validity;
the scientific verdict remains **FAIL: matched-child improvement absent**.

The [additive audit binding](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_findings_v3_001/audit_binding_001.json)
binds the completed audit to both retained figure generations and the new
[fixed-root saved-trajectory GIF](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_ending_video_v3_001/output_001/root65000_actual_closed_loop.gif).
Original image footers correctly record their earlier pending-at-rendering
status and remain unchanged. The animation shows cheap scalar Ending root65000,
not a physical robot: independent/actor/guided complete costs are20.193578,
1.965067 and1.848146, with captures0/1,1/1,1/1. Rendering its26 saved state
boundaries took2.385190 s and added75 saved cost rows, zero simulator controls,
neural rows or fits. The audit SHA256 is
`c6672ee401dab14ce5dc7183d8e4efcff267ba004e6860a7abdbcbac615ca8bd`.

## Addendum: temporal acquisition, precision and interpolation all retain negatives

The physical temporal-proposal assay has now **closed**, superseding its earlier
active-at-writing status above. Both short knots and independent blocks
obtained0/2 local approaches and0/2 final captures. Mean complete costs were
1541.621314 and1543.042040, respectively. All **235,407,776 physics controls in
9,600 queries** completed, with zero unknown/error/rejected work and no actor
or optimizer work; runner wall was **3,694.512729 s**. The two conditions each
paid117,703,888 controls. This fixed proposal intervention did not repair
discovery. It supplies no locally competent/ending-failing contrast, no
successful long-lookahead feasibility witness and no ordinary Acrobot learner.
[Closed result](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_temporal_summary_001/assay_result.json),
[protocol](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_temporal_summary_001/protocol.json),
[short-knot ledger](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_temporal_summary_001/evidence/short_knots/ledger.jsonl),
[block ledger](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_temporal_summary_001/evidence/independent_blocks/ledger.jsonl).

Its complete remote [raw arithmetic audit](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_temporal_summary_001/audit_001.json)
subsequently passed in **298.515744 s**, adding zero physics/inference/fits.
The audit binds all9,600 queries and235,407,776 controls, while the full raw
arrays remain preserved remotely. Audit SHA256 is
`2301279d4489472946e966fe25cbe31c5a084c416a8bf95a2b2486ad46bcf39c`.
The [native convention/energy/phase diagnosis](/Users/jaisel/Documents/mppi-gps/docs/bidirectional_acrobot_discovery_diagnosis.md)
finds consistent action/angle conventions and a persistent nominal plan; it
does not recast the failed direct energy heuristic as validated Spong control
or prove physical infeasibility. A separate32-slot native phase representation
has been implemented for review, without executing any new physical control;
it is not a successful repair claim.

The [closed Ending precision assay](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_ending_precision_001/assay_result.json)
also failed its matched-child improvement criterion. It paid **21,888 scalar
controls,60,768 actor rows and80 Adam steps**, with21.120423 s body and24.418930 s
launch-to-end wall. Its [independent audit](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_ending_precision_001/audit_001.json)
passed in16.969929 s. Those new data are not independent rediscovery of the
reused v3 ancestry.

The following [interpolation assay](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_ending_interpolation_001/assay_result.json)
performed no fitting and retained the full frozen candidate grid. Qualification
selected guided alpha.25 and raw alpha.5; the fresh deterministic mean costs
were parent1.958114344, guided(alpha.25)1.942188129, same-alpha raw1.945330581,
and independently selected best raw(alpha.5)1.932183223. All captured64/64.
Thus the guided interpolation improves on the parent and same-alpha raw, but
loses to the stronger separately qualified raw candidate by.010004906 mean
cost. Its declared comparison remains **FAIL**, rather than claiming success
from the weaker raw comparison. Stochastic validation likewise gives64/64
captures in all four arms and worse guided than best-raw mean cost.

Interpolation paid **21,600 scalar controls and21,600 actor rows, zero model
queries and zero Adam steps**, in20.698579 s body/22.065609 s launch-to-end wall.
The [arithmetic audit](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_ending_interpolation_001/audit_001.json)
passed in16.508544 s. No pipeline end-only holdout or second feedback generation
was consumed by these diagnostics. The outcome motivates keeping the actor's
step direction, step acceptance and auxiliary information separate; it does
not establish that the guided experience has added useful learning signal.

The additive [work-index revision005](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_work_index_001/revision_005/README.md)
binds67 recorded runs,332,478 actual query-request rows and700,398 work-request
rows, including the newest closed suite012's338 passes and all retained prior
test failures. Its distinct recorded totals are **693,201,668 physics controls,
3,220,884 scalar controls,3,226,364 requested actor rows and11,876 Adam calls**.
One requested actor row remains unresolved in the historical development
record; earlier unmetered work remains unknown. The index's temporal audit
was pending when built; the additive closure above supersedes that status
without altering its snapshot. Building/verifying this index took22.663739/
2.264614 s and performed no physics, inference, fitting or tests. Index SHA256
is`8faa365d796505afce57fbf5f1242635785e07a693817131183697145d487e72`.


Addendum — own-occupancy credit falsifier and closed phase discovery assay. The complete-parent action-gradient test also closed negative. It measured full-to-ending and four-step differences from the same paid action probes at 400 actual parent-visited states, then matched full/local/null children on the same parent, old owned MC data, optimizer state, random draws and 200 unit-weight rows. No teacher action labels entered this test. At parameter fraction .5, deterministic costs were full 1.937574002, local 1.937587794 and null 1.937726787, versus parent 1.957999. Full credit's gain over null was .000152784 and over local .000013792, both below the predeclared .001. All full proposals at fraction 1 failed capture. No full-credit fraction qualified in both deterministic and stochastic modes, so its reserved fresh final panels were not run. These are qualification results, not a successful standalone update or a negative result about all long-term gradients.

This assay paid 23,600 scalar controls (10,400 model and 13,200 live), 51,960 actor rows and 60 Adam steps; body wall was 24.813 s. Its independent audit passed in 27.874 s, separately checking 37,944 saved transition rows and executing no simulator, actor or optimizer call. [All fixed-fraction curves and raw CSVs](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_ending_learning_diagnostics_001/output_002/README.md) retain the small full/local/null differences on a separate gain scale and preserve all capture failures. The prior figure revision is unchanged.

The independent persistent-phase Acrobot proposal assay completed with 0/2 local success and 0/2 capture. Its mean complete cost was 1499.119766463 versus the explicitly reused short-proposal baseline's 1541.621313862; lower cost did not meet either original competence threshold. The new condition paid 117,703,888 physics controls over 4,800 queries, with 1,525.754 s body wall and zero actor/optimizer work. The already paid baseline's 117,703,888 controls are reused, not newly executed or added twice; the whole earlier temporal study remains charged separately. The source-bound all-raw audit passed in 144.845 s and performed no new simulator/inference/fits. Physical discovery remains unsupported for these reset panels, so this study provides no physical-task global-learning or bidirectional-success result. [Closed phase result](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_phase_summary_001/assay_result.json) and [independent audit](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_phase_summary_001/audit_001.json) remain separate from this interpretation.


### Closed early-context formulation v5

The exact unsolved bootstrap was tested with no new teacher-action supervision. At nearly matched proposal cost (short slightly more), H25 independent search captured on all eight source episodes; H4 captured on none. Only actual boundary-6 states became intervention contexts. Two provisional complete-return updates per arm learned deterministic endings: both half-step actors captured on 16/16 fresh ordinary qualification starts. The long-search-data child cost 1.950693837353 versus 1.940476101216 for the independently selected short-search-data child. The required .001 guided-data advantage failed, so no feedback round or reserved 64-root actor confirmation ran. This is not standalone 64-start qualification or successful bidirectionality.

The closed run used 269,272 cheap controls (255,240 model and 14,032 actual), 41,792 actor rows, 96 Adam updates and four diagnostic gradient calls in 17.185710500 seconds. Independent arithmetic audit001 passed, including every proposal bank, return ownership, LOO targets, optimizer/RNG chronology and strongest-raw selection. It performed no model, actor or optimizer work. The version remains negative. Source, raw ledgers and checkpoints are in `runs/2026-10-27_bidirectional_ending_early_context_001`; source-bound plots are in `runs/2026-10-27_bidirectional_early_context_figures_001/output_001`.


### Closed coherent-return and native affine tests

The coherent whole-policy-return assay closed negative for discovery-data advantage. Ordinary-only, both short-search-data mixtures and long-data weight.25 produced the same final policy hash8a82a0ade429d1cf2fbad629237188feb873a02a5b4d58220ec2fffd768a0d30. All four had deterministic16/16 capture and mean cost1.973795477 on fresh qualification starts. Their stochastic deployment had15/16 capture and mean cost3.161597435. Long-data weight1 retained more noise: deterministic behavior was identical, but stochastic capture was11/16 and cost7.913157895. The parent captured0/16 deterministically and4/16 stochastically. Ending learning occurred; the tested source-data advantage did not. No reserved64-root confirmation or feedback round ran.

This test paid201,360 actual scalar controls and actor rows, zero model controls and zero Adam/backward calls, in214.467982375 seconds body wall. It measured195 candidate versions over15 policy-search iterations;14 selections changed the center and one retained it. The independent audit passed in177.932309917 seconds internally (180.228871208 seconds wrapper), adding no physics, actor or fitting work. The source-bound saved-array diagnosis found useful ending contrasts in255/1,560 deterministic and200/1,560 stochastic long-state outcomes, versus0/1,560 in each short-state mode. Many source improvements were rejected for ordinary-start regression. These facts separate available evidence from successful use of it. Raw policy-head finite responses were mostly bias-aligned on24 initial states, but had nonzero residual directions; no global incapacity claim follows. See runs/2026-10-27_bidirectional_ending_coherent_001 and runs/2026-10-27_bidirectional_ending_coherent_diagnosis_001.

The native affine-feedback reference also closed negative. All four actual800-control nominal episodes failed local approach and ending capture. Fresh defect-aware feedforward corrections were far larger than the native action bound and actual actions saturated on94–97% of controls. It paid38,448 native controls/frames in61.748708709 seconds, with zero actor or Adam work. Audit001 retained a verifier path-shadowing error; repaired audit002 passed in23.014473125 seconds wrapper without rerunning the producer or adding physics. This is evidence of failed executable correction, not proof of infeasibility. Source, query ledgers, both verifier outputs and offline reconstruction remain under runs/2026-10-27_bidirectional_native_affine_001 and runs/2026-10-27_bidirectional_native_affine_diagnosis_001.

Accountable suite024 subsequently passed652 checks. Its additional44,778 scalar controls,19,867 actor rows and128 Adam calls are separately paid software-test work; native physics was zero. This validates implementation checks, not research success. A new frozen hard-constrained native feasibility reference has a3,900,848-control maximum and a separate state-informed return-search formulation is being implemented. Neither is retrospectively assigned a positive result. No physical actor, repeated beneficial bidirectional loop, Walker transfer or all-cost efficiency advantage has been established.


### State-informed geometry and hard-constrained native closure

State-informed whole-policy return search closed with a real reliability–cost tradeoff and a failed predeclared joint gate. All four learned deterministic actors captured16/16 ordinary qualification starts. Complete deterministic costs were fixed1.976042355, ordinary1.890182169, short-state1.897981107 and long-state1.905381161. Stochastic captures were14/16,10/16,10/16 and15/16 respectively; long-state stochastic cost3.100410387 beat ordinary9.022768164. Long versus ordinary gained five stochastic captures and lost none on these paired draws, but increased deterministic cost by.015198992, mostly the unchanged greed component. The rule required improvement in both modes, so the assay remains FAIL, with no64-root final evaluation or feedback round. These16-root panels do not establish broad robustness or full standalone qualification.

The geometry run paid66,600 executed scalar controls and66,888 actor rows, with zero model controls, Adam calls or backwards, in65.813015500 seconds body wall. Its independent raw audit passed in55.409431333 seconds internally and56.125820416 seconds wrapper. A separately instrumented functional neural replay paid288 additional actor rows and0 simulator controls, exactly reproducing all saved hidden features, means and log standard deviations in.952198084 seconds. The raw auditor binds that external paid verification rather than pretending it is free arithmetic. The original geometry criterion was not changed after these outcomes. Evidence is under runs/2026-10-27_bidirectional_ending_geometry_001, runs/2026-10-27_bidirectional_geometry_feature_verification_001 and runs/2026-10-27_bidirectional_ending_geometry_diagnosis_001.

The hard-constrained native reference also closed negative. Its two4000-variable transcriptions left maximum scaled dynamics defects.004896154/.006265165; guessed final positions were near target, while actual minimum tip distances were3.808599/3.707837m. All four actual replays failed local approach and capture, so no tracker or fresh final panel ran. It paid3,036,848 native controls/frames and0 actor/Adam work in128.215505459 seconds. All316 function/Jacobian callbacks,456 BFGS updates and396 iterations are counted as numerical work separately. Audit001 failed to resolve the explicit query_archive sibling namespace; files remained present and the heterogeneous layer passed. The repaired verifier's27 pure cases passed, and audit002 passed in22.271477542 seconds internally (22.825613209 wrapper), with0 new physics or inference. Both audit outputs and both auditor source packets are retained. The diagnosis remains an optimization/executable-reference failure, not an infeasibility proof or a negative learner result.

Recorded work-index revision008 independently verifies the snapshot closed at10:50:53 UTC:815,751,348 physics controls/frames,4,333,048 scalar controls,3,860,292 requested neural rows (one historical row unresolved), and13,608 Adam calls. Physics, scalar controls, inference and optimizer calls are distinct units. The index includes all retained test failures and204 separately instrumented pure auditor checks. Build/verification took52.698113/6.714653 seconds externally, with no scientific queries. Earlier unmetered work remains unknown, so no total-efficiency claim is supported. Later native_rollout001 and pure suite026 are outside that point-in-time snapshot.

New development is explicit and separate. A control-only native shooting reference removes transcription gaps by simulating every800-control forecast under native physics; it has its own4,281,648-control cap and actual-outcome gates. A conditional deterministic-controller feedback test is being formulated using a fixed competent common parent and a strong baseline that resimulates complete stored successful continuations. It will not require the already-failed initial long-data learner advantage as a prerequisite. Its target controller and any promotion rule will be declared before new calls; it cannot relabel the old both-mode tradeoff as a pass. No two beneficial physical GPS feedback rounds, physical standalone actor, Walker transfer or all-cost sample-efficiency result has yet been established.


## Addendum004: native forecast closure and conditional teacher negatives

The native control-only rollout reference is independently audited negative (audit004 PASS/scientific FAIL). Its 3,272,048 controls,717 paid forecasts,303 fresh Jacobians and eight failed actual replays are closed; saved forecasts and replays agree exactly. The earlier namespace/completion-envelope verifier failures remain preserved. No producer rerun repaired the result. Body800.077 seconds; final audit55.220 seconds, with earlier verifier work separately paid.

Conditional mean-controller001 gave RAW complete episodic suffix memory in addition to its private MPPI bank. GUIDE cost1.756854 versus RAW1.797309 and actor1.888878, all16 captures, but only five pairs improved and11 tied. It failed the frozen75% gate and performed no child learning. Its 290,731 scalar controls/20,622 neural rows/zero Adam work took53.037 seconds; the independent raw audit passed in80.419 seconds, scientific success false.

An explicitly outcome-informed selective variant changed the teacher criterion and fresh root namespace only. It required mean gain>=.01 and no observed paired cost loss. It also failed: eight gains/seven ties/one loss (root802009, -.000903338 cost gain), despite mean gain.043447 and16/16 capture. No confirmation or child updates ran. New work290,599 scalar controls/20,469 neural rows/zero Adam,53.948 seconds. Audit pending; the loss criterion is not being loosened. This supplies a concrete test of continuation ownership against strong replanning, not a successful full feedback loop.

The separate classical collocated-PFL reference passed24 instantaneous native parity cases, then failed all eight actual nominals (four fixed gains/two ordinary roots). It spent6400 native controls in11.658 seconds, plus40 direct zero-time forwards,24 mass expansions and one CARE solve; no actor work, teacher promotion or final evaluation. Audit pending. It neither repairs MPPI acquisition nor proves infeasibility.

The bidirectional child-learning link, two accepted feedback rounds, physical standalone policy, three-seed frozen comparison and all-cost efficiency advantage remain unestablished. All earlier runs and negative verdicts remain unchanged.
