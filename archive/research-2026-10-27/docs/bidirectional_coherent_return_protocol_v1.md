# Coherent whole-policy return learning: finite learner test

This new development version follows the negative early-context v5 test. Both v5 learners acquired deterministic endings, but long-search states demanded stronger recovery and produced a more costly ordinary deterministic controller. That was a valid-target, deployment/objective tradeoff; neither invalid returns nor absent gradients explains it. The new test changes the actor update mechanism and makes its two-controller objective explicit. It does not retry v5 or claim a completed bidirectional loop.

## Learned object and measured targets

Start from the same exact unsolved bootstrap, SHA256 `e415ae406f8d1db5cd829f64b042da9abbfa1fd7710bfe46dfd7e64832322c83`, including its paid history. Preserve the bounded 128×128 actor and freeze its feature layers. Search only a fixed six-dimensional output subspace: output bias, four seeded orthonormal directions in its128 output weights, and log standard deviation. Generate the weight basis from PCG64 seed1701 and sign-canonical QR; it is independent of policy outcomes, discovered actions and acquired states. A coordinate changes head/bias by .10 or log standard deviation by .50. These scales cannot adapt after outcomes.

Each candidate is a complete named action-policy version. Run it to the true terminal under both deterministic mean control and stochastic behavior, separately, from eight ordinary starts and sixteen acquired starts. The acquired contexts are exactly boundary6 of the eight actual short and eight actual long discovery episodes from independently audited v5. They are state-only interventions. No planner/teacher action, old actor return or hypothetical plan labels a new candidate. The historical bootstrap's2,000 auxiliary updates remain explicitly paid ancestry; no new action supervision is introduced.

The learned scores are measured complete returns of each exact candidate/controller/context. They are not a fitted value function, gradient estimator, planner-replanning return or updated actor's relabeled old return. Deterministic and stochastic controller identities and outcomes stay separate. Their explicitly defined training score averages their costs with weights .5/.5. Ordinary scores divide by25 and acquired suffix scores by19; the reset-distribution objective is disclosed rather than called ordinary deployment performance.

## Fixed search and strong controls

Run five matched learning chains: ordinary-only (β0), short-search contexts (β.25 or1), and long-search contexts (β.25 or1). Each starts at zero absolute coordinates around the same bootstrap. At each of three steps, construct and checkpoint13 candidate versions before queries: current center, then ±one unit along each of the six coordinates. No diagonal proposals or extra radius are added. The absolute coordinate L1 reachability limit is3 after three steps; floating-point construction always starts from the original checkpoint rather than accumulating rounding drift.

Every chain pays all24 contexts in both modes for every candidate, even if versions repeat. The selector receives ordinary data and only its named acquired source; ordinary-only receives only ordinary data. Other paid panels remain hidden from that selector. Repeated controllers/noise draws are correlated repeats, not independent replications or free cached evidence. Independent MPPI is absent from this learner assay; its old discovery work remains charged.

First restrict proposals to complete ordinary-cost and aggregate capture nonregression against the freshly queried current center in **each** controller mode. Among those, minimize (ordinary joint cost/25 + β×source joint cost/19)/(1+β), with center-first stable ties. The ordinary feasibility filter can block needed intermediate policy changes; that is an optimization failure this formulation must retain. Acquired experience can change the chosen center and therefore later policy proposals. This is why three search steps, rather than reranking one common exhaustive bank, are tested. The method performs numerical parameter optimization despite using zero Adam updates/backward calls; count candidates, selections, changed centers and stays.

## Qualification, confirmation and stopping

After exactly three steps, test the parent and five final-chain actors on16 fresh ordinary roots611000–611015, in deterministic and stochastic modes with noise root613000. Select the strongest qualified ordinary/short-data control independently; parent is its fallback. Select a long-data actor only if it beats both parent and that strongest control by at least .001 complete mean cost in each mode, with paired capture nonregression and at least90% deterministic capture. No intermediate search checkpoint may enter this roster.

If none qualifies, close this version negative. Otherwise confirm the chosen long-data actor, parent and chosen strongest control on64 untouched roots612000–612063, both modes, noise root614000. Require the same complete-cost, paired capture and deterministic90% gates. Final outcomes cannot choose another weight or actor. This is a learner/data-transfer criterion; even a pass cannot establish policy-guided search or two beneficial feedback rounds.

Freeze maxima before execution: **210,960 actual scalar controls and210,960 actor rows**, zero model controls, zero Adam/backward calls. The full schedule comprises196,560 training controls (195 queried actor versions,9,360 complete episodes),4,800 qualification controls (192 episodes), and9,600 reserved confirmation controls (384 episodes). Keep every requested/completed/failed query, actual pre/post state, raw cost component, actor/checkpoint hash, proposal/selection state, sampling RNG and inference call. Numerical search work is reported separately from gradient-optimizer work. Unused reservations cannot fund retries, a new basis or larger steps. Report wall time and storage, including audit work. Original v3/v5 lineage is charged once; other research remains paid in the global index.

Required assumptions are that the frozen representation/subspace contains useful coordinated behavior, the bounded proposals reach it, and measured acquired-state effects improve fresh ordinary behavior. Each may fail. The ordinary-only control prevents assigning a generic parameter-search benefit to model-discovered experience. This is not a matched superiority test against PPO, an efficiency experiment, a physical result, or a novelty result. Parameter-space policy learning is established: [Sehnke et al.](https://people.idsia.ch/~juergen/icann2008sehnke.pdf), [Theodorou et al.](https://proceedings.mlr.press/v9/theodorou10a.html), [Mania et al.](https://arxiv.org/abs/1803.07055). The combined protected, qualified learning-feedback loop remains the research target.

Separately, failure of this initial long-versus-short data advantage does not prohibit testing feedback from a common competent parent. The main bidirectional comparison shares initial discovery data. Any later conditional feedback test must keep this assay's verdict unchanged, requalify its common parent, use actual new raw/guided trajectories and fresh matched child outcomes, and establish two beneficial rounds. It cannot claim that the failed initial data advantage or physical discovery succeeded.

```sh
.venv/bin/python -m scripts.diagnostics.run_ending_coherent_assay \
  --early-run runs/2026-10-27_bidirectional_ending_early_context_001 \
  --config configs/bidirectional_ending_coherent_v1.json \
  --out runs/2026-10-27_bidirectional_ending_coherent_001
```
