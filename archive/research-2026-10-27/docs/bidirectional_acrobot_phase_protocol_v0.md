# Prospective native phase-proposal coverage assay, v0

This is a prospective **development coverage protocol**, prepared after the
audited temporal-proposal negative and before any phase-assay physics. It
supersedes none of the existing results. The earlier
[DRAFT](/Users/jaisel/Documents/mppi-gps/docs/bidirectional_phase_proposal_protocol_DRAFT.md)
remains unchanged. Source/configuration, runtime and input hashes must be
frozen together before launch. This document does not authorize a larger
search or an actor-learning campaign.

## Evidence and baseline choice

The closed [temporal assay](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_temporal_summary_001/assay_result.json)
paid235,407,776 controls across9,600 queries and four complete800-control
episodes. Both conditions achieved0/2 local approaches and0/2 captures.
Short knots had mean complete cost1541.621313862480; blocks had1543.042039918575.
Choose the original **short-knot condition** as the paid baseline because it
has lower observed complete cost and neither condition has a capture/local
advantage. This choice is made after those aggregate results; it is not an
untouched scientific holdout or an independently replicated superiority claim.

The [complete raw arithmetic audit](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_temporal_summary_001/audit_001.json)
is PASS with no errors/pending items,298.515743915 s audit wall, and zero new
physics/inference/fits. It covers both original conditions. Its SHA256 is
`2301279d4489472946e966fe25cbe31c5a084c416a8bf95a2b2486ad46bcf39c`;
manifest/config SHAs are
`54c3300817dbab81436c75f95b49c68d461c41818abd91d6ea160992e5252323` /
`96f9fe36ccba32eaa646a00e9b64781cd132c39953a3c847f3121ba298123986`.

## Fixed intervention

The task remains native Acrobot: T800,.01 s controls, unchanged XML/RK4/motor,
native actions[-1,1] and gear2, near-hanging reset distribution, original
POST-action cost and final100-control distance<=.2/speed<=.5 capture. Local
approach still requires distance<=.5 by600. No reward, ending penalty,
actuation, model, policy or qualified-backend change is allowed.

Retain H144,K512 Gaussian slots,16 memory/fallback slots,32 broad slots, one
paid mean verification and the lower-scored verified-mean/best-bank champion
rule. Each new episode starts with its own empty memory and zero initial
nominal. Nominal retention/shift and constant tail fill are unchanged. The
default `SearchConfig`, independent search, task and backend remain unchanged;
the new subclass consumes the base sampler's **entire original random draw
stream**, then replaces only rows512:544.

Those32 rows are2 periods×2 waveform types×8 phases, in that fixed order:

```
period P = .9134159022626218 or2.356877965498118 seconds
shape = sin or square
phase j = 0,...,7; offset = 2*pi*j/8
theta(k) = 2*pi*((elapsed+k)*.01)/P + offset
u(k) = sin(theta(k)) or (+1 if sin(theta(k))>=0 else -1).
```

Absolute episode time preserves each template's phase across adjacent
planning windows. It does not force the controller to keep selecting a
template. The periods are disclosed undamped hanging-rest analytic scales
from the [saved-parameter receipt](/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_acrobot_static_diagnosis_001/analysis_001.json),
not tuned nonlinear swing-up frequencies or a feasible controller witness.
No actor, critic, energy-controller action, successful teacher or witness
trajectory enters construction. There are no unmetered proposal rollouts.
The existing queried mean may combine phases; its own measured score must
still beat the best bank row. No additional phase-conditioned mean or
commitment change is introduced in this assay.

## Fixed roots, reuse and accounting

Execute exactly two new complete phase episodes, in root-index order:

| Root index | Reset seed | Private planner seed | Paid reused baseline |
| --- | ---: | ---: | --- |
|0|110100|30000|short_knots_root0_0000|
|1|110101|30001|short_knots_root1_0000|

Do not rerun the baseline, replace a failed root, share memory across episodes,
add samples or retune after reading phase outcomes. Any error stops this
formulation; retain its requested/unknown/failed work. All actual model/live
calls go through the same durable `QueryExecutor`.

Per complete episode, the control cap is
`561*sum(r=1..800,min(144,r))+800 = 58,851,944`.
The **new phase cap is117,703,888 controls**, zero actor rows/Adam steps.
The two reused short-knot episodes already paid117,703,888; the matched
two-condition attribution is235,407,776. Reuse never makes those costs free.
The full prior temporal acquisition paid235,407,776, including the blocks
that informed this development choice. New phase plus that full prior run
is353,111,664 distinct controls. Backend qualification separately paid1,033,024;
including it gives354,144,688 distinct controls for this scoped ancestry,
before other separately indexed project work. These overlapping attributions
must not be added twice. Report actual wall/storage and any new setup/audit
work separately. The old temporal runner wall was3694.512729424 s; its
launch-to-end wall3697.636248827 s. No new phase wall is assumed in advance.

## Mandatory prelaunch identity gate

Require both old raw condition archives closed/PASS, their ledgers and episode
packets SHA-bound by that audit, and all four original outcomes complete.
Verify the exact original task/reset/planner/search recipe and runtime package
versions. Bind original independent-search/task/backend/physics/evidence
source bytes to the new frozen source; compare the executed workflow/helper
ASTs where the source-inventory function has changed. The source manifest
must include the actual phase subclass, assay, CLI and tests.

The [new CLI](/Users/jaisel/Documents/mppi-gps/scripts/diagnostics/run_independent_phase_assay.py)
requires `--baseline-run` and `--backend-qualification`; freeze/reexecute and
recheck before constructing a task. Its input packet copies compact metadata,
ledgers and episode evidence, while binding the original full raw archive
without copying its approximately40 GB of query arrays. The backend003 gate
must still cover K560/K1,H144 and the same source/runtime/GPU. Missing,
pending, mismatched or contradictory evidence prohibits execution.

## Outcomes and permissible interpretation

Report all two phase outcomes alongside both reused baseline outcomes:
actual local approach, full final capture, complete cost, first-step
model/live consistency, native actions and raw terminal events. Two of two
new actual captures is the declared development lead;0/2 retains a discovery
negative. Also report1/2 candidly, without promoting it to broad reliability.
No smaller cost or closer hypothetical endpoint alone qualifies capture.

The separate fixed raw temporal diagnosis samples all four old executed
episodes and32 prespecified banks before phase physics. It can distinguish
sampled coverage, finite-score ranking, model/live discrepancy and executed
replanning behavior; it cannot establish which unexecuted intervention works.
Its output must not silently change this finite basis or panel.

A strong complete **long independent capture witness is still missing**.
Neither the failed direct energy/LQR reference nor old position-only tuning
scores supplies one. This two-root reused comparison is not a fresh frozen
scientific comparison, a horizon-gap qualification, a successful actor/GPS
pipeline, a sample-efficiency result or a novelty claim. Stop before ordinary
actor learning if the required discovery/competence link remains unsupported.
