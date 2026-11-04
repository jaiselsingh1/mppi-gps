# P1.1 — worker-native baseline revalidation

This prerequisite follows the audited strict **R0.1 origin replay pass**. It is
separately frozen before execution; the
[RV configuration](../configs/studies/trajectory_pi_native_baselines_p1_1.json)
binds its actual source, inputs, runtime and finite limits. A failed check stops
P1.1. The original [P1.0 failure](trajectory_pi_p1_result.md) remains unchanged.

The purpose is to establish reproducible native trajectories on `robopil5090`
for the **same two original complete control sequences and initial contexts**.
This is deterministic replay and validation, not teacher search or policy
training. No original control sequence is replaced, optimized or resampled.

## Bindings and chronology

Before the first control, bind the frozen P1.1 and RV configurations, source
export, original H4 recordings and P1.0 prepared manifest. Retain all original
arrays. Bind the R0.1 result, independent audit receipt/findings and external
summary. R0.1 must retain `ORIGIN_REPLAY_PASS`, an independently passed audit and
its original source/runtime seals. Its already paid Linux comparator is reused,
not rerun as an R0 diagnostic.

Use the declared Linux CPU runtime, task dictionary, unchanged model XML and
native execution functions. Construct one environment and preserve one compiled
MJB, its checksum, native binary identities and compiled-model/options metadata.
Record zero external forces and the supported Walker dimensions9/9/6; use the
actual `mjSTATE_INTEGRATION` enum and its91-element integration vector.

Process slots0 and1 in their fixed original order:

1. **Capture:** restore that slot's original complete initial context byte
   exactly and execute its original1,000 float64 controls once. Preserve every
   N+1 boundary, warmstart, integration context, observation, control, native cost,
   terminal flag and outcome. This defines the worker-native baseline.
2. **Repeat:** restore the identical original initial context and execute the
   same1,000 controls a second time. Compare the repeat against the capture;
   neither trajectory is compared to the old macOS path using an equality gate.
3. **Forecast:** restore that same context and perform one K1/H1000 native batch
   forecast of the same controls. Compare its returned post-action states and
   complete cost with the capture. Check that forecasting leaves the live full
   context byte-identical. This is not a PI iteration.

The original float64 controls and initial contexts must match exact bytes in
all three operations. Both captures must complete1,000 controls, remain healthy,
and satisfy the unchanged walking criterion: mean absolute forward-velocity
error at most.3 around1.5m/s. A fall or invalid live value stops immediately.
There is no replacement slot, retry, alternative source or tolerance search.

## Exact comparisons and outputs

Capture versus repeat uses `rtol=0, atol=1e-10` for all1,001 physical states,
warmstarts and integration contexts, all saved observations and all1,000 native
costs. Controls and initial restoration remain byte-exact. Preserve separately
stored terminal boundaries; do not shift pre-action observations by one control.

The K1 forecast uses the existing batch/native contracts: all1,000 post-action
states at `rtol=0, atol=1e-10`, and aggregate complete cost at
`rtol=0, atol=1e-9`. The existing batch interface returns neither warmstarts nor
integration arrays; this comparison must not claim to verify unavailable
outputs. Derive pre-action observations from the exact initial state and the
preceding post-action states. Preserve raw model outputs and compared controls.

Report original macOS versus worker-native costs and state differences as
cross-platform diagnostics. The immutable **worker-native capture cost** becomes
that slot's initial cost reference for the unchanged P1.1 admission and late-cost
ratio gates. This reference change is explicit; original costs are never
rewritten or claimed equal.

The `trajectory-pi-native-validation-1` proof must have `status=PASS` and bind
`config_sha256` to the RV configuration, `p1_config_sha256` to P1.1, and the
source-manifest SHA. It retains the original prepared/config bindings,
`origin_replay` evidence, `native_audit` receipt/findings bindings, each slot's
raw evidence, comparisons, competence gates and complete paid-work receipt.
An independent saved-only audit recomputes comparisons, outcomes, original
context/control identities, provenance and accounting before preparation accepts
these baselines. It performs no new physics or learning.

## Frozen work ceiling

| Work | Maximum |
| --- | ---: |
| Capture native controls |2,000 |
| Repeat native controls |2,000 |
| Total native controls |4,000 |
| Two K1/H1000 model forecasts |2,000 model controls |
| Physics substeps |24,000 |
| Environment constructions / compiled MJB saves |1 /1 |
| Complete restores / external `mj_forward` calls |6 /6 |
| Model queries / PI passes |2 /0 |
| Neural rows / gradients / Adam updates |0 /0 /0 |
| Wall / new output |600 seconds /64 MiB |

All calls reserve work before entry. Retain paid prefixes, failed attempts and
unknown partial work. These phase limits fit inside P1.1's unchanged total
project envelope; they are not extra compute outside it.

After this proof passes and is audited, prepare two versioned native NPZ files:
N+1 states/warmstarts/integrations, N controls/costs and N **pre-action**
observations. Bind them separately from the original macOS files. Copy all nine
existing parent stream files unchanged. The subsequent all-eight-check P0 then
executes both prepared control sequences again: **2,000 additional native
controls**, explicitly budgeted to validate the exact serialized inputs entering
the study. This third execution is not hidden or replaced by reusing the RV
receipt. Remaining P0 checks and fresh capacity must also pass before the
[unchanged bounded alternating comparison](trajectory_pi_p1_1_protocol.md).
