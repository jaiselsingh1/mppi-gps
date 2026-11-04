# R0.1 — authorized origin-host replay

This freezes the [reviewed proposal](trajectory_pi_replay_r0_1_proposal.md).
P1.0, its failed prerequisite, source, results and audit remain unchanged.

Run one existing teacher-slot-0 control sequence on the original ARM platform.
The [configuration](../configs/studies/trajectory_pi_replay_r0_1.json) binds the
original H4 NPZ, original launch receipt, already-paid Linux H5, exact task and
create-only output root. Source and native interpreter/MuJoCo/NumPy binaries are
sealed before launch. One constructed model is saved as MJB, with options and
hashes durably bound before the first control. Historical native-binary hashes
were not recorded, so current binary binding is not proof of historical identity.

Limits: one attempt, 1,000 native controls, 4,000 physics substeps, 600 external
wall seconds and 64 MiB total output including frozen inputs/source. Model
controls, PI passes, neural rows, gradients and Adam updates are zero. There is
one environment construction, one restore/external forward and one model save.
The external supervisor reserves at most 540 seconds for replay and 60 seconds
for its independent saved-evidence audit within the same 600-second envelope.
The already-paid Linux trace is read only. No Linux replay or teacher search runs.

Initial state/warmstart/integration and controls must match exact bytes. All
N+1 states, warmstarts, integration contexts and observations, and N costs, must
match the original at `rtol=0, atol=1e-10`. The terminal observation is derived
from its separately saved terminal physical state using the unchanged feature
map. Walking requires all 1,000 controls healthy and velocity MAE at most .3
around 1.5; report it separately from numerical equality.

Any missing binding, failed restoration, live numerical error, fall, strict
comparison failure, budget exhaustion or evidence failure stops. Preserve all
paid prefixes and audit saved arrays/bindings/accounting independently. Source
failure is retained even when the saved-evidence audit closes successfully.

An origin-host pass combined with the existing Linux divergence supports a
practical platform/runtime replayability distinction. It identifies no unique
numerical cause. If origin replay also fails, stop: restoration/provenance remain
unresolved. No further diagnostic launches automatically.

The user's conditional P1.1 authorization is separate. Only an audited R0 pass
permits freezing worker-native baseline capture/replay/candidate checks with
their own finite budget. The scientific optimizer, alpha, fitting procedure,
parents, seeds and planned alternating rounds must remain unchanged.
