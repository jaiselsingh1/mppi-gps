# C1.0 execution contract

The user authorized implementation and one bounded execution on 2026-11-02. The
[review protocol](student_consequence_c1_protocol.md), its
[source bindings](proposals/student_consequence_c1_sources.json), census and
storage contract remain unchanged historical documents. Their review-only
status describes that earlier checkpoint. The executable
[recipe](../configs/studies/student_consequence_c1_0.json) retains their scientific
values and adds explicit execution and accounting bindings.

Only the fitting loss changes: uniform loss, one frozen native-cost-effect
weight, and the same weight multiset shuffled within the declared strata. No
parents are refitted and no teachers are reacquired. Independent H5 sources
remain the sole new training corpus. C1 is a learner diagnostic; it does not
test adaptive teacher–student coupling.

Before any substitution or fitting, source extraction, saved prediction reuse,
normalization, checkpoint/Adam/RNG continuation, seed freshness, runtime,
capacity, durable recording and synthetic continuation checks must pass. The
paid replay prelude then restores and executes all twelve complete sources.
It must pass the original byte-exact restoration and `rtol=0, atol=1e-10`
boundary/cost comparison, plus full walking outcomes. A durable replay gate
admits the main stage only after all twelve pass. The same process owns one
environment and the cumulative six-hour worker ceiling; the twelve replays
are included in its 1,584,600-control reservation. They are never repeated.

Any failed prerequisite or infrastructure boundary stops the study, retaining
its paid prefix. Complete all 12,000 substitutions before applying the existing
constant/identical-shuffle stop gate. A failed gate does not authorize new data,
seeds, tolerances, weights, updates or a replacement run.

The final horizon-band ordering is literal: after nine ascending 100-control
blocks, consume PCG64 permutations for remaining horizons 1–10, 11–25, 26–50,
51–100, then ascending error quartiles. This resolves an ordering ambiguity
before seeing outcomes. The same at least two streams must qualify on both
primary and confirmation panels. Walking, full survival, fitting MSE and cost
remain distinct outcomes.

Native actions are promoted to float64 before both physics and native cost
calculation, matching the inherited runtime interface. `native_cost` sums
actual returned costs including the fall terminal once; `raw_cost` subtracts
that terminal. Complete cost adds only the declared five-per-missing-control
absorbing tail. No post-fall states are invented.

The existing 512 MiB preparation read ceiling includes the already paid local
seed inventory (12,530,329 bytes), a bounded selected-source extraction and a
conservative continuation/remote-metadata allowance. Preparation has a 600 s,
512 MiB RSS, 256 MiB output ceiling. Worker output remains at most 8 GiB,
independent audit at most 4 GiB, and all study-owned output at most 16 GiB.
There is one input deployment and one compact result collection, each bounded
by 128 MiB; raw arrays and checkpoints stay on the remote host.

Actual sampled indices and PCG states are durable before each training
forward. Actual training predictions are also staged before marking a returned
forward complete, within the existing 512 MiB external-file reservation.
The audit can recompute fitting residual losses without reproducing training.
Gradient norms remain finite, hash-bound worker measurements; gradients and
Adam updates are not independently rerun. Independent nontraining inference
keeps the original 522,000-row ceiling, including one-dimensional deployment
inputs. A returned call whose output cannot be persisted is reported separately
from genuinely unknown partial computation.

Commands use the existing public dispatcher:

```sh
python scripts/train_mppi_gps.py study student-consequence --help
python scripts/train_mppi_gps.py study student-consequence freeze --help
python scripts/train_mppi_gps.py study student-consequence launch --run /absolute/frozen/run
python scripts/train_mppi_gps.py study student-consequence audit --run /absolute/frozen/run --out /absolute/fresh/audit
python scripts/train_mppi_gps.py study student-consequence report --run /absolute/frozen/run --out /absolute/fresh/report
python scripts/train_mppi_gps.py inspect --run /absolute/frozen/run --verify
```

Run and audit entrypoints require their bounded coordinator ownership. Debug
current code with synthetic fixtures; set native breakpoints at
`NativeConsequence.trace`, `same_replay`, and `ConsequenceMeter.measure`.
Inspect a frozen worker from its own `source/` tree. Use a fresh directory for
every software fixture; never restart a retained paid prefix. Audit receipts
separate evidence integrity, completed scientific contrast and confirmed
walking benefit. Publication follows that audit. No outcome launches a
follow-up automatically.
