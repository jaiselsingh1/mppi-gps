# R0.1 — proposed origin-host replay diagnostic

**Proposed; not frozen or launched.** This does not authorize restarting P1.
Its purpose is to distinguish replayability on the source platform from the
cross-host divergence observed in the [preserved P1 stop](trajectory_pi_p1_result.md).

Use only P1's existing teacher slot 0, exact recorded float64 actions and full
initial context. Bind the original H4 archive/runtime records, XML, relevant
source functions, current native binary and compiled-model/options metadata.
Perform one deterministic open-loop replay on the original macOS ARM platform
using its recorded MuJoCo 3.6.0 environment. The already paid Linux P0 trace is
the comparator; do not repeat it. No policy checkpoint, neural inference,
optimizer, teacher acquisition or fitting is needed.

Before launch, freeze the exact source/environment bindings and a create-only
output directory. The proposed finite cap is 600 wall seconds, 1,000 native
controls, 4,000 physics substeps, one environment construction, and 64 MiB of new
output. Model controls, PI passes, neural rows, gradients and Adam updates are
zero. Preserve all N+1 states, warmstarts, integration contexts, actions and costs.

Require initial restoration to match exact bytes. Compare all recorded states,
warmstarts, integration contexts and native costs with `rtol=0, atol=1e-10`;
keep the original P1 tolerance. Also record first error/threshold crossings,
walking, survival and complete cost separately. Stop on any missing binding,
failed restoration, live numerical failure, fall, budget or evidence error.
There is one attempt, no source replacement and no tolerance search.

If origin-host replay passes while the preserved Linux trace diverges, the
platform/runtime distinction is supported as a practical replayability issue;
it does not identify a unique floating-point operation or prove every native
option identical. If origin-host replay also fails, missing context, source or
native-model provenance remains an active explanation. Either result is useful.
Independent saved-array/accounting review must close the diagnostic.

No outcome automatically restarts the similarity comparison. A subsequent
version would need an explicitly reviewed, replay-eligible input/runtime binding
while preserving P1.0's failed prerequisite and unchanged scientific hypothesis.
