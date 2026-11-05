# Acrobot probe: accounting and preserved failures

The [scientific result](acrobot_energy_discovery_result.md) uses ten complete
episodes: four near upright and two hanging starts for each of three recipes.
The [machine-readable accounting](../results/acrobot_energy_discovery_001/complete_accounting.json)
separates canonical outcomes, all paid execution, prerequisites, transfers,
saved review and publication charges.

| Paid work | Count |
|---|---:|
| Executed native controls | 6,029 |
| Verification/control-equivalent evaluations | 108 |
| Forecast controls | 771,788,800 |
| Native model queries | 6,052 |
| Physics substeps | 1,543,589,874 |
| Neural forwards / gradients / optimizer updates | 0 / 0 / 0 |

The native workers consumed **8,163.79 CPU seconds**, including process setup and
finalization. Energy execution took **34m30s wall time**; the two near-upright
attempts ran concurrently with it. The conservative total charges all reserved
support allocations—120 CPU seconds for prerequisites/tests, 180 for saved
review/rendering and 60 for engineering/transfers/publication—giving
**8,523.79 CPU seconds**, below the frozen **11,160-second** shared cap. These
charges include unmeasured small support operations; they are not claimed as
measured use. Detailed nested timers are not added twice.

## Storage-only completion

The original near-upright recorder stopped before its next write would exceed
128 MiB. It retained two complete failures and 429 controls of start 2. Its
`STOPPED_RESOURCES` status and exact output remain unchanged. There was no unknown
partial physics work; one paid query's diagnostic bank is incomplete.

The separately committed [1.1R protocol](acrobot_energy_completion_protocol.md)
retained complete starts 0/1 and ran only unfinished starts 2/3 from identical
states and seeds using the original `05fe64d` worker. It compared the repeated
429-control prefix before start 3: states, warmstarts, integration context,
actions, costs and pre-action observations all matched exactly. The completion
used 443.31 internally measured CPU seconds against its 635-second cap.

The repeated attempt adds 429 executed controls, 432 queries and **33,178,458
physics substeps**. The original **1,510,411,416** physics ceiling remains in the
historical protocol; the explicit pre-execution amendment raised it to
**1,543,589,874**. Shared CPU and **2 GiB** output ceilings stayed fixed. Raw
transfers streamed without retaining a third archive; local and remote copies
are indexed separately. No scientific recipe, tolerance or success test changed.

## Saved-review correction

Attempt 001 stopped on whole-JSON equality of `inputs.json`: the original CLI
stored a relative configuration path and the completion stored its exact native
absolute spelling. All content hashes, source bindings and environment fields
matched. The failed attempt, 3.308214 child CPU seconds, and its source remain
preserved.

Reviewer `0f748a6` accepts only those two exact bound spellings and still requires
every other input field to match. Wrong roots, names and hashes are rejected by
regression tests. Attempt 002 passed with no repeated physics. Its full child
CPU census includes plotting/font-cache work; the shorter in-process timer is
nested within it. The [capsule](../results/acrobot_energy_discovery_001/README.md)
preserves both attempts and exact reviewer versions.

The first visualization is also preserved. A format-only second dashboard moves
legends outside the data panels so the deepest near-upright height excursion is
visible. No traces or measurements changed; the four motion GIFs use saved
states, with no physics replay. Rendering costs and source hashes are recorded.
