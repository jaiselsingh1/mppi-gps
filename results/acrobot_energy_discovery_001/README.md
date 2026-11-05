# Acrobot balance and energy-discovery probes

**Acrobot remains a teacher problem: near-upright balance was 0/4, and all three energy recipes were 0/2.** No policy was fitted and no alternating loop was run. The frozen kill rule applied. The saved-only independent audit passed 667,386 checks; exact payload hashes and [complete accounting](complete_accounting.json) accompany the result.

| Recipe | Required height holds | Best height |
|---|---:|---:|
| Prior dense controller, near upright | 0/4 | 4.000 m |
| Matched energy cost, 6 s horizon | 0/2 | 3.652 m |
| Matched energy cost, 8 s horizon | 0/2 | 3.154 m |
| Double energy cost, 8 s horizon | 0/2 | 3.195 m |

Success remains tip height at least 3.8 m at **every final-two-second control boundary**. Brief near-upright height and target energy do not count as capture. The energy probes lasted 12 s; near-upright probes lasted 10 s. Planner settings also differ between the dense and energy probes, so these results do not isolate the cost term causally.

Read [the audited report](report.md), [episode measurements](episodes.csv), [planner-pass measurements](planner_passes.csv), and [audit summary](independent_audit_summary.json). The [figure](figure.png) includes every canonical episode. Full audit files, arrays and raw journals remain outside Git with measured sizes, checksums and availability in [the evidence index](external_evidence.json).

The original near-upright run stopped before the next write would exceed its storage limit after two complete episodes and a 429-control partial third. Its exact `STOPPED_RESOURCES` result remains in `run/near-upright/`. A separately frozen storage-only completion repeated the third start and matched all 429 saved controls and complete contexts with zero error, then ran the fourth start. Canonical near-upright outcomes use original trials 0/1 plus completion trials 2/3. Paid repeated work and the unexecuted partially written bank remain explicit; neither is silently discarded or treated as a fifth outcome.

The initial independent reviewer stopped at an overly broad path-spelling check. That incomplete attempt, its launch/exit receipts and its error remain in `audit_attempt_001/`. The exact relative path and its frozen-source-root absolute spelling were then explicitly bound, with every other input field and all content hashes and numerical tolerances unchanged. Both immutable reviewer versions are retained under [audit_versions/](audit_versions/README.md). Attempt 002 passed. This repair repeated no physics, fitting or evaluation.

Worker source is `05fe64d`; completion wrapper source is `adf1cfc`; the final reviewer source is `0f748a6`. Exact compact per-part inputs, runtime bindings, configurations, results, prefix checks, launch, transfer and process receipts are retained. Only two complete candidate trajectories per bank were saved; scalar summaries cover all candidates. The saved-only audit cannot reconstruct unretained physics. Exploratory two-seed results would not have qualified the required fresh 9/10 teacher even if successful.

For the generic inspector, `run/config.json`, `run/result.json`, `run/environment.json` and `run/inputs.json` are byte-exact aliases of `run/energy/`; `run/launch.json` is the original joint launch receipt. The aliased result covers the energy worker only. Complete accounting covers all preflight, original failed work, recovery, energy, transfer, audit and engineering work separately.

The [dashboard](probe_dashboard.png) and [PDF](probe_dashboard.pdf) show height, total/potential/kinetic energy, balance gate and torques for all ten episodes. This format-only v2 moves legends out of the data panels; the original figure and all four external recorded-state GIFs are preserved. See the [scientific interpretation](../../docs/acrobot_energy_discovery_result.md) and [accounting appendix](../../docs/acrobot_energy_discovery_accounting.md).
