# SAC continuation map, run 001

Small evidence capsule for the [SAC continuation results](../../docs/sac_continuation_results.md), run from the [frozen protocol](../../docs/sac_continuation_protocol.md) at commit `18482df`.

| File | Contents |
|---|---|
| [summary.json](summary.json) | The run's own summary, copied byte for byte |
| [continuation_map.png](continuation_map.png) | The run's figure, copied byte for byte |
| [posthoc_analysis.py](posthoc_analysis.py) | Analyses outside the frozen protocol; reads the saved arrays only |
| [posthoc.json](posthoc.json) | Its output |
| [capsule.json](capsule.json) | Paths, sizes and SHA-256 of the inputs, the run outputs and these files |

The per-start arrays (`results.npz`, 93 MB) and the K1 inputs stay outside Git under `runs/`. To regenerate the post hoc output from them:

```bash
python results/sac_continuation_001/posthoc_analysis.py
```
