# Saved-only LQR certificate finalization — R1

The native `a12b207` stage completed all 362 declared episodes, all 37 local-map
measurements and the twelve-control replay: 181,000 executed controls, 49
verification controls and 362,098 physics substeps. Its final serialization then
raised `FileExistsError` because `lqr.json` already held the initial lock and the
shared writer is create-only. The original `STOPPED_INVALID_EXECUTION` result,
source, arrays, initial lock and paid costs remain unchanged.

This [bounded recovery](../configs/studies/acrobot_attractor_finalization_r1.json)
performs **zero physics, inference or fitting**. It independently verifies exact
source/configuration/result/H5/initial-lock hashes, native finite differences,
replay, deployed controls, costs and all 362 saved outcomes. It reconstructs the
234 calibration inputs and 128 confirmation inputs using the original seeds and
unchanged selection rule. Original native A/B, P and K are copied exactly; only
the missing predeclared level-set metadata is derived.

Create `recovered-lqr/` with a derived `result.json`, the exact original
`config.json` and final `lqr.json`. Its result records the original failure and
inherited work separately from zero new native work. It must pass every original
local-lock requirement; no failed capture, tolerance or seed may be replaced.
The old worker is not resumed or rewritten.

Limits are **20 CPU seconds, 120 wall seconds and 8 MiB**; preserve any failed
recovery. The support reservation remains unchanged: 20 CPU seconds recovery,
110 final saved review and 20 tests fit the original 150-second allocation.
No native or total storage budget increases.

After successful independent finalization, the remaining already-authorized
stages execute the **original frozen worker and original configuration**, binding
the new certificate path explicitly. A versioned launch supervisor changes only
that argument. The original worker itself verifies the lock equals its derived
result and that the original configuration bytes match. All acquisition settings,
seeds, costs, gates, stage order and conditional work remain as frozen.

The active implementation separately fixes the future writer lifecycle by naming
the initial packet `lqr_initial.json`. This correction is not applied to the
immutable worker used by the present study and repeats none of its physics.
