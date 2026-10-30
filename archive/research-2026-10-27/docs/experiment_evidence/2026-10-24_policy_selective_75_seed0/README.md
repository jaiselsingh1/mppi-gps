# Policy-selective MPPI distillation: completed 75-iteration study

Both trained networks learned standalone Walker locomotion in this experiment. Anchor completed 96/100 eight-second episodes; Selective completed 95/100. Both met the predeclared observed-mean development gate of at least 90% survival and velocity MAE at most 0.30 m/s. Selective tracked speed more accurately, with no clear survival or duration advantage over Anchor.

**These are two policies trained with one training seed (0). All intervals describe evaluation-reset variability conditional on those policies; they do not measure training-seed robustness.** This study does not establish an exploration or sample-efficiency advantage, superiority to independent MPPI, or replacement of critic-based policy-gradient methods. Selective's action-change RMS was higher, so lower speed error does not establish a smoother or more natural gait.

## Frozen protocol

The two arms ran on `robopil5090` from 2026-10-24 20:51:14 EDT to 23:13:55 EDT. Both completed exactly 75 ordered iterations. Their saved configurations differ only in `selection_mode`.

| Setting | Value |
| --- | --- |
| Initialization | Identical random seed-0 tanh MLP; 17 → 256 → 256 → 6 |
| Teacher | GPU MPPI, K=256, H=32, policy tail T=100, ESS target 0.2K |
| Exploration scale | 0.3 / sqrt(1 + k/25), zero-based k |
| Collection | 10 episodes × at most 400 controls per iteration, policy execution noise 0.1 |
| Labels and replay | Every two controls; two recent iterations |
| Regression | 2,000 Adam updates, batch 256, learning rate 0.0003 |
| Training monitor | 10 fixed development resets every five completed iterations |
| Final checkpoint rule | `policy_074.pt`; no outcome-based checkpoint selection |
| Final evaluation | 100 identical fresh development resets 8200000–8200099, at most 1,000 controls, dt=0.008 s |
| Deployment during evaluation | Deterministic network alone, no planner or critic |
| Video reset | Preselected reset 8200000 |
| Reserved final panel | 5000000–5000199; untouched by this comparison |

The executed implementation was frozen before the later research-telemetry additions. These outcomes are not a rerun of the subsequent instrumented implementation. The driver reports `source_unchanged=true` after checking all 115 frozen source/configuration/asset fingerprints. The server environment was Python 3.11.16, Torch 2.10.0, MuJoCo 3.6.0, mujoco-warp 3.7.0.1, and NumPy 2.4.3.

## Final standalone results

All failed episodes remain in the averages. Velocity MAE is the mean over each episode's executed controls, then the equal-weight episode mean. Adjusted return includes the analytic failure tail and is a secondary training diagnostic. Executed total cost alone is not compared across unequal episode lengths.

| Policy | Eight-second survival | 95% Wilson survival interval | Mean duration | Velocity MAE | Adjusted return |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 0/100 | [0.00%, 3.70%] | 0.51640 s | 1.736125 m/s | -1255.62751 |
| Anchor | 96/100 | [90.16%, 98.43%] | 7.89312 s | 0.223838 m/s | -621.48759 |
| Selective | 95/100 | [88.82%, 97.85%] | 7.80552 s | 0.202904 m/s | -533.71441 |

Both arms pass the gate on observed means. This is not confidence-bound certification: Selective's survival interval extends below 90%.

Paired differences below are Selective minus Anchor. Continuous intervals use 10,000 paired-reset percentile bootstrap draws with fixed seed 20261024; survival uses the paired Newcombe score interval.

| Metric | Paired mean difference | 95% interval |
| --- | ---: | ---: |
| Survival | -1.000 percentage point | [-7.742511, +5.612466] pp |
| Duration | -0.087600 s | [-0.314012, +0.123688] s |
| Velocity MAE | -0.020933525 m/s | [-0.027084990, -0.014436125] m/s |
| Adjusted return | +87.773182 | [+29.224300, +146.073641] |

There were 91 paired resets where both survived, four where only Selective survived, and five where only Anchor survived. Anchor used 228,307 collected controls and 114,275 labelled states; Selective used 245,173 controls and 122,684 labelled states. These counts exclude the substantial internal MPPI simulation budget and therefore are not a measure of total sample efficiency. Neither arm had an unreachable ESS-target state; mean ESS remained approximately 51.2. Arm training times were 8,405.07 s and 8,528.08 s, respectively, with the arms running concurrently.

## Audit and local artifacts

Independent review verified 75 ordered metric rows per arm, complete status files, configurations differing only by mode, all three checkpoint/report hashes, exactly matched ordered evaluation seeds, the 1,000-control horizon, recomputed episode means, and paired uncertainty. The six report regression tests passed, including the actual 0/100 survival boundary. Both final figures were visually checked. A report-only rendering fix plots absolute confidence intervals independently of the mean bars; it changes no numeric result or executed training source.

The following generated files remain in the ignored run directory, following [the artifact policy](../../artifact_layout.md). They are local evidence and are not included in the published repository:

- [Full comparison report](../../../runs/2026-10-24_policy_selective_75_seed0_001/comparison.md), [numeric comparison](../../../runs/2026-10-24_policy_selective_75_seed0_001/comparison.json), [learning curves](../../../runs/2026-10-24_policy_selective_75_seed0_001/training_curves.png), and [standalone metrics plot](../../../runs/2026-10-24_policy_selective_75_seed0_001/final_metrics.png).
- [Labeled side-by-side video](../../../runs/2026-10-24_policy_selective_75_seed0_001/videos/standalone_side_by_side.mp4); original [initial](../../../runs/2026-10-24_policy_selective_75_seed0_001/videos/initial.mp4), [Anchor](../../../runs/2026-10-24_policy_selective_75_seed0_001/videos/anchor.mp4), and [Selective](../../../runs/2026-10-24_policy_selective_75_seed0_001/videos/selective.mp4) videos.
- Local root: `/Users/jaisel/Documents/mppi-gps/runs/2026-10-24_policy_selective_75_seed0_001`. Video paths are under its `videos/` directory. Both trained policies complete the preselected video reset.

Rebuild the report from the preserved local study with:

```sh
python -m scripts.diagnostics.compare_selective_policies \
  --study-dir runs/2026-10-24_policy_selective_75_seed0_001
```

Checkpoints, raw episode reports, numeric logs, and plots are intentionally not copied into Git. The hashes below identify the preserved evidence; the complete frozen source inventory is in the local `protocol.json`.

| Local study artifact | SHA-256 |
| --- | --- |
| `protocol.json` | `65cc3099678b3eada21580dd2235869fed4cfab53a5b4ef2f0e8254291f7a593` |
| `study_status.json` | `61c8274293a07d69824137442dc802eccaeabc4ed0bc0a388ab9f9a17f86dc58` |
| `anchor/metrics.jsonl` | `ef4c647903fdb3efc4093f8d7694618fd951a2b0499e56b84b95874ee9b5f89e` |
| `selective/metrics.jsonl` | `69c5cf4de0d07be45838ab27bd06e9ce5ab0ea78137ee51e4ce8d93610896dce` |
| `policy_initial.pt` | `b05c727765f0e2fed0032b633a048e53abab3d1d0130fea743ee0da0932cee5d` |
| `anchor/policy_074.pt` | `9021a386b879d812e698ed59e854689e70f16a57614ce2db7ea7729d433286cb` |
| `selective/policy_074.pt` | `49e72d0e2db46a7202d517c9093f6412f75c70152846c83688f2740e1823d648` |
| `evaluation/initial.json` | `4f24f140f945b8417e165cdeb2e6000af01343c3561746f8449cc7ca1a98b5dd` |
| `evaluation/anchor.json` | `aa0f793f7ca6641696961727197a80e854e4e62660990a73dc8e6b5763715085` |
| `evaluation/selective.json` | `9bb51a0e5b388835c285604fe7aa5729350f3c54ba149dd3e22155058942eda7` |
| `comparison.json` | `1de7f0d29bfc5d73ded9ca972a3a82c2fcb54feb63edc564c1b273f6e6898c27` |
| `training_curves.png` | `c936dcb0ef87eb2a12688a5848d68d923b46ce1eab9a42113ec85f0e927cbe46` |
| `final_metrics.png` | `bfbe2491730ddf5bf607f6f9ef74fa7c393419e481d86c1c994b6b0de483e163` |

The canonical SHA-256 of all 115 frozen `source_sha256` entries is `0d4c44513d5d12c85ae520d9329e983c4c7e8ba2b962601a68db836702daa496`, computed from `json.dumps(source_sha256, sort_keys=True, separators=(',', ':')).encode()`. Selected executed-source fingerprints follow; later source changes in this repository do not retroactively change this run.

| Executed source/configuration | Frozen SHA-256 |
| --- | --- |
| `scripts/training/train_policy_selective.py` | `719c7185ad9ea8398b9b74d5804da2a62149cacbf4462499b1f04494f277b011` |
| `scripts/training/train_walker_mdgps.py` | `66cb1134e16a24c8942799364536815f55dc843ff4b4c5243d5ec126bed49ecf` |
| `scripts/runners/eval_policy_walker2d.py` | `0f8538b9bf467057ef6f3c2156bd1f75f6b1c10cad892c8c1862cedaa2e6d565` |
| `src/gps/gpu_estep.py` | `9957f9fefb44d89727b458fb911438ff6ec000526706eab29f01a0ff3d4b54cd` |
| `src/gps/mdgps.py` | `088e4533cc0826a30a21b145eb7728bc82484107ade8dc2cf87784dbe44c4325` |
| `src/gps/policy_tail.py` | `fc1fd4f059ddd03ce086810aebc2060a3e02a13f20daa69380398ba9fa4f0481` |
| `src/gps/selection.py` | `49d55810fe5e884618644538805e7ab59f8c262e7de01da1e91d467457d36d00` |
| `src/gps/policy.py` | `3e7403a0630a7d7f323e85552641166159b00b2e1a7fad53bb4a7101e8240276` |
| `src/envs/walker2d.py` | `477c3d2d8735171825392c9fe011941bc26e386f2113b4dd54df8a8c1e03ebca` |
| `src/envs/mujoco_env.py` | `ebc5833d2cf0dd7242e2e4674cc1099dda964cfd4ad8bcc3f59c1c319d61e502` |
| `configs/walker2d_task.json` | `546f6902405daaddb37fb9c458d77fb5159234d1218ec73abf9a625ec9f8f27a` |
| `configs/walker2d_best.json` | `16435775c846c8037cc17be218cdf1255fa9abb971174ff04f4df363449a3ad6` |
| `assets/walker2d.xml` | `4f33bf5069a0bf65056871ab4579f09c62696e7f46c3af6936efedacc572d590` |
