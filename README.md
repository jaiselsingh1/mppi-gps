# MPPI–GPS: one Walker development loop

The v2 workflow tests fitting, corrective experience and early teacher–student co-adaptation on native MuJoCo Walker. The protected MPPI bank remains policy-independent.

**V2 is closed and negative for standalone walking:** the retained policy scored **0/16 held successes**, averaging **230.8125 of 1,000 controls**. Better reconstruction did not produce a qualified walking policy.

Both higher-weight guided teachers walked successfully; their +1.76% mean cost passed the frozen 5% cost guard. They were rejected solely because policy disagreement improved only 0.0655%, below the required 5%. The lower weight produced 1/2 successes. Consequently there was no confirmation, feedback child fit or child comparison; training-data usefulness was not tested.

Source and summary figures are versioned. Raw run/artifact links refer to separately preserved evidence; see [evidence availability](docs/evidence_availability.md).

Read the [algorithm](docs/algorithm.md), [code map](docs/code_map.md), [results](docs/results.md) and [paper mapping](docs/paper_mapping.md).

![Closed v2 results](docs/figures/walker_v2_summary.png)

[Closed result](runs/2026-10-29_cio_feedback_v2_001/result.json), [configuration](runs/2026-10-29_cio_feedback_v2_001/config.json), [retained checkpoint](runs/2026-10-29_cio_feedback_v2_001/checkpoints/final.pt), [fitting curves](runs/2026-10-29_cio_fit_v2_001/fits/fit_mse.png) and [saved-outcome audit](artifacts/2026-10-29_v2_outcome_audit_001/receipt_001/receipt.json) are available. The audit passed 4,092,016 checks with no new simulation or inference.

Videos show the [independent teacher walking](runs/2026-10-29_cio_feedback_v2_001/videos/independent_mppi_walk.mp4), [higher-weight teacher walking](runs/2026-10-29_cio_feedback_v2_001/videos/guided_mppi_high_weight_walk.mp4), [lower-weight teacher falling](runs/2026-10-29_cio_feedback_v2_001/videos/guided_mppi_low_weight_fall.mp4) and [standalone policy falling](runs/2026-10-29_cio_feedback_v2_001/videos/standalone_policy_fall.mp4). [Executed traces](runs/2026-10-29_cio_feedback_v2_001/exports/executed_traces.h5) and the [full raw-evidence locator](runs/2026-10-29_cio_feedback_v2_001/exports/remote_evidence.json) preserve the outcomes.

Future recovery should mean completing the task after a disturbance, rather than returning to one saved trajectory. Preserve complete executed outcomes and judge training-data usefulness through actual standalone child improvement; action agreement is diagnostic. These are next-formulation requirements, not changes to frozen v2.

The data-dependent stages require the preserved raw v1 evidence on the remote host. From that host, use the canonical working folder and existing interpreter:

~~~sh
ssh robopil5090
cd /home/jaisel/mppi-gps-workspace/current/mppi-gps
MPPI_GPS_PY=/home/jaisel/mppi-gps/.venv/bin/python

# First: saved-data fitting only. Requires the v1 raw evidence.
"$MPPI_GPS_PY" scripts/train_mppi_gps.py train --stage fit --experience runs/2026-10-29_walker_cio_mppi_001 --out runs/NEW_fit

# Then: matched actor evaluation, recovery, provisional teachers and next children.
"$MPPI_GPS_PY" scripts/train_mppi_gps.py train --stage feedback --experience runs/2026-10-29_walker_cio_mppi_001 --fit-run runs/NEW_fit --out runs/NEW_feedback

# Remaining public commands; choose a fresh output path for each run.
"$MPPI_GPS_PY" scripts/train_mppi_gps.py baseline --out runs/NEW_baseline
"$MPPI_GPS_PY" scripts/train_mppi_gps.py evaluate --checkpoint PATH_TO_POLICY.pt --out runs/NEW_evaluation
"$MPPI_GPS_PY" scripts/train_mppi_gps.py render --run runs/NEW_feedback --out runs/NEW_feedback/videos/recorded_episode.mp4
~~~

The 5090 canonical working folder is /home/jaisel/mppi-gps-workspace/current/mppi-gps; its existing interpreter is /home/jaisel/mppi-gps/.venv/bin/python. Raw v1 physics evidence is retained on that machine. Local results are mirrored for review.

New v2 work, including the auxiliary diagnostic, paid **78,815,397 controls and 50,694,380 neural rows**, with 15,360,000 backward rows and 60,000 Adam steps. Imported v1 work and validations remain separate; see [cost accounting](artifacts/2026-10-29_v2_report_001/cost_accounting.json).

The same CLI provides baseline, evaluate and render. Runs and source/configuration snapshots are create-only. The v1 negative and all earlier evidence remain [preserved](docs/history.md). No ending-task, population-wide success or efficiency result is claimed.
