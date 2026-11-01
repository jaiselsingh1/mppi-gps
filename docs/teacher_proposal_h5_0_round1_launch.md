# H5.0 round-one launch decision

The user asked to continue the research after preparation, validation and the approved off-host backup. This authorizes **one bounded first-round H5.0 comparison**, including the explicit weak-learner ordering exception described in the [execution protocol](teacher_proposal_h5_0_execution_protocol.md). The useful-learning prerequisite remains **unmet**. No second acquisition round, altered metric, new learner or automatic follow-up is authorized.

The [authorized recipe](../configs/studies/teacher_proposal_h5_0_round1.json) preserves every scientific field of the [preparation recipe](../configs/studies/teacher_proposal_h5_0.json): all seeds, task, observations, normalization, targets, optimizer, update counts, controls, coefficients, budgets, outcomes and stopping rules are identical. Only authorization, status and explicit recipe ownership differ. The original preparation and its false launch flag remain unchanged.

## Fixed question and decision

Does placing the same conditional action-agreement term inside the isolated second proposal stream improve the child after exactly 2,000 updates, compared with selector-only guidance? Independent task-only search is the third control and the protected producer remains actor-free. Keep descriptive motion distance R, conditional agreement D and usefulness after learning separate. No agreement/reconstruction gate can establish useful teaching.

Execute three fresh parent training streams once each, all 36 acquisition episodes before any child fit, and at most nine matched child fits. Each child inherits the same complete incoming policy, normalization, Adam/RNG and minibatch index stream within its contrast. If the complete acquisition corpus is unavailable or a teacher fails walking, finish the fixed acquisition roster and stop before learning children. Otherwise execute the fixed primary, confirmation and paired-gap rosters. A retention-only gain remains partial. All walking, survival, teacher-admission and regression gates in the original protocol remain fixed.

The full-roster upper reservations remain **1,339,255,104 native controls**, **5,357,020,416 physics substeps**, **4,005,571,104 combined worker/audit neural rows**, **24,000 Adam updates** and **512 GiB** evidence storage. The native worker has six wall-hours and 16 GiB RSS on 16 pinned CPU cores; its independent audit has two wall-hours and 4 GiB RSS. These are hard reservations, not a forecast of completion time or a claim of work already spent. No paid operation may be retried.

## Create-only execution and evidence

New run ID is `2026-11-01_teacher_proposal_h5_0_r1_001`, distinct from the unlaunched preparation. First commit/push the authorized recipe and recipe-binding validation. Then prepare locally from the original sealed W2 owner and transfer the full new source/config/input export to `robopil5090`, verifying every file before launch. Both pinned runtime metadata and sufficient disk space must be rechecked; changes stop execution.

```sh
python -B scripts/train_mppi_gps.py study teacher-proposal prepare \
  --recipe teacher_proposal_h5_0_round1 \
  --source-run runs/2026-10-31_student_update_w2_001 \
  --runtime-probe artifacts/2026-11-01_teacher_proposal_h5_0_r1_001/engineering/001_runtime_probe/stdout.txt \
  --seed-review artifacts/2026-11-01_teacher_proposal_h5_0_preparation_001/seed_review.json \
  --out runs/2026-11-01_teacher_proposal_h5_0_r1_001
```

The unchanged seed review retains its original availability limits. Verify that the old preparation has no worker, checkpoint, launch marker or raw stream; merely declaring its unused seeds was not a research run. Do not resample seeds or count a self-collision with that preparation as prior scientific use.

On the experiment host, use the new run's **frozen** `source/scripts/train_mppi_gps.py study teacher-proposal run --out RUN`, with `RUN/source` as the working directory. Its authenticated outer coordinator owns the live worker, CPU affinity, deadline and RAM limit. Invoke no hidden worker directly. Keep bytecode disabled. Coordinator logs, audit outputs and reports belong outside the sealed run; existing directories are never reused.

Monitoring reads bounded ledger tails and small result/exit files only. It performs no new inference or physics and has separate engineering accounting. After a completed contrast or normal teacher-competence stop, run the frozen audit CLI with `--mode round` in a new external output directory. It verifies all saved predictions within its own fixed census and deadline; it does not repeat training or simulation. Preserve complete costs, checkpoints, update curves, failures and unknown overhead. Publish compact evidence after audit; keep raw arrays/checkpoints/ledgers external.

An infrastructure, provenance, numerical, storage or deadline error makes the contrast incomplete. Preserve the paid prefix and outer exit receipt; do not call it an audited scientific negative, relax a tolerance, restart, or change the mechanism within this run. Close with the evidence and an explicitly versioned next recommendation for review. A successful first round permits preparing a second-round protocol, never launching it automatically. Obsidian receives one brief result/link after closure.
