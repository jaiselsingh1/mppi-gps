# Scalar-layout numerical repair, development revision 3

Preserve dev0/dev1 learning negatives and the interrupted dev2 metric run.
Dev2 accepted its first metric update through actual complete training outcomes,
then stopped at the next strict action/feature consistency check. The batch
feature forward and scalar executed controller accumulated float32 operations
differently after the mean head changed. No teacher or next-student comparison
completed in that interrupted run. Its actual work and successful first update
do not establish full-loop success.

Change ONLY the metric feature forward layout: use the exact single-observation
actor call used by actual execution, collecting the same80hidden-state rows.
Keep the original2e-7 action check. No tolerance, direction, training objective,
metric, margin, task reward, actor architecture, projection, fraction ladder,
promotion criterion or expenditure ceiling changes. Existing dev2 config still
selects its original batch method; the new explicit layout field selects this
repair. Start again from the same pinned pure-MC parent, preserving all prior
costs; do not silently resume/overwrite the interrupted run.

Before a new full run, a finite changed-controller qualification selects the
first accepted checkpoint and the following completed base traces by chronology.
Compare the repaired mean-head analytic cost and eight endpoint derivatives
against nine autograd calls at THAT changed actor. Both methods must match at
3e-6absolute/3e-5relative. This costs160neural rows, zero simulator transitions,
80analytic policy and80physical derivative rows. Then compare that fixed actor
and original parent on16fresh paired ordinary roots8100000--8100015:800actual
controls and800NN rows. Require complete mean-cost gain>=.001, >=90%capture and
capture nonregression. No parameter/fraction is selected. Failure blocks this
repair run; retain the qualification packet. Total qualification cap800controls,
960NN rows,nine autograd calls,zero optimizer. All work is recorded before calls.

The complete study remains
[dev2](bidirectional_deployment_adjoint_protocol_dev2.md), with two actual
teacher and matched-child rounds even if a scientific gate fails. New actual
reset namespaces7100000baseline,7110000training,7200000qualification,
7300000/7500000feedback,7800000--7800063final are disjoint from previous actual
resets. Auxiliary changed-actor qualification roots are separate from this
study and never become its final or qualification rosters.

Caps remain3,222,784scalar controls (3,000,816model;221,968actual),464,960NN rows,
128Adam/128backwards/32diagnostic calls for MC control,192metric updates and
solves,98,304halfspace checks,12,288ball checks,15,360analytic policy and15,360
physical Jacobian rows. The scalar layout changes call count, not neural-row
count. Every lookup, verification, failed earlier run and numerical replay
remains charged. A positive first update is not standalone-policy, search,
next-student, physical-task, novelty or efficiency evidence for the full loop.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.diagnostics.run_deployment_adjoint_loop \
  --geometry-run runs/2026-10-27_bidirectional_ending_geometry_001 \
  --parent-run runs/2026-10-27_bidirectional_return_curriculum_003 \
  --config configs/bidirectional_deployment_adjoint_dev3.json \
  --out runs/2026-10-27_bidirectional_deployment_adjoint_004
```
