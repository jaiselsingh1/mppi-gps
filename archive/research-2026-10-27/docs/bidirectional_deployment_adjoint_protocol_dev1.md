# Measured ending-constraint projection, development revision 1

This prospective revision follows observed dev0 rejection, without modifying
that still bounded run. At its first step the weighted-barrier proposal reduced
ordinary complete cost2.307269->2.286717 while worsening worst acquired ending
violation.432009->.446871. All four tested fractions failed the constraint gate.
At the development snapshot used for this formulation,128steps had no accepted
update. The full dev0 schedule continues and its negative/operational result,
all rejected work and exact frozen source remain retained.

## Different learning formulation

Keep the deployment mean objective and measured closed-loop tangent, but remove
the weighted barrier from the UPDATE DIRECTION. Compute the ordinary running
cost gradient g and eight separate full-remaining-trajectory ending gradients
A_i from the actual current actor's complete outcomes. These are analytic
model/actor derivatives, not critic predictions or teacher action labels.
The known target defines the ending constraints from the outset.

Start with d=-.03*g/||g||. Apply64cycles of projections onto the eight measured
linearized halfspaces A_i*d<=b_i. For ordinary contexts b_i=max(slack_i,0),
where slack_i=-.101-measured_final_y. For acquired contexts b_i=slack_i when
feasible, otherwise -min(-slack_i,.01). Thus the tentative direction may use
ordinary ending slack while restoring acquired infeasibility, instead of
averaging competing constraint and task gradients. This is a finite halfspace
feasibility method, not an exact-QP/convergence claim. It depends on sufficient
actor parameter flexibility and accurate local derivatives.

After64cycles require max linear violation<=1e-7, g*d<-1e-9 and ||d||<=.03+1e-7.
If these fail, the direction is zero. Never claim a projected direction is an
executed improvement. Try only fractions1,.5,.25,.125, each through actual
complete changed-actor executions. The SAME actual ordinary cost/capture and
acquired violation gates as dev0 accept or reject. Rejected updates restore
exact parent state; all computations remain charged. The surrogate binary
ending reward is not differentiated and the simulator reward never changes.

No Adam step implements the projected update. Store the cost gradient, eight
ending gradients, bounds, proposed direction, numerical feasibility/descent,
512halfspace checks per attempted update and every actual outcome. Nine
autograd.grad calls and80neural/Jacobian rows occur per step. The log-standard
deviation remains unchanged because deployment uses the deterministic mean.
The initial external MC control still uses its old own-return PPO procedure.

The revised executable falsifier separately compares BOTH running/constraint
surrogate and ending-state derivatives against actual finite parameter
perturbations. It also tests a feasible redistribution direction and refusal
when constraints admit no descent. The new test packet costs125controls,
137neural rows and eight autograd calls, plus pure roster/acceptance checks.
This repeated qualification is required by the new endpoint-gradient/projection
mechanism, not a repeat to obtain a better learning result. Keep036--038.

Packet039 passed both actual derivative checks (125controls/137NN/eight
autograd calls) but one pure fixture failed because NumPy refused converting a
length-one array to float. Correct only that fixture and move pure projection
checks into the integration file. Packet040 runs those pure checks without
repeating the125controls. The production tangent/projection source is identical
between039 and040. Both packets and the fixture failure remain charged.

## Unchanged full-loop test and frozen caps

Use the exact same paid pure-MC parent and independent discovery corpus as
[dev0](bidirectional_deployment_adjoint_protocol_dev0.md). The initial common
Adam reset,64new-learner versus32MC updates, two32-step matched-child rounds,
independent H4K32+smooth/memory search, full-memory RAW,50candidate forecasts,
support/promotion/invalidation, actual prefix4-then-actor21 handovers, paired
teacher/root panels, fresh16qualification and64final outcomes, minimum gain
.001 and90%capture thresholds are unchanged. Both feedback rounds run even if
the new method fails. These outcomes select no new step norm, projection count,
fraction, margin, task cost or budget.

New actual-reset namespaces:5100000baseline,5110000training,
5200000qualification,5300000/5500000feedback,5800000--5800063final. Previous
private training-noise seed integers are not actual reset/evaluation roots.
One actor parent/cheap task remains a mechanism-development scope; neither an
exploration/efficiency nor physical/three-seed conclusion is justified.

Caps remain3,222,784scalar controls =3,000,816model+221,968actual and464,960NN
rows. Work changes to128Adam/128training backward calls (MC control only),
1,760autograd.grad calls =32MC diagnostics+192*9full-tangent calls,
192projected parameter-update attempts,98,304halfspace checks and15,360active
analytic Jacobian rows. All four fraction tests at EVERY step are budgeted,
including invalid directions; unused reservations cannot fund extra work.
Independently audit raw arithmetic, own-controller target ownership, numerical
projection and actual acceptance, exact common parents, trust and causal links.
Do not claim independent neural-gradient replay from a source-bound audit.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.diagnostics.run_deployment_adjoint_loop \
  --geometry-run runs/2026-10-27_bidirectional_ending_geometry_001 \
  --parent-run runs/2026-10-27_bidirectional_return_curriculum_003 \
  --config configs/bidirectional_deployment_adjoint_dev1.json \
  --out runs/2026-10-27_bidirectional_deployment_adjoint_002
```
