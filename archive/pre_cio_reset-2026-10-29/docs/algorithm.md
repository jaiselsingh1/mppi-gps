# Current formulation: outcome-trained independent MPPI ↔ Walker policy

**Status: source implementation in progress; no Walker training run for this formulation has started.** This page describes the selected method. [Results](results.md) distinguish executed evidence from this design. The current public CLI still exposes Point Mass qualification only.

## Task

Use the existing MuJoCo Walker2d XML, six bounded controls and 0.008-second controls. Native running-cost weights are unchanged. The prospective finite task has 600 controls: target speed 1.5m/s for 400 controls, then 0 for 200. Walking competence requires at least 1m displacement at control 400 and mean forward speed at least 0.75m/s over controls 125–399. Ending requires health, absolute forward speed at most 0.2m/s and an unscaled native non-forward velocity norm at most 1 throughout the final 125 controls. That norm includes root vertical velocity and angular velocities; it is a declared mixed-unit numerical metric, not a joint-speed norm in rad/s.

The task adds a declared1,000 final failure charge if walking competence and settling are not both attained. Native unhealthy terminal cost20 is charged once; an early fall also incurs5 per unexecuted remaining control. Every controller receives the same task and model. These settings must be frozen before the first experiment; they are not qualified results.

Snapshots include full MuJoCo integration state, solver warmstart, time and task history. Actor observations contain the native17 Walker features plus target speed, remaining time, time to stop, walk progress, accumulated walk-speed context and settling progress. There is no custom transition equation.

## Search → policy

Independent MPPI owns its proposals, private RNG, nominal sequence and trajectory memory. Retain the tuned Gaussian channel and additional independent smooth/memory proposals. Simulate retrieved action continuations from the current physical snapshot. Verify the synthesized independent mean and retain the better task-scored result. Only independent scores update that private planner state.

Actual executed trajectories provide acquired reset states and measured remaining teacher costs. The teacher return belongs to the teacher that executed that continuation. It is a scalar performance bound, not a measured return of the actor and not an action label.

The global component is a bounded23-input,128×128 action network with six outputs. It starts without an imported imitation policy. The currently selected critic-free update perturbs all mean-network parameters in four paired Gaussian directions, keeping deterministic-unused log-standard-deviation parameters fixed. Each version is evaluated on the same four complete contexts: two ordinary and two acquired.

For actor-owned cost J and a measured teacher continuation bound B at exactly the same snapshot, the proposed per-context objective is J/2800 plus the squared positive excess max(J−B,0)²/2800² where a bound exists. Ordinary starts have no invented teacher bound. Paired outcomes estimate a parameter-space smoothed-return gradient; Adam proposes a new actor. Returns of perturbation actors are never relabeled as outcomes of that new actor.

This method uses complete measured outcomes to learn preparation and finishing. It assumes discovery visits useful, recoverable states and that the actor can represent a useful feedback law. It does not assume a neural policy invents evidence of unseen success. Parameter perturbation is an existing optimization component, not the research novelty.

## Policy → search

Start each policy version with zero influence. Measure uninterrupted actor-only suffixes on acquired states and small physical perturbations; bind the exact policy/task/family hash, remaining horizon and tested support.

A copied immutable candidate bank may be scored with original short-window task cost plus weighted policy-action agreement. The policy cannot seed, restrict or overwrite the independent proposals, nominal update or memory. Restoring an identical physical and planner snapshot must reproduce the same independent bank regardless of the actor.

Suffix competence permits an experimental guidance trial only. Complete paired executed replanning and fresh confirmation must show lower complete cost without capture regression before promotion. Unsupported states, changed versions, numerical errors or failed requalification revoke influence. A hypothetical actor-tail score is not evidence about the executed MPC controller.

## The bidirectional test

Freeze the incoming actor, discovery data, learner budget and evaluation roots. Train one child using independent-search experience and another using qualified guided-search experience, with the same update procedure. Test both standalone on fresh ordinary starts. Record whether guidance changed executed decisions, improved control and produced a better next standalone actor; require two actual changed-policy feedback rounds for the full loop claim.

Training acceptance, standalone qualification, guidance eligibility, controller promotion and bidirectional benefit are separate outcomes. Partial curriculum or suffix success is not standalone deployment. Failure of any link stays a negative; the task or budget is not silently changed.

## Implemented boundary

- Existing real Walker physics, native MPPI and tuned configurations are available.
- Walker task/query adapter and critic-free actor-update helper have been written and are being reviewed.
- Six-action proposal and variable-observation support have been added to the generic core.
- Physical Walker validation, the integrated orchestrator, fresh training and both feedback comparisons remain unfinished.
- No trained actor, critic, successful current Walker run or sample-efficiency advantage is claimed.

The retired Ending method and older value/imitation experiments are historical approaches, not alternative active recipes. The [read-only CIO-GPS comparison](../artifacts/2026-10-29_ciogps_readonly_comparison_001/comparison.md) explains the other branch's distinct action-matching loop. Its later changes are outside that pinned audit.
