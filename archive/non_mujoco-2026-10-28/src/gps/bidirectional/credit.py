"""Owned complete-feedback credit and explicit derivative qualification.

Arithmetic primitives and gradient functions read complete actor-owned
outcomes; they do not execute controllers or update parameters. Named
qualifiers execute separate versioned probe controllers through the query
executor, while leaving the base controller unchanged. The teacher supplies
only its exact-context component envelope. Credit uses Ending's local model
and actor derivatives without a critic, teacher action labels, or a fabricated
return.

The result is a smooth-arithmetic derivative on the recorded branch. Literal
float32 rounding is not differentiated. Clipping boundaries and the ending-lane
threshold are nonsmooth; the indicator derivatives are zero only away from that
threshold. Initial complete-feedback finite differences check the formula on
their tested directions. Later actor-specific derivatives remain local
proposals: replay checks their owners and arithmetic, and complete executions
must qualify each changed actor. Initial checks do not certify later branches.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .coupling import finite, primal_step_admissible
from .ending import EndingSnapshot
from .meter import attach_meter


def _finite(value, name, shape=None):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all() or (shape is not None and result.shape != shape):
        raise ValueError(f'Finite, matching {name} required')
    return result


def ending_training_metrics(terminal_y, ending_y, reserve):
    """A stricter learner target; task capture labels are never replaced.

    The signed training residual is literally ``(y - ending_y) + reserve``.
    This is a development target, not a bound on native arithmetic error.
    """
    y=_finite(terminal_y,'complete ending coordinates')
    if y.ndim!=1 or not y.size or isinstance(reserve,(bool,np.bool_)):
        raise ValueError('Nonempty ending vector and scalar nonnegative reserve required')
    if np.asarray(ending_y).ndim or np.asarray(reserve).ndim:
        raise ValueError('Scalar ending threshold and reserve required')
    threshold,margin=_finite([ending_y,reserve],'ending training target',(2,)).tolist()
    if margin<0.:raise ValueError('Nonnegative ending reserve required')
    with np.errstate(over='ignore',invalid='ignore'):residual=(y-threshold)+margin
    _finite(residual,'signed ending training residuals')
    target=float(_finite(threshold-margin,'ending training target'))
    flags=residual<=0.
    return dict(training_ending_margin=margin,training_ending_target=target,
        training_72_residuals=residual.tolist(),worst_72_training_residual=float(residual.max()),
        worst_72_training_violation=float(max(0.,residual.max())),
        all_72_training_capture=bool(flags.all()),all_72_training_capture_flags=flags.tolist())


def realized_primal_certificate(p,Q,A,b,delta,*,phase,radius,feasibility_atol,proposal_certificate):
    """Check the local constraints at the actual changed actor's displacement.

    A line-search fraction is a different point from the solver's full proposal.
    This is a primal arithmetic check, with the unchanged numerical tolerance;
    it neither certifies optimizer optimality nor establishes actual endings.
    Phase I recomputes its free nonnegative epigraph at this point. Complete
    executed ending progress/preservation and original-cost gates remain with
    the caller. The original backend must still have returned finite success.
    """
    p,Q,A,b,delta=(finite(v) for v in (p,Q,A,b,delta))
    radius,tolerance=finite([radius,feasibility_atol]).tolist()
    if (p.ndim!=1 or not len(p) or Q.ndim!=2 or Q.shape[0]!=len(p)
            or delta.ndim!=1 or not len(delta) or Q.shape[1]!=len(delta)
            or A.ndim!=2 or A.shape[1]!=len(delta) or b.shape!=(len(A),)
            or np.any(b<0.) or radius<=0. or tolerance<0.
            or phase not in ('feasibility','improvement')
            or (phase=='improvement' and np.any(p>0.))):
        raise ValueError('Finite matching realized-step arrays and valid original bounds required')
    with np.errstate(over='ignore',invalid='ignore'):
        performance_linear=Q@delta;safety_linear=A@delta
        performance=p+performance_linear;safety=safety_linear-b
        norm=float(np.linalg.norm(delta))
    finite_values=bool(all(np.isfinite(v).all() for v in (performance,safety,performance_linear,safety_linear,[norm])))
    if not finite_values:raise ValueError('Nonfinite realized-step primal arithmetic')
    epigraph=max(0.,float(performance.max())) if phase=='feasibility' else 0.
    shifted=performance-epigraph
    violation=max(0.,float(shifted.max()),float(safety.max()) if len(safety) else 0.,norm-radius)
    proposal=proposal_certificate
    backend=bool(isinstance(proposal,dict)
        and proposal.get('schema')=='performance-feasibility-dual-certificate-v1'
        and isinstance(proposal.get('checks'),dict) and isinstance(proposal.get('solver_status'),dict)
        and proposal['checks'].get('finite_values') is True
        and proposal['checks'].get('backend_success') is True
        and proposal['solver_status'].get('success') is True
        and proposal.get('phase')==phase and proposal.get('radius')==radius
        and proposal.get('feasibility_atol')==tolerance)
    checks=dict(finite_values=True,original_backend_success=backend,
        primal_feasible=bool(violation<=tolerance and norm<=radius+tolerance))
    return dict(schema='realized-candidate-primal-v1',admission_point='realized_float32_actor_displacement',
        phase=phase,checks=checks,primal_admissible=bool(all(checks.values())),certified=False,
        primal_max_violation=violation,actual_norm=norm,radius=radius,feasibility_atol=tolerance,
        epigraph=epigraph,linearized_performance=performance.tolist(),
        performance_residuals=shifted.tolist(),safety_residuals=safety.tolist(),
        performance_linear_values=performance_linear.tolist(),safety_linear_values=safety_linear.tolist(),
        original_full_proposal_primal_admissible=primal_step_admissible(proposal),
        optimality_claimed=False,ending_attainment_claimed=False,
        arithmetic_work=dict(performance_matvecs=1,safety_matvecs=1,norm_evaluations=1,
            performance_rows=len(p),safety_rows=len(A),parameter_dimension=len(delta)),
        scope='Original local inequalities at the executed candidate parameters only; no dual/optimality or true-outcome certificate')



def teacher_excess_nonregression(current, candidate, budgets, *, atol):
    """Check each measured teacher envelope separately, without trading debt.

    An existing excess may stay unchanged. A policy already below its teacher
    bound may use only that measured headroom. Passing does not imply that an
    initially unmet teacher bound has been attained. Ownership is the caller's
    responsibility; these arrays alone cannot establish it.
    """
    current=_finite(current,'current teacher-context costs')
    if current.ndim!=1 or not current.size:
        raise ValueError('Nonempty teacher-context cost vectors required')
    candidate=_finite(candidate,'candidate teacher-context costs',current.shape)
    budgets=_finite(budgets,'measured teacher budgets',current.shape)
    if any(np.any(value<0.) for value in (current,candidate,budgets)):
        raise ValueError('Teacher-context costs and budgets must be nonnegative')
    if isinstance(atol,(bool,np.bool_)) or np.asarray(atol).ndim!=0:
        raise ValueError('A finite nonnegative scalar cost tolerance is required')
    tolerance=float(atol)
    if not np.isfinite(tolerance) or tolerance<0.:
        raise ValueError('A finite nonnegative scalar cost tolerance is required')
    before=np.maximum(current-budgets,0.);after=np.maximum(candidate-budgets,0.)
    with np.errstate(over='ignore'):limit=before+tolerance
    if not np.isfinite(limit).all():raise ValueError('Finite teacher-excess limits required')
    per_context=after<=limit
    return dict(before_excess=before,after_excess=after,slack=np.maximum(budgets-current,0.),
        per_context=per_context,passed=bool(np.all(per_context)))


def stage_reference_budget(current_costs, reference_costs):
    """Mean-cost headroom on one measured, matched reset cohort.

    The caller binds both actor versions and the identical initial states.
    Numerical shape equality alone does not establish that ownership. A
    reference advantage belongs only to this call's cohort: no state or credit
    is carried between updates. This is neither a per-state cost guarantee nor
    a teacher action/return target for the current actor.
    """
    try:
        current=_finite(current_costs,'current ordinary complete costs')
        reference=_finite(reference_costs,'reference ordinary complete costs',current.shape)
        if current.ndim!=1 or not current.size or np.any(current<0.) or np.any(reference<0.):
            raise ValueError('Matching nonempty nonnegative cost vectors required')
        with np.errstate(over='ignore',invalid='ignore'):
            current_mean=float(current.mean());reference_mean=float(reference.mean())
        if not np.isfinite([current_mean,reference_mean]).all():
            raise ValueError('Finite measured cohort means required')
    except (TypeError,ValueError,OverflowError) as error:
        raise ValueError('Invalid stage-reference cost evidence: '+str(error)) from error
    return dict(current_mean=current_mean,reference_mean=reference_mean,
        limit=max(current_mean,reference_mean),slack=max(0.,reference_mean-current_mean))


def outcome_witness_scales(owned_costs, teacher_costs, *, floor=1e-6):
    """One shared, frozen normalization from measured whole-episode outcomes.

    Teacher rows are data arms; columns are the SAME initial contexts. These
    arithmetic inputs do not assert that any teacher or student succeeded.
    The executor/orchestrator must establish that from actual traces first.
    """
    owned = _finite(owned_costs, 'owned complete costs')
    teachers = _finite(teacher_costs, 'teacher complete costs')
    if (owned.ndim != 1 or not owned.size or teachers.ndim != 2
            or teachers.shape[1] != owned.size or not teachers.shape[0]
            or np.any(owned < 0.) or np.any(teachers < 0.)
            or not np.isfinite(floor) or floor <= 0.):
        raise ValueError('Matching nonnegative complete outcomes and positive scale floor required')
    complete = float(teachers.mean())
    if complete <= 0.:
        raise ValueError('Positive measured complete-cost normalization required')
    deficit = np.maximum(owned[None] - teachers, 0.)
    return dict(complete=complete, witness=max(float(np.sqrt(np.square(deficit).mean())), float(floor)))


def outcome_witness_terms(costs, budgets, *, complete_scale, witness_scale, beta,
                          context_gradients=None):
    """Whole-outcome hinge on OWN returns; teacher returns are fixed bounds.

    Optional Jacobian rows must belong to those exact owned returns. In
    particular, mean(residual) * mean(gradient) is not this derivative.
    """
    costs = _finite(costs, 'owned complete costs')
    budgets = _finite(budgets, 'teacher complete budgets', costs.shape)
    if (costs.ndim != 1 or not costs.size or np.any(costs < 0.) or np.any(budgets < 0.)
            or not np.isfinite([complete_scale,witness_scale,beta]).all()
            or min(complete_scale,witness_scale) <= 0. or beta < 0.):
        raise ValueError('Nonnegative complete outcomes, positive scales and nonnegative weight required')
    deficit = np.maximum(costs - budgets, 0.)
    penalty = float(np.square(deficit / witness_scale).mean())
    result = dict(outcome_excess=(deficit / witness_scale), outcome_penalty=penalty,
        objective=float(costs.mean()/complete_scale + beta*penalty))
    if context_gradients is not None:
        jacobian = _finite(context_gradients, 'owned context Jacobian')
        if jacobian.ndim != 2 or jacobian.shape[0] != costs.size:
            raise ValueError('One owned complete-cost gradient per context required')
        original = jacobian.mean(axis=0)
        teaching = (2*deficit[:,None]*jacobian / witness_scale**2).mean(axis=0)
        result.update(original_cost_gradient=original, teaching_gradient=teaching,
            gradient=original/complete_scale + beta*teaching)
    return result


def mixed_outcome_terms(global_costs, auxiliary_costs, budgets, *, complete_scale,
                        witness_scale, beta, global_gradient=None,
                        auxiliary_gradients=None):
    """Original mean return on one roster, plus exact-context teacher bounds.

    ``global_gradient`` is the already equal-trajectory mean derivative of
    ``global_costs`` across all complete clock groups. Auxiliary Jacobian rows
    instead belong to the matching auxiliary costs and measured budgets. The
    auxiliary mean return is not added to the objective a second time.
    Ownership and source-context provenance are established by the caller.
    """
    try:
        costs = _finite(global_costs, 'global complete costs')
        settings = _finite([complete_scale, witness_scale, beta], 'mixed outcome scales', (3,))
        if (costs.ndim != 1 or not costs.size or np.any(costs < 0.)
                or min(settings[:2]) <= 0. or settings[2] < 0.
                or ((global_gradient is None) != (auxiliary_gradients is None))):
            raise ValueError('Nonempty original costs and matching optional derivative evidence required')
        complete_scale, witness_scale, beta = settings.tolist()
        gradient = None if global_gradient is None else _finite(global_gradient, 'global mean gradient')
        if gradient is not None and (gradient.ndim != 1 or not gradient.size):
            raise ValueError('One finite global mean parameter gradient is required')
        auxiliary = outcome_witness_terms(auxiliary_costs, budgets,
            complete_scale=complete_scale, witness_scale=witness_scale, beta=beta,
            context_gradients=auxiliary_gradients)
        if gradient is not None and auxiliary['teaching_gradient'].shape != gradient.shape:
            raise ValueError('Global and auxiliary gradients must share the named parameter dimension')
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Invalid mixed outcome evidence: ' + str(error)) from error
    result = dict(global_mean_cost=float(costs.mean()),
        outcome_excess=auxiliary['outcome_excess'], outcome_penalty=auxiliary['outcome_penalty'],
        objective=float(costs.mean()/complete_scale + beta*auxiliary['outcome_penalty']))
    if gradient is not None:
        teaching = auxiliary['teaching_gradient']
        result.update(original_cost_gradient=gradient, teaching_gradient=teaching,
            gradient=gradient/complete_scale + beta*teaching)
    if (not np.isfinite([result['global_mean_cost'],result['outcome_penalty'],result['objective']]).all()
            or not np.isfinite(result['outcome_excess']).all()
            or (gradient is not None and not np.isfinite(result['gradient']).all())):
        raise ValueError('Mixed outcome evidence produced nonfinite arithmetic')
    return result


def unit_loss_gradient(gradient):
    """Separate proximal parameter step units from arbitrary loss units.

    This positive rescaling preserves the loss descent direction, not the
    displacement produced by a fixed proximal coefficient. A zero derivative
    stays zero; no direction or successful outcome is invented.
    """
    value=_finite(gradient,'loss gradient')
    if value.ndim!=1 or not value.size:raise ValueError('A finite nonempty gradient vector is required')
    norm=float(np.linalg.norm(value))
    if not np.isfinite(norm):raise ValueError('A finite gradient norm is required')
    return (value/norm if norm>0. else value.copy()),norm


def ending_cotangents(task_dynamics, pre_states, actions, components, budgets,
                      policy_dy, *, beta, return_scale, scales, budget_free=False):
    """Return per-trajectory action cotangents without simulating a transition.

    Inputs describe B complete trajectories with H saved pre-action states.
    ``policy_dy`` is the derivative of the *bounded mean action* with respect to
    the pre-action position, including the actor's observation transformation.
    The fixed initial contexts and teacher budgets are not differentiated.

    For C=sum_t components, the objective is
    sum(C)/return_scale + beta*sum(max((C-budget)/scales, 0)**2).
    Its component weights are 1/return_scale +
    2*beta*max(C-budget, 0)/scales**2. A reverse adjoint includes future feedback:
    r_t=l_a + f_a*lambda_next;
    lambda_t=l_y + f_y*lambda_next + policy_dy_t*r_t.
    ``action_cotangents`` are r_t, before averaging across trajectories.

    At an exact clip boundary we use the strict-interior convention, and at an
    exact lane threshold we use zero indicator derivative. Such rows have
    ``smooth_branch=False`` and must be rejected for gradient qualification.
    ``clip_min_distance=2`` is a finite sentinel when no movable slot exists.
    Explicit ``budget_free=True`` accepts ``budgets=None`` only at beta zero;
    no teacher target is manufactured for original-cost credit.
    """
    pre = _finite(pre_states, 'pre-action states')
    if pre.ndim != 3 or pre.shape[-1] != 2 or min(pre.shape[:2]) < 1:
        raise ValueError('Pre-action states must have shape (B,H,2), B,H >= 1')
    batch, horizon = pre.shape[:2]
    action = _finite(actions, 'bounded actions')
    if action.shape == (batch, horizon, 1):
        action = action[..., 0]
    if action.shape != (batch, horizon) or np.any(np.abs(action) > 1):
        raise ValueError('Bounded actions must have shape (B,H) or (B,H,1)')
    component = _finite(components, 'complete cost components', (batch, horizon, 4))
    if np.any(component < 0):
        raise ValueError('Nonnegative measured cost components required')
    if budget_free:
        if budget_free is not True or budgets is not None or beta != 0.:
            raise ValueError('Budget-free arithmetic requires absent budgets and beta zero')
        budget = None
    else:
        budget = _finite(budgets, 'teacher budgets', (batch, 4))
        if np.any(budget < 0):
            raise ValueError('Nonnegative teacher budgets required')
    sensitivity = _finite(policy_dy, 'bounded-policy position derivatives', (batch, horizon))
    scale = _finite(scales, 'component scales', (4,))
    if (np.any(scale <= 0) or not np.isfinite([beta, return_scale]).all()
            or beta < 0 or return_scale <= 0):
        raise ValueError('Positive scales/return scale and nonnegative beta required')

    clock, position = pre[..., 0], pre[..., 1]
    expected = clock[:, :1] + np.arange(horizon)[None, :]
    if (np.any(clock != np.floor(clock)) or np.any(clock < 0)
            or np.any(clock >= task_dynamics.T) or np.any(np.abs(position) > 1)
            or not np.array_equal(clock, expected)
            or np.any(clock[:, -1] != task_dynamics.T - 1)):
        raise ValueError('Complete unpadded pre-action trajectories ending at clock T-1 required')

    totals = component.sum(axis=1)
    weights = np.full((batch, 4), 1. / return_scale, dtype=np.float64)
    objective = totals.sum(axis=1) / return_scale
    if beta != 0:
        excess = np.maximum(totals - budget, 0.)
        weights += 2. * beta * excess / scale**2
        objective += beta * np.square(excess / scale).sum(axis=1)

    movable = clock < task_dynamics.gate
    unclipped = position + task_dynamics.move * action
    interior = movable & (unclipped > -1.) & (unclipped < 1.)
    # After the gate, position is copied and controls have no causal effect.
    f_y = np.where(movable, interior.astype(np.float64), 1.)
    f_a = task_dynamics.move * interior
    greed_y_next = -.5 * task_dynamics.greed * weights[:, 1, None]
    l_y = np.where(movable, greed_y_next * f_y, 0.)
    l_a = (2. * task_dynamics.effort * weights[:, 0, None] * action * movable
           + greed_y_next * f_a)
    cotangent = np.zeros((batch, horizon), dtype=np.float64)
    adjoint = np.zeros(batch, dtype=np.float64)
    for step in range(horizon - 1, -1, -1):
        cotangent[:, step] = l_a[:, step] + f_a[:, step] * adjoint
        adjoint = (l_y[:, step] + f_y[:, step] * adjoint
                   + sensitivity[:, step] * cotangent[:, step])

    clip_distance = np.minimum(np.abs(unclipped - 1.), np.abs(unclipped + 1.))
    clip_min = np.where(movable, clip_distance, 2.).min(axis=1)
    # Every complete trajectory contains the final pre-state after the gate.
    ending_signed = task_dynamics.ending_y - position[:, -1]
    ending_distance = np.abs(ending_signed)
    # A complete trace's next saved pre-position is its current post-position;
    # the last position is unchanged because the final control is after gate.
    post_position = np.concatenate((position[:, 1:], position[:, -1:]), axis=1)
    margins = dict(
        clip_min_distance=clip_min,
        ending_signed_margin=ending_signed,
        ending_abs_distance=ending_distance,
        movable_steps=movable.sum(axis=1),
        smooth_branch=(clip_min > 0.) & (ending_distance > 0.),
        clip_regions=np.where(movable, np.where(unclipped < -1., -1,
                              np.where(unclipped > 1., 1, 0)), 0).astype(np.int8),
        movable_mask=movable,
        good_lane=post_position <= task_dynamics.ending_y,
        good_lane_pre=position <= task_dynamics.ending_y,
    )
    if not all(np.isfinite(value).all() for value in
               (weights, objective, cotangent, adjoint)):
        raise FloatingPointError('Nonfinite deterministic feedback credit')
    return dict(action_cotangents=cotangent, component_weights=weights,
                objective_per_episode=objective, branch_margins=margins)


def terminal_position_cotangents(task_dynamics, pre_states, actions, policy_dy,
                                  selected_rows):
    """Differentiate a weighted sum of measured complete terminal positions.

    ``selected_rows`` has shape (B,). The neural adapter uses one global one-hot
    row across all groups, so its objective is the single measured y_T. Here the
    terminal adjoint is initialized to these fixed row weights, with no local
    cost and no normalization by the number of trajectories. The actor feedback
    derivative still enters every reverse step. Teacher actions and Ending's
    discontinuous success indicator do not enter this endpoint objective.

    A lane-threshold equality is harmless for y_T differentiability. Exact
    clipping boundaries remain nonsmooth and are reported in ``smooth_branch``.
    As in ``ending_cotangents``, this differentiates smooth arithmetic and omits
    literal floating point rounding.
    """
    pre = _finite(pre_states, 'pre-action states')
    if pre.ndim != 3 or pre.shape[-1] != 2 or min(pre.shape[:2]) < 1:
        raise ValueError('Pre-action states must have shape (B,H,2), B,H >= 1')
    batch, horizon = pre.shape[:2]
    action = _finite(actions, 'bounded actions')
    if action.shape == (batch, horizon, 1):
        action = action[..., 0]
    if action.shape != (batch, horizon) or np.any(np.abs(action) > 1):
        raise ValueError('Bounded actions must have shape (B,H) or (B,H,1)')
    sensitivity = _finite(policy_dy, 'bounded-policy position derivatives', (batch, horizon))
    selected = _finite(selected_rows, 'fixed terminal row weights', (batch,))
    clock, position = pre[..., 0], pre[..., 1]
    expected = clock[:, :1] + np.arange(horizon)[None, :]
    if (np.any(clock != np.floor(clock)) or np.any(clock < 0)
            or np.any(clock >= task_dynamics.T) or np.any(np.abs(position) > 1)
            or not np.array_equal(clock, expected)
            or np.any(clock[:, -1] != task_dynamics.T - 1)):
        raise ValueError('Complete unpadded pre-action trajectories ending at clock T-1 required')

    movable = clock < task_dynamics.gate
    unclipped = position + task_dynamics.move * action
    interior = movable & (unclipped > -1.) & (unclipped < 1.)
    f_y = np.where(movable, interior.astype(np.float64), 1.)
    f_a = task_dynamics.move * interior
    cotangent = np.zeros((batch, horizon), dtype=np.float64)
    adjoint = selected.copy()
    for step in range(horizon - 1, -1, -1):
        cotangent[:, step] = f_a[:, step] * adjoint
        adjoint = (f_y[:, step] * adjoint
                   + sensitivity[:, step] * cotangent[:, step])

    clip_distance = np.minimum(np.abs(unclipped - 1.), np.abs(unclipped + 1.))
    clip_min = np.where(movable, clip_distance, 2.).min(axis=1)
    ending_signed = task_dynamics.ending_y - position[:, -1]
    post_position = np.concatenate((position[:, 1:], position[:, -1:]), axis=1)
    margins = dict(
        clip_min_distance=clip_min,
        ending_signed_margin=ending_signed,
        ending_abs_distance=np.abs(ending_signed),
        movable_steps=movable.sum(axis=1),
        smooth_branch=clip_min > 0.,
        clip_regions=np.where(movable, np.where(unclipped < -1., -1,
                              np.where(unclipped > 1., 1, 0)), 0).astype(np.int8),
        movable_mask=movable,
        good_lane=post_position <= task_dynamics.ending_y,
        good_lane_pre=position <= task_dynamics.ending_y,
    )
    objective = selected * position[:, -1]
    if not all(np.isfinite(value).all() for value in (cotangent, adjoint, objective)):
        raise FloatingPointError('Nonfinite terminal-position feedback credit')
    # Keeping the detail shapes consistent makes their absence of cost credit
    # explicit. The endpoint is a task-state constraint, not a component cost.
    return dict(action_cotangents=cotangent,
                component_weights=np.zeros((batch, 4), dtype=np.float64),
                objective_per_episode=objective, branch_margins=margins)


def original_cost_cotangents(task_dynamics, pre_states, actions, components,
                             policy_dy, selected_rows, *, allow_multiple=False,
                             budget_free=False):
    """Branch-local credit for one original, unnormalized complete return.

    A group may contain its single selected row or no selected row. The neural
    adapter retains every original batch row and applies one global one-hot
    selection across groups. Unselected rows have zero cost weights, objective
    and action cotangents. There is no teacher penalty or trajectory averaging.
    Discrete detour/ending jumps have no local derivative away from their
    boundaries; the original measured costs and branch margins remain visible.
    Explicit ``allow_multiple=True`` keeps each selected row's unaveraged
    credit for a later weighted aggregate. ``budget_free=True`` represents the
    absence of teacher evidence without a dummy budget array.
    """
    pre = _finite(pre_states, 'pre-action states')
    if pre.ndim != 3 or pre.shape[-1] != 2 or min(pre.shape[:2]) < 1:
        raise ValueError('Pre-action states must have shape (B,H,2), B,H >= 1')
    selected = _finite(selected_rows, 'fixed original-cost row weights', (len(pre),))
    if (type(allow_multiple) is not bool or type(budget_free) is not bool
            or np.any((selected != 0.) & (selected != 1.))
            or (not allow_multiple and selected.sum() > 1.)):
        raise ValueError('Original-cost row weights must be zero or one, at most one per group')
    result = ending_cotangents(task_dynamics, pre, actions, components,
        None if budget_free else np.zeros((len(pre), 4), dtype=np.float64), policy_dy,
        beta=0., return_scale=1., scales=np.ones(4, dtype=np.float64), budget_free=budget_free)
    result['action_cotangents'] *= selected[:, None]
    result['component_weights'] *= selected[:, None]
    result['objective_per_episode'] *= selected
    return result


def _running_window_bounds(task_dynamics, start, stop):
    if (any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for value in (start, stop)) or not 0 <= start < stop <= task_dynamics.T):
        raise ValueError('An absolute integer clock window must satisfy 0 <= start < stop <= T')
    return int(start), int(stop)


def running_window_cotangents(task_dynamics, pre_states, actions, components,
                              policy_dy, selected_rows, *, start, stop):
    """Original effort+greed cost in one fixed absolute-clock interval.

    The supplied trajectories remain complete. Only immediate running-cost
    terms in [start,stop) enter the objective; earlier actions may affect those
    terms through the state and actor feedback, while later actions get zero
    credit. ``selected_rows`` is zero/one with at most one selected trajectory
    per original group, as for original_cost_cotangents. A selected complete
    trace must cover the entire requested window. No teacher return or action
    target is introduced, and neither a transition nor a controller is called.
    """
    start, stop = _running_window_bounds(task_dynamics, start, stop)
    pre = _finite(pre_states, 'pre-action states')
    if pre.ndim != 3 or pre.shape[-1] != 2 or min(pre.shape[:2]) < 1:
        raise ValueError('Pre-action states must have shape (B,H,2), B,H >= 1')
    selected = _finite(selected_rows, 'fixed running-window row weights', (len(pre),))
    if (np.any((selected != 0.) & (selected != 1.)) or selected.sum() > 1.
            or np.any((selected != 0.) & (pre[:, 0, 0] > start))):
        raise ValueError('At most one complete row covering the whole window may be selected')
    # Reuse the unchanged complete-trace/component validation and branch
    # diagnostics. The zeros are unused beta-zero arithmetic, not teacher data.
    result = ending_cotangents(task_dynamics, pre, actions, components,
        np.zeros((len(pre), 4)), policy_dy, beta=0., return_scale=1., scales=np.ones(4))
    action = np.asarray(actions, dtype=np.float64)
    if action.ndim == 3:
        action = action[..., 0]
    component = np.asarray(components, dtype=np.float64)
    sensitivity = np.asarray(policy_dy, dtype=np.float64)
    clock, position = pre[..., 0], pre[..., 1]
    window = (clock >= start) & (clock < stop)
    local_weight = window * selected[:, None]
    movable = clock < task_dynamics.gate
    unclipped = position + task_dynamics.move * action
    interior = movable & (unclipped > -1.) & (unclipped < 1.)
    f_y = np.where(movable, interior.astype(np.float64), 1.)
    f_a = task_dynamics.move * interior
    greed_y_next = -.5 * task_dynamics.greed
    l_y = np.where(movable, local_weight * greed_y_next * f_y, 0.)
    l_a = local_weight * (2. * task_dynamics.effort * action * movable + greed_y_next * f_a)
    cotangent = np.zeros(pre.shape[:2], dtype=np.float64)
    adjoint = np.zeros(len(pre), dtype=np.float64)
    for step in range(pre.shape[1] - 1, -1, -1):
        cotangent[:, step] = l_a[:, step] + f_a[:, step] * adjoint
        adjoint = (l_y[:, step] + f_y[:, step] * adjoint
            + sensitivity[:, step] * cotangent[:, step])
    nominal = np.array([1., 1., 0., 0.])
    objective = (component[..., :2].sum(axis=-1) * window).sum(axis=1) * selected
    if not all(np.isfinite(value).all() for value in (cotangent, adjoint, objective)):
        raise FloatingPointError('Nonfinite running-window feedback credit')
    result.update(action_cotangents=cotangent,
        component_weights=selected[:, None] * nominal,
        component_masks=local_weight[..., None] * nominal,
        nominal_component_mask=nominal, window_mask=window,
        objective_per_episode=objective, cost_window=[start, stop],
        objective_kind='original_running_window_cost')
    return result


def _owned_query_records(archive, query_ids, owner):
    """Bind each supplied result to a paid complete deterministic execution."""
    requested, completed, failed = {}, {}, set()
    selected = set(query_ids)
    for line in archive.ledger_path.read_text().splitlines():
        record = json.loads(line)
        query = record.get('query_id')
        if query not in selected:
            continue
        kind = record['kind']
        if kind == 'query_requested':
            if query in requested:
                raise ValueError('Duplicate actor-query request')
            requested[query] = record
        elif kind == 'query_complete':
            if query in completed:
                raise ValueError('Duplicate actor-query completion')
            completed[query] = record
        elif kind == 'query_error':
            failed.add(query)
    if set(requested) != selected or set(completed) != selected or failed:
        raise ValueError('Complete local actor-query evidence required')
    for record in requested.values():
        metadata = record.get('metadata', {})
        if (record.get('mode') != 'executed' or record.get('domain') != 'scalar'
                or metadata.get('actor_hash') != owner
                or metadata.get('deterministic') is not True
                or metadata.get('complete_continuation') is not True
                or metadata.get('semantics') != 'actor_owned_complete'
                or metadata.get('actor', {}).get('family') != 0):
            raise ValueError('Owned uninterrupted deterministic actor execution required')
    return requested, completed


def _saved_arrays(archive, reference):
    """Read an existing evidence packet after checking its frozen byte hash."""
    path = archive.path / reference['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Actor-query evidence hash mismatch')
    with np.load(path, allow_pickle=False) as packet:
        return {name: packet[name].copy() for name in packet.files}


def feedback_gradient(archive, actor, task, groups, config, *, beta=None, tag,
                      ending_row=None, cost_row=None, cost_window=None,
                      budget_free=False):
    """Compute two metered VJPs on already executed deterministic actor traces.

    Forward batches retain the live executor's per-time batch shape, avoiding a
    different flattened GEMM. The first VJP supplies all input sensitivities.
    The second supplies the complete named parameter gradient from detached
    analytic action cotangents. There is no optimizer and no simulator call.
    All parameters retain their positions; unused deterministic ``log_std`` and
    unused input weights receive explicit zeros. Trajectories are weighted
    equally across all groups, independent of their remaining horizons.

    With ``ending_row=(group_index, row)`` the objective becomes that one actual
    complete terminal position. All groups keep their original forward batch
    shapes, while one selected terminal adjoint supplies scalar endpoint credit
    without trajectory averaging. Exact-context teacher evidence remains bound
    and validated, but contributes neither a cost nor action target in this mode.

    With ``cost_row=(group_index, row)`` the objective is instead that row's
    original complete sum of costs, with beta zero and return scale one. No
    trajectory averaging is applied. The supplied paid query is never sliced
    or relabelled: every original row keeps its forward batch shape and cost,
    while all other row cotangents are zero. ``cost_row`` and ``ending_row``
    are mutually exclusive; an explicit nonzero beta is rejected.

    ``cost_window=(start,stop)`` is valid only with ``cost_row``. It selects
    original effort+greed costs whose absolute pre-action clocks lie in that
    fixed interval, retaining the complete trajectory and its feedback credit.
    An explicit query inference contract is replayed rather than flattened.

    With ``budget_free=True``, every context has budget=None and an exact
    prewritten ordinary/acquired-state metadata sidecar. Beta must be zero and
    return_scale one. With no selected row, this computes the equal-trajectory
    mean of original complete costs across arbitrary complete start clocks;
    ending_row and original cost_row remain available without averaging.
    Teacher envelopes and running windows are not admitted in this mode.
    """
    if ending_row is not None and cost_row is not None:
        raise ValueError('Select either a terminal row or an original-cost row')
    if type(budget_free) is not bool or (budget_free and cost_window is not None):
        raise ValueError('Explicit budget-free mode does not admit a running-cost window')
    if cost_row is not None and beta is not None and float(beta) != 0.:
        raise ValueError('Original-cost row credit requires beta zero')
    if cost_window is not None:
        if (cost_row is None or ending_row is not None
                or not isinstance(cost_window, (tuple, list)) or len(cost_window) != 2):
            raise ValueError('A running window requires one original-cost row and (start,stop)')
        cost_window = _running_window_bounds(task.dynamics, *cost_window)
    # The NumPy cotangent primitive stays usable without loading Torch or the
    # learner. Importing this module performs no neural or optimizer work.
    import torch
    from .learning import work_request
    from .policy import actor_forward, policy_hash

    groups = list(groups)
    if not groups:
        raise ValueError('At least one deterministic actor-owned group required')
    selected_beta = 0. if cost_row is not None else config.beta if beta is None else float(beta)
    if not np.isfinite(selected_beta) or selected_beta < 0:
        raise ValueError('Nonnegative finite envelope weight required')
    if budget_free and (selected_beta != 0. or config.return_scale != 1.):
        raise ValueError('Budget-free original credit requires beta zero and return scale one')
    owner = policy_hash(actor)
    queries = [group['result'].query_id for group in groups]
    if len(set(queries)) != len(queries):
        raise ValueError('Duplicate actor-owned query in gradient groups')
    requests, completions = _owned_query_records(archive, queries, owner)
    named_parameters = list(actor.named_parameters())
    if not named_parameters or any(not parameter.requires_grad for _, parameter in named_parameters):
        raise ValueError('All actor parameters must be differentiable')
    if actor.action_scale.dtype != torch.float32:
        raise ValueError('Credit must match the deployed float32 actor arithmetic')

    prepared = []
    trajectory_count = 0
    context_metadata = []
    teacher_sources = {}

    def budget_free_arrays(reference):
        from pathlib import Path
        if not isinstance(reference, dict):
            raise ValueError('A byte-bound local budget-free packet is required')
        relative = Path(reference['file']); path = archive.path / relative
        if (relative.is_absolute() or '..' in relative.parts
                or path.stat().st_size != reference['bytes']):
            raise ValueError('Invalid budget-free evidence descriptor')
        return _saved_arrays(archive, reference)

    # Validate all ownership/context/state bindings before doing neural work.
    for group in groups:
        raw, contexts = group['result'], list(group['contexts'])
        pre = _finite(raw.pre_states, 'saved pre-action states')
        if pre.ndim != 3 or pre.shape[-1] != 2 or min(pre.shape[:2]) < 1:
            raise ValueError('Saved pre-action states must have shape (B,H,2)')
        batch, horizon = pre.shape[:2]
        states = _finite(raw.states, 'saved post-action states', pre.shape)
        actions = _finite(raw.actions, 'saved executed actions', (batch, horizon, 1))
        components = _finite(raw.components, 'saved components', (batch, horizon, 4))
        read_arrays = budget_free_arrays if budget_free else _saved_arrays
        if budget_free:
            saved_inputs = read_arrays(requests[raw.query_id]['inputs'])
            saved_result = read_arrays(completions[raw.query_id]['result'])
        else:
            saved_inputs = read_arrays(archive, requests[raw.query_id]['inputs'])
            saved_result = read_arrays(archive, completions[raw.query_id]['result'])
        if (raw.actor_owner != owner or len(contexts) != batch
                or not np.array_equal(states[:, :-1], pre[:, 1:])
                or not np.array_equal(saved_inputs['initial_states'], pre[:, 0])
                or any(not np.array_equal(value, saved_result[name]) for name, value in
                       (('pre_states', pre), ('states', states), ('actions', actions),
                        ('components', components)))
                or np.any(states[:, -1, 0] != task.config.T)
                or np.any(states[..., 0] != pre[..., 0] + 1.)
                or np.any(pre[:, -1, 0] != task.config.T - 1)
                or np.any(pre[..., 0] != pre[:, :1, 0] + np.arange(horizon)[None, :])
                or requests[raw.query_id]['requested_controls'] != batch * horizon
                or completions[raw.query_id]['completed_controls'] != batch * horizon):
            raise ValueError('Aligned complete generating-actor traces required')
        if budget_free:
            metadata = requests[raw.query_id]['metadata']
            binding = budget_free_arrays(group['bindings'])
            actor_metadata = metadata['actor']
            # Verify the generating checkpoint's recorded bytes, without
            # constructing another actor or changing inference batch shapes.
            actor_reference = actor_metadata['raw']
            from pathlib import Path
            actor_relative = Path(actor_reference['file']); actor_path = archive.path / actor_relative
            if (actor_relative.is_absolute() or '..' in actor_relative.parts
                    or actor_path.stat().st_size != actor_reference['bytes']
                    or hashlib.sha256(actor_path.read_bytes()).hexdigest() != actor_reference['sha256']
                    or actor_metadata.get('actor_hash') != owner
                    or metadata.get('budget_free') is not True
                    or metadata.get('contexts_ref') != group['bindings']
                    or json.loads(str(saved_inputs['metadata_json'])) != metadata
                    or not np.array_equal(binding['initial_states'], pre[:, 0])
                    or np.any(np.abs(actions) > 1.) or np.any(components < 0.)
                    or np.any(np.abs(pre[..., 1]) > 1.) or np.any(np.abs(states[..., 1]) > 1.)
                    or np.any(pre[..., 0] != np.floor(pre[..., 0])) or np.any(pre[..., 0] < 0)
                    or not np.array_equal(saved_result['stage_costs'], components.sum(axis=-1))
                    or not np.array_equal(saved_result['totals'], components.sum(axis=-1).sum(axis=1))):
                raise ValueError('Exact budget-free query, checkpoint, and context sidecar bindings required')
            rows = json.loads(str(binding['context_metadata_json']))
            if not isinstance(rows, list) or len(rows) != batch:
                raise ValueError('One prewritten budget-free metadata row per trajectory required')
            for row, (context, entry) in enumerate(zip(contexts, rows)):
                key = EndingSnapshot(pre[row, 0], task.task_hash).context_id
                source = entry.get('source') if isinstance(entry, dict) else None
                if (not isinstance(entry, dict)
                        or set(entry) != {'family', 'task_hash', 'context_key', 'source'}
                        or type(entry['family']) is not int or entry['family'] != 0
                        or entry['task_hash'] != task.task_hash or entry['context_key'] != key
                        or context.budget is not None or not np.array_equal(context.state, pre[row, 0])
                        or context.metadata != entry or not isinstance(source, dict)
                        or (('root' in source) == ('teacher_ref' in source))
                        or ('root' in source and (source.get('kind') != 'ordinary_reset'
                            or type(source['root']) is not int
                            or source['root'] < 0 or pre[row, 0, 0] != 0.))
                        or ('teacher_ref' in source and (source.get('kind') != 'executed_teacher_state'
                            or not isinstance(source['teacher_ref'], dict)
                            or type(source.get('elapsed')) is not int or source['elapsed'] < 0
                            or source['elapsed'] != pre[row, 0, 0]))):
                    raise ValueError('Budget-free rows must bind exact state/task/family and actual source provenance')
                provenance = {}
                if 'teacher_ref' in source:
                    reference = source['teacher_ref']
                    source_key = json.dumps(reference, sort_keys=True)
                    if source_key not in teacher_sources:
                        teacher = budget_free_arrays(reference)
                        teacher_meta = json.loads(str(teacher['metadata_json']))
                        teacher_states = _finite(teacher['states'], 'actual teacher states', (task.config.T + 1, 2))
                        teacher_actions = _finite(teacher['actions'], 'actual teacher actions', (task.config.T, 1))
                        teacher_components = _finite(teacher['components'], 'actual teacher components', (task.config.T, 4))
                        teacher_queries = teacher_meta.get('query_ids')
                        if (teacher_meta.get('task_hash') != task.task_hash
                                or type(teacher_meta.get('family')) is not int or teacher_meta['family'] != 0
                                or teacher_meta.get('complete') is not True
                                or teacher_meta.get('semantics') not in ('executed_replanning', 'executed_prefix_then_actor')
                                or not isinstance(teacher_meta.get('controller_hash'), str) or not teacher_meta['controller_hash']
                                or type(teacher_meta.get('root')) is not int or teacher_meta['root'] < 0
                                or not isinstance(teacher_queries, list) or len(teacher_queries) != task.config.T
                                or any(not isinstance(q, str) or not q for q in teacher_queries)
                                or len(set(teacher_queries)) != len(teacher_queries)
                                or not np.array_equal(teacher_states[:, 0], np.arange(task.config.T + 1))
                                or np.any(np.abs(teacher_states[:, 1]) > 1.)
                                or np.any(np.abs(teacher_actions) > 1.) or np.any(teacher_components < 0.)
                                or type(teacher_meta.get('capture')) is not bool
                                or teacher_meta['capture'] != bool(teacher_states[-1, 1] <= task.dynamics.ending_y)):
                            raise ValueError('Byte-bound complete executed teacher source required for acquired states')
                        teacher_sources[source_key] = (teacher_states, teacher_meta)
                    teacher_states, teacher_meta = teacher_sources[source_key]
                    elapsed = source['elapsed']
                    if not 0 <= elapsed < task.config.T or not np.array_equal(teacher_states[elapsed], pre[row, 0]):
                        raise ValueError('Acquired context differs from the bound actual teacher state at elapsed clock')
                    provenance = dict(source_suffix_query_ids=teacher_meta['query_ids'][elapsed:],
                        source_controller_hash=teacher_meta['controller_hash'], source_semantics=teacher_meta['semantics'],
                        source_guiding_policy_hash=teacher_meta.get('guiding_policy_hash'))
                context_metadata.append(dict(entry, state=pre[row, 0].tolist(),
                    query_id=raw.query_id, query_row=row, **provenance))
            budgets = None
        else:
            budgets = []
            for row, context in enumerate(contexts):
                budget = context.budget
                if (not np.array_equal(context.state, pre[row, 0])
                        or EndingSnapshot(pre[row, 0], task.task_hash).context_id != budget.context_key
                        or budget.task_hash != task.task_hash or budget.family != 0
                        or budget.names != ('effort', 'greed', 'detour', 'ending')):
                    raise ValueError('Exact teacher context/task/family/component binding required')
                budgets.append(budget.costs)
        prepared.append(dict(raw=raw, pre=pre, actions=actions, components=components,
                             budgets=None if budget_free else np.stack(budgets), observations=[], action_tensors=[],
                             inference_contract=requests[raw.query_id]['metadata']['actor'].get('inference_contract', 'native')))
        trajectory_count += batch

    selected_row = cost_row if cost_row is not None else ending_row
    if selected_row is not None:
        if (not isinstance(selected_row, (tuple, list)) or len(selected_row) != 2
                or any(isinstance(index, (bool, np.bool_))
                       or not isinstance(index, (int, np.integer)) for index in selected_row)):
            raise ValueError('A selected row requires integer (group_index, row)')
        group_index, row = (int(index) for index in selected_row)
        if cost_row is not None:cost_row = (group_index, row)
        else:ending_row = (group_index, row)
        if (not 0 <= group_index < len(prepared)
                or not 0 <= row < len(prepared[group_index]['pre'])):
            raise ValueError('Selected row lies outside the complete groups')
        for index, item in enumerate(prepared):
            item['selected_rows'] = np.zeros(len(item['pre']), dtype=np.float64)
            if index == group_index:
                item['selected_rows'][row] = 1.
                if cost_window is not None and item['pre'][row, 0, 0] > cost_window[0]:
                    raise ValueError('The selected complete query must cover the entire running window')
            if ending_row is not None and not np.array_equal(item['raw'].states[:, -1, 1], item['pre'][:, -1, 1]):
                raise ValueError('The recorded final position must be unchanged after the gate')

    row_metadata = {} if cost_row is None else dict(
        objective_kind='original_complete_cost', cost_row=list(cost_row),
        selected_query_id=queries[cost_row[0]], selected_query_row=cost_row[1],
        effective_beta=0., effective_return_scale=1., selected_trajectory_count=1)
    if cost_window is not None:
        row_metadata.update(objective_kind='original_running_window_cost',
            cost_window=list(cost_window), window_components=['effort', 'greed'])
    if budget_free:
        row_metadata.update(budget_free=True, teacher_budget_supplied=False,
            effective_beta=0., effective_return_scale=1.,
            objective_kind='terminal_position' if ending_row is not None else
                'original_complete_cost' if cost_row is not None else 'original_complete_cost_mean')

    all_observations, all_actions = [], []
    for item in prepared:
        pre = item['pre']
        for step in range(pre.shape[1]):
            observation = task.observation(EndingSnapshot(pre[:, step], task.task_hash))
            obs = torch.as_tensor(observation, dtype=torch.float32,
                                  device=actor.action_scale.device).detach().requires_grad_(True)
            attach_meter(actor, archive, 'deterministic_feedback_credit')
            mean, _ = actor_forward(actor, obs, 0, inference_contract=item['inference_contract'])
            action = actor._bounded(mean)
            observed_action = action.detach().cpu().numpy()
            matches = (np.array_equal(observed_action, item['actions'][:, step])
                if item['inference_contract'] != 'native' else
                np.allclose(observed_action, item['actions'][:, step], atol=1e-6, rtol=0.))
            if not matches:
                raise ValueError('Recomputed bounded means differ from recorded deterministic actions')
            item['observations'].append(obs)
            item['action_tensors'].append(action)
            all_observations.append(obs)
            all_actions.append(action)

    request = work_request(archive, 'backward', purpose='deterministic_credit_input_vjp',
                           tag=tag, actor_hash=owner, query_ids=queries, **row_metadata)
    input_vjps = torch.autograd.grad(tuple(all_actions), tuple(all_observations),
                                    grad_outputs=tuple(torch.ones_like(a) for a in all_actions),
                                    retain_graph=True, create_graph=False)
    archive.append('backward_complete', request_id=request,
                   purpose='deterministic_credit_input_vjp', tag=tag, **row_metadata)
    cursor = 0
    credit = []
    action_weights = []
    for item in prepared:
        horizon = item['pre'].shape[1]
        # observation[0]=y; all other coordinates depend only on the fixed clock.
        policy_dy = np.stack([value.detach().cpu().numpy()[:, 0]
                              for value in input_vjps[cursor:cursor + horizon]], axis=1)
        cursor += horizon
        if cost_row is not None:
            if cost_window is None:
                result = original_cost_cotangents(task.dynamics, item['pre'], item['actions'],
                    item['components'], policy_dy, item['selected_rows'], budget_free=budget_free)
            else:
                result = running_window_cotangents(task.dynamics, item['pre'], item['actions'],
                    item['components'], policy_dy, item['selected_rows'],
                    start=cost_window[0], stop=cost_window[1])
        elif ending_row is None:
            if budget_free:
                result = original_cost_cotangents(task.dynamics, item['pre'], item['actions'],
                    item['components'], policy_dy, np.ones(len(item['pre'])),
                    allow_multiple=True, budget_free=True)
            else:
                result = ending_cotangents(task.dynamics, item['pre'], item['actions'],
                                           item['components'], item['budgets'], policy_dy,
                                           beta=selected_beta, return_scale=config.return_scale,
                                           scales=config.scales)
        else:
            result = terminal_position_cotangents(task.dynamics, item['pre'],
                item['actions'], policy_dy, item['selected_rows'])
            result['objective_per_episode'] = (item['selected_rows']
                                               * item['raw'].states[:, -1, 1])
        credit.append(result)
        for step, action in enumerate(item['action_tensors']):
            # Coefficients are fixed cotangents: differentiating the adjoint
            # again would introduce spurious higher-order terms.
            action_weights.append(torch.as_tensor(
                result['action_cotangents'][:, step, None]
                / (trajectory_count if ending_row is None and cost_row is None else 1),
                dtype=action.dtype, device=action.device))

    request = work_request(archive, 'backward', purpose='deterministic_credit_parameter_vjp',
                           tag=tag, actor_hash=owner, query_ids=queries, **row_metadata)
    parameter_vjps = torch.autograd.grad(tuple(all_actions),
        tuple(parameter for _, parameter in named_parameters),
        grad_outputs=tuple(action_weights), allow_unused=True, create_graph=False)
    archive.append('backward_complete', request_id=request,
                   purpose='deterministic_credit_parameter_vjp', tag=tag, **row_metadata)
    gradient = np.concatenate([
        np.zeros(parameter.numel(), dtype=np.float64) if value is None
        else value.detach().cpu().numpy().astype(np.float64).reshape(-1)
        for (_, parameter), value in zip(named_parameters, parameter_vjps)])
    if not np.isfinite(gradient).all() or policy_hash(actor) != owner:
        raise FloatingPointError('Nonfinite gradient or changed generating actor')
    result = dict(
        gradient=gradient,
        action_cotangents=[result['action_cotangents'] for result in credit],
        component_weights=[result['component_weights'] for result in credit],
        objective_per_episode=[result['objective_per_episode'] for result in credit],
        branch_margins=[result['branch_margins'] for result in credit],
        objective=float(np.concatenate([result['objective_per_episode'] for result in credit]).mean()),
        actor_owner=owner, query_ids=queries,
        parameter_names=[name for name, _ in named_parameters],
        parameter_shapes=[tuple(parameter.shape) for _, parameter in named_parameters],
        trajectory_count=trajectory_count,
        beta=selected_beta,
        derivative='smooth arithmetic on recorded branch; complete feedback finite differences required',
    )
    if ending_row is not None:
        result.update(objective=float(np.concatenate(result['objective_per_episode']).sum()),
            objective_kind='terminal_position', ending_row=list(ending_row),
            derivative='smooth arithmetic terminal position on recorded clipping branch; complete feedback finite differences required')
    elif cost_row is not None:
        result.update(row_metadata,
            objective=float(np.concatenate(result['objective_per_episode']).sum()),
            selected_rows=[item['selected_rows'].copy() for item in prepared],
            derivative='branch-local original complete cost; discrete task-mode jumps are not differentiated; complete feedback qualification required')
        if cost_window is not None:
            result.update(component_masks=[item['component_masks'] for item in credit],
                window_mask=[item['window_mask'] for item in credit],
                nominal_component_mask=credit[cost_row[0]]['nominal_component_mask'],
                derivative='Original effort+greed in a fixed absolute-clock window, with complete actor feedback; smooth-arithmetic clipping convention requires complete-policy qualification')
    if budget_free:
        result.update(row_metadata, context_metadata=context_metadata,
            bindings=[group['bindings'] for group in groups],
            component_totals=[item['components'].sum(axis=1) for item in prepared],
            component_names=['effort', 'greed', 'detour', 'ending'],
            neural_forward_rows=sum(len(item['pre']) * item['pre'].shape[1] for item in prepared),
            backward_calls=2,
            derivative='Budget-free branch-local original complete-feedback credit; discrete task-mode jumps not differentiated; complete-policy qualification required'
                if ending_row is None else result['derivative'])
    return result


def hybrid_feedback_gradient(archive, actor, task, groups, config, *, tag, beta=None):
    """Three metered VJPs on complete current-actor stochastic executions.

    A group contains one original ``result``, its ``bindings`` NPZ descriptor,
    and one expanded context dictionary per query row. Context dictionaries
    contain context_id, state, task_hash, family, replicates, repeat_index,
    beta and budget. An ordinary beta-zero context may have budget=None;
    positive beta requires an exact-context OutcomeBudget. Distinct context
    IDs remain distinct even when their states coincide. All repeats for one
    ID must occur within one original query, with indices 0 through R-1.
    An explicit beta=0 is an off-envelope diagnostic on these same outcomes;
    original sidecar/budget metadata is still validated and retained unchanged.

    The prewritten binding packet contains initial_states and a JSON row list
    in context_metadata_json. The query's contexts_ref must equal that packet
    descriptor. Its JSON rows omit state and contain the computed context_key
    and the complete budget metadata, including costs and original query IDs.

    Effort/greed envelope terms receive complete-feedback pathwise credit.
    Only detour/ending envelope terms receive same-context leave-one-out score
    credit, and only pre-gate actions enter that score. Observations are saved,
    detached leaves; the score is a partial parameter derivative at fixed
    observations and fixed recorded latents. No total-loss score is added.

    The forward latent is EXACTLY the recorded latent, with local derivative
    supplied by z_saved+(mean-stop(mean))+(std-stop(std))*eps and detached
    eps=(z_saved-stop(mean))/stop(std). This is an explicit smooth-rounding
    convention, not a derivative of float32 quantization. Continuous frozen-
    noise complete-feedback FD and conditional score qualification remain
    required. Native inference shapes/precision are recorded; deployment still
    requires fresh actual deterministic qualification, not an assumed bitwise
    invariance across different inference batch shapes.
    No simulation, random draw, optimizer, parameter update or packet write is
    performed. The caller must durably archive this returned packet.
    """
    import torch
    from pathlib import Path
    from .coupling import OutcomeBudget
    from .learning import work_request
    from .policy import policy_hash

    groups=list(groups)
    if not groups:raise ValueError('Stochastic owned context groups required')
    if beta is not None and float(beta)!=0.:raise ValueError('Only the explicit beta-zero diagnostic override is supported')
    beta_override=None if beta is None else 0.
    owner=policy_hash(actor);queries=[g['result'].query_id for g in groups]
    if len(set(queries))!=len(queries):raise ValueError('Duplicate stochastic query')
    named=list(actor.named_parameters())
    if (not named or any(not p.requires_grad for _,p in named)
            or actor.action_scale.dtype!=torch.float32 or actor.act_dim!=1):
        raise ValueError('Differentiable scalar float32 Gaussian actor required')
    scale=_finite(config.scales,'component scales',(4,))
    if (np.any(scale<=0) or not np.isfinite([config.beta,config.return_scale]).all()
            or config.beta<0 or config.return_scale<=0):raise ValueError('Invalid envelope settings')
    private_rng=actor._generator.get_state().clone();global_rng=torch.get_rng_state().clone()
    requested={};completed={};failed=set();selected=set(queries)
    for line in archive.ledger_path.read_text().splitlines():
        event=json.loads(line);q=event.get('query_id')
        if q not in selected:continue
        target=requested if event['kind']=='query_requested' else completed if event['kind']=='query_complete' else None
        if target is not None:
            if q in target:raise ValueError('Duplicate stochastic query evidence')
            target[q]=event
        elif event['kind']=='query_error':failed.add(q)
    if set(requested)!=selected or set(completed)!=selected or failed:
        raise ValueError('Resolved complete local stochastic evidence required')

    def packet(ref):
        relative=Path(ref['file']);path=archive.path/relative
        if (relative.is_absolute() or '..' in relative.parts or path.stat().st_size!=ref['bytes']):
            raise ValueError('Invalid stochastic evidence descriptor')
        return _saved_arrays(archive,ref)

    prepared=[];contexts=[];seen_contexts=set();names=('effort','greed','detour','ending')
    for gi,group in enumerate(groups):
        raw=group['result'];rows=list(group['contexts']);request=requested[raw.query_id]
        meta=request.get('metadata',{});actor_meta=meta.get('actor',{})
        if (request.get('mode')!='executed' or request.get('domain')!='scalar'
                or meta.get('actor_hash')!=owner or actor_meta.get('actor_hash')!=owner
                or actor_meta.get('family')!=0 or meta.get('deterministic') is not False
                or meta.get('complete_continuation') is not True or meta.get('semantics')!='actor_owned_complete'
                or actor_meta.get('sampling')!='numpy-PCG64-noise-float32-latent'
                or type(meta.get('seed')) is not int or meta.get('contexts_ref')!=group['bindings']):
            raise ValueError('Current stochastic actor and prewritten contexts must own the query')
        inp=packet(request['inputs']);saved=packet(completed[raw.query_id]['result']);binding=packet(group['bindings'])
        pre=_finite(raw.pre_states,'pre-action states')
        if pre.ndim!=3 or pre.shape[-1]!=2 or min(pre.shape[:2])<1:raise ValueError('Nonempty (B,H,2) traces required')
        batch,horizon=pre.shape[:2]
        states=_finite(raw.states,'post-action states',pre.shape)
        actions=_finite(raw.actions,'actions',(batch,horizon,1))
        components=_finite(raw.components,'components',(batch,horizon,4))
        latent=_finite(raw.latent,'recorded latents',(batch,horizon,1))
        logps=_finite(raw.log_probs,'recorded log densities',(batch,horizon))
        arrays=dict(pre_states=pre,states=states,actions=actions,components=components,latent=latent,log_probs=logps,
            scores=_finite(raw.scores,'unused recorded scores',(batch,horizon,0)),
            stage_costs=_finite(raw.stage_costs,'stage costs',(batch,horizon)),
            totals=_finite(raw.totals,'complete costs',(batch,)))
        if (raw.actor_owner!=owner or len(rows)!=batch or np.any(np.abs(actions)>1.) or np.any(components<0.)
                or any(not np.array_equal(value,saved[key]) for key,value in arrays.items())
                or json.loads(str(inp['metadata_json']))!=meta
                or not np.array_equal(inp['initial_states'],pre[:,0])
                or not np.array_equal(binding['initial_states'],pre[:,0])
                or np.any(pre[...,0]!=np.floor(pre[...,0])) or np.any(pre[...,0]<0)
                or np.any(pre[...,0]>=task.config.T) or np.any(np.abs(pre[...,1])>1.)
                or np.any(np.abs(states[...,1])>1.)
                or not np.array_equal(states[:,:-1],pre[:,1:])
                or not np.array_equal(states[...,0],pre[...,0]+1.)
                or not np.array_equal(pre[...,0],pre[:,:1,0]+np.arange(horizon)[None,:])
                or np.any(pre[:,-1,0]!=task.config.T-1) or np.any(states[:,-1,0]!=task.config.T)
                or not np.array_equal(states[:,-1,1],pre[:,-1,1])
                or not np.array_equal(arrays['stage_costs'],components.sum(axis=-1))
                or not np.array_equal(arrays['totals'],arrays['stage_costs'].sum(axis=1))
                or request['requested_controls']!=batch*horizon
                or completed[raw.query_id]['completed_controls']!=batch*horizon):
            raise ValueError('Byte-bound uninterrupted complete stochastic traces required')
        totals=components.sum(axis=1);budgets=totals.copy();betas=np.empty(batch);metadata=[];local={}
        for ri,context in enumerate(rows):
            if not isinstance(context,dict):raise ValueError('Expanded context dictionaries required')
            state=_finite(context['state'],'context state',(2,));key=EndingSnapshot(state,task.task_hash).context_id
            cid=context['context_id'];replicates=context['replicates'];repeat=context['repeat_index'];beta=float(context['beta'])
            if (not isinstance(cid,str) or not cid or context['task_hash']!=task.task_hash
                    or type(context['family']) is not int or context['family']!=0
                    or type(replicates) is not int or replicates<2 or type(repeat) is not int
                    or not 0<=repeat<replicates or not np.array_equal(state,pre[ri,0])
                    or not np.isfinite(beta) or beta not in (0.,config.beta)):
                raise ValueError('Exact fixed context, repeat and envelope metadata required')
            budget=context['budget'];budget_meta=None
            if budget is None:
                if beta!=0.:raise ValueError('Positive envelope weight needs a real teacher budget')
            else:
                if (not isinstance(budget,OutcomeBudget) or budget.context_key!=key
                        or budget.task_hash!=task.task_hash or budget.family!=0 or budget.names!=names):
                    raise ValueError('Teacher envelope does not own this exact context')
                budgets[ri]=_finite(budget.costs,'teacher component budget',(4,))
                budget_meta=dict(costs=budgets[ri].tolist(),names=list(budget.names),context_key=key,
                    task_hash=budget.task_hash,family=budget.family,teacher_controller_hash=budget.teacher_controller_hash,
                    query_ids=list(budget.query_ids))
            entry=dict(context_id=cid,context_key=key,task_hash=task.task_hash,family=0,
                replicates=replicates,repeat_index=repeat,beta=beta,budget=budget_meta)
            metadata.append(entry);betas[ri]=beta if beta_override is None else 0.;local.setdefault(cid,[]).append(ri)
        if json.loads(str(binding['context_metadata_json']))!=metadata:
            raise ValueError('Context dictionaries differ from the prewritten query binding')
        item=dict(raw=raw,pre=pre,actions=actions,components=components,latent=latent,logps=logps,
            totals=totals,budgets=budgets,betas=betas,row_metadata=metadata,contexts=[],observations=[],actions_graph=[],logps_graph=[],noise=[])
        for cid,indices in local.items():
            first=metadata[indices[0]];base={k:v for k,v in first.items() if k!='repeat_index'}
            if (cid in seen_contexts or len(indices)!=first['replicates']
                    or sorted(metadata[r]['repeat_index'] for r in indices)!=list(range(len(indices)))
                    or any({k:v for k,v in metadata[r].items() if k!='repeat_index'}!=base for r in indices)):
                raise ValueError('IID repeat membership may not merge or split context identities')
            seen_contexts.add(cid)
            record=dict(base,query_id=raw.query_id,group_index=gi,row_indices=indices,state=pre[indices[0],0].tolist(),
                effective_beta=base['beta'] if beta_override is None else 0.)
            contexts.append(record);item['contexts'].append(record)
        prepared.append(item)

    count=len(contexts);all_obs=[];all_actions=[];all_logps=[]
    for item in prepared:
        for step in range(item['pre'].shape[1]):
            observation=task.observation(EndingSnapshot(item['pre'][:,step],task.task_hash))
            obs=torch.as_tensor(observation,dtype=torch.float32,device=actor.action_scale.device).detach().requires_grad_(True)
            attach_meter(actor,archive,'hybrid_feedback_credit');mean,logstd=actor(obs,0);std=logstd.exp()
            saved=torch.as_tensor(item['latent'][:,step],dtype=mean.dtype,device=mean.device).detach()
            noise=((saved-mean.detach())/std.detach()).detach()
            if not bool(torch.isfinite(noise).all()):raise FloatingPointError('Nonfinite reconstructed sampled noise')
            reparameterized=saved+(mean-mean.detach())+(std-std.detach())*noise
            action=actor._bounded(reparameterized)
            logp,_,_=actor._evaluated(mean,logstd,saved)
            if (not np.array_equal(action.detach().cpu().numpy(),item['actions'][:,step])
                    or not np.array_equal(logp.detach().cpu().numpy(),item['logps'][:,step])):
                raise ValueError('Native-batch recorded actions/log densities do not reproduce exactly')
            item['observations'].append(obs);item['actions_graph'].append(action);item['logps_graph'].append(logp)
            item['noise'].append(noise.detach().cpu().numpy())
            all_obs.append(obs);all_actions.append(action);all_logps.append(logp)

    def vjp(purpose,outputs,inputs,weights,retain_graph):
        request=work_request(archive,'backward',purpose=purpose,tag=tag,actor_hash=owner,query_ids=queries)
        try:
            result=torch.autograd.grad(tuple(outputs),tuple(inputs),grad_outputs=tuple(weights),
                retain_graph=retain_graph,create_graph=False,allow_unused=True)
        except BaseException as error:
            archive.append('backward_error',request_id=request,purpose=purpose,tag=tag,error_type=type(error).__name__);raise
        archive.append('backward_complete',request_id=request,purpose=purpose,tag=tag)
        return result

    input_vjp=vjp('hybrid_credit_input_vjp',all_actions,all_obs,[torch.ones_like(a) for a in all_actions],True)
    path_weights=[];score_weights=[];cursor=0;details=[]
    for item in prepared:
        batch,horizon=item['pre'].shape[:2]
        sensitivity=np.stack([np.zeros(batch) if value is None else value.detach().cpu().numpy()[:,0]
            for value in input_vjp[cursor:cursor+horizon]],axis=1);cursor+=horizon
        cotangents=np.zeros((batch,horizon));weights=np.zeros((batch,4));margins={};row_weights=np.zeros(batch)
        terms=item['totals']/config.return_scale
        terms=terms+item['betas'][:,None]*np.square(np.maximum((item['totals']-item['budgets'])/scale,0.))
        continuous=terms[:,:2].sum(axis=1);discrete=terms[:,2:].sum(axis=1)
        baseline=np.zeros(batch);advantage=np.zeros(batch)
        for context in item['contexts']:
            rows=np.asarray(context['row_indices']);beta=context['effective_beta'];R=len(rows)
            credit=ending_cotangents(task.dynamics,item['pre'][rows],item['actions'][rows],item['components'][rows],
                item['budgets'][rows],sensitivity[rows],beta=beta,return_scale=config.return_scale,scales=scale)
            cotangents[rows]=credit['action_cotangents'];weights[rows]=credit['component_weights'];row_weights[rows]=1./(count*R)
            for key,value in credit['branch_margins'].items():
                if key not in margins:margins[key]=np.empty((batch,*value.shape[1:]),dtype=value.dtype)
                margins[key][rows]=value
            baseline[rows]=(discrete[rows].sum()-discrete[rows])/(R-1);advantage[rows]=discrete[rows]-baseline[rows]
            context.update(discrete_outcomes_nonconstant=bool(np.any(discrete[rows]!=discrete[rows[0]])),
                continuous_objective=float(continuous[rows].mean()),discrete_objective=float(discrete[rows].mean()))
        if not all(np.isfinite(v).all() for v in (terms,baseline,advantage,row_weights)):
            raise FloatingPointError('Nonfinite hybrid component losses or leave-one-out coefficients')
        margins['deterministic_full_cost_smooth_branch']=margins['smooth_branch'].copy()
        margins['smooth_branch']=margins['clip_min_distance']>0.
        margins['discrete_boundary_hit']=margins['ending_abs_distance']==0.
        causal=item['pre'][...,0]<task.dynamics.gate
        for step,(action,logp) in enumerate(zip(item['actions_graph'],item['logps_graph'])):
            path_weights.append(torch.as_tensor(cotangents[:,step,None]*row_weights[:,None],dtype=action.dtype,device=action.device))
            score_weights.append(torch.as_tensor(advantage*row_weights*causal[:,step],dtype=logp.dtype,device=logp.device).detach())
        path_component_weights=weights.copy();path_component_weights[:,2:]=0.
        details.append(dict(action_cotangents=cotangents,component_weights=weights,pathwise_component_weights=path_component_weights,
            policy_dy=sensitivity,continuous_objective_per_episode=continuous,discrete_objective_per_episode=discrete,
            objective_per_episode=continuous+discrete,discrete_baselines=baseline,discrete_advantages=advantage,
            row_weights=row_weights,causal_score_mask=causal,branch_margins=margins))
    params=[p for _,p in named]
    path_vjp=vjp('hybrid_credit_pathwise_parameter_vjp',all_actions,params,path_weights,True)
    score_vjp=vjp('hybrid_credit_score_parameter_vjp',all_logps,params,score_weights,False)
    def flatten(values):
        return np.concatenate([np.zeros(p.numel(),dtype=np.float64) if value is None else
            value.detach().cpu().numpy().astype(np.float64).reshape(-1) for (_,p),value in zip(named,values)])
    path_gradient=flatten(path_vjp);score_gradient=flatten(score_vjp);gradient=path_gradient+score_gradient
    if (not np.isfinite(gradient).all() or policy_hash(actor)!=owner
            or not torch.equal(private_rng,actor._generator.get_state()) or not torch.equal(global_rng,torch.get_rng_state())):
        raise FloatingPointError('Nonfinite hybrid credit or changed generating actor/RNG')
    result=dict(schema='hybrid-feedback-gradient-v1',gradient=gradient,pathwise_gradient=path_gradient,score_gradient=score_gradient,
        actor_owner=owner,query_ids=queries,parameter_names=[n for n,_ in named],parameter_shapes=[tuple(p.shape) for _,p in named],
        context_metadata=contexts,row_context_metadata=[i['row_metadata'] for i in prepared],context_count=count,
        trajectory_count=sum(len(i['pre']) for i in prepared),bindings=[g['bindings'] for g in groups],
        beta_override=beta_override,effective_beta=[i['betas'] for i in prepared],
        observations=[np.stack([o.detach().cpu().numpy() for o in i['observations']],axis=1) for i in prepared],
        actions=[i['actions'] for i in prepared],latent=[i['latent'] for i in prepared],
        reconstructed_noise=[np.stack(i['noise'],axis=1) for i in prepared],recorded_log_probs=[i['logps'] for i in prepared],
        objective=float(sum(np.sum(d['row_weights']*d['objective_per_episode']) for d in details)),
        derivative='recorded-latent exact-forward smooth-rounding pathwise derivative plus detached-state/latent Gaussian discrete-only score',
        qualification_required=['native inference shapes/float32 precision recorded',
            'continuous frozen-noise complete-feedback finite differences','conditional Gaussian score derivative',
            'stochastic sampling and ordinary-state coverage','fresh actual deterministic qualification'],
        teacher_action_labels=False,optimizer_steps=0,random_draws=0,backward_calls=3,
        neural_forward_rows=sum(i['pre'].shape[0]*i['pre'].shape[1] for i in prepared))
    for key in details[0]:result[key]=[d[key] for d in details]
    return result


def qualify_hybrid_credit(executor, actor, groups, learner, arm, config, beta0_credit):
    """One frozen complete-feedback/conditional-score derivative qualification.

    Two changed float32 actors execute the original initial states with the
    original PCG seeds and native batch shapes. Continuous component losses
    are measured from these new complete outcomes. The discrete score FD is
    a separate fixed-data log-density surrogate on the ORIGINAL observations,
    latents and LOO advantages. Neither comparison differentiates a measured
    discrete outcome through its indicator. Clipping changes are diagnostics;
    only the declared derivative-error tolerances decide the FD checks.
    """
    import torch
    from .learning import NeuralPolicy,save_pt
    from .policy import FamilyGaussianActor,policy_hash
    from .search import state_json

    expected=dict(epsilon=.001,path_atol=1e-4,score_atol=1e-3,rtol=.02,
        minimum_envelope_contrast=1e-8,required=True)
    if (config.get('initial_credit_gate')!=expected or arm not in ('raw','guided')
            or learner.actor is not actor or learner.archive is not executor.archive
            or learner.task is not executor.task or state_json(learner.config)!=config['learning']):
        raise ValueError('Exact frozen hybrid-credit qualification settings and learner ownership required')
    archive=executor.archive;task=executor.task;groups=list(groups);owner=policy_hash(actor)
    queries=[g['result'].query_id for g in groups];named=list(actor.named_parameters())
    names=[name for name,_ in named];shapes=[tuple(p.shape) for _,p in named]
    if (beta0_credit.get('actor_owner')!=owner or beta0_credit.get('query_ids')!=queries
            or beta0_credit.get('beta_override')!=0. or beta0_credit.get('parameter_names')!=names
            or [tuple(s) for s in beta0_credit.get('parameter_shapes',[])]!=shapes):
        raise ValueError('The off-envelope comparison must belong to the same current complete batch')
    tag=f'{arm}_initial_credit';epsilon=expected['epsilon']
    request=archive.append('hybrid_credit_qualification_requested',tag=tag,actor_hash=owner,
        query_ids=queries,settings=expected,complete_probe_actors=2,optimizer_steps=0)
    credit=hybrid_feedback_gradient(archive,actor,task,groups,learner.config,tag=tag+'_base')
    base_ref=learner._packet(tag+'_base',credit)
    if (credit['bindings']!=beta0_credit.get('bindings')
            or credit['row_context_metadata']!=beta0_credit.get('row_context_metadata')):
        raise ValueError('The envelope contrast changed its source contexts or budgets')
    gradient=np.asarray(credit['gradient']);path=np.asarray(credit['pathwise_gradient']);score=np.asarray(credit['score_gradient'])
    off=_finite(beta0_credit['gradient'],'same-batch beta-zero gradient',gradient.shape)
    norm=float(np.linalg.norm(gradient));score_norm=float(np.linalg.norm(score));contrast=float(np.linalg.norm(gradient-off))
    coverage=[c['discrete_outcomes_nonconstant'] for c in credit['context_metadata']]
    checks=dict(finite_nonzero_full_gradient=bool(np.isfinite(gradient).all() and np.isfinite(norm) and norm>0.),
        finite_nonzero_score_gradient=bool(np.isfinite(score).all() and np.isfinite(score_norm) and score_norm>0.),
        discrete_outcomes_nonconstant=bool(any(coverage)),
        envelope_gradient_contrast=bool(np.isfinite(contrast) and contrast>expected['minimum_envelope_contrast']),
        pathwise_finite_difference=False,score_finite_difference=False)
    diagnostics=dict(full_gradient_norm=norm,pathwise_gradient_norm=float(np.linalg.norm(path)),score_gradient_norm=score_norm,
        envelope_gradient_contrast_norm=contrast,context_discrete_coverage=coverage,
        context_ids=[c['context_id'] for c in credit['context_metadata']],probe_skipped_reason=None)
    probes={};parameter_ref=None
    if checks['finite_nonzero_full_gradient']:
        original_requests={e['query_id']:e for line in archive.ledger_path.read_text().splitlines()
            if (e:=json.loads(line))['kind']=='query_requested' and e['query_id'] in set(queries)}
        direction=gradient/norm
        before=torch.nn.utils.parameters_to_vector(actor.parameters()).detach().cpu().numpy().copy()
        parent_ref=save_pt(archive,f'learning/{tag}_parent_actor.pt',actor.policy_state())
        values={};vectors={};clip_changes={}
        for sign,label in ((-1.,'minus'),(1.,'plus')):
            probe_tag=f'{tag}_{label}'
            construction=archive.append('hybrid_credit_fd_actor_requested',tag=probe_tag,parent_actor=owner,
                parent_checkpoint=parent_ref,nominal_step=sign*epsilon,optimizer_steps=0)
            probe=FamilyGaussianActor.from_checkpoint(actor.policy_state(),device=actor.action_scale.device)
            with torch.no_grad():
                target=torch.as_tensor(before.astype(np.float64)+sign*epsilon*direction,
                    dtype=torch.float32,device=actor.action_scale.device).clone()
                torch.nn.utils.vector_to_parameters(target,probe.parameters())
            changed=policy_hash(probe);after=torch.nn.utils.parameters_to_vector(probe.parameters()).detach().cpu().numpy().copy()
            checkpoint=save_pt(archive,f'learning/{probe_tag}_actor.pt',probe.policy_state())
            displacement=archive.save_npz(f'learning/{probe_tag}_parameters.npz',before=before,after=after,
                direction=direction,nominal_step=np.array(sign*epsilon))
            archive.append('hybrid_credit_fd_actor_complete',request_id=construction,tag=probe_tag,
                parent_actor=owner,changed_actor=changed,checkpoint=checkpoint,displacement=displacement,
                parameter_names=names,parameter_shapes=shapes,optimizer_steps=0)
            adapter=NeuralPolicy(probe,archive,task);actual=[];query_details=[]
            for gi,group in enumerate(groups):
                q=group['result'].query_id;meta=original_requests[q]['metadata'];seed=meta['seed']
                reserved={'actor','actor_hash','deterministic','seed','complete_continuation','semantics'}
                extra={k:v for k,v in meta.items() if k not in reserved}
                extra.update(hybrid_credit_fd_parent_query=q,hybrid_credit_fd_parent_actor=owner)
                raw=executor.feedback(group['result'].pre_states[:,0],adapter,
                    kind=f'{probe_tag}_group{gi}',mode='executed',deterministic=False,seed=seed,owner_metadata=extra)
                if raw.actor_owner!=changed:raise ValueError('Changed actor does not own its complete FD probe')
                actual.append(raw);query_details.append(dict(query_id=raw.query_id,original_query_id=q,
                    seed=seed,bindings=group['bindings'],native_batch_rows=len(raw.pre_states)))
            arrays={};continuous_value=0.;score_value=0.;changes=[]
            for gi,(raw,group) in enumerate(zip(actual,groups)):
                totals=raw.components.sum(axis=1);terms=totals[:,:2]/learner.config.return_scale
                for ri,context in enumerate(group['contexts']):
                    if context['beta']!=0.:
                        budget=context['budget']
                        if budget is None:raise ValueError('A probe cannot invent an ordinary-context teacher budget')
                        terms[ri]+=context['beta']*np.square(np.maximum(
                            (totals[ri,:2]-budget.costs[:2])/np.asarray(learner.config.scales[:2]),0.))
                continuous_rows=terms.sum(axis=1);weights=credit['row_weights'][gi]
                continuous_value+=float(weights@continuous_rows)
                observed=credit['observations'][gi];latent=credit['latent'][gi]
                direct=[]
                with torch.no_grad():
                    for step in range(observed.shape[1]):
                        attach_meter(probe,archive,'hybrid_credit_score_finite_difference')
                        obs=torch.as_tensor(observed[:,step],dtype=torch.float32,device=probe.action_scale.device)
                        mean,logstd=probe(obs,0)
                        z=torch.as_tensor(latent[:,step],dtype=mean.dtype,device=mean.device)
                        logp,_,_=probe._evaluated(mean,logstd,z)
                        direct.append(logp.cpu().numpy())
                direct=np.stack(direct,axis=1)
                score_coefficients=(credit['discrete_advantages'][gi]*weights)[:,None]*credit['causal_score_mask'][gi]
                score_value+=float(np.sum(direct.astype(np.float64)*score_coefficients))
                pre=raw.pre_states;movable=pre[...,0]<task.dynamics.gate
                unclipped=pre[...,1]+task.dynamics.move*raw.actions[...,0]
                regions=np.where(movable,np.where(unclipped<-1.,-1,np.where(unclipped>1.,1,0)),0).astype(np.int8)
                changes.append(int(np.count_nonzero((regions!=credit['branch_margins'][gi]['clip_regions'])&movable)))
                arrays[f'group{gi}_continuous_objective']=continuous_rows
                arrays[f'group{gi}_score_log_probs']=direct
                arrays[f'group{gi}_score_coefficients']=score_coefficients
                arrays[f'group{gi}_clip_regions']=regions
            all_requests={};all_completions={}
            for line in archive.ledger_path.read_text().splitlines():
                event=json.loads(line)
                if event['kind']=='query_requested':all_requests[event['query_id']]=event
                elif event['kind']=='query_complete':all_completions[event['query_id']]=event
            for item in query_details:
                item.update(inputs=all_requests[item['query_id']]['inputs'],result=all_completions[item['query_id']]['result'])
            evidence=archive.save_npz(f'learning/{probe_tag}_evidence.npz',
                metadata_json=np.array(json.dumps(dict(actor_hash=changed,queries=query_details,
                    continuous_objective=continuous_value,score_surrogate=score_value),sort_keys=True)),**arrays)
            archive.append('hybrid_credit_fd_probe',tag=probe_tag,actor_hash=changed,checkpoint=checkpoint,
                displacement=displacement,evidence=evidence,queries=query_details,
                continuous_objective=continuous_value,score_surrogate=score_value,
                score_semantics='Fixed original observations/latents and detached original discrete LOO coefficients',
                new_complete_outcomes_used_for='Continuous effort/greed component objectives only')
            probes[label]=dict(actor_hash=changed,checkpoint=checkpoint,displacement=displacement,evidence=evidence,
                queries=query_details,continuous_objective=continuous_value,score_surrogate=score_value)
            vectors[label]=after;values[label]=(continuous_value,score_value);clip_changes[label]=changes
        realized=(vectors['plus'].astype(np.float64)-vectors['minus'].astype(np.float64))/(2*epsilon)
        numeric=np.subtract(values['plus'],values['minus'])/(2*epsilon)
        predicted=np.array([path@realized,score@realized]);errors=np.abs(numeric-predicted)
        tolerances=np.array([expected['path_atol'],expected['score_atol']])+expected['rtol']*np.abs(predicted)
        finite=bool(all(np.isfinite(v).all() for v in (realized,numeric,predicted,errors,tolerances)))
        checks['pathwise_finite_difference']=bool(finite and errors[0]<=tolerances[0])
        checks['score_finite_difference']=bool(finite and errors[1]<=tolerances[1])
        parameter_ref=archive.save_npz(f'learning/{tag}_finite_differences.npz',minus=vectors['minus'],plus=vectors['plus'],
            realized_parameter_direction=realized,nominal_direction=direction,epsilon=np.array(epsilon),
            pathwise_gradient=path,score_gradient=score,gradient=gradient,beta0_gradient=off,
            numerical=numeric,predicted=predicted,absolute_error=errors,tolerance=tolerances)
        diagnostics.update(pathwise=dict(numerical=float(numeric[0]),predicted=float(predicted[0]),
            absolute_error=float(errors[0]),tolerance=float(tolerances[0])),
            score=dict(numerical=float(numeric[1]),predicted=float(predicted[1]),
                absolute_error=float(errors[1]),tolerance=float(tolerances[1])),
            clipping_branch_changes=clip_changes,clipping_branches_stable=not any(sum(v) for v in clip_changes.values()),
            clipping_stability_is_gate=False,realized_direction_norm=float(np.linalg.norm(realized)),
            realized_vs_nominal_direction_norm=float(np.linalg.norm(realized-direction)))
    else:diagnostics['probe_skipped_reason']='nonfinite_or_zero_full_gradient'
    if policy_hash(actor)!=owner:raise RuntimeError('Qualification changed its generating actor')
    result=dict(schema='hybrid-credit-qualification-v1',passed=bool(all(checks.values())),checks=checks,
        diagnostics=diagnostics,settings=expected,basecreditref=base_ref,parameter_evidence=parameter_ref,
        probes=probes,actor_owner=owner,query_ids=queries,optimizer_steps=0,
        semantics='Complete stochastic continuous-objective FD and fixed-data discrete score-surrogate FD; no deployment qualification')
    archive.append('hybrid_credit_qualification_complete',request_id=request,arm=arm,**result)
    return result


def deterministic_deployment_gradient(archive, actor, task, groups, *, tag):
    """Two VJPs for original cost on paid native-batch1 ordinary executions.

    Every group is one ORIGINAL current-actor complete deterministic query.
    Its prewritten ``bindings`` packet contains initial_states of shape (1,2)
    and context_metadata_json with one row: context_id, state, task_hash,
    family=0, budget=None. No teacher budget or stochastic trajectory is
    substituted for a native-batch1 deployment execution. Optional ``contexts``
    must equal that same row. Contexts receive equal weight.

    Native (1,8) forwards must reproduce saved means, actions and densities
    exactly before either reverse pass. The adjoint differentiates smooth
    arithmetic of the original complete cost; discrete ending/detour jumps
    remain measured and are not differentiated. The caller must qualify the
    derivative convention and every changed complete deployment separately.
    This function makes no transition, sampling, update or packet-write call.
    """
    import torch
    from pathlib import Path
    from .learning import work_request
    from .policy import policy_hash
    from .search import state_json

    groups = list(groups)
    if not groups:
        raise ValueError('Nonempty current-owned native-batch1 ordinary groups required')
    owner = policy_hash(actor)
    queries = [group['result'].query_id for group in groups]
    if len(set(queries)) != len(queries):
        raise ValueError('A deployment query may contribute only once')
    requests, completions = _owned_query_records(archive, queries, owner)
    named = list(actor.named_parameters())
    if (not named or actor.action_scale.dtype != torch.float32
            or any(not parameter.requires_grad or parameter.dtype != torch.float32 for _, parameter in named)):
        raise ValueError('A differentiable float32 deployment actor is required')
    private_rng = actor._generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    current_checkpoint = actor.policy_state()

    def local_path(reference):
        relative = Path(reference['file'])
        path = archive.path / relative
        if (relative.is_absolute() or '..' in relative.parts
                or path.stat().st_size != reference['bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']):
            raise ValueError('Invalid byte-bound local deployment descriptor')
        return path

    def arrays(reference):
        with np.load(local_path(reference), allow_pickle=False) as packet:
            return {key: packet[key].copy() for key in packet.files}

    def equal_checkpoint(left, right):
        if isinstance(left, torch.Tensor):
            return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
        if isinstance(left, dict):
            return (isinstance(right, dict) and left.keys() == right.keys()
                and all(equal_checkpoint(left[key], right[key]) for key in left))
        if isinstance(left, (list, tuple)):
            return (type(left) is type(right) and len(left) == len(right)
                and all(equal_checkpoint(a, b) for a, b in zip(left, right)))
        return type(left) is type(right) and left == right

    prepared, context_metadata, seen_contexts, checked_checkpoints = [], [], set(), set()
    for group in groups:
        raw = group['result']
        request, completion = requests[raw.query_id], completions[raw.query_id]
        metadata = request['metadata']
        actor_metadata = metadata['actor']
        if (metadata.get('contexts_ref') != group['bindings']
                or actor_metadata.get('actor_hash') != owner
                or actor_metadata.get('sampling') != 'numpy-PCG64-noise-float32-latent'):
            raise ValueError('Deployment actor checkpoint and prewritten ordinary context must own the query')
        actor_reference = actor_metadata['raw']
        actor_path = local_path(actor_reference)
        if str(actor_path) not in checked_checkpoints:
            recorded_checkpoint = torch.load(actor_path, map_location='cpu', weights_only=True)
            if not equal_checkpoint(recorded_checkpoint, current_checkpoint):
                raise ValueError('The generating checkpoint differs from the full current actor state')
            checked_checkpoints.add(str(actor_path))
        saved_inputs = arrays(request['inputs'])
        saved_result = arrays(completion['result'])
        binding = arrays(group['bindings'])
        pre = _finite(raw.pre_states, 'deployment pre-action states')
        if pre.ndim != 3 or pre.shape[0] != 1 or pre.shape[2] != 2 or pre.shape[1] < 1:
            raise ValueError('Original native-batch1 complete traces are required')
        horizon = pre.shape[1]
        post = _finite(raw.states, 'deployment post-action states', pre.shape)
        actions = _finite(raw.actions, 'deployment actions', (1, horizon, 1))
        components = _finite(raw.components, 'deployment components', (1, horizon, 4))
        latent = _finite(raw.latent, 'deployment mean latents', (1, horizon, 1))
        log_probs = _finite(raw.log_probs, 'deployment log densities', (1, horizon))
        raw_arrays = dict(pre_states=pre, states=post, actions=actions, components=components,
            latent=latent, log_probs=log_probs,
            scores=_finite(raw.scores, 'unused deployment scores', (1, horizon, 0)),
            stage_costs=_finite(raw.stage_costs, 'deployment stage costs', (1, horizon)),
            totals=_finite(raw.totals, 'deployment complete costs', (1,)))
        if (raw.actor_owner != owner or horizon != task.config.T or pre[0, 0, 0] != 0.
                or np.any(np.abs(actions) > 1.) or np.any(components < 0.)
                or any(not np.array_equal(value, saved_result[key]) for key, value in raw_arrays.items())
                or json.loads(str(saved_inputs['metadata_json'])) != metadata
                or not np.array_equal(saved_inputs['initial_states'], pre[:, 0])
                or not np.array_equal(binding['initial_states'], pre[:, 0])
                or not np.array_equal(pre[:, 1:], post[:, :-1])
                or not np.array_equal(pre[..., 0], np.arange(task.config.T)[None, :])
                or not np.array_equal(post[..., 0], pre[..., 0] + 1.)
                or np.any(np.abs(pre[..., 1]) > 1.) or np.any(np.abs(post[..., 1]) > 1.)
                or post[0, -1, 0] != task.config.T or post[0, -1, 1] != pre[0, -1, 1]
                or not np.array_equal(raw_arrays['stage_costs'], components.sum(axis=-1))
                or not np.array_equal(raw_arrays['totals'], raw_arrays['stage_costs'].sum(axis=1))
                or request['requested_controls'] != horizon or completion['completed_controls'] != horizon):
            raise ValueError('Byte-bound uninterrupted ordinary native-batch1 traces are required')
        rows = json.loads(str(binding['context_metadata_json']))
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            raise ValueError('One original ordinary context row is required')
        row = rows[0]
        if (set(row) != {'context_id', 'state', 'task_hash', 'family', 'budget'}
                or not isinstance(row['context_id'], str) or not row['context_id']
                or row['context_id'] in seen_contexts or row['task_hash'] != task.task_hash
                or type(row['family']) is not int or row['family'] != 0 or row['budget'] is not None
                or not np.array_equal(_finite(row['state'], 'ordinary context state', (2,)), pre[0, 0])
                or ('contexts' in group and state_json(group['contexts']) != rows)):
            raise ValueError('Distinct ordinary context identities must bind exact state/task/family and budget=None')
        seen_contexts.add(row['context_id'])
        context_metadata.append(dict(row, query_id=raw.query_id, native_batch_size=1))
        prepared.append(dict(raw=raw, pre=pre, actions=actions, components=components,
            latent=latent, log_probs=log_probs, observations=[], action_tensors=[]))

    all_observations, all_actions = [], []
    for item in prepared:
        for step in range(item['pre'].shape[1]):
            observation = task.observation(EndingSnapshot(item['pre'][:, step], task.task_hash))
            if observation.shape != (1, 8):
                raise ValueError('The deployment credit forward must preserve native (1,8) inference')
            obs = torch.as_tensor(observation, dtype=torch.float32,
                device=actor.action_scale.device).detach().requires_grad_(True)
            attach_meter(actor, archive, 'deterministic_deployment_credit')
            mean, logstd = actor(obs, 0)
            action = actor._bounded(mean)
            logp, _, _ = actor._evaluated(mean, logstd, mean)
            if (not np.array_equal(mean.detach().cpu().numpy(), item['latent'][:, step])
                    or not np.array_equal(action.detach().cpu().numpy(), item['actions'][:, step])
                    or not np.array_equal(logp.detach().cpu().numpy(), item['log_probs'][:, step])):
                raise ValueError('Native-batch1 saved mean/actions/log densities do not reproduce exactly')
            item['observations'].append(obs)
            item['action_tensors'].append(action)
            all_observations.append(obs)
            all_actions.append(action)

    def vjp(purpose, inputs, weights, retain_graph, allow_unused):
        request = work_request(archive, 'backward', purpose=purpose, tag=tag,
            actor_hash=owner, query_ids=queries, inference_contract='native_batch1',
            objective_kind='original_complete_ordinary_cost', effective_beta=0., effective_return_scale=1.)
        try:
            values = torch.autograd.grad(tuple(all_actions), tuple(inputs), grad_outputs=tuple(weights),
                retain_graph=retain_graph, create_graph=False, allow_unused=allow_unused)
        except BaseException as error:
            archive.append('backward_error', request_id=request, purpose=purpose,
                tag=tag, error_type=type(error).__name__)
            raise
        archive.append('backward_complete', request_id=request, purpose=purpose, tag=tag)
        return values

    input_vjp = vjp('deployment_credit_input_vjp', all_observations,
        [torch.ones_like(action) for action in all_actions], True, False)
    credits, policy_dy, action_weights, cursor = [], [], [], 0
    for item in prepared:
        horizon = item['pre'].shape[1]
        sensitivity = np.stack([value.detach().cpu().numpy()[:, 0]
            for value in input_vjp[cursor:cursor + horizon]], axis=1)
        cursor += horizon
        # The shared arithmetic primitive accepts a budget-shaped argument.
        # beta=0 makes these zeros unused: they are NOT teacher evidence,
        # a return target, or an OutcomeBudget, and none is represented as such.
        value = ending_cotangents(task.dynamics, item['pre'], item['actions'], item['components'],
            np.zeros((1, 4)), sensitivity, beta=0., return_scale=1., scales=np.ones(4))
        credits.append(value)
        policy_dy.append(sensitivity)
        for step, action in enumerate(item['action_tensors']):
            action_weights.append(torch.as_tensor(value['action_cotangents'][:, step, None] / len(prepared),
                dtype=action.dtype, device=action.device))
    parameter_vjp = vjp('deployment_credit_parameter_vjp', [parameter for _, parameter in named],
        action_weights, False, True)
    gradient = np.concatenate([np.zeros(parameter.numel(), dtype=np.float64) if value is None
        else value.detach().cpu().numpy().astype(np.float64).reshape(-1)
        for (_, parameter), value in zip(named, parameter_vjp)])
    if (not np.isfinite(gradient).all() or policy_hash(actor) != owner
            or not torch.equal(private_rng, actor._generator.get_state())
            or not torch.equal(global_rng, torch.get_rng_state())):
        raise FloatingPointError('Nonfinite deployment credit or changed generating actor/RNG')
    totals = np.concatenate([item['components'].sum(axis=1) for item in prepared])
    objectives = totals.sum(axis=1)
    return dict(schema='deterministic-deployment-gradient-v1', gradient=gradient,
        objective=float(objectives.mean()), objective_per_context=objectives, component_totals=totals,
        component_names=['effort', 'greed', 'detour', 'ending'],
        action_cotangents=[value['action_cotangents'] for value in credits],
        component_weights=[value['component_weights'] for value in credits],
        branch_margins=[value['branch_margins'] for value in credits], policy_dy=policy_dy,
        observations=[np.stack([o.detach().cpu().numpy() for o in item['observations']], axis=1) for item in prepared],
        actions=[item['actions'] for item in prepared], context_metadata=context_metadata,
        bindings=[group['bindings'] for group in groups], actor_owner=owner, query_ids=queries,
        parameter_names=[name for name, _ in named], parameter_shapes=[tuple(parameter.shape) for _, parameter in named],
        context_count=len(prepared), trajectory_count=len(prepared), native_batch_size=1,
        inference_contract='native_batch1', objective_kind='original_complete_ordinary_cost',
        effective_beta=0., effective_return_scale=1., teacher_budget_supplied=False,
        neural_forward_rows=sum(item['pre'].shape[1] for item in prepared), backward_calls=2,
        random_draws=0, optimizer_steps=0,
        derivative='Native-batch1 smooth-arithmetic complete-feedback original-cost derivative; discrete mode jumps not differentiated; complete-policy FD and actual qualification required')


def projected_cost_risk_step(gD, gR, alpha, radius):
    """Exact Euclidean proximal cost step in a homogeneous risk halfspace.

    Minimize gD·delta + ||delta||²/(2*alpha), subject to gR·delta <= 0
    and ||delta|| <= radius. Project center=-alpha*gD onto the halfspace,
    then radially cap that projection. Homogeneity makes this the unique
    strongly convex optimum, including the zero-step opposed-gradient case.

    The certificate uses Euclidean projection stationarity with risk_dual
    for gR·delta and norm_dual for .5*(||delta||²-radius²). It certifies this
    arithmetic subproblem only, never an actual return/risk/capture outcome.
    All exposed NumPy dot/norm/finite-check calls and wall time are returned;
    exceptions retain partial work in error.projection_work. No RNG or solver
    backend is invoked.
    """
    import time
    started = time.perf_counter()
    work = dict(input_validation_requests=1, input_validation_completions=0,
        numpy_dot_calls=0, numpy_norm_calls=0, finite_checks=0,
        halfspace_projection_calls=0, radial_cap_calls=0, parameter_dimension=0,
        random_draws=0, neural_rows=0, backend_solver_calls=0)

    def fail(message):
        error = ValueError(message)
        error.projection_work = dict(work, wall_seconds=time.perf_counter() - started)
        raise error

    def finite(value):
        work['finite_checks'] += 1
        return bool(np.isfinite(value).all())

    def dot(left, right):
        work['numpy_dot_calls'] += 1
        return float(np.dot(left, right))

    def norm(value):
        work['numpy_norm_calls'] += 1
        return float(np.linalg.norm(value))

    try:
        cost = np.asarray(gD, dtype=np.float64)
        risk = np.asarray(gR, dtype=np.float64)
        alpha, radius = float(alpha), float(radius)
    except (TypeError, ValueError, OverflowError):
        fail('Finite matching gradient vectors and positive alpha/radius are required')
    if (cost.ndim != 1 or len(cost) < 1 or risk.shape != cost.shape
            or not finite(cost) or not finite(risk) or not finite([alpha, radius])
            or alpha <= 0. or radius <= 0.):
        fail('Finite matching gradient vectors and positive alpha/radius are required')
    work['parameter_dimension'] = len(cost)
    work['input_validation_completions'] = 1
    center = -alpha * cost
    if not finite(center):
        fail('The cost-step center is not representable in finite float64 arithmetic')
    center_norm, risk_norm = norm(center), norm(risk)
    if not finite([center_norm, risk_norm]):
        fail('Gradient/step norms are not representable in finite float64 arithmetic')
    # A unit normal avoids squaring the original risk-gradient norm when
    # computing the projection. Positive risk-gradient scaling changes no step.
    normal = risk / risk_norm if risk_norm > 0. else np.zeros_like(risk)
    normal_center = dot(normal, center)
    normal_dual = max(normal_center, 0.)
    projection_active = normal_dual > 0.
    work['halfspace_projection_calls'] += 1
    projected = center - normal_dual * normal
    projected_norm = norm(projected)
    if not finite([normal_center, normal_dual, projected_norm]) or not finite(projected):
        fail('The risk-halfspace projection is not finite')
    scale = min(1., radius / projected_norm) if projected_norm > 0. else 1.
    if scale <= 0. or not finite(scale):
        fail('The norm-cap scale is not representable in finite float64 arithmetic')
    work['radial_cap_calls'] += 1
    delta = projected * scale
    norm_dual = 1. / scale - 1.
    risk_dual = normal_dual / risk_norm if risk_norm > 0. else 0.
    step_norm = norm(delta)
    risk_derivative = dot(risk, delta)
    normalized_risk_residual = dot(normal, delta)
    cost_derivative = dot(cost, delta)
    residual = delta - center + normal_dual * normal + norm_dual * delta
    stationarity = norm(residual)
    squared_norm = dot(delta, delta)
    norm_constraint = .5 * (squared_norm - radius * radius)
    risk_complementarity = normal_dual * normalized_risk_residual
    norm_complementarity = norm_dual * norm_constraint
    complementarity = max(abs(risk_complementarity), abs(norm_complementarity))
    difference = delta - center
    primal = .5 * dot(difference, difference)
    shifted = center - normal_dual * normal
    dual = (.5 * dot(center, center) - .5 * dot(shifted, shifted) / (1. + norm_dual)
        - .5 * norm_dual * radius * radius)
    gap = primal - dual
    objective = cost_derivative + .5 * squared_norm / alpha
    # This tolerance qualifies only floating arithmetic of the closed form.
    # Actual float32 displacement and outcomes are the caller's separate gates.
    arithmetic_atol = 128. * np.finfo(np.float64).eps * max(1., center_norm, radius)
    gap_atol = arithmetic_atol * max(1., center_norm, radius)
    metrics = [risk_dual, norm_dual, step_norm, risk_derivative, normalized_risk_residual,
        cost_derivative, stationarity, norm_constraint, complementarity, primal, dual, gap,
        objective, arithmetic_atol, gap_atol]
    if not finite(delta) or not finite(metrics):
        fail('The projection certificate is not representable in finite float64 arithmetic')
    checks = dict(finite=True, risk_halfspace=normalized_risk_residual <= arithmetic_atol,
        norm_cap=step_norm <= radius + arithmetic_atol,
        dual_nonnegative=risk_dual >= 0. and norm_dual >= 0.,
        stationarity=stationarity <= arithmetic_atol,
        complementarity=complementarity <= gap_atol,
        duality_gap=abs(gap) <= gap_atol)
    certificate = dict(schema='projected-cost-risk-step-v1', certified=bool(all(checks.values())),
        checks=checks, alpha=alpha, radius=radius, parameter_dimension=len(cost),
        projection_active=projection_active, norm_cap_active=scale < 1.,
        risk_dual=risk_dual, norm_dual=norm_dual,
        stationarity_residual=stationarity, risk_residual=risk_derivative,
        normalized_risk_residual=normalized_risk_residual,
        norm_residual=step_norm-radius, complementarity_residual=complementarity,
        primal_objective=primal, dual_objective=dual, duality_gap=gap,
        cost_proximal_objective=objective, cost_directional_derivative=cost_derivative,
        risk_directional_derivative=risk_derivative, step_norm=step_norm,
        center_norm=center_norm, halfspace_projection_norm=projected_norm,
        radial_scale=scale, arithmetic_atol=arithmetic_atol, gap_atol=gap_atol,
        dual_convention='Euclidean projection: risk gR·delta<=0; norm .5*(||delta||²-radius²)<=0',
        scope='Closed-form arithmetic subproblem only; actual realized displacement and complete outcomes require separate checks',
        projection_work=dict(work, wall_seconds=time.perf_counter()-started))
    return delta, certificate


def hybrid_outcome_gradient(context_groups):
    """Combine continuous pathwise credit with discrete-only score credit.

    Each group supplies ``continuous_pathwise_gradients`` and detached
    ``trajectory_scores`` with shape (blocks, repeats, parameters), plus
    ``discrete_losses`` with shape (blocks, repeats). A single block may omit
    its leading dimension. Repeats must be independent complete executions
    of one current actor at the same context; ownership, causal scores and
    the continuous/discrete partition must be verified by the caller.

    Leave-one-repeat-out baselines use only discrete losses from the same
    block. Blocks are independent replicates of that context's estimator.
    Contexts receive equal weight even when repeat/block counts differ. No
    random draw or policy update occurs here. Per-group ``random_draws`` is
    explicitly caller-declared accounting, not an observed RNG count.
    Invalid arithmetic raises ValueError with partial ``hybrid_work``.
    """
    import time
    from .coupling import finite
    started=time.perf_counter()
    work=dict(input_validation_requests=0,input_validation_completions=0,
        loo_requests=0,loo_completions=0,score_product_requests=0,score_product_completions=0,
        block_reduction_requests=0,block_reduction_completions=0,
        context_reduction_requests=0,context_reduction_completions=0,
        context_groups=0,replicate_blocks=0,trajectory_samples=0,parameter_dimension=0,
        scalar_score_products=0,caller_declared_random_draws=0,
        helper_random_draws=0,helper_rng_calls=0,wall_seconds=0.)
    def fail(message):
        work['wall_seconds']=time.perf_counter()-started
        error=ValueError(message);error.hybrid_work=dict(work);raise error
    if not isinstance(context_groups,(list,tuple)) or not context_groups:
        fail('At least one repeated-context group is required')
    context_gradients=[];pathwise_gradients=[];score_gradients=[]
    block_gradients=[];baselines=[];advantages=[];coverage=[]
    dimension=None
    for group in context_groups:
        work['input_validation_requests']+=1
        if not isinstance(group,dict):fail('Each repeated-context group must be a dictionary')
        try:
            continuous=finite(group['continuous_pathwise_gradients'])
            losses=finite(group['discrete_losses']);scores=finite(group['trajectory_scores'])
            draws=group['random_draws']
        except (KeyError,TypeError,ValueError,OverflowError):
            fail('Finite pathwise gradients, discrete losses, scores and draw metadata are required')
        if continuous.ndim==2:
            continuous=continuous[None];losses=losses[None];scores=scores[None]
        if (continuous.ndim!=3 or scores.shape!=continuous.shape
                or losses.shape!=continuous.shape[:2] or min(continuous.shape)<1
                or continuous.shape[1]<2 or type(draws) is not int or draws<0
                or (dimension is not None and continuous.shape[2]!=dimension)):
            fail('Aligned blocks/repeats/parameters, at least two repeats and nonnegative draw counts required')
        blocks,repeats,dimension=continuous.shape
        work['input_validation_completions']+=1
        work['context_groups']+=1;work['replicate_blocks']+=blocks
        work['trajectory_samples']+=blocks*repeats;work['parameter_dimension']=dimension
        work['caller_declared_random_draws']+=draws
        work['loo_requests']+=1
        with np.errstate(over='ignore',invalid='ignore'):
            baseline=(losses.sum(axis=1,keepdims=True)-losses)/(repeats-1)
            advantage=losses-baseline
        if not np.isfinite(baseline).all() or not np.isfinite(advantage).all():
            fail('Nonfinite discrete leave-one-out arithmetic')
        work['loo_completions']+=1;work['score_product_requests']+=1
        with np.errstate(over='ignore',invalid='ignore'):
            products=advantage[...,None]*scores
        if not np.isfinite(products).all():fail('Nonfinite discrete score product')
        work['score_product_completions']+=1;work['scalar_score_products']+=products.size
        work['block_reduction_requests']+=1
        with np.errstate(over='ignore',invalid='ignore'):
            path_blocks=continuous.mean(axis=1);score_blocks=products.mean(axis=1)
            combined=path_blocks+score_blocks
            path_mean=path_blocks.mean(axis=0);score_mean=score_blocks.mean(axis=0)
            context_mean=combined.mean(axis=0)
        if not all(np.isfinite(v).all() for v in (path_blocks,score_blocks,combined,path_mean,score_mean,context_mean)):
            fail('Nonfinite hybrid gradient reduction')
        work['block_reduction_completions']+=1
        pathwise_gradients.append(path_mean);score_gradients.append(score_mean)
        context_gradients.append(context_mean);block_gradients.append(combined)
        baselines.append(baseline);advantages.append(advantage)
        nonconstant=np.any(losses!=losses[:,:1],axis=1)
        coverage.append(dict(blocks=blocks,repeats=repeats,
            nonconstant_discrete_blocks=int(nonconstant.sum()),constant_discrete_blocks=int((~nonconstant).sum())))
    work['context_reduction_requests']+=1
    with np.errstate(over='ignore',invalid='ignore'):
        contexts=np.stack(context_gradients);gradient=contexts.mean(axis=0)
    if not np.isfinite(gradient).all():fail('Nonfinite equal-context gradient')
    work['context_reduction_completions']+=1;work['wall_seconds']=time.perf_counter()-started
    return dict(schema='hybrid-outcome-gradient-v1',gradient=gradient,context_gradients=contexts,
        context_pathwise_gradients=np.stack(pathwise_gradients),context_score_gradients=np.stack(score_gradients),
        block_gradients=block_gradients,baselines=baselines,advantages=advantages,coverage=coverage,work=work)
