"""Outcome-envelope coupling; teacher controls are never actor labels.

This module contains arithmetic only. Callers must bind complete outcomes and
actor-owned score functions to durable executions in the query archive.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .search import frozen


def finite(value, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise ValueError('Finite, matching cost/score arrays required')
    return array


def primal_step_admissible(certificate):
    """A finite feasible proposal; executed descent must be checked separately.

    Optimizer optimality remains diagnostic. This predicate alone cannot admit
    a policy update, establish capture, or grant controller authority.
    """
    if (not isinstance(certificate,dict) or not isinstance(certificate.get('checks'),dict)
            or not isinstance(certificate.get('solver_status'),dict)):
        return False
    try:
        checks=certificate['checks']
        values=finite([certificate['primal_max_violation'],certificate['actual_norm'],
            certificate['radius'],certificate['feasibility_atol']])
        violation,norm,radius,tolerance=values
        return bool(certificate['schema']=='performance-feasibility-dual-certificate-v1'
            and all(checks.get(key) is True for key in ('finite_values','backend_success','primal_feasible'))
            and certificate['solver_status']['success'] is True
            and radius>0. and norm>=0. and tolerance>=0.
            and 0.<=violation<=tolerance and norm<=radius+tolerance)
    except (KeyError,TypeError,ValueError,OverflowError):
        return False


def bounded_feasibility_ray(p,Q,A,b,*,radius,feasibility_atol=1e-8):
    """Bound a negative first-worst endpoint-gradient ray, without optimization.

    ``p`` contains measured ending residuals, ``Q`` their current-actor gradients,
    and ``A@delta<=b`` zero-feasible safety constraints. The step uses the full
    radius unless a positive safety slope limits its length. It never aims for
    an exact ending boundary. Descent here concerns only the selected residual;
    actual complete outcomes must separately establish worst-ending progress
    and capture. No solver optimality or task-feasibility claim is returned.

    Invalid/nonfinite arithmetic raises ValueError with completed/partial
    exposed kernel counts in ``error.ray_work``. No SciPy or actor is accessed.
    """
    import time
    begun=time.perf_counter()
    work={name:0 for name in ('input_validation_requests','input_validation_completions',
        'worst_selection_requests','worst_selection_completions','norm_requests','norm_completions',
        'direction_requests','direction_completions','directional_dot_requests','directional_dot_completions',
        'safety_matvec_requests','safety_matvec_completions','safety_bounds_requests','safety_bounds_completions',
        'step_requests','step_completions','performance_matvec_requests','performance_matvec_completions',
        'certificate_requests','certificate_completions')}
    work.update(internal_blas_operations='unobserved inside NumPy kernels',wall_seconds=0.)
    def fail(message):
        work['wall_seconds']=time.perf_counter()-begun
        error=ValueError(message);error.ray_work=dict(work);raise error
    work['input_validation_requests']+=1
    try:
        p,Q,A,b=finite(p),finite(Q),finite(A),finite(b)
        settings=finite([radius,feasibility_atol])
    except (TypeError,ValueError,OverflowError) as error:
        fail('Invalid feasibility ray input: '+str(error))
    radius,tolerance=settings.tolist()
    if (p.ndim!=1 or not len(p) or Q.ndim!=2 or Q.shape[0]!=len(p) or not Q.shape[1]
            or A.ndim!=2 or A.shape[1]!=Q.shape[1] or b.shape!=(A.shape[0],)
            or np.any(b<0.) or radius<=0. or tolerance<0.):
        fail('Finite matching residuals/gradients, zero-feasible safety and positive radius required')
    work['worst_selection_requests']+=1
    worst=int(np.argmax(p))
    work['worst_selection_completions']+=1
    if p[worst]<=0.:fail('Feasibility ray requires a positive measured ending residual')
    work['input_validation_completions']+=1
    def norm(value):
        work['norm_requests']+=1
        with np.errstate(over='ignore',invalid='ignore'):value=float(np.linalg.norm(value))
        if not np.isfinite(value):fail('Nonfinite feasibility ray norm')
        work['norm_completions']+=1
        return value
    gradient=Q[worst];gradient_norm=norm(gradient)
    work['direction_requests']+=1
    direction=-gradient/gradient_norm if gradient_norm>0. else np.zeros(Q.shape[1])
    work['direction_completions']+=1
    direction_norm=norm(direction)
    work['directional_dot_requests']+=1
    directional_derivative=float(gradient@direction)
    work['directional_dot_completions']+=1
    work['safety_matvec_requests']+=1
    with np.errstate(over='ignore',invalid='ignore'):slopes=A@direction
    if not np.isfinite(slopes).all():fail('Nonfinite feasibility ray safety slopes')
    work['safety_matvec_completions']+=1
    work['safety_bounds_requests']+=1
    bounds=np.full(len(b),radius);positive=slopes>0.
    # Infinite nonlimiting ratios are capped before being saved. This equals
    # min(radius,b/slope) and never relaxes a positive-slope safety bound.
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        bounds[positive]=np.minimum(radius,b[positive]/slopes[positive])
    length=min(radius,float(bounds.min())) if len(bounds) else radius
    if gradient_norm==0.:length=0.
    if not np.isfinite(bounds).all() or not np.isfinite(length):fail('Nonfinite feasibility ray bound')
    work['safety_bounds_completions']+=1
    work['step_requests']+=1
    delta=length*direction
    work['step_completions']+=1
    actual_norm=norm(delta)
    work['performance_matvec_requests']+=1
    with np.errstate(over='ignore',invalid='ignore'):changes=Q@delta;residuals=p+changes
    if not np.isfinite(changes).all() or not np.isfinite(residuals).all():
        fail('Nonfinite feasibility ray residual prediction')
    work['performance_matvec_completions']+=1
    work['safety_matvec_requests']+=1
    with np.errstate(over='ignore',invalid='ignore'):safety_residuals=A@delta-b
    if not np.isfinite(safety_residuals).all():fail('Nonfinite feasibility ray safety residual')
    work['safety_matvec_completions']+=1
    work['certificate_requests']+=1
    violation=max(0.,actual_norm-radius,float(safety_residuals.max()) if len(b) else 0.)
    if not np.isfinite([directional_derivative,violation]).all():fail('Nonfinite feasibility ray certificate')
    checks=dict(finite_values=True,primal_norm=bool(actual_norm<=radius+tolerance),
        primal_safety=bool(np.all(safety_residuals<=tolerance)),
        descent=bool(gradient_norm>0. and length>0. and changes[worst]<0.))
    admissible=bool(all(checks.values()))
    reason=None if admissible else ('zero_worst_gradient' if gradient_norm==0. else
        'ordinary_safety_blocks_ray' if length==0. else 'no_admissible_numerical_descent')
    work['certificate_completions']+=1;work['wall_seconds']=time.perf_counter()-begun
    return delta,dict(schema='bounded-feasibility-ray-v1',method='negative_first_worst_endpoint_gradient',
        certified=False,checks=checks,primal_admissible=admissible,block_reason=reason,
        worst_row=worst,worst_residual=float(p[worst]),gradient_norm=gradient_norm,
        direction=direction.tolist(),direction_norm=direction_norm,directional_derivative=directional_derivative,
        length=length,step_bound=length,radius=radius,actual_norm=actual_norm,
        original_residuals=p.tolist(),performance_changes=changes.tolist(),linearized_residuals=residuals.tolist(),
        safety_slopes=slopes.tolist(),radius_capped_safety_bounds=bounds.tolist(),
        safety_residuals=safety_residuals.tolist(),primal_max_violation=violation,
        feasibility_atol=tolerance,ray_work=work,
        descent_scope='Selected first-argmax residual only; actual complete outcomes must verify worst-ending progress')


def bounded_common_descent_ray(p, Q, A, b, *, ordinary_gradient, radius,
                               feasibility_atol=1e-8):
    """Bound the negative sum of unit endpoint and ordinary-cost gradients.

    The endpoint is the first maximum positive residual in p. Both selected
    gradients must be nonzero and have a strictly common descent direction;
    zero or opposed gradients return a blocked zero step. Safety bounds must
    contain zero. A caller can include its fixed incoming ordinary-cost slack
    among these bounds and MUST verify that anchor through complete execution.
    No task capture, minimax progress, or optimizer optimality is certified.
    Invalid inputs retain partial exposed work in ValueError.ray_work.
    """
    import time
    begun = time.perf_counter()
    work = {name: 0 for name in (
        'input_validation_requests', 'input_validation_completions',
        'worst_selection_requests', 'worst_selection_completions',
        'norm_requests', 'norm_completions', 'direction_requests', 'direction_completions',
        'directional_dot_requests', 'directional_dot_completions',
        'safety_matvec_requests', 'safety_matvec_completions',
        'safety_bounds_requests', 'safety_bounds_completions',
        'step_requests', 'step_completions',
        'performance_matvec_requests', 'performance_matvec_completions',
        'certificate_requests', 'certificate_completions')}
    work.update(internal_blas_operations='unobserved inside NumPy kernels', wall_seconds=0.)
    def fail(message):
        work['wall_seconds'] = time.perf_counter() - begun
        error = ValueError(message)
        error.ray_work = dict(work)
        raise error
    work['input_validation_requests'] += 1
    try:
        p, Q, A, b = finite(p), finite(Q), finite(A), finite(b)
        c = finite(ordinary_gradient)
        settings = finite([radius, feasibility_atol])
    except (TypeError, ValueError, OverflowError) as error:
        fail('Invalid common-descent ray input: ' + str(error))
    radius, tolerance = settings.tolist()
    if (p.ndim != 1 or not len(p) or Q.ndim != 2 or Q.shape[0] != len(p) or not Q.shape[1]
            or c.shape != (Q.shape[1],) or A.ndim != 2 or A.shape[1] != Q.shape[1]
            or b.shape != (A.shape[0],) or np.any(b < 0.) or radius <= 0. or tolerance < 0.):
        fail('Matching finite gradients, zero-feasible safety and positive radius required')
    work['worst_selection_requests'] += 1
    worst = int(np.argmax(p))
    work['worst_selection_completions'] += 1
    if p[worst] <= 0.:
        fail('Common-descent ray requires a positive measured ending residual')
    work['input_validation_completions'] += 1
    def norm(value):
        work['norm_requests'] += 1
        with np.errstate(over='ignore', invalid='ignore'):
            result = float(np.linalg.norm(value))
        if not np.isfinite(result):
            fail('Nonfinite common-descent ray norm')
        work['norm_completions'] += 1
        return result
    def dot(first, second):
        work['directional_dot_requests'] += 1
        with np.errstate(over='ignore', invalid='ignore'):
            result = float(first @ second)
        if not np.isfinite(result):
            fail('Nonfinite common-descent ray derivative')
        work['directional_dot_completions'] += 1
        return result
    g = Q[worst]
    g_norm, c_norm = norm(g), norm(c)
    work['direction_requests'] += 1
    unit_sum = g / g_norm + c / c_norm if g_norm > 0. and c_norm > 0. else np.zeros(Q.shape[1])
    unit_sum_norm = norm(unit_sum)
    direction = -unit_sum / unit_sum_norm if unit_sum_norm > 0. else np.zeros(Q.shape[1])
    endpoint_derivative, ordinary_derivative = dot(g, direction), dot(c, direction)
    common = g_norm > 0. and c_norm > 0. and endpoint_derivative < 0. and ordinary_derivative < 0.
    if not common:
        direction = np.zeros(Q.shape[1])
        endpoint_derivative = ordinary_derivative = 0.
    work['direction_completions'] += 1
    direction_norm = norm(direction)
    work['safety_matvec_requests'] += 1
    with np.errstate(over='ignore', invalid='ignore'):
        slopes = A @ direction
    if not np.isfinite(slopes).all():
        fail('Nonfinite common-descent safety slopes')
    work['safety_matvec_completions'] += 1
    work['safety_bounds_requests'] += 1
    bounds = np.full(len(b), radius)
    positive = slopes > 0.
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        bounds[positive] = np.minimum(radius, b[positive] / slopes[positive])
    length = min(radius, float(bounds.min())) if len(bounds) else radius
    if not common:
        length = 0.
    if not np.isfinite(bounds).all() or not np.isfinite(length):
        fail('Nonfinite common-descent safety bound')
    work['safety_bounds_completions'] += 1
    work['step_requests'] += 1
    delta = length * direction
    work['step_completions'] += 1
    actual_norm = norm(delta)
    work['performance_matvec_requests'] += 1
    with np.errstate(over='ignore', invalid='ignore'):
        changes = Q @ delta
        residuals = p + changes
    if not np.isfinite(changes).all() or not np.isfinite(residuals).all():
        fail('Nonfinite common-descent residual prediction')
    work['performance_matvec_completions'] += 1
    ordinary_change = dot(c, delta)
    work['safety_matvec_requests'] += 1
    with np.errstate(over='ignore', invalid='ignore'):
        safety_residuals = A @ delta - b
    if not np.isfinite(safety_residuals).all():
        fail('Nonfinite common-descent safety residual')
    work['safety_matvec_completions'] += 1
    work['certificate_requests'] += 1
    violation = max(0., actual_norm - radius, float(safety_residuals.max()) if len(b) else 0.)
    if not np.isfinite(violation):
        fail('Nonfinite common-descent certificate')
    checks = dict(finite_values=True, primal_norm=bool(actual_norm <= radius + tolerance),
        primal_safety=bool(np.all(safety_residuals <= tolerance)),
        endpoint_descent=bool(common and length > 0. and changes[worst] < 0.),
        ordinary_descent=bool(common and length > 0. and ordinary_change < 0.))
    admissible = bool(all(checks.values()))
    reason = None if admissible else ('zero_endpoint_gradient' if g_norm == 0. else
        'zero_ordinary_gradient' if c_norm == 0. else 'opposed_gradients' if unit_sum_norm == 0. else
        'safety_blocks_ray' if length == 0. and common else 'no_representable_common_descent')
    work['certificate_completions'] += 1
    work['wall_seconds'] = time.perf_counter() - begun
    return delta, dict(schema='bounded-common-descent-ray-v1',
        method='negative_sum_of_unit_endpoint_and_ordinary_gradients', certified=False,
        checks=checks, primal_admissible=admissible, block_reason=reason, worst_row=worst,
        worst_residual=float(p[worst]), endpoint_gradient_norm=g_norm, ordinary_gradient_norm=c_norm,
        unit_sum_norm=unit_sum_norm, direction=direction.tolist(), direction_norm=direction_norm,
        endpoint_directional_derivative=endpoint_derivative, ordinary_directional_derivative=ordinary_derivative,
        length=length, step_bound=length, radius=radius, actual_norm=actual_norm,
        endpoint_linearized_change=float(changes[worst]), ordinary_linearized_change=ordinary_change,
        original_residuals=p.tolist(), performance_changes=changes.tolist(), linearized_residuals=residuals.tolist(),
        safety_slopes=slopes.tolist(), radius_capped_safety_bounds=bounds.tolist(),
        safety_residuals=safety_residuals.tolist(), primal_max_violation=violation,
        feasibility_atol=tolerance, ray_work=work,
        descent_scope='Selected endpoint and ordinary linearized cost only; actual fixed-anchor and ending checks remain mandatory')


@dataclass(frozen=True)
class OutcomeBudget:
    """Cost-component envelope from one complete successful executed teacher."""
    costs: np.ndarray
    names: tuple[str, ...]
    context_key: str
    task_hash: str
    family: int
    teacher_controller_hash: str
    query_ids: tuple[str, ...]

    def __post_init__(self):
        costs = finite(self.costs)
        if (costs.ndim != 1 or len(costs) != len(self.names) or len(costs) == 0
                or np.any(costs < 0) or len(set(self.names)) != len(self.names)):
            raise ValueError('Named nonnegative component costs required')
        if not all(isinstance(x,str) and x for x in (
                self.context_key,self.task_hash,self.teacher_controller_hash,*self.names,*self.query_ids)):
            raise ValueError('Explicit evidence identities required')
        if not self.query_ids or len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError('Unique executed query references required')
        if type(self.family) is not int or self.family < 0:
            raise ValueError('Persistent family ID required')
        object.__setattr__(self,'costs',frozen(costs))
        object.__setattr__(self,'names',tuple(self.names))
        object.__setattr__(self,'query_ids',tuple(self.query_ids))

    @classmethod
    def from_execution(cls, episode, names):
        if (episode['semantics'] not in ('executed_replanning','executed_prefix_then_actor') or not episode['complete']
                or not episode['capture'] or episode.get('actor_owner') is not None):
            raise ValueError('Budget requires complete successful executed teacher evidence')
        components = finite(episode['components'])
        states = finite(episode['states'])
        actions = finite(episode['actions'])
        if (components.ndim != 2 or components.shape != (len(actions),len(names))
                or len(states) != len(actions)+1 or len(actions) != episode['remaining']
                or len(episode['query_ids']) != len(actions)):
            raise ValueError('Complete pre-action trajectory alignment required')
        return cls(components.sum(axis=0),tuple(names),episode['context_key'],
            episode['task_hash'],episode['family'],episode['controller_hash'],tuple(episode['query_ids']))


def exceedance(costs, budget, scales):
    """One-sided discrepancy; lower component costs are allowed, not fitted."""
    costs, budget, scales = finite(costs),finite(budget),finite(scales)
    if (costs.ndim < 1 or budget.ndim != 1 or costs.shape[-1] != len(budget)
            or scales.shape != budget.shape or np.any(scales <= 0)):
        raise ValueError('Matching components and positive fixed scales required')
    return np.square(np.maximum((costs-budget)/scales,0.)).sum(axis=-1)


def actor_losses(component_costs, budget: OutcomeBudget, *, context_key, task_hash,
                 family, behavior_owner, expected_owner, complete, scales, beta=1., return_scale=1.):
    """Returns of the behavior actor only, with an explicit teacher constraint.

    The returned per-episode penalty is nonadditive. Its score-function gradient
    multiplies the WHOLE trajectory score; it is not relabeled cost-to-go for a
    different actor, nor detached into a made-up per-state value target.
    """
    costs = finite(component_costs)
    if costs.ndim != 2 or costs.shape[1:] != budget.costs.shape:
        raise ValueError('Complete per-episode component totals required')
    if (context_key,task_hash,family) != (budget.context_key,budget.task_hash,budget.family):
        raise ValueError('Teacher budget is outside this exact context/task/family')
    if not complete or not behavior_owner or behavior_owner != expected_owner:
        raise ValueError('Complete outcomes must belong to the generating actor')
    if not np.isfinite([beta,return_scale]).all() or beta < 0 or return_scale <= 0:
        raise ValueError('Nonnegative beta and positive return scale required')
    raw = costs.sum(axis=1)/return_scale
    # Zero really removes teacher dependence, including floating-point effects.
    return raw if beta == 0 else raw+beta*exceedance(costs,budget.costs,scales)


def score_gradient(losses, trajectory_scores):
    """Unbiased repeated-context REINFORCE with leave-one-trajectory-out baseline."""
    losses, scores = finite(losses),finite(trajectory_scores)
    if losses.ndim != 1 or len(losses) < 2 or scores.ndim != 2 or len(scores) != len(losses):
        raise ValueError('At least two repeated executions of ONE context required')
    baseline = (losses.sum()-losses)/(len(losses)-1)
    advantages = losses-baseline
    return (advantages[:,None]*scores).mean(axis=0),baseline


def clipped_step(parameters, gradient, *, learning_rate, max_norm):
    parameters,gradient = finite(parameters),finite(gradient)
    if (parameters.shape != gradient.shape or not np.isfinite([learning_rate,max_norm]).all()
            or min(learning_rate,max_norm) <= 0):
        raise ValueError('Matching parameters and positive finite step settings required')
    norm = float(np.linalg.norm(gradient))
    return parameters-learning_rate*gradient*min(1.,max_norm/max(norm,1e-300))


def project_ending_step(direction, terminal_gradient, *, residual, step, margin):
    """Nearest cost step satisfying ONE linearized measured ending constraint.

    ``residual`` is actual y_T minus the task threshold. This arithmetic only
    proposes parameters; complete feedback executions must verify feasibility.
    """
    direction,a=finite(direction),finite(terminal_gradient)
    if (direction.ndim!=1 or direction.shape!=a.shape or not len(direction)
            or not np.isfinite([residual,step,margin]).all() or step<=0 or margin<0):
        raise ValueError('Finite matching directions, positive step and nonnegative guard required')
    original=step*direction;violation=max(residual+margin+float(a@original),0.)
    norm2=float(a@a)
    if norm2==0:
        if violation>0:raise ValueError('Vanishing terminal sensitivity cannot repair the measured constraint')
        return original,0.
    coefficient=violation/norm2
    return original-coefficient*a,coefficient


def project_ending_and_ordinary_step(direction, terminal_gradient, ordinary_gradient,
                                     *, residual, step, margin,
                                     feasibility_atol=1e-10, condition_limit=1e10):
    """Nearest step satisfying two fixed, measured linearized constraints.

    For x=step*direction, project onto
    terminal_gradient @ delta <= -residual-margin, and
    ordinary_gradient @ delta <= min(ordinary_gradient @ x, 0).
    Thus an existing predicted ordinary improvement is preserved, and an
    original uphill direction receives a non-worsening ordinary bound.

    This is parameter-vector arithmetic only. The gradients and residual stay
    fixed; no actor, transition, outcome, or trust authority is constructed.
    Complete executions must still validate nonlinear ending feasibility and
    ordinary performance after the actual float32 parameter assignment.

    Enumerating zero, one, and two active KKT constraints gives the Euclidean
    projection. Feasible zero/single-active candidates also cover zero normals,
    redundant parallel constraints, and opposing constraints defining a slab.
    A two-active solve is used only below the declared raw-Gram condition limit;
    no ridge or constraint relaxation is applied. ``gram_condition=None`` means
    the condition was singular/nonfinite, including a zero normal. Such a case
    can still have a certified zero/single-active solution.
    """
    direction, a, c = finite(direction), finite(terminal_gradient), finite(ordinary_gradient)
    if (direction.ndim != 1 or a.shape != direction.shape or c.shape != direction.shape
            or not len(direction) or not np.isfinite(
                [residual, step, margin, feasibility_atol, condition_limit]).all()
            or step <= 0 or margin < 0 or feasibility_atol < 0 or condition_limit < 1):
        raise ValueError('Finite matching gradients, positive step, and valid projection tolerances required')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        original = step * direction
        original_ordinary = float(c @ original)
        ordinary_bound = min(original_ordinary, 0.)
        matrix = np.stack((a, c))
        bounds = np.asarray([-residual - margin, ordinary_bound], dtype=np.float64)
        gram = matrix @ matrix.T
    if not all(np.isfinite(value).all() for value in (original, bounds, gram)):
        raise ValueError('Numerically unstable projection arithmetic')
    if not np.isfinite(original_ordinary):
        raise ValueError('Numerically unstable ordinary prediction')
    solver_work = dict(candidate_checks=0, condition_estimates=1,
                       linear_solve_requests=0, linear_solve_completions=0)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        try:
            condition = float(np.linalg.cond(gram))
        except np.linalg.LinAlgError:
            condition = float('inf')
    gram_condition = condition if np.isfinite(condition) else None
    zero_normals = np.diag(gram) == 0.
    if np.any(zero_normals & (bounds < -feasibility_atol)):
        raise ValueError('Infeasible projection: violated constraint has zero sensitivity')

    candidates = []

    def consider(active, multipliers):
        solver_work['candidate_checks'] += 1
        multipliers = np.asarray(multipliers, dtype=np.float64)
        if (not np.isfinite(multipliers).all()
                or np.any(multipliers < -feasibility_atol)):
            return
        # Eliminate only roundoff-sized negative multipliers, then recheck every
        # primal constraint and each declared active equality on the new step.
        multipliers = np.maximum(multipliers, 0.)
        with np.errstate(over='ignore', invalid='ignore'):
            delta = original - matrix.T @ multipliers
            constraint_residuals = matrix @ delta - bounds
            correction_norm = float(np.linalg.norm(delta - original))
        if (not np.isfinite(delta).all() or not np.isfinite(constraint_residuals).all()
                or not np.isfinite(correction_norm)
                or np.any(constraint_residuals > feasibility_atol)
                or any(abs(constraint_residuals[index]) > feasibility_atol for index in active)):
            return
        candidates.append((correction_norm, tuple(active), delta, multipliers,
                           constraint_residuals))

    consider((), np.zeros(2))
    for index in range(2):
        if zero_normals[index]:
            continue
        multipliers = np.zeros(2)
        multipliers[index] = float((matrix[index] @ original - bounds[index]) / gram[index, index])
        consider((index,), multipliers)
    if np.isfinite(condition) and condition <= condition_limit:
        solver_work['linear_solve_requests'] += 1
        try:
            multipliers = np.linalg.solve(gram, matrix @ original - bounds)
        except np.linalg.LinAlgError:
            pass
        else:
            solver_work['linear_solve_completions'] += 1
            consider((0, 1), multipliers)
    if not candidates:
        if not np.isfinite(condition) or condition > condition_limit:
            raise ValueError('Infeasible or numerically unstable two-constraint projection')
        raise ValueError('Infeasible two-constraint projection or failed KKT feasibility check')
    # Stable enumeration order resolves numerically equal candidates without
    # changing the objective or introducing a new selection criterion.
    best = min(enumerate(candidates), key=lambda item: (item[1][0], item[0]))[1]
    correction_norm, active, delta, multipliers, constraint_residuals = best
    certificate = dict(
        ordinary_bound=ordinary_bound,
        duals=multipliers.tolist(),
        active_constraints=list(active),
        gram_condition=gram_condition,
        constraint_residuals=constraint_residuals.tolist(),
        correction_norm=correction_norm,
        original_ordinary_prediction=original_ordinary,
        projected_ordinary_prediction=float(c @ delta),
        projected_ending_residual=float(residual + margin + a @ delta),
        solver_work=solver_work,
    )
    return delta, certificate


def performance_feasibility_step(performance_residuals, performance_gradients,
                                 safety_gradients, safety_bounds, *, alpha, radius,
                                 phase='feasibility', max_iterations=200,
                                 solver_ftol=1e-12, feasibility_atol=1e-8,
                                 certificate_atol=1e-8, certificate_rtol=1e-8,
                                 objective_gradient=None, dual_objective_scale=1.):
    """Propose one certified, norm-bounded step from fixed measured gradients.

    For p of shape (M,), Q of shape (M,N), A of shape (K,N), and b of
    shape (K,), the feasibility phase minimizes

        t + ||delta||**2/(2*alpha)
        subject to t >= 0, p + Q@delta <= t, A@delta <= b,
                   ||delta|| <= radius.

    The improvement phase requires p <= 0 and minimizes

        mean(Q,axis=0)@delta + ||delta||**2/(2*alpha)
        subject to p + Q@delta <= 0, A@delta <= b,
                   ||delta|| <= radius.

    The latter objective differs from squared distance to -alpha*mean(Q)
    only by a constant. Safety bounds must be nonnegative: zero is a feasible
    safety step, and this helper never demands a deeper ending margin.

    Improvement alone accepts an optional finite objective_gradient of shape
    (N,). It replaces mean(Q) in the objective without changing any constraint.
    This q is one fixed final basis row, with coefficient one; it adds no dual
    variable. Its carrier row is excluded from all safety residuals.

    Feasibility alone permits a positive dual_objective_scale. It multiplies
    only the SLSQP objective and Jacobian callbacks, preserving their minimizer.
    Every returned primal/dual certificate remains in the original units.

    A few-variable dual uses [Q;A]@[Q;A].T, with nonnegative performance
    multipliers lambda and safety multipliers mu. Feasibility additionally
    requires sum(lambda) <= 1. With v=Q.T@lambda+A.T@mu (plus mean(Q) in
    improvement, or the explicit q), delta=-min(alpha,radius/||v||)*v. The norm-constrained
    quadratic conjugate supplies a dual lower bound. Singular/repeated/zero
    gradient rows need no inverse, ridge, or relaxed primal constraint.

    Returns (delta, certificate). Backend success is insufficient: finite
    arithmetic, original primal/dual feasibility, gap, stationarity and
    complementarity must all pass. Normal backend nonconvergence returns a
    finite step with certified=False; callers MUST reject it. Invalid inputs
    or nonfinite arithmetic raise ValueError with partial solver_work attached.
    A certified Phase-I optimum can have positive teacher violation or a zero
    step. Solver certification is not teacher attainment or actor admission.

    Only exposed callbacks/kernels are counted. SLSQP's internal factorizations
    and operations are not exposed by SciPy and are explicitly unobserved;
    their wall time is retained. No model, network, optimizer state, or actor
    parameter is accessed or changed. Old projection paths do not import SciPy.
    """
    import time
    begun = time.perf_counter()
    work = {name: 0 for name in (
        'scipy_import_requests', 'scipy_import_completions',
        'gram_requests', 'gram_completions', 'solver_requests',
        'solver_completions', 'solver_errors',
        'objective_requests', 'objective_completions',
        'jacobian_requests', 'jacobian_completions',
        'constraint_requests', 'constraint_completions',
        'constraint_jacobian_requests', 'constraint_jacobian_completions',
        'iteration_callback_requests', 'iteration_callback_completions',
        'dual_evaluation_requests', 'dual_evaluation_completions',
        'dual_cache_lookups', 'dual_cache_hits',
        'gram_matvec_requests', 'gram_matvec_completions',
        'span_quadratic_requests', 'span_quadratic_completions',
        'span_norm_fallback_requests', 'span_norm_fallback_completions',
        'dual_projection_requests', 'dual_projection_completions',
        'dual_simplex_normalization_passes',
        'certificate_requests', 'certificate_completions',
        'full_span_product_requests', 'full_span_product_completions',
        'condition_estimate_requests', 'condition_estimate_completions',
        'explicit_linear_solve_requests', 'explicit_linear_solve_completions')}
    work.update(internal_factorizations='unobserved inside scipy.optimize.SLSQP',
        internal_operations='unobserved inside scipy.optimize.SLSQP',
        solver_wall_seconds=0., wall_seconds=0.)

    def fail(message):
        work['wall_seconds'] = time.perf_counter() - begun
        error = ValueError(message)
        error.solver_work = dict(work)
        raise error

    try:
        p, Q = finite(performance_residuals), finite(performance_gradients)
        A, b = finite(safety_gradients), finite(safety_bounds)
        custom_objective = objective_gradient is not None
        q = finite(objective_gradient) if custom_objective else None
        settings = finite([alpha, radius, solver_ftol, feasibility_atol,
                           certificate_atol, certificate_rtol, dual_objective_scale])
    except (TypeError, ValueError, OverflowError) as error:
        fail('Invalid performance-step input: ' + str(error))
    (alpha, radius, solver_ftol, feasibility_atol, certificate_atol,
     certificate_rtol, dual_objective_scale) = settings.tolist()
    if (p.ndim != 1 or not len(p) or Q.ndim != 2 or Q.shape[0] != len(p)
            or not Q.shape[1] or A.ndim != 2 or A.shape[1] != Q.shape[1]
            or b.shape != (A.shape[0],) or np.any(b < 0)
            or min(alpha, radius, solver_ftol, dual_objective_scale) <= 0
            or min(feasibility_atol, certificate_atol, certificate_rtol) < 0
            or type(max_iterations) is not int or max_iterations < 1
            or phase not in ('feasibility', 'improvement')
            or (phase == 'improvement' and np.any(p > 0))
            or (dual_objective_scale != 1. and phase != 'feasibility')):
        fail('Finite matching gradients, zero-feasible bounds, and valid performance-step settings required')
    scaled_dual = dual_objective_scale != 1.
    if scaled_dual:
        work.update(objective_scaling_calls=0, jacobian_scaling_calls=0)
    M, N = Q.shape
    K = A.shape[0]
    D = M + K
    if custom_objective and (phase != 'improvement' or q.shape != (N,)):
        fail('Custom objective gradient requires the improvement phase and finite shape (N,)')
    matrix = np.concatenate((Q, A), axis=0)
    if custom_objective:
        matrix = np.concatenate((matrix, q[None, :]), axis=0)
        work.update(basis_rows=D + 1, fixed_objective_rows=1,
            dual_variables=D, parameter_dimension=N)
    offset = np.zeros(M + K + int(custom_objective))
    if phase == 'improvement':
        if custom_objective:
            offset[-1] = 1.
        else:
            offset[:M] = 1. / M
    linear = np.concatenate((p, -b))
    work['gram_requests'] += 1
    with np.errstate(over='ignore', invalid='ignore'):
        gram = matrix @ matrix.T
    if not np.isfinite(gram).all():
        fail('Nonfinite performance-step Gram matrix')
    work['gram_completions'] += 1
    cache = {}

    def radial(norm_value):
        scale = alpha if norm_value == 0 else min(alpha, radius / norm_value)
        if norm_value <= radius / alpha:
            phi = .5 * norm_value * (alpha * norm_value)
        else:
            phi = radius * (norm_value - .5 * radius / alpha)
        if not np.isfinite([scale, phi]).all() or scale <= 0:
            fail('Nonfinite performance-step norm-conjugate arithmetic')
        return scale, phi

    def evaluate(value):
        work['dual_cache_lookups'] += 1
        x = np.asarray(value, dtype=np.float64)
        if 'x' in cache and np.array_equal(x, cache['x']):
            work['dual_cache_hits'] += 1
            return cache['objective'], cache['jacobian']
        work['dual_evaluation_requests'] += 1
        if x.shape != (M + K,) or not np.isfinite(x).all():
            fail('Nonfinite performance-step dual iterate')
        z = (np.concatenate((x, np.zeros(1))) if custom_objective else x) + offset
        work['gram_matvec_requests'] += 1
        product = gram @ z
        work['gram_matvec_completions'] += 1
        work['span_quadratic_requests'] += 1
        norm2 = float(z @ product)
        work['span_quadratic_completions'] += 1
        # Recompute a negative roundoff quadratic using the original rows.
        # No clipping to a PSD surrogate, ridge or constraint relaxation occurs.
        if norm2 < 0:
            work['span_norm_fallback_requests'] += 1
            vector = matrix.T @ z
            norm2 = float(vector @ vector)
            work['span_norm_fallback_completions'] += 1
        if not np.isfinite(norm2) or norm2 < 0 or not np.isfinite(product).all():
            fail('Nonfinite performance-step span arithmetic')
        scale, phi = radial(float(np.sqrt(norm2)))
        objective = float(phi - linear @ x)
        jacobian = scale * product[:D] - linear
        if not np.isfinite(objective) or not np.isfinite(jacobian).all():
            fail('Nonfinite performance-step dual objective or derivative')
        cache.update(x=x.copy(), objective=objective, jacobian=jacobian)
        work['dual_evaluation_completions'] += 1
        return objective, jacobian

    def objective(value):
        work['objective_requests'] += 1
        result = evaluate(value)[0]
        if scaled_dual:
            work['objective_scaling_calls'] += 1
            result *= dual_objective_scale
            if not np.isfinite(result):
                fail('Nonfinite scaled performance-step dual objective')
        work['objective_completions'] += 1
        return result

    def jacobian(value):
        work['jacobian_requests'] += 1
        result = evaluate(value)[1].copy()
        if scaled_dual:
            work['jacobian_scaling_calls'] += 1
            result *= dual_objective_scale
            if not np.isfinite(result).all():
                fail('Nonfinite scaled performance-step dual Jacobian')
        work['jacobian_completions'] += 1
        return result

    def simplex(value):
        work['constraint_requests'] += 1
        result = float(1. - np.sum(value[:M]))
        if not np.isfinite(result):
            fail('Nonfinite performance-step simplex constraint')
        work['constraint_completions'] += 1
        return result

    def simplex_jacobian(value):
        work['constraint_jacobian_requests'] += 1
        result = np.concatenate((-np.ones(M), np.zeros(K)))
        work['constraint_jacobian_completions'] += 1
        return result

    def iteration_callback(value):
        work['iteration_callback_requests'] += 1
        if not np.isfinite(value).all():
            fail('Nonfinite performance-step iteration callback')
        work['iteration_callback_completions'] += 1

    work['scipy_import_requests'] += 1
    try:
        from scipy import optimize
        from scipy import __version__ as scipy_version
    except ImportError as error:
        fail('Performance-step solver unavailable: ' + str(error))
    work['scipy_import_completions'] += 1
    constraints = ([dict(type='ineq', fun=simplex, jac=simplex_jacobian)]
                   if phase == 'feasibility' else [])
    bounds = ([(0., 1.)] * M + [(0., None)] * K if phase == 'feasibility'
              else [(0., None)] * (M + K))
    work['solver_requests'] += 1
    solver_begun = time.perf_counter()
    try:
        result = optimize.minimize(objective, np.zeros(M + K), method='SLSQP',
            jac=jacobian, bounds=bounds, constraints=constraints,
            callback=iteration_callback,
            options=dict(maxiter=max_iterations, ftol=solver_ftol, disp=False))
    except BaseException as error:
        work['solver_errors'] += 1
        work['solver_wall_seconds'] = time.perf_counter() - solver_begun
        fail('Performance-step solver raised ' + type(error).__name__ + ': ' + str(error))
    work['solver_wall_seconds'] = time.perf_counter() - solver_begun
    work['solver_completions'] += 1
    work['certificate_requests'] += 1
    raw_duals = np.asarray(result.x, dtype=np.float64)
    if raw_duals.shape != (M + K,) or not np.isfinite(raw_duals).all():
        fail('Performance-step solver returned nonfinite duals')
    # Ensure the reported lower bound uses truly feasible duals, even if the
    # backend has a tiny bound/simplex roundoff error. Recompute every check
    # and the primal step after this disclosed dual projection.
    work['dual_projection_requests'] += 1
    duals = np.maximum(raw_duals, 0.)
    if phase == 'feasibility' and duals[:M].sum() > 1.:
        duals[:M] /= duals[:M].sum()
        work['dual_simplex_normalization_passes'] += 1
        # Division can round upward. Each counted downward scaling strictly
        # reduces positive entries until the summed simplex bound is exact.
        while duals[:M].sum() > 1.:
            duals[:M] *= np.nextafter(1., 0.)
            work['dual_simplex_normalization_passes'] += 1
    work['dual_projection_completions'] += 1
    work['full_span_product_requests'] += 1
    basis_duals = np.concatenate((duals, np.zeros(1))) if custom_objective else duals
    vector = matrix.T @ (basis_duals + offset)
    work['full_span_product_completions'] += 1
    vector_norm = float(np.linalg.norm(vector))
    scale, phi = radial(vector_norm)
    delta = -scale * vector
    work['full_span_product_requests'] += 1
    predictions = matrix @ delta
    work['full_span_product_completions'] += 1
    norm2 = float(delta @ delta)
    actual_norm = float(np.sqrt(norm2))
    linear_performance = p + predictions[:M]
    epigraph = max(0., float(linear_performance.max())) if phase == 'feasibility' else 0.
    performance_residual = linear_performance - epigraph
    safety_residual = predictions[M:D] - b
    primal = (epigraph if phase == 'feasibility' else float(offset @ predictions)) + norm2 / (2. * alpha)
    dual = float(linear @ duals - phi)
    gap = float(primal - dual)
    gap_tolerance = certificate_atol + certificate_rtol * max(1., abs(primal), abs(dual))
    lambda_sum = float(duals[:M].sum())
    epigraph_dual = max(0., 1. - lambda_sum) if phase == 'feasibility' else 0.
    norm_dual = max(0., vector_norm / radius - 1. / alpha)
    stationarity = delta / alpha + vector + norm_dual * delta
    stationarity_norm = float(np.linalg.norm(stationarity))
    stationarity_inf = float(np.max(np.abs(stationarity)))
    stationarity_tolerance = certificate_atol + certificate_rtol * max(
        1., actual_norm / alpha, vector_norm, norm_dual * actual_norm)
    performance_complementarity = duals[:M] * performance_residual
    safety_complementarity = duals[M:] * safety_residual
    epigraph_complementarity = epigraph_dual * epigraph
    norm_complementarity = norm_dual * .5 * (norm2 - radius * radius)
    span_coefficients = -scale * (basis_duals + offset)
    dual_projection_norm = float(np.linalg.norm(duals - raw_duals))
    norm_residual = actual_norm - radius
    primal_violation = max(0., float(performance_residual.max()),
        float(safety_residual.max()) if K else 0., norm_residual, -epigraph)
    dual_violation = max(0., float(-duals.min()),
        lambda_sum - 1. if phase == 'feasibility' else 0., -epigraph_dual, -norm_dual)
    complementarity = max(float(np.abs(performance_complementarity).max()),
        float(np.abs(safety_complementarity).max()) if K else 0.,
        abs(epigraph_complementarity), abs(norm_complementarity))
    finite_values = bool(all(np.isfinite(value).all() for value in (
        vector, delta, predictions, performance_residual, safety_residual, span_coefficients,
        performance_complementarity, safety_complementarity,
        [primal, dual, gap, gap_tolerance, actual_norm, norm_residual, lambda_sum,
         epigraph, epigraph_dual, norm_dual, stationarity_norm, stationarity_inf,
         stationarity_tolerance, epigraph_complementarity, norm_complementarity,
         primal_violation, dual_violation, complementarity, dual_projection_norm])))
    if not finite_values:
        fail('Nonfinite performance-step primal/dual certificate')
    checks = dict(finite_values=True, backend_success=bool(result.success),
        primal_feasible=primal_violation <= feasibility_atol,
        dual_feasible=dual_violation <= feasibility_atol,
        duality_gap=-gap_tolerance <= gap <= gap_tolerance,
        stationarity=stationarity_norm <= stationarity_tolerance,
        complementarity=complementarity <= gap_tolerance)
    work['certificate_completions'] += 1
    work.update(solver_iterations=int(getattr(result, 'nit', 0)),
        solver_reported_objective_evaluations=int(getattr(result, 'nfev', 0)),
        solver_reported_jacobian_evaluations=int(getattr(result, 'njev', 0)),
        wall_seconds=time.perf_counter() - begun)
    certificate = dict(schema='performance-feasibility-dual-certificate-v1',
        phase=phase, certified=bool(all(checks.values())), checks=checks,
        alpha=alpha, radius=radius, parameter_dimension=N, performance_rows=M,
        safety_rows=K, gram=gram.tolist(), span_coefficients=span_coefficients.tolist(),
        epigraph=epigraph, linearized_performance=linear_performance.tolist(),
        teacher_violation=max(0., float(linear_performance.max())),
        primal_objective=float(primal), dual_objective=dual, duality_gap=gap,
        gap_tolerance=gap_tolerance, feasibility_atol=feasibility_atol,
        performance_residuals=performance_residual.tolist(), safety_residuals=safety_residual.tolist(),
        actual_norm=actual_norm, norm_residual=norm_residual,
        primal_max_violation=primal_violation, dual_max_violation=dual_violation,
        raw_duals=raw_duals.tolist(), dual_performance=duals[:M].tolist(),
        dual_safety=duals[M:].tolist(), dual_epigraph=epigraph_dual,
        dual_norm_squared=norm_dual, dual_projection_norm=dual_projection_norm,
        stationarity_norm=stationarity_norm, stationarity_inf=stationarity_inf,
        stationarity_tolerance=stationarity_tolerance,
        performance_complementarity=performance_complementarity.tolist(),
        safety_complementarity=safety_complementarity.tolist(),
        epigraph_complementarity=float(epigraph_complementarity),
        norm_complementarity=float(norm_complementarity),
        complementarity_max=float(complementarity),
        solver_status=dict(backend='scipy.optimize.SLSQP', scipy_version=scipy_version,
            success=bool(result.success), status=int(result.status), message=str(result.message),
            max_iterations=max_iterations, solver_ftol=solver_ftol),
        solver_work=work)
    if custom_objective:
        certificate.update(objective_kind='custom_linear_gradient',
            objective_gradient=q.tolist(), objective_basis_row=D,
            dual_variables=D, basis_rows=D + 1)
    if scaled_dual:
        certificate['dual_objective_scale'] = dual_objective_scale
    return delta, certificate


def teacher_scores(prefix_costs, full_component_costs, actor_component_costs, *,
                   scales, influence, agreement=1.):
    """Policy-constrained completions of independent prefixes.

    At influence=1: J(prefix + actor tail) + agreement*D(C_actor,C_completion).
    At zero: the independent prefix scores exactly; actor arrays are unused.
    Actual executed replanning must qualify this surrogate separately.
    """
    prefix = finite(prefix_costs)
    if prefix.ndim != 1 or not np.isfinite([influence,agreement]).all() or min(influence,agreement) < 0:
        raise ValueError('Finite prefix vector and nonnegative coupling required')
    if influence == 0:
        return prefix.copy()
    full,actor = finite(full_component_costs),finite(actor_component_costs)
    if full.ndim != 2 or len(full) != len(prefix) or actor.shape != full.shape[1:]:
        raise ValueError('Aligned complete prefix/actor continuations required')
    # Reverse one-sided discrepancy: teacher may remain somewhat suboptimal but
    # closer to the student's achieved component profile. No action distance.
    discrepancy = np.array([exceedance(actor,row,scales) for row in full])
    tail = full.sum(axis=1)-prefix
    return prefix+influence*(tail+agreement*discrepancy)


def select_teaching_pairs(pairs, *, maximum=4, minimum_gain=1e-6, state_atol=1e-8):
    """Select measured interventions in declared order, never held-out scores.

    Both arms retain their own acquired states and executed suffix budgets.
    Complete-root superiority does not imply superiority at different suffix
    contexts. A fixed 64-rollout batch uses 1, 2 or 4 distinct contexts evenly.
    This is conditional teaching acquisition, not an unbiased control sample.
    """
    if maximum!=4 or not np.isfinite([minimum_gain,state_atol]).all() or min(minimum_gain,state_atol)<=0:
        raise ValueError('Declared four-context maximum and positive tolerances required')
    eligible=[];decisions=[];seen=set()
    for index,pair in enumerate(pairs):
        a,b=pair['raw'],pair['guided']
        if (a['context_key'],a['root'],a['task_hash'],a['family'])!=(b['context_key'],b['root'],b['task_hash'],b['family']):
            raise ValueError('Teaching pair has different initial context/task/family')
        if not np.array_equal(a['states'][0],b['states'][0]):raise ValueError('Teaching roots differ')
        complete=all(e['complete'] and e['capture'] for e in (a,b))
        if complete:
            for episode in (a,b):OutcomeBudget.from_execution(episode,('effort','greed','detour','ending'))
            if set(a['query_ids']).intersection(b['query_ids']):raise ValueError('Paired executions share query evidence')
        gain=float(np.asarray(a['components']).sum()-np.asarray(b['components']).sum())
        changed=(np.shape(a['states'])==np.shape(b['states']) and
            np.max(np.abs(np.asarray(a['states'])-np.asarray(b['states'])))>state_atol)
        useful=bool(complete and gain>minimum_gain and changed)
        distinct=a['context_key'] not in seen
        if useful and distinct:
            eligible.append(index);seen.add(a['context_key'])
        decisions.append(dict(index=index,root=a['root'],context=a['context_key'],complete_capture=bool(complete),
            root_cost_gain=gain,state_changed=bool(changed),useful=useful,first_useful_context=bool(useful and distinct)))
    count=max((n for n in (1,2,4) if n<=len(eligible)),default=0)
    return eligible[:count],decisions
