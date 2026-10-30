"""Finite core contract checks. Real physics and neural execution are forbidden."""
from __future__ import annotations
from dataclasses import asdict,is_dataclass
import functools
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

CORE_TESTS = tuple(f'tests/test_{name}.py' for name in (
    'math_properties', 'math_hypothesis', 'stats', 'gait', 'mppi_snapshot',
    'mppi_iterations', 'bidirectional_search_proposals', 'bidirectional_trust',
    'bidirectional_task_evidence', 'repository_layout', 'outcome_coupling','execution_contracts',
    'performance_feasibility','feasibility_ray','interior_ending'))

SOURCE_FILES=('src/gps/bidirectional/validation.py','src/gps/bidirectional/coupling.py','src/gps/bidirectional/credit.py',
    'src/gps/bidirectional/loop.py','tests/test_execution_contracts.py',
    'src/gps/bidirectional/policy.py','src/gps/bidirectional/learning.py',
    'src/gps/bidirectional/global_learning.py','src/gps/bidirectional/teaching.py','src/gps/bidirectional/study.py',
    'src/gps/bidirectional/endpoint.py','src/gps/bidirectional/performance.py','scripts/training/train_bidirectional_gps.py',
    'tests/test_performance_feasibility.py','tests/test_outcome_coupling.py','tests/test_feasibility_ray.py',
    'tests/test_repository_layout.py','tests/test_interior_ending.py')


def snapshot_sources(root,output,label):
    files=[]
    for name in SOURCE_FILES:
        content=(root/name).read_bytes();snapshot=output/label/name
        snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
        files.append(dict(file=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),
            snapshot=str(snapshot.relative_to(output))))
    manifest=output/(label+'.json')
    with manifest.open('x') as stream:json.dump(files,stream,indent=2);stream.write('\n')
    return files


class EnginesForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mujoco', 'warp', 'mujoco_warp', 'torch'}:
            raise RuntimeError(f'Core checks forbid physics/neural runtime: {fullname}')


class FixtureWork:
    """Count outermost synthetic model calls, retaining failed reservations."""
    def __init__(self, path, cap=100_000):
        self.stream = path.open('x')
        self.output = path.parent
        self.cap = cap
        self.depth = self.requested = self.completed = self.failed = 0
        self.reports = []
        self.originals = []
        self.projections=dict(requested=0,completed=0,failed=0,candidate_checks=0,
            condition_estimates=0,linear_solve_requests=0,linear_solve_completions=0)
        self.performance=dict(requested=0,completed=0,failed=0,rejected=0,
            certified=0,uncertified=0,request_cap=32,expected_requests=27,
            exposed_solver_work={},unobserved_solver_work={},unknown_partial_requests=[],wall_seconds=0.)
        self.rays=dict(requested=0,completed=0,failed=0,request_cap=8,expected_requests=4,
            exposed_ray_work={},unknown_partial_requests=[],wall_seconds=0.)
        self.common_rays=dict(requested=0,completed=0,failed=0,request_cap=16,expected_requests=10,
            exposed_ray_work={},unknown_partial_requests=[],wall_seconds=0.)
        self.hybrid=dict(requested=0,completed=0,failed=0,rejected=0,helper_returns=0,
            request_cap=1,expected_requests=1,expected_caller_declared_random_draws=786432,
            expected_estimator_rows=786436,caller_declared_random_draws=0,requested_estimator_rows=0,
            completed_estimator_rows=0,helper_random_draws=0,helper_rng_calls=0,
            exposed_hybrid_work={},unknown_partial_requests=[],raw_archives=[],wall_seconds=0.)
        self.cost_risk=dict(requested=0,completed=0,failed=0,request_cap=4,expected_requests=4,
            exposed_projection_work={},unknown_partial_requests=[],raw_archives=[],wall_seconds=0.)
        self.running_windows=dict(requested=0,completed=0,failed=0,rejected=0,
            expected_range_errors=0,unexpected_errors=0,request_cap=4,expected_requests=4,
            expected_completions=2,expected_failures=2,raw_archives=[],wall_seconds=0.)
        self.realized_primal=dict(requested=0,completed=0,failed=0,unexpected_errors=0,
            request_cap=40,expected_requests=33,expected_completions=16,expected_failures=17,wall_seconds=0.)
        self.teacher_excess=dict(requested=0,completed=0,failed=0,unexpected_errors=0,
            request_cap=32,expected_requests=27,expected_completions=7,expected_failures=20,wall_seconds=0.)

    def realized_primal_call(self,function,*,totals=None,prefix='realized_primal'):
        """Meter pure gate checks, including deliberately invalid inputs."""
        totals=self.realized_primal if totals is None else totals
        @functools.wraps(function)
        def call(*args,**kwargs):
            totals['requested']+=1;number=totals['requested']
            begun=time.perf_counter()
            event=dict(event=prefix+'_fixture_requested',request=number,
                arguments_repr=repr((args,kwargs)),controls=0,neural_rows=0,solver_calls=0)
            self.stream.write(json.dumps(event)+'\n');self.stream.flush()
            try:
                if number>totals['request_cap']:raise RuntimeError('Fixed realized-primal fixture cap exceeded')
                value=function(*args,**kwargs)
            except BaseException as error:
                totals['failed']+=1
                if not isinstance(error,ValueError):totals['unexpected_errors']+=1
                event=dict(event=prefix+'_fixture_failed',request=number,
                    error=type(error).__name__,message=str(error),expected_input_rejection=isinstance(error,ValueError))
                raise
            else:
                totals['completed']+=1
                event=dict(event=prefix+'_fixture_complete',request=number,result=value)
                return value
            finally:
                elapsed=time.perf_counter()-begun;totals['wall_seconds']+=elapsed
                event.update(wall_seconds=elapsed,controls=0,neural_rows=0,solver_calls=0)
                self.stream.write(json.dumps(event,default=lambda value:value.tolist())+'\n');self.stream.flush()
        return call

    def running_window_call(self,function):
        @functools.wraps(function)
        def call(*args,**kwargs):
            totals=self.running_windows;totals['requested']+=1;number=totals['requested']
            begun=time.perf_counter();inputs=outputs=failure=None
            event=dict(event='running_window_fixture_requested',request=number,
                scope='handwritten complete arrays; pure cotangent arithmetic',controls=0,neural_rows=0)
            self.stream.write(json.dumps(event)+'\n')
            self.stream.flush()
            try:
                if number>totals['request_cap']:
                    totals['rejected']+=1;raise RuntimeError('Fixed running-window fixture cap exceeded')
                inputs=self.hybrid_archive(f'running_window_{number:03d}_inputs.npz',dict(args=args,kwargs=kwargs))
                result=function(*args,**kwargs)
                outputs=self.hybrid_archive(f'running_window_{number:03d}_outputs.npz',result)
            except BaseException as error:
                totals['failed']+=1
                expected=isinstance(error,ValueError) and 'absolute integer clock window' in str(error)
                totals['expected_range_errors' if expected else 'unexpected_errors']+=1
                detail=dict(error=type(error).__name__,message=str(error),expected_range_error=expected)
                try:failure=self.hybrid_archive(f'running_window_{number:03d}_failure.npz',detail)
                except BaseException as archive_error:detail['failure_archive_error']=str(archive_error)
                event=dict(event='running_window_fixture_failed',request=number,inputs=inputs,
                    failure_raw=failure,**detail)
                raise
            else:
                totals['completed']+=1
                event=dict(event='running_window_fixture_complete',request=number,inputs=inputs,outputs=outputs)
                return result
            finally:
                elapsed=time.perf_counter()-begun;totals['wall_seconds']+=elapsed
                totals['raw_archives'].append(dict(request=number,inputs=inputs,outputs=outputs,failure=failure))
                event.update(wall_seconds=elapsed,controls=0,neural_rows=0)
                self.stream.write(json.dumps(event)+'\n');self.stream.flush()
        return call

    def cost_risk_call(self,function):
        @functools.wraps(function)
        def call(*args,**kwargs):
            totals=self.cost_risk;totals['requested']+=1;number=totals['requested']
            begun=time.perf_counter();projection_work={};inputs=outputs=None
            event=dict(event='cost_risk_fixture_requested',request=number,controls=0,neural_rows=0)
            self.stream.write(json.dumps(event)+'\n');self.stream.flush()
            try:
                if number>totals['request_cap']:raise RuntimeError('Fixed cost-risk projection fixture cap exceeded')
                inputs=self.hybrid_archive(f'cost_risk_{number:03d}_inputs.npz',dict(args=args,kwargs=kwargs))
                projection_work=None
                result=function(*args,**kwargs);projection_work=result[1]['projection_work']
                if not isinstance(projection_work,dict):raise ValueError('Missing exposed cost-risk projection work')
                outputs=self.hybrid_archive(f'cost_risk_{number:03d}_outputs.npz',result)
            except BaseException as error:
                totals['failed']+=1;projection_work=getattr(error,'projection_work',projection_work)
                if projection_work is None:totals['unknown_partial_requests'].append(number)
                event=dict(event='cost_risk_fixture_failed',request=number,inputs=inputs,
                    error=type(error).__name__,message=str(error),projection_work=projection_work)
                raise
            else:
                totals['completed']+=1
                event=dict(event='cost_risk_fixture_complete',request=number,inputs=inputs,
                    outputs=outputs,projection_work=projection_work)
                return result
            finally:
                elapsed=time.perf_counter()-begun;totals['wall_seconds']+=elapsed
                if isinstance(projection_work,dict):
                    for key,value in projection_work.items():
                        if isinstance(value,(int,float)):
                            known=totals['exposed_projection_work'];known[key]=known.get(key,0)+value
                totals['raw_archives'].append(dict(request=number,inputs=inputs,outputs=outputs))
                event.update(wall_seconds=elapsed,controls=0,neural_rows=0)
                self.stream.write(json.dumps(event)+'\n');self.stream.flush()
        return call

    def hybrid_archive(self, name, value):
        """Store a complete nested fixture as non-pickled arrays and JSON."""
        import numpy as np
        arrays={}
        def pack(item):
            if is_dataclass(item) and not isinstance(item,type):
                return {'dataclass_type':type(item).__qualname__,'fields':pack(asdict(item))}
            if isinstance(item,SimpleNamespace):
                return {'namespace_fields':pack(vars(item))}
            if isinstance(item,np.ndarray):
                if item.dtype.hasobject:raise ValueError('Object arrays are forbidden in fixture archives')
                key=f'array_{len(arrays):04d}';arrays[key]=item
                return {'npz_array':key}
            if isinstance(item,np.generic):return item.item()
            if isinstance(item,dict):return {key:pack(v) for key,v in item.items()}
            if isinstance(item,tuple):return {'tuple_items':[pack(v) for v in item]}
            if isinstance(item,list):return [pack(v) for v in item]
            return item
        metadata=dict(schema='nested-synthetic-fixture-v1',value=pack(value))
        arrays['metadata_json']=np.array(json.dumps(metadata,sort_keys=True,allow_nan=False))
        path=self.output/'hybrid_fixtures'/name;path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
        return dict(file=str(path.relative_to(self.output)),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def hybrid_call(self,function):
        @functools.wraps(function)
        def call(*args,**kwargs):
            totals=self.hybrid;totals['requested']+=1;request=totals['requested']
            self.stream.write(json.dumps(dict(event='hybrid_fixture_requested',request=request,
                scope='synthetic arithmetic',controls=0,neural_rows=0))+'\n');self.stream.flush()
            begun=time.perf_counter();hybrid_work={};input_ref=None;output_ref=None;failure_ref=None;helper_returned=False
            try:
                if request>totals['request_cap']:
                    totals['rejected']+=1;raise RuntimeError('Fixed hybrid-gradient fixture cap exceeded')
                # Input draw declarations count all generated fixture data,
                # even if a helper later fails before processing every group.
                groups=args[0] if args else kwargs.get('context_groups',[])
                if isinstance(groups,(list,tuple)):
                    for group in groups:
                        if not isinstance(group,dict):continue
                        draws=group.get('random_draws')
                        if type(draws) is int and draws>=0:totals['caller_declared_random_draws']+=draws
                        shape=getattr(group.get('continuous_pathwise_gradients'),'shape',())
                        if len(shape) in (2,3):totals['requested_estimator_rows']+=shape[0]*(shape[1] if len(shape)==3 else 1)
                input_ref=self.hybrid_archive(f'request_{request:03d}_inputs.npz',dict(args=args,kwargs=kwargs))
                self.stream.write(json.dumps(dict(event='hybrid_fixture_inputs',request=request,raw=input_ref,
                    caller_declared_random_draws=totals['caller_declared_random_draws'],
                    synthetic_estimator_rows=totals['requested_estimator_rows'],controls=0,neural_rows=0))+'\n');self.stream.flush()
                hybrid_work=None
                result=function(*args,**kwargs);helper_returned=True;totals['helper_returns']+=1;hybrid_work=result['work']
                if not isinstance(hybrid_work,dict):raise ValueError('Missing exposed hybrid arithmetic work')
                output_ref=self.hybrid_archive(f'request_{request:03d}_outputs.npz',result)
            except BaseException as error:
                totals['failed']+=1;hybrid_work=getattr(error,'hybrid_work',hybrid_work)
                if hybrid_work is None:totals['unknown_partial_requests'].append(request)
                failure=dict(error=type(error).__name__,message=str(error),hybrid_work=hybrid_work,
                    wall_seconds=time.perf_counter()-begun)
                try:failure_ref=self.hybrid_archive(f'request_{request:03d}_failure.npz',failure)
                except BaseException as archive_error:failure['failure_archive_error']=str(archive_error)
                event=dict(event='hybrid_fixture_failed',request=request,inputs=input_ref,failure_raw=failure_ref,
                    controls=0,neural_rows=0,**failure)
                raise
            else:
                totals['completed']+=1
                event=dict(event='hybrid_fixture_complete',request=request,inputs=input_ref,outputs=output_ref,
                    hybrid_work=hybrid_work,controls=0,neural_rows=0)
                return result
            finally:
                elapsed=time.perf_counter()-begun;totals['wall_seconds']+=elapsed
                if isinstance(hybrid_work,dict):
                    for key,value in hybrid_work.items():
                        if isinstance(value,(int,float)):
                            known=totals['exposed_hybrid_work'];known[key]=known.get(key,0)+value
                    if helper_returned:totals['completed_estimator_rows']+=hybrid_work.get('trajectory_samples',0)
                    totals['helper_random_draws']+=hybrid_work.get('helper_random_draws',0)
                    totals['helper_rng_calls']+=hybrid_work.get('helper_rng_calls',0)
                totals['raw_archives'].append(dict(request=request,inputs=input_ref,outputs=output_ref,failure=failure_ref))
                event['wall_seconds']=elapsed;self.stream.write(json.dumps(event)+'\n');self.stream.flush()
        return call

    def ray_call(self,function,*,common=False):
        @functools.wraps(function)
        def call(*args,**kwargs):
            totals=self.common_rays if common else self.rays
            prefix='common_ray_fixture' if common else 'ray_fixture'
            totals['requested']+=1;request=totals['requested'];ray_work=None
            self.stream.write(json.dumps(dict(event=prefix+'_requested',request=request,controls=0))+'\n');self.stream.flush()
            begun=time.perf_counter()
            try:
                if request>totals['request_cap']:
                    ray_work={};raise RuntimeError('Fixed feasibility-ray fixture cap exceeded')
                result=function(*args,**kwargs);ray_work=result[1]['ray_work']
                if not isinstance(ray_work,dict):raise ValueError('Missing exposed ray work')
            except BaseException as error:
                totals['failed']+=1;ray_work=getattr(error,'ray_work',ray_work)
                if ray_work is None:totals['unknown_partial_requests'].append(request)
                event=dict(event=prefix+'_failed',request=request,error=type(error).__name__,
                    message=str(error),ray_work=ray_work,controls=0)
                raise
            else:
                totals['completed']+=1
                event=dict(event=prefix+'_complete',request=request,ray_work=ray_work,
                    primal_admissible=result[1]['primal_admissible'],controls=0)
                return result
            finally:
                elapsed=time.perf_counter()-begun;totals['wall_seconds']+=elapsed
                if isinstance(ray_work,dict):
                    for key,value in ray_work.items():
                        if isinstance(value,(int,float)):
                            known=totals['exposed_ray_work'];known[key]=known.get(key,0)+value
                event['wall_seconds']=elapsed;self.stream.write(json.dumps(event)+'\n');self.stream.flush()
        return call

    def performance_call(self,function):
        @functools.wraps(function)
        def call(*args,**kwargs):
            totals=self.performance;totals['requested']+=1;request=totals['requested']
            self.stream.write(json.dumps(dict(event='performance_fixture_requested',request=request,
                method='norm_bounded_performance_dual',controls=0))+'\n');self.stream.flush()
            begun=time.perf_counter();solver_work=None
            try:
                if request>totals['request_cap']:
                    totals['rejected']+=1;solver_work={}
                    raise RuntimeError('Fixed performance-solver fixture cap exceeded')
                result=function(*args,**kwargs);certificate=result[1]
                solver_work=certificate['solver_work'];certified=bool(certificate['certified'])
                if not isinstance(solver_work,dict):raise ValueError('Missing exposed solver work')
            except BaseException as error:
                totals['failed']+=1;solver_work=getattr(error,'solver_work',solver_work)
                if solver_work is None:totals['unknown_partial_requests'].append(request)
                event=dict(event='performance_fixture_failed',request=request,error=type(error).__name__,
                    message=str(error),solver_work=solver_work,controls=0)
                raise
            else:
                totals['completed']+=1;totals['certified' if certified else 'uncertified']+=1
                event=dict(event='performance_fixture_complete',request=request,certified=certified,
                    solver_status=certificate['solver_status'],solver_work=solver_work,controls=0)
                return result
            finally:
                elapsed=time.perf_counter()-begun;totals['wall_seconds']+=elapsed
                if isinstance(solver_work,dict):
                    for key,value in solver_work.items():
                        if isinstance(value,(int,float)):
                            known=totals['exposed_solver_work'];known[key]=known.get(key,0)+value
                        else:totals['unobserved_solver_work'][key]=value
                event['wall_seconds']=elapsed
                self.stream.write(json.dumps(event)+'\n');self.stream.flush()
        return call

    def wrap(self, function, kind):
        import numpy as np
        @functools.wraps(function)
        def call(env, *args, **kwargs):
            if self.depth:
                return function(env, *args, **kwargs)
            count = int(np.prod(np.shape(args[1])[:2])) if kind == 'batch_rollout' else 1
            if self.requested + count > self.cap:
                raise RuntimeError('Fixed fixture-control cap exceeded')
            self.requested += count
            self.stream.write(json.dumps(dict(event='requested', kind=kind,
                controls=count, fixture=type(env).__name__))+'\n')
            self.stream.flush()
            started = time.perf_counter()
            self.depth += 1
            try:
                result = function(env, *args, **kwargs)
            except BaseException as error:
                self.failed += 1
                self.stream.write(json.dumps(dict(event='failed', kind=kind,
                    requested_controls=count, completed_controls=None,
                    error=type(error).__name__, wall_seconds=time.perf_counter()-started))+'\n')
                raise
            else:
                self.completed += count
                self.stream.write(json.dumps(dict(event='completed', kind=kind,
                    controls=count, wall_seconds=time.perf_counter()-started))+'\n')
                return result
            finally:
                self.depth -= 1
                self.stream.flush()
        return call

    def pytest_collection_modifyitems(self, items):
        from . import coupling,credit
        realized_function=credit.realized_primal_certificate
        self.originals.append((credit,'realized_primal_certificate',realized_function))
        credit.realized_primal_certificate=self.realized_primal_call(realized_function)
        excess_function=credit.teacher_excess_nonregression
        self.originals.append((credit,'teacher_excess_nonregression',excess_function))
        credit.teacher_excess_nonregression=self.realized_primal_call(excess_function,
            totals=self.teacher_excess,prefix='teacher_excess')
        projection_function=coupling.project_ending_and_ordinary_step
        @functools.wraps(projection_function)
        def projection(*args,**kwargs):
            self.projections['requested']+=1
            if self.projections['requested']>100:raise RuntimeError('Fixed arithmetic QP fixture cap exceeded')
            self.stream.write(json.dumps(dict(event='projection_fixture_requested',method='two_halfspace_active_set'))+'\n');self.stream.flush()
            try:result=projection_function(*args,**kwargs)
            except BaseException as error:
                self.projections['failed']+=1
                self.stream.write(json.dumps(dict(event='projection_fixture_failed',error=type(error).__name__,partial_solver_work='unknown'))+'\n')
                self.stream.flush();raise
            self.projections['completed']+=1
            for key,value in result[1]['solver_work'].items():self.projections[key]+=value
            self.stream.write(json.dumps(dict(event='projection_fixture_complete',solver_work=result[1]['solver_work']))+'\n');self.stream.flush()
            return result
        self.originals.append((coupling,'project_ending_and_ordinary_step',projection_function))
        coupling.project_ending_and_ordinary_step=projection
        performance_function=coupling.performance_feasibility_step
        self.originals.append((coupling,'performance_feasibility_step',performance_function))
        coupling.performance_feasibility_step=self.performance_call(performance_function)
        ray_function=coupling.bounded_feasibility_ray
        self.originals.append((coupling,'bounded_feasibility_ray',ray_function))
        coupling.bounded_feasibility_ray=self.ray_call(ray_function)
        common_function=coupling.bounded_common_descent_ray
        self.originals.append((coupling,'bounded_common_descent_ray',common_function))
        coupling.bounded_common_descent_ray=self.ray_call(common_function,common=True)
        hybrid_function=credit.hybrid_outcome_gradient
        self.originals.append((credit,'hybrid_outcome_gradient',hybrid_function))
        credit.hybrid_outcome_gradient=self.hybrid_call(hybrid_function)
        risk_function=credit.projected_cost_risk_step
        self.originals.append((credit,'projected_cost_risk_step',risk_function))
        credit.projected_cost_risk_step=self.cost_risk_call(risk_function)
        window_function=credit.running_window_cotangents
        self.originals.append((credit,'running_window_cotangents',window_function))
        credit.running_window_cotangents=self.running_window_call(window_function)
        seen = set()
        for item in items:
            for name in ('FakeAcrobot', 'RecordingEnv', 'LinearEnv'):
                cls = getattr(item.module, name, None)
                if cls is None or cls in seen:
                    continue
                seen.add(cls)
                for method in ('batch_rollout', 'step'):
                    if hasattr(cls, method):
                        original = getattr(cls, method)
                        self.originals.append((cls, method, original))
                        setattr(cls, method, self.wrap(original, method))

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or report.failed or report.skipped:
            self.reports.append(dict(test=report.nodeid, when=report.when, outcome=report.outcome))

    def close(self):
        for cls, method, original in self.originals:
            setattr(cls, method, original)
        self.stream.close()


def validate(root: Path, output: Path):
    """One finite roster, fresh directory; no training or physical simulation."""
    if Path.cwd().resolve() != root.resolve():
        raise ValueError('Run validation from the repository root')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    sources_before=snapshot_sources(root,output,'source_before')
    guard = EnginesForbidden()
    if any(name in sys.modules for name in ('mujoco','torch','warp','mujoco_warp')):
        raise RuntimeError('Run core validation in a fresh process')
    sys.meta_path.insert(0, guard)
    work = FixtureWork(output/'fixture_queries.jsonl')
    exit_code = 1
    try:
        import pytest
        exit_code = int(pytest.main(['-q','-p','no:cacheprovider','--basetemp',
            str(output/'pytest_tmp'),*CORE_TESTS], plugins=[work]))
    finally:
        work.close()
        sys.meta_path.remove(guard)
        sources_after=snapshot_sources(root,output,'source_after')
        before={row['file']:row['sha256'] for row in sources_before}
        after={row['file']:row['sha256'] for row in sources_after}
        changed=[name for name in SOURCE_FILES if before[name]!=after[name]]
        work.performance['expected_request_count_met']=work.performance['requested']==work.performance['expected_requests']
        work.rays['expected_request_count_met']=work.rays['requested']==work.rays['expected_requests']
        work.common_rays['expected_request_count_met']=work.common_rays['requested']==work.common_rays['expected_requests']
        work.hybrid['expected_request_count_met']=work.hybrid['requested']==work.hybrid['expected_requests']
        work.cost_risk['expected_request_count_met']=bool(work.cost_risk['requested']==work.cost_risk['expected_requests']
            and work.cost_risk['completed']==4 and work.cost_risk['failed']==0 and not work.cost_risk['unknown_partial_requests'])
        work.running_windows['expected_request_count_met']=bool(
            work.running_windows['requested']==work.running_windows['expected_requests']
            and work.running_windows['completed']==work.running_windows['expected_completions']
            and work.running_windows['failed']==work.running_windows['expected_failures']
            and work.running_windows['expected_range_errors']==work.running_windows['expected_failures']
            and work.running_windows['unexpected_errors']==work.running_windows['rejected']==0)
        work.realized_primal['expected_request_count_met']=bool(
            work.realized_primal['requested']==work.realized_primal['expected_requests']
            and work.realized_primal['completed']==work.realized_primal['expected_completions']
            and work.realized_primal['failed']==work.realized_primal['expected_failures']
            and work.realized_primal['unexpected_errors']==0)
        work.teacher_excess['expected_request_count_met']=bool(
            work.teacher_excess['requested']==work.teacher_excess['expected_requests']
            and work.teacher_excess['completed']==work.teacher_excess['expected_completions']
            and work.teacher_excess['failed']==work.teacher_excess['expected_failures']
            and work.teacher_excess['unexpected_errors']==0)
        work.hybrid['expected_arithmetic_counts_met']=bool(work.hybrid['completed']==work.hybrid['helper_returns']==1
            and work.hybrid['failed']==0 and not work.hybrid['unknown_partial_requests']
            and work.hybrid['caller_declared_random_draws']==work.hybrid['expected_caller_declared_random_draws']
            and work.hybrid['requested_estimator_rows']==work.hybrid['completed_estimator_rows']==work.hybrid['expected_estimator_rows']
            and work.hybrid['helper_random_draws']==work.hybrid['helper_rng_calls']==0)
        if (changed or not all(v['expected_request_count_met'] for v in (work.performance,work.rays,work.common_rays,work.hybrid,work.cost_risk,work.running_windows,work.realized_primal,work.teacher_excess))
                or not work.hybrid['expected_arithmetic_counts_met']):exit_code=1
        summary = dict(schema='core-validation-v1', exit_code=exit_code,
            roster=list(CORE_TESTS), reports=work.reports,
            wall_seconds=time.perf_counter()-started,
            requested_fixture_controls=work.requested,
            completed_fixture_controls=work.completed, failed_fixture_queries=work.failed,
            failed_query_completed_fixture_controls='unknown; requests retained',
            real_physics_controls=0, actor_rows=0, optimizer_steps=0,
            engine_import_guard=['mujoco','warp','mujoco_warp','torch'])
        summary['arithmetic_projection_fixtures']=dict(work.projections,
            scope='Pure two-halfspace unit fixtures; no actor training or simulation',
            failed_partial_solver_work='unknown; failed intents retained')
        summary['performance_solver_fixtures']=dict(work.performance,
            scope='Pure finite analytic/certificate fixtures; zero simulated controls, actor rows or updates',
            internal_solver_work='Only exposed callbacks/kernels counted; SciPy internals remain unobserved')
        summary['feasibility_ray_fixtures']=dict(work.rays,scope='Pure ray geometry; no simulated controls or actor calls')
        summary['common_descent_fixtures']=dict(work.common_rays,
            scope='Pure fixed-anchor arithmetic only; no neural probe, training or planning authority')
        summary['hybrid_outcome_fixtures']=dict(work.hybrid,
            scope='Synthetic Gaussian gradient arithmetic; estimator rows are neither neural rows nor physics controls')
        summary['cost_risk_projection_fixtures']=dict(work.cost_risk,
            scope='Four analytic halfspace/ball projections; no actor, simulator, or deployment-feasibility claim')
        summary['running_window_cotangent_fixtures']=dict(work.running_windows,
            scope='Four handwritten-array calls: two arithmetic results and two expected range errors; zero model or neural calls')
        summary['realized_primal_fixtures']=dict(work.realized_primal,
            scope='33 handwritten-point calls, including17 expected input rejections; no solve, actor, derivative or control execution')
        summary['teacher_excess_fixtures']=dict(work.teacher_excess,
            scope='27 handwritten-cost checks, including20 expected input rejections; no solve, actor, derivative or control execution')
        summary['source_binding']=dict(unchanged=not changed,changed_files=changed,
            files=list(SOURCE_FILES),before_manifest='source_before.json',after_manifest='source_after.json',
            before_manifest_sha256=hashlib.sha256((output/'source_before.json').read_bytes()).hexdigest(),
            after_manifest_sha256=hashlib.sha256((output/'source_after.json').read_bytes()).hexdigest(),
            bound_but_not_exercised=['src/gps/bidirectional/policy.py','src/gps/bidirectional/learning.py',
                'src/gps/bidirectional/global_learning.py','src/gps/bidirectional/teaching.py','src/gps/bidirectional/study.py'],
            canonical_forward_coverage='Source-bound only; Torch import guard excludes actor numerical/autograd execution')
        with (output/'summary.json').open('x') as stream:
            json.dump(summary,stream,indent=2); stream.write('\n')
    return exit_code
