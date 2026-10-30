"""Executed temporal teaching, without teacher action or state regression.

The reusable step accepts an executor and current teacher episodes. It does not
open another archive, import planning authority, or pin a historical learner.
The small protocol at the bottom tests this interface before a feedback cycle.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .coupling import performance_feasibility_step, primal_step_admissible
from .credit import feedback_gradient, outcome_witness_scales, outcome_witness_terms, unit_loss_gradient
from .ending import EndingExecutor, EndingSnapshot, ScalarArchive
from .evidence import file_sha256, verify_experiment
from .experience import import_qualified_teachers
from .learning import LearningConfig, NeuralPolicy, context_from_episode, save_pt
from .meter import ledger_work_snapshot
from .policy import FamilyGaussianActor, policy_hash
from .search import state_json
from .sensitivity import displaced, parameters


CONFIG = dict(schema='temporal-outcome-teaching-interface-v32', updates=2,
    task=dict(gate=12,delay=12,move=.2,effort=.02,greed=.2,detour=.2,
        detour_steps=5,penalty=20.,ending_y=-.1),
    inference_contract='native_batch1_rows', beta=1.,
    alphas=[.1,.03,.01,.003], fractions=[1.,.5], radius=.01,
    qualification_root=90000000, evaluation_root=91000000,
    qualification_roots=16,evaluation_roots=64,minimum_child_gain=.001,capture_min=.9,
    minimum_training_gain=1e-10,actual_norm_atol=1e-8,
    solver=dict(max_iterations=256,solver_ftol=1e-12,feasibility_atol=1e-8,
        certificate_atol=1e-8,certificate_rtol=1e-8),
    derivative=dict(epsilon=.001,atol=1e-4,rtol=.02),
    max_controls=100000,max_neural_rows=200000,max_backward_calls=128,max_optimizer_calls=0,
    max_verification_controls=100000,max_verification_neural_rows=200000,
    max_verification_backward_calls=128,max_verification_optimizer_calls=0,
    scope='Same-context temporal teaching interface; no acquired-state learning, new search authority, or completed-loop claim')

FULL_CONFIG=deepcopy(CONFIG)
FULL_CONFIG.update(schema='temporal-outcome-coupled-cycle-v33',cycle=True,updates=32,rounds=2,
    qualification_every=4,root_offset=94000000,seed_offset=94000000,
    short_search=dict(K=32,H=4,temperature=.2,sigma=.8,smooth_candidates=8,
        memory_candidates=16,knot_spacing=4,independent_block_lengths=[]),
    full_memory_candidates=16,planner_seed=910000,agreement=1.,agreement_ladder=[.1,1.,10.],
    selector_scales=[1.,1.,1.,20.],
    handoff_times=[0,4,8],handoff_perturbations=[-.01,0.,.01],support_radius=.15,
    control_trial_roots=8,control_confirmation_roots=8,control_evaluation_roots=16,
    minimum_control_gain=.001,ordinary_qualification_roots=16,standalone_roots=64,
    acquisition_strategy='qualified_trial_pairs',teaching_minimum_gain=1e-6,teaching_state_atol=1e-8,
    policy_inference_contract='native_batch1_rows',temporal_teaching=True,
    execution_mode='commit_prefix_then_actor',commitment_state_atol=1e-5,
    max_controls=10000000,max_neural_rows=2000000,max_backward_calls=4000,
    max_verification_controls=10000000,max_verification_neural_rows=2000000,
    max_verification_backward_calls=4000,
    scope='Finite one-seed temporal-outcome teaching and two actual feedback rounds; Ending only; prior development costs additional; no efficiency claim')

OUTCOME_CONFIG=deepcopy(FULL_CONFIG)
OUTCOME_CONFIG.update(schema='complete-outcome-coupled-cycle-v34',
    complete_outcome_teaching=True,bootstrap_search=True,bootstrap_round_index=100,
    teaching_contexts=8,acquisition_seed_pool=512,updates=16,
    root_offset=130000000,seed_offset=130000000,witness_scale_floor=1e-6,
    max_controls=15000000,max_neural_rows=4000000,
    max_verification_controls=15000000,max_verification_neural_rows=4000000,
    roster_bound=dict(search_phases=3,controller_episodes=342,guided_episodes=219,
        independent_controls_per_episode=29433,guided_tail_rows_per_episode=11511,
        learning_controls=214600,learning_neural_rows=521800,suffix_rows=9072,
        other_context_rows=20000,scalar_controls=12830667,neural_rows=3077256,
        backward_calls=3072,performance_solver_calls=384,
        scope='Conservative complete-roster bound per phase, including bootstrap and both changed-policy rounds; actual work is charged from the ledger'),
    derivative=dict(epsilon=1e-5,atol=1e-4,rtol=.02),
    scope='Finite one-seed complete-outcome teaching with fresh stratified ordinary-context acquisition, bootstrap search and two changed-policy feedback rounds; Ending only; no efficiency claim')

UNIT_CONFIG=deepcopy(OUTCOME_CONFIG)
UNIT_CONFIG.update(schema='complete-outcome-unit-step-cycle-v35',unit_proposal_gradient=True,
    root_offset=140000000,seed_offset=140000000,
    scope='Finite one-seed complete-outcome coupled cycle with unit loss-gradient proximal proposals; bootstrap plus two changed-policy feedback rounds; Ending only; no efficiency claim')

GLOBAL_CONFIG=deepcopy(UNIT_CONFIG)
GLOBAL_CONFIG.update(schema='global-outcome-reference-cycle-v38',global_outcome_teaching=True,
    unit_feasibility_metric=True,stage_reference_budget=True,
    ordinary_training_roots=32,acquired_training_clocks=[2,4,6,8],learner_training_root=6000000,
    minimum_feasibility_gain=1e-10,ordinary_training_atol=1e-10,
    root_offset=180000000,seed_offset=180000000,
    max_neural_rows=6000000,max_verification_neural_rows=6000000,
    credit_interface=dict(ordinary_roots=2,source_arm='raw',source_clock=8,source_episodes=4,
        max_controls=3000,max_neural_rows=6000,max_backward_calls=8,max_optimizer_calls=0,
        scalar_controls_bound=1298,neural_rows_bound=1584,backward_calls_bound=6,
        performance_solver_calls_bound=4,actor_versions=11,failed_credit_checks=2),
    reference_interface=dict(source_schema='global-outcome-unit-ending-cycle-v37',stage=0,arm='guided',
        iteration=7,alpha_index=0,fraction_index=1,ordinary_roots=32,
        max_controls=2000,max_neural_rows=2000,max_backward_calls=0,max_optimizer_calls=0,
        scalar_controls_bound=800,neural_rows_bound=800,backward_calls_bound=0,performance_solver_calls_bound=0),
    roster_bound=dict(search_phases=3,controller_episodes=342,guided_episodes=219,
        independent_controls_per_episode=29433,guided_tail_rows_per_episode=11511,
        learning_controls=1553240,learning_neural_rows=2213720,suffix_rows=9072,
        other_context_rows=20000,scalar_controls=14169307,neural_rows=4769176,
        backward_calls=3648,performance_solver_calls=384,
        scope='Conservative per-phase bound for bootstrap plus two changed-policy rounds;64 own mixed-reset and eight auxiliary trajectories per actor trial;32 separately measured stage-reference outcomes per attempt; four FD actors per stage'),
    scope='Finite one-seed global actor-owned outcome learning with freshly measured same-context stage-reference cost budgets, unit endpoint feasibility, independent search, exact-context teacher bounds, executed trust and two changed-policy feedback rounds; Ending only; no efficiency claim')

# A diagnostic stage, injected into its frozen configuration only. It does not
# change the current full-study protocol or import a frontier for deployment.
ENDPOINT_INTERFACE=dict(source_schema='global-outcome-reference-cycle-v38',stage=0,arm='guided',
    iteration=11,missing_slot=48,selected_slot=56,
    derivative=dict(epsilon=1e-5,atol=1e-4,rtol=.02),
    gradient_match_atol=1e-8,gradient_match_rtol=1e-6,
    max_controls=256,max_neural_rows=512,max_backward_calls=8,max_optimizer_calls=0,
    scalar_controls_bound=108,neural_rows_bound=144,backward_calls_bound=4,
    parameter_constructions_bound=2,queries_bound=6,performance_solver_calls_bound=0)

# The previous full protocol remains an exact reproduction input. This version
# changes endpoint constraint coverage, not the task, actor loss or promotion.
COVERAGE_CONFIG=deepcopy(GLOBAL_CONFIG)
COVERAGE_CONFIG.update(schema='global-outcome-active-endpoints-cycle-v40',
    root_offset=190000000,seed_offset=190000000,
    active_endpoint_constraints=dict(max_gradients=9,initial_failed_max=8,
        add_after_fraction_pair=True,actual_margins=True,numerical_reserve=0.),
    scope='Finite one-seed bidirectional loop with baseline-owned active ending inequalities, complete outcome teaching, protected independent search and unchanged fresh qualification; Ending only; no efficiency claim')
COVERAGE_CONFIG['roster_bound'].update(learning_neural_rows=2613080,neural_rows=5168536,
    scope='Conservative per-phase three-stage bound; at most nine baseline-group endpoint packets per attempt, four solves and eight complete candidate trials; other v38 work unchanged')

COVERAGE_INTERFACE=dict(source_schema='global-outcome-reference-cycle-v38',stage=0,arm='guided',iteration=11,
    max_controls=16000,max_neural_rows=28000,max_backward_calls=40,max_optimizer_calls=0,
    scalar_controls_bound=15560,neural_rows_bound=26600,backward_calls_bound=38,
    parameter_constructions_bound=8,queries_bound=55,performance_solver_calls_bound=4,
    check_credit=False,scope='One actual active-constraint update on the exact stopped v38 actor/roster; no qualification or search authority')

GROUP_COVERAGE_CONFIG=deepcopy(COVERAGE_CONFIG)
GROUP_COVERAGE_CONFIG.update(schema='global-outcome-group-endpoints-cycle-v41',root_offset=200000000,seed_offset=200000000,
    scope='Finite one-seed bidirectional loop with original-query-group endpoint coverage and one measured pivot per trial pair; original outcome loss, protected independent search and qualification unchanged; Ending only')
GROUP_COVERAGE_CONFIG['active_endpoint_constraints'].update(initial_strategy='query_group_worst',maximum_new_after_pair=1)

ADMISSION_CONFIG=deepcopy(GROUP_COVERAGE_CONFIG)
ADMISSION_CONFIG.update(schema='global-outcome-realized-admission-cycle-v42',root_offset=210000000,seed_offset=210000000,
    realized_candidate_primal=True,
    scope='Finite bidirectional loop with original local primal tolerance evaluated at each executed actor displacement; complete outcome teaching, independent search and fresh qualification unchanged; Ending only')

ADMISSION_INTERFACE=dict(source_schema='global-outcome-group-endpoints-cycle-v41',stage=0,arm='guided',iteration=3,
    max_controls=16000,max_neural_rows=28000,max_backward_calls=40,max_optimizer_calls=0,
    scalar_controls_bound=15560,neural_rows_bound=26600,backward_calls_bound=38,
    parameter_constructions_bound=8,queries_bound=55,performance_solver_calls_bound=4,
    check_credit=False,scope='One complete actual update at the fixed v41 stopping case with realized-candidate primal admission; no deployment qualification or planning authority')

CONSTRAINT_ANCHOR_INTERFACE=dict(ADMISSION_INTERFACE,
    source_schema='global-outcome-realized-admission-cycle-v42',iteration=13,
    scope='Bound v42 GUIDE13 before actor and complete original contexts; inherited evidence supplies no authority')
CONSTRAINT_PROBE_INTERFACE=dict(max_controls=4000,max_neural_rows=4000,
    max_backward_calls=0,max_optimizer_calls=0,scalar_controls_bound=3280,
    neural_rows_bound=3280,backward_calls_bound=0,parameter_constructions_bound=2,
    queries_bound=12,performance_solver_calls_bound=0,fractions=[1.,.5],
    scope='Two executed ordinary-first candidates from one previously paid teacher-constrained proposal; no fresh qualification or search authority')

BRANCH_CREDIT_INTERFACE=dict(max_controls=75,max_neural_rows=75,
    max_backward_calls=0,max_optimizer_calls=0,scalar_controls_bound=75,
    neural_rows_bound=75,backward_calls_bound=0,parameter_constructions_bound=2,
    queries_bound=3,performance_solver_calls_bound=0,fractions=[-.125,-.0625],
    auxiliary_row=0,projection_relative_atol=.01,derivative=dict(atol=1e-4,rtol=.02),
    baseline_replay='deterministic_control_bits_log_density_diagnostic_v2',
    scope='Fresh native single-row baseline and two negative fractions of the inherited proposal at the fixed GUIDE13 actor; same-branch credit diagnostic only; no learning, qualification or search authority')

# A new development formulation; the active v42 configuration is retained.
# The reserve is a learner target, not a changed task threshold or task cost.
INTERIOR_CONFIG=deepcopy(ADMISSION_CONFIG)
INTERIOR_CONFIG.update(schema='global-outcome-interior-ending-cycle-v43',
    root_offset=220000000,seed_offset=220000000,training_ending_margin=1e-4,
    ordinary_teacher_constraints=True,requires_interior_interface=True,
    scope='Interior-ending restoration followed by ordinary-return learning under measured teacher constraints; independent search, task costs and fresh qualification unchanged; Ending development only')
INTERIOR_CONFIG['active_endpoint_constraints']['numerical_reserve']=1e-4
INTERIOR_INTERFACE=dict(CONSTRAINT_ANCHOR_INTERFACE,
    scope='One actual v43 update at the exact GUIDE13 before actor; actual interior-ending progress and original capture/cost protection; no standalone qualification or planning authority')


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(state_json(value),stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')


def load_config(path):
    value=json.loads(Path(path).read_text());base=deepcopy(value);base.pop('source',None)
    stage=base.pop('execution_stage',None)
    if stage=='endpoint_coverage_interface' and base.pop('endpoint_interface',None)!=ENDPOINT_INTERFACE:
        raise ValueError('The declared finite endpoint-coverage diagnostic is required')
    if stage=='branch_credit_probe':
        if base.pop('branch_credit_interface',None)!=BRANCH_CREDIT_INTERFACE or base!=ADMISSION_CONFIG:
            raise ValueError('The declared finite native branch-credit probe is required')
    elif stage=='ordinary_constraints_probe':
        if (base.pop('admission_interface',None)!=CONSTRAINT_ANCHOR_INTERFACE
                or base.pop('constraint_probe_interface',None)!=CONSTRAINT_PROBE_INTERFACE or base!=ADMISSION_CONFIG):
            raise ValueError('The declared finite ordinary-first teacher-constraint probe is required')
    elif stage=='realized_admission_update':
        expected=INTERIOR_INTERFACE if base.get('schema')==INTERIOR_CONFIG['schema'] else ADMISSION_INTERFACE
        profile=INTERIOR_CONFIG if base.get('schema')==INTERIOR_CONFIG['schema'] else ADMISSION_CONFIG
        if base.pop('admission_interface',None)!=expected or base!=profile:
            raise ValueError('The declared finite realized-displacement update is required')
    elif stage=='active_ending_update':
        if (base.pop('endpoint_interface',None)!=ENDPOINT_INTERFACE
                or base.pop('coverage_interface',None)!=COVERAGE_INTERFACE or base not in (COVERAGE_CONFIG,GROUP_COVERAGE_CONFIG)):
            raise ValueError('The declared finite actual endpoint update is required')
    elif stage is not None and (stage not in ('global_credit_interface','stage_reference_interface','endpoint_coverage_interface') or base!=GLOBAL_CONFIG):
        raise ValueError('Only the declared global credit component stage is accepted')
    if base not in (CONFIG,FULL_CONFIG,OUTCOME_CONFIG,UNIT_CONFIG,GLOBAL_CONFIG,COVERAGE_CONFIG,GROUP_COVERAGE_CONFIG,ADMISSION_CONFIG,INTERIOR_CONFIG):raise ValueError('Only the declared outcome teaching protocols are supported')
    return value


def bind(source,config):
    """Bind preserved *inputs*, not an unqualified output or inherited trust."""
    source=Path(source).resolve();verify_experiment(source/'source',source)
    if not json.loads((source/'audit.json').read_text())['passed']:
        raise ValueError('Input supplier needs a completed paid audit')
    inputs=source/'inputs';packet=json.loads((inputs/'teachers/manifest.json').read_text())
    if packet['trust_authority_imported']:raise ValueError('Teacher evidence cannot import authority')
    cfg=deepcopy(config);cfg['source']=dict(run=str(source),
        files=[dict(file=str(p.relative_to(inputs)),sha256=file_sha256(p),bytes=p.stat().st_size)
            for p in sorted(inputs.rglob('*')) if p.is_file()],
        input_costs_additional=True,uses_supplier_frontier=False,planning_authority_imported=False)
    return cfg


def _interior_component_proof(directory, config, parent):
    """Read a useful paid component; importing its actor is never admission."""
    import ast
    import hashlib
    directory=Path(directory);files={}
    def checked(name,descriptor=None):
        relative=Path(name);path=directory/relative
        if relative.is_absolute() or '..' in relative.parts or not path.is_file():raise ValueError('Missing or unsafe interior admission evidence')
        row=dict(file=str(relative),sha256=file_sha256(path),bytes=path.stat().st_size)
        if descriptor is not None and (row['sha256']!=descriptor['sha256'] or ('bytes' in descriptor and row['bytes']!=descriptor['bytes'])):
            raise ValueError('Interior admission evidence changed')
        if files.setdefault(str(relative),row)!=row:raise ValueError('Interior admission evidence changed during reading')
        return path
    def read(name):return json.loads(checked(name).read_text())
    def packet(descriptor):
        with np.load(checked('producer/'+descriptor['file'],descriptor),allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
    component=read('config.json');result=read('result.json');audit=read('audit.json');manifest=read('manifest.json');launch=read('launch.json');study=read('producer_study.json')
    base=deepcopy(component);base.pop('source',None)
    if (base.pop('execution_stage',None)!='realized_admission_update' or base.pop('admission_interface',None)!=INTERIOR_INTERFACE
            or base!=INTERIOR_CONFIG or result['schema']!='realized-admission-update-interface-v43'
            or study['schema']!=result['schema'] or not all(result.get(k) is True for k in ('audit_pass','interface_pass','component_pass','useful_executed_update'))
            or result.get('producer_failed') is not False or audit.get('passed') is not True or launch.get('returncode')!=0
            or manifest['config_sha256']!=file_sha256(directory/'config.json') or not all(study['interface_checks'].values())):
        raise ValueError('A closed useful paid v43 actual-update component is required')
    source=component['source']['update_origin'];step=study['step'];lines=step['lines'];selected=next((r for r in lines if r['eligible']),None)
    if (source['source_schema']!=INTERIOR_INTERFACE['source_schema'] or any(source[k]!=INTERIOR_INTERFACE[k] for k in ('stage','arm','iteration'))
            or source['current_actor']!=study['source_actor'] or step['owner']!=study['source_actor']
            or source['incoming_actor']==step['owner'] or source.get('planning_authority_imported') is not False
            or source.get('frontier_authority_imported') is not False or len(lines)!=8 or selected is None
            or [r['tag'] for r in lines if r['selected']]!=[selected['tag']] or step['selected_line']!=selected['tag']
            or not step['accepted'] or not all(selected['checks'].values()) or not selected['eligible']
            or selected['actor_hash']==step['owner'] or study['candidate_actor']!=selected['actor_hash'] or study['planning_influence']!=0.):
        raise ValueError('Interior admission needs the original diagnostic roles and first actual eligible candidate')
    original_files={r['file']:r for r in source['files']}
    original=json.loads(checked('inputs/update_origin/study.json',original_files['study.json']).read_text());old_stage=original['stages'][source['stage']]
    old_step=next(r for r in old_stage['steps'] if r['arm']==source['arm'] and r['iteration']==source['iteration'])
    if (original['schema']!=source['source_schema'] or old_step['tag']!=source['step'] or old_step['owner']!=step['owner']
            or old_stage['incoming_actor']!=source['incoming_actor'] or source['reference_checkpoint_role']!='stage_qualified_checkpoints.parent'
            or source['reference_checkpoint']!=old_stage['qualified_checkpoints']['parent'] or source['current_checkpoint']!=old_step['before_checkpoint']):
        raise ValueError('Interior diagnostic checkpoint roles differ from the bound original GUIDE13 step')
    component_parent=checked('inputs/parent.pt');full_parent=Path(parent)
    parent_state=torch.load(full_parent,map_location='cpu',weights_only=True)
    if (file_sha256(full_parent)!=file_sha256(component_parent) or parent_state['actor']['policy_hash']!=source['incoming_actor']
            or parent_state['actor']['policy_hash']==step['owner']):
        raise ValueError('Full learning must retain the qualified input parent, not the diagnostic frontier')
    critical={r['file']:r for r in manifest['files']};implementation=Path(__file__).resolve().parent;source_hashes={}
    for name in ('credit.py','policy.py','coupling.py','ending.py','global_learning.py'):
        relative='src/gps/bidirectional/'+name;old=checked('source/'+relative,critical[relative]);current=implementation/name
        if name=='global_learning.py':
            def segments(path):
                text=path.read_text();tree=ast.parse(text)
                return {n.name:hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest() for n in tree.body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('metrics','_ActiveEndpoints','global_step')}
            previous,now=segments(old),segments(current)
            if len(previous)!=3 or previous!=now:raise ValueError('Interior component used a different learner implementation')
            source_hashes[name]=previous
        else:
            if file_sha256(old)!=file_sha256(current):raise ValueError('Interior component implementation changed: '+name)
            source_hashes[name]=file_sha256(old)
    before=step['before'];after=selected['after'];reference=step['ordinary_reference'];wanted=set(before['query_ids']+before['auxiliary']['query_ids']+after['query_ids']+after['auxiliary']['query_ids']+reference['query_ids']);requests={};done={};last=None
    with checked('producer/ledger.jsonl').open() as stream:
        for text in stream:
            event=json.loads(text);last=event
            if event.get('query_id') in wanted and event['kind'] in ('query_requested','query_complete'):
                target=requests if event['kind']=='query_requested' else done
                if event['query_id'] in target:raise ValueError('Duplicate interior evidence query')
                target[event['query_id']]=event
    if last is None or last['kind']!='archive_close' or requests.keys()!=wanted or done.keys()!=wanted:raise ValueError('Incomplete interior admission query evidence')
    def measured(qids,owner):
        values=[]
        for q in qids:
            request=requests[q];meta=request['metadata'];raw=packet(done[q]['result']);inputs=packet(request['inputs']);contexts=packet(meta['contexts_ref']);checked('producer/'+meta['actor']['raw']['file'],meta['actor']['raw'])
            if (request['mode']!='executed' or not meta['deterministic'] or not meta['complete_continuation']
                    or meta['actor_hash']!=meta['actor']['actor_hash'] or meta['actor_hash']!=owner
                    or meta['actor'].get('inference_contract')!='native_batch1_rows' or meta['semantics']!='actor_owned_complete'
                    or json.loads(str(inputs['metadata_json']))!=meta or not np.array_equal(inputs['initial_states'],contexts['initial_states'])
                    or not np.array_equal(raw['pre_states'][:,0],inputs['initial_states']) or not np.array_equal(raw['pre_states'][:,1:],raw['states'][:,:-1])
                    or not np.all(raw['states'][:,-1,0]==25.) or request['requested_controls']!=done[q]['completed_controls']):
                raise ValueError('Interior evidence has a different owner, context or continuation')
            values.append(raw)
        costs=np.concatenate([r['components'].sum(axis=(1,2)) for r in values]);y=np.concatenate([r['states'][:,-1,1] for r in values])
        if not np.isfinite(costs).all() or not np.isfinite(y).all():raise ValueError('Nonfinite interior evidence')
        return costs,y,values
    old,y0,old_groups=measured(before['query_ids']+before['auxiliary']['query_ids'],step['owner']);new,y1,new_groups=measured(after['query_ids']+after['auxiliary']['query_ids'],selected['actor_hash']);ref,_,ref_groups=measured(reference['query_ids'],source['incoming_actor'])
    if (len(old)!=72 or len(new)!=72 or len(ref)!=32 or len(old_groups)!=6 or len(new_groups)!=6
            or any(not np.array_equal(a['pre_states'][:,0],b['pre_states'][:,0]) for a,b in zip(old_groups,new_groups))
            or not np.array_equal(ref_groups[0]['pre_states'][:,0],old_groups[0]['pre_states'][:,0])
            or not np.array_equal(old[:64],before['costs']) or not np.array_equal(new[:64],after['costs'])
            or not np.array_equal(old[64:],before['auxiliary']['costs']) or not np.array_equal(new[64:],after['auxiliary']['costs'])):
        raise ValueError('Interior actual before/after cohorts or costs differ')
    rho=config['training_ending_margin'];ending=config['task']['ending_y'];v0=float(max(0.,((y0-ending)+rho).max()));v1=float(max(0.,((y1-ending)+rho).max()));limit=max(float(old[:32].mean()),float(ref.mean()))
    B=np.asarray(before['outcome_budgets']);delta=packet(selected['actual_step'])['delta'];inputs=packet(selected['proposal_inputs']);certificate=selected['proposal_certificate'];point=selected['candidate_primal_certificate'];phase=selected['phase']
    p,Q,A,b=(inputs[k] for k in ('performance_residual','performance_gradients','safety_gradients','safety_bounds'));performance=p+Q@delta;safety=A@delta-b;norm=float(np.linalg.norm(delta));epigraph=max(0.,float(performance.max())) if phase=='feasibility' else 0.
    primal=max(0.,float((performance-epigraph).max()),float(safety.max()) if len(safety) else 0.,norm-config['radius'])
    if (not all(np.isfinite(x).all() for x in (delta,p,Q,A,b,performance,safety,[norm])) or not point['primal_admissible']
            or point['schema']!='realized-candidate-primal-v1' or point['phase']!=phase
            or point['radius']!=config['radius'] or point['feasibility_atol']!=config['solver']['feasibility_atol']
            or not all(certificate['checks'].get(k) is True for k in ('finite_values','backend_success')) or certificate['solver_status']['success'] is not True
            or certificate['schema']!='performance-feasibility-dual-certificate-v1' or certificate['phase']!=phase
            or certificate['radius']!=config['radius'] or certificate['feasibility_atol']!=config['solver']['feasibility_atol']
            or primal>config['solver']['feasibility_atol'] or norm>config['radius']+config['actual_norm_atol']
            or not np.all((y1<=ending)[y0<=ending]) or not np.all(y1<=ending) or new[:32].mean()>limit+config['ordinary_training_atol']):
        raise ValueError('Interior actual primal, physical capture or ordinary protection failed')
    if phase=='feasibility':useful=bool(v0>0. and v0-v1>config['minimum_feasibility_gain'])
    elif phase=='improvement':useful=bool(v0==v1==0. and old[:32].mean()-new[:32].mean()>config['minimum_training_gain']
        and np.all(np.maximum(new[64:]-B,0.)<=np.maximum(old[64:]-B,0.)+config['ordinary_training_atol']))
    else:useful=False
    if not useful:raise ValueError('No measured interior progress or eligible ordinary/teacher update')
    return dict(source_actor=step['owner'],candidate_actor=selected['actor_hash'],parent_actor=source['incoming_actor'],parent_sha256=file_sha256(full_parent),selected_line=selected['tag'],phase=phase,training_violation_before=v0,training_violation_after=v1,ordinary_gain=float(old[:32].mean()-new[:32].mean()),implementation=source_hashes,planning_authority_imported=False,frontier_actor_imported=False,historical_costs_additional=True),list(files.values())


def bind_interior_interface(componentrun,config):
    """Require a useful actual component before freezing a new full v43 run."""
    componentrun=Path(componentrun).resolve();base=deepcopy(config);base.pop('source',None)
    if base!=INTERIOR_CONFIG or config['source'].get('uses_supplier_frontier') is not False or config['source'].get('planning_authority_imported') is not False:
        raise ValueError('Only the declared input-parent v43 full learner consumes interior admission')
    verify_experiment(componentrun/'source',componentrun)
    proof,files=_interior_component_proof(componentrun,config,Path(config['source']['run'])/'inputs/parent.pt')
    result=deepcopy(config);result['source']['interior_interface']=dict(run=str(componentrun),files=files,proof=proof)
    result['source']['files'].extend(dict(row,file='interior_interface/'+row['file']) for row in files)
    return result


def verify_interior_interface(out,config):
    """Recheck copied proof and current learner before any full-run queries."""
    binding=config.get('source',{}).get('interior_interface')
    if not binding:raise ValueError('Full v43 learning requires a useful actual interior component')
    root=Path(out)/'inputs/interior_interface'
    for row in binding['files']:
        relative=Path(row['file']);path=root/relative
        if relative.is_absolute() or '..' in relative.parts or not path.is_file() or path.stat().st_size!=row['bytes'] or file_sha256(path)!=row['sha256']:
            raise ValueError('Copied interior admission input changed')
    proof,_=_interior_component_proof(root,config,Path(out)/'inputs/parent.pt')
    if proof!=binding['proof']:raise ValueError('Interior admission role or implementation changed')
    return proof


def copy_inputs(out,config):
    shutil.copytree(Path(config['source']['run'])/'inputs',Path(out)/'inputs')
    for name in ('reference_origin','endpoint_origin','update_origin','constraint_proposal','branch_credit_origin','interior_interface'):
        origin=config['source'].get(name)
        if not origin:continue
        for row in origin['files']:
            target=Path(out)/'inputs'/name/row['file'];target.parent.mkdir(parents=True,exist_ok=True)
            source=Path(origin['run'])/row['file']
            if 'source_file' in row:
                if name!='branch_credit_origin' or row['file']!='specification.json':
                    raise ValueError('Only the bound external branch-credit specification may override its source path')
                source=Path(row['source_file'])
                if not source.is_absolute() or file_sha256(source)!=row['sha256'] or source.stat().st_size!=row['bytes']:
                    raise ValueError('Bound external branch-credit specification changed')
            shutil.copy2(source,target)
    verify_inputs(out,config)


def verify_inputs(out,config):
    root=Path(out)/'inputs'
    for row in config['source']['files']:
        relative=Path(row['file']);path=root/relative
        if (relative.is_absolute() or '..' in relative.parts or not path.is_file()
                or path.stat().st_size!=row['bytes'] or file_sha256(path)!=row['sha256']):
            raise ValueError('Bound temporal teaching input changed')


def handover_windows(teacher_root,teachers):
    """Recover the event from original executed controls, before actor queries."""
    packet=json.loads((Path(teacher_root)/'manifest.json').read_text())
    ledger=Path(teacher_root)/'origin/producer/ledger.jsonl'
    # The importer retains the original ledger. Never infer a handover from a
    # favorable cost difference, a policy forecast, or future student outcomes.
    events=[json.loads(line) for line in ledger.read_text().splitlines()]
    requests={r['query_id']:r for r in events if r['kind']=='query_requested'}
    commits={r['raw']['file']:r for r in events if r['kind']=='continuation_commit'}
    windows=[]
    for raw,guided in zip(teachers['raw'],teachers['guided']):
        if (not raw['capture'] or not guided['capture'] or not raw['complete'] or not guided['complete']
                or not np.array_equal(raw['states'][0],guided['states'][0])):
            raise ValueError('Paired complete successful teachers from one starting state required')
        original=guided.get('original_query_ids')
        if original is None:
            namespace=packet['query_namespace'];original=[q.removeprefix(namespace) for q in guided['query_ids']]
        found=[]
        for query in original:
            metadata=requests[query]['metadata'];reference=metadata.get('commitment')
            if reference is not None and reference['file'] in commits:
                event=commits[reference['file']]
                request=next(r for r in events if r['id']==event['request_id'])
                found.append((int(request['start']),int(request['start']+request['prefix_horizon'])))
        if not found:raise ValueError('Successful guided execution has no bound actual handover')
        start,stop=min(found)
        if start!=0 or not 0<stop<len(guided['actions']):
            raise ValueError('This interface tests one proper prefix from the ordinary start')
        windows.append((start,stop))
    return windows


def collect(executor,actor,episodes,tag):
    contexts=[context_from_episode(e,executor.task,0,1) for e in episodes]
    states=np.stack([c.state for c in contexts])
    binding=executor.archive.save_npz(f'teaching/{tag}_contexts.npz',initial_states=states,
        teacher_budgets=np.stack([c.budget.costs for c in contexts]),
        teacher_query_ids_json=np.array(json.dumps([c.budget.query_ids for c in contexts])))
    policy=NeuralPolicy(actor,executor.archive,executor.task,inference_contract='native_batch1_rows')
    result=executor.feedback(states,policy,kind=tag,mode='executed',deterministic=True,
        owner_metadata=dict(contexts_ref=binding,inference_contract='native_batch1_rows'))
    return [dict(result=result,contexts=contexts,bindings=binding)]


def metrics(groups,windows,budgets,scales,beta,task):
    raw=groups[0]['result'];J=raw.components.sum(axis=(1,2))
    W=np.array([raw.components[i,(raw.pre_states[i,:,0]>=a)&(raw.pre_states[i,:,0]<b),:2].sum()
        for i,(a,b) in enumerate(windows)])
    residual=np.maximum(W-budgets,0.)/scales['window']
    return dict(costs=J.tolist(),mean_cost=float(J.mean()),prefix_costs=W.tolist(),
        prefix_budgets=budgets.tolist(),prefix_excess=residual.tolist(),
        prefix_penalty=float(np.square(residual).mean()),
        objective=float(J.mean()/scales['complete']+beta*np.square(residual).mean()),
        captures=int((raw.states[:,-1,1]<=task.dynamics.ending_y).sum()),episodes=len(J),
        terminal_y=raw.states[:,-1,1].tolist(),query_ids=[raw.query_id])


def packet(archive,tag,credit):
    arrays={k:v for k,v in credit.items() if isinstance(v,np.ndarray)}
    metadata={k:state_json(v) for k,v in credit.items() if k not in arrays}
    return archive.save_npz(f'teaching/{tag}.npz',**arrays,metadata_json=np.array(json.dumps(metadata,sort_keys=True)))


def outcome_scales(executor,parent,data,tag):
    """Pay for the incoming actor once; freeze common scales for both arms."""
    groups=collect(executor,parent,data['raw'],tag+'_normalization')
    raw=groups[0]['result'];costs=raw.components.sum(axis=(1,2))
    teachers=np.array([[e['components'].sum() for e in data[a]] for a in ('raw','guided')])
    scales=outcome_witness_scales(costs,teachers,floor=1e-6)
    scales.update(query_id=raw.query_id,owner=policy_hash(parent),
        incoming_costs=costs.tolist(),teacher_costs=teachers.tolist(),frozen_for_stage=True)
    executor.archive.append('complete_outcome_scale',tag=tag,scales=scales,
        source_binding=groups[0]['bindings'],planning_influence=0.)
    return scales


def outcome_metrics(groups,budgets,scales,beta,task):
    raw=groups[0]['result'];costs=raw.components.sum(axis=(1,2))
    terms=outcome_witness_terms(costs,budgets,complete_scale=scales['complete'],
        witness_scale=scales['witness'],beta=beta)
    return dict(costs=costs.tolist(),mean_cost=float(costs.mean()),outcome_budgets=budgets.tolist(),
        outcome_excess=terms['outcome_excess'].tolist(),outcome_penalty=terms['outcome_penalty'],
        objective=terms['objective'],captures=int((raw.states[:,-1,1]<=task.dynamics.ending_y).sum()),
        episodes=len(costs),terminal_y=raw.states[:,-1,1].tolist(),query_ids=[raw.query_id])


def outcome_step(executor,actor,episodes,windows,scales,config,tag,*,check_credit=False):
    """Learn against measured whole-teacher outcomes, without action labels.

    Windows retain actual handover provenance only. The loss permits useful
    early expenditure when it improves the complete original outcome.
    """
    archive=executor.archive;groups=collect(executor,actor,episodes,tag+'_base')
    budgets=np.array([e['components'].sum() for e in episodes])
    before=outcome_metrics(groups,budgets,scales,config['beta'],executor.task)
    original=LearningConfig(beta=0.,return_scale=1.);C=[];E=[];refs=[]
    for i in range(len(episodes)):
        cost=feedback_gradient(archive,actor,executor.task,groups,original,beta=0.,
            tag=f'{tag}_cost{i}',cost_row=(0,i))
        ending=feedback_gradient(archive,actor,executor.task,groups,original,
            tag=f'{tag}_ending{i}',ending_row=(0,i))
        C.append(cost['gradient']);E.append(ending['gradient'])
        refs.extend((packet(archive,f'{tag}_cost{i}',cost),packet(archive,f'{tag}_ending{i}',ending)))
    C=np.stack(C);E=np.stack(E)
    terms=outcome_witness_terms(before['costs'],budgets,complete_scale=scales['complete'],
        witness_scale=scales['witness'],beta=config['beta'],context_gradients=C)
    gradient=terms['gradient'];gJ=terms['original_cost_gradient']
    proposal_gradient=gradient;proposal_fields={}
    if config.get('unit_proposal_gradient'):
        proposal_gradient,proposal_norm=unit_loss_gradient(gradient)
        proposal_fields=dict(proposal_gradient=proposal_gradient,proposal_gradient_norm=np.array(proposal_norm))
    raw=archive.save_npz(f'teaching/{tag}_proposal_inputs.npz',gradient=gradient,
        original_cost_gradient=gJ,context_gradients=C,ending_gradients=E,
        outcome_budgets=budgets,costs=np.asarray(before['costs']),teaching_gradient=terms['teaching_gradient'],**proposal_fields)
    credit_check=None
    if check_credit:
        norm=float(np.linalg.norm(gradient))
        if not np.isfinite(norm) or norm<=0.:raise ValueError('Finite nonzero whole-outcome derivative required')
        direction=gradient/norm;probes=[];changed=[];same_branch=True
        for sign,label in ((-1,'minus'),(1,'plus')):
            candidate=displaced(executor,actor,direction,sign*config['derivative']['epsilon'],
                tag+'_outcome_fd_'+label,'finite_difference')
            measured=collect(executor,candidate,episodes,tag+'_outcome_fd_'+label)
            probes.append(outcome_metrics(measured,budgets,scales,config['beta'],executor.task));changed.append(candidate)
            same_branch &= np.array_equal(groups[0]['result'].components[:,:,2:],measured[0]['result'].components[:,:,2:])
        realized=(parameters(changed[1]).astype(float)-parameters(changed[0]).astype(float))/(2*config['derivative']['epsilon'])
        predicted=float(gradient@realized)
        numerical=float((probes[1]['objective']-probes[0]['objective'])/(2*config['derivative']['epsilon']))
        tolerance=config['derivative']['atol']+config['derivative']['rtol']*abs(predicted)
        credit_check=dict(passed=bool(same_branch and np.isfinite([predicted,numerical]).all()
                and abs(predicted-numerical)<=tolerance),same_discrete_branch=bool(same_branch),
            predicted=predicted,numerical=numerical,tolerance=tolerance,probes=probes,
            scope='Whole complete-outcome objective on the same measured discrete branch; a changed branch fails this admission')
        archive.append('complete_outcome_credit_check',tag=tag,result=credit_check)
    p=np.asarray(before['terminal_y'])-executor.task.dynamics.ending_y
    A=np.concatenate((E,gJ[None]));b=np.zeros(len(A));lines=[];selected=None;chosen=None
    if (before['captures']==before['episodes'] and np.isfinite(gradient).all() and np.linalg.norm(gradient)>0
            and (credit_check is None or credit_check['passed'])):
        for ai,alpha in enumerate(config['alphas']):
            request=archive.append('performance_solver_requested',tag=f'{tag}_alpha{ai}',raw=raw,
                alpha=alpha,radius=config['radius'],phase='improvement',settings=config['solver'],
                **(dict(objective_gradient_contract='unit_whole_loss_v0',objective_gradient_norm=proposal_norm)
                   if config.get('unit_proposal_gradient') else {}))
            try:
                delta,certificate=performance_feasibility_step(p,E,A,b,alpha=alpha,
                    radius=config['radius'],phase='improvement',objective_gradient=proposal_gradient,**config['solver'])
            except BaseException as error:
                archive.append('performance_solver_error',request_id=request,error=type(error).__name__,
                    solver_work=getattr(error,'solver_work',None));raise
            delta_ref=archive.save_npz(f'teaching/{tag}_alpha{ai}_proposal.npz',delta=delta)
            archive.append('performance_solver_complete',request_id=request,raw=delta_ref,
                certificate=certificate,solver_work=certificate['solver_work'])
            for fi,fraction in enumerate(config['fractions']):
                label=f'{tag}_alpha{ai}_fraction{fi}'
                candidate=displaced(executor,actor,delta,fraction,label,'line_trial')
                actual=collect(executor,candidate,episodes,label)
                after=outcome_metrics(actual,budgets,scales,config['beta'],executor.task)
                realized=parameters(candidate).astype(float)-parameters(actor).astype(float)
                checks=dict(proposal_primal_admissible=primal_step_admissible(certificate),
                    bounded_actual_step=bool(np.isfinite(realized).all() and np.linalg.norm(realized)
                        <=config['radius']+config['actual_norm_atol']),
                    complete_capture=after['captures']==before['episodes'],
                    original_cost_nonregression=after['mean_cost']<=before['mean_cost']+config['minimum_training_gain'],
                    outcome_penalty_progress=before['outcome_penalty']-after['outcome_penalty']>config['minimum_training_gain'],
                    actual_merit_progress=before['objective']-after['objective']>config['minimum_training_gain'])
                row=dict(tag=label,alpha=alpha,fraction=fraction,actor_hash=policy_hash(candidate),
                    before=before,after=after,checks=checks,eligible=bool(all(checks.values())),selected=False,
                    realized_linear_residuals=(A@realized).tolist(),proposal=delta_ref,
                    actual_step=archive.save_npz(f'teaching/{label}_actual_step.npz',delta=realized))
                row['selected']=bool(row['eligible'] and selected is None);lines.append(row)
                archive.append('temporal_teaching_candidate',**row)
                if row['selected']:selected=row;chosen=candidate
    record=dict(tag=tag,owner=policy_hash(actor),before=before,gradient_packets=refs,
        proposal_inputs=raw,lines=lines,accepted=chosen is not None,
        selected_line=None if selected is None else selected['tag'],
        stop_reason=None if chosen is not None else 'finite_candidate_grid_exhausted',credit_check=credit_check,
        teacher_action_labels=False,teacher_state_targets=False,planning_influence=0.)
    archive.append('temporal_teaching_step',**record)
    return actor if chosen is None else chosen,record


def outcome_coupling_checks(archive,first,scales,config):
    owned={}
    for arm in ('raw','guided'):
        ref=first[arm]['proposal_inputs']
        if file_sha256(archive.path/ref['file'])!=ref['sha256']:raise ValueError('Owned whole-outcome packet changed')
        with np.load(archive.path/ref['file'],allow_pickle=False) as saved:
            owned[arm]={key:saved[key].copy() for key in saved.files}
    raw,guide=owned['raw'],owned['guided']
    exchange=outcome_witness_terms(raw['costs'],guide['outcome_budgets'],
        complete_scale=scales['complete'],witness_scale=scales['witness'],beta=config['beta'],
        context_gradients=raw['context_gradients'])['gradient']
    checks=dict(same_owned_cost_gradient=np.array_equal(raw['original_cost_gradient'],guide['original_cost_gradient']),
        same_owned_costs=np.array_equal(raw['costs'],guide['costs']),
        same_owned_context_gradients=np.array_equal(raw['context_gradients'],guide['context_gradients']),
        budget_exchange_reproduces_guided_gradient=np.array_equal(exchange,guide['gradient']),
        beta_zero_removes_teacher_dependence=np.array_equal(raw['original_cost_gradient']/scales['complete'],guide['original_cost_gradient']/scales['complete']),
        complete_outcome_derivatives_pass=all(first[a]['credit_check']['passed'] for a in ('raw','guided')))
    contrast=float(np.linalg.norm(exchange-raw['gradient']))
    archive.append('complete_outcome_coupling_controls',checks=checks,gradient_contrast=contrast,new_queries=0)
    return checks,contrast


def temporal_step(executor,actor,episodes,windows,scales,config,tag,*,check_credit=False):
    """One teacher-dependent proposal, assessed by complete changed-actor runs.

    This step is reusable inside the actual feedback orchestrator. All input
    episodes must already be verified; only their state, original outcomes and
    provenance reach credit. Their controls are never regression labels.
    """
    archive=executor.archive;groups=collect(executor,actor,episodes,tag+'_base')
    budgets=np.array([e['components'][a:b,:2].sum() for e,(a,b) in zip(episodes,windows)])
    before=metrics(groups,windows,budgets,scales,config['beta'],executor.task)
    original=LearningConfig(beta=0.,return_scale=1.)
    cost=feedback_gradient(archive,actor,executor.task,groups,original,beta=0.,tag=tag+'_cost')
    gJ=cost['gradient'];W=[];E=[];refs=[packet(archive,tag+'_cost',cost)]
    for i,window in enumerate(windows):
        prefix=feedback_gradient(archive,actor,executor.task,groups,original,beta=0.,
            tag=f'{tag}_prefix{i}',cost_row=(0,i),cost_window=window)
        ending=feedback_gradient(archive,actor,executor.task,groups,original,
            tag=f'{tag}_ending{i}',ending_row=(0,i))
        W.append(prefix['gradient']);E.append(ending['gradient'])
        refs.extend([packet(archive,f'{tag}_prefix{i}',prefix),packet(archive,f'{tag}_ending{i}',ending)])
    W=np.stack(W);E=np.stack(E);excess=np.maximum(np.asarray(before['prefix_costs'])-budgets,0.)
    teaching=(2*excess[:,None]*W/(scales['window']**2)).mean(axis=0)
    gradient=gJ/scales['complete']+config['beta']*teaching
    raw=archive.save_npz(f'teaching/{tag}_proposal_inputs.npz',gradient=gradient,
        original_cost_gradient=gJ,prefix_gradients=W,ending_gradients=E,prefix_budgets=budgets,
        prefix_costs=np.asarray(before['prefix_costs']),teaching_gradient=teaching)
    credit_check=None
    if check_credit:
        direction=W.mean(axis=0);norm=float(np.linalg.norm(direction));probes=[];changed=[]
        if not np.isfinite(norm) or norm<=0.:raise ValueError('Finite nonzero preparation derivative required')
        for sign,label in ((-1,'minus'),(1,'plus')):
            candidate=displaced(executor,actor,direction/norm,sign*config['derivative']['epsilon'],
                tag+'_window_fd_'+label,'finite_difference')
            measured=collect(executor,candidate,episodes,tag+'_window_fd_'+label)
            probes.append(metrics(measured,windows,budgets,scales,config['beta'],executor.task));changed.append(candidate)
        realized=(parameters(changed[1]).astype(float)-parameters(changed[0]).astype(float))/(2*config['derivative']['epsilon'])
        predicted=float(W.mean(axis=0)@realized)
        numerical=float((np.mean(probes[1]['prefix_costs'])-np.mean(probes[0]['prefix_costs']))/(2*config['derivative']['epsilon']))
        tolerance=config['derivative']['atol']+config['derivative']['rtol']*abs(predicted)
        credit_check=dict(passed=bool(np.isfinite([predicted,numerical]).all() and abs(predicted-numerical)<=tolerance),
            predicted=predicted,numerical=numerical,tolerance=tolerance,probes=probes,
            scope='Original proper-prefix running cost only; complete probe outcomes retained separately')
        archive.append('temporal_credit_check',tag=tag,result=credit_check)
    # Retain current successful endpoints to first order, rather than aiming at
    # the capture boundary. This is a conservative proposal condition only.
    # Realized linear residuals are diagnostics; actual complete outcomes gate
    # acceptance, avoiding a float32 sign test being mistaken for control risk.
    p=np.asarray(before['terminal_y'])-executor.task.dynamics.ending_y
    A=np.concatenate((E,gJ[None]));b=np.zeros(len(A));lines=[];chosen=None;selected=None
    if (before['captures']==before['episodes'] and np.isfinite(gradient).all() and np.linalg.norm(gradient)>0
            and (credit_check is None or credit_check['passed'])):
        for ai,alpha in enumerate(config['alphas']):
            request=archive.append('performance_solver_requested',tag=f'{tag}_alpha{ai}',raw=raw,
                alpha=alpha,radius=config['radius'],phase='improvement',settings=config['solver'])
            try:
                delta,certificate=performance_feasibility_step(p,E,A,b,alpha=alpha,
                    radius=config['radius'],phase='improvement',objective_gradient=gradient,**config['solver'])
            except BaseException as error:
                archive.append('performance_solver_error',request_id=request,error=type(error).__name__,
                    solver_work=getattr(error,'solver_work',None));raise
            delta_ref=archive.save_npz(f'teaching/{tag}_alpha{ai}_proposal.npz',delta=delta)
            archive.append('performance_solver_complete',request_id=request,raw=delta_ref,
                certificate=certificate,solver_work=certificate['solver_work'])
            for fi,fraction in enumerate(config['fractions']):
                label=f'{tag}_alpha{ai}_fraction{fi}'
                candidate=displaced(executor,actor,delta,fraction,label,'line_trial')
                actual=collect(executor,candidate,episodes,label)
                after=metrics(actual,windows,budgets,scales,config['beta'],executor.task)
                realized=parameters(candidate).astype(float)-parameters(actor).astype(float)
                checks=dict(proposal_primal_admissible=primal_step_admissible(certificate),
                    bounded_actual_step=bool(np.isfinite(realized).all() and np.linalg.norm(realized)
                        <=config['radius']+config['actual_norm_atol']),
                    complete_capture=after['captures']==before['episodes'],
                    original_cost_nonregression=after['mean_cost']<=before['mean_cost']+config['minimum_training_gain'],
                    temporal_penalty_progress=before['prefix_penalty']-after['prefix_penalty']>config['minimum_training_gain'],
                    actual_merit_progress=before['objective']-after['objective']>config['minimum_training_gain'])
                row=dict(tag=label,alpha=alpha,fraction=fraction,actor_hash=policy_hash(candidate),
                    before=before,after=after,checks=checks,eligible=bool(all(checks.values())),selected=False,
                    realized_linear_residuals=(A@realized).tolist(),proposal=delta_ref,
                    actual_step=archive.save_npz(f'teaching/{label}_actual_step.npz',delta=realized))
                row['selected']=bool(row['eligible'] and selected is None);lines.append(row)
                archive.append('temporal_teaching_candidate',**row)
                if row['selected']:selected=row;chosen=candidate
    record=dict(tag=tag,owner=policy_hash(actor),before=before,gradient_packets=refs,
        proposal_inputs=raw,lines=lines,accepted=chosen is not None,
        selected_line=None if selected is None else selected['tag'],
        stop_reason=None if chosen is not None else 'finite_candidate_grid_exhausted',
        credit_check=credit_check,
        teacher_action_labels=False,teacher_state_targets=False,planning_influence=0.)
    archive.append('temporal_teaching_step',**record)
    return actor if chosen is None else chosen,record


def panel(executor,actor,roots,tag):
    states=np.stack([executor.task.reset(r).state for r in roots])
    raw=executor.feedback(states,NeuralPolicy(actor,executor.archive,executor.task,
        inference_contract='native_batch1_rows'),kind=tag,mode='executed',deterministic=True)
    return dict(actor_hash=policy_hash(actor),query_id=raw.query_id,roots=roots,episodes=len(roots),
        captures=int((raw.states[:,-1,1]<=executor.task.dynamics.ending_y).sum()),
        mean_cost=float(raw.components.sum(axis=(1,2)).mean()),costs=raw.components.sum(axis=(1,2)).tolist(),
        inference_contract='native_batch1_rows')


def produce(out,config,phase):
    from .study import configure_archive
    from .performance import _performance_work
    started=time.perf_counter();incoming=torch.load(out/'inputs/parent.pt',map_location='cpu',weights_only=True)
    with ScalarArchive(out/phase) as archive:
        configure_archive(archive,config,verification=phase=='verification')
        executor=EndingExecutor(archive,max_controls=config['max_verification_controls' if phase=='verification' else 'max_controls'],task_config=config['task'])
        teachers=import_qualified_teachers(archive,out/'inputs/teachers',executor.task.task_hash)
        windows=handover_windows(out/'inputs/teachers',teachers)
        parent=FamilyGaussianActor.from_checkpoint(incoming['actor']);actors={a:FamilyGaussianActor.from_checkpoint(incoming['actor']) for a in ('raw','guided')}
        if (len(windows)!=4 or any(e['guiding_policy_hash']!=policy_hash(parent) for e in teachers['guided'])
                or any(e['guiding_policy_hash'] is not None for e in teachers['raw'])):
            raise ValueError('Four original paired teachers must bind the actual incoming policy')
        if config.get('cycle'):
            from .search import MemoryFragment,TrajectoryMemory
            from .study import produce_temporal_cycle
            saved=json.loads((out/'inputs/memory.json').read_text());memory=TrajectoryMemory()
            memory.fragments=[MemoryFragment(np.asarray(f['observation']),f['remaining'],np.asarray(f['actions']),
                f['episode_id'],f['family']) for f in saved['memory']]
            archive.append('temporal_cycle_input',incoming_actor=policy_hash(parent),
                memory_fragments=len(memory.fragments),memory_file_sha256=file_sha256(out/'inputs/memory.json'),
                planning_authority_imported=False,prior_costs_additional=True)
            result,final_full_state=produce_temporal_cycle(executor,incoming,teachers,windows,memory,config)
            if phase=='verification':
                for arm in ('raw','guided'):
                    for i,e in enumerate(teachers[arm]):
                        replay=executor.model_rollout(EndingSnapshot(e['states'][0],executor.task.task_hash),e['actions'][None],
                            kind=f'historical_{arm}_{i}',mode='verification')
                        if any(not np.array_equal(a,b) for a,b in ((replay.states[0],e['states'][1:]),(replay.actions[0],e['actions']),(replay.components[0],e['components']))):
                            raise ValueError('Original executed teacher replay changed')
            result.update(work=dict(archive.totals(),**_performance_work(archive)),wall_seconds=time.perf_counter()-started)
            write(out/('study.json' if phase=='producer' else 'verification_study.json'),result)
            return result
        scales=dict(complete=float(np.mean([e['components'].sum() for a in actors for e in teachers[a]])),
            window=float(np.mean([e['components'][lo:hi,:2].sum() for a in actors for e,(lo,hi) in zip(teachers[a],windows)])))
        if min(scales.values())<=0:raise ValueError('Positive shared measured normalization costs required')
        archive.append('temporal_teaching_initialization',windows=windows,scales=scales,
            current_actor=policy_hash(parent),planning_influence=0.,teacher_action_labels=False,
            unchanged_optimizer_history=True,scope=config['scope'])
        stopped={a:False for a in actors};records=[]
        for iteration in range(1,config['updates']+1):
            for arm in actors:
                if stopped[arm]:continue
                actors[arm],row=temporal_step(executor,actors[arm],teachers[arm],windows,scales,config,
                    f'{arm}_iter{iteration:03d}',check_credit=iteration==1)
                row.update(arm=arm,iteration=iteration);records.append(row);stopped[arm]=not row['accepted']
        first={r['arm']:r for r in records if r['iteration']==1};owned={}
        for arm,row in first.items():
            with np.load(archive.path/row['proposal_inputs']['file'],allow_pickle=False) as data:
                owned[arm]={k:data[k].copy() for k in data.files}
        first_raw,first_guide=owned['raw'],owned['guided']
        exchanged=first_raw['original_cost_gradient']/scales['complete']+config['beta']*(
            2*np.maximum(first_raw['prefix_costs']-first_guide['prefix_budgets'],0.)[:,None]
            *first_raw['prefix_gradients']/scales['window']**2).mean(axis=0)
        coupling_checks=dict(same_complete_owned_cost_gradient=np.array_equal(first_raw['original_cost_gradient'],first_guide['original_cost_gradient']),
            same_owned_prefix_costs=np.array_equal(first_raw['prefix_costs'],first_guide['prefix_costs']),
            same_owned_prefix_derivatives=np.array_equal(first_raw['prefix_gradients'],first_guide['prefix_gradients']),
            budget_exchange_reproduces_guided_gradient=np.array_equal(exchanged,first_guide['gradient']),
            budget_exchange_changes_gradient=bool(np.linalg.norm(exchanged-first_raw['gradient'])>1e-8),
            zero_weight_removes_teacher_dependence=np.array_equal(first_raw['original_cost_gradient']/scales['complete'],first_guide['original_cost_gradient']/scales['complete']),
            prefix_target_active=bool(np.all(first_guide['prefix_costs']>first_guide['prefix_budgets'])),
            window_derivatives_pass=all(r['credit_check']['passed'] for r in first.values()))
        archive.append('temporal_coupling_controls',checks=coupling_checks,
            gradient_contrast=float(np.linalg.norm(exchanged-first_raw['gradient'])),new_queries=0)
        frontiers=dict(parent=parent,**actors);qualification={};evaluation={};checkpoints={}
        for role,actor in frontiers.items():
            state=deepcopy(incoming);state['actor']=actor.policy_state()
            state['temporal_teaching']=dict(windows=windows,accepted_updates=sum(r['accepted'] for r in records if r['arm']==role),
                planning_influence=0.,inference_contract=config['inference_contract'],original_optimizer_history_untouched=True)
            checkpoints[role]=save_pt(archive,f'checkpoints/{role}_frontier.pt',state)
            qualification[role]=panel(executor,actor,list(range(config['qualification_root'],config['qualification_root']+config['qualification_roots'])),role+'_qual')
        base=qualification['parent'];qualified={a:bool(qualification[a]['captures']>=base['captures']
            and qualification[a]['captures']>=np.ceil(config['capture_min']*config['qualification_roots'])
            and base['mean_cost']-qualification[a]['mean_cost']>config['minimum_child_gain']) for a in actors}
        chosen=min((a for a in actors if qualified[a]),key=lambda a:qualification[a]['mean_cost'],default=None)
        for role,actor in frontiers.items():
            evaluation[role]=panel(executor,actor,list(range(config['evaluation_root'],config['evaluation_root']+config['evaluation_roots'])),role+'_frontier_eval')
        scientific=bool(chosen=='guided' and evaluation['guided']['captures']>=max(evaluation[a]['captures'] for a in ('parent','raw'))
            and evaluation['guided']['captures']>=np.ceil(config['capture_min']*config['evaluation_roots'])
            and all(evaluation[a]['mean_cost']-evaluation['guided']['mean_cost']>config['minimum_child_gain'] for a in ('parent','raw')))
        if phase=='verification':
            for arm in actors:
                for i,e in enumerate(teachers[arm]):
                    replay=executor.model_rollout(EndingSnapshot(e['states'][0],executor.task.task_hash),e['actions'][None],kind=f'historical_{arm}_{i}',mode='verification')
                    if any(not np.array_equal(a,b) for a,b in ((replay.states[0],e['states'][1:]),(replay.actions[0],e['actions']),(replay.components[0],e['components']))):
                        raise ValueError('Original executed teacher replay changed')
        result=dict(schema=config['schema'],windows=windows,scales=scales,steps=records,coupling_checks=coupling_checks,
            interface_pass=bool(all(coupling_checks.values()) and any(r['accepted'] for r in records)),
            accepted_updates={a:sum(r['accepted'] for r in records if r['arm']==a) for a in actors},
            qualification=qualification,qualified=qualified,selected_child=chosen,frontier_evaluation=evaluation,
            checkpoints=checkpoints,guided_next_policy_advantage=scientific,beneficial_bidirectional_loop=False,
            planning_influence=0.,scope=config['scope'],work=dict(archive.totals(),**_performance_work(archive)),
            wall_seconds=time.perf_counter()-started)
        attempts=len(records);candidates=sum(len(r['lines']) for r in records)
        fd_actors=2*sum(r['credit_check'] is not None for r in records)
        historical=sum(len(e['actions']) for a in actors for e in teachers[a]) if phase=='verification' else 0
        expected=dict(completed_scalar_controls=100*(attempts+candidates+fd_actors)+6000+historical,
            neural_forward_completed_rows=1000*attempts+100*(candidates+fd_actors)+6000,
            backward_calls=18*attempts,optimizer_steps=0,
            queries=attempts+candidates+fd_actors+6+(8 if phase=='verification' else 0),
            performance_solver_calls=sum(bool(r['lines']) for r in records)*len(config['alphas']))
        result['expected_work']=expected;result['work_roster_checks']={k:result['work'][k]==v for k,v in expected.items()}
        if not all(result['work_roster_checks'].values()):raise ValueError('Temporal stage differs from its declared work roster')
        write(out/('study.json' if phase=='producer' else 'verification_study.json'),result);return result


def run(root,out,config):
    """Paid whole-protocol repeat; no changed criterion after a failed result."""
    from .audit import compare_states
    from .performance import _without_wall
    out=Path(out);verify_experiment(root,out);verify_inputs(out,config)
    if config.get('requires_interior_interface'):verify_interior_interface(out,config)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();results={};failure=None;equal=False
    try:
        for phase in ('producer','verification'):results[phase]=produce(out,config,phase)
        a,b=deepcopy(results['producer']),deepcopy(results['verification'])
        # Each phase has already checked its own declared accounting roster.
        # Verification intentionally adds original-teacher replay controls and
        # queries; those expected counts are not controller/learning outputs.
        for value in (a,b):value.pop('work');value.pop('expected_work',None)
        equal=_without_wall(a)==_without_wall(b);arrays=0;checkpoints=0
        for path in sorted((out/'producer').rglob('*.npz')):
            other=out/'verification'/path.relative_to(out/'producer')
            with np.load(path,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
                equal &= set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
            arrays+=1
        for path in sorted((out/'producer').rglob('*.pt')):
            other=out/'verification'/path.relative_to(out/'producer')
            equal &= compare_states(torch.load(path,map_location='cpu',weights_only=True),torch.load(other,map_location='cpu',weights_only=True));checkpoints+=1
        audit=dict(passed=bool(equal),array_pairs=arrays,full_checkpoint_pairs=checkpoints,
            work=results['verification']['work'],scope=(
                'Paid complete executed-search/acquisition/actor/optimizer-history replay; numerical repeat, not independent physical validation'
                if config.get('complete_outcome_teaching') else
                'Paid complete-actor/model/optimizer-history replay; no search qualification'))
    except BaseException as error:
        import traceback
        phase='verification' if 'producer' in results else 'producer'
        from types import SimpleNamespace
        from .performance import _performance_work
        failed_archive=SimpleNamespace(ledger_path=out/phase/'ledger.jsonl');durable=ledger_work_snapshot(failed_archive)
        if failed_archive.ledger_path.is_file():durable['work'].update(_performance_work(failed_archive))
        failure=dict(phase=phase,type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),durable_work=durable)
        write(out/(phase+'_failure.json'),failure);audit=dict(passed=False,failure=failure)
    verify_inputs(out,config);verify_experiment(root,out);write(out/'audit.json',audit)
    result=dict(schema=config['schema'],producer_failed=failure is not None,audit_pass=audit['passed'],
        interface_pass=bool(audit['passed'] and results['producer'].get('interface_pass',False)),
        guided_next_policy_advantage=bool(audit['passed'] and results['producer'].get('guided_next_policy_advantage',False)),
        beneficial_bidirectional_loop=bool(audit['passed'] and results['producer'].get('beneficial_bidirectional_loop',False)),
        planning_influence=0.,work={p:r['work'] for p,r in results.items()},
        wall_seconds=time.perf_counter()-started,scope=config['scope'])
    write(out/'result.json',result)
    return bool(result['beneficial_bidirectional_loop'] if config.get('cycle') else result['interface_pass'])
