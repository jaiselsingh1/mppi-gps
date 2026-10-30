"""Independent saved-native collocated-PFL audit, NumPy/stdlib only.

No producer, engine, actor, SciPy or CARE solver is imported. Native transition
arrays are not reintegrated. Saved compiled arrays bind the mechanics; the
generic raw auditor separately recomputes all original task costs and endings.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from .shooting_audit import Reader,require,same,canonical,identity,digest,_counts,reconcile_budget,ordinary_state
from .affine_audit import PARENT_SOURCE,PARENT_RESULT,PARENT_AUDIT,CRITICAL

GAINS=[[1.,1.,.1],[1.,1.,1.],[4.,2.,.1],[4.,2.,1.]]
STATES=[[np.pi,0.,0.,0.],[np.pi+.05,.02,.04,-.03],[0.,0.,0.,0.],[.1,-.15,.2,-.3],
    [.4,-.7,3.,-2.],[2.4,1.2,-4.,1.],[-1.3,2*np.pi+.4,1.5,-3.],[1.3,-2*np.pi-.4,-1.5,3.]]


def fixed_config():
    return dict(T=800,dt=.01,max_controls=57600,max_parity_forwards=24,max_mass_expansions=24,gains=GAINS,
        parity_states=STATES,parity_actions=['minus_one','plus_one','pfl_clipped_gain_index_mod4'],parity_atol=1e-9,
        parity_rtol=1e-10,acceleration_atol=1e-8,switch_angle_norm=.3,switch_velocity_norm=2.,Q=[10.,10.,1.,1.],R=.1,
        nominal_seeds=[210000,210001],selection_seeds=list(range(211000,211008)),final_seeds=list(range(212000,212032)),final_min_admissible=29)


def fixed_bound():
    return dict(nominal_controls=6400,selection_controls=25600,final_controls=25600,total=57600,parity_mj_forward_calls=24,
        parity_mj_fullM_calls=24,compiled_parameter_reads=1,care_solves=1,controller_constructions=4,native_model_constructors=1,
        native_environment_data_constructors=2,parity_data_constructors=1,native_rollout_context_constructors=1,
        max_episodes=72,max_episode_reset_restore_forwards=144,max_direct_zero_time_forwards=168)


def finite(a,shape=None):
    a=np.asarray(a);require(a.dtype.kind in 'biuf' and np.isfinite(a).all() and (shape is None or a.shape==shape),'Nonfinite or wrong-shaped saved mechanics array');return a


def parameter_arrays(raw,xml):
    """Recover COM inertias using the saved principal axes, not logged scalars."""
    tree=ET.fromstring(xml);bodies=tree.findall('./worldbody//body');joints=tree.findall('./worldbody//joint');motors=tree.findall('./actuator/motor')
    require([b.attrib['name'] for b in bodies]==['upper_arm','lower_arm'] and [j.attrib['name'] for j in joints]==['shoulder','elbow']
        and len(motors)==1 and motors[0].attrib['name']==motors[0].attrib['joint']=='elbow','Unsupported compiled-object XML ordering')
    shapes=dict(body_mass=(3,),body_ipos=(3,3),body_pos=(3,3),body_inertia=(3,3),body_iquat=(3,4),body_quat=(3,4),
        body_parentid=(3,),jnt_axis=(2,3),jnt_dofadr=(2,),dof_damping=(2,),dof_armature=(2,),actuator_gear=(1,6),actuator_ctrlrange=(1,2),gravity=(3,))
    a={name:finite(raw[name],shape) for name,shape in shapes.items()}
    require(np.array_equal(a['body_parentid'],[0,0,1]) and np.array_equal(a['jnt_dofadr'],[0,1]),'Compiled native body/DOF ancestry changed')
    same(a['jnt_axis'],[[0.,1.,0.],[0.,1.,0.]],'Compiled joint axes changed',atol=1e-12,rtol=0)
    same(a['body_quat'][1:],[[1.,0.,0.,0.]]*2,'Compiled body orientation changed',atol=1e-12,rtol=0)
    require(np.max(abs(a['body_ipos'][1:,:2]))<=1e-12 and np.max(abs(a['body_pos'][1:,:2]))<=1e-12,'Compiled COM/link axes changed')
    same(a['gravity'][:2],[0.,0.],'Compiled gravity axes changed',atol=1e-12,rtol=0)
    same(a['actuator_ctrlrange'],[[-1.,1.]],'Native actuator bounds changed',atol=0,rtol=0)
    same(a['actuator_gear'],[[2.,0.,0.,0.,0.,0.]],'Native gear changed',atol=0,rtol=0)
    same(a['dof_damping'],[.05,.05],'Native damping changed',atol=0,rtol=0)
    require(a['gravity'][2]<0 and np.all(a['body_mass'][1:]>0) and np.all(a['body_inertia'][1:]>0)
        and np.all(a['dof_armature']>=0),'Invalid native physical parameters')
    inertia=[]
    for i in (1,2):
        w,x,y,z=a['body_iquat'][i];require(abs(w*w+x*x+y*y+z*z-1)<=1e-10,'Principal inertia quaternion is not normalized')
        rotation=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
            [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
        inertia.append(float((rotation@np.diag(a['body_inertia'][i])@rotation.T)[1,1]))
    p=dict(m1=float(a['body_mass'][1]),m2=float(a['body_mass'][2]),c1=float(a['body_ipos'][1,2]),c2=float(a['body_ipos'][2,2]),
        l1=float(a['body_pos'][2,2]),I1=inertia[0],I2=inertia[1],gravity=float(-a['gravity'][2]),
        damping1=float(a['dof_damping'][0]),damping2=float(a['dof_damping'][1]),gear=float(a['actuator_gear'][0,0]),
        pivot_height=float(a['body_pos'][1,2]),armature1=float(a['dof_armature'][0]),armature2=float(a['dof_armature'][1]))
    require(min(p[k] for k in ('m1','m2','c1','c2','l1','I1','I2','gravity'))>0,'Invalid recovered native masses/geometry/inertia')
    stored=json.loads(str(raw['parameters_json']));require(set(stored)==set(p),'Saved parameter scalar roster differs')
    for key,value in p.items():same(stored[key],value,'Compiled parameter scalar differs: '+key,atol=1e-12,rtol=1e-12)
    return p


def mechanics(p,q,w):
    q,w=finite(q,(2,)),finite(w,(2,));cos=np.cos(q[1]);sin=np.sin(q[1]);d=p['I2']+p['m2']*p['c2']**2;b=p['m2']*p['l1']*p['c2']
    M=np.array([[p['I1']+p['m1']*p['c1']**2+p['m2']*p['l1']**2+d+2*b*cos+p['armature1'],d+b*cos],[d+b*cos,d+p['armature2']]])
    C=b*sin*np.array([-w[1]*(2*w[0]+w[1]),w[0]*w[0]])
    G=-p['gravity']*np.array([(p['m1']*p['c1']+p['m2']*p['l1'])*np.sin(q[0])+p['m2']*p['c2']*np.sin(q[0]+q[1]),p['m2']*p['c2']*np.sin(q[0]+q[1])])
    D=np.array([p['damping1'],p['damping2']])*w;h=C+G+D
    E=float(.5*np.dot(w,M@w)+p['gravity']*((p['m1']+p['m2'])*p['pivot_height']+(p['m1']*p['c1']+p['m2']*p['l1'])*np.cos(q[0])+p['m2']*p['c2']*np.cos(q.sum())))
    S=float(np.linalg.det(M)/M[0,0]);require(np.linalg.eigvalsh(M).min()>0 and S>0,'Nonpositive recovered mass/Schur inertia')
    return dict(M=M,Cw=C,G=G,damping=D,h=h,S=S,energy=E)


def pfl(p,q,w,gain,target):
    q,w=finite(q,(2,)),finite(w,(2,));k=finite(gain,(3,));m=mechanics(p,q,w)
    v=float(-k[0]*q[1]-k[1]*w[1]+k[2]*w[0]*(m['energy']-target));tau=float(m['S']*v+m['h'][1]-m['M'][1,0]*m['h'][0]/m['M'][0,0])
    u=float(np.clip(tau/p['gear'],-1,1))
    return dict(m,virtual_acceleration=v,tau_requested=tau,normalized_unclipped_action=tau/p['gear'],action=u,
        tau_applied=p['gear']*u,saturated=bool(abs(tau/p['gear'])>1),target_energy=float(target),energy_error=m['energy']-target,
        q2_unwrapped=float(q[1]),gain_triple=k.tolist())


def native_fields(p,terms,u):
    rhs=np.array([-terms['h'][0],p['gear']*u-terms['h'][1]]);a=np.linalg.solve(terms['M'],rhs)
    eliminated=float((p['gear']*u-terms['h'][1]+terms['M'][1,0]*terms['h'][0]/terms['M'][0,0])/terms['S'])
    return dict(torque=np.array([0.,p['gear']*u]),acceleration=a,elbow_elimination=eliminated,
        clip_residual=float((p['gear']*u-terms['tau_requested'])/terms['S']))


def parity_comparison(p,terms,u,native,matrix):
    ex=native_fields(p,terms,u)
    fields=(('mass',matrix,terms['M'],1e-9),('bias',native['bias'],terms['Cw']+terms['G'],1e-9),
        ('passive',native['passive'],-terms['damping'],1e-9),('actuator',native['actuator'],ex['torque'],1e-9),
        ('qacc',native['qacc'],ex['acceleration'],1e-8),('elimination',native['qacc'][1],ex['elbow_elimination'],1e-8),
        ('clipping',native['qacc'][1]-terms['virtual_acceleration'],ex['clip_residual'],1e-8),('energy',np.sum(native['energy']),terms['energy'],1e-8))
    checks={};errors={}
    for name,a,b,atol in fields:
        a,b=np.asarray(a),np.asarray(b);require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid parity field '+name)
        checks[name]=bool(np.allclose(a,b,atol=atol,rtol=1e-10));errors[name]=float(np.max(abs(a-b)))
    return checks,errors,ex


def unpack_native_mass(packed):
    """MuJoCo3.6 qM, two-DOF chain: row diagonal followed by ancestors.

    engine_support.c mj_fullM visits (i,j)=(0,0),(1,1),(1,0).
    This is legacy data.qM, not newer data.M CSR storage.
    """
    a=finite(packed,(3,));return np.array([[a[0],a[2]],[a[2],a[1]]])


def lqr_arrays(p):
    M=mechanics(p,[0.,0.],[0.,0.])['M'];b=p['m2']*p['c2'];a=p['m1']*p['c1']+p['m2']*p['l1']
    stiffness=p['gravity']*np.array([[a+b,b],[b,b]])
    A=np.zeros((4,4));A[:2,2:]=np.eye(2);A[2:,:2]=np.linalg.solve(M,stiffness);A[2:,2:]=-np.linalg.solve(M,np.diag([p['damping1'],p['damping2']]))
    B=np.zeros((4,1));B[2:,0]=np.linalg.solve(M,[0.,p['gear']]);return A,B


def verify_care(A,B,P,Q,R,K):
    A,B,P,Q,K=(finite(x) for x in (A,B,P,Q,K));n=len(A)
    require(A.shape==P.shape==Q.shape==(n,n) and B.shape==(n,1) and K.shape==(1,n) and R>0,'CARE array shapes differ')
    same(P,P.T,'CARE P is asymmetric',atol=1e-8,rtol=1e-10);require(np.linalg.eigvalsh(P).min()>=-1e-8,'CARE P is not positive semidefinite')
    same(K,B.T@P/R,'CARE gain stationarity differs',atol=2e-8,rtol=2e-8)
    residual=A.T@P+P@A-P@B@B.T@P/R+Q
    scale=1+max(float(abs(x).max()) for x in (A.T@P,P@A,P@B@B.T@P/R,Q));relative=float(abs(residual).max()/scale)
    require(relative<=1e-8,'Saved CARE solution has a large normalized residual')
    eig=np.linalg.eigvals(A-B@K);require(np.max(eig.real)<0,'Saved continuous LQR is not stabilizing')
    return residual,eig,relative


def controller_action(p,gain,K,state,target):
    state=finite(state,(5,));q,w=state[1:3],state[3:];terms=pfl(p,q,w,gain,target)
    wrapped=q-2*np.pi*np.floor((q+np.pi)/(2*np.pi));local=bool(np.linalg.norm(wrapped)<.3 and np.linalg.norm(w)<2.)
    raw=-float((K@np.r_[wrapped,w]).item()) if local else terms['normalized_unclipped_action']
    return np.array([np.clip(raw,-1,1)]),dict(mode='upright_lqr' if local else 'collocated_pfl',normalized_unclipped_action=raw,
        tau_requested=p['gear']*raw if local else terms['tau_requested'],saturated=bool(abs(raw)>1),energy=terms['energy'],
        energy_error=terms['energy_error'],q2_unwrapped=float(q[1]),wrapped_angle_norm=float(np.linalg.norm(wrapped)),velocity_norm=float(np.linalg.norm(w)),
        virtual_acceleration=None if local else terms['virtual_acceleration'],schur_inertia=terms['S'],virtual_law_active=not local)


def close_intents(rows):
    """No hidden successful completion of failed/inflight zero-time work."""
    requests={};resolved={};index={r['id']:i for i,r in enumerate(rows)}
    require(len(index)==len(rows),'Duplicate native evidence IDs')
    for row in rows:
        kind=row['kind']
        if kind.startswith('pfl_') and kind.endswith('_requested'):requests[row['id']]=kind
        if kind.startswith('pfl_') and (kind.endswith('_complete') or kind.endswith('_error')) and 'request_id' in row:
            request=row['request_id'];require(request in requests and request not in resolved and index[request]<index[row['id']],'Unbound/duplicate/reversed PFL work completion')
            expected=requests[request][:-len('_requested')]
            require(kind in (expected+'_complete',expected+'_error'),'PFL work resolution kind differs');resolved[request]=row
    return requests,resolved,sorted(set(requests)-set(resolved))


def work_counts(rows):
    kinds=dict(Counter(r['kind'] for r in rows));n=lambda name:kinds.get(name,0);analytic={};saturation={};modes={}
    for row in rows:
        if row['kind'].startswith('pfl_') and row['kind'].endswith('_requested'):
            for key in ('analytic_mass_matrices','implicit_analytic_mass_matrices','analytic_energy_evaluations','energy_evaluations','arithmetic_rows','linear_solves','linear_solves_in_linearization'):
                analytic[key]=analytic.get(key,0)+int(row.get(key,0))
        if row['kind']=='pfl_action_complete':
            key=str(row['index']);saturation[key]=saturation.get(key,0)+int(row['saturated']);key=row['mode'];modes[key]=modes.get(key,0)+1
    _,_,pending=close_intents(rows)
    return dict(event_counts=kinds,parity_mj_forward_requested=n('pfl_native_forward_requested'),parity_mj_forward_completed=n('pfl_native_forward_complete'),
        mj_fullM_requested=n('pfl_native_mass_expansion_requested'),mj_fullM_completed=n('pfl_native_mass_expansion_complete'),
        compiled_parameter_reads_requested=n('pfl_compiled_parameters_read_requested'),compiled_parameter_reads_completed=n('pfl_compiled_parameters_read_complete'),
        care_solves_requested=n('pfl_care_requested'),care_solves_completed=n('pfl_care_complete'),
        controller_constructions_requested=n('pfl_controller_requested'),controller_constructions_completed=n('pfl_controller_complete'),
        analytic_actions_requested=n('pfl_action_requested'),analytic_actions_completed=n('pfl_action_complete'),analytic_action_errors=n('pfl_action_error'),
        episode_reset_forward_requests=n('zero_time_reset_requested'),episode_restore_forward_requests=n('zero_time_restore_requested'),
        direct_zero_time_forward_requests=n('pfl_native_forward_requested')+n('zero_time_reset_requested')+n('zero_time_restore_requested'),
        scratch_resetData_requests=n('pfl_scratch_reset_requested'),scratch_resetData_completions=n('pfl_scratch_reset_complete'),
        scratch_data_constructors_requested=n('pfl_scratch_data_requested'),scratch_data_constructors_completed=n('pfl_scratch_data_complete'),
        analytic_requested_work=analytic,saturated_completed_actions_by_gain_index=saturation,completed_action_modes=modes,pending_pfl_request_ids=pending)


def audit_pfl_reference(run):
    from .audit import audit_archive
    started=time.perf_counter();run=Path(run).resolve();reader=Reader(run);errors=[];pending=[];layers={};counts={};details={};verdict=False
    analytic=dict(compiled_parameter_array_sets=0,instantaneous_parity_cases=0,care_residual_rows=0,actual_controller_action_rows=0,ordinary_reset_rows=0)
    def guard(label,fn):
        try:return fn()
        except Exception as exc:errors.append(label+': '+type(exc).__name__+': '+str(exc));return None
    def finish():
        return dict(schema='independent-native-collocated-pfl-audit-v1',run=str(run),status='FAIL' if errors else 'PENDING' if pending else 'PASS',
            arithmetic_valid=not errors,scientific_verdict='PENDING' if errors or pending else 'PASS' if verdict else 'FAIL',
            native_reference_pass=bool(verdict and not errors and not pending),errors=errors,pending=list(dict.fromkeys(pending)),counts=counts,
            details=details,component_audits=layers,analytic_verification_rows=analytic,input_sha256=reader.hashes,
            auditor_sha256=digest(Path(__file__).read_bytes()),wall_seconds=time.perf_counter()-started,
            additional_physics_controls=0,additional_mj_forward_calls=0,additional_mass_expansions=0,additional_care_solves=0,
            additional_actor_inference_rows=0,additional_fits=0,additional_optimizer_calls=0,
            limitations=['Native transition/compiled-model bytes are source-bound, not independently recompiled or reintegrated.',
                'CARE algebra/eigenvalues are verified without solving CARE; fixed parity only tests saved instantaneous cases.',
                'Classical sampled saturated reference outcomes are separate from MPPI acquisition, GPS and policy learning.',
                'A finite negative is not task infeasibility; ledger checks cannot attest to uninstrumented bypassed work.'])
    if not run.is_dir():pending.append('Native PFL run directory absent');return finish()
    names=('manifest.json','config.json','protocol.json','input_provenance.json','pfl_result.json','native_parity.json','launch.json','stage_started.json','stage_complete.json')
    for name in names:
        if not (run/name).is_file():pending.append('Missing closed artifact: '+name)
    archive=run/'evidence/native'
    if not (archive/'ledger.jsonl').is_file():pending.append('Missing native paid ledger');return finish()
    raw_audit=guard('generic native query/task arithmetic',lambda:audit_archive(archive,source_root=run/'source',allowed_root=run))
    if raw_audit is not None:
        layers['native']=raw_audit;errors.extend('native: '+x for x in raw_audit.get('errors',[]));pending.extend('native: '+x for x in raw_audit.get('pending',[]))
        reader.hashes.update(raw_audit.get('input_sha256',{}))
    blob=guard('native ledger bytes',lambda:reader.file(archive/'ledger.jsonl'))
    rows=guard('native ledger JSON',lambda:[json.loads(line) for line in blob.splitlines()]) if blob is not None else None
    if rows is None or pending:return finish()
    totals=guard('native paid query counts',lambda:_counts(rows))
    if totals is not None:counts.update(totals)
    docs=guard('closed PFL metadata',lambda:{name:reader.json(run/name) for name in names})
    if docs is None or totals is None or raw_audit is None:return finish()
    cfg,manifest,proto,provenance,result,stage=(docs[name] for name in ('config.json','manifest.json','protocol.json','input_provenance.json','pfl_result.json','stage_complete.json'))
    source=manifest['source_fingerprint'];index={r['id']:i for i,r in enumerate(rows)}
    def one(kind,**match):
        found=[r for r in rows if r['kind']==kind and all(r.get(k)==v for k,v in match.items())]
        require(len(found)==1,'Require one '+kind+' '+str(match));return found[0]
    def before(a,b):require(index[a['id']]<index[b['id']],'Native PFL work chronology differs')
    def payload(row):return {k:v for k,v in row.items() if k not in ('id','kind','utc','wall_seconds','request_id')}
    def header():
        require(cfg==fixed_config() and proto['schema']=='native-collocated-pfl-reference-v1' and proto['config']==cfg
            and proto['work_bound']==fixed_bound() and proto['all_traces_excluded_from_teacher'] is True,'Prospective PFL recipe/caps/exclusion differ')
        required={'source/src/gps/bidirectional/pfl_reference.py','source/scripts/diagnostics/run_collocated_pfl_reference.py',
            'source/configs/bidirectional_native_pfl_v0.json','source/docs/bidirectional_native_pfl_protocol_v0.md'}
        refs={r['file']:r for r in manifest['files']};require(len(refs)==len(manifest['files']) and required<=set(refs),'PFL source inventory incomplete/duplicated')
        for name,ref in refs.items():
            path=Path(name);require(not path.is_absolute() and '..' not in path.parts and path.parts[0]=='source','Unsafe frozen source path')
            data=reader.file(run/path);require(digest(data)==ref['sha256'] and len(data)==ref['bytes'],'Frozen PFL source bytes differ')
        pins={'source/src/gps/bidirectional/pfl_reference.py':'39bf6066fc0ae75b8afed1a68d1613fe3b4b7974d8f4721ef59c4dba9b173acb',
            'source/scripts/diagnostics/run_collocated_pfl_reference.py':'6acfed2d6d3f632acec078173624f76f7bb94d4f66f74f896ae293e4cf49a934',
            'source/configs/bidirectional_native_pfl_v0.json':'81e0dd463e9f3a3375ea0614bb4691b7057b884e070e18d813721feee2f53deb'}
        require(all(refs[name]['sha256']==h for name,h in pins.items()),'Auditor covers only the declared PFL producer/CLI/config version')
        require(reader.json(run/'source/configs/bidirectional_native_pfl_v0.json')==cfg,'Repository config and actual frozen config differ')
        for field,name in (('config_sha256','config.json'),('protocol_sha256','protocol.json'),('input_provenance_sha256','input_provenance.json')):
            require(manifest[field]==digest(reader.file(run/name)),'Frozen metadata hash differs')
        fingerprint=identity(dict(files=manifest['files'],config_sha256=manifest['config_sha256'],protocol_sha256=manifest['protocol_sha256'],dependency_versions=manifest['dependency_versions']))
        launch,start=docs['launch.json'],docs['stage_started.json']
        require(source==fingerprint==result['source_fingerprint']==launch['source_fingerprint']==start['source_fingerprint']
            and launch['source_manifest_sha256']==digest(reader.file(run/'manifest.json')) and launch['input_provenance_sha256']==manifest['input_provenance_sha256'],
            'Frozen launch/runtime identity differs')
        require(launch['frozen_reexecution_required'] is True and Path(launch['frozen_source_root']).name=='source'
            and start['stage']=='native_collocated_pfl_reference' and start['max_requested_controls']==57600
            and start['max_parity_mj_forward_calls']==start['max_mj_fullM_calls']==24 and start['no_retry'] is True,'Frozen CLI/reexecution stage differs')
        require(stage['stage']=='native_collocated_pfl_reference' and stage['source_unchanged'] is True and stage['parent_copies_unchanged'] is True
            and stage['result_sha256']==digest(reader.file(run/'pfl_result.json')),'Closed result/source binding differs')
        require(result['schema']=='native-collocated-pfl-reference-result-v1' and result['accounting']==totals and result['archive_accounting']==dict(native=totals)
            and rows[-1]['kind']=='archive_close' and rows[-1]['accounting']==totals,'Paid PFL ledger/result totals differ')
        require(result['actor_inference_rows']==result['adam_steps']==0 and result['gradient_optimizer_used'] is False
            and result['numerical_control_optimization_used'] is False and result['teacher_promotion_performed'] is False
            and result['all_traces_excluded_from_teacher'] is True,'Reference scope/teacher ownership differs')
        require(not any(r['kind'].startswith(('neural_','optimizer_','fit_','actor_','rollout_','shooting_')) for r in rows),'Undeclared policy/optimizer/model-search work')
        require(all(r['mode']=='live' and r['query_kind']=='executed_control' and r['scope']=='native_collocated_pfl_reference'
            and r['requested_controls']==r['requested_physics_controls']==r['requested_physics_frames']==1 for r in rows if r['kind']=='query_requested'),
            'PFL physical work bypassed sole live executor or contains model forecasts')
        shared=reconcile_budget(rows,[],57600);details['shared_budget']=shared
        require(result['shared_budget']['requested_controls']==shared['requested_controls']==totals['requested_controls']<=57600
            and result['budget_matches_paid_ledger'] is True,'Shared cap does not own every physical slot')
        work=work_counts(rows);reported=dict(result['analytic_and_zero_time_work']);reported.pop('scope')
        require(work==reported,'Zero-time/action arithmetic work summary differs');counts['zero_time_and_analytic']=work
        require(work['parity_mj_forward_requested']<=24 and work['mj_fullM_requested']<=24 and work['direct_zero_time_forward_requests']<=168
            and work['care_solves_requested']<=1 and work['compiled_parameter_reads_requested']==1,'Zero-time/CARE/parameter-read cap exceeded')
        if work['pending_pfl_request_ids']:pending.extend('Unclosed PFL intent: '+x for x in work['pending_pfl_request_ids'])
        construct=one('backend_construction_requested');done=one('backend_constructed');before(construct,done)
        require(done['request_id']==construct['id'] and construct['model_constructors']==done['model_constructors']==1
            and construct['native_data_constructors']==done['native_data_constructors']==2 and construct['rollout_context_constructors']==done['rollout_context_constructors']==1
            and construct['mj_state_size_reads']==done['mj_state_size_reads']==1 and construct['nthread']==1
            and done['task_hash']==result['task_hash'] and done['source_fingerprint']==source,'Native backend construction evidence differs')
        require(done['native_model']['nq']==done['native_model']['nv']==2 and done['native_model']['nu']==done['native_model']['frame_skip']==1
            and done['native_model']['timestep']==.01 and done['native_model']['integrator']==1 and done['native_model']['disableflags']&1,'Native backend RK4/constraint boundary differs')
    guard('PFL source/runtime/accounting closure',header)
    def ancestry():
        require(provenance['schema']=='native-pfl-parent-provenance-v1' and provenance['parent_source_fingerprint']==PARENT_SOURCE
            and provenance['parent_result_sha256']==PARENT_RESULT and provenance['parent_audit_sha256']==PARENT_AUDIT,'Wrong fixed native ancestry')
        refs=provenance['files'];copies=provenance['frozen_copies'];require(len(copies)==len(refs) and {r['file'] for r in copies}=={'parent_inputs/'+name for name in refs},'Copied parent metadata roster differs')
        for row in copies:
            name=row['parent_relative'];require(not name.endswith('.npz'),'PFL initialized from old numeric guesses');old=refs[name];data=reader.file(run/row['file'])
            require(digest(data)==row['sha256']==old['sha256'] and len(data)==row['bytes']==old['bytes'],'Parent metadata/source copy differs')
        packet=run/'parent_inputs';old=reader.json(packet/'reference_result.json');oldmanifest=reader.json(packet/'manifest.json');audit=reader.json(packet/'audit_001.json')
        require(digest(reader.file(packet/'reference_result.json'))==PARENT_RESULT and digest(reader.file(packet/'audit_001.json'))==PARENT_AUDIT
            and old['source_fingerprint']==oldmanifest['source_fingerprint']==PARENT_SOURCE and audit['status']=='PASS' and audit['arithmetic_valid'] is True
            and audit['scientific_verdict']=='FAIL' and not audit['errors'] and not audit['pending'],'Closed negative native ancestry changed')
        require(old['status']=='complete' and old['error'] is None and not old['close_errors'] and old['native_reference_pass'] is False,
            'PFL source ancestry is not the fixed negative')
        require(provenance['parent_dependency_versions']==manifest['dependency_versions']==oldmanifest['dependency_versions']
            and result['task_hash']==provenance['parent_task_hash']==old['task_hash'],'Native source/task/runtime changed')
        oldsources={r['file']:r['sha256'] for r in oldmanifest['files']};require(set(provenance['critical_source_sha256'])==set(CRITICAL),'Critical source roster differs')
        for name in CRITICAL:
            value=digest(reader.file(run/'source'/name));require(value==digest(reader.file(packet/'source'/name))==oldsources['source/'+name]==provenance['critical_source_sha256'][name],
                'Native source changed from qualified parent: '+name)
        prefixes=[Path(name).parent for name,h in audit['input_sha256'].items() if Path(name).name=='manifest.json' and h==refs['manifest.json']['sha256']]
        require(any(all(audit['input_sha256'].get(str(prefix/name))==row['sha256'] for name,row in refs.items() if name!='audit_001.json') for prefix in prefixes),
            'Copied ancestry not bound by one original audit namespace')
        excluded={str(Path('evidence/native')/r['solve']['raw']['file']) for r in old['candidates']}
        require(set(provenance['excluded_old_numeric_guess_files'])==excluded and not excluded.intersection(refs),'Old numeric initialization was not fully excluded')
        require(result['inherited_parent_work']==provenance['parent_work']==old['accounting'] and result['inherited_logger_attempt_controls']==352848,
            'Paid ancestry work was omitted or changed')
    guard('unchanged task/compiled-source ancestry',ancestry)
    if result.get('error') or result.get('close_errors') or totals['unknown_controls']:
        pending.append('Operationally incomplete PFL study; no complete scientific certificate');return finish()
    if errors or pending:return finish()
    data={}
    def parameters_and_parity():
        pr,pd=one('pfl_compiled_parameters_read_requested'),one('pfl_compiled_parameters_read_complete');before(pr,pd)
        require(pd['request_id']==pr['id'] and pr['parameter_reads']==pd['parameter_reads']==1 and pr['new_mj_forward_calls']==pr['new_integration_controls']==0
            and pd['raw']==result['compiled_parameters'],'Fresh compiled-parameter read linkage/work differs')
        raw=reader.arrays(archive,pd['raw']);p=parameter_arrays(raw,reader.file(run/'source/assets/acrobot.xml'));data['parameters']=p
        require(pd['parameters']==p,'Logged compiled-parameter summary differs');analytic['compiled_parameter_array_sets']+=1
        tr,td=one('pfl_target_energy_requested'),one('pfl_target_energy_complete');before(pd,tr);before(tr,td)
        target=mechanics(p,[0.,0.],[0.,0.])['energy'];same(td['target_energy'],target,'Upright energy convention differs')
        require(td['request_id']==tr['id'] and tr['analytic_energy_evaluations']==tr['implicit_analytic_mass_matrices']==1 and tr['new_simulator_controls']==0,
            'Target-energy arithmetic work differs');data['target']=target
        begin,end=one('pfl_parity_requested'),one('pfl_parity_complete');before(td,begin);before(begin,end)
        require(begin['cases']==begin['mj_forward_cap']==begin['mj_fullM_cap']==24 and begin['new_integration_controls']==0,'Instantaneous inspection cap differs')
        saved=docs['native_parity.json'];require(saved==result['qualification']==payload(end),'Parity result/event/raw metadata differ')
        require(len(saved['rows'])==saved['mj_forward_calls']==saved['mj_fullM_calls']==24 and saved['new_integration_controls']==0,'Fixed inspection roster/work incomplete')
        same(saved['target_energy'],target,'Parity target differs')
        cr,cd=one('pfl_scratch_data_requested'),one('pfl_scratch_data_complete');before(begin,cr);before(cr,cd)
        require(cd['request_id']==cr['id'] and cr['data_constructors']==1 and cr['new_integration_controls']==0,'Scratch constructor work differs')
        passed=[]
        for i,state in enumerate(STATES):
            plan,planned=one('pfl_parity_plan_requested',state_index=i),one('pfl_parity_plan_complete',state_index=i);before(cd,plan);before(plan,planned)
            require(planned['request_id']==plan['id'] and plan['gain_index']==i%4 and plan['analytic_mass_matrices']==2
                and plan['energy_evaluations']==1 and plan['new_integration_controls']==0,'Parity action-planning work differs')
            q,w=np.asarray(state[:2]),np.asarray(state[2:]);terms=pfl(p,q,w,GAINS[i%4],target)
            same(planned['q'],q,'Parity q differs');same(planned['velocity'],w,'Parity velocity differs')
            require(set(planned['terms'])==set(terms),'Declared PFL instantaneous field roster differs')
            for key,value in terms.items():same(planned['terms'][key],value,'PFL planned analytic field differs: '+key,atol=1e-10,rtol=1e-10)
            for j,u in enumerate((-1.,1.,terms['action'])):
                case=3*i+j;row=saved['rows'][case]
                rr,rd=one('pfl_scratch_reset_requested',case=case),one('pfl_scratch_reset_complete',case=case)
                fr,fd=one('pfl_native_forward_requested',case=case),one('pfl_native_forward_complete',case=case)
                mr,md=one('pfl_native_mass_expansion_requested',case=case),one('pfl_native_mass_expansion_complete',case=case)
                ar,ad=one('pfl_parity_arithmetic_requested',case=case),one('pfl_parity_arithmetic_complete',case=case)
                for a,b in ((planned,rr),(rr,rd),(rd,fr),(fr,fd),(fd,mr),(mr,md),(md,ar),(ar,ad),(ad,end)):before(a,b)
                require(rd['request_id']==rr['id'] and fd['request_id']==fr['id'] and md['request_id']==mr['id'] and ad['request_id']==ar['id'],
                    'Zero-time native work lacks its before-call intent')
                require(rr['native_resetData_calls']==1 and fr['forward_calls']==fd['forward_calls']==mr['mj_fullM_calls']==md['mj_fullM_calls']==1
                    and fr['cumulative_forward_requests']==mr['cumulative_mass_requests']==case+1 and ar['linear_solves']==1,
                    'Native inspection/linear-algebra work count differs')
                require(fr['state_index']==i and fr['action_index']==j and all(r['new_integration_controls']==0 for r in (rr,fr,mr,ar)),
                    'Zero-time inspection gained integration or wrong case')
                same(fr['q'],q,'Native forward q differs');same(fr['velocity'],w,'Native forward velocity differs');same(fr['action'],u,'Native inspection control differs')
                native=fd['native'];require(native['time']==0,'Native inspection advanced time')
                for key in ('qacc','bias','passive','actuator','energy'):finite(native[key],(2,))
                M=finite(md['mass_matrix'],(2,2));same(M,unpack_native_mass(native['packed_mass']),'Native qM expansion differs',atol=0,rtol=0)
                checks,errors_local,expected=parity_comparison(p,terms,u,native,M)
                require(row['case']==case and row['state_index']==i and row['action_index']==j and row['gain_index']==i%4
                    and row['pfl_requested_saturated'] is terms['saturated'] and row['checks']==checks and row['passed'] is all(checks.values()),
                    'Saved parity comparison/qualification label differs')
                same(row['action'],u,'Parity action summary differs');same(row['pfl_requested_action'],terms['action'],'PFL third inspection control differs')
                require(set(row['max_absolute_errors'])==set(errors_local),'Parity error roster differs')
                for key,value in errors_local.items():same(row['max_absolute_errors'][key],value,'Parity arithmetic error differs: '+key,atol=2e-11,rtol=1e-9)
                require({k:v for k,v in payload(ad).items()}==row,'Parity raw-row event differs')
                raw=reader.arrays(archive,row['raw'])
                expected_raw=dict(q=q,velocity=w,action=np.array([u]),gain_triple=np.asarray(GAINS[i%4]),M_analytic=terms['M'],M_native=M,
                    Cw=terms['Cw'],G=terms['G'],damping=terms['damping'],h=terms['h'],S=terms['S'],native_qacc=native['qacc'],native_bias=native['bias'],
                    native_passive=native['passive'],native_actuator=native['actuator'],native_energy=native['energy'],analytic_energy=terms['energy'],
                    target_energy=target,virtual_acceleration=terms['virtual_acceleration'],tau_requested=terms['tau_requested'],qacc_analytic=expected['acceleration'],
                    elbow_elimination=expected['elbow_elimination'],clip_residual=expected['clip_residual'])
                require(set(raw)==set(expected_raw),'Parity saved numeric field roster differs')
                for key,value in expected_raw.items():same(raw[key],value,'Saved parity native/analytic array differs: '+key,atol=2e-10,rtol=2e-10)
                passed.append(row['passed']);analytic['instantaneous_parity_cases']+=1
        require(saved['passed'] is all(passed),'Parity promotion result differs');data['parity_passed']=saved['passed'];data['parity_end']=end
        if not saved['passed']:
            require(result['lqr_raw'] is None and result['controllers']==[] and result['selected_controller'] is None
                and not result['final_outcomes'] and result['final_admissible']==0 and totals['requested_controls']==0,'Failed parity consumed forbidden CARE/controller/episode work')
            require(not any(r['kind'].startswith(('pfl_care_','pfl_controller_','pfl_action_','pfl_episode_','zero_time_reset_','zero_time_restore_')) for r in rows),
                'Parity failure did not halt before CARE/controller/episode work')
    guard('compiled native mechanics and24 fixed instantaneous inspections',parameters_and_parity)
    if errors:return finish()
    if not data['parity_passed']:
        require(result['native_reference_pass'] is stage['native_reference_pass'] is False and result['stopped_at']=='native_instantaneous_parity_failed'
            and result['status']=='complete','Closed failed-parity verdict differs')
        return finish()
    def lqr_and_controllers():
        p,target=data['parameters'],data['target'];req,end=one('pfl_care_requested'),one('pfl_care_complete');before(data['parity_end'],req);before(req,end)
        require(end['request_id']==req['id'] and req['care_solves']==end['care_solves']==1 and req['upright_linearizations']==1
            and req['implicit_analytic_mass_matrices']==1 and req['linear_solves_in_linearization']==3 and req['new_integration_controls']==0
            and end['raw']==result['lqr_raw'],'Single shared CARE work linkage differs')
        raw=reader.arrays(archive,end['raw']);A,B=lqr_arrays(p)
        same(raw['A'],A,'Native upright linearization differs');same(raw['B'],B,'Native gear/linearized action differs')
        same(raw['Q'],np.diag([10.,10.,1.,1.]),'LQR Q differs');same(raw['R'],.1,'LQR R differs')
        residual,eig,relative=verify_care(A,B,raw['P'],raw['Q'],.1,raw['K'])
        same(raw['care_residual'],residual,'Saved CARE residual arithmetic differs',atol=2e-7,rtol=2e-8)
        order=lambda x:np.asarray(sorted(x,key=lambda a:(a.real,a.imag)))
        same(order(raw['closed_loop_eigenvalues']),order(eig),'Saved CARE closed-loop eigenvalues differ',atol=2e-7,rtol=2e-8)
        same(end['max_closed_loop_real_eigenvalue'],float(max(eig.real)),'CARE stability summary differs')
        same(end['maximum_abs_care_residual'],float(abs(residual).max()),'CARE residual summary differs',atol=2e-7,rtol=2e-8)
        data.update(K=raw['K'],care_end=end);details['care_normalized_residual']=relative;analytic['care_residual_rows']+=4
        require(len(result['controllers'])==4,'Four fixed gain controller identities were not constructed')
        constructors=[]
        for i,record in enumerate(result['controllers']):
            request,done=one('pfl_controller_requested',index=i),one('pfl_controller_complete',index=i);before(end,request);before(request,done)
            require(done['request_id']==request['id'] and request['gains']==GAINS[i] and request['controller_constructions']==1
                and request['new_integration_controls']==0,'Fixed controller constructor work differs')
            expected=dict(algorithm='native-collocated-pfl-upright-lqr-v1',parameters=p,gain_triple=GAINS[i],K=raw['K'].tolist(),target_energy=target,
                task_hash=result['task_hash'],source_fingerprint=source,q2_feedback='UNWRAPPED',energy_error='E-Etarget',gear_action='u=clip(tau/gear,-1,1)',
                switch_angle_norm=.3,switch_velocity_norm=2.,Q=[10.,10.,1.,1.],R=.1,excitation=0,lqr_raw=result['lqr_raw'])
            require(record['index']==i and record['gain_triple']==GAINS[i] and record['identity']==done['identity']==expected
                and record['controller_hash']==done['controller_hash']==identity(expected),'Actual native PFL controller identity differs')
            constructors.append(done)
        data['constructors']=constructors
    guard('shared saved CARE solution and fixed controller versions',lqr_and_controllers)
    if errors:return finish()
    used=set();expected_episodes=set()
    def actual_episodes_and_gates():
        p,K,target=data['parameters'],data['K'],data['target'];queries={r['query_id']:r for r in rows if r['kind']=='query_requested'}
        qdone={r['query_id']:r for r in rows if r['kind']=='query_complete'}
        generic={r['episode_id']:r for r in raw_audit['episodes']};saved={r['episode_id']:r for r in rows if r['kind']=='episode_saved'}
        outcomes={r['episode_id']:r for r in rows if r['kind']=='reference_episode_outcome'}
        require(len(saved)==sum(r['kind']=='episode_saved' for r in rows) and len(outcomes)==sum(r['kind']=='reference_episode_outcome' for r in rows),'Duplicate reference episode identity')
        all_constructed=max(index[r['id']] for r in data['constructors']);episode_ends={};ordinary_resets=[];restore_rows=[]
        def actual(i,seed,panel,after=None):
            label=f'pfl{i}_{panel}_{seed}';expected_episodes.add(label);record=result['controllers'][i];owner=record['controller_hash']
            request,end=one('pfl_episode_requested',episode_id=label),one('pfl_episode_complete',episode_id=label);before(request,end)
            require(index[request['id']]>all_constructed and (after is None or index[after['id']]<index[request['id']]),'Episode ran before frozen controller/selection')
            require(end['request_id']==request['id'] and request['index']==end['index']==i and request['panel']==end['panel']==panel and request['seed']==end['seed']==seed
                and request['controller_hash']==owner and request['planned_integration_controls']==800 and request['no_early_termination'] is True,'Actual fixed episode request differs')
            event=outcomes[label];ep=generic[label];raw=reader.arrays(archive,saved[label]['raw']);n=len(raw['actions'])
            require(n==800 and raw['states'].shape==(801,5) and raw['warmstarts'].shape==(801,2) and len(raw['query_ids'])==800,
                'Actual reference shortened or omitted raw boundaries')
            require(ep['context_id']==f'ordinary:{seed}' and ep['controller_hash']==owner and ep['actor_hash'] is None and ep['complete'] is True
                and ep['execution_semantics']=='native_shooting_reference' and ep['excluded_from_teacher_aux'] is True,'Actual continuation ownership differs')
            same(raw['states'][0],ordinary_state(seed),'Fresh ordinary reset differs',atol=0,rtol=0);analytic['ordinary_reset_rows']+=1
            require(np.array_equal(raw['elapsed'],np.arange(801)),'Actual reference clock rows differ')
            source_data=json.loads(str(raw['source_json']));expected_source=dict(algorithm='native-collocated-pfl-upright-lqr-v1',panel=panel,gain_index=i,gain_triple=GAINS[i],
                fixed_seed=seed,excluded_from_teacher=True,formula_identity=owner,parity_only_is_not_success=True,controller='native_shooting_reference',excluded_from_teacher_aux=True)
            require(source_data==expected_source,'Actual reference source/exclusion differs')
            expected=dict(episode_id=label,context_id=f'ordinary:{seed}',cost=ep['cost'],capture=ep['capture'],local=ep['local'],complete=ep['complete'],
                admissible=bool(ep['complete'] and ep['capture'] and ep['local']),raw=saved[label]['raw'],controller_hash=owner,actor_hash=None,
                task_hash=result['task_hash'],query_ids=list(ep['query_ids']),excluded_from_teacher_aux=True)
            require(set(payload(event))==set(expected),'Reference episode outcome field roster differs')
            same(event['cost'],expected['cost'],'Complete reference task cost differs')
            for key,value in expected.items():
                if key!='cost':require(event[key]==value,'Reference actual outcome differs: '+key)
            require(end['outcome']==payload(event),'Episode completion/outcome differs');before(event,end)
            qids=[str(q) for q in raw['query_ids']];require(not used.intersection(qids),'Actual episodes reused paid queries');used.update(qids)
            action_requests=[r for r in rows if r['kind']=='pfl_action_requested' and r['episode_id']==label]
            action_done=[r for r in rows if r['kind']=='pfl_action_complete' and r['episode_id']==label]
            require(len(action_requests)==len(action_done)==800,'Action arithmetic does not cover all actual controls')
            ar={r['elapsed']:r for r in action_requests};ad={r['elapsed']:r for r in action_done}
            require(set(ar)==set(ad)==set(range(800)),'Action arithmetic clock reused/omitted')
            for t,qid in enumerate(qids):
                request_action,action=ar[t],ad[t];q=queries[qid]
                before(request,request_action);before(request_action,action);before(action,q);before(q,qdone[qid]);before(qdone[qid],event)
                require(action['request_id']==request_action['id'] and request_action['index']==action['index']==i
                    and request_action['controller_hash']==action['controller_hash']==owner and request_action['analytic_mass_matrices']==action['analytic_mass_matrices']==2
                    and request_action['energy_evaluations']==action['energy_evaluations']==request_action['arithmetic_rows']==action['arithmetic_rows']==1
                    and request_action['actor_inference_rows']==request_action['new_integration_controls']==0,'Before-call analytic action work differs')
                same(request_action['state'],raw['states'][t],'Action arithmetic used a foreign state',atol=0,rtol=0)
                predicted,fields=controller_action(p,GAINS[i],K,raw['states'][t],target)
                same(raw['actions'][t],predicted,'Executed action is not fixed clipped PFL/LQR law',atol=2e-10,rtol=2e-10)
                same(action['action'],predicted,'Action log/execution differs',atol=2e-10,rtol=2e-10)
                same(action['target_energy'],target,'Action target changed');same(action['tau_applied'],p['gear']*predicted[0],'Applied torque log differs')
                for key,value in fields.items():
                    if value is None or type(value) in (bool,str):require(action[key]==value,'Actual action branch/diagnostic differs: '+key)
                    else:same(action[key],value,'Actual action arithmetic differs: '+key,atol=2e-9,rtol=2e-10)
                require(q['metadata']['episode_id']==label and q['metadata']['source']==source_data and q['metadata']['excluded_from_teacher_aux'] is True
                    and q['owner']==dict(actor_hash=None,controller_hash=owner,semantics='native_shooting_reference',excluded_from_teacher_aux=True),
                    'Sole live executor actual control owner/source differs')
                analytic['actual_controller_action_rows']+=1
            interval=[r for r in rows if index[request['id']]<index[r['id']]<index[ar[0]['id']]]
            reset=[r for r in interval if r['kind']=='zero_time_reset_requested'];reset_done=[r for r in interval if r['kind']=='zero_time_reset_complete'];restore=[r for r in interval if r['kind']=='zero_time_restore_requested']
            require(len(reset)==len(reset_done)==len(restore)==1,'Actual episode reset/restore forward intent coverage differs')
            rr,rd,rs=reset[0],reset_done[0],restore[0];before(rr,rd);before(rd,rs);before(rs,ar[0])
            require(rr['seed']==rd['seed']==seed and rr['context_id']==rd['context_id']==rs['context_id']==f'ordinary:{seed}'
                and rr['direct_mj_forward_calls']==rd['direct_mj_forward_calls']==rs['direct_mj_forward_calls']==1
                and rr['new_simulator_controls']==rd['new_simulator_controls']==rs['new_simulator_controls']==0 and rd['overlap_with_request'] is True,
                'Reset/restore zero-time forward work differs')
            snap=json.loads(str(raw['initial_snapshot_json']));require(rd['snapshot']==snap,'Saved episode does not start at returned reset boundary')
            ordinary_resets.append(rr);restore_rows.append(rs);episode_ends[label]=end
            return payload(event)
        nominal_ends=[];eligible=[]
        for i,record in enumerate(result['controllers']):
            nominal=[actual(i,seed,'nominal') for seed in cfg['nominal_seeds']]
            require(record['nominal']==nominal,'Fixed two-root nominal summary differs');nominal_ends.extend(episode_ends[r['episode_id']] for r in nominal)
            require(record['nominal_qualified'] is all(r['admissible'] for r in nominal),'Nominal gain eligibility differs')
        for i,record in enumerate(result['controllers']):
            selected=[]
            if record['nominal_qualified']:
                for seed in cfg['selection_seeds']:
                    outcome=actual(i,seed,'selection');selected.append(outcome)
                    for end in nominal_ends:before(end,one('pfl_episode_requested',episode_id=outcome['episode_id']))
            require(record['selection']==selected and record['selection_qualified'] is (len(selected)==8 and all(r['admissible'] for r in selected)),
                'Unqualified gain got fresh selection or selection gate differs')
            mean=float(np.mean([r['cost'] for r in selected])) if selected else None
            require(record['selection_mean_cost']==mean,'Selection complete mean cost differs')
            if record['selection_qualified']:eligible.append(record)
        frozen=[r for r in rows if r['kind']=='pfl_controller_frozen'];chosen=None
        if not eligible:
            require(not frozen and result['selected_controller'] is None and not result['final_outcomes'] and result['final_admissible']==0,'Failed selection consumed untouched final data')
        else:
            chosen=min(eligible,key=lambda r:(r['selection_mean_cost'],r['index']));require(len(frozen)==1,'Eligible reference lacked unique final freeze');freeze=frozen[0]
            require(freeze['index']==result['selected_controller']==chosen['index'] and freeze['controller_hash']==chosen['controller_hash']
                and freeze['selection_mean_cost']==chosen['selection_mean_cost'] and freeze['tie_rule']=='lowestdeclaredgainindex' and freeze['before_fresh_final_seeds']==cfg['final_seeds'],
                'Fresh final controller was not selected by fixed full-cost rule')
            for record in result['controllers']:
                for out in record['selection']:before(episode_ends[out['episode_id']],freeze)
            final=[actual(chosen['index'],seed,'fresh_final',after=freeze) for seed in cfg['final_seeds']]
            require(result['final_outcomes']==final and result['final_admissible']==sum(r['admissible'] for r in final),'Fresh32 actual outcome roster/count differs')
        require(used==set(queries)==set(qdone) and expected_episodes==set(generic)==set(saved)==set(outcomes),'Paid controls/episodes outside exact PFL roster')
        require(len(ordinary_resets)==len(restore_rows)==len(expected_episodes)<=72,'Reset/restore episode-work bound differs')
        details.update(actual_episode_count=len(expected_episodes),eligible_gain_indices=[r['index'] for r in eligible],
            selected_gain_index=None if chosen is None else chosen['index'],
            restore_completion_proof='Inherited helper logs before restore only; the subsequent actual first-action state and exact raw/query initial boundary prove return, not an independently observed engine completion hook.')
        return bool(chosen is not None and result['final_admissible']>=29)
    actual=guard('actual PFL actions, shared fresh panels and original gates',actual_episodes_and_gates)
    if actual is not None:verdict=actual
    def final_closure():
        require(result['native_reference_pass'] is stage['native_reference_pass'] is verdict and result['status']=='complete'
            and result['within_declared_bounds'] is True,'Reported feasibility verdict differs')
        stop=('qualified_collocated_pfl_reference' if verdict else 'no_qualifying_nominal_pfl_controller' if not any(r['nominal_qualified'] for r in result['controllers'])
            else 'no_qualifying_fresh_selection_pfl_controller' if result['selected_controller'] is None else 'fresh_final_pfl_qualification_failed')
        require(result['stopped_at']==stop,'Bounded stopping gate differs')
        expected=dict(declared_gain_candidates=4,constructed=4,nominal_panels_completed=4,
            selection_panels_completed=sum(len(r['selection'])==8 for r in result['controllers']),selected_index=result['selected_controller'])
        require(result['fixed_controller_selection_work']==expected,'Fixed numerical controller selection work omitted/changed')
        work=counts['zero_time_and_analytic']
        require(work['analytic_actions_requested']==work['analytic_actions_completed']==totals['requested_controls']
            and work['analytic_action_errors']==0 and work['episode_reset_forward_requests']==work['episode_restore_forward_requests']==len(expected_episodes)
            and work['direct_zero_time_forward_requests']==24+2*len(expected_episodes),'Actual action/forward accounting closure differs')
    if not errors:guard('PFL complete verdict and all-work closure',final_closure)
    return finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();report=audit_pfl_reference(args.run)
    with args.out.open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','scientific_verdict','native_reference_pass','counts','wall_seconds')},indent=2))
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
