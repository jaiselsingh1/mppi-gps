"""Saved-only final all-candidate reduction/ledger review; no actor or physics."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

T0, C0 = time.monotonic(), time.process_time()
HERE = Path(__file__).resolve().parent
OWNER = Path('/Users/jaisel/Documents/workbench/mppi-local-coupling-diagnosis')
BASE = OWNER/'runs/2026-11-03_local_coupling_diagnosis_002'
RUN = OWNER/'runs/2026-11-03_local_calibration_audit_001'
READS, CHECKS, FAILURES = {}, [], []
SUMMARY = {}


def guard():
    if time.process_time()-C0 > 20 or time.monotonic()-T0 > 30:
        raise RuntimeError('Final reduction review reservation stop')


def bind(p):
    guard()
    p = Path(p)
    if str(p) not in READS:
        data = p.read_bytes()
        READS[str(p)] = dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    return READS[str(p)]


def read(p):
    bind(p)
    return json.loads(Path(p).read_text())


def check(name, actual, expected, atol=0):
    guard()
    passed = actual == expected if isinstance(actual,dict) or isinstance(expected,dict) else bool(np.allclose(actual,expected,atol=atol,rtol=0))
    CHECKS.append(dict(name=name,passed=passed))
    if not passed:
        FAILURES.append(name)


try:
    audit, work, previous = read(RUN/'audit.json'),read(RUN/'work.json'),read(BASE/'work.json')
    source = OWNER/'scripts/studies/audit_local_calibration.py'
    bind(source)
    bind(OWNER/'scripts/studies/local_coupling_diagnosis.py')
    bind(OWNER/'docs/local_coupling_protocol.md')
    bind(RUN/'native_reductions.npz');bind(BASE/'calibration.npz')
    per_candidate=[]
    with np.load(RUN/'native_reductions.npz',allow_pickle=False) as native, np.load(BASE/'calibration.npz',allow_pickle=False) as bank:
        c,d,n,done=(native[k] for k in ('costs','D','lengths','done'))
        cb,db,nb=bank['C'],bank['D'],bank['lengths']
        posts,actions=bank['posts'],bank['actions']
        healthy=(posts[...,2]>.8)&(posts[...,2]<2)&(posts[...,3]>-1)&(posts[...,3]<1)
        mask=np.ones(healthy.shape,bool);mask[:,1:]=np.logical_and.accumulate(healthy[:,:-1],axis=1)
        check('256_row_shapes',[len(c),len(d),len(n),len(done),len(cb)],[256]*5)
        check('exact_all_prefix_lengths',n,nb)
        check('prefix_lengths_from_raw_bank',n,mask.sum(1))
        check('native_done_vs_raw_bank',done,~healthy.all(1))
        check('native_C_vs_bank',c,cb,1e-8)
        check('native_D_vs_bank',d,db,5e-4)
        check('reported_C_max',audit['C_max_abs'],float(np.max(np.abs(c-cb))),1e-15)
        check('reported_D_max',audit['D_max_abs'],float(np.max(np.abs(d-db))),1e-15)
        check('14_full_native_survivors',int(np.sum(~done)),14)
        check('reported_survivors',audit['full_survivors'],int(np.sum(~done)))
        check('native_controls_6667',int(n.sum()),6667)
        check('reported_controls',audit['controls'],int(n.sum()))
        check('paid_native_controls',work['completed']['native_controls'],int(n.sum()))
        check('paid_forward_rows',work['completed']['neural_rows'],int(n.sum()))
        check('paid_substeps',work['completed']['physics_substeps'],4*int(n.sum()))
        check('paid_restores',work['completed']['restores'],256)
        check('no_new_model_controls',work['completed'].get('model_controls',0),0)
        for k in range(256):
            per_candidate.append(dict(candidate=k,length=int(n[k]),done=bool(done[k]),native_cost=float(c[k]),bank_cost=float(cb[k]),native_D=float(d[k]),bank_D=float(db[k])))
        SUMMARY['calibration_native_reductions']=dict(candidates=256,survivors=int(np.sum(~done)),controls=int(n.sum()),maximum_cost_difference=float(np.max(np.abs(c-cb))),maximum_D_difference=float(np.max(np.abs(d-db))),reported_native_state_max_abs=audit['native_state_max_abs'],candidate_order_binding='Implicit ordinal 0..255 from frozen replay source enumerate(bank.actions); no separate per-row source hash.')
    ledger_path=RUN/'ledger.jsonl';bind(ledger_path)
    events=[json.loads(line) for line in ledger_path.read_text().splitlines()]
    pending={};completed={};unknown={};requested={}
    for row in events:
        kind=row['event']
        if kind=='request':
            check('unique_request_'+str(row['id']),row['id'] not in pending,True)
            pending[row['id']]=row['counts'];dest=requested
        elif kind in ('complete','error'):
            check('matching_completion_'+str(row['id']),pending.pop(row['id'],None)==row['counts'],True)
            dest=completed if kind=='complete' else unknown
        else:
            continue
        for k,v in row['counts'].items():
            dest[k]=dest.get(k,0)+v
    check('no_pending',bool(pending),False)
    check('exact_completed_ledger',completed,work['completed'])
    check('exact_unknown_ledger',unknown,work['unknown_partial'])
    check('prior_CPU_carried_once',work['preparation_reservation']['prior_process_cpu_seconds'],previous['charged_cpu_seconds']-300,1e-12)
    check('same_elapsed_anchor',work['preparation_reservation']['prep_wall_start_upper_epoch'],previous['preparation_reservation']['prep_wall_start_upper_epoch'])
    combined={}
    for counts in (previous['completed'],work['completed']):
        for k,v in counts.items():
            combined[k]=combined.get(k,0)+v
    check('combined_native',combined['native_controls'],6730)
    check('combined_model',combined['model_controls'],12288)
    check('combined_substeps',combined['physics_substeps'],76072)
    check('combined_neural',combined['neural_rows'],13334)
    part_b=read(BASE/'part_b.json')
    check('unchanged_B_stop',part_b['status']=='STOPPED' and part_b['classification']=='unclassified' and not part_b['episodes'],True)
    check('C_not_run',not (BASE/'part_c.json').exists(),True)
    SUMMARY['combined_scientific_calls_attempt002_plus_256_audit']=combined
    SUMMARY['attempt001_additional_nonphysical_calls']=dict(environment_constructions=1,checkpoint_loads=1)
    SUMMARY['owner_global_accounting']=dict(charged_cpu_seconds=work['charged_cpu_seconds'],preparation_cpu_precharge=300,process_cpu_all_owner_phases=work['charged_cpu_seconds']-300,charged_global_elapsed_seconds=work['charged_elapsed_seconds'],combine_rule='Latest charged work already includes 001/002 process CPU and preparation reservation once. Do not sum charged work receipts.',reviewer_cost='Separate own receipts; final external wrapper measured below. Early reviewer imports/orchestration remain unmeasured.')
    SUMMARY['limitations']=['No independent policy forwards or physics: native D/C reductions matched, not policy arithmetic or simulator evolution reproduced by this reviewer.','Native per-step states, observations, predictions, actions and native costs are not preserved in the added reduction NPZ. The reported state maximum cannot be independently recomputed from those reductions.','No full context or native binary/runtime manifest sealed in this study run; input/source/config checks are separate.','No reconstruction learning, action-effect test, survival comparison or C diagnostic was reached.']
    SUMMARY['owner_all256_source_semantics']='Source appends obs(env.get_state()) before env.step, with 17-vector velocity clipping; deployed Policy.predict uses checkpoint normalization and final tanh; D uses actual saved actions promoted float64. C sums native prefix step costs and adds five per unused control. No post-fall policy rows.'
    (HERE/'native_per_candidate.json').write_text(json.dumps(per_candidate,indent=2)+'\n')
except BaseException as error:
    FAILURES.append(repr(error))
finally:
    cpu,wall=time.process_time()-C0,time.monotonic()-T0
    findings=dict(schema='independent-local-calibration-reduction-review-1',passed=not FAILURES,checks=CHECKS,failures=FAILURES,summary=SUMMARY)
    (HERE/'native_findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    (HERE/'native_input_bindings.json').write_text(json.dumps(READS,indent=2)+'\n')
    previous_receipt=json.loads((HERE/'closure_receipt.json').read_text())
    receipt=dict(schema='independent-local-calibration-reduction-receipt-1',passed=not FAILURES,process_cpu_seconds=cpu,active_process_wall_seconds=wall,total_numeric_region_cpu_seconds=previous_receipt['total_review_cpu_seconds']+cpu,total_numeric_region_wall_seconds=previous_receipt['total_active_process_wall_seconds']+wall,checks=len(CHECKS),files_read=len(READS),logical_unique_bytes=sum(v['bytes'] for v in READS.values()),neural_forward_rows=0,model_controls=0,native_controls=0,physics_substeps=0,gradients=0,adam_updates=0,owner_code_executed=False)
    (HERE/'native_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));print(json.dumps(dict(failures=FAILURES,summary=SUMMARY)))
