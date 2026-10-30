"""Create a scoped, additive accounting receipt from closed evidence only."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path.cwd());parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();started=time.perf_counter();root=args.root.resolve();out=args.out.resolve();out.mkdir(exist_ok=False)
    (out/'build.py').write_bytes(Path(__file__).read_bytes());inputs={}
    def read(path):
        path=root/path;inputs[str(path.relative_to(root))]=sha(path);return json.loads(path.read_text())
    totals=dict(model_controls=0,executed_controls=0,total_controls=0,neural_rows=0,adam_steps=0,
        diagnostic_backwards=0,query_attempts=0,query_errors=0,rejected_queries=0,unknown_controls=0)
    studies=[];first=None
    for i in range(1,4):
        folder=f'runs/2026-10-27_bidirectional_return_curriculum_{i:03d}'
        result=read(folder+'/curriculum_result.json');audit=read(folder+'/audit_001.json');stage=read(folder+'/stage_complete.json')
        assert audit['status']=='PASS' and result['source_unchanged'] and not result['passed'] and result['stage']=='complete_measured_loop_with_negative_links'
        a=result['accounting'];row=dict(model_controls=result['model_controls'],executed_controls=result['executed_controls'],
            total_controls=a['completed_controls'],neural_rows=result['actor_rows_requested'],adam_steps=result['adam_attempts'],
            diagnostic_backwards=result['diagnostic_backward_attempts'],query_attempts=a['query_attempts'],query_errors=a['error_queries'],
            rejected_queries=a['rejected_queries'],unknown_controls=a['unknown_controls'])
        for k,v in row.items():totals[k]+=v
        studies.append(dict(run=folder,charges=row,body_wall_seconds=result['body_wall_seconds'],
            launch_to_end_wall_seconds=stage['launch_to_end_wall_seconds'],audit_body_wall_seconds=audit['audit_wall_seconds']))
        first=result if first is None else first
    tests=[];test_controls=0;test_model=0;test_live=0
    for i in range(30,36):
        folder=f'runs/2026-10-27_bidirectional_accounted_tests_{i:03d}';result=read(folder+'/test_result.json')
        n=result['accounting']['scalar_controls_requested'];correction=None
        if i==32:
            correction=read(folder+'/accounting_addendum_001.json')
            assert correction['input_sha256']['test_result.json']==inputs[folder+'/test_result.json']
            n=correction['requested_scalar_controls']
        test_controls+=n
        if n:test_model+=25;test_live+=50
        assert result['accounting']['actor_forward_rows_requested']==result['accounting']['optimizer_updates_requested']==0
        tests.append(dict(run=folder,passed=result['passed'],scalar_controls=n,correction_applied=correction is not None,
            body_wall_seconds=result['wall_seconds'],launch_to_end_wall_seconds=result['launch_to_end_wall_seconds']))
    packets=[]
    for i in range(1,5):
        folder=f'runs/2026-10-27_bidirectional_return_curriculum_auditor_{i:03d}';fixture=read(folder+'/test_receipt.json')
        execution_path=Path(folder)/'execution_receipt.json'
        execution=read(execution_path) if (root/execution_path).exists() else None
        packets.append(dict(run=folder,fixture_receipt=fixture,execution_receipt=execution))
    ancestor=first['reused_paid_ancestry'];controls=ancestor['unique_lineage_requested_scalar_controls']
    rows=ancestor['unique_lineage_actor_rows'];adam=ancestor['unique_lineage_adam_attempts']
    replay=read('runs/2026-10-27_bidirectional_geometry_feature_verification_001/stage_complete.json')
    assert replay['actor_rows_completed']==replay['actor_rows_requested']==288
    reports=[]
    for i in (1,2):
        path=f'runs/2026-10-27_bidirectional_return_curriculum_report_{i:03d}/receipt.json'
        if (root/path).exists():reports.append(dict(file=path,receipt=read(path)))
    diagnosis=read('runs/2026-10-27_bidirectional_return_curriculum_search_diagnosis_001/diagnosis.json')
    receipt=dict(schema='scoped-return-curriculum-campaign-costs-v1',studies=studies,incremental_study_totals=totals,
        associated_tests=tests,test_scalar_controls=test_controls,test_model_controls=test_model,test_executed_controls=test_live,
        shared_paid_ancestry_once=dict(total_controls=controls,neural_rows=rows,adam_steps=adam,
            body_wall_seconds=first['reused_body_wall_seconds'],separate_feature_verification_rows=288),
        scoped_lineage_totals=dict(total_controls=totals['total_controls']+test_controls+controls,
            neural_rows=totals['neural_rows']+rows+288,adam_steps=totals['adam_steps']+adam),
        auditor_packets=packets,reports=reports,candidate_diagnosis_wall_seconds=diagnosis['wall_seconds'],
        input_sha256=inputs,source_sha256=sha(out/'build.py'),processing_wall_seconds=time.perf_counter()-started,
        new_controls=0,new_actor_rows=0,new_optimizer_steps=0,
        limitations=['Scoped lineage only; all other earlier physical/search/learning experiments remain paid outside it.',
            'Historical unmetered development is unknown; this cannot establish all-cost efficiency.',
            'Body, launch, wrapper and concurrent processing wall scopes overlap; do not add them as serial calendar elapsed.',
            'Test032 local75controls are corrected once, never added again on top of an observed global75.',
            'All failed fixtures/collection attempts remain recorded; no earlier study or summary is overwritten.'])
    with (out/'costs.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({k:receipt[k] for k in ('incremental_study_totals','scoped_lineage_totals','processing_wall_seconds')},indent=2))


if __name__=='__main__':main()
