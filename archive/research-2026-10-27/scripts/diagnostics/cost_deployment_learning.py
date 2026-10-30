"""Additive costs for deployment-learning development, including failures."""
import argparse,hashlib,json,time
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args();out=a.out.resolve();out.mkdir(exist_ok=False);start=time.perf_counter();(out/'build.py').write_bytes(Path(__file__).read_bytes());root=Path.cwd();inputs={}
 def read(p):
  file=root/p;inputs[p]=sha(file);return json.loads(file.read_text())
 totals=dict(model_controls=0,executed_controls=0,total_controls=0,neural_rows=0,adam_steps=0,backward_calls=0,autograd_grad_calls=0,query_attempts=0,error_queries=0,rejected_queries=0,unknown_controls=0,manual_parameter_updates=0,analytic_physical_jacobian_rows_completed=0,analytic_policy_jacobian_rows_completed=0,metric_solves_completed=0,metric_halfspace_checks_completed=0,metric_ball_checks_completed=0)
 studies=[]
 for i in range(1,5):
  folder=f'runs/2026-10-27_bidirectional_deployment_adjoint_{i:03d}';r=read(folder+'/curriculum_result.json');stage=read(folder+'/stage_complete.json');history=[dict(file=str(p.relative_to(root)),receipt=read(str(p.relative_to(root)))) for p in sorted((root/folder).glob('audit_[0-9][0-9][0-9].json'))];audit=history[-1]['receipt'];q=r['accounting']
  charges=dict(model_controls=r['model_controls'],executed_controls=r['executed_controls'],total_controls=q['charged_requested_controls'],neural_rows=r['actor_rows_requested'],adam_steps=r['adam_attempts'],backward_calls=r['backward_attempts'],autograd_grad_calls=r['diagnostic_backward_attempts'],query_attempts=q['query_attempts'],error_queries=q['error_queries'],rejected_queries=q['rejected_queries'],unknown_controls=q['unknown_controls'])
  charges['manual_parameter_updates']=audit['counts'].get('manual_parameter_updates',0)
  charges['analytic_physical_jacobian_rows_completed']=audit['counts'].get('analytic_dynamics_jacobian_rows_rechecked',0)
  charges['analytic_policy_jacobian_rows_completed']=audit['counts'].get('analytic_policy_jacobian_rows_rechecked',0)
  charges['metric_solves_completed']=audit['counts'].get('metric_solves_rechecked',0)
  charges['metric_halfspace_checks_completed']=512*charges['metric_solves_completed']
  charges['metric_ball_checks_completed']=64*charges['metric_solves_completed']
  for k,v in charges.items():totals[k]+=v
  studies.append(dict(run=folder,stage=r['stage'],scientific_success=r['passed'],audit_status=audit['status'],audit_errors=audit['errors'],audit_history=[dict(file=z['file'],status=z['receipt']['status'],errors=z['receipt']['errors'],wall_seconds=z['receipt']['audit_wall_seconds']) for z in history],charges=charges,body_wall_seconds=r['body_wall_seconds'],launch_to_end_wall_seconds=stage['launch_to_end_wall_seconds'],audit_wall_seconds=audit['audit_wall_seconds']))
 tests=[];associated=dict(scalar_controls=0,neural_rows=0,adam_steps=0,autograd_grad_calls=0)
 for i in range(36,43):
  folder=f'runs/2026-10-27_bidirectional_accounted_tests_{i:03d}';r=read(folder+'/test_result.json');q=r['accounting'];charges=dict(scalar_controls=q['scalar_controls_requested'],neural_rows=q['actor_forward_rows_requested'],adam_steps=q['optimizer_updates_requested'],autograd_grad_calls=q['autograd_grad_calls_requested'])
  for k,v in charges.items():associated[k]+=v
  tests.append(dict(run=folder,passed=r['passed'],charges=charges,body_wall_seconds=r['wall_seconds'],launch_to_end_wall_seconds=r['launch_to_end_wall_seconds']))
 other=[]
 paths=[('bidirectional_actor_gap_diagnosis_001','diagnosis.json'),('bidirectional_adjoint_direction_diagnosis_001','diagnosis.json'),('bidirectional_actor_geometry_replay_001','receipt.json'),('bidirectional_metric_geometry_check_001','receipt.json'),('bidirectional_changed_actor_qualification_001','receipt.json'),('bidirectional_changed_actor_qualification_audit_001','audit.json')]
 for name,file in paths:
  folder='runs/2026-10-27_'+name;r=read(folder+'/'+file);other.append(dict(run=folder,receipt=r))
 # The two diagnostics and cached metric repair have zero new actor forwards.
 # The 80-row geometry replay and 960-row changed-actor qualification are paid.
 associated['neural_rows']+=80+960;associated['scalar_controls']+=800;associated['autograd_grad_calls']+=9
 associated['physical_jacobian_rows']=80+80;associated['analytic_policy_jacobian_rows']=80+80
 packet_receipts=[]
 for packet in sorted(root.glob('runs/2026-10-27_bidirectional_deployment_auditor_*')):
  for file in sorted(packet.glob('*.json')):
   if file.name=='test_receipt.json' or file.name.startswith(('execution_','failure_','scheduling_','begin_')):
    rel=str(file.relative_to(root));packet_receipts.append(dict(file=rel,receipt=read(rel)))
 report_path='runs/2026-10-27_bidirectional_deployment_report_001/receipt.json';report=read(report_path)
 diagnosis=read('runs/2026-10-27_bidirectional_deployment_learning_diagnosis_001/diagnosis.json')
 endpoints=read('runs/2026-10-27_bidirectional_deployment_endpoints_001/diagnosis.json')
 margins=read('runs/2026-10-27_bidirectional_deployment_margins_report_001/receipt.json')
 previous='runs/2026-10-27_bidirectional_return_curriculum_costs_001/costs.json';prior=read(previous)
 result=dict(schema='scoped-deployment-learning-costs-v1',studies=studies,incremental_study_totals=totals,tests=tests,associated_nonstudy_totals=associated,other_evidence=other,auditor_packets=packet_receipts,report=report,measured_update_diagnosis=diagnosis,endpoint_diagnosis=endpoints,margin_plot=margins,
  previous_scoped_cost_receipt=previous,previous_scoped_lineage_totals=prior['scoped_lineage_totals'],
  current_scoped_lineage_totals=dict(total_controls=prior['scoped_lineage_totals']['total_controls']+totals['total_controls']+associated['scalar_controls'],neural_rows=prior['scoped_lineage_totals']['neural_rows']+totals['neural_rows']+associated['neural_rows'],adam_steps=prior['scoped_lineage_totals']['adam_steps']+totals['adam_steps']+associated['adam_steps']),
  input_sha256=inputs,source_sha256=sha(out/'build.py'),processing_wall_seconds=time.perf_counter()-start,new_scalar_controls=0,new_actor_rows=0,new_optimizer_steps=0,
  limits=['Scoped lineage only; earlier physical/learning failures outside the prior receipt remain paid separately. Historical unmetered development is unknown, so no all-cost efficiency claim.',
   'Study3 stopped on neural feature-layout mismatch; its 24408 closed controls are retained. It did not complete a scientific loop.',
   'Requested/completed/query/NN/control scopes overlap; do not sum work categories describing the same call.',
   'Concurrent body/audit and launch wall times overlap; this is no claim of serial calendar elapsed.',
   'Analytic Jacobian arithmetic, linear solves and projection checks are in study work records and audit numeric counters. They are not free neural forwards or additional physics controls.',
   'Reports, audits, diagnoses and failed tests are retained; test fixtures and numerical replay are not scientific training seeds.'])
 with (out/'costs.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({k:result[k] for k in ('incremental_study_totals','associated_nonstudy_totals','current_scoped_lineage_totals')},indent=2))

if __name__=='__main__':main()
