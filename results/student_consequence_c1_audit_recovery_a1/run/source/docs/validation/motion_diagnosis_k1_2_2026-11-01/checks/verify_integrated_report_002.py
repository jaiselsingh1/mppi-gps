from pathlib import Path
import json,hashlib
root=Path.cwd();art=root/'artifacts/2026-11-01_motion_diagnosis_k1_2_001';run=root/'runs/2026-11-01_motion_diagnosis_k1_2_001'
r=json.loads((run/'result.json').read_text());a=json.loads((art/'audit_001/receipt.json').read_text());p=root/'docs/motion_diagnosis_k1_2_results.md';s=p.read_text();checks=0
for source,label in [('qualified_reference','Qualified reference'),('mppi','Walking MPPI')]:
 rows=[x for x in r['trajectories'] if x['source']==source or (source=='mppi' and x['index']<14)]
 rr=[v['within']['best']['R'] for v in rows];aa=[v['within']['best']['action_mse'] for v in rows]
 line=next(v for v in s.splitlines() if v.startswith('| '+label+' |'))
 for value in [str(len(rows)),'666',f'{min(rr):.6f}–{max(rr):.6f}',f'{min(aa):.6f}–{max(aa):.6f}']:
  assert value in line,(label,value);checks+=1
pair=next(v for v in r['cross_pairs'] if v['first']=='v2_r0_trial_raw_0' and v['second']=='v2_r0_trial_raw_1')['best']
for key in ['R','action_mse','first_reference_E','second_reference_E']:
 assert f'{pair[key]:.6f}' in s;checks+=1
for value in pair['squared_contributions'].values():assert f'{value:.6f}' in s;checks+=1
for name in ['v2_r0_trial_raw_0','v2_r0_trial_raw_1']:
 t=next(v for v in r['trajectories'] if v['id']==name);assert f'{t["E"]:.9f}' in s;checks+=1
for value in [r['work']['source_hash_bytes'],r['work']['source_io_bytes'],r['work']['decoded_bytes'],r['work']['queries_completed'],r['work']['aligned_completed'],r['peak_rss_bytes'],a['checks'],a['work']['total_hash_bytes'],a['work']['source_io_bytes'],a['work']['compact_io_bytes'],a['peak_rss_bytes'],r['motion_archive']['bytes']]:
 assert f'{value:,}' in s;checks+=1
assert f'{r["wall_seconds"]:.6f}' in s;checks+=1
assert f'{a["wall_seconds"]:.6f}' in s;checks+=1
out={'schema':'walker-motion-report-number-check-v1','passed':True,'checks':checks,'document_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Read-only formatting checks against independently audited saved result/receipt; no raw arrays, distance queries or scientific rerun.'}
with (art/'integrated_report_verification_002.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
