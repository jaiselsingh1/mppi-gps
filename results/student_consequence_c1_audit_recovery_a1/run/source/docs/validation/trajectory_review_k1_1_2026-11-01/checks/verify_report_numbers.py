from pathlib import Path
import hashlib,json
root=Path.cwd(); art=root/'artifacts/2026-11-01_trajectory_review_k1_1_001'
p=root/'docs/trajectory_review_k1_1_results.md';text=p.read_text();r=json.loads((root/'runs/2026-11-01_trajectory_review_k1_1_001/result.json').read_text());n=0
mapping={'Original discovery':'original_discovery','Later independent':'later_independent','Later guided':'later_guided','H4 independent':'h4_independent','H4 fixed-high':'h4_fixed_high','H4 backoff':'h4_backoff','Pooled MPPI, descriptive mixture':'pooled_MPPI_14'}
for label,key in mapping.items():
 g=next(v for v in r['groups'] if v['source']==key);d=g['decomposition'];line=next(v for v in text.splitlines() if v.startswith('| '+label+' |'))
 values=[str(g['count']),f"{g['E_spread']:.6f}",f"{d['covariance_diag_contribution']:.9f}",f"{d['covariance_offdiag_contribution']:+.9f}",f"{d['var_between_full']:.9f}"]
 assert [v.strip() for v in line.split('|')[2:-1]]==values,(label,line,values);n+=len(values)
g=next(v for v in r['groups'] if v['source']=='later_independent');pair=g['two_trajectory_difference']
for b in pair['bin_deltas']:
 prefix=f"| {b['start_control']}–{b['stop_control_exclusive']} |"
 line=next(v for v in text.splitlines() if v.startswith(prefix))
 assert f"{b['delta_E']:+.6f}" in line;n+=1
for t in r['trajectories']:
 if t['id'] in [pair['first'],pair['second']]:assert f"{t['E']:.9f}" in text;n+=1
assert f"{pair['full_delta']:.9f}" in text;n+=1
for field in ['rows','bins','actuator_entries']:
 assert f"{r['analysis_counts'][field]:,}" in text;n+=1
assert r['execution_complete'] is True and r['error'] is None;n+=1
result={'schema':'walker-maintained-report-numeric-verification-v1','checks':n,'passed':True,'document':str(p.relative_to(root)),'document_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scope':'Read-only formatting/identity checks against saved audited result; no new science, raw-array reads or result rewrite.'}
with (art/'report_number_verification.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
