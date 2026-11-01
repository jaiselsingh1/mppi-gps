from pathlib import Path
import hashlib,json,shutil,time

ROOT=Path('/Users/jaisel/Documents/mppi-gps')
ART=ROOT/'artifacts/2026-11-01_trajectory_review_k1_1_001'
RUN=ROOT/'runs/2026-11-01_trajectory_review_k1_1_001'
CAP=ROOT/'results/trajectory_review_k1_1'

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def binding(p):return {'sha256':sha(p),'bytes':p.stat().st_size}

assert read(RUN/'worker_exit.json')['execution_complete'] is True
assert read(ART/'audit_001/receipt.json')['passed'] is False
assert read(ART/'audit_002/receipt.json')['passed'] is True
assert not CAP.exists()
receipts=[]
for p in sorted((ART/'engineering').glob('*/receipt.json')):
 d=read(p)
 for n in ['stdout','stderr']:
  assert sha(p.parent/(n+'.txt'))==d[n+'_sha256'],p
 receipts.append({'directory':str(p.parent.relative_to(ART)),**d})
for p in sorted((ART/'report_001').glob('*.csv')):
 assert p.read_bytes()==(ART/'report_002'/p.name).read_bytes()
assert (ART/'report_001/report.md').read_bytes()==(ART/'report_002/report.md').read_bytes()
result=read(RUN/'result.json');old=read(RUN/'inputs/result.json')
closed_files=[p for p in RUN.rglob('*') if p.is_file()]
closed_bytes=sum(p.stat().st_size for p in closed_files)
assert closed_bytes==read(ART/'audit_002/findings.json')['closed_run_bytes']
accounting={
 'schema':'walker-trajectory-review-accounting-v1',
 'scope':'New saved-only K1.1 analysis, failed/passed audit, original/corrected report and explicitly metered command overhead. Inherited research counters are separate; no global efficiency or billing claim.',
 'run':{'attempts':1,'execution_complete':True,'work':result['work'],'worker_wall_seconds':result['wall_seconds'],'peak_rss_bytes':result['peak_rss_bytes'],'owner':read(RUN/'worker_exit.json'),'closed_files':len(closed_files),'closed_bytes':closed_bytes},
 'audits':{n:read(ART/n/'receipt.json') for n in ['audit_001','audit_002']},
 'reports':{n:read(ART/n/'receipt.json') for n in ['report_001','report_002']},
 'report_correction':'One cumulative-plot layout fix. All seven CSVs and report.md byte-identical; other PDF changes limited to timestamps. Both report versions retained.',
 'audit_correction':'Original frozen checker incorrectly treated confirmation as primary-style policy-to-summary mapping. Separately frozen corrected checker handles the historical schemas explicitly. Science inputs, metrics, thresholds and tolerances unchanged.',
 'inherited_k1':{'work_by_phase':old['work'],'worker_wall_seconds':old['wall_seconds'],'stored_bytes_declared':old['stored_bytes'],'result_sha256':sha(RUN/'inputs/result.json'),'ancestors':'Original discovery, reference training and student fitting remain in their historical ledgers/capsules; no summed reconstructed total or repeat cost.'},
 'metered_commands':receipts,
 'metered_command_wall_seconds_sum':sum(r['wall_seconds'] for r in receipts),
 'overlap':'Command wall times include subprocess time. Do not add worker/audit/report internal elapsed to their enclosing command elapsed, nor equate this sum to elapsed investigation time.',
 'storage':'Original full run local; exact complete new run plus source/input bytes copied to capsule. Reports/tools/engineering copies charged separately by capsule file census. Inherited K1 raw arrays remain outside Git and were not opened.',
 'unknown_not_zero':['Uninstrumented editing/inspection and subagent development wall time','Global synthetic-test model/physics/arithmetic counts (tests include small training and simulator checks)','Model/tool billing and CPU energy cost','Unrecorded historical ancestor overhead'],
 'later_validation':'Final clean-export/CI/publication overhead will be retained separately under artifacts/2026-11-01_trajectory_review_k1_1_001/engineering and docs/validation; this accounting snapshot and capsule are not mutated to append later costs.'}
write(ART/'accounting.json',accounting)
CAP.mkdir()
files={}
def copy(p,q):
 target=CAP/q;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 b=binding(target);assert b==binding(p)
 files[str(q)]={'source':str(p),**b}
def tree(p,q):
 for f in sorted(p.rglob('*')):
  if f.is_file():copy(f,Path(q)/f.relative_to(p))
tree(RUN,'run')
tree(ART/'report_002','report')
tree(ART/'report_001','historical/report_001')
for n in ['audit_001','audit_002']:tree(ART/n,n)
tree(ART/'analysis_source_002','analysis_source_002')
tree(ART/'engineering','engineering')
copy(ART/'accounting.json',Path('accounting/accounting.json'))
copy(ART/'prelaunch_publication.json',Path('publication/prelaunch_publication.json'))
copy(Path(__file__),Path('publication/publish_capsule.py'))
copy(RUN/'source/src/gps/mppi_gps/audits/trajectory_review.py',Path('audit_source/original_001.py'))
copy(ART/'analysis_source_002/src/gps/mppi_gps/audits/trajectory_review.py',Path('audit_source/corrected_002.py'))
readme='''# K1.1 temporal agreement review\n\nComplete saved-only analysis of all 34 K1 trajectories, 30,070 controls and 304 bins. Some near-equal full scores conceal opposing local differences. This does not establish phase, recurrence, consequential errors, teachability or useful coupling.\n\n- [Result and causal verdict](../../docs/trajectory_review_k1_1_results.md); [frozen protocol](../../docs/trajectory_review_k1_1_protocol.md).\n- [Figures and seven complete tables](report/report.md); [frozen result](run/result.json).\n- [Passed independent audit](audit_002/receipt.json); [original failed audit](audit_001/receipt.json).\n- [Complete accounting](accounting/accounting.json), [prelaunch publication](publication/prelaunch_publication.json), [corrected tools source](analysis_source_002/source_manifest.json).\n- [Next learning hypothesis, unlaunched](../../docs/student_visitation_s1_2_hypothesis.md).\n\nThis capsule contains the entire new run, including exact nine saved inputs and frozen source. No raw arrays/checkpoints/videos were generated or consumed by this analysis. Existing K1 bulk evidence remains [external](../../docs/evidence_availability.md); original capsules and archives are unchanged. The original checker/report and the separate corrected versions are retained. Do not execute a worker inside this closed evidence directory.\n\nPortable read-only inspection and plotting:\n\n```sh\npython -B scripts/train_mppi_gps.py inspect --run results/trajectory_review_k1_1 --verify --json\npython -B scripts/diagnostics/report_trajectory_review.py --run results/trajectory_review_k1_1 --out /absolute/fresh/external/report\n```\n\nA new analysis must be prepared at a fresh path using the frozen recipe and explicitly located original inputs; see the protocol. The auditor recomputes saved metadata only and cannot certify unsaved motions or replay networks. Later publication-validation receipts live separately so this capsule remains immutable.\n'''
(CAP/'README.md').write_text(readme)
files['README.md']={'source':'publication-authored navigation',**binding(CAP/'README.md')}
cap={'schema':'walker-result-capsule-v1','study':'trajectory_review_k1_1','run_id':RUN.name,'status':'saved_analysis_complete','integrity_status':'independent_saved_metadata_audit_passed','implementation_commit':'e73cc5bdf3efbd7029773375d6b4f8e74c6860fc','checker_report_correction_commit':'3dcec3618c647c847badf5b7fb5a543af6059494','scope':'Saved fixed-reference score decomposition only. No new controller, learning or optimization feedback. All data previously viewed; descriptive rather than blind confirmation.','files':files,'copied_file_count':len(files),'copied_bytes':sum(v['bytes'] for v in files.values()),'bulk_evidence':{'local_directory':str(RUN),'remote_directory':None,'remote_copy_verified':False,'omitted':[],'scope':'Entire new saved-only run is copied here. No new raw arrays, checkpoints or videos exist. Inherited K1 bulk remains local-only and outside Git.'},'new_physics':0,'new_inference':0,'new_training':0,'operation_scope':'Create-only exact-copy publication. Capsule registry excludes itself to avoid circular self-hashing.'}
write(CAP/'capsule.json',cap)
print(json.dumps({'capsule':str(CAP),'files':len(files),'bytes':cap['copied_bytes'],'capsule_sha256':sha(CAP/'capsule.json')}))
