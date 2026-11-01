"""Publish exact small H5 preparation metadata; omit benchmark arrays."""
from pathlib import Path
import hashlib
import json
import shutil
import time

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT/'artifacts/2026-11-01_teacher_proposal_h5_0_preparation_001'
RUN = ROOT/'runs/2026-11-01_teacher_proposal_h5_0_preparation_001'
CAP = ROOT/'results/teacher_proposal_h5_0_preparation'
started = time.perf_counter()
CAP.mkdir(exist_ok=False)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')
copied = {}
def copy(source, name):
    target = CAP/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source,target)
    assert sha(source)==sha(target)
    copied[name]={'source':str(source),'sha256':sha(target),'bytes':target.stat().st_size}
for name in ('config.json','manifest.json','input_manifest.json','result.json','call_census.json','launch_receipt.json'):
    copy(RUN/name,'run/'+name)
for path in sorted((RUN/'metadata').iterdir()):
    copy(path,'run/metadata/'+path.name)
for folder, target in (('audit_preparation_001','audit'),('report_preparation_001','report'),('engineering','engineering')):
    for path in sorted((ART/folder).rglob('*')):
        if path.is_file(): copy(path,target+'/'+str(path.relative_to(ART/folder)))
for name in ('seed_review.py','runtime_probe.py'):
    copy(ART/name,'preflight/'+name)
for name in ('teacher_proposal_h5_0_execution_protocol.md','teacher_proposal_h5_0_hypothesis.md','teacher_proposal_h5_0_metric_review.md'):
    copy(RUN/'source/docs'/name,'protocol/'+name)
catalog = {str(p.relative_to(RUN)):{'bytes':p.stat().st_size,'sha256':sha(p)}
           for p in sorted(RUN.rglob('*')) if p.is_file()}
write(ART/'publication/full_preparation_catalog.json',catalog)
copy(ART/'publication/full_preparation_catalog.json','external/full_preparation_catalog.json')
copy(Path(__file__),'publication/publish_preparation.py')
receipts = {str(p.parent.name):json.loads(p.read_text()) for p in sorted((ART/'engineering').glob('*/receipt.json'))}
accounting = {
 'schema':'walker-h5-preparation-accounting-v1',
 'scope':'Engineering preparation only; prospective scientific work is not executed work.',
 'h5_research_executed':{'parent_fits':0,'child_fits':0,'network_forward_rows':0,'native_controls':0,'mppi_queries':0,'optimizer_steps':0,'round_audit_neural_rows':0},
 'preparation':json.loads((RUN/'result.json').read_text()),
 'preparation_audit':{k:v for k,v in json.loads((ART/'audit_preparation_001/receipt.json').read_text()).items() if k!='bindings'},
 'metadata_seed_review':{'source_read_bytes':12512791,'wall_seconds':0.181359375,'proposed_seeds':94,'top_level_configurations':234,'conservative_intervals':3326,'collisions':0},
 'enclosing_metered_commands':receipts,
 'enclosing_command_wall_seconds_sum':sum(r['wall_seconds'] for r in receipts.values()),
 'wall_time_interpretation':'Command times include nested subprocesses and must not be added to their internal timers or interpreted as total investigation elapsed time.',
 'engineering_tests':{'latest_full_suite':{'passed':979,'skipped':1,'xfailed':1,'receipt':'engineering/009_stable_full_suite/receipt.json'},'synthetic_and_baseline_fitting_or_physics':'Tests include small real Torch fitting/checkpoint continuation and existing native physics checks. Their aggregate neural/control/update census was not instrumented and is unknown; the H5 research-work zero does not describe those tests.'},
 'retained_failures':[
  '006_full_engineering: 2 audit fixture failures while required integration/warmstart evidence was missing; fixed fixtures, preserved audit contracts; 966 passed.',
  '013_export_smokes: wrong test filename; no tests ran, exit4; corrected invocation preserved separately.',
  'Proposal agent review retains initial synthetic fixture failures; later passing receipts supersede the engineering status without erasing failures.',
  '021 remote backup rejected by automatic approval review before execution; no archive or transfer was made.'
 ],
 'prospective_scientific_reservations':json.loads((RUN/'call_census.json').read_text()),
 'prospective_storage_reservation':json.loads((RUN/'config.json').read_text())['storage_review']['census'],
 'preparation_files':{'count':len(catalog),'bytes':sum(r['bytes'] for r in catalog.values())},
 'offhost_backup':{'status':'not_performed','reason':'Automatic approval review rejected exporting W2 inputs and metadata; optional approval question pending. Published code is on origin/main; full local preparation is intact.','bytes_transferred':0},
 'uninstrumented_or_unknown':['Interactive editing/tool/agent overhead','Aggregate CPU arithmetic and native/NN calls in engineering tests','Peak RAM for local engineering/preparation processes','Global filesystem reads/cache costs beyond recorded phases','Billing, energy and any later publication/CI overhead'],
 'no_efficiency_or_learning_benefit_claim':True
}
write(ART/'publication/accounting.json',accounting)
copy(ART/'publication/accounting.json','accounting/accounting.json')
readme = '''# H5.0 engineering preparation — unlaunched

The fixed-metric three-arm implementation and first-round protocol are prepared. This capsule records **engineering validation**, not an H5 scientific result. The useful-learning prerequisite remains unmet; the proposed one-round weak-learner exception and launch authorization remain pending. No parent fit, acquisition, child fit or full prediction audit ran.

- [Exact frozen recipe](run/config.json), [call census](run/call_census.json), [preparation result](run/result.json) and [input seal](run/input_manifest.json).
- [Execution protocol](protocol/teacher_proposal_h5_0_execution_protocol.md), [hypothesis](protocol/teacher_proposal_h5_0_hypothesis.md), and [fixed-metric review](protocol/teacher_proposal_h5_0_metric_review.md). These are exact frozen snapshots; their relative links resolve in the original source tree. Maintained versions are in [docs](../../docs/teacher_proposal_h5_0_execution_protocol.md).
- [Independent preparation audit](audit/receipt.json): 490 metadata/hash/census checks, zero array decodes, inference, physics or updates. This is not the future full scientific audit.
- [Engineering suite](engineering/009_stable_full_suite/stdout.txt): 979 passed, one skip, one expected failure; [Ruff](engineering/010_final_ruff/stdout.txt), [formatting](engineering/011_final_format/stdout.txt), [preservation checks](engineering/012_preserved_evidence/stdout.txt), and retained earlier failures.
- [Read-only report](report/report.md), [complete phase accounting](accounting/accounting.json), and [full external preparation catalog](external/full_preparation_catalog.json).

I uses task-only proposals and selector; S adds agreement only to the selector; P also guides the isolated second proposal stream. The protected producer remains actor-free. D, its coefficient, representation and temporal reduction stay fixed. Useful learning is measured by fresh standalone walking after exactly 2,000 child updates, not agreement or reconstruction. All controls, budgets, stop rules and admission gates are frozen before any future launch.

Full-roster prospective maxima are 1,339,255,104 native controls, 4,005,571,104 worker-plus-audit neural rows and 24,000 Adam updates. The conditional uncompressed evidence bound is 491,755,966,800 bytes under a512GiB ceiling; worker plus audit reserve eight wall-hours on16 cores. These quantities were **not spent** in preparation. Small synthetic/baseline tests have separate, incompletely instrumented compute costs.

The exact preparation is local at `/Users/jaisel/Documents/mppi-gps/runs/2026-11-01_teacher_proposal_h5_0_preparation_001`. W2 NPZ inputs and the full473-file source export remain outside Git; their hashes are retained. Every frozen source file matches commit `a19ad4ca31c07d393497f610f228810289839eb2`. The optional full off-host backup was blocked before transfer by automatic approval review. No H5 worker workspace was launched on `robopil5090`.

Use the read-only inspector on this capsule; never use a capsule or existing run as an execution output. A later authorized run needs a new frozen configuration, explicit authorization and its authenticated watchdog. This preparation cannot authorize itself.
'''
(ART/'publication/README.md').write_text(readme)
copy(ART/'publication/README.md','README.md')
capsule = {
 'schema':'walker-result-capsule-v1','study':'teacher_proposal_h5_0_preparation',
 'run_id':RUN.name,'status':'prepared_unlaunched','integrity_status':'preparation_metadata_audit_passed',
 'implementation_commit':'a19ad4ca31c07d393497f610f228810289839eb2',
 'scientific_round_audit_complete':False,'research_execution_authorized':False,
 'files':copied,'copied_file_count':len(copied),'copied_bytes':sum(r['bytes'] for r in copied.values()),
 'bulk_evidence':{'local_directory':str(RUN),'remote_directory':None,'remote_copy_verified':False,
   'omitted':['inputs/train.npz','inputs/validation.npz','inputs/normalization.npz','inputs/task.json','source/'],
   'scope':'Exact benchmark inputs and full frozen source remain local external evidence. Their complete per-file hashes and sizes are in external/full_preparation_catalog.json. No research arrays, checkpoints or ledgers exist yet.'},
 'new_research_physics':0,'new_research_inference':0,'new_research_training':0,
 'scope':'Metadata preparation, code and synthetic validation only; no scientific result or admitted child.'
}
write(CAP/'capsule.json',capsule)
receipt={'files':len(copied),'bytes':capsule['copied_bytes'],'capsule_sha256':sha(CAP/'capsule.json'),
         'wall_seconds':time.perf_counter()-started,'full_preparation_files':len(catalog),
         'full_preparation_bytes':sum(r['bytes'] for r in catalog.values())}
write(ART/'publication/publication_receipt.json',receipt)
print(json.dumps(receipt,indent=2))
