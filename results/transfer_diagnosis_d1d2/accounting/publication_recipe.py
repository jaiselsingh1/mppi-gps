from pathlib import Path
import hashlib,json,shutil,time
ROOT=Path('/Users/jaisel/Documents/mppi-gps')
RUN=ROOT/'runs/2026-10-31_transfer_diagnosis_d1d2_001'
ART=ROOT/'artifacts/2026-10-31_transfer_diagnosis_d1d2_001'
OUT=ROOT/'results/transfer_diagnosis_d1d2'

def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()

def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

if __name__=='__main__':
    started=time.perf_counter()
    r=json.loads((RUN/'result.json').read_text())
    audit=json.loads((ART/'audit_001/receipt.json').read_text())
    report=json.loads((ART/'report_001/receipt.json').read_text())
    full=json.loads((ART/'local_full_run_file_manifest.json').read_text())
    assert r['execution_complete'] and not r['error'] and audit['passed']
    assert (ART/'local_full_run_file_manifest.json').read_bytes()==(ART/'remote_full_run_file_manifest.json').read_bytes()
    engineering={
        'schema':'walker-transfer-accounting-v1',
        'new_research':r['work'],
        'stage_resources':{k:{n:r[k][n] for n in ['worker_wall_seconds','peak_memory_bytes']} for k in ['d1','d2']},
        'external_launch':json.loads((RUN/'launch_exit.json').read_text()),
        'independent_audit':audit,
        'full_report':report,
        'prelaunch_validation':json.loads((ART/'engineering/clean_export.json').read_text()),
        'prelaunch_ci':json.loads((ART/'engineering/prelaunch_ci.json').read_text()),
        'preparation_and_transfers':{
            'input_validation':'Three separate saved-input preparation/ownership validations, including independent13,254-check lineage validation; zero neural/physics/fitting. Temporary validated copies remain external; exact global engineering arithmetic and full elapsed time were not metered.',
            'sealed_input_files':295,'portable_validation_input_bytes':124298288,
            'initial_export_files_and_directories':566,'initial_export_logical_bytes':128194476,
            'closed_full_run_files':full['file_count'],'closed_full_run_bytes_per_mirror':full['total_bytes'],
            'closed_mirror_additional_transferred_file_bytes':769982515,
            'audit_mirror_file_bytes':87259490,
            'full_report_bytes':sum(p.stat().st_size for p in (ART/'report_001').iterdir() if p.is_file()),
            'full_report_derived_curve_cache_bytes':(ART/'report_001/derived_curves.json.gz').stat().st_size,
            'hash_verification':'Independent complete-file SHA256 manifests match on local and remote hosts.',
            'known_artifact_storage_under_8GiB':True,
            'storage_scope':'Worker enforces run/input/source/evidence ceiling. Physical local/remote mirrors, validation copies, audit registry, reports and Git copies are separate preservation overhead; cloud/CI retention footprint is unavailable.'
        },
        'historical_lineage':{
            'unique_direct_source_runs':['2026-10-31_action_compatibility_k1_001','2026-10-31_student_update_w2_001'],
            'source_capsules':['results/action_compatibility_k1','results/student_update_w2'],
            'accounting':'Historical acquisition, learning and evaluation remain charged in original ledgers/capsules. Shared older ancestors are references, not summed again. Imported predictions are not new inference or fitting. No lifetime efficiency total is claimed.'
        },
        'engineering_validation':'Synthetic tests, saved-only audits, plotting, metadata inspection, hashing and transfer are separately identified; global synthetic neural/physics operation counts are unavailable, not zero. No real roster was inferred or stepped before sealing.',
        'unknowns':['Engineering elapsed time before complete external receipts','Global synthetic fixture arithmetic and cache-building operations','Model/tool billing','CI/cloud resource use and retention storage'],
        'no_efficiency_claim':True,
    }
    write(ART/'engineering_accounting.json',engineering)
    OUT.mkdir(exist_ok=False)
    bindings={}
    def copy(source,relative):
        target=OUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        assert sha(target)==sha(source)
        bindings[relative]={'source':str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),'sha256':sha(target),'bytes':target.stat().st_size}
    for name in ['config.json','manifest.json','input_manifest.json','result.json','prelaunch_receipt.json','prelaunch_inspection.json','launch_receipt.json','launch_exit.json','worker_started.json','launcher_started.json','launch.py']:
        copy(RUN/name,'run/'+name)
    for name in ['inventory.json','task.json']:
        copy(RUN/'inputs'/name,'run/inputs/'+name)
    excluded={'coverage_samples.csv','reconstruction_samples.csv','temporal_physical.csv','survivor_counts.csv'}
    for source in sorted((ART/'report_001').iterdir()):
        if source.is_file() and source.name not in excluded:copy(source,'report/'+source.name)
    for name in ['receipt.json','findings.json','file_bindings.json']:
        copy(ART/'audit_001'/name,'audit/'+name)
    copy(ART/'local_full_run_file_manifest.json','external/full_run_file_manifest.json')
    copy(ART/'mirror_receipt.json','external/mirror_receipt.json')
    copy(ART/'descriptive_summary.json','report/descriptive_summary.json')
    copy(ART/'engineering_accounting.json','accounting/engineering_accounting.json')
    for source in sorted((ART/'engineering').iterdir()):
        if source.is_file():copy(source,'accounting/prelaunch/'+source.name)
    for source,name in [(Path('/tmp/summarize_transfer.py'),'descriptive_aggregation.py'),(Path('/tmp/manifest_transfer_run.py'),'manifest_recipe.py'),(Path(__file__),'publication_recipe.py')]:
        copy(source,'accounting/'+name)
    capsule={
        'schema':'walker-result-capsule-v1','study':'transfer_diagnosis_d1d2','run_id':RUN.name,
        'scope':'Exact selected original metadata, audit summaries, figures, summary CSVs and hash-bound derived plotting curves. Complete native arrays/checkpoints/ledgers/full-row CSVs remain external.',
        'files':bindings,
        'bulk_evidence':{'local_run':str(RUN),'remote_host':'robopil5090','remote_run':'/home/jaisel/mppi-gps-workspace/runs/'+RUN.name,'local_artifacts':str(ART),'full_run_files':full['file_count'],'full_run_bytes':full['total_bytes'],'omitted':['evidence/raw.h5','evidence/ledger.jsonl','source/','inputs/checkpoints/','inputs/k1/','inputs/w2/','inputs/teachers/','audit_001/array_bindings.json']+[f'report_001/{name}' for name in sorted(excluded)]},
        'operation_scope':'Worker95f0a55 executed once; capsule copies and portable rendering perform no policy inference, fitting, MPPI search or native stepping.',
        'new_physics':0,'experimental_new_live_controls':21938,'new_inference':0,'new_training':0,
        'capsule_copy_wall_seconds':time.perf_counter()-started,
    }
    write(OUT/'capsule.json',capsule)
    (OUT/'README.md').write_text('''# D1/D2 exact result capsule

The diagnostic completed once and passed its independent audit. It changes no learner or planner. [Main interpretation](../../docs/transfer_diagnosis_d1d2_results.md) · [frozen protocol](../../docs/transfer_diagnosis_d1d2_protocol.md) · [full report](report/report.md) · [audit receipt](audit/receipt.json) · [next S1 proposal, unlaunched](../../docs/student_source_s1_protocol.md).

All 20 teacher replays had zero recorded numerical error before any substitution. All 240 branches completed their stopping logic: 21,938 native controls, 87,752 substeps; zero new inference, gradients, Adam or MPPI queries. D1 completed69,701 saved-support queries. Audit:1,061,961 assertions,531boundfiles,289,363arrays; assertion counts are not task replications.

Support is descriptive. All recorded first-100 portions of the2k students are inside the calibrated support threshold despite walking failure. D2 establishes single-action effects only under the saved open-loop teacher continuation; its ordering is not a validated teaching score or closed-loop competence ranking.

## What is tracked

Original configuration/results/source and input manifests, launch/exit receipts, nine PNG/PDF figure families, summary and consequence CSVs, independent audit summaries, full external file hashes, cost accounting, and a deterministic2.74MB derived-curve cache. The cache contains eight derived D1 plotting fields; it contains no full simulator states, actions, checkpoints or native D2 arrays. All selected original copies are byte-bound by `capsule.json`.

## Full external evidence

The complete run is544files /898,176,991bytes locally at `/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_transfer_diagnosis_d1d2_001` and on `robopil5090` at `/home/jaisel/mppi-gps-workspace/runs/2026-10-31_transfer_diagnosis_d1d2_001`. Both full manifests are byte-identical. [File hashes](external/full_run_file_manifest.json) · [mirror proof](external/mirror_receipt.json).

Full reports, every raw-row CSV and the87MBaudit-array registry are under the corresponding local `artifacts/2026-10-31_transfer_diagnosis_d1d2_001/` directory; the independent original audit is also retained in the remote workspace's artifacts directory. The full report receipt lists exact omitted-table hashes. Temporal/contributor tables can be regenerated from the tracked derived cache. Native arrays, ledgers, checkpoints and frozen source stay outside Git. Original discovery's missing local terminal/context fields remain explicitly missing and are not used forD2.

## Clean-checkout commands

```sh
python scripts/train_mppi_gps.py inspect --run results/transfer_diagnosis_d1d2 --verify
python scripts/diagnostics/report_transfer_diagnosis.py --run results/transfer_diagnosis_d1d2 --out /tmp/walker-transfer-report-fresh-001
```

Portable rendering verifies metadata/cache hashes and reproduces the nine figure families and summary/temporal tables without the ignored evidence directories. Full raw-row CSVs and a full independent scientific audit require the external run. [Research and engineering accounting](accounting/engineering_accounting.json) separates those operations and records unavailable overhead rather than assigning it zero.
''')
    print(json.dumps({'capsule':str(OUT),'bound_files':len(bindings),'bound_bytes':sum(v['bytes'] for v in bindings.values()),'full_run_manifest_sha256':sha(OUT/'external/full_run_file_manifest.json')},indent=2))
