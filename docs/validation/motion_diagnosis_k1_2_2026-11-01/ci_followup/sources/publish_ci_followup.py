"""Preserve a CI failure and bounded synthetic reproductions without array reads."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path('/Users/jaisel/Documents/mppi-gps')
ART=ROOT/'artifacts/2026-11-01_motion_diagnosis_k1_2_001'
DEST=ROOT/'docs/validation/motion_diagnosis_k1_2_2026-11-01/ci_followup'
DEST.mkdir(parents=True,exist_ok=False)
files={}
walls=[]
def copy(source,name):
    target=DEST/name
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open('rb') as inp,target.open('xb') as out:shutil.copyfileobj(inp,out)
    data=target.read_bytes()
    files[name]=dict(source=str(source),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

for folder in sorted((ART/'engineering').iterdir()):
    if folder.name[:3].isdigit() and 43<=int(folder.name[:3])<=50:
        for path in sorted(folder.iterdir()):
            if path.is_file():copy(path,f'commands/{folder.name}/{path.name}')
        if (folder/'receipt.json').is_file():walls.append(json.loads((folder/'receipt.json').read_text())['wall_seconds'])
for name in ('reproduce_report_ci.py','reproduce_report_ci_002.py','reproduce_report_ci_003.py','retrieve_ci_synthetic_receipts.py','publish_ci_followup.py'):
    copy(ART/name,'sources/'+name)

rows=[]
retrieved=json.loads((ART/'engineering/048_linux_receipt_retrieval/stdout.txt').read_text())
for case,items in retrieved.items():
    for name,binding in items.items():
        data=binding['content'].encode()
        assert len(data)==binding['bytes'] and hashlib.sha256(data).hexdigest()==binding['sha256']
        target=DEST/'synthetic'/case/name
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(data)
        files[str(target.relative_to(DEST))]=dict(remote_source=f'robopil5090:artifacts/2026-11-01_motion_diagnosis_k1_2_001/{case}/{name}',bytes=len(data),sha256=binding['sha256'])
    receipt=json.loads(items['report/receipt.json']['content'])
    rows.append(dict(case=case,execution_complete=receipt['execution_complete'],peak_rss_bytes=receipt['peak_rss_bytes'],raw_read_bytes=receipt['raw_read_bytes'],report_wall_seconds=receipt['wall_seconds']))
record=dict(
    schema='walker-k1-2-ci-isolation-followup-v1',
    failed_commit='d95e68cc32a032c24bd31b551ceb41012188d3c6',
    failed_workflow_url='https://github.com/jaiselsingh1/mppi-gps/actions/runs/41280422071',
    failed_job_id=124771048213,
    failed_ci_tests=dict(passed=813,failed=1,skipped=1,xfailed=1,wall_seconds=149.15),
    original_child_error_available=False,
    diagnosis='Linux high-water RSS inheritance is reproduced under bounded parent memory pressure; same frozen report passes from a light parent and after isolated grandchild launch. This supports the CI diagnosis; original CI child stderr/receipt were hidden, not observed.',
    implementation_change='Test helper only: lightweight intermediary plus stdout/stderr/receipt failure diagnostics. No active scientific/report production module, recipe, budget or frozen payload changed.',
    focused_tests_passed=13,
    ruff_active_files=202,
    synthetic_attempts=3,
    synthetic_controls_per_fixture=308,
    synthetic_pair_queries=162,
    synthetic_aligned_state_pairs=3438,
    synthetic_parent_pressure_bytes_per_pressure_case=600*1024**2,
    cases=rows,
    new_original_evidence_reads_or_research_calls=0,
    metered_enclosing_local_command_wall_seconds_sum=sum(walls),
    scope='Three explicitly counted synthetic probes only; global full-suite fixture work remains unknown. Enclosing command walls include remote/report subprocesses; do not add internal walls again. Original CI149.15s tests are separate from these local commands. Remaining CI setup/cleanup job wall and billing are not equated with CPU work. Original failed CI logs retained.',
    later_overhead='Closes before own commit/push. Final CI and later publication costs remain append-only in local engineering directory outside Git.',
    files=files,
)
with (DEST/'receipt.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(dict(files=len(files),metered_wall_seconds=sum(walls),cases=rows)))
