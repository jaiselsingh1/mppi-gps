"""Version publication metadata after Git correctly excludes an external journal.

No scientific run, source snapshot, result, audit, report or accounting payload
is changed. The ignored local journal copy is left untouched.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT/'artifacts/2026-11-01_motion_diagnosis_k1_2_001'
PUB = ART/'publication'
CAP = ROOT/'results/motion_diagnosis_k1_2'
RUN = ROOT/'runs/2026-11-01_motion_diagnosis_k1_2_001'
started = time.perf_counter()
old_manifest_bytes = (CAP/'capsule.json').read_bytes()
old_readme_bytes = (CAP/'README.md').read_bytes()
old = json.loads(old_manifest_bytes)
old_sha = hashlib.sha256(old_manifest_bytes).hexdigest()
assert old_sha=='c8b5505b7d17abbaec39841fd948f4261a15e55448a77f433efa40b13971c993'
assert hashlib.sha256(old_readme_bytes).hexdigest()==old['files']['README.md']['sha256']
assert old['copied_file_count']==495 and old['copied_bytes']==43600350
journal = dict(old['files']['run/work.jsonl'])
assert journal['bytes']==1478297
assert journal['sha256']=='a167565202a7b939fdee362bc93c57e3a110c6fd115c2d0b8b0632d3f24703e6'
assert (RUN/'work.jsonl').stat().st_size==journal['bytes']
protected = {key:value for key,value in old['files'].items() if key not in ('README.md','run/work.jsonl')}
before_stats = {key:((CAP/key).stat().st_size,(CAP/key).stat().st_mtime_ns) for key in protected}
journal_before = {str(path):path.stat().st_mtime_ns for path in (RUN/'work.jsonl',CAP/'run/work.jsonl')}
new = json.loads(old_manifest_bytes)
new['files'].pop('run/work.jsonl')


def save_exact(source: Path, target: Path, data: bytes) -> dict:
    source.parent.mkdir(parents=True,exist_ok=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    with source.open('xb') as stream:
        stream.write(data)
    with target.open('xb') as stream:
        stream.write(data)
    return dict(source=str(source),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


for name,data in (('capsule.json',old_manifest_bytes),('README.md',old_readme_bytes)):
    relative = Path('historical/publication_001')/name
    new['files'][str(relative)] = save_exact(PUB/'historical_publication_001'/name,CAP/relative,data)
script_data = Path(__file__).read_bytes()
script_relative = Path('publication/portability_correction_002.py')
with (CAP/script_relative).open('xb') as stream:
    stream.write(script_data)
new['files'][str(script_relative)] = dict(source=str(Path(__file__).resolve()),sha256=hashlib.sha256(script_data).hexdigest(),bytes=len(script_data))
correction = dict(
    schema='walker-motion-publication-portability-correction-v2',
    scope='Publication manifest/navigation only; no scientific reexecution or altered evidence.',
    discovery='Clean Git export027 passed814 tests, then repository verification found the ignored work.jsonl missing despite the v1 manifest binding. The global *.jsonl exclusion correctly keeps journals outside Git.',
    failed_export_receipt=str(ART/'engineering/027_results_export_validation/receipt.json'),
    prior_publication=dict(manifest='historical/publication_001/capsule.json',sha256=old_sha,
                           bytes=len(old_manifest_bytes),readme='historical/publication_001/README.md',
                           readme_sha256=hashlib.sha256(old_readme_bytes).hexdigest(),
                           payload_files=495,payload_bytes=43600350),
    change='Remove run/work.jsonl from the published payload contract and bind its original RUN location externally. Preserve both prior publication documents and archive this correction source.',
    external_work_journal=dict(file='work.jsonl',local_file=str(RUN/'work.jsonl'),
                               sha256=journal['sha256'],bytes=journal['bytes'],
                               available_local=True,in_git=False,remote_copy_verified=False),
    accounting_snapshot=dict(file='accounting/accounting.json',publication_version=1,
                              sha256=old['files']['accounting/accounting.json']['sha256'],
                              scope='Unchanged scientific/accounting record. Its publication495-file snapshot remains true for v1; capsule.json governs the corrected portable v2 payload.'),
    scientific_payloads='All run configurations/results/source manifests, audit/report outputs and accounting bytes are unchanged. The full original run and its373-file catalog remain valid.',
    ignored_local_copy='Existing capsule/run/work.jsonl is left untouched as an ignored local extra; it is not part of the corrected published contract.',
    policy='No force-add and no change to .gitignore. No rerun, array read, audit or new PDF authoring.',
    cost_capture_external=str(ART/'engineering/029_portability_correction/receipt.json'),
)
correction_data = (json.dumps(correction,indent=2,allow_nan=False)+'\n').encode()
relative = Path('publication/portability_correction_002.json')
new['files'][str(relative)] = save_exact(PUB/'portability_correction_002.json',CAP/relative,correction_data)
readme = old_readme_bytes.decode()
original = 'The entire new run is copied here except `evidence/motion.npz`: 6,064,785 bytes, SHA-256 `55337bee1245059c96e208865fa04730364be0267b8489c42558073b16aff783`. The [full run catalog](external/full_run_file_manifest.json) binds that local-only raw archive and all 373 closed-run files.'
replacement = 'Two new-run files remain external: `evidence/motion.npz` (6,064,785 bytes, SHA-256 `55337bee1245059c96e208865fa04730364be0267b8489c42558073b16aff783`) and `work.jsonl` (1,478,297 bytes, SHA-256 `a167565202a7b939fdee362bc93c57e3a110c6fd115c2d0b8b0632d3f24703e6`). Both remain in the full local RUN. The [full run catalog](external/full_run_file_manifest.json) binds all 373 closed-run files.'
assert original in readme
readme = readme.replace(original,replacement)
readme += '\nPublication v2 [corrects the portable payload](publication/portability_correction_002.json): the global journal ignore rule excluded `work.jsonl` from Git export. The exact [v1 manifest](historical/publication_001/capsule.json) and [v1 navigation](historical/publication_001/README.md) are preserved. Scientific evidence and the original accounting snapshot are unchanged; that accounting describes publication v1. The current manifest defines publication v2.\n'
readme_data = readme.encode()
readme_source = PUB/'README_capsule_002.md'
with readme_source.open('xb') as stream:
    stream.write(readme_data)
# Only these two current publication/navigation files are intentionally replaced.
(CAP/'README.md').write_bytes(readme_data)
new['files']['README.md'] = dict(source=str(readme_source),sha256=hashlib.sha256(readme_data).hexdigest(),bytes=len(readme_data))
new['bulk_evidence']['omitted'] = ['evidence/motion.npz','work.jsonl']
new['bulk_evidence']['work_journal'] = correction['external_work_journal']
new['bulk_evidence']['scope'] = 'Entire new run except its compact raw archive and work journal; both remain external/local. Full raw atlas and all published scientific payloads are unchanged.'
new['publication_version'] = 2
new['publication_correction'] = str(relative)
new['publication_history'] = [correction['prior_publication']]
new['accounting_publication_snapshot'] = correction['accounting_snapshot']
new['operation_scope'] = 'Publication v2 metadata/navigation correction. Original v1 manifest/navigation are preserved. Registry excludes itself; ignored local extras are not published payload. Scientific payloads and the original full run are unchanged.'
new['copied_file_count'] = len(new['files'])
new['copied_bytes'] = sum(value['bytes'] for value in new['files'].values())
assert new['copied_file_count']==498
assert {key:new['files'][key] for key in protected}==protected
for key,stat in before_stats.items():
    path = CAP/key
    assert (path.stat().st_size,path.stat().st_mtime_ns)==stat
for path,mtime in journal_before.items():
    assert Path(path).stat().st_mtime_ns==mtime
manifest_data = (json.dumps(new,indent=2,allow_nan=False)+'\n').encode()
(CAP/'capsule.json').write_bytes(manifest_data)
receipt = dict(schema='walker-motion-publication-correction-receipt-v2',execution_complete=True,
               publication_version=2,prior_capsule_sha256=old_sha,
               capsule_sha256=hashlib.sha256(manifest_data).hexdigest(),manifest_bytes=len(manifest_data),
               payload_files=new['copied_file_count'],payload_bytes=new['copied_bytes'],
               run_files_published=371,run_bytes_published=18705678,
               external_work_journal=correction['external_work_journal'],
               protected_payload_entries_unchanged=len(protected),
               metadata_source_read_bytes=len(old_manifest_bytes)+len(old_readme_bytes)+len(script_data),
               raw_array_reads=0,journal_payload_reads=0,scientific_reexecutions=0,
               wall_seconds=time.perf_counter()-started,
               scope='Current manifest/README versioned; old bytes and all scientific/accounting payloads preserved. Detailed file-system/tool overhead is not completely instrumented, not zero.')
with (PUB/'portability_correction_receipt_002.json').open('x') as stream:
    json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps(receipt,indent=2))
