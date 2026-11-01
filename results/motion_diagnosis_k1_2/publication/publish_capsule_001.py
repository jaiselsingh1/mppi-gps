"""Create-only publication of closed K1.2 evidence; never decode motion arrays."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT/'artifacts/2026-11-01_motion_diagnosis_k1_2_001'
RUN = ROOT/'runs/2026-11-01_motion_diagnosis_k1_2_001'
CAP = ROOT/'results/motion_diagnosis_k1_2'
PUB = ART/'publication'
STARTED = time.perf_counter()
metadata_read_bytes = 0
copy_read_bytes = 0
copy_write_bytes = 0
destination_hash_read_bytes = 0
files: dict[str, dict] = {}


def read(path: Path) -> dict:
    global metadata_read_bytes
    data = path.read_bytes()
    metadata_read_bytes += len(data)
    return json.loads(data)


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path: Path) -> str:
    global destination_hash_read_bytes
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024**2):
            destination_hash_read_bytes += len(chunk)
            value.update(chunk)
    return value.hexdigest()


def copy(source: Path, relative: Path, expected: dict | None = None) -> None:
    global copy_read_bytes, copy_write_bytes
    assert source.is_file() and not source.is_symlink(), source
    assert source.suffix.lower() not in ('.npz','.npy','.h5','.hdf5','.pt','.pth','.mp4','.webm'), source
    target = CAP/relative
    assert target.resolve().is_relative_to(CAP.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    before = source.stat()
    source_hash = hashlib.sha256()
    with source.open('rb') as reader, target.open('xb') as writer:
        while chunk := reader.read(1024**2):
            copy_read_bytes += len(chunk)
            copy_write_bytes += len(chunk)
            source_hash.update(chunk)
            writer.write(chunk)
    after = source.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns), source
    binding = dict(sha256=source_hash.hexdigest(),bytes=before.st_size)
    assert digest(target)==binding['sha256'] and target.stat().st_size==binding['bytes']
    if expected is not None:
        assert all(binding[key]==expected[key] for key in binding), source
    key = str(relative)
    assert key not in files
    files[key] = dict(source=str(source),**binding)


def tree(source: Path, relative: Path) -> None:
    for path in sorted(source.rglob('*')):
        if path.is_file():
            copy(path,relative/path.relative_to(source))


assert not CAP.exists(), 'Never replace a published capsule'
result = read(RUN/'result.json')
owner = read(RUN/'worker_exit.json')
audit = read(ART/'audit_001/receipt.json')
findings = read(ART/'audit_001/findings.json')
audit_bindings = read(ART/'audit_001/file_bindings.json')
report = read(ART/'report_001/receipt.json')
preflight = read(ART/'preflight_accounting_007.json')
independent_metadata = read(ART/'independent_metadata_review_001/receipt.json')
publication = read(ART/'prelaunch_publication.json')
inspection = read(ART/'visual_qa/inspection_001.json')
assert result['execution_complete'] and result['error'] is None
assert owner['execution_complete'] and owner['error'] is None and owner['exit_code']==0
assert audit['passed'] and audit['error'] is None
assert report['execution_complete'] and report['error'] is None and inspection['passed']
raw = result['motion_archive']
assert raw == dict(bytes=6064785,file='evidence/motion.npz',sha256='55337bee1245059c96e208865fa04730364be0267b8489c42558073b16aff783')
assert (RUN/raw['file']).stat().st_size==raw['bytes']
closed = [path for path in sorted(RUN.rglob('*')) if path.is_file()]
closed_bytes = sum(path.stat().st_size for path in closed)
assert len(closed)==373 and closed_bytes==findings['closed_run_bytes']==26248760
assert all(str(path.resolve()) in audit_bindings for path in closed)
commands = []
for path in sorted((ART/'engineering').glob('*/receipt.json')):
    receipt = read(path)
    for stream in ('stdout','stderr'):
        data = (path.parent/(stream+'.txt')).read_bytes()
        metadata_read_bytes += len(data)
        assert hashlib.sha256(data).hexdigest()==receipt[stream+'_sha256']
    commands.append(dict(directory=str(path.parent.relative_to(ART)),**receipt))
worker_io = result['work']['source_hash_bytes']+result['work']['source_io_bytes']
audit_io = audit['work']['total_hash_bytes']+audit['work']['source_io_bytes']+audit['work']['compact_io_bytes']
assert worker_io+audit_io==2942449292
accounting = dict(
    schema='walker-motion-diagnosis-accounting-v1',
    scope='One new saved-motion worker, independent recomputation, one report and visual QA. New file reads are paid work; historical acquisition, inference and fitting remain inherited separate records.',
    run=dict(attempts=1,execution_complete=True,work=result['work'],counts=result['counts'],
             worker_wall_seconds=result['wall_seconds'],peak_rss_bytes=result['peak_rss_bytes'],
             owner=owner,closed_files=len(closed),closed_bytes=closed_bytes,
             compact_raw_archive=raw),
    audit=dict(attempts=1,final_receipt=audit,earlier_findings_work_snapshot=findings['independent_work'],
               final_receipt_authoritative=True,
               snapshot_difference='The final audit receipt includes the last closed-run rehash after the findings snapshot. Preserve both; use final total_hash_bytes for accounting.'),
    report=report,
    visual_qa=dict(inspection=inspection,
                   page_render_receipts={kind:read(ART/'visual_qa'/(kind+'_pages_001')/'receipt.json') for kind in ('raw','derived')},
                   contact_sheets=read(ART/'visual_qa/contact_sheets_001/receipt.json')),
    logical_file_io=dict(worker=worker_io,audit_final=audit_io,worker_plus_audit=worker_io+audit_io,
                         report_compact_reads=report['raw_read_bytes'],
                         decoded_arrays_are_not_extra_file_io=True,
                         convention='Audit total_hash_bytes already contains source_hash_bytes. Add its source_io_bytes and compact_io_bytes once. Do not add decoded/heap bytes or archive size again.'),
    preflight=preflight,
    independent_metadata_preflight=independent_metadata,
    metered_commands=commands,
    failed_metered_commands=[row['directory'] for row in commands if row['exit_code']!=0],
    metered_command_wall_seconds_sum=sum(row['wall_seconds'] for row in commands),
    overlap='Command times enclose their worker/audit/report subprocess times. Do not add nested times again or equate their sum with elapsed investigation time.',
    measurement_failures='Two metadata shorthand-path failures and the failed prelaunch full-test deadline case remain preserved. The fixed deadline/report regression commands do not rerun the scientific worker.',
    inherited_evidence=dict(
        k1='results/action_compatibility_k1/capsule.json',
        k1_1='results/trajectory_review_k1_1/capsule.json',
        scope='Inherited teacher discovery, reference training, previous student fits and earlier diagnostics are not pooled with new K1.2 costs or claimed paid again. K1 raw HDF/NPZ/checkpoints remain local-only.'),
    publication=dict(source='publication/publish_capsule_001.py',
                     receipt_external=str(PUB/'publication_receipt_001.json'),
                     scope='Create-only exact copies; final copied payload counts are in capsule.json. Publication receipt is external to avoid a circular manifest hash.'),
    unknown_not_zero=['Uninstrumented source editing, helper/source-generation and interactive visual-review wall time',
                      'Interpreter/import IO, Poppler IO beyond named source-file hash reads, tool/token billing and energy',
                      'Global synthetic-test training/physics/arithmetic counts; tests use independent small fixtures',
                      'Unretained historical ancestor overhead'],
    later_validation='Subsequent publication/export/CI receipts are retained separately. This capsule and accounting snapshot are not mutated to append them.',
)
write(PUB/'accounting_001.json',accounting)
CAP.mkdir()
for path in closed:
    relative = path.relative_to(RUN)
    if str(relative)==raw['file']:
        continue
    copy(path,Path('run')/relative,audit_bindings[str(path.resolve())])
tree(ART/'report_001',Path('report'))
tree(ART/'audit_001',Path('audit_001'))
for command in commands:
    # A live publisher/validation command has no closed receipt yet and remains external.
    tree(ART/command['directory'],Path(command['directory']))
tree(ART/'independent_metadata_review_001',Path('independent_metadata_review_001'))
for path in sorted(ART.iterdir()):
    if path.is_file() and path.name.startswith(('preflight','motion_input_roster','normalize_preflight')):
        copy(path,Path('preflight')/path.name)
tree(ART/'visual_qa/contact_sheets_001',Path('visual_qa/contact_sheets_001'))
copy(ART/'visual_qa/inspection_001.json',Path('visual_qa/inspection_001.json'))
for kind in ('raw','derived'):
    folder = ART/'visual_qa'/(kind+'_pages_001')
    for name in ('receipt.json','stdout.txt','stderr.txt'):
        copy(folder/name,Path('visual_qa')/folder.name/name)
for name in ('render_atlas_pages_001.py','contact_sheets_001.py'):
    copy(ART/'visual_qa'/name,Path('visual_qa/scripts')/name)
copy(ART/'prelaunch_publication.json',Path('publication/prelaunch_publication.json'))
copy(PUB/'accounting_001.json',Path('accounting/accounting.json'))
copy(Path(__file__).resolve(),Path('publication/publish_capsule_001.py'))
catalog = dict(schema='walker-motion-full-run-file-catalog-v1',run_id=RUN.name,local_directory=str(RUN),
               file_count=len(closed),total_bytes=closed_bytes,
               files={str(path.relative_to(RUN)):audit_bindings[str(path.resolve())] for path in closed},
               integrity='Every closed-run file is independently bound by audit001, including the omitted raw archive. Copies in this capsule were compared with those exact bindings.',
               omitted_from_git=[raw['file']])
write(PUB/'full_run_file_manifest_001.json',catalog)
copy(PUB/'full_run_file_manifest_001.json',Path('external/full_run_file_manifest.json'))
readme='''# K1.2 saved-motion diagnosis

One source-frozen diagnosis of all 34 K1 trajectories, 30,070 controls and 561 cross-trajectory pairs. No new inference, physics, fitting or acquisition ran. The independent audit passed 380,112 checks. Motion and short-window similarity do not establish gait phase, teachability, a transfer cause or useful coupling.

- [Result and causal verdict](../../docs/motion_diagnosis_k1_2_results.md); [frozen protocol](../../docs/motion_diagnosis_k1_2_protocol.md).
- [Figures and complete tables](report/report.md); [full actual-motion atlas](report/full_motion_atlas.pdf); [derived summary atlas](report/derived_curve_atlas.pdf).
- [Frozen result](run/result.json), [independent audit](audit_001/receipt.json) and [accounting](accounting/accounting.json).
- [All-page visual review](visual_qa/inspection_001.json) and [ordered contact sheets](visual_qa/contact_sheets_001/receipt.json).
- [Next H5.0 hypothesis](../../docs/teacher_proposal_h5_0_hypothesis.md), untested and not authorized for launch.

The entire new run is copied here except `evidence/motion.npz`: 6,064,785 bytes, SHA-256 `55337bee1245059c96e208865fa04730364be0267b8489c42558073b16aff783`. The [full run catalog](external/full_run_file_manifest.json) binds that local-only raw archive and all 373 closed-run files. Original K1 raw HDF/NPZ/checkpoints remain external and local-only; no legacy raw arrays were copied to Git. Four discovery endpoints remain missing. Available terminal states do not create terminal actions.

Clean-checkout reporting renders saved block means/min/max and copies the exact bound original full-motion atlas and fixed-pair figures. It explicitly discloses this provenance; it does not reconstruct raw motion from block summaries. All 68 page PNGs remain external with hashes in the visual-review receipts; only compact contact sheets are included here.

```sh
python -B scripts/train_mppi_gps.py inspect --run results/motion_diagnosis_k1_2 --verify --json
python -B scripts/diagnostics/report_motion_diagnosis.py --run results/motion_diagnosis_k1_2 --out /absolute/fresh/external/report
```

The two failed preflight path attempts, independent metadata-check timing limitation, failed deadline test, corrected regression checks, source publication, report marker, rendering and QA costs are preserved. Do not execute workers inside this immutable capsule. Later publication validation stays separate.
'''
write(PUB/'readme_source_001.json',dict(text=readme))
readme_source = PUB/'README_capsule_001.md'
with readme_source.open('x') as stream:
    stream.write(readme)
copy(readme_source,Path('README.md'))
capsule = dict(schema='walker-result-capsule-v1',study='motion_diagnosis_k1_2',run_id=RUN.name,
               status='saved_motion_analysis_complete',integrity_status='independent_motion_audit_passed',
               implementation_commit=publication['implementation_commit'],
               scope='Saved motion inspection and one frozen ordered-segment probe only. No new control, policy learning, teacher acquisition or optimization feedback.',
               files=files,copied_file_count=len(files),copied_bytes=sum(row['bytes'] for row in files.values()),
               bulk_evidence=dict(local_directory=str(RUN),remote_directory=None,remote_copy_verified=False,
                                  omitted=[raw['file']],raw_archive=raw,
                                  scope='Entire new run except its compact raw archive. Original full raw atlas is preserved here as an inspectable exact asset; inherited K1 raw remains local-only outside Git.'),
               new_physics=0,new_inference=0,new_training=0,
               operation_scope='Create-only exact-copy publication. Registry excludes itself; source files, closed runs and older capsules are never modified.')
write(CAP/'capsule.json',capsule)
manifest_bytes = (CAP/'capsule.json').read_bytes()
publication_receipt = dict(schema='walker-motion-capsule-publication-v1',execution_complete=True,
                           capsule=str(CAP),copied_files=len(files),copied_bytes=capsule['copied_bytes'],
                           capsule_sha256=hashlib.sha256(manifest_bytes).hexdigest(),capsule_manifest_bytes=len(manifest_bytes),
                           copy_source_read_bytes=copy_read_bytes,copy_destination_write_bytes=copy_write_bytes,
                           copied_destination_hash_read_bytes=destination_hash_read_bytes,
                           metadata_input_read_bytes=metadata_read_bytes,
                           wall_seconds=time.perf_counter()-STARTED,
                           raw_motion_array_reads=0,omitted_raw_archive_rehashed=False,
                           source_contract='Raw archive hash was inherited from the passed audit; only its size was inspected here. The publisher reads and byte-verifies selected nonraw files.',
                           scope='Payload-copy and metadata-reader counters are explicit logical IO. Interpreter/import, manual source authoring and output JSON serialization overhead are not fully instrumented; not zero.')
write(PUB/'publication_receipt_001.json',publication_receipt)
print(json.dumps(publication_receipt,indent=2))
