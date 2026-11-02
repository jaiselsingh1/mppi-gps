"""Create-only late engineering closure; never opens scientific array payloads."""

from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT / 'artifacts/2026-11-01_motion_diagnosis_k1_2_001'
DEST = ROOT / 'docs/validation/motion_diagnosis_k1_2_2026-11-01'
DEST.mkdir(parents=True, exist_ok=False)
files = {}


def copy(source, name):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as inp, target.open('xb') as out:
        shutil.copyfileobj(inp, out)
    data = target.read_bytes()
    files[name] = dict(source=str(source), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


commands = []
for folder in sorted((ART / 'engineering').iterdir()):
    # Earlier metered work is already in the immutable capsule.
    if folder.name[:3].isdigit() and int(folder.name[:3]) >= 25:
        receipt = json.loads((folder / 'receipt.json').read_text())
        commands.append(dict(name=folder.name, **receipt))
        for path in sorted(folder.iterdir()):
            if path.is_file():
                copy(path, f'commands/{folder.name}/{path.name}')

for name in [
    'integrated_report_verification.json', 'integrated_report_verification_002.json',
    'verify_integrated_report.py', 'verify_integrated_report_002.py',
    'obsidian_update.json', 'update_obsidian.py', 'verify_obsidian_update_002.py',
    'mirror_catalog.py',
]:
    copy(ART / name, 'checks/' + name)
copy(ART / 'offhost_verification.json', 'offhost_verification.json')
copy(ART / 'publication/mirror_catalog_local_001.json', 'checks/mirror_catalog_local_001.json')
copy(ART / 'publication/portability_correction_002.json', 'checks/portability_correction_002.json')
copy(ART / 'publication/portability_correction_receipt_002.json', 'checks/portability_correction_receipt_002.json')
copy(Path('/tmp/m12_capture.py'), 'sources/capture_command.py')
copy(Path(__file__), 'sources/publish_validation.py')

record = dict(
    schema='walker-k1-2-publication-validation-v1',
    prelaunch_commit='a77aa8831870674867d9a8af5564867bc742ec40',
    scientific_result_commit='daf2345782427f765b88b3d1ad1c5e335420895d',
    clean_export_commit='1983140a3c3d99b918dae533fa5d2d42019044b4',
    tests=dict(passed=814, skipped=1, xfailed=1),
    clean_export_commands=31,
    capsule_files_verified=3148,
    historical_files_verified=162,
    maintained_links_verified_in_export=985,
    numeric_report_fields_checked=31,
    independent_scientific_audit_checks=380112,
    scientific_worker_attempts=1,
    scientific_reruns_after_closure=0,
    publication_manifest_sha256='68e03441b3bec7d51331a1544a028ef1d0a5c232eb78109e200d837abf0d54cf',
    earlier_failed_export='027 passed814 tests then rejected a declared full journal ignored by Git. Publication-only v2 removes that file from portable bindings and explicitly references external evidence; original manifest/README and every scientific payload remain preserved.',
    earlier_notes_check='All intended writes completed before an incorrect final scratch-marker assertion. Read-only comparisons prove intended edits only; original script, failure explanation and successful verification retained.',
    late_metered_command_wall_seconds_sum=sum(c['wall_seconds'] for c in commands),
    sum_scope='Commands025 onward through this snapshot only, including failures. Earlier measured commands are in the immutable capsule. Enclosing wall times include their subprocesses: do not add internal timings again or interpret sum as total investigation elapsed time. Uninstrumented editing, synthetic-test scientific work, wire bytes, billing and energy remain unknown.',
    new_offhost_scientific_execution=0,
    full_append_only_engineering_location=str(ART / 'engineering'),
    later_overhead='Closes before its own commit/push. Later publication/remote verification/CI receipts remain append-only outside Git at the explicit engineering location; no frozen capsule or scientific result is rewritten.',
    files=files,
)
with (DEST / 'receipt.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(files=len(files), late_wall_seconds=record['late_metered_command_wall_seconds_sum'])))
