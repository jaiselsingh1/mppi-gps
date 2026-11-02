"""Record the explicitly approved backup without changing frozen evidence."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
ART = ROOT/'artifacts/2026-11-01_teacher_proposal_h5_0_preparation_001'
MIRROR = ART/'mirror_001'
OUT = ROOT/'docs/validation/teacher_proposal_h5_0_2026-11-01/backup_001'
OUT.mkdir(exist_ok=False)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def copy(source, name):
    target=OUT/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    return {'bytes':target.stat().st_size,'sha256':sha(target),'source':str(source)}
bindings={}
for name in ('local_catalog.json','remote_catalog.json','receipt.json','stderr.txt'):
    bindings['original/'+name]=copy(MIRROR/name,'original/'+name)
for name in ('receipt.json','stdout.txt','stderr.txt'):
    bindings['command/'+name]=copy(ART/'engineering/027_approved_backup'/name,'command/'+name)
bindings['mirror_preparation.py']=copy(Path('/tmp/h50_mirror_preparation.py'),'mirror_preparation.py')
bindings['publish_backup.py']=copy(Path(__file__),'publish_backup.py')
original=json.loads((MIRROR/'receipt.json').read_text())
local=json.loads((MIRROR/'local_catalog.json').read_text())
remote=json.loads((MIRROR/'remote_catalog.json').read_text())
assert local==remote
assert original['exit_code']==0 and original['exact_catalog_match']
verification={k:v for k,v in original.items() if k!='metadata_backup_only'}
verification.update(
    schema='walker-h5-approved-preparation-backup-v1',
    authorization='User explicitly approved copying the frozen H5 preparation, including W2 inputs and metadata, to robopil5090; research execution remains blocked.',
    preparation_backup_only=True,
    includes_frozen_benchmark_inputs=True,
    raw_receipt_field_correction='Original metadata_backup_only is an imprecise field name. This was a full preparation backup including the sealed W2 NPZ inputs, not metadata alone. Raw receipt and executed helper are preserved unchanged.',
    remote_catalog_sha256=sha(MIRROR/'remote_catalog.json'),
    launch_authorized=False,
    eligibility_decision='pending_explicit_review',
    useful_learning_prerequisite_satisfied=False,
    scientific_work={'network_forwards':0,'physics_controls':0,'optimizer_steps':0,'mppi_queries':0},
    archive_payload_bytes=original['archive_bytes'],
    file_pass_accounting={'local_pretransfer_catalog_hash_bytes':original['uncompressed_bytes'],'local_archive_payload_read_bytes':original['uncompressed_bytes'],'remote_decompressed_file_write_bytes':original['uncompressed_bytes'],'remote_verification_hash_bytes':original['uncompressed_bytes'],'scope':'Named helper passes only. Preflight, independent verification and publication hashing are additional; SSH protocol overhead, peak RAM, editing time and billing are not measured.'},
    cost_scope='One successful transfer attempt within128MiB payload and120s deadline. Enclosing command wall time includes the inner backup timer; do not add them. Previous automatic-review rejection occurred before execution and transferred zero bytes.',
    files=bindings,
)
(OUT/'verified_backup.json').write_text(json.dumps(verification,indent=2)+'\n')
readme='''# Approved H5.0 preparation backup

After explicit user approval, the complete frozen preparation was copied to a **new** directory on `robopil5090`: `/home/jaisel/mppi-gps-workspace/runs/2026-11-01_teacher_proposal_h5_0_preparation_001`.

All **489 files /10,332,583 bytes** match by path, byte size and SHA-256. The transferred compressed archive was **4,031,472 bytes**; the backup took **1.544872 seconds** inside a **1.577764-second** enclosing command. These timers overlap. One transfer attempt completed within the128MiB payload limit and120-second deadline. SSH overhead, peak RAM and billing were not metered.

- [Verified backup and accounting](verified_backup.json).
- [Local file catalog](original/local_catalog.json), [remote file catalog](original/remote_catalog.json), and [unchanged original receipt](original/receipt.json).
- [Exact executed helper](mirror_preparation.py) and [enclosing command receipt](command/receipt.json).

The original receipt's `metadata_backup_only` field is an imprecise name: this was the **full preparation**, including frozen W2 NPZ inputs and source. The verified receipt uses `preparation_backup_only` and `includes_frozen_benchmark_inputs`; it preserves the raw receipt rather than rewriting it.

No inference, physics, fitting, acquisition or scientific audit ran. Both copies retain `launch_authorized=false`, the pending weak-learner ordering exception and the unmet useful-learning prerequisite. Existing remote workspaces, local frozen evidence and the tracked capsule were not modified. The earlier capsule/validation availability fields describe their pre-approval snapshot; this supplemental record documents the later backup. Original inherited H4/K1/W1/W2 bulk evidence was not mirrored by this operation.
'''
(OUT/'README.md').write_text(readme)
index=ROOT/'docs/evidence_index.json'
data=json.loads(index.read_text())
entry=data['studies']['teacher_proposal_h5_0_preparation']
entry.update(
    remote_directory=original['remote_directory'],remote_copy_verified=True,
    availability='Full489-file/10332583-byte preparation is byte-verified locally and onrobopil5090 following explicit backup approval. Compact metadata tracked; NPZ inputs and full473-file source export remain outsideGit. Research execution remains blocked.',
    verified_backup='docs/validation/teacher_proposal_h5_0_2026-11-01/backup_001/verified_backup.json',
    verified_backup_sha256=sha(OUT/'verified_backup.json'),
    original_capsule_availability_scope='Pre-approval publication snapshot, preserved unchanged.'
)
index.write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'backup_record':str(OUT),'files':original['files'],'bytes':original['uncompressed_bytes'],'verified_backup_sha256':sha(OUT/'verified_backup.json')},indent=2))
