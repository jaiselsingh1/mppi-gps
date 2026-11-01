import json
from pathlib import Path
from src.gps.mppi_gps.audits.motion_diagnosis import MotionAudit, verify_catalog
cfg=json.loads(Path('configs/studies/motion_diagnosis.json').read_text())
metadata={name:json.loads(Path(binding['source']).read_text()) for name,binding in cfg['metadata'].items()}
a=MotionAudit();verify_catalog(a,metadata)
print(json.dumps({'checks':a.checks,'numeric_dataset_attempts':a.work['dataset_attempts'],'source_hash_bytes':a.work['source_hash_bytes']}))
