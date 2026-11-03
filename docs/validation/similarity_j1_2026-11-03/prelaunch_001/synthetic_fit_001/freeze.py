from pathlib import Path
import hashlib,json,shutil
base=Path(__file__).parent;root=Path.cwd();target=base/'source_snapshot';target.mkdir(exist_ok=False)
files={}
for source in sorted((root/'src').rglob('*')):
 if not source.is_file() or '__pycache__' in source.parts:continue
 relative=source.relative_to(root);destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
 data=source.read_bytes();destination.write_bytes(data);files[str(relative)]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
source_bytes=sum(row['bytes'] for row in files.values());assert source_bytes<8*1024**2
(base/'source_manifest.json').write_text(json.dumps(dict(schema='walker-j1-synthetic-fit-source-v1',files=files,bytes=source_bytes),indent=2)+'\n')
recipe=dict(schema='walker-j1-synthetic-fit-recipe-v1',scope='One engineering integration ofthe unchanged2000-update wrapper on synthetic labels; zeroWalker learning cases',policy_seed=2026129960,fit_seed=2026129961,data_seed=2026129950,global_torch_seed=2026129960,observations=1000,observation_dimensions=17,action_dimensions=6,normalization='fixedidentitysynthetic',architecture='17-128tanh-128tanh-6tanh',targets='tanh(xW^T+b); PCG64 normal W sd.02; x sd.2; b linspace(-.03,.03,6)',updates=2000,batch_size=256,forward_rows=512000,backward_rows=512000,adam_updates=2000,native_controls=0,model_controls=0,wall_seconds=180,rss_bytes=2*1024**3,output_bytes=64*1024**2,meter_evidence_bytes=64*1024**2-source_bytes-1024**2,source_manifest_sha256=hashlib.sha256((base/'source_manifest.json').read_bytes()).hexdigest(),worker_sha256=hashlib.sha256((base/'worker.py').read_bytes()).hexdigest(),stopping='Anyfailure stops and retains paidwork. No secondfit or automaticretry.',success='All2000update/tape/PCG/Adam/parent/globalRNG checks pass; finite changed child and exact counters; no simulation module import.')
(base/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n');print(json.dumps(dict(files=len(files),source_bytes=source_bytes,worker_sha256=recipe['worker_sha256'])))
