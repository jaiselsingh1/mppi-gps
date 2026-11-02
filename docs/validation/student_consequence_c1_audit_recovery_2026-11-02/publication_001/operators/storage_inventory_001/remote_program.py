import pathlib,json,time
base=pathlib.Path('/home/jaisel/mppi-gps-workspace')
names=['runs/2026-11-02_c1_audit_recovery_001','artifacts/2026-11-02_c1_audit_recovery_001','runs/2026-11-02_c1_certificate_closure_a2_001','artifacts/2026-11-02_c1_certificate_closure_a2_001']
result={}
for name in names:
 p=base/name;files=[f for f in p.rglob('*') if f.is_file() and not f.is_symlink()]
 result[name]={'files':len(files),'logical_bytes':sum(f.stat().st_size for f in files)}
print(json.dumps({'scope':'Metadata-only stat inventory; no scientific payload reads. Original run unchanged; original storage retained from its closed census.','trees':result,'new_scientific_work':0,'observed_unix':time.time()}))
