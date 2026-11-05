import pathlib,json,hashlib,resource,time
resource.setrlimit(resource.RLIMIT_CPU,(10,10));t=time.process_time();root=pathlib.Path('/home/jaisel/mppi-gps-codex-acrobot/external_refs_2026-11-05/collocation_warmstart_v2_0');m=json.load(open(root/'preparation/manifest.json'));auth=json.load(open(root/'authorization.json'));rows=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
for f in m['files']:
 p=root/'source'/f['path'];rows.append({'path':f['path'],'expected':f['sha256'],'actual':sha(p),'size_matches':p.stat().st_size==f['bytes']})
for f in m['prefixes']:
 p=root/'preparation'/f['path'];rows.append({'path':'preparation/'+f['path'],'expected':f['sha256'],'actual':sha(p),'size_matches':True})
ms=sha(root/'preparation/manifest.json');print(json.dumps({'source_commit':auth['source_commit'],'manifest_sha256':ms,'authorized_manifest':auth['approved_manifest_sha256'],'manifest_matches_authorization':ms==auth['approved_manifest_sha256'],'files':rows,'all_hashes_match':all(r['expected']==r['actual'] and r['size_matches'] for r in rows),'prefix1_final_candidate_exists':(root/'solves/prefix_1/candidate.npz').exists(),'prefix1_final_result_exists':(root/'solves/prefix_1/result.json').exists(),'final_route_summary_exists':(root/'solves/result.json').exists(),'native_output_directory_exists':(root/'native').exists(),'new_native_physics_steps':0,'new_optimizer_calls':0,'cpu_seconds':time.process_time()-t},indent=2))
