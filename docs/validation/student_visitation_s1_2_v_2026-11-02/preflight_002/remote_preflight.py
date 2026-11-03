import hashlib,importlib.metadata,json,os,pathlib,platform,resource,shutil,signal,sys,time
signal.alarm(40)
resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
start=time.monotonic()
result={'schema':'walker-s1.2-v-remote-preflight-v1','passed':False,'read_bytes':0,'checks':[],'scientific_work':0,'remote_writes':False}
try:
 result['runtime']={'python':platform.python_version(),'platform':sys.platform,'machine':platform.machine(),'packages':{name:importlib.metadata.version(name) for name in ['torch','mujoco','numpy','scipy']},'available_cpus':sorted(os.sched_getaffinity(0))}
 assert result['runtime']['python']=='3.11.16','Python runtime differs'
 for name,expected in {'torch':'2.10.0+cpu','mujoco':'3.6.0','numpy':'2.4.3','scipy':'1.17.1'}.items():assert result['runtime']['packages'][name]==expected,'Runtime package differs: '+name
 assert set(range(16))<=set(result['runtime']['available_cpus']),'Required CPU affinity unavailable'
 for binding in [{'name': 'parent_0', 'path': '/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_consequence_c1_001/prepared/parents/s0_parent.pt', 'sha256': 'f76ff22c816f0554517355721b2ad1c92e89cf3e58b5751eb0b79211a22594c2'}, {'name': 'parent_1', 'path': '/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_consequence_c1_001/prepared/parents/s1_parent.pt', 'sha256': 'dfff0bf7b76f775bebef892a0c058df65faec6d68078d2e7039886b80e18dfa5'}, {'name': 'parent_2', 'path': '/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_consequence_c1_001/prepared/parents/s2_parent.pt', 'sha256': 'c7c42a8365d936df18eed06f15572ebcc420b6688ad3a53ad1b863d161a9a55f'}, {'name': 'reference', 'path': '/home/jaisel/mppi-gps-workspace/runs/2026-10-31_student_source_s1_1_001/inputs/s1/inputs/reference.pt', 'sha256': '174e988618a478d41b846cf428a63b91fef5a0a0f94ae0b36637708856ef363f'}, {'name': 'normalization', 'path': '/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_consequence_c1_001/prepared/inputs/normalization.npz', 'sha256': '652fb44b1796ecf685c631ed86d1c9c9909580fffed29bbb0d339f140ae4ae73'}]:
  p=pathlib.Path(binding['path']);assert p.is_absolute() and p.is_file() and not p.is_symlink(),'Missing/nonordinary input: '+binding['name']
  before=p.stat();assert before.st_size<=32*1024**2-result['read_bytes'],'Input read reservation exceeded'
  data=p.read_bytes();result['read_bytes']+=len(data);actual=hashlib.sha256(data).hexdigest();after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Input changed while read'
  row=dict(binding,bytes=len(data),actual_sha256=actual);result['checks'].append(row);assert actual==binding['sha256'],'Chosen input hash differs: '+binding['name']
 disk=shutil.disk_usage('/home/jaisel/mppi-gps-workspace');result['free_disk_bytes']=disk.free;assert disk.free>=16*1024**3,'Insufficient study output capacity'
 result['passed']=True
except BaseException as error:result['error']={'type':type(error).__name__,'message':str(error)}
result['wall_seconds']=time.monotonic()-start;result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
print(json.dumps(result,allow_nan=False));sys.exit(0 if result['passed'] else 1)
