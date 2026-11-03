from pathlib import Path
import hashlib,importlib.metadata,json,os,platform,shutil,signal,sys,time
signal.alarm(15);s=time.monotonic();paths={'reference':Path('/home/jaisel/mppi-gps-workspace/runs/2026-10-31_student_source_s1_1_001/inputs/s1/inputs/reference.pt'),'normalization':Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_consequence_c1_001/prepared/inputs/normalization.npz')};files={}
for k,p in paths.items():
 data=p.read_bytes();files[k]={'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
print(json.dumps({'python':sys.version,'host':platform.node(),'platform':platform.platform(),'distributions':{n:importlib.metadata.version(n) for n in ['numpy','scipy','mujoco','torch','h5py','matplotlib','pytest']},'affinity':sorted(os.sched_getaffinity(0)),'free_bytes':shutil.disk_usage('/home/jaisel/mppi-gps-workspace').free,'ffmpeg':shutil.which('ffmpeg'),'new_run_exists':Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-03_similarity_j1_001').exists(),'files':files,'file_read_bytes':sum(v['bytes'] for v in files.values()),'wall_seconds':time.monotonic()-s,'new_physics':0,'new_neural':0,'new_fitting':0}))
