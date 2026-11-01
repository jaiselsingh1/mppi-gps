from pathlib import Path
import json, subprocess, sys, time, hashlib

root=Path('/Users/jaisel/Documents/mppi-gps')
folder=root/'artifacts/2026-11-01_trajectory_review_k1_1_001/engineering'/sys.argv[1]
folder.mkdir(parents=True,exist_ok=False)
command=sys.argv[2:]
started=time.perf_counter()
with (folder/'stdout.txt').open('xb') as out,(folder/'stderr.txt').open('xb') as err:
    process=subprocess.run(command,cwd=root,stdout=out,stderr=err)
record={'command':command,'cwd':str(root),'exit_code':process.returncode,'wall_seconds':time.perf_counter()-started,
        'stdout_sha256':hashlib.sha256((folder/'stdout.txt').read_bytes()).hexdigest(),
        'stderr_sha256':hashlib.sha256((folder/'stderr.txt').read_bytes()).hexdigest()}
with (folder/'receipt.json').open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
print(json.dumps(record))
print((folder/'stdout.txt').read_text()[-8000:])
print((folder/'stderr.txt').read_text()[-8000:])
sys.exit(process.returncode)
