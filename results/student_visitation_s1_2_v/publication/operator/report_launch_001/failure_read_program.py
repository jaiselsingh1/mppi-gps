from pathlib import Path
import json
p=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_student_visitation_s1_2_v_002')
print(json.dumps({n:(p/n).read_text() for n in ['report_stdout.txt','report_exit.json']}))
