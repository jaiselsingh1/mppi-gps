import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,signal
signal.alarm(10)
run=Path('/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_visitation_s1_2_v_002');art=Path('/home/jaisel/mppi-gps-workspace/artifacts/2026-11-02_student_visitation_s1_2_v_002')
sys.path.insert(0,str(run/'source'))
from src.gps.mppi_gps.experiments.student_visitation import _check_child,_check_coordinator
from src.gps.mppi_gps.provenance import sha
operator=json.loads((art/'worker_operator_receipt.json').read_text());assert operator['passed'] is True
coordinator=_check_coordinator(run,'worker',run/'worker_coordinator_exit.json')
child=_check_child(run,'worker',run/'worker_exit.json',run/'worker_result.json')
worker=json.loads((run/'worker_result.json').read_text());assert worker['error'] is None and worker['status']=='COMPLETED_V_FIXED_ROSTER'
print(json.dumps(dict(passed=True,coordinator=coordinator,child=child,worker_result_sha256=sha(run/'worker_result.json'),source_commit='70075397bfb2966dff97e325c4104915f03c9581',scientific_work=0)))
