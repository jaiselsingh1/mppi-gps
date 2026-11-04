"""Append the completed eight-round saved-data roster; never run a worker."""
from pathlib import Path
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CPU, (20, 870))
started = time.monotonic()


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


cfg = json.loads((BASE / "config.json").read_text())
run = Path(cfg["continuation_run"])
while True:
    paths = [(run / f"worker/streams/s{s}/result.json").resolve() for s in range(3)]
    data = [json.loads(p.read_text()) for p in paths]
    if any(row["status"] == "FAILED_STOPPED" for row in data):
        raise RuntimeError("Existing study stopped; no absent rounds will be inferred")
    if all(row["status"] == "COMPLETED_ALL_ROUNDS" for row in data):
        break
    if time.monotonic() - started > 900:
        raise TimeoutError("Saved final roster not available; initial review remains preserved")
    time.sleep(5)
cfg["initial_review_config"] = binding(BASE / "config.json")
cfg["snapshot_roster"] = [dict(stream=s, snapshot=binding(p), completed_rounds=list(range(1, 9))) for s, p in enumerate(paths)]
cfg["limits"]["root_cpu_seconds"] = 870
cfg["limits"]["cold_start_cpu_seconds"] = 150
cfg["final_roster_note"] = "Same frozen analysis rules; all completed rounds, with latest primary+confirmation fall states. Initial outputs remain unchanged."
final = BASE / "final_snapshot"
final.mkdir(exist_ok=False)
write(final / "config.json", cfg)

source = (BASE / "bank_analysis.py").read_text().replace("(800, 801)", "(120, 121)")
(final / "bank_analysis.py").write_text(source)
contexts = (BASE / "contexts_admission/source_002.py").read_text()
contexts = contexts.replace("(350, 360)", "(15, 16)").replace("cpu_limit_seconds=360", "cpu_limit_seconds=15")
contexts = contexts.replace('for entry in latest["evaluations"]:', 'for entry in latest["evaluations"] + snapshot.get("final_evaluations", []):')
contexts = contexts.replace('entry["kind"] != "child"', 'entry["kind"] not in ("child", "final")')
(final / "contexts.py").write_text(contexts)

spectrum = final / "spectrum_conflict"
spectrum.mkdir()
for name in ("action_spectrum.original.py", "label_conflict.original.py"):
    shutil.copyfile(BASE / "spectrum_conflict" / name, spectrum / name)
adapter = (BASE / "spectrum_conflict/adapter.py").read_text().replace("(355, 355)", "(45, 45)")
adapter = adapter.replace('source=bind(__file__), config=bind(ROOT / "analysis_config.json"),', 'source=bindings[str(bind(__file__))], config=bindings[str(bind(ROOT / "analysis_config.json"))],')
adapter = adapter.replace("rlimit_cpu_seconds=355", "rlimit_cpu_seconds=45").replace("aggregate_subcap_cpu_seconds=360", "aggregate_subcap_cpu_seconds=50")
(spectrum / "adapter.py").write_text(adapter)
scfg = json.loads((BASE / "spectrum_conflict/analysis_config.json").read_text())
scfg["review_config"] = binding(final / "config.json")
scfg["adapter"] = binding(spectrum / "adapter.py")
scfg["archived_sources"] = [binding(spectrum / name) for name in ("action_spectrum.original.py", "label_conflict.original.py")]
scfg["final_packaging_fix"] = "Serialize known source/config Path values as their existing binding dictionaries; formulas and metrics unchanged."
write(spectrum / "analysis_config.json", scfg)
cold = final / "cold_start"
cold.mkdir()
shutil.copyfile(BASE / "cold_start/review.py", cold / "review.py")

commands = [
    ("banks", [sys.executable, "-B", str(final / "bank_analysis.py")], 300),
    ("contexts", [sys.executable, "-B", str(final / "contexts.py"), "--config", str(final / "config.json"), "--out", str(final / "contexts_admission")], 90),
    ("bands", [sys.executable, "-B", str(spectrum / "adapter.py")], 180),
    ("cold", [sys.executable, "-B", str(cold / "review.py"), "--config", str(final / "config.json"), "--cpu-cap", "20"], 90),
]
phases = []
for name, cmd, timeout in commands:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    clock = time.monotonic()
    code, failure = None, None
    try:
        with (final / f"{name}.stdout.log").open("x") as out, (final / f"{name}.stderr.log").open("x") as err:
            code = subprocess.run(cmd, stdout=out, stderr=err, timeout=timeout).returncode
    except BaseException as error:
        failure = repr(error)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    row = dict(phase=name, argv=cmd, exit_code=code, error=failure,
               cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
               wall_seconds=time.monotonic()-clock)
    phases.append(row)
    write(final / f"{name}.external.json", row)
    print(json.dumps(row), flush=True)
    if code != 0:
        break
write(final / "external_receipt.json", dict(
    passed=len(phases)==4 and all(row["exit_code"]==0 for row in phases), phases=phases,
    measured_child_cpu_seconds=sum(row["cpu_seconds"] for row in phases),
    wall_seconds=time.monotonic()-started, physics=0, neural_forwards=0, policy_updates=0,
    source=binding(__file__), final_config=binding(final / "config.json"),
    scope="Saved-only final roster extension; source formulas, native tolerances and running experiment unchanged. CPU charged to root review allocation."
))
sys.exit(any(row["exit_code"] != 0 for row in phases))
