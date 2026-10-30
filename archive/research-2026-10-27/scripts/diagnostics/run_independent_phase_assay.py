"""Freeze and run two native phase episodes against an audited paid baseline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ENTRY_TIME = time.time()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--backend-qualification", required=True, type=Path)
    parser.add_argument("--baseline-audit", type=Path)
    parser.add_argument("--internal-frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.config = args.config.resolve(); out = args.out.resolve()
    args.baseline_run = args.baseline_run.resolve()
    args.backend_qualification = args.backend_qualification.resolve()
    from src.gps.bidirectional.phase_assay import (
        input_provenance, load_config, protocol, run_assay, snapshot_inputs, verify_inputs, _sha,
    )
    from src.gps.bidirectional.workflow import freeze_source, write_json
    config = load_config(args.config); protocol(config)
    if not args.internal_frozen:
        # No task/engine exists yet. A pending or changed baseline fails here.
        provenance = input_provenance(args.baseline_run, args.backend_qualification,
            baseline_audit=None if args.baseline_audit is None else args.baseline_audit.resolve())
        from src.gps.bidirectional.temporal_assay import check_backend
        check_backend(config, args.backend_qualification)
        freeze_source(out, config, "independent_phase_discovery_assay")
        try:
            script = out/"source/scripts/diagnostics/run_independent_phase_assay.py"
            if not script.is_file():
                raise RuntimeError("Frozen source inventory omits phase launcher")
            snapshot_inputs(provenance, out)
        except BaseException as error:
            write_json(out/"setup_failure.json", dict(error_type=type(error).__name__, error=str(error),
                physics_controls=0, actor_rows=0, optimizer_steps=0))
            raise
        write_json(out/"launch.json", dict(entry_epoch_seconds=ENTRY_TIME,
            source_preparation_seconds=time.time()-ENTRY_TIME, stage="independent_phase_discovery_assay",
            baseline_run=str(args.baseline_run), input_provenance_sha256=_sha(out/"input_provenance.json")))
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; os.chdir(out/"source")
        os.execv(sys.executable, [sys.executable, "-m", "scripts.diagnostics.run_independent_phase_assay",
            "--config", str(out/"config.json"), "--out", str(out), "--baseline-run", str(args.baseline_run),
            "--backend-qualification", str(args.backend_qualification), "--internal-frozen"])
    if not Path(__file__).resolve().is_relative_to(out/"source"):
        parser.error("Internal launcher must be the frozen copy")
    launch = json.loads((out/"launch.json").read_text())
    if _sha(out/"input_provenance.json") != launch["input_provenance_sha256"]:
        raise ValueError("Input provenance changed across source reexecution")
    verify_inputs(out)
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    result = run_assay(config, out)
    write_json(out/"stage_complete.json", dict(stage="independent_phase_discovery_assay", scientific_pass=False,
        result_file="assay_result.json", development_lead=result["development_lead"],
        body_wall_seconds=result["wall_seconds"], launch_to_end_wall_seconds=time.time()-launch["entry_epoch_seconds"]))
    print(json.dumps({k: v for k, v in result.items() if k != "conditions"}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
