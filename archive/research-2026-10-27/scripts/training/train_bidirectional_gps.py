"""Run a fresh, source-bound stage; historical runs are never modified."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

_ENTRY_TIME=time.time()
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

from src.gps.bidirectional.workflow import (freeze_source, load_config, write_json,
    qualify, acquire, develop, report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=("validate","qualify","acquire","develop","compare","report"))
    parser.add_argument("--config",type=Path)
    parser.add_argument("--out",type=Path,required=True,help="New directory; report reads an existing directory")
    parser.add_argument("--corpus",type=Path)
    parser.add_argument("--qualification",type=Path)
    parser.add_argument("--backend-qualification",type=Path,
        help="Passed, task/source-bound CPU/Warp comparison required for Warp physics")
    parser.add_argument("--resume",type=Path,help="Trusted local round-commit checkpoint; resumed work uses a NEW output directory")
    parser.add_argument("--internal-frozen",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.stage=="report":
        result=report(args.out)
        from src.gps.bidirectional.reporting import render_report
        render_report(args.out,result)
        print(json.dumps({k:v for k,v in result.items() if k!="events"},indent=2));return
    for key in ("out","config","corpus","qualification","backend_qualification","resume"):
        value=getattr(args,key)
        if value is not None:setattr(args,key,value.resolve())
    config=load_config(args.config)
    if config["backend"]=="warp" and args.stage!="validate":
        if args.backend_qualification is None:parser.error("Warp physics requires --backend-qualification")
        from src.gps.bidirectional.task import TaskConfig
        from src.gps.bidirectional.evidence import file_sha256
        import src.gps.bidirectional.backend as backend
        evidence=json.loads(args.backend_qualification.read_text())
        if (not evidence.get("qualified") or evidence.get("task_hash")!=TaskConfig.from_dict(config["task"]).task_hash
                or evidence.get("source_sha256")!=file_sha256(backend.__file__)):
            parser.error("Backend qualification failed or does not bind the current task/backend source")
        import importlib.metadata
        runtime=json.loads((args.backend_qualification.parent/"manifest.json").read_text())
        for package in ("mujoco","mujoco-warp","warp-lang","numpy"):
            if runtime.get("dependency_versions",{}).get(package)!=importlib.metadata.version(package):
                parser.error("Backend qualification uses a different physics/runtime dependency: "+package)
        if runtime.get("cuda_device")!=torch.cuda.get_device_name():
            parser.error("Backend qualification uses a different GPU model")
    if args.stage in ("develop","compare") and args.corpus is None:
        parser.error("--corpus must identify complete independent acquisition evidence")
    if args.stage=="compare" and args.qualification is None:
        parser.error("Frozen comparison requires a passed --qualification")
    if args.stage=="compare" and (config["schema"]!="bidirectional-acrobot-v0"
            or len(config["seeds"])!=3 or config["development"]["rounds"]!=2
            or config["evaluation"]["roots"]!=64 or config["evaluation"]["capture_min"]!=.9):
        parser.error("The declared frozen comparison requires the full task, three seeds, two rounds and 64-root 90% actor criterion")
    if args.internal_frozen:
        from src.gps.bidirectional.evidence import file_sha256
        if not Path(__file__).resolve().is_relative_to(args.out/"source"):
            parser.error("Internal execution must use the run's frozen source tree")
        manifest=json.loads((args.out/"manifest.json").read_text())
        if file_sha256(args.out/"config.json")!=manifest["config_sha256"] or any(
                file_sha256(args.out/r["file"])!=r["sha256"] for r in manifest["files"]):
            parser.error("Frozen source/configuration was modified")
    else:
        freeze_source(args.out,config,args.stage)
    inputs={}
    from src.gps.bidirectional.evidence import file_sha256
    for name in ("qualification","backend_qualification","resume"):
        value=getattr(args,name)
        if value is not None:
            inputs[name]=dict(file=str(value.resolve()),sha256=file_sha256(value),
                parent_run=str(value.resolve().parents[1] if name=="resume" else value.resolve().parent))
            if name in ("qualification","backend_qualification"):
                parent_manifest=value.parent/"manifest.json"
                inputs[name]["parent_manifest_sha256"]=file_sha256(parent_manifest)
                inputs[name]["parent_config_sha256"]=file_sha256(value.parent/"config.json")
    if args.qualification is not None:
        from src.gps.bidirectional.task import TaskConfig
        parent_config=load_config(args.qualification.parent/"config.json")
        if TaskConfig.from_dict(parent_config["task"]).task_hash!=TaskConfig.from_dict(config["task"]).task_hash:
            parser.error("Task qualification does not match this task/model")
        if any(parent_config["search"][k]!=v for k,v in config["search"].items() if k!="H"):
            parser.error("Task qualification does not match this independent proposal process")
    if args.corpus is not None:
        inputs["corpus"]=dict(parent_run=str(args.corpus.resolve()),
            manifest_sha256=file_sha256(args.corpus/"manifest.json"),
            ledgers=[dict(file=str(p.relative_to(args.corpus)),sha256=file_sha256(p))
                for p in sorted(args.corpus.rglob("ledger.jsonl"))])
    if args.internal_frozen and inputs!=json.loads((args.out/"input_provenance.json").read_text()):
        parser.error("External input evidence changed across source freezing and execution")
    if not args.internal_frozen:
        write_json(args.out/"input_provenance.json",inputs)
        write_json(args.out/"launch.json",dict(entry_epoch_seconds=_ENTRY_TIME,
            source_preparation_wall_seconds=time.time()-_ENTRY_TIME))
        command=[sys.executable,"-m","scripts.training.train_bidirectional_gps",
            "--internal-frozen","--stage",args.stage,"--config",str(args.out/"config.json"),"--out",str(args.out)]
        for key in ("corpus","qualification","backend_qualification","resume"):
            value=getattr(args,key)
            if value is not None:command.extend(["--"+key.replace("_","-"),str(value)])
        os.chdir(args.out/"source")
        os.execv(sys.executable,command)
    started=time.perf_counter()
    try:
        qualification=json.loads(args.qualification.read_text()) if args.qualification else None
        if args.stage=="validate":
            from src.gps.bidirectional.validation import validate
            result=validate(config,args.out)
        elif args.stage=="qualify":result=qualify(config,args.out,qualification)
        elif args.stage=="acquire":result=acquire(config,args.out,qualification)
        else:result=develop(config,args.out,args.corpus,qualification,args.resume)
        write_json(args.out/"stage_complete.json",dict(stage=args.stage,
            wall_seconds=time.perf_counter()-started,
            launch_to_end_wall_seconds=time.time()-json.loads((args.out/"launch.json").read_text())["entry_epoch_seconds"],
            result_file={"develop":"development.json","compare":"development.json",
                "qualify":"qualification.json","acquire":"acquisition.json"}.get(args.stage,args.stage+".json"),
            scientific_pass=bool(result.get("qualified",False) if args.stage in ("qualify","acquire") else
                result.get("standalone_pass",False) and not result.get("mechanism_only",True)
                and result.get("task_qualification_passed",False))))
        print(json.dumps(result,indent=2,default=lambda x:x.tolist()))
    except BaseException as error:
        write_json(args.out/"stage_failed.json",dict(stage=args.stage,wall_seconds=time.perf_counter()-started,
            error_type=type(error).__name__,error=str(error)))
        raise


if __name__=="__main__":main()
