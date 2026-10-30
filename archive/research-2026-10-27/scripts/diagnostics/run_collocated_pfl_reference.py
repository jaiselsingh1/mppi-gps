"""Freeze/re-execute the separate bounded native collocated-PFL reference.

No old guessed state or control arrays initialize this controller. The closed
reference002 packet binds native task/source/runtime identity and paid ancestry.
--freeze-only performs hashing/copying, never engine construction or a CARE.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
from src.gps.bidirectional.pfl_reference import (PFLConfig,CRITICAL,PARENT_AUDIT,PARENT_RESULT,PARENT_SOURCE,
    canonical,file_sha,identity,parent_provenance,protocol,require)

ROOT=Path(__file__).resolve().parents[2]
PACKAGES=("mujoco","numpy","scipy","mujoco-warp","warp-lang","jaxtyping")
FILES=("src/__init__.py","src/envs/__init__.py","src/gps/__init__.py","scripts/__init__.py","scripts/diagnostics/__init__.py",
    "src/envs/acrobot.py","src/envs/base.py","src/envs/mujoco_env.py",
    "src/gps/bidirectional/__init__.py","src/gps/bidirectional/backend.py","src/gps/bidirectional/task.py",
    "src/gps/bidirectional/evidence.py","src/gps/bidirectional/hetero_executor.py",
    "src/gps/bidirectional/shooting_reference.py","src/gps/bidirectional/affine_reference.py",
    "src/gps/bidirectional/reference.py","src/gps/bidirectional/pfl_reference.py",
    "scripts/diagnostics/run_collocated_pfl_reference.py","tests/test_bidirectional_pfl_reference.py",
    "docs/bidirectional_native_pfl_protocol_v0.md","configs/bidirectional_native_pfl_v0.json",
    "assets/acrobot.xml","assets/common/visual.xml","assets/common/skybox.xml","assets/common/materials.xml")


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n");stream.flush();os.fsync(stream.fileno())


def versions():return {name:importlib.metadata.version(name) for name in PACKAGES}


def copy_exact(source,destination,sha,size):
    source,destination=Path(source),Path(destination)
    require(source.is_file() and not source.is_symlink() and file_sha(source)==sha and source.stat().st_size==size,"Source input differs before copy")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with source.open("rb") as original,destination.open("xb") as target:
        shutil.copyfileobj(original,target);target.flush();os.fsync(target.fileno())
    require(file_sha(destination)==sha and destination.stat().st_size==size,"Copied bytes differ")


def safe_relative(name,first=None):
    path=Path(name)
    require(not path.is_absolute() and ".." not in path.parts and bool(path.parts) and (first is None or path.parts[0]==first),"Unsafe source/input relative path")
    return path


def freeze(out,config,parent,*,entry_epoch):
    out,parent=Path(out).resolve(),Path(parent).resolve();provenance=parent_provenance(parent)
    # Hash validation above includes the old audited guesses; never copy or load
    # their numeric arrays. Their actions/states are not PFL initialization.
    excluded=sorted(name for name in provenance["files"] if name.endswith(".npz"))
    provenance["files"]={name:row for name,row in provenance["files"].items() if name not in excluded}
    provenance["excluded_old_numeric_guess_files"]=excluded
    provenance["schema"]="native-pfl-parent-provenance-v1"
    provenance["data_scope"]="Only metadata/source identities; no prior state/control arrays loaded or copied, no teacher actions, no reference initialization"
    provenance["all_cost_scope"]="New57,600-control maximum plus24zero-timeparityforwards/24mass expansions is additional; paid parent and other historical studies remain ancestry, not added twice to these query totals"
    dependencies=versions()
    require(dependencies==provenance["parent_dependency_versions"],"Native runtime versions differ from exact reference002")
    require(not parent.is_relative_to(out) and not out.is_relative_to(parent),"Keep immutable parent and new run separate")
    for name in FILES:require((ROOT/name).is_file() and not (ROOT/name).is_symlink(),"Missing regular source: "+name)
    out.mkdir(parents=True,exist_ok=False);write_json(out/"config.json",asdict(config));write_json(out/"protocol.json",protocol(config))
    copies=[]
    for name,row in provenance["files"].items():
        relative=safe_relative(name);target=Path("parent_inputs")/relative
        copy_exact(parent/relative,out/target,row["sha256"],row["bytes"])
        copies.append(dict(file=str(target),parent_relative=str(relative),sha256=row["sha256"],bytes=row["bytes"]))
    provenance["frozen_copies"]=copies;write_json(out/"input_provenance.json",provenance)
    inventory=[]
    for name in FILES:
        source=ROOT/name;sha,size=file_sha(source),source.stat().st_size
        copy_exact(source,out/"source"/name,sha,size);inventory.append(dict(file="source/"+name,sha256=sha,bytes=size))
    for name,sha in provenance["critical_source_sha256"].items():require(file_sha(out/"source"/name)==sha,"Native source changed during freeze")
    manifest=dict(schema="native-collocated-pfl-source-v1",files=inventory,dependency_versions=dependencies,
        config_sha256=file_sha(out/"config.json"),protocol_sha256=file_sha(out/"protocol.json"),
        input_provenance_sha256=file_sha(out/"input_provenance.json"),python=sys.version,
        engine="Native CPU MuJoCo RK4 .01/frame_skip1; unchanged task/source; explicit zero-time parity before CARE",
        new_physics_controls=0,actor_inference_rows=0,adam_steps=0)
    manifest["source_fingerprint"]=identity(dict(files=inventory,config_sha256=manifest["config_sha256"],protocol_sha256=manifest["protocol_sha256"],dependency_versions=dependencies))
    write_json(out/"manifest.json",manifest)
    write_json(out/"launch.json",dict(entry_epoch_seconds=entry_epoch,source_manifest_sha256=file_sha(out/"manifest.json"),
        input_provenance_sha256=manifest["input_provenance_sha256"],source_fingerprint=manifest["source_fingerprint"],
        original_source_root=str(ROOT),frozen_source_root=str(out/"source"),original_argv=sys.argv,frozen_reexecution_required=True,
        parent_source_fingerprint=PARENT_SOURCE,numerical_thread_environment={name:os.environ.get(name) for name in
            ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS")}))
    return manifest


def verify_frozen(out):
    out=Path(out).resolve();manifest=json.loads((out/"manifest.json").read_text());launch=json.loads((out/"launch.json").read_text())
    provenance=json.loads((out/"input_provenance.json").read_text())
    require(file_sha(out/"manifest.json")==launch["source_manifest_sha256"],"Frozen manifest changed")
    require(versions()==manifest["dependency_versions"]==provenance["parent_dependency_versions"],"Frozen package versions changed")
    require(file_sha(out/"input_provenance.json")==manifest["input_provenance_sha256"]==launch["input_provenance_sha256"],"Frozen ancestry changed")
    seen=set()
    for row in manifest["files"]:
        relative=safe_relative(row["file"],"source");require(str(relative) not in seen,"Repeated source row");seen.add(str(relative));path=out/relative
        require(path.is_file() and not path.is_symlink() and file_sha(path)==row["sha256"] and path.stat().st_size==row["bytes"],"Frozen source changed: "+str(relative))
    require(seen=={"source/"+name for name in FILES},"Source roster differs")
    require(file_sha(out/"config.json")==manifest["config_sha256"] and file_sha(out/"protocol.json")==manifest["protocol_sha256"],"Config/protocol changed")
    require(json.loads((out/"config.json").read_text())==json.loads(canonical(asdict(PFLConfig())))
        and json.loads((out/"protocol.json").read_text())==json.loads(canonical(protocol())),"Prospective fixed PFL recipe changed")
    computed=identity(dict(files=manifest["files"],config_sha256=manifest["config_sha256"],protocol_sha256=manifest["protocol_sha256"],dependency_versions=manifest["dependency_versions"]))
    require(computed==manifest["source_fingerprint"]==launch["source_fingerprint"],"Frozen source fingerprint differs")
    seen=set()
    for row in provenance["frozen_copies"]:
        relative=safe_relative(row["file"],"parent_inputs");require(str(relative) not in seen,"Repeated input copy");seen.add(str(relative));path=out/relative
        require(relative.suffix!=".npz" and path.is_file() and not path.is_symlink() and file_sha(path)==row["sha256"] and path.stat().st_size==row["bytes"],"Frozen metadata/source input changed")
    require(seen=={"parent_inputs/"+name for name in provenance["files"]},"Parent metadata/source roster differs")
    packet=out/"parent_inputs"
    require(file_sha(packet/"reference_result.json")==PARENT_RESULT and file_sha(packet/"audit_001.json")==PARENT_AUDIT
        and provenance["parent_source_fingerprint"]==PARENT_SOURCE,"Wrong exact parent identities")
    require(set(provenance["critical_source_sha256"])==set(CRITICAL),"Native ancestor roster differs")
    for name,sha in provenance["critical_source_sha256"].items():require(file_sha(out/"source"/name)==sha==file_sha(packet/"source"/name),"Native ancestor source changed")
    return manifest,launch


def main():
    entry=time.time();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path);parser.add_argument("--parent-run",type=Path);parser.add_argument("--config",type=Path)
    parser.add_argument("--freeze-only",action="store_true");parser.add_argument("--internal-frozen",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.out.resolve();values=asdict(PFLConfig()) if args.config is None else json.loads(args.config.read_text())
    if json.loads(canonical(values))!=json.loads(canonical(asdict(PFLConfig()))):parser.error("Only the fixed prospective collocated-PFL recipe is permitted")
    config=PFLConfig(**values)
    if not args.internal_frozen:
        if args.parent_run is None:parser.error("--parent-run requires the exact closed audited reference002 packet")
        freeze(out,config,args.parent_run,entry_epoch=entry)
        if args.freeze_only:
            print(json.dumps(dict(status="FROZEN_NOT_EXECUTED",out=str(out),new_physics_controls=0,mj_forward_calls=0,mj_fullM_calls=0,care_solves=0,actor_inference_rows=0,adam_steps=0),sort_keys=True));return 0
        environment=dict(os.environ,PYTHONPATH=str(out/"source"),PYTHONDONTWRITEBYTECODE="1");os.chdir(out/"source")
        os.execvpe(sys.executable,[sys.executable,"-m","scripts.diagnostics.run_collocated_pfl_reference","--out",str(out),"--config",str(out/"config.json"),"--internal-frozen"],environment)
    if not Path(__file__).resolve().is_relative_to(out/"source"):parser.error("Actual work must execute from frozen source")
    manifest,launch=verify_frozen(out)
    write_json(out/"stage_started.json",dict(stage="native_collocated_pfl_reference",source_fingerprint=manifest["source_fingerprint"],epoch_seconds=time.time(),
        max_requested_controls=config.max_controls,max_parity_mj_forward_calls=24,max_mj_fullM_calls=24,no_retry=True))
    from src.gps.bidirectional.pfl_reference import run_native_reference
    result=run_native_reference(out,source_fingerprint=manifest["source_fingerprint"],config=config)
    verify_frozen(out)
    write_json(out/"stage_complete.json",dict(stage="native_collocated_pfl_reference",body_wall_seconds=result["reference_wall_seconds"],
        launch_to_end_wall_seconds=time.time()-launch["entry_epoch_seconds"],native_reference_pass=result["native_reference_pass"],
        result_sha256=file_sha(out/"pfl_result.json"),source_unchanged=True,parent_copies_unchanged=True,scientific_scope=result["scientific_scope"],
        actor_inference_rows=0,adam_steps=0,numerical_control_optimization_used=False))
    print(json.dumps({key:result[key] for key in ("native_reference_pass","final_admissible","stopped_at","error","accounting","analytic_and_zero_time_work","reference_wall_seconds")},indent=2))
    return 1 if result["error"] or result["close_errors"] else 0


if __name__=="__main__":raise SystemExit(main())
