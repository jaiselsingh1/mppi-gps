"""Run the frozen bidirectional tests with durable actual-work accounting.

This launcher never runs tests on import. Use a NEW output directory. Engine
imports are prohibited during the test session; cheap scalar Ending transitions
are separately metered and keep their own raw pre/post ledger in test_evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.abc
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import weakref


_ENTRY_EPOCH = time.time()
_BLOCKED_ENGINES = frozenset(("mujoco", "mujoco_warp", "warp"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


class DurableWork:
    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.path.open("x", encoding="utf-8")
        self.sequence = 0
        self.started = time.perf_counter()
        self.current_test = None

    def append(self, kind, **values):
        identifier = f"work-{self.sequence:09d}"
        row = dict(id=identifier,kind=kind,test=self.current_test,
            utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=time.perf_counter()-self.started,**values)
        self.stream.write(json.dumps(row,sort_keys=True,separators=(",", ":"),allow_nan=False)+"\n")
        self.stream.flush(); os.fsync(self.stream.fileno())
        self.sequence += 1
        return identifier

    def close(self):
        if not self.stream.closed:
            self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()


class NoEngines(importlib.abc.MetaPathFinder):
    def __init__(self, work): self.work=work

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".",1)[0] in _BLOCKED_ENGINES:
            self.work.append("engine_import_blocked",module=fullname,
                             new_mujoco_controls=0,new_warp_controls=0)
            raise ImportError(f"Accountable synthetic tests prohibit engine import: {fullname}")
        return None


class WorkPlugin:
    """Global hooks observe existing calls; they never request an extra forward."""
    def __init__(self, work):
        import numpy as np
        import torch
        from src.gps.bidirectional.policy import FamilyGaussianActor
        self.np,self.torch,self.actor_type=np,torch,FamilyGaussianActor
        self.work=work;self.handles=[];self.scalar_originals=[]
        self.pending=weakref.WeakKeyDictionary();self.actor_ids=weakref.WeakKeyDictionary()
        self.actor_sequence=0;self.original_adam=None;self.guard=None
        self.original_autograd={}

    def install(self):
        loaded=[name for name,module in sys.modules.items()
                if module is not None and name.split(".",1)[0] in _BLOCKED_ENGINES]
        if loaded:
            self.work.append("engine_isolation_rejected",already_loaded=sorted(loaded))
            raise RuntimeError("Engine modules already loaded: use a fresh frozen test process")
        self.guard=NoEngines(self.work);sys.meta_path.insert(0,self.guard)
        self.work.append("engine_isolation_installed",blocked_roots=sorted(_BLOCKED_ENGINES),
                         already_loaded=[],scope="Engine modules unavailable, not merely unobserved")
        module_hooks=self.torch.nn.modules.module
        self.handles=[module_hooks.register_module_forward_pre_hook(self._before),
            module_hooks.register_module_forward_hook(self._after,always_call=True)]
        self.original_adam=self.torch.optim.Adam.step
        original=self.original_adam;work=self.work
        def counted_adam(optimizer,*args,**kwargs):
            parameters=[p for group in optimizer.param_groups for p in group["params"]]
            request=work.append("optimizer_update_requested",optimizer="torch.optim.Adam",
                updates=1,parameter_scalars=sum(p.numel() for p in parameters),
                gradient_scalars=sum(p.numel() for p in parameters if p.grad is not None))
            start=time.perf_counter()
            try: result=original(optimizer,*args,**kwargs)
            except BaseException as error:
                work.append("optimizer_update_error",request_id=request,updates=1,
                    partial_update_unknown=True,error_type=type(error).__name__,error=str(error),
                    compute_wall_seconds=time.perf_counter()-start)
                raise
            work.append("optimizer_update_complete",request_id=request,updates=1,
                        compute_wall_seconds=time.perf_counter()-start)
            return result
        self.counted_adam=counted_adam;self.torch.optim.Adam.step=counted_adam
        for name in ("grad","backward"):
            original_function=getattr(self.torch.autograd,name)
            self.original_autograd[name]=original_function
            def counted_autograd(*args,_name=name,_original=original_function,**kwargs):
                supplied=args[0] if args else kwargs.get("outputs" if _name=="grad" else "tensors")
                # Do not consume generators or inspect tensor values: either
                # would alter the actual differentiation request or synchronize
                # work merely for accounting.
                values=[supplied] if self.torch.is_tensor(supplied) else (
                    supplied if isinstance(supplied,(tuple,list)) else None)
                tensors=[] if values is None else [v for v in values if self.torch.is_tensor(v)]
                request=work.append("autograd_"+_name+"_requested",calls=1,
                    output_tensor_count=None if values is None else len(tensors),
                    output_scalars=None if values is None else sum(t.numel() for t in tensors),
                    shape_accounting="opaque iterable not consumed" if values is None else "tensor metadata only")
                start=time.perf_counter()
                try:result=_original(*args,**kwargs)
                except BaseException as error:
                    work.append("autograd_"+_name+"_error",request_id=request,calls=1,
                        partial_work_unknown=True,error_type=type(error).__name__,error=str(error),
                        compute_wall_seconds=time.perf_counter()-start)
                    raise
                work.append("autograd_"+_name+"_complete",request_id=request,calls=1,
                            compute_wall_seconds=time.perf_counter()-start)
                return result
            setattr(self.torch.autograd,name,counted_autograd)
        self.work.append("work_hooks_installed",actor_class="FamilyGaussianActor",
            actor_scope="Top-level nn.Module calls only; child Linear/Tanh modules are not double counted",
            optimizer_scope="Every torch.optim.Adam.step call, including fitting and verification",
            autograd_scope="Separate torch.autograd.grad and torch.autograd.backward API calls; no extra differentiation",
            extra_actor_forwards=0)

    def _before(self,module,args):
        if not isinstance(module,self.actor_type):return
        if not args:
            self.work.append("unmeterable_actor_call_rejected",reason="Global pre-hook cannot see keyword-only observations")
            raise RuntimeError("Actor observations must be positional for before-call test accounting")
        shape=tuple(self.np.shape(args[0]))
        rows=int(self.np.prod(shape[:-1])) if len(shape)>1 else 1
        if module not in self.actor_ids:
            self.actor_ids[module]=f"actor-{self.actor_sequence:06d}";self.actor_sequence+=1
        request=self.work.append("neural_requested",actor_instance=self.actor_ids[module],
            actor_class=type(module).__name__,rows=rows,observation_shape=list(shape),
            device=str(module.action_scale.device))
        self.pending.setdefault(module,[]).append((request,rows,time.perf_counter()))

    def _after(self,module,args,output):
        if not isinstance(module,self.actor_type):return
        stack=self.pending.get(module,[])
        if not stack:return
        request,rows,start=stack.pop()
        self.work.append("neural_complete",request_id=request,rows=rows,
            returned=output is not None,partial_work_unknown=output is None,
            compute_wall_seconds=time.perf_counter()-start)

    def pytest_collection_modifyitems(self,session,config,items):
        classes={id(getattr(item.module,"CountedEnding")):getattr(item.module,"CountedEnding")
                 for item in items if hasattr(item.module,"CountedEnding")}
        for cls in classes.values():
            original=cls.transition;work=self.work;np=self.np
            def transition(instance,states,actions,_original=original):
                shape=np.shape(states)
                controls=int(np.prod(shape[:-1])) if len(shape)>1 else 1
                request=work.append("scalar_transition_requested",controls=controls,
                    scope=instance.ledger.scope,raw_ledger=str(instance.ledger.path),
                    prestates=np.asarray(states).tolist(),actions=np.asarray(actions).tolist())
                start=time.perf_counter()
                try: result=_original(instance,states,actions)
                except BaseException as error:
                    work.append("scalar_transition_error",request_id=request,controls=controls,
                        partial_work_unknown=True,error_type=type(error).__name__,error=str(error))
                    raise
                # The test's own full pre/post ledger is retained and made
                # durable here; this does not execute a second transition.
                instance.ledger.stream.flush();os.fsync(instance.ledger.stream.fileno())
                work.append("scalar_transition_complete",request_id=request,controls=controls,
                    compute_wall_seconds=time.perf_counter()-start)
                return result
            self.scalar_originals.append((cls,original));cls.transition=transition
        self.work.append("frozen_test_collection",tests=[item.nodeid for item in items],
                         scalar_transition_classes=[cls.__name__ for cls in classes.values()])

    def pytest_runtest_logstart(self,nodeid,location):
        self.work.current_test=nodeid;self.work.append("test_requested",nodeid=nodeid)

    def pytest_runtest_logreport(self,report):
        self.work.append("test_phase_complete",nodeid=report.nodeid,phase=report.when,
                         outcome=report.outcome,phase_wall_seconds=report.duration)

    def uninstall(self):
        for handle in self.handles:handle.remove()
        for cls,original in self.scalar_originals:cls.transition=original
        if self.original_adam is not None:self.torch.optim.Adam.step=self.original_adam
        for name,original in self.original_autograd.items():setattr(self.torch.autograd,name,original)
        if self.guard in sys.meta_path:sys.meta_path.remove(self.guard)
        self.work.current_test=None
        self.work.append("work_hooks_removed",extra_actor_forwards=0)


class Tee(io.TextIOBase):
    def __init__(self,original,stream):self.original,self.stream=original,stream
    @property
    def encoding(self):return getattr(self.original,"encoding","utf-8")
    def write(self,value):
        self.original.write(value);self.stream.write(value);self.stream.flush();return len(value)
    def flush(self):self.original.flush();self.stream.flush()
    def isatty(self):return False


def work_summary(path,evidence):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    def select(kind):return [r for r in rows if r["kind"]==kind]
    neural=select("neural_requested");complete=select("neural_complete")
    updates=select("optimizer_update_requested");done=select("optimizer_update_complete")
    scalar=select("scalar_transition_requested");returned=select("scalar_transition_complete")
    closed={r["request_id"] for r in returned+select("scalar_transition_error")}
    raw_ledgers=[]
    for raw in sorted(Path(evidence).rglob("raw_ending_ledger.jsonl")):
        values=[json.loads(line) for line in raw.read_text().splitlines()]
        attempts=[r for r in values if r["kind"]=="transition_attempt"]
        raw_ledgers.append(dict(file=str(raw),sha256=sha256(raw),bytes=raw.stat().st_size,
            attempted_controls=sum(r["controls"] for r in attempts),transition_attempts=len(attempts),
            raw_transition_completions=sum(r["kind"]=="transition_complete" for r in values)))
    isolation=bool(select("engine_isolation_installed")) and not select("engine_isolation_rejected")
    raw_controls=sum(r["attempted_controls"] for r in raw_ledgers)
    requested_scalar=sum(r["controls"] for r in scalar)
    differentiation={}
    for name in ("grad","backward"):
        requested=len(select("autograd_"+name+"_requested"))
        completed=len(select("autograd_"+name+"_complete"));failed=len(select("autograd_"+name+"_error"))
        differentiation.update({"autograd_"+name+"_calls_requested":requested,
            "autograd_"+name+"_calls_completed":completed,"autograd_"+name+"_calls_failed":failed,
            "autograd_"+name+"_calls_inflight":requested-completed-failed})
    return dict(**differentiation,actor_forward_calls_requested=len(neural),actor_forward_rows_requested=sum(r["rows"] for r in neural),
        actor_forward_calls_returned=sum(r["returned"] for r in complete),
        actor_forward_rows_returned=sum(r["rows"] for r in complete if r["returned"]),
        actor_forward_rows_failed_or_unknown=sum(r["rows"] for r in neural)
            -sum(r["rows"] for r in complete if r["returned"]),
        optimizer_updates_requested=len(updates),optimizer_updates_completed=len(done),
        optimizer_updates_failed=len(select("optimizer_update_error")),
        optimizer_updates_inflight=len(updates)-len(done)-len(select("optimizer_update_error")),
        scalar_controls_requested=requested_scalar,
        scalar_controls_completed=sum(r["controls"] for r in returned),
        scalar_controls_failed=sum(r["controls"] for r in select("scalar_transition_error")),
        scalar_controls_inflight=sum(r["controls"] for r in scalar if r["id"] not in closed),
        scalar_raw_controls=raw_controls,scalar_raw_matches_global_requests=raw_controls==requested_scalar,
        scalar_raw_ledgers=raw_ledgers,engine_import_isolation_verified=isolation,
        blocked_engine_import_attempts=len(select("engine_import_blocked")),
        actual_mujoco_controls=0 if isolation else None,actual_warp_controls=0 if isolation else None,
        control_scope="Real scalar Ending transitions are charged separately; injected fake transitions are not MuJoCo controls",
        local_meter_overlap="Per-test neural/optimizer meters observe some same work; never add them to this global hook total")


def select_test_roster(directory,names=None):
    """Restrict a frozen run to explicit regular bidirectional test basenames."""
    directory=Path(directory)
    available={p.name:p for p in directory.glob("test_bidirectional_*.py") if p.is_file() and not p.is_symlink()}
    if names is None:return [available[name] for name in sorted(available)]
    if not names or len(set(names))!=len(names):raise ValueError("Nonempty unique test-file roster required")
    if any(type(name) is not str or Path(name).name!=name or name not in available for name in names):
        raise ValueError("Explicit tests must be frozen bidirectional test basenames")
    return [available[name] for name in sorted(names)]


def run_frozen(out,config):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"]="1"
    os.environ["PYTHONDONTWRITEBYTECODE"]="1"
    evidence=out/"test_evidence"
    if evidence.exists():raise FileExistsError("Test evidence is create-only; pytest must not clear an existing basetemp")
    launch=json.loads((out/"launch.json").read_text())
    roster=select_test_roster(Path("tests"),launch.get("selected_test_files"))
    if not roster:raise RuntimeError("Frozen test roster is empty")
    write_json(out/"test_roster.json",dict(files=[dict(file=str(p),sha256=sha256(p)) for p in roster],
        options=["-q","--basetemp",str(evidence),"-p","no:cacheprovider"],
        pytest_plugin_autoload=False,source_tree=str(Path.cwd())))
    work=DurableWork(out/"work_ledger.jsonl")
    work.append("test_session_requested",tests=len(roster),engine_policy="imports prohibited",
        hardware=platform.platform(),python=platform.python_version(),torch=str(torch.__version__),
        cpu_count=os.cpu_count(),cuda_available=torch.cuda.is_available())
    plugin=WorkPlugin(work);start=time.perf_counter();exit_code=None;failure=None
    stdout_path,stderr_path=out/"pytest_stdout.txt",out/"pytest_stderr.txt"
    oldout,olderr=sys.stdout,sys.stderr
    with stdout_path.open("x",encoding="utf-8") as stdout,stderr_path.open("x",encoding="utf-8") as stderr:
        try:
            plugin.install()
            sys.stdout,sys.stderr=Tee(oldout,stdout),Tee(olderr,stderr)
            import pytest
            exit_code=int(pytest.main([str(p) for p in roster]+["-q","--basetemp",str(evidence),
                "-p","no:cacheprovider"],plugins=[plugin]))
        except BaseException as error:
            failure=dict(error_type=type(error).__name__,error=str(error));exit_code=1
            work.append("test_session_error",**failure)
        finally:
            sys.stdout,sys.stderr=oldout,olderr
            plugin.uninstall()
            work.append("test_session_complete",pytest_exit_code=exit_code,
                        session_wall_seconds=time.perf_counter()-start)
            work.close()
            stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
    accounting=work_summary(work.path,evidence)
    manifest=json.loads((out/"manifest.json").read_text())
    source_unchanged=all(sha256(out/r["file"])==r["sha256"] for r in manifest["files"])
    result=dict(schema="bidirectional-accountable-tests-v1",pytest_exit_code=exit_code,
        passed=exit_code==0 and source_unchanged and accounting["scalar_raw_matches_global_requests"],
        failure=failure,accounting=accounting,source_unchanged=source_unchanged,
        manifest_sha256=sha256(out/"manifest.json"),config_sha256=sha256(out/"config.json"),
        work_ledger_sha256=sha256(work.path),stdout_sha256=sha256(stdout_path),stderr_sha256=sha256(stderr_path),
        wall_seconds=time.perf_counter()-start,
        launch_to_end_wall_seconds=time.time()-json.loads((out/"launch.json").read_text())["entry_epoch_seconds"],
        status_scope="Software/cheap scalar mechanism checks only; no Acrobot/Walker competence or learned-controller scientific success")
    write_json(out/"test_result.json",result)
    print(json.dumps(result,indent=2,allow_nan=False))
    return exit_code if result["passed"] else (exit_code or 1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--test-file",action="append",help="Explicit test_bidirectional_*.py basename; repeat to select only new checks")
    parser.add_argument("--internal-frozen",action="store_true",help=argparse.SUPPRESS)
    args=parser.parse_args();out=args.out.resolve()
    from src.gps.bidirectional.workflow import freeze_source,load_config
    config=load_config(args.config.resolve() if args.config else None)
    if not args.internal_frozen:
        if args.test_file is not None:select_test_roster(Path(__file__).resolve().parents[2]/"tests",args.test_file)
        freeze_source(out,config,"tests")
        frozen=out/"source/scripts/diagnostics/run_bidirectional_tests.py"
        if not frozen.is_file():
            write_json(out/"setup_failure.json",dict(error="freeze_source must copy this diagnostic launcher",
                tests_started=False,actor_forwards=0,optimizer_updates=0,simulator_controls=0))
            raise RuntimeError("Frozen source omits run_bidirectional_tests.py")
        write_json(out/"launch.json",dict(entry_epoch_seconds=_ENTRY_EPOCH,
            source_preparation_wall_seconds=time.time()-_ENTRY_EPOCH,
            launcher_source_sha256=sha256(__file__),selected_test_files=args.test_file))
        os.environ["PYTHONDONTWRITEBYTECODE"]="1"
        command=[sys.executable,"-m","scripts.diagnostics.run_bidirectional_tests",
            "--internal-frozen","--out",str(out),"--config",str(out/"config.json")]
        os.chdir(out/"source");os.execv(sys.executable,command)
    if not Path(__file__).resolve().is_relative_to(out/"source"):
        parser.error("Internal tests must execute the frozen source copy")
    manifest=json.loads((out/"manifest.json").read_text())
    if sha256(out/"config.json")!=manifest["config_sha256"] or any(
            sha256(out/r["file"])!=r["sha256"] for r in manifest["files"]):
        parser.error("Frozen source/configuration changed before tests")
    raise SystemExit(run_frozen(out,config))


if __name__=="__main__":main()
