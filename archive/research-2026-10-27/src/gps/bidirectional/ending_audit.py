"""Independent raw arithmetic audit for the learned Ending pilot.

Only NumPy and the standard library are imported. No simulator, producer,
policy, learner, Torch, or planner is imported or executed. PASS certifies
closed arithmetic evidence, not successful learning or a complete feedback loop.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import time
import zipfile

import numpy as np


COMPONENTS = ("effort", "greed", "detour", "ending")
TASK_FIELDS = {"gate", "delay", "move", "effort", "greed", "detour", "detour_steps", "penalty", "ending_y"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(actual, expected, message, atol=2e-12):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype.kind in "biuf" and np.isfinite(a).all()
            and np.allclose(a, b, atol=atol, rtol=2e-12), message)


def numeric(value, shape=None):
    a = np.asarray(value)
    require(a.dtype.kind in "biuf" and np.isfinite(a).all()
            and (shape is None or a.shape == shape), "Nonfinite or incorrectly shaped numeric array")
    return a


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, name+" must be an integer >= "+str(minimum))
    return value


def packed(raw, key):
    a = np.asarray(raw[key])
    require(a.shape == () and a.dtype.kind in "US", "Invalid packed JSON "+key)
    return json.loads(str(a))


def utc(row):
    return datetime.fromisoformat(row["utc"])


def validate_task(config):
    require(isinstance(config, dict) and set(config) == TASK_FIELDS, "Task fields differ")
    for key in ("gate", "delay", "detour_steps"):
        integer(config[key], key, int(key == "gate"))
    require(all(type(v) in (int, float) and math.isfinite(v) for v in config.values()), "Invalid task value")
    require(config["move"] > 0 and -1 <= config["ending_y"] <= 1
            and all(config[k] >= 0 for k in ("effort", "greed", "detour", "penalty")), "Invalid task bounds")
    return config["gate"]+config["delay"]+1


def transition_arithmetic(states, actions, config):
    """Elementwise arithmetic reconstruction of recorded boundaries, no rollout."""
    T = validate_task(config)
    # The frozen transition contract converts native neural float32 actions
    # and states to float64 before arithmetic. Reconstruct that conversion.
    s = numeric(states).astype(np.float64,copy=False)
    require(s.ndim >= 1 and s.shape[-1] == 2, "State must end in [clock,y]")
    clock, y = s[..., 0], s[..., 1]
    require(np.all((clock >= 0) & (clock <= T) & (clock == np.floor(clock)))
            and np.all(np.abs(y) <= 1), "State clock or bounds invalid")
    a = numeric(actions).astype(np.float64,copy=False)
    if a.shape == s.shape[:-1]+(1,):
        a = a[..., 0]
    require(a.shape == s.shape[:-1] and np.all(np.abs(a) <= 1), "Action bounds or alignment invalid")
    active, moving = clock < T, clock < config["gate"]
    next_y = np.where(active & moving, np.clip(y+config["move"]*a, -1, 1), y)
    next_t = np.where(active, clock+1, clock)
    good = next_y <= config["ending_y"]
    ending = active & (clock == T-1)
    parts = {
        "effort": np.where(active & moving, config["effort"]*a*a, 0.),
        "greed": np.where(active & moving, .5*config["greed"]*(1-next_y), 0.),
        "detour": np.where(active & (clock >= config["gate"])
                           & (clock < config["gate"]+config["detour_steps"]) & good, config["detour"], 0.),
        "ending": np.where(ending & ~good, config["penalty"], 0.),
    }
    flags = dict(ending_event=ending, terminated=next_t >= T, ending_good=ending & good, good_lane=good)
    return np.stack((next_t, next_y), axis=-1), parts, sum(parts.values()), flags


def observations(states, T):
    s = numeric(states)
    obs = np.zeros(s.shape[:-1]+(8,))
    obs[..., 0], obs[..., 1] = s[..., 1], s[..., 0]/T
    obs[..., 2] = 1-obs[..., 1]
    return obs


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.hashes = {}

    def file(self, path, expected=None):
        p = Path(path)
        p = p.resolve() if p.is_absolute() else (self.root/p).resolve()
        require(p.is_relative_to(self.root) and p.is_file(), "Missing or escaped evidence file: "+str(path))
        value = sha(p)
        require(expected is None or value == expected, "Evidence SHA differs: "+str(path))
        self.hashes[str(p.relative_to(self.root))] = value
        return p

    def json(self, path):
        return json.loads(self.file(path).read_text())

    def npz(self, base, ref):
        require(isinstance(ref, dict) and set(ref) >= {"file", "sha256", "bytes"}, "Missing durable array reference")
        relative = Path(ref["file"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe array reference")
        p = self.file(Path(base)/relative, ref["sha256"])
        require(p.stat().st_size == ref["bytes"], "Array byte count differs")
        with np.load(p, allow_pickle=False) as z:
            result = {key: z[key] for key in z.files}
        require(not any(a.dtype.hasobject for a in result.values()), "Object arrays prohibited")
        return result

    def rows(self, path):
        p = self.file(path)
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        require(len({r["id"] for r in rows}) == len(rows), "Duplicate ledger IDs")
        prefix = "work" if rows and rows[0]["id"].startswith("work-") else "event"
        for i, row in enumerate(rows):
            require(row["id"] == f"{prefix}-{i:09d}" and math.isfinite(row["wall_seconds"])
                    and row["wall_seconds"] >= 0, "Ledger order or clock invalid")
            if i:
                require(utc(rows[i-1]) <= utc(row) and rows[i-1]["wall_seconds"] <= row["wall_seconds"], "Ledger time reversed")
        return rows


class _StorageType:
    def __init__(self, dtype): self.dtype = np.dtype(dtype)


def _numeric_dtype(spec,*args):
    dtype=np.dtype(spec,*args)
    require(dtype.kind in "biuf" and dtype.fields is None and dtype.subdtype is None
        and dtype.itemsize<=8,"Unsupported serialized numeric dtype")
    return dtype


def _numeric_scalar(dtype,data):
    require(isinstance(dtype,np.dtype) and dtype.kind in "biuf" and dtype.fields is None
        and dtype.subdtype is None and dtype.itemsize<=8 and isinstance(data,bytes)
        and len(data)==dtype.itemsize,"Unsupported serialized numeric scalar")
    return np.frombuffer(data,dtype=dtype,count=1)[0].item()


def _latin1_bytes(value,encoding,errors="strict"):
    require(type(value) is str and encoding in ("latin1","latin-1") and errors=="strict",
        "Unsupported serialized byte encoding")
    return value.encode("latin1")


class _NumericArray(np.ndarray):
    """Only numeric, contiguous pickle array state is admitted by the decoder."""
    def __setstate__(self, state):
        require(isinstance(state, tuple) and len(state) == 5, "Unsupported serialized array state")
        version, shape, dtype, fortran, data = state
        require(version == 1 and isinstance(shape, tuple) and len(shape) <= 8
            and all(type(n) is int and n >= 0 for n in shape)
            and isinstance(dtype, np.dtype) and dtype.kind in "biuf"
            and dtype.fields is None and dtype.subdtype is None and dtype.itemsize <= 8
            and type(fortran) is bool and isinstance(data, bytes)
            and len(data) == math.prod(shape)*dtype.itemsize,
            "Unsupported serialized numeric array")
        np.ndarray.__setstate__(self, state)


def _numeric_array_reconstruct(subtype, shape, typecode):
    require(subtype is _NumericArray and shape == (0,) and typecode == b"b",
        "Unsupported serialized array reconstruction")
    return np.ndarray.__new__(_NumericArray, (0,), dtype=np.uint8)


def _tensor(storage, offset, size, stride, requires_grad, hooks, metadata=None):
    require(type(offset) is int and offset >= 0 and len(size) == len(stride), "Invalid serialized tensor layout")
    require(all(type(n) is int and n >= 0 for n in size) and all(type(n) is int and n >= 0 for n in stride), "Invalid tensor shape/stride")
    span = offset+(sum((n-1)*s for n, s in zip(size, stride)) if all(size) else -1)
    require(span < len(storage), "Tensor storage exceeds archive")
    return np.ndarray(tuple(size), dtype=storage.dtype, buffer=storage, offset=offset*storage.dtype.itemsize,
                      strides=tuple(s*storage.dtype.itemsize for s in stride)).copy()


def checkpoint_numpy(path):
    """Restricted Torch ZIP storage decoding, without importing Torch or models."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), "Duplicate checkpoint ZIP names")
        pickle_names = [name for name in names if name.endswith("/data.pkl")]
        require(len(pickle_names) == 1, "Unsupported checkpoint container")
        parent = pickle_names[0][:-len("data.pkl")]
        require(archive.read(parent+"byteorder") == b"little", "Unsupported checkpoint byte order")
        class Reader(pickle.Unpickler):
            def find_class(self, module, name):
                allowed = {("collections", "OrderedDict"): OrderedDict,
                           ("_codecs", "encode"): _latin1_bytes,
                           ("numpy", "dtype"): _numeric_dtype,
                           ("numpy", "ndarray"): _NumericArray,
                           ("numpy._core.multiarray", "_reconstruct"): _numeric_array_reconstruct,
                           ("numpy.core.multiarray", "_reconstruct"): _numeric_array_reconstruct,
                           ("numpy._core.multiarray", "scalar"): _numeric_scalar,
                           ("numpy.core.multiarray", "scalar"): _numeric_scalar,
                           ("torch._utils", "_rebuild_tensor_v2"): _tensor,
                           ("torch", "FloatStorage"): _StorageType("<f4"),
                           ("torch", "DoubleStorage"): _StorageType("<f8"),
                           ("torch", "LongStorage"): _StorageType("<i8"),
                           ("torch", "ByteStorage"): _StorageType("u1")}
                require((module, name) in allowed, "Checkpoint global prohibited: "+module+"."+name)
                return allowed[module, name]

            def persistent_load(self, value):
                require(isinstance(value, tuple) and len(value) == 5 and value[0] == "storage"
                        and isinstance(value[1], _StorageType) and value[3] == "cpu", "Unsupported storage reference")
                key, count = str(value[2]), value[4]
                require(key.isdigit() and type(count) is int and count >= 0, "Invalid storage key/count")
                data = archive.read(parent+"data/"+key)
                require(len(data) == count*value[1].dtype.itemsize, "Storage byte count differs")
                return np.frombuffer(data, dtype=value[1].dtype)
        return Reader(io.BytesIO(archive.read(pickle_names[0]))).load()


def policy_identity(policy):
    require(policy["policy_type"] == "bidirectional-family-gaussian-v1" and policy["format_version"] == 1, "Unsupported actor checkpoint")
    header = dict(version=policy["policy_type"], obs_dim=policy["obs_dim"], hidden_dim=policy["hidden_dim"],
                  num_families=policy["num_families"], log_std_bounds=[-5., 1.])
    h = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    for name, array in sorted(policy["actor"].items()):
        a = np.ascontiguousarray(numeric(array))
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    result = h.hexdigest()
    require(result == policy["policy_hash"], "Checkpoint parameter owner hash differs")
    return result


def controller_identity(owner):
    return hashlib.sha256(json.dumps(dict(policy_hash=owner["actor_hash"], family=owner["family"],
        deterministic=owner["deterministic"], version="bidirectional-family-gaussian-v1"), sort_keys=True).encode()).hexdigest()


def resolutions(rows, prefix, unit, pending):
    requested = {r["id"]: r for r in rows if r["kind"] == prefix+"_requested"}
    resolved = {}
    for r in rows:
        if r["kind"] not in (prefix+"_complete", prefix+"_error"):
            continue
        key = r["request_id"]
        require(key in requested and key not in resolved and utc(requested[key]) <= utc(r), "Invalid "+prefix+" resolution")
        require(r[unit] == requested[key][unit], "Work resolution charge differs")
        resolved[key] = r
        if r["kind"].endswith("_error") or r.get("returned") is False:
            pending.append(prefix+" failed or returned no output: "+key)
    missing = set(requested)-set(resolved)
    pending.extend(prefix+" unresolved: "+key for key in sorted(missing))
    return requested, resolved


def audit(run, *, feature_verification=None):
    started = time.perf_counter(); e = Evidence(run)
    errors, pending, queries, episodes, checkpoint_rows, mc_rows = [], [], [], [], [], []
    completion_rows,matched_children=[],[];precision_assay=None;interpolation_assay=None;qfilter_assay=None;own_occupancy=None;early_loop=None;coherent_assay=None;geometry_assay=None;geometry_feature_verification=None;conditional_assay=None
    counts = {}; result = {}; config = {}; rows = []; work = []
    def check(label, function):
        try: return function()
        except (ValueError, KeyError, TypeError, IndexError, OSError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
            errors.append(label+": "+str(exc)); return None
    try:
        config = e.json("config.json"); T = validate_task(config["task"])
        precision=config.get("schema")=="bidirectional-ending-precision-assay-v0"
        interpolation=config.get("schema")=="bidirectional-ending-interpolation-assay-v0"
        qfilter=config.get("schema")=="bidirectional-ending-qfilter-assay-v0"
        own=config.get("schema")=="bidirectional-ending-own-occupancy-assay-v0"
        early=config.get("schema")=="bidirectional-ending-early-context-loop-v5"
        coherent=config.get("schema")=="bidirectional-ending-coherent-head-assay-v1"
        geometry=config.get("schema")=="bidirectional-ending-state-geometry-assay-v1"
        conditional=config.get("schema") in ("bidirectional-ending-conditional-mean-feedback-v1","bidirectional-ending-conditional-mean-feedback-v2-selective")
        require(feature_verification is None or geometry, "Feature-verification receipt applies only to the geometry assay")
        manifest = e.json("manifest.json"); result = e.json("loop_result.json" if early else "assay_result.json" if precision or interpolation or qfilter or own or coherent or geometry or conditional else "pilot_result.json")
        require(sha(e.file("config.json")) == manifest["config_sha256"] == result["config_sha256"], "Frozen config binding differs")
        require(sha(e.file("manifest.json")) == result["manifest_sha256"], "Manifest result binding differs")
        for ref in manifest["files"]: e.file(ref["file"], ref["sha256"])
        if config.get("selector")=="one_action_then_complete_frozen_actor":
            require({"source/src/gps/bidirectional/first_action_completion.py",
                "source/src/gps/bidirectional/completion_pilot.py"}<=set(e.hashes),"Completion selector/protocol frozen source omitted")
        if precision:
            require({"source/src/gps/bidirectional/learner_assay.py",
                "source/scripts/diagnostics/run_ending_precision_assay.py"}<=set(e.hashes),"Precision assay frozen source omitted")
        if interpolation:
            require({"source/src/gps/bidirectional/interpolation_assay.py",
                "source/scripts/diagnostics/run_ending_interpolation_assay.py"}<=set(e.hashes),"Interpolation assay frozen source omitted")
        if qfilter:
            require({"source/src/gps/bidirectional/qfilter_assay.py",
                "source/scripts/diagnostics/run_ending_qfilter_assay.py"}<=set(e.hashes),"Q-filter assay frozen source omitted")
        if own:
            require({"source/src/gps/bidirectional/own_occupancy_assay.py",
                "source/src/gps/bidirectional/own_gradient.py",
                "source/scripts/diagnostics/run_ending_own_occupancy_assay.py"}<=set(e.hashes),"Own-occupancy critical source omitted")
        if early:
            require({"source/src/gps/bidirectional/early_context_loop.py",
                "source/src/gps/bidirectional/first_action_completion.py",
                "source/scripts/diagnostics/run_ending_early_context_loop.py"}<=set(e.hashes),"Early-loop frozen producer/selector omitted")
        if coherent:
            require({"source/src/gps/bidirectional/coherent_return.py",
                "source/src/gps/bidirectional/coherent_assay.py",
                "source/scripts/diagnostics/run_ending_coherent_assay.py"}<=set(e.hashes),"Coherent frozen parameter search/selector omitted")
        if geometry:
            require({"source/src/gps/bidirectional/geometry_return.py",
                "source/src/gps/bidirectional/geometry_assay.py",
                "source/scripts/diagnostics/run_ending_geometry_assay.py"}<=set(e.hashes),"Geometry frozen feature/parameter search omitted")
        if conditional:
            require({"source/src/gps/bidirectional/conditional_feedback.py",
                "source/src/gps/bidirectional/first_action_completion.py",
                "source/scripts/diagnostics/run_ending_conditional_feedback.py"}<=set(e.hashes),"Conditional mean frozen source omitted")
        require({"source/src/ending/core.py", "source/src/gps/bidirectional/ending_pilot.py",
            "source/src/gps/bidirectional/policy.py", "source/src/gps/bidirectional/learning.py",
            "source/src/gps/bidirectional/search.py", "source/src/gps/bidirectional/trust.py",
            "source/src/gps/bidirectional/evidence.py", "source/src/gps/bidirectional/meter.py",
            "source/scripts/diagnostics/run_bidirectional_tests.py",
            "source/scripts/diagnostics/run_learned_ending_pilot.py"} <= set(e.hashes), "Critical frozen sources omitted")
        require(result["source_unchanged"] is True, "Producer detected source mutation")
        base = Path("evidence/conditional" if conditional else "evidence/loop" if early else "evidence/assay" if precision or interpolation or qfilter or own or coherent or geometry else "evidence/pilot")
        rows = e.rows(base/"ledger.jsonl"); work = e.rows("work_ledger.jsonl")
        require(e.hashes[str(base/"ledger.jsonl")] == result["query_ledger_sha256"]
                and e.hashes["work_ledger.jsonl"] == result["work_ledger_sha256"], "Result ledger bindings differ")
        require(rows[0]["kind"] == "archive_begin" and rows[-1]["kind"] == "archive_close", "Archive not closed")
        require(work[-1]["kind"] == "work_hooks_removed" and any(r["kind"] == "engine_isolation_installed" and r["already_loaded"] == [] for r in work), "Global work/engine isolation not closed")
        require(not any(r["kind"] in ("engine_isolation_rejected", "engine_import_blocked") for r in work), "Engine isolation violation")
        task_hash = hashlib.sha256(canonical(dict(task=config["task"], adapter="ending-obs8-v1",
            source_sha256=e.hashes["source/src/gps/bidirectional/ending_pilot.py"],
            dynamics_sha256=e.hashes["source/src/ending/core.py"])).encode()).hexdigest()
        qreq = {r["query_id"]: r for r in rows if r["kind"] == "query_requested"}
        qdone = {r["query_id"]: r for r in rows if r["kind"] == "query_complete"}
        qerror = {r["query_id"]: r for r in rows if r["kind"] == "query_error"}
        require(len(qreq) == sum(r["kind"] == "query_requested" for r in rows)
                and len(qdone) == sum(r["kind"] == "query_complete" for r in rows)
                and not set(qdone)&set(qerror) and (set(qdone)|set(qerror)) <= set(qreq), "Duplicate or unbound query resolution")
        pending.extend("Unclosed/error query: "+q for q in sorted(set(qreq)-set(qdone)))
        scalar_req, scalar_done = resolutions(work, "scalar_transition", "controls", pending)
        by_query = defaultdict(list)
        for r in scalar_req.values(): by_query[r["query_id"]].append(r)
        require(set(by_query) <= set(qreq), "Scalar work refers to unknown query")
        counts.update(queries=len(qreq), completed_queries=len(qdone), error_queries=len(qerror),
            rejected_queries=sum(r["kind"] == "query_rejected" for r in rows),
            requested_controls=sum(r["requested_controls"] for r in qreq.values()),
            completed_controls=sum(r["completed_controls"] for r in qdone.values()),
            scalar_transition_requests=len(scalar_req),
            requested_scalar_controls=sum(r["controls"] for r in scalar_req.values()),
            completed_scalar_controls=sum(r["controls"] for r in scalar_done.values() if r["kind"].endswith("_complete")))
        live = {}
        def one_query(qid):
            request, complete = qreq[qid], qdone[qid]
            require(utc(request) < utc(complete), "Query resolved before request")
            inputs, raw = e.npz(base, request["inputs"]), e.npz(base, complete["result"])
            require(packed(inputs, "task_json") == config["task"]
                    and packed(inputs, "metadata_json") == request["owner"], "Query configuration/owner differs")
            state, actions = numeric(inputs["initial_state"], (2,)), numeric(inputs["actions"])
            same(raw["actions"], actions, "Result actions differ from requested actions")
            model = request["mode"] == "model"
            require(model or request["mode"] == "live", "Unknown query mode")
            if model:
                require(actions.ndim == 3 and actions.shape[2] == 1, "Model plan shape differs")
                K, H = actions.shape[:2]
                states = numeric(raw["states"], (K, H, 2))
                prestates = np.concatenate((np.broadcast_to(state, (K, 1, 2)), states[:, :-1]), axis=1)
            else:
                K, H = 1, 1; require(actions.shape == (1,), "Live action shape differs")
                states, prestates = numeric(raw["states"], (2,)), state
            require(K > 0 and H > 0 and H <= T-int(state[0]), "Terminal padding/incomplete horizon incorrectly counted")
            expected, parts, costs, flags = transition_arithmetic(prestates, actions, config["task"])
            same(states, expected, "Post-state transition formula differs")
            if model:
                same(raw["stage_costs"], costs, "Model stage costs differ")
                same(raw["totals"], costs.sum(axis=1), "Model total costs differ")
                for k, value in parts.items(): same(raw["component_"+k], value, "Model component differs: "+k)
            else:
                same(raw["costs"], costs, "Live cost differs")
                same(raw["components"], np.stack([parts[k] for k in COMPONENTS]), "Live cost components differ")
            for k, value in flags.items(): same(raw["flag_"+k], value, "Raw transition flag differs: "+k)
            n = K*H
            require(request["requested_controls"] == request["requested_physics_controls"] == complete["completed_controls"] == n
                    and request["requested_physics_frames"] == complete["completed_physics_frames"] == 0, "Query controls/frame accounting differs")
            calls = by_query[qid]
            require(len(calls) == H, "Scalar before-call ledger does not cover query horizon")
            for t, call in enumerate(calls):
                done = scalar_done[call["id"]]
                require(done["kind"] == "scalar_transition_complete" and done["query_id"] == qid
                        and call["controls"] == K and call["scope"] == request["scope"], "Scalar execution owner/count mismatch")
                require(utc(request) <= utc(call) <= utc(done) <= utc(complete), "Scalar work lies outside query timestamps")
                p, a, s, c = (prestates[:, t], actions[:, t], states[:, t], costs[:, t]) if model else (prestates, actions, states, costs)
                same(call["prestates"], p, "Global scalar prestate differs")
                same(call["actions"], a, "Global scalar action differs")
                same(done["states"], s, "Global scalar poststate differs")
                same(done["costs"], c, "Global scalar cost differs")
                for k, value in parts.items(): same(done["cost_components"][k], value[:, t] if model else value, "Global component differs: "+k)
                for k, value in flags.items(): same(done["flags"][k], value[:, t] if model else value, "Global flag differs: "+k)
            summary = dict(query_id=qid, mode=request["mode"], kind=request["query_kind"], scope=request["scope"],
                charged_controls=n, raw_ref=complete["result"], owner=request["owner"])
            queries.append(summary)
            counts["analytic_query_transition_row_checks"]=counts.get("analytic_query_transition_row_checks",0)+n
            if not model: live[qid] = dict(request=request, complete=complete, state=state, action=actions, raw=raw)
        for qid in qdone: check(qid, lambda qid=qid: one_query(qid))
        for prefix, unit in (("neural", "rows"), ("optimizer_update", "updates"), ("autograd_backward", "calls"), ("autograd_grad", "calls")):
            req, done = resolutions(work, prefix, unit, pending)
            counts[prefix+"_requests"] = len(req)
            counts[prefix+"_units"] = sum(r[unit] for r in req.values())
            counts[prefix+"_completed_units"] = sum(r[unit] for r in done.values() if r["kind"].endswith("_complete"))
        for prefix, unit in (("neural", "rows"), ("optimizer", "updates"), ("gradient_backward", "passes"), ("diagnostic_backward", "passes")):
            req, done = resolutions(rows, prefix, unit, pending)
            counts["local_"+prefix+"_units"] = sum(r[unit] for r in req.values())
        require(counts["local_neural_units"] == counts["neural_units"]
                and counts["local_optimizer_units"] == counts["optimizer_update_units"]
                and counts["local_gradient_backward_units"] == counts["autograd_backward_units"]
                and counts["local_diagnostic_backward_units"] == counts["autograd_grad_units"], "Overlapping local/global work observers disagree")
        counts.update(model_controls=sum(q["charged_controls"] for q in queries if q["mode"] == "model"),
                      live_controls=sum(q["charged_controls"] for q in queries if q["mode"] == "live"),
                      new_mujoco_controls=0, new_warp_controls=0, audit_actor_rows=0, audit_physics_controls=0)
        require(counts["requested_controls"] == counts["requested_scalar_controls"] == counts["completed_scalar_controls"]
                and counts["requested_controls"] <= config["max_controls"]
                and counts["neural_units"] <= config["max_actor_rows"]
                and counts["optimizer_update_units"] <= config["max_adam_steps"], "Paid work ledgers/caps differ")
        for key, value in rows[-1]["accounting"].items():
            if key in counts: require(value == counts[key], "Archive close count differs: "+key)
        reported = result["accounting"]
        for key, value in dict(requested_scalar_controls=counts["requested_scalar_controls"], completed_scalar_controls=counts["completed_scalar_controls"],
            reserved_query_controls=counts["requested_controls"], actor_rows=counts["neural_units"], actor_calls=counts["neural_requests"],
            adam_attempts=counts["optimizer_update_units"], adam_completed=counts["optimizer_update_completed_units"],
            diagnostic_backward_attempts=counts["local_diagnostic_backward_units"], gradient_backward_attempts=counts["local_gradient_backward_units"],
            global_autograd_grad_calls=counts["autograd_grad_requests"], global_autograd_backward_calls=counts["autograd_backward_requests"]).items():
            require(reported[key] == value, "Result work summary differs: "+key)
        known = {};saved_checkpoints={}
        fingerprint = hashlib.sha256(canonical(dict(files=sorted((r["file"], r["sha256"]) for r in manifest["files"]),
            dependencies=manifest.get("dependency_versions", {}))).encode()).hexdigest()
        for row in rows:
            if row["kind"] != "checkpoint": continue
            def one_checkpoint(row=row):
                p = e.file(row["file"], row["sha256"]); saved = checkpoint_numpy(p)
                require(saved["source_fingerprint"] == fingerprint, "Checkpoint frozen source fingerprint differs")
                actor_only=interpolation or ((qfilter or own or early or coherent or geometry or conditional) and "actor" in saved)
                policy=saved["actor"] if actor_only else saved["learner"]["policy"]
                owner = policy_identity(policy)
                if not actor_only:
                    require(saved["learner"]["config"] == config["learning"], "Checkpoint learning config differs")
                # Candidate checkpoints are committed after their paid actor
                # qualification. The preceding fit event already declares the
                # proposal hash; require both that earlier version declaration
                # and the later serialized parameter identity, without falsely
                # treating the post-panel checkpoint as pre-panel work.
                versions = [utc(r) for r in rows if r["kind"] == "fit_work"
                            and r.get("proposal_policy_hash") == owner and utc(r) <= utc(row)]
                earliest=[utc(row)]+versions+([known[owner][0]] if owner in known else [])
                known[owner] = (min(earliest), saved, row)
                saved_checkpoints[row["label"]]=saved
                checkpoint_rows.append(dict(label=row["label"], policy_hash=owner, updates=0 if actor_only else saved["learner"]["updates"],
                    file=str(p.relative_to(e.root)), sha256=row["sha256"]))
            check("checkpoint "+row["label"], one_checkpoint)
        outcome_by_file = {r["raw"]["file"]: r for r in rows if r["kind"] == "episode_outcome"}
        decisions_by_episode = defaultdict(list)
        for row in rows:
            if row["kind"] == "decision": decisions_by_episode[row["episode"]].append(row)
        used_live = set()
        behavior = {};actual_boundaries={}
        for path in sorted((e.root/base/"episodes").glob("*.npz")):
            def one_episode(path=path):
                rel = str(path.relative_to(e.root/base)); event = outcome_by_file.get(rel)
                ref = event["raw"] if event is not None else dict(file=rel, sha256=sha(path), bytes=path.stat().st_size)
                raw = e.npz(base, ref); actions = numeric(raw["actions"])
                n = len(actions); states = numeric(raw["states"], (n+1, 2)); costs = numeric(raw["raw_costs"], (n,))
                require(n > 0 and n == T-int(states[0, 0]) and states[-1, 0] == T and bool(raw["complete"]), "Episode is incomplete or padded")
                same(raw["initial_state"], states[0], "Episode initial state differs")
                same(raw["observations"], observations(states, T), "Actor observation alignment differs")
                expected, parts, expected_cost, flags = transition_arithmetic(states[:-1], actions, config["task"])
                same(states[1:], expected, "Episode transitions differ"); same(costs, expected_cost, "Episode return costs differ")
                same(raw["cost_components"], np.stack([parts[k] for k in COMPONENTS], axis=-1), "Episode components differ")
                counts["analytic_episode_transition_row_checks"]=counts.get("analytic_episode_transition_row_checks",0)+n
                require(bool(raw["captured"]) == bool(flags["ending_good"][-1]) and int(np.count_nonzero(flags["ending_event"])) == 1, "Episode ending counted incorrectly")
                qids = raw["query_ids"].tolist(); require(len(qids) == n and len(set(qids)) == n, "Episode query references duplicated/missing")
                require(not used_live.intersection(qids), "One live query reused by different episodes")
                owner = packed(raw, "owner_json"); context = str(raw["context_id"])
                for t, qid in enumerate(qids):
                    q = live[qid]; same(q["state"], states[t], "Episode live prestate differs")
                    same(q["action"], actions[t], "Episode live action differs")
                    same(q["raw"]["states"], states[t+1], "Episode live poststate differs")
                    require(q["request"]["owner"] == owner, "Continuation changed owner/controller")
                    if t: require(utc(live[qids[t-1]]["complete"]) <= utc(q["request"]), "Episode control times overlap/reverse")
                used_live.update(qids)
                if owner["actor_hash"] is not None:
                    require(owner["actor_hash"] in known, "Actor checkpoint owner missing")
                    if owner["semantics"] == "actor_suffix":
                        require(owner["controller_hash"] == controller_identity(owner), "Actor controller identity differs")
                    else:
                        require(owner["semantics"] == "executed_replanning" and owner["deterministic"] is True,
                            "Unknown actor-influenced continuation")
                    require(known[owner["actor_hash"]][0] <= utc(live[qids[0]]["request"]), "Actor rollout predates its owner checkpoint")
                    if owner["deterministic"] is False:
                        numeric(raw["old_log_probs"], (n,)); latent = numeric(raw["pre_tanh"], actions.shape)
                        same(actions, np.tanh(latent), "Recorded stochastic latent does not produce its action", atol=2e-7)
                    else: require(raw["pre_tanh"].size == raw["old_log_probs"].size == 0, "Deterministic episode has stochastic labels")
                else: require(raw["pre_tanh"].size == raw["old_log_probs"].size == 0, "Planner trajectory supplied actor likelihood labels")
                decisions = decisions_by_episode[path.stem]
                if owner["semantics"] == "executed_replanning" and not path.stem.startswith("teacher_"):
                    require(len(decisions) == n and [r["elapsed"] for r in decisions] == states[:-1, 0].tolist(),
                        "Replanning decisions do not cover actual episode boundaries")
                    changed = 0
                    for t, decision in enumerate(decisions):
                        d = e.npz(base, decision["raw"])
                        same(d["executed_action"], actions[t], "Replanning decision/action differs")
                        independent = numeric(d["independent_plan"])
                        require(independent.ndim == 2 and independent.shape[1] == 1
                            and len(independent) <= T-int(states[t, 0]) and np.all(np.abs(independent) <= 1),
                            "Independent decision plan is invalid")
                        # Conditional labels compare against the full RAW-memory winner;
                        # its independent helper checks that winner before accepting this flag.
                        differs = decision["changed"] if conditional else not np.array_equal(actions[t], independent[0])
                        require(decision["changed"] == differs, "Decision influence label differs from executed action")
                        changed += differs
                        actual_boundaries[(path.stem,decision["elapsed"])]=(states[t].copy(),actions[t].copy(),owner)
                        for qid in decision["model_query_ids"]+decision["guided_query_ids"]:
                            require(qid in qdone and qreq[qid]["mode"] == "model"
                                and utc(qdone[qid]) <= utc(live[qids[t]]["request"]),
                                "Replanning verification not completed before actual action")
                    if event is not None:
                        require(event["changed_controls"] == changed, "Episode changed-control summary differs")
                    counts["audited_replanning_controls"] = counts.get("audited_replanning_controls", 0)+n
                    counts["audited_changed_controls"] = counts.get("audited_changed_controls", 0)+changed
                if event is None:
                    pending.append("Saved complete raw episode has no closed Outcome: "+rel)
                else:
                    out = event["outcome"]
                    require(event["owner"] == owner and out["actor_hash"] == owner["actor_hash"]
                            and out["controller_hash"] == owner["controller_hash"] and out["semantics"] == owner["semantics"]
                            and out["deterministic"] == owner["deterministic"] and out["family_id"] == owner["family"]
                            and out["context_id"] == context and out["task_hash"] == task_hash
                            and out["remaining_steps"] == n and out["query_ids"] == qids and out["complete"] is True,
                            "Closed Outcome ownership/continuation differs")
                    same(out["cost"], costs.sum(), "Closed Outcome return differs")
                    require(out["capture"] == bool(raw["captured"]) and utc(live[qids[-1]]["complete"]) <= utc(event), "Closed Outcome ending/time differs")
                ep = dict(episode=path.stem, context_id=context, controls=n, cost=float(costs.sum()), capture=bool(raw["captured"]),
                    actor_hash=owner["actor_hash"], deterministic=owner["deterministic"], semantics=owner["semantics"],
                    closed_outcome=event is not None, query_ids=qids, raw_ref=ref)
                episodes.append(ep)
                if owner["actor_hash"] is not None and owner["deterministic"] is False:
                    behavior[path.stem] = (ep, raw, event)
            check("episode "+path.stem, one_episode)
        require(set(live) == used_live, "Live controls missing from saved complete episode evidence")
        from .ending_completion_audit import behavior_stem,is_mc_fit,validate_completion,validate_child_pair
        completion_requests={row["id"]:row for row in rows if row["kind"]=="completion_requested"}
        completion_decisions=[row for row in rows if row["kind"]=="completion_decision"]
        outer_decisions={tuple(row["guided_query_ids"]):row for row in rows
            if row["kind"]=="decision" and row["guided_query_ids"]}
        used_completion_queries=set();closed_completion_requests=set()
        for event in completion_decisions:
            def one_completion(event=event):
                request=completion_requests[event["request_id"]]
                require(request["id"] not in closed_completion_requests,"Completion request resolved twice")
                if early or conditional:
                    candidates=[d for episode_decisions in decisions_by_episode.values() for d in episode_decisions
                        if d["guided_query_ids"][-len(event["query_ids"]):]==event["query_ids"]]
                    require(len(candidates)==1,"Early completion is not uniquely linked after eligibility queries")
                    actual_decision=candidates[0]
                    decision=dict(actual_decision,guided_query_ids=event["query_ids"])
                else:
                    decision=outer_decisions[tuple(event["query_ids"])]
                initial,action,owner=actual_boundaries[(decision["episode"],decision["elapsed"])]
                require(owner["actor_hash"]==event["actor_hash"] and event["actor_hash"] in known,
                    "Completion actual controller actor/checkpoint differs")
                require(known[event["actor_hash"]][0]<=utc(request)<=utc(event)<=utc(decision),
                    "Completion evidence predates actor or follows actual decision")
                require(not used_completion_queries.intersection(event["query_ids"]),"Completion queries reused by another decision")
                raw=e.npz(base,event["raw"]);independent=e.npz(base,decision["raw"])["independent_plan"]
                if conditional:
                    # The actor pool is hypothetical; mixed RAW-vs-pool execution
                    # is independently checked by conditional_audit.
                    action=raw["candidate_first_actions"][int(raw["selected_index"])]
                bank_qid=decision["model_query_ids"][0]
                bank=e.npz(base,qdone[bank_qid]["result"])["actions"]
                def one_query_reference(qid):
                    require(qid in qreq and qid in qdone and utc(request)<=utc(qreq[qid])<=utc(qdone[qid])<=utc(event),
                        "Completion query outside its request/decision")
                    return dict(request=qreq[qid],inputs=e.npz(base,qreq[qid]["inputs"]),result=e.npz(base,qdone[qid]["result"]))
                checked=validate_completion(raw,event,request,decision,initial,action,bank,independent,
                    one_query_reference,T,task_hash)
                require(str(raw["actor_controller_hash"])==controller_identity(dict(actor_hash=owner["actor_hash"],family=0,deterministic=True)),
                    "Completion deterministic tail controller identity differs")
                completion_rows.append(dict(request_id=request["id"],episode=decision["episode"],elapsed=decision["elapsed"],**checked))
                used_completion_queries.update(event["query_ids"]);closed_completion_requests.add(request["id"])
            check("completion "+event["request_id"],one_completion)
        failed_completion_events=[row for row in rows if row["kind"] in ("completion_error","completion_revocation")]
        for event in failed_completion_events:
            if event.get("raw") is not None:check("partial completion evidence",lambda event=event:e.npz(base,event["raw"]))
            if event.get("request_id") is not None:pending.append("Completion did not produce a complete selected continuation: "+event["request_id"])
        pending.extend("Unclosed completion request: "+key for key in sorted(set(completion_requests)-closed_completion_requests-
            {row.get("request_id") for row in failed_completion_events}))
        counts["audited_completion_decisions"]=len(completion_rows)
        counts["audited_completion_model_controls"]=sum(row["model_controls"] for row in completion_rows)
        partial_refs={qid for row in failed_completion_events for qid in row.get("completed_query_ids",[])}
        pending.extend("Unbound completion model query: "+qid for qid,row in qreq.items()
            if row["query_kind"]=="actor_completion_step" and qid not in used_completion_queries|partial_refs)
        for fit in [r for r in rows if r["kind"] == "fit_work" and is_mc_fit(r["purpose"])]:
            def one_mc(fit=fit):
                prior_phases = [r for r in rows if r["kind"] in ("pilot_phase","assay_phase") and utc(r) < utc(fit)]
                phase = prior_phases[-1]["phase"]
                stem = behavior_stem(fit["purpose"],phase)
                panel = [v for k, v in behavior.items() if k.startswith(stem)]
                require(len(panel) == fit["complete_episodes"] and all(v[0]["actor_hash"] == fit["behavior_policy_hash"]
                    and v[2] is not None and utc(v[2]) < utc(fit) for v in panel), "MC panel not complete earlier own-actor behavior")
                returns, groups, full = [], defaultdict(list), []
                for ep, raw, event in panel:
                    values = np.cumsum(raw["raw_costs"][::-1], dtype=np.float64)[::-1]
                    full.append(values[0])
                    for t, value in enumerate(values):
                        groups[(ep["context_id"], 0, len(values), len(values)-t)].append(len(returns)); returns.append(value)
                values = np.asarray(returns); baseline = np.empty(len(values))
                for indices in groups.values():
                    require(len(indices) >= config["learning"]["minimum_replicates"], "MC leave-one-out replicate coverage absent")
                    baseline[indices] = (values[indices].sum()-values[indices])/(len(indices)-1)
                advantages = values-baseline
                require(fit["controls"] == len(values) and fit["target_owner"] == "behavior_actor", "MC target ownership/count differs")
                for key, expected in dict(mc_return_mean=values.mean(), raw_mc_advantage_std=advantages.std(),
                    raw_mc_advantage_range=[advantages.min(), advantages.max()], episode_full_cost_range=[min(full), max(full)]).items():
                    same(fit[key], expected, "MC summary differs: "+key)
                mc_rows.append(dict(phase=phase,purpose=fit["purpose"],behavior_episode_names=sorted(v[0]["episode"] for v in panel),
                    behavior_policy_hash=fit["behavior_policy_hash"], proposal_policy_hash=fit["proposal_policy_hash"],
                    complete_episodes=len(panel), controls=len(values), mc_return_mean=float(values.mean()),
                    raw_mc_advantage_std=float(advantages.std()), gradient_steps=fit["gradient_steps"],
                    first_minibatch_pg_gradient_norm=fit["first_minibatch_pg_gradient_norm"]))
            check("MC fit "+fit["id"], one_mc)
        child_rounds=sorted({int(match.group(1)) for label in saved_checkpoints
            if (match:=re.fullmatch(r"completion_r(\d+)_child_(raw|guided)",label))})
        for round_id in child_rounds:
            prefix=f"completion_r{round_id}"
            if not all(prefix+suffix in saved_checkpoints for suffix in ("_child_raw","_child_guided","_parent")):
                pending.append("Incomplete matched completion child/parent checkpoint set: "+prefix)
                continue
            def one_child_pair(round_id=round_id):
                prefix=f"completion_r{round_id}"
                left,right,parent=(saved_checkpoints[prefix+suffix] for suffix in ("_child_raw","_child_guided","_parent"))
                fits=[row for row in mc_rows if row.get("purpose") in (prefix+"_matched_child_raw",prefix+"_matched_child_guided")]
                require(len(fits)==2 and fits[0]["behavior_episode_names"]==fits[1]["behavior_episode_names"]
                    and fits[0]["behavior_policy_hash"]==fits[1]["behavior_policy_hash"]==left["parent_actor_hash"]
                    and fits[0]["gradient_steps"]==fits[1]["gradient_steps"],"Matched children did not use identical audited own MC/optimizer work")
                aux=e.npz(base,left["auxiliary_raw"])
                shared=[e.npz(base,ref) for ref in left["shared_teacher_refs"]]
                raw_new=[e.npz(base,ref) for ref in left["new_teacher_refs"]]
                guided_new=[e.npz(base,ref) for ref in right["new_teacher_refs"]]
                for ref in left["shared_teacher_refs"]+left["new_teacher_refs"]+right["new_teacher_refs"]:
                    source=outcome_by_file[ref["file"]]
                    require(source["owner"]["semantics"]=="executed_replanning" and source["outcome"]["complete"],
                        "Matched auxiliary source is not a complete executed teacher")
                require([str(raw["context_id"]) for raw in raw_new]==[str(raw["context_id"]) for raw in guided_new],
                    "Matched new teacher contexts differ")
                require(all(packed(raw,"owner_json")["actor_hash"] is None for raw in raw_new)
                    and all(packed(raw,"owner_json")["actor_hash"]==left["parent_actor_hash"] for raw in guided_new),
                    "Matched raw/guided teacher policy provenance differs")
                checked=validate_child_pair(left,right,parent["learner"],aux,shared,raw_new,guided_new,config["child_auxiliary_rows"])
                matched_children.append(dict(round=round_id,**checked))
            check("matched completion children "+str(round_id),one_child_pair)
        fit_steps = sum(r.get("updates", r.get("gradient_steps", 0)) for r in rows if r["kind"] == "fit_work")
        require(fit_steps == counts["optimizer_update_units"], "Overlapping completed fit summaries disagree with metered steps")
        if precision:
            from .ending_precision_audit import validate_precision_assay
            precision_assay=check("precision assay",lambda:validate_precision_assay(e,base,rows,work,
                result,config,saved_checkpoints,episodes,mc_rows))
            if precision_assay is not None:
                counts["analytic_reused_episode_transition_row_checks"]=precision_assay["analytic_reused_episode_transition_row_checks"]
        if interpolation:
            from .ending_interpolation_audit import validate_interpolation_assay
            interpolation_assay=check("interpolation assay",lambda:validate_interpolation_assay(e,base,rows,work,
                result,config,saved_checkpoints,episodes))
        if qfilter:
            from .ending_qfilter_audit import validate_qfilter_assay
            qfilter_assay=check("Q-filter assay",lambda:validate_qfilter_assay(e,base,rows,work,
                result,config,saved_checkpoints,episodes,qreq,qdone,task_hash))
            if qfilter_assay is not None:
                counts["analytic_reused_episode_transition_row_checks"]=qfilter_assay["analytic_reused_episode_transition_row_checks"]
        if own:
            from .own_occupancy_audit import validate_own_occupancy
            own_occupancy=check("own occupancy",lambda:validate_own_occupancy(e,base,rows,work,
                result,config,saved_checkpoints,episodes,qreq,qdone,task_hash))
            if own_occupancy is not None:
                counts["analytic_reused_episode_transition_row_checks"]=own_occupancy["analytic_reused_episode_transition_row_checks"]
        if early:
            from .early_context_audit import validate_early_context_loop
            early_loop=check("early context loop",lambda:validate_early_context_loop(e,base,rows,work,
                result,config,saved_checkpoints,episodes,qreq,qdone,task_hash))
            if early_loop is not None:
                counts["analytic_reused_episode_transition_row_checks"]=early_loop["analytic_reused_episode_transition_row_checks"]
        if coherent:
            from .coherent_audit import validate_coherent_assay
            coherent_assay=check("coherent head assay",lambda:validate_coherent_assay(e,base,rows,work,
                result,config,saved_checkpoints,episodes,qreq,qdone,task_hash))
            if coherent_assay is not None:
                counts["analytic_reused_episode_transition_row_checks"]=coherent_assay["analytic_reused_episode_transition_row_checks"]
        if geometry:
            from .geometry_audit import validate_geometry_assay
            geometry_assay=check("state geometry assay",lambda:validate_geometry_assay(e,base,rows,work,
                result,config,saved_checkpoints,episodes,qreq,qdone,task_hash))
            if geometry_assay is not None:
                counts["analytic_reused_episode_transition_row_checks"]=geometry_assay["analytic_reused_episode_transition_row_checks"]
            if feature_verification is not None:
                from .geometry_audit import bind_feature_verification
                geometry_feature_verification=check("paid feature verification receipt",lambda:bind_feature_verification(feature_verification,e,result))
                if geometry_feature_verification is not None:
                    pending.extend(geometry_feature_verification["pending"])
        if conditional:
            from .conditional_audit import validate_conditional_initial
            if config["schema"]=="bidirectional-ending-conditional-mean-feedback-v2-selective":
                from .conditional_selective_audit import validate_selective_initial
                conditional_validator=validate_selective_initial
            else:conditional_validator=validate_conditional_initial
            conditional_assay=check("conditional mean initial stage",lambda:conditional_validator(e,base,rows,work,
                result,config,saved_checkpoints,episodes,qreq,qdone,task_hash))
            if conditional_assay is not None:
                counts["analytic_reused_episode_transition_row_checks"]=conditional_assay["analytic_reused_episode_transition_row_checks"]
        if result["stage"] == "execution_error": pending.append("Pilot ended in execution_error; no scientific success can be assigned")
        require(result["passed"] is False or not pending, "Producer claimed success with incomplete evidence")
        if not precision and not interpolation and not qfilter and not own and not early and not coherent and not geometry and not conditional:
            require(result.get("teacher_successes") == sum(ep["capture"] for ep in episodes if ep["episode"].startswith("teacher_")), "Teacher discovery success summary differs")
        for field, prefix in (("bootstrap_capture_rate", "bootstrap_qualification_"), ("final_actor_capture_rate", "final_actor_qualification_")):
            panel = [ep for ep in episodes if ep["episode"].startswith(prefix) and ep["closed_outcome"]]
            if field in result:
                require(len(panel) == config["actor_qualification_roots"], "Actor qualification panel incomplete")
                same(result[field], np.mean([ep["capture"] for ep in panel]), "Actor competence result differs")
    except (ValueError, KeyError, TypeError, IndexError, OSError, zipfile.BadZipFile, pickle.UnpicklingError) as exc:
        errors.append("audit: "+str(exc))
    modes = Counter(q["mode"] for q in queries)
    return dict(schema="independent-ending-arithmetic-v1", run=str(e.root),
        status="FAIL" if errors else ("PENDING" if pending else "PASS"), evidence_arithmetic_valid=not errors,
        scientific_success=result.get("passed", False), producer_stage=result.get("stage"), errors=errors, pending=pending,
        counts=counts, query_modes=dict(modes), queries=queries, episodes=episodes, checkpoints=checkpoint_rows, monte_carlo=mc_rows,
        completions=completion_rows,matched_children=matched_children,
        precision_assay=precision_assay,
        interpolation_assay=interpolation_assay,
        qfilter_assay=qfilter_assay,
        own_occupancy=own_occupancy,early_context_loop=early_loop,
        coherent_assay=coherent_assay,
        geometry_assay=geometry_assay,
        conditional_mean_initial=conditional_assay,
        geometry_feature_verification=geometry_feature_verification,
        input_sha256=e.hashes, auditor_sha256=sha(__file__), audit_wall_seconds=time.perf_counter()-started,
        completion_auditor_sha256=sha(Path(__file__).with_name("ending_completion_audit.py")),
        precision_auditor_sha256=sha(Path(__file__).with_name("ending_precision_audit.py")) if precision_assay is not None or interpolation_assay is not None or qfilter_assay is not None or own_occupancy is not None or early_loop is not None else None,
        interpolation_auditor_sha256=sha(Path(__file__).with_name("ending_interpolation_audit.py")) if interpolation_assay is not None or qfilter_assay is not None or own_occupancy is not None or early_loop is not None else None,
        qfilter_auditor_sha256=sha(Path(__file__).with_name("ending_qfilter_audit.py")) if qfilter_assay is not None else None,
        own_occupancy_auditor_sha256=sha(Path(__file__).with_name("own_occupancy_audit.py")) if own_occupancy is not None else None,
        early_context_auditor_sha256=sha(Path(__file__).with_name("early_context_audit.py")) if early_loop is not None else None,
        coherent_auditor_sha256=sha(Path(__file__).with_name("coherent_audit.py")) if coherent_assay is not None else None,
        geometry_auditor_sha256=sha(Path(__file__).with_name("geometry_audit.py")) if geometry_assay is not None else None,
        conditional_auditor_sha256=sha(Path(__file__).with_name("conditional_audit.py")) if conditional_assay is not None else None,
        conditional_selective_auditor_sha256=sha(Path(__file__).with_name("conditional_selective_audit.py")) if conditional_assay is not None and config.get("schema")=="bidirectional-ending-conditional-mean-feedback-v2-selective" else None,
        work_overlap="Global neural/Adam/autograd and local learner events observe the same work; fit_work is a summary. Counts are not added.",
        limitations=["No policy forward, gradient, optimizer, simulator, MuJoCo, or Warp call was executed by this audit.",
            "Deterministic actor outputs and stochastic log densities are checkpoint/source/chronology bound, not independently recomputed.",
            "Stochastic tanh latent/action alignment, complete own-return targets and leave-one-out summaries are independently checked.",
            "Arithmetic PASS does not establish successful actor learning, transfer, curriculum benefit, sample efficiency or a completed feedback loop."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path); parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-verification", type=Path,
        help="Optional closed, separately paid geometry feature-verification packet; reads receipt only")
    args = parser.parse_args()
    require(not args.out.exists(), "Audit output must be create-only")
    report = audit(args.run, feature_verification=args.feature_verification)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(canonical({k: report[k] for k in ("status", "evidence_arithmetic_valid", "scientific_success", "producer_stage", "counts", "errors", "pending", "audit_wall_seconds")}))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__": main()
