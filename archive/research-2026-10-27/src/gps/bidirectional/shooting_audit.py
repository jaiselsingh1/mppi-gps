"""Independent arithmetic/ownership audit of a saved native shooting reference.

No task, producer, engine, optimizer or policy module is imported. This composes
two independent saved-query auditors and checks the reference-specific math.
It performs no dynamics reintegration and never creates missing experience.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def identity(value):
    return digest(canonical(value).encode())


def same(actual, expected, message, *, atol=1e-10, rtol=2e-8):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=atol, rtol=rtol), message)


def unpack(raw, key):
    value = raw[key]
    require(value.shape == (), "Packed metadata is not scalar: "+key)
    return json.loads(str(value))


def distance_squared(states):
    x = np.asarray(states)
    a, b = x[..., 0], x[..., 1]
    return (np.sin(a)+np.sin(a+b))**2+(np.cos(a)+np.cos(a+b)-2.)**2


def distance_gradient(states):
    x = np.asarray(states)
    a, b = x[..., 0], x[..., 1]
    horizontal = np.sin(a)+np.sin(a+b)
    vertical = np.cos(a)+np.cos(a+b)-2.
    out = np.zeros_like(x, dtype=float)
    out[..., 0] = 2*horizontal*(np.cos(a)+np.cos(a+b))-2*vertical*(np.sin(a)+np.sin(a+b))
    out[..., 1] = 2*horizontal*np.cos(a+b)-2*vertical*np.sin(a+b)
    return out


def constraints(states, cfg):
    states = np.asarray(states)
    tail = states[-cfg["capture_steps"]:]
    residual = [np.maximum(distance_squared(tail)/cfg["capture_distance"]**2-1., 0.),
                np.maximum(np.sum(tail[:, 2:]**2, axis=1)/cfg["capture_speed"]**2-1., 0.)]
    if cfg["require_local_at_600_residual"]:
        residual.append(np.maximum([distance_squared(states[cfg["local_by"]])/cfg["local_distance"]**2-1.], 0.))
    return np.concatenate(residual)


def fd_inputs(states, actions, state_fd, action_fd):
    states, actions = np.asarray(states), np.asarray(actions)
    require(states.ndim == 2 and states.shape[1] == 5 and actions.shape == (len(states), 1), "FD shape mismatch")
    require(np.isfinite(states).all() and np.isfinite(actions).all() and np.max(abs(actions)) <= 1., "Invalid native FD inputs")
    eps = np.asarray(state_fd)
    require(eps.shape == (4,) and np.isfinite(eps).all() and np.min(eps) > 0
            and np.isfinite(action_fd) and action_fd > 0, "Invalid FD step")
    ss = np.broadcast_to(states[:, None], (len(states), 10, 5)).copy()
    uu = np.broadcast_to(actions[:, None], (len(states), 10, 1)).copy()
    for j in range(4):
        ss[:, 2*j, j+1] += eps[j]
        ss[:, 2*j+1, j+1] -= eps[j]
    upper, lower = np.minimum(actions[:, 0]+action_fd, 1.), np.maximum(actions[:, 0]-action_fd, -1.)
    uu[:, 8, 0], uu[:, 9, 0] = upper, lower
    return ss.reshape(-1, 5), uu.reshape(-1, 1), upper-lower


def derivatives(outputs, denominator, state_fd):
    out = np.asarray(outputs).reshape(-1, 10, 5)[:, :, 1:]
    denominator = np.asarray(denominator)
    require(denominator.shape == (len(out),) and np.all(denominator > 0), "FD denominator mismatch")
    A = np.empty((len(out), 4, 4))
    for j, step in enumerate(state_fd):
        A[:, :, j] = (out[:, 2*j]-out[:, 2*j+1])/(2*step)
    return A, ((out[:, 8]-out[:, 9])/denominator[:, None])[:, :, None]


def expected_csr(states, A, B, cfg):
    """Construct the sparse residual Jacobian independently, using NumPy only."""
    T, count = cfg["T"], cfg["capture_steps"]
    scale = np.asarray(cfg["dynamics_scale"])
    rows, cols, data = [], [], []
    for t in range(T):
        for j in range(4):
            rows.append(4*t+j); cols.append(4*t+j); data.append(1./scale[j])
            if t > 0:
                for k in range(4):
                    rows.append(4*t+j); cols.append(4*(t-1)+k); data.append(-A[t, j, k]/scale[j])
            rows.append(4*t+j); cols.append(4*T+t); data.append(-B[t, j, 0]/scale[j])
    tail = np.asarray(states)[-count:]
    distances = distance_squared(tail)/cfg["capture_distance"]**2-1.
    speed = np.sum(tail[:, 2:]**2, axis=1)/cfg["capture_speed"]**2-1.
    gradients = distance_gradient(tail)/cfg["capture_distance"]**2
    for k in range(count):
        node = T-count+k
        for j in range(4):
            rows.append(4*T+k); cols.append(4*node+j); data.append(gradients[k, j] if distances[k] > 0 else 0.)
        for j in range(2):
            rows.append(4*T+count+k); cols.append(4*node+2+j)
            data.append(2*tail[k, 2+j]/cfg["capture_speed"]**2 if speed[k] > 0 else 0.)
    nrows = 4*T+2*count
    if cfg["require_local_at_600_residual"]:
        node = cfg["local_by"]
        active = distance_squared(states[node]) > cfg["local_distance"]**2
        gradient = distance_gradient(states[node])/cfg["local_distance"]**2
        for j in range(4):
            rows.append(nrows); cols.append(4*(node-1)+j); data.append(gradient[j] if active else 0.)
        nrows += 1
    rows, cols, data = np.asarray(rows), np.asarray(cols), np.asarray(data)
    order = np.lexsort((cols, rows))
    rows, cols, data = rows[order], cols[order], data[order]
    require(len(set(zip(rows.tolist(), cols.tolist()))) == len(rows), "Unexpected duplicate Jacobian coordinate")
    pointers = np.r_[0, np.cumsum(np.bincount(rows, minlength=nrows))]
    return data, cols, pointers, (nrows, 5*T)


def gains(A, B, Q, R):
    A, B = np.asarray(A), np.asarray(B)
    require(A.ndim == 3 and A.shape[1:] == (4, 4) and B.shape == (len(A), 4, 1), "Riccati shape mismatch")
    P, W = np.diag(Q), np.diag(Q)
    K = np.empty((len(A), 1, 4))
    for t in reversed(range(len(A))):
        H = R+B[t].T@P@B[t]
        K[t] = np.linalg.solve(H, B[t].T@P@A[t])
        P = W+A[t].T@P@(A[t]-B[t]@K[t])
        P = (P+P.T)/2.
    require(np.isfinite(K).all(), "Nonfinite independently reconstructed gains")
    return K


def clipped_action(state, elapsed, reference_states, reference_actions, K):
    require(type(elapsed) is int and 0 <= elapsed < len(reference_actions), "Tracker elapsed outside its reference")
    state, reference_states = np.asarray(state), np.asarray(reference_states)
    require(state.shape == (5,) and reference_states.shape == (len(reference_actions)+1, 5), "Tracker state layout differs")
    require(abs(state[0]-elapsed*.01) <= 1e-9, "Tracker state clock differs")
    error = state[1:]-reference_states[elapsed, 1:]
    # Preserve the declared half-open [-pi,pi) interval at exact branch ties.
    error[:2] -= 2*np.pi*np.floor((error[:2]+np.pi)/(2*np.pi))
    return np.clip(np.asarray(reference_actions)[elapsed]-np.asarray(K)[elapsed]@error, -1., 1.)


def fixed_config():
    return dict(T=800, dt=.01, max_controls=1600000, reset_seeds=[191000, 191001],
        qualification_seeds=list(range(192000, 192008)), final_seeds=list(range(193000, 193032)),
        max_residual_calls=80, max_jacobian_calls=40, state_fd=[1e-5]*4, action_fd=1e-5,
        q_bound=4*np.pi, velocity_bound=30., dynamics_scale=[1., 1., 5., 5.],
        capture_steps=100, capture_distance=.2, capture_speed=.5, local_by=600,
        local_distance=.5, require_local_at_600_residual=True, qualification_atol=2e-10,
        defect_tolerance=1e-6, Q=[10., 10., 1., 1.], R=.1, final_min_capture_and_local=29)


def ordinary_state(seed):
    rng=np.random.default_rng(seed)
    return np.r_[0.,[np.pi,0.]+rng.uniform(-.1,.1,2),rng.uniform(-.05,.05,2)]


class Reader:
    def __init__(self, run):
        self.run, self.hashes = Path(run), {}

    def file(self, path):
        path = Path(path)
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(self.run.resolve()), "Missing/unsafe file: "+str(path))
        data = path.read_bytes(); self.hashes[str(path.resolve())] = digest(data)
        return data

    def json(self, path):
        return json.loads(self.file(path))

    def arrays(self, archive, ref):
        require(isinstance(ref, dict) and set(ref) == {"file", "sha256", "bytes"}, "Incomplete raw reference")
        name = Path(ref["file"])
        require(not name.is_absolute() and ".." not in name.parts, "Unsafe raw relative path")
        data = self.file(Path(archive)/name)
        require(digest(data) == ref["sha256"] and len(data) == ref["bytes"], "Raw file hash/bytes differ")
        with np.load(io.BytesIO(data), allow_pickle=False) as raw:
            require(len(set(raw.files)) == len(raw.files), "Duplicate raw array name")
            return {k:raw[k].copy() for k in raw.files}


def _counts(rows):
    requests = {r["query_id"]:r for r in rows if r["kind"] == "query_requested"}
    completed = {r["query_id"]:r for r in rows if r["kind"] == "query_complete"}
    failed = {r["query_id"]:r for r in rows if r["kind"] == "query_error"}
    require(not set(completed)&set(failed) and (set(completed)|set(failed)) <= set(requests), "Query resolution mismatch")
    unresolved = set(requests)-set(completed)-set(failed)
    returned = {q:r for q,r in failed.items() if r["call_returned"]}
    known = {**completed, **returned}
    requested = sum(r["requested_controls"] for r in requests.values())
    physical = sum(r["requested_physics_controls"] for r in requests.values())
    return dict(requested_controls=requested, charged_requested_controls=requested,
        requested_physics_controls=physical, requested_physics_frames=sum(r["requested_physics_frames"] for r in requests.values()),
        completed_controls=sum(r["completed_controls"] for r in known.values()),
        completed_physics_frames=sum(r["completed_physics_frames"] for r in known.values()),
        cutoff_skipped_controls=requested-physical,
        error_requested_controls=sum(requests[q]["requested_controls"] for q in failed),
        unknown_error_controls=sum(requests[q]["requested_physics_controls"] for q in failed if q not in returned),
        inflight_or_interrupted_controls=sum(requests[q]["requested_physics_controls"] for q in unresolved),
        unknown_controls=sum(requests[q]["requested_physics_controls"] for q in unresolved|(set(failed)-set(returned))),
        queries=len(requests), query_attempts=len(requests)+sum(r["kind"] == "query_rejected" for r in rows),
        completed_queries=len(completed), error_queries=len(failed), inflight_or_interrupted_queries=len(unresolved),
        rejected_queries=sum(r["kind"] == "query_rejected" for r in rows), incomplete_final_line=False)


def reconcile_budget(native_rows, heterogeneous_rows, cap):
    """Count each globally scoped reservation once, including partial runs."""
    reservations = []
    requests = {}
    for scope, rows in (("native",native_rows),("heterogeneous",heterogeneous_rows)):
        for row in rows:
            if row["kind"] == "query_requested":
                key=scope+":"+row["query_id"]
                require(key not in requests,"Duplicate scoped query intent")
                requests[key]=row
    for i,row in enumerate(native_rows):
        if row["kind"] == "reference_budget_reserved":
            require(i+1<len(native_rows) and native_rows[i+1]["kind"]=="query_requested",
                    "Ordinary reservation lacks its immediately following durable query")
            request=native_rows[i+1]
            require(row["requested"]==request["requested_controls"],"Ordinary reservation/request work differs")
            reservations.append((row["budget_reservation_id"],row["requested"],row["budget_requested_cumulative"],"native:"+request["query_id"]))
    for row in heterogeneous_rows:
        if row["kind"]=="query_requested":
            reservations.append((row["budget_reservation_id"],row["requested_controls"],row["budget_requested_cumulative"],"heterogeneous:"+row["query_id"]))
    reservations.sort(key=lambda x:int(x[0][1:]))
    require(len(reservations)==len(requests) and {r[3] for r in reservations}==set(requests),
            "Scoped query/reservation coverage differs")
    cumulative=0
    for i,(token,n,running,_) in enumerate(reservations):
        cumulative+=n
        require(token==f"b{i:09d}" and running==cumulative<=cap,"Shared durable budget sequence differs")
    return dict(reservations=len(reservations),requested_controls=cumulative)


def audit_shooting_reference(run):
    from .audit import audit_archive
    from .hetero_audit import audit_heterogeneous
    started = time.perf_counter(); run = Path(run).resolve(); reader = Reader(run)
    errors, pending, layers, counts, details = [], [], {}, {}, {}
    analytic = dict(residual_rows=0, finite_difference_rows=0, tracking_action_rows=0, reset_rng_rows=0)
    verdict = False
    def guard(label, callback):
        try:
            return callback()
        except Exception as error:
            errors.append(label+": "+type(error).__name__+": "+str(error)); return None
    def finish():
        return dict(schema="independent-native-shooting-reference-audit-v0",
            status="FAIL" if errors else "PENDING" if pending else "PASS", arithmetic_valid=not errors,
            scientific_verdict="PENDING" if errors or pending else "PASS" if verdict else "FAIL",
            native_reference_pass=bool(verdict and not errors and not pending), errors=errors,
            pending=list(dict.fromkeys(pending)), counts=counts, work=dict(counts), details=details,
            component_audits=layers, input_sha256=reader.hashes,
            source_sha256=digest(Path(__file__).read_bytes()), wall_seconds=time.perf_counter()-started,
            analytic_verification_rows=analytic, additional_physics_controls=0,
            additional_actor_inference_rows=0, additional_fits=0,
            limitations=["Independent saved arithmetic/provenance/ownership checks; no dynamics reintegration.",
                "A qualified finite tested reference is not an MPPI/GPS/actor or efficiency result.",
                "Failed finite shooting starts do not prove task infeasibility."])
    if not run.is_dir():
        pending.append("Reference run directory is absent"); return finish()
    # Detect declared-source corruption even when execution stopped before a
    # ledger/result existed. Missing execution evidence still remains pending.
    if (run/"manifest.json").is_file():
        def early_source_identity():
            manifest = reader.json(run/"manifest.json")
            require(isinstance(manifest.get("files"),list),"Source manifest lacks file inventory")
            seen = set()
            for row in manifest["files"]:
                name = Path(row["file"])
                require(not name.is_absolute() and ".." not in name.parts and name.parts[0] == "source"
                        and str(name) not in seen,"Unsafe/duplicate source inventory")
                seen.add(str(name)); blob = reader.file(run/name)
                require(digest(blob) == row["sha256"] and len(blob) == row["bytes"],"Frozen source hash/bytes differ: "+str(name))
        guard("early frozen source inventory",early_source_identity)
    for name, function in (("native", audit_archive), ("heterogeneous", audit_heterogeneous)):
        path = run/"evidence"/name
        if not (path/"ledger.jsonl").is_file():
            pending.append("Missing "+name+" ledger"); continue
        report = guard(name+" independent query audit", lambda f=function,p=path:
            f(p, source_root=run/"source", **({"allowed_root":run} if name == "native" else {"max_requested_controls":1600000})))
        if report is None: continue
        layers[name] = report
        errors.extend(name+": "+x for x in report.get("errors",[]))
        pending.extend(name+": "+x for x in report.get("pending",[]))
        for path, sha in report.get("input_sha256",{}).items():
            if path in reader.hashes and reader.hashes[path] != sha: errors.append("Conflicting consumed input hash")
            reader.hashes[path] = sha
    ledgers = {}
    for name in ("native","heterogeneous"):
        path=run/"evidence"/name/"ledger.jsonl"
        if not path.is_file():continue
        blob = guard(name+" raw paid ledger",lambda p=path:reader.file(p))
        if blob is None:continue
        try: ledgers[name] = [json.loads(line) for line in blob.splitlines()]
        except ValueError:
            pending.append("Incomplete/corrupt raw ledger prevents reference composition"); return finish()
        local = guard(name+" raw paid totals", lambda n=name:_counts(ledgers[n]))
        if local is not None:
            for key, value in local.items():
                if type(value) is int: counts[key] = counts.get(key, 0)+value
    if len(layers)!=2 or len(ledgers)!=2:return finish()
    if errors:
        return finish()
    for name in ("manifest.json", "config.json", "protocol.json", "reference_result.json", "stage_complete.json", "launch.json", "stage_started.json"):
        if not (run/name).is_file(): pending.append("Missing closed artifact: "+name)
    if pending:
        return finish()
    documents=guard("closed reference documents",lambda:{name:reader.json(run/name) for name in
        ("config.json","protocol.json","manifest.json","reference_result.json","stage_complete.json","launch.json","stage_started.json")})
    if documents is None:return finish()
    if guard("closed document types",lambda:require(all(isinstance(x,dict) for x in documents.values()),"Closed reference documents must be objects")) is None and errors:
        return finish()
    cfg,protocol=documents["config.json"],documents["protocol.json"]
    manifest,result,stage=documents["manifest.json"],documents["reference_result.json"],documents["stage_complete.json"]
    source_fp=manifest["source_fingerprint"]
    native, hetero = run/"evidence/native", run/"evidence/heterogeneous"
    nr, hr = ledgers["native"], ledgers["heterogeneous"]
    def header():
        require(cfg == fixed_config() and set(cfg) == set(fixed_config()), "Frozen finite reference recipe differs")
        require(protocol["schema"] == "native-rk4-shooting-reference-v0" and protocol["config"] == cfg,
                "Frozen reference protocol/config binding differs")
        expected_bound = dict(residual_controls=256000, jacobian_controls=1280000,
            candidate_live_replays=6400, tracker_jacobian_controls=16000,
            nominal_and_selection_live_controls=8000, final_live_controls=25600,
            qualification_allowance=8000, total=1600000, maximum_heterogeneous_solver_batches=480)
        require(protocol["work_bound"] == expected_bound and protocol["actor_rows"] == protocol["adam_steps"] == 0,
                "Prospective work vector changed")
        require(result["source_fingerprint"] == source_fp and stage["source_unchanged"] is True,
                "Runtime/source identity or source closure differs")
        launch,started_event=documents["launch.json"],documents["stage_started.json"]
        require(launch["source_manifest_sha256"]==digest(reader.file(run/"manifest.json"))
                and launch["source_fingerprint"]==source_fp==started_event["source_fingerprint"]
                and started_event["max_requested_controls"]==1600000 and started_event["no_retry"] is True,
                "Launch/runtime/source protocol binding differs")
        task_hashes={row["task_hash"] for rows in (nr,hr) for row in rows if row["kind"]=="query_requested"}
        require(task_hashes=={result["task_hash"]},"Result does not name the exact queried native task")
        require(stage["result_sha256"] == digest(reader.file(run/"reference_result.json")), "Stage does not bind exact result")
        require(result["accounting"] == {**counts, "incomplete_final_line":False}, "Producer paid totals differ from both raw ledgers")
        require(result["shared_budget"]["requested_controls"] == counts["requested_controls"] <= 1600000,
                "Shared paid budget differs/exceeded")
        require(result["actor_inference_rows"] == result["adam_steps"] == 0, "Reference claims actor/Adam work")
    guard("frozen protocol/accounting", header)
    sequence=guard("global disjoint paid sequence",lambda:reconcile_budget(nr,hr,1600000))
    if sequence is not None:
        details["global_budget_reservations"]=sequence["reservations"]
        guard("global paid sequence total",lambda:require(sequence["requested_controls"]==counts["requested_controls"],"Shared sequence omits paid work"))
    if result.get("error") or result.get("close_errors") or counts.get("unknown_controls"):
        pending.append("Execution/evidence error or unresolved paid work; no completed reference verdict")
        return finish()
    if errors: return finish()
    qmaps = {}
    for name, rows in ledgers.items():
        requests = {r["query_id"]:r for r in rows if r["kind"] == "query_requested"}
        completions = {r["query_id"]:r for r in rows if r["kind"] == "query_complete"}
        qmaps[name] = {q:dict(request=r, completion=completions[q],
            inputs=reader.arrays(run/"evidence"/name, r["inputs"]),
            raw=reader.arrays(run/"evidence"/name, completions[q]["result"])) for q,r in requests.items()}
    nq, hq = qmaps["native"], qmaps["heterogeneous"]
    used_native,used_heterogeneous=set(),set()
    event_index = {id(row):i for i,row in enumerate(nr)}
    episodes = {r["episode_id"]:r for r in nr if r["kind"] == "episode_saved"}
    verified_episodes = {r["episode_id"]:r for r in layers["native"]["episodes"]}
    episode_raw = {name:reader.arrays(native, event["raw"]) for name,event in episodes.items()}
    outcome_rows = {r["episode_id"]:r for r in nr if r["kind"] == "reference_episode_outcome"}
    def outcomes():
        require(set(episodes) == set(verified_episodes) == set(outcome_rows), "Custom outcomes are not exactly verified actual episodes")
        for name,event in outcome_rows.items():
            verified = verified_episodes[name]
            for key in ("context_id", "cost", "capture", "local", "complete", "controller_hash", "actor_hash", "task_hash", "query_ids"):
                require(canonical(event[key]) == canonical(verified[key]), "Custom actual outcome differs: "+name+"/"+key)
            require(event["admissible"] is bool(verified["complete"] and verified["capture"] and verified["local"]), "Reference admissibility differs")
            require(event["raw"] == episodes[name]["raw"] and event["excluded_from_teacher_aux"] is True, "Custom outcome raw/exclusion differs")
        details["verified_actual_episodes"] = len(episodes)
    guard("actual complete owner returns", outcomes)
    begin_solves = [r for r in nr if r["kind"] == "shooting_solve_begin"]
    ends = {r["solve_id"]:r for r in nr if r["kind"] == "shooting_solve_end"}
    def warm_qualification():
        events = [r for r in nr if r["kind"] == "native_heterogeneous_qualification"]
        require(len(events) == 1 and len(begin_solves) == 4, "Missing fixed qualification/four solve starts")
        event = events[0]
        require(event_index[id(event)] < min(event_index[id(r)] for r in begin_solves), "Solver starts before paid qualification")
        state_cases = ((np.pi,0.), (0.,0.), (.4,-.7), (2.4,1.2))
        velocity_cases = ((0.,0.), (.2,-.3), (3.,-2.), (-4.,1.))
        qrows = [(qid,q) for qid,q in hq.items() if q["request"]["purpose"] == "heterogeneous_backend_qualification"]
        require(len(qrows) == 1 and len(event["rows"]) == 16 and event["controls"] == 48, "Fixed16 paid backend roster differs")
        qualification_qid,qrow=qrows[0];raw=qrow["raw"]
        require(qrow["request"]["budget_reservation_id"]=="b000000000","Fixed native qualification was not first paid map call")
        used_heterogeneous.add(qualification_qid)
        for i,row in enumerate(event["rows"]):
            require(row["heterogeneous_query_id"]==qualification_qid and row["scalar_query_archive"]=="evidence/native"
                    and row["heterogeneous_query_archive"]=="evidence/heterogeneous","Qualifier scoped query references differ")
            state = np.array([((53*i)%800)*.01, *state_cases[i//4], *velocity_cases[i%4]])
            u = (-1.,-.3,.4,1.)[i%4]
            same(raw["initial_states"][i], state, "Fixed qualification state differs", atol=0, rtol=0)
            same(raw["actions"][i], [[u]], "Fixed qualification control differs", atol=0, rtol=0)
            results = []
            for j,qid in enumerate(row["scalar_query_ids"]):
                q = nq[qid]; pre = q["inputs"]
                used_native.add(qid)
                same(pre["initial_state"], state, "Scalar qualifier state differs", atol=0, rtol=0)
                same(pre["initial_warmstart"], [0.,0.] if j == 0 else [.125,-.25], "Warmstart intervention differs", atol=0, rtol=0)
                same(pre["actions"], [[[u]]], "Scalar qualifier action differs", atol=0, rtol=0)
                require(nr.index(q["completion"]) < event_index[id(event)], "Qualification claims future scalar result")
                results.append(q["raw"]["states"][0,0])
            require(len(results) == 2, "Missing warmstart intervention pair")
            err1 = float(np.max(abs(results[0]-raw["states"][i,0])))
            err2 = float(np.max(abs(results[0]-results[1])))
            same(row["heterogeneous_error"], err1, "Qualifier error arithmetic differs")
            same(row["warmstart_intervention_error"], err2, "Warmstart error arithmetic differs")
            require(max(err1,err2) <= cfg["qualification_atol"] and row["passed"] is True, "Native qualification did not pass")
        require(event["passed"] is True and result["qualification"] == {k:v for k,v in event.items() if k not in ("id","kind","utc","wall_seconds")}, "Qualifier summary differs")
    guard("native scalar/heterogeneous warmstart qualification", warm_qualification)
    def unpack_variables(raw, initial):
        variables = raw["variables"]
        require(variables.shape == (4000,) and variables.dtype == np.dtype("float64")
                and np.isfinite(variables).all(), "Shooting variable layout differs")
        physical = np.vstack((initial[1:], variables[:3200].reshape(800,4)))
        actions = variables[3200:,None]
        require(np.max(abs(actions)) <= 1 and np.max(abs(physical[1:,:2])) <= cfg["q_bound"]
                and np.max(abs(physical[1:,2:])) <= cfg["velocity_bound"], "Optimizer variables escaped declared search box")
        return variables, physical, actions
    callback_packets = {}
    def callbacks():
        expected_ids = [f"root{i}_winding{side}" for i in range(2) for side in ("minus","plus")]
        require([r["solve_id"] for r in begin_solves] == expected_ids and set(ends) == set(expected_ids), "Fixed solve order differs")
        for i,begin in enumerate(begin_solves):
            require(begin["winding"]==(-1 if i%2==0 else 1) and begin["max_nfev"]==80
                    and begin["max_jacobian_calls"]==40 and begin["initial_guess_is_experience"] is False,
                    "Solve initialization/callback allowance differs")
        for root_index,seed in enumerate(cfg["reset_seeds"]):
            initial = ordinary_state(seed)
            analytic["reset_rng_rows"]+=1
            for side in ("minus", "plus"):
                sid = f"root{root_index}_winding{side}"
                records = []
                for kind, limit in (("residual",80),("jacobian",40)):
                    starts = [r for r in nr if r["kind"] == f"shooting_{kind}_requested" and r["solve_id"] == sid]
                    done = [r for r in nr if r["kind"] == f"shooting_{kind}_complete" and r["solve_id"] == sid]
                    require(len(starts) == len(done) <= limit and [r["callback"] for r in starts] == list(range(len(starts)))
                            and [r["callback"] for r in done] == list(range(len(done))), "Callback work missing/reordered/over cap")
                    for i,row in enumerate(done):
                        require(event_index[id(begin_solves[expected_ids.index(sid)])]<event_index[id(starts[i])]
                                <event_index[id(row)]<event_index[id(ends[sid])],"Callback escaped its solve chronology")
                        require(row["query_archive"] == "evidence/heterogeneous", "Callback query is unscoped")
                        q = hq[row["query_id"]]
                        used_heterogeneous.add(row["query_id"])
                        require(q["request"]["purpose"] == "shooting_"+kind and q["request"]["metadata"]["solve_id"] == sid
                                and q["request"]["metadata"]["callback"] == i, "Callback names wrong native map work")
                        packet = reader.arrays(native, row["raw"])
                        variables, physical, actions = unpack_variables(packet, initial)
                        prestates = np.column_stack((np.arange(800)*.01, physical[:-1]))
                        if kind == "residual":
                            same(q["inputs"]["initial_states"], prestates, "Residual prestate input differs", atol=0, rtol=0)
                            same(q["inputs"]["actions"], actions[:,None,:], "Residual action input differs", atol=0, rtol=0)
                            defects = (physical[1:]-q["raw"]["states"][:,0,1:])/np.array(cfg["dynamics_scale"])
                            residual = np.r_[defects.ravel(), constraints(physical,cfg)]
                            same(packet["residual"],residual,"Residual does not use measured native map")
                            same(packet["dynamics_defects"],defects,"Dynamics defects differ")
                            same(packet["guessed_states"],physical,"Guessed state layout differs",atol=0,rtol=0)
                            require(packet["guessed_states_are_experience"].item() is False,"Guessed states promoted as experience")
                            norm = float(np.linalg.norm(residual)); same(row["residual_norm"],norm,"Residual norm differs")
                            records.append((norm,variables,physical,actions,row,defects))
                            analytic["residual_rows"] += len(residual)
                        else:
                            ss,uu,denominator = fd_inputs(prestates,actions,cfg["state_fd"],cfg["action_fd"])
                            same(q["inputs"]["initial_states"],ss,"FD perturbation states differ",atol=0,rtol=0)
                            same(q["inputs"]["actions"],uu[:,None,:],"FD perturbation actions differ",atol=0,rtol=0)
                            A,B = derivatives(q["raw"]["states"][:,0],denominator,cfg["state_fd"])
                            same(packet["A"],A,"Native A derivative differs");same(packet["B"],B,"Native B derivative differs")
                            same(packet["action_denominator"],denominator,"Bounded action FD denominator differs",atol=0,rtol=0)
                            data,indices,indptr,shape = expected_csr(physical,A,B,cfg)
                            same(packet["csr_data"],data,"Sparse Jacobian values differ")
                            for key,expected in (("csr_indices",indices),("csr_indptr",indptr),("csr_shape",shape)):
                                same(packet[key],expected,"Sparse Jacobian structure differs",atol=0,rtol=0)
                            analytic["finite_difference_rows"] += 8000
                require(records,"No measured residual candidate")
                best = min(records,key=lambda x:x[0]); end=ends[sid]
                require(end["residual_calls"] == len(records) and end["jacobian_calls"] == len([r for r in nr if r["kind"]=="shooting_jacobian_complete" and r["solve_id"]==sid]), "Solver callback summary differs")
                candidate = reader.arrays(native,end["raw"])
                same(end["best_residual_norm"],best[0],"Best residual score differs")
                defect_max=float(np.max(abs(best[5])))
                constraint_max=float(np.max(constraints(best[2],cfg)))
                same(end["best_scaled_dynamics_defect_max"],defect_max,"Reported best native defect differs")
                same(end["best_capture_local_violation_max"],constraint_max,"Reported best constraint violation differs")
                require(end["guessed_constraints_closed"] is bool(max(defect_max,constraint_max)<=cfg["defect_tolerance"]),
                        "Optimizer residual closure claim differs")
                same(candidate["actions"],best[3],"Candidate not best measured residual",atol=0,rtol=0)
                same(candidate["guessed_states"],best[2],"Candidate guessed trajectory differs",atol=0,rtol=0)
                require(candidate["guessed_states_are_experience"].item() is False and end["accepted_as_experience"] is False,"Solver status promoted guessed experience")
                callback_packets[sid] = dict(initial=initial,actions=best[3],end=end)
        details["finite_solve_callbacks_verified"] = len(callback_packets)
    guard("all finite native shooting callbacks",callbacks)
    if errors: return finish()
    def candidate_replays():
        require(len(result["candidates"]) == 4,"Missing actual candidate replay roster")
        for index,candidate in enumerate(result["candidates"]):
            sid = list(callback_packets)[index]; packet=callback_packets[sid]
            require(candidate["root_index"]==index//2 and candidate["seed"]==cfg["reset_seeds"][index//2],"Candidate ordinary root roster differs")
            require(candidate["solve"] == {k:v for k,v in packet["end"].items() if k not in ("id","kind","utc","wall_seconds")},"Candidate solve record not bound to raw event")
            expected_hash=identity(dict(algorithm="native-shooting-open-loop",solve_id=sid,
                actions_sha256=digest(np.ascontiguousarray(packet["actions"]).tobytes()),task_hash=result["task_hash"],source_fingerprint=source_fp))
            require(len(candidate["replays"]) == 2,"Candidate actual duplicate replay count differs")
            admissible=[]
            for repeat,outcome in enumerate(candidate["replays"]):
                name=f"{sid}_replay{repeat}"; event=outcome_rows[name]; raw=episode_raw[name]
                require(outcome == {k:v for k,v in event.items() if k not in ("id","kind","utc","wall_seconds")},"Candidate outcome differs from independently verified actual episode")
                same(raw["states"][0],packet["initial"],"Replay reset/root differs",atol=1e-15,rtol=0)
                same(raw["actions"],packet["actions"],"Actual replay did not execute optimized native actions",atol=0,rtol=0)
                require(event["controller_hash"] == expected_hash,"Replay controller owner differs")
                require(nr.index(nq[event["query_ids"][0]]["request"])>event_index[id(packet["end"])],
                        "Actual candidate replay preceded completed optimizer record")
                used_native.update(event["query_ids"])
                admissible.append(event["admissible"])
            require(candidate["accepted"] is all(admissible),"Candidate accepted without both actual captures/local approach")
    guard("actual candidate ownership/replays",candidate_replays)
    tracker_data = {}
    def tracker_math():
        expected_candidates=[]
        for root_index in range(2):
            options=[(float(np.mean([r["cost"] for r in c["replays"]])),i) for i,c in enumerate(result["candidates"])
                     if c["root_index"] == root_index and c["accepted"]]
            if options: expected_candidates.append(min(options,key=lambda x:(x[0],x[1]))[1])
        require([t["candidate_index"] for t in result["trackers"]] == expected_candidates,"Tracker does not preserve best successful per-root family")
        for index,tracker in enumerate(result["trackers"]):
            require(tracker["index"] == index,"Tracker indices reorder")
            raw=reader.arrays(native,tracker["raw"])
            candidate=result["candidates"][tracker["candidate_index"]]
            reference=episode_raw[candidate["replays"][-1]["episode_id"]]
            require(tracker["reference_episode_raw"] == candidate["replays"][-1]["raw"]
                    and tracker["linearization_query_archive"] == "evidence/heterogeneous","Tracker ancestry/raw map missing")
            same(raw["states"],reference["states"],"Tracker reference is not actual replay",atol=0,rtol=0)
            same(raw["actions"],reference["actions"],"Tracker reference actions differ",atol=0,rtol=0)
            q=hq[tracker["linearization_query_id"]]
            require(q["request"]["purpose"]=="native_tracker_linearization","Tracker names non-linearization work")
            used_heterogeneous.add(tracker["linearization_query_id"])
            ss,uu,denom=fd_inputs(raw["states"][:-1],raw["actions"],cfg["state_fd"],cfg["action_fd"])
            same(q["inputs"]["initial_states"],ss,"Tracker FD states not from actual reference",atol=0,rtol=0)
            same(q["inputs"]["actions"],uu[:,None,:],"Tracker FD actions differ",atol=0,rtol=0)
            A,B=derivatives(q["raw"]["states"][:,0],denom,cfg["state_fd"])
            same(raw["A"],A,"Tracker A differs");same(raw["B"],B,"Tracker B differs")
            same(raw["Q"],cfg["Q"],"Tracker Q retuned",atol=0,rtol=0);same(raw["R"],cfg["R"],"Tracker R retuned",atol=0,rtol=0)
            K=gains(A,B,cfg["Q"],cfg["R"])
            same(raw["gains"],K,"Tracker Riccati/gains differ",atol=2e-8,rtol=2e-8)
            owner=identity(dict(algorithm="native-shooting-tvlqr-clipped",raw=tracker["raw"],source_fingerprint=source_fp,task_hash=result["task_hash"]))
            require(tracker["controller_hash"] == owner,"Tracker controller identity differs")
            tracker_data[index]=(raw,K)
            analytic["finite_difference_rows"] += 8000
    guard("independent tracker derivative/Riccati math",tracker_math)
    def tracking_panels():
        for index,tracker in enumerate(result["trackers"]):
            names=[f"tracker{index}_nominal"]+[f"tracker{index}_selection{s}" for i,s in enumerate(cfg["qualification_seeds"]) if i%len(result["trackers"])==index]
            require(tracker["nominal"] == {k:v for k,v in outcome_rows[names[0]].items() if k not in ("id","kind","utc","wall_seconds")},"Nominal tracker outcome differs")
            require([r["episode_id"] for r in tracker["qualification"]] == names[1:],"Fresh selection seed allocation differs")
            for outcome in tracker["qualification"]:
                actual=outcome_rows[outcome["episode_id"]]
                require(outcome=={k:v for k,v in actual.items() if k not in ("id","kind","utc","wall_seconds")},"Tracker selection summary differs from measured actual outcome")
            raw,K=tracker_data[index]
            for name in names:
                episode=episode_raw[name]
                require(outcome_rows[name]["controller_hash"] == tracker["controller_hash"],"Tracking episode switched owner")
                if name.endswith("_nominal"):
                    same(episode["states"][0],raw["states"][0],"Nominal tracker root differs",atol=0,rtol=0)
                else:
                    seed=int(name.split("_selection")[1])
                    same(episode["states"][0],ordinary_state(seed),"Fresh selection reset is not fixed PCG root",atol=1e-15,rtol=0)
                    analytic["reset_rng_rows"]+=1
                same(episode["warmstarts"][0],np.zeros(2),"Fresh ordinary reset warmstart differs",atol=0,rtol=0)
                used_native.update(outcome_rows[name]["query_ids"])
                for t in range(800):
                    same(episode["actions"][t],clipped_action(episode["states"][t],t,raw["states"],raw["actions"],K),"Actual tracking action differs",atol=2e-8,rtol=2e-8)
                analytic["tracking_action_rows"] += 800
        eligible=[t for t in result["trackers"] if t["nominal"]["admissible"] and t["qualification"]
                  and all(r["admissible"] for r in t["qualification"])]
        frozen=[r for r in nr if r["kind"] == "native_tracker_frozen"]
        if not eligible:
            require(not frozen and not result["final_outcomes"],"Unqualified tracker received final promotion")
            require(result["final_admissible"]==0,"Empty final panel reports invented successes")
            details["selected_tracker"] = None
            return False
        chosen=min(eligible,key=lambda t:(float(np.mean([r["cost"] for r in t["qualification"]])),t["index"]))
        require(len(frozen)==1 and frozen[0]["tracker_index"] == chosen["index"] and frozen[0]["controller_hash"] == chosen["controller_hash"]
                and frozen[0]["before_fresh_final_seeds"] == cfg["final_seeds"],"Selected tracker/fresh panel was not frozen")
        require([r["episode_id"] for r in result["final_outcomes"]] == [f"final{s}" for s in cfg["final_seeds"]],"Final fresh32 roster differs")
        raw,K=tracker_data[chosen["index"]]
        for seed,outcome in zip(cfg["final_seeds"],result["final_outcomes"]):
            name=f"final{seed}"; episode=episode_raw[name]; event=outcome_rows[name]
            require(outcome == {k:v for k,v in event.items() if k not in ("id","kind","utc","wall_seconds")},"Final summary differs from verified episode")
            require(event["context_id"] == f"ordinary:{seed}" and event["controller_hash"] == chosen["controller_hash"],"Final panel reset/owner differs")
            same(episode["states"][0],ordinary_state(seed),"Final reset is not fixed fresh PCG root",atol=1e-15,rtol=0)
            same(episode["warmstarts"][0],np.zeros(2),"Final ordinary warmstart differs",atol=0,rtol=0)
            used_native.update(event["query_ids"]);analytic["reset_rng_rows"]+=1
            request=nq[event["query_ids"][0]]["request"]
            require(nr.index(request)>event_index[id(frozen[0])],"Final execution preceded controller freeze")
            for t in range(800):
                same(episode["actions"][t],clipped_action(episode["states"][t],t,raw["states"],raw["actions"],K),"Fresh final action is not frozen feedback controller",atol=2e-8,rtol=2e-8)
            analytic["tracking_action_rows"] += 800
        successes=sum(outcome_rows[f"final{s}"]["admissible"] for s in cfg["final_seeds"])
        require(result["final_admissible"] == successes,"Fresh final success count differs")
        details.update(selected_tracker=chosen["index"],fresh_final_admissible=successes,
            fresh_final_roots=32,criterion=">=29/32 both original local and capture")
        return successes>=29
    if not errors:
        measured=guard("complete actual tracking/selection/fresh final",tracking_panels)
        verdict=bool(measured)
        guard("exact finite query roster",lambda:require(used_native==set(nq) and used_heterogeneous==set(hq),
            "Queries outside fixed qualification/callback/replay/tracker/final roster"))
        guard("reported reference verdict",lambda:require(result["native_reference_pass"] is verdict
            and stage["native_reference_pass"] is verdict,"Producer verdict differs from raw executed panel"))
        expected_stop=("no_admissible_native_candidate" if not any(c["accepted"] for c in result["candidates"])
            else "no_admissible_feedback_tracker" if not result["final_outcomes"]
            else "tested_native_reference_qualified" if verdict else "fresh_final_feedback_qualification_failed")
        guard("reported stopping link",lambda:require(result["status"]=="complete" and result["stopped_at"]==expected_stop,
            "Reported first unsupported reference link differs"))
        kinds=("shooting_solve_begin","shooting_solve_end","shooting_residual_requested",
               "shooting_residual_complete","shooting_jacobian_requested","shooting_jacobian_complete")
        guard("numerical optimization work summary",lambda:require(result["scipy_work_counts"]==
            {kind:sum(r["kind"]==kind for r in nr) for kind in kinds},"SciPy callback work summary differs from durable intents"))
    details["duplicate_candidate_replays_are_independent_evidence"] = False
    details["full_native_control_scope"] = "800native controls; no action-policy proposals or imagined-return targets"
    return finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path);parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError("Audits are create-only")
    report=audit_shooting_reference(args.run)
    with args.out.open("x") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write("\n")
    print(json.dumps({k:report[k] for k in ("status","scientific_verdict","native_reference_pass","counts","wall_seconds")},indent=2))


if __name__ == "__main__":
    main()
