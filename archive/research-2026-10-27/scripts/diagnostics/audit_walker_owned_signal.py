"""Independent arithmetic/action audit of a bounded candidate-owned Walker assay.

Imports only NumPy and the standard library. No simulator, Torch, project
runner, checkpoint loader or training code is imported. Optional NumPy action
verification is separately charged and is required for a final signal PASS.
Recorded state transitions are not an independent dynamics replay.

CLI: audit_walker_owned_signal.py RUN --source-root FROZEN_PACKET
     --verify-actions --out NEW_AUDIT.json
The manifest defaults to RUN/frozen_manifest.json. Verification also creates
NEW_AUDIT.inferences.jsonl and refuses an existing ledger even if the JSON
result is absent. Interrupted reserved rows remain charged in that journal.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import time

import numpy as np


CLASS_ID = "src.gps.policy.TanhMLPPolicy"
PARAMETERS = ("fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc_mean.weight", "fc_mean.bias")
BUFFERS = ("action_low", "action_high", "action_scale", "action_bias", "obs_mean", "obs_std")
PARTS = ("velocity", "vertical_velocity", "angle", "unhealthy", "control", "fall")
TASK = dict(frame_skip=4, cost_style="target_velocity", target_velocity=1.5,
    vel_cost_weight=2., vertical_velocity_cost_weight=.5, angle_cost_weight=.1,
    unhealthy_cost_weight=5., ctrl_cost_weight=.001, terminal_unhealthy_weight=20.,
    apply_terminal_cost_on_done=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_array(value, shape=None):
    result = np.asarray(value)
    require(result.dtype.kind in "biuf" and np.isfinite(result).all(), "nonfinite/nonnumeric array")
    if shape is not None:
        require(result.shape == shape, f"array shape {result.shape} != {shape}")
    return result


def same(actual, expected, label, *, atol=1e-9, rtol=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all()
            and np.allclose(a, b, atol=atol, rtol=rtol), label)


def safe_path(root, relative):
    require(isinstance(relative, str), "relative evidence path must be a string")
    part = PurePosixPath(relative)
    require(not part.is_absolute() and ".." not in part.parts and part.parts,
            "unsafe relative evidence path")
    result = (root / relative).resolve()
    require(result.is_relative_to(root.resolve()), "evidence path escapes run")
    return result


def jsonl(path):
    """Complete lines only; an interrupted tail remains explicit pending work."""
    rows, partial = [], False
    lines = path.read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            require(index == len(lines)-1 and not line.endswith(b"\n"), "corrupt complete JSONL line")
            partial = True
    return rows, partial


def snapshot_hash(state, warmstart):
    h = hashlib.sha256()
    for name, value in (("state", state), ("warmstart", warmstart)):
        a = np.asarray(value, dtype="<f8")
        h.update(name.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def healthy(states):
    states = np.asarray(states)
    return (states[..., 2] > .8) & (states[..., 2] < 2.) & (states[..., 3] > -1.) & (states[..., 3] < 1.)


def observations(states):
    states = np.asarray(states)
    return np.concatenate((states[..., 2:10], np.clip(states[..., 10:19], -10., 10.)), axis=-1)


def absorbing_scale(gamma, remaining):
    require(type(remaining) is int and remaining >= 0 and type(gamma) in (int, float)
            and 0 <= gamma < 1 and math.isfinite(gamma), "invalid finite horizon/discount")
    # Direct finite sum is deliberately independent of the runner's expm1 form.
    return float(5. * np.sum(float(gamma) ** np.arange(remaining)))


def cost_components(poststates, actions):
    s, a = np.asarray(poststates), np.asarray(actions)
    bad = ~healthy(s)
    return np.column_stack((2.*np.abs(1.5-s[:, 10]), .5*s[:, 11]**2,
        .1*s[:, 3]**2, 5.*bad, .001*np.sum(a*a, axis=1), 20.*bad))


def audit_trace(trace, result, context):
    """Recompute complete own-suffix costs from raw state/action transitions."""
    n, gamma = result["executed_steps"], result["gamma"]
    remaining = context["remaining"]
    require(type(n) is int and 0 <= n <= remaining, "invalid executed horizon")
    require(result.get("remaining") == remaining and result.get("complete") is True
            and result.get("cutoff") is False and result.get("continuation_owner") == "candidate"
            and result.get("target_kind") == "complete_changed_actor_return", "not a complete candidate-owned suffix")
    s = finite_array(trace["states"], (n+1, 19)).astype(np.float64)
    w = finite_array(trace["warmstarts"], (n+1, 9)).astype(np.float64)
    o = finite_array(trace["observations"], (n+1, 17)).astype(np.float64)
    a = finite_array(trace["actions"], (n, 6)).astype(np.float64)
    c = finite_array(trace["raw_costs"], (n,)).astype(np.float64)
    parts = finite_array(trace["cost_components"], (n, 6)).astype(np.float64)
    require(np.all(np.abs(a) <= 1.), "action outside native actuator bounds")
    same(o, observations(s), "state-to-observation mismatch", atol=0, rtol=0)
    same(s[:,0],s[0,0]+.008*np.arange(n+1),"raw control-boundary clock mismatch",atol=1e-10,rtol=0)
    same(np.diff(s[:,0]),np.full(n,.008),"raw control timestep mismatch",atol=1e-12,rtol=0)
    require(snapshot_hash(s[0], w[0]) == context["snapshot_digest"] == result["snapshot_digest"], "query snapshot digest mismatch")
    if "state" in context:
        same(s[0], context["state"], "context initial state mismatch", atol=0, rtol=0)
        same(w[0], context["warmstart"], "context initial warmstart mismatch", atol=0, rtol=0)
    health = healthy(s)
    terminated, truncated = result["terminated"], result["truncated"]
    require(type(terminated) is bool and type(truncated) is bool, "termination flags must be bool")
    require(terminated == bool(not health[-1]) and truncated == bool(n == remaining and not terminated), "health/termination/truncation mismatch")
    require((n == 0 and terminated and not health[0]) or (n >= 1 and health[:-1].all()), "rollout continues after a fall or stops before healthy action")
    require(n == remaining or terminated, "healthy incomplete suffix")
    expected = cost_components(s[1:], a)
    same(parts, expected, "raw cost-component mismatch")
    same(c, expected.sum(axis=1), "raw cost mismatch")
    executed = float(np.dot(float(gamma)**np.arange(n), c))
    m = remaining-n if terminated else 0
    absorption = float(float(gamma)**n * absorbing_scale(gamma, m))
    same(result["executed_cost"], executed, "executed return mismatch")
    same(result["absorbing_cost"], absorption, "finite absorption mismatch")
    same(result["cost"], executed+absorption, "total own-return mismatch")
    require(result["absorbing_steps"] == m and result["unhealthy_cost"] == 5., "absorbing metadata mismatch")
    return dict(cost=executed+absorption, executed_steps=n, absorption=absorption,
                terminated=terminated, states=s, warmstarts=w, observations=o, actions=a,
                raw_costs=c, cost_components=parts)


def owner_hash(arrays, class_identity=CLASS_ID):
    h = hashlib.sha256(); h.update(class_identity.encode())
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        require(a.dtype.kind in "biuf" and np.isfinite(a).all(), "nonfinite actor state")
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def validate_actor(arrays, *, hidden=256):
    require(set(arrays) == set(PARAMETERS+BUFFERS), "actor keys/layout differ from declared class")
    shapes = dict(zip(PARAMETERS, ((hidden,17),(hidden,),(hidden,hidden),(hidden,),(6,hidden),(6,))))
    for name, shape in shapes.items():
        a = finite_array(arrays[name], shape)
        require(a.dtype == np.dtype("float32"), "actor parameters must preserve float32")
    for name, expected in dict(action_low=np.full(6,-1.), action_high=np.ones(6),
        action_scale=np.ones(6), action_bias=np.zeros(6), obs_mean=np.zeros(17), obs_std=np.ones(17)).items():
        a = finite_array(arrays[name], expected.shape)
        require(a.dtype == np.dtype("float32") and np.array_equal(a, expected), "actor buffers/identity transform mismatch")
    return np.concatenate([arrays[name].astype(np.float64).reshape(-1) for name in PARAMETERS])


def numpy_actions(arrays, obs, *, batch_size=256):
    """Independent float32 forward arithmetic; callers separately count rows."""
    x = finite_array(obs).astype(np.float32)
    require(x.ndim == 2 and x.shape[1] == 17, "action-verification observations shape")
    result = []
    for start in range(0, len(x), batch_size):
        h = np.tanh(x[start:start+batch_size] @ arrays["fc1.weight"].T + arrays["fc1.bias"])
        h = np.tanh(h @ arrays["fc2.weight"].T + arrays["fc2.bias"])
        result.append(np.tanh(h @ arrays["fc_mean.weight"].T + arrays["fc_mean.bias"])
                      * arrays["action_scale"] + arrays["action_bias"])
    return np.concatenate(result) if result else np.empty((0,6), dtype=np.float32)


def audit_journal(rows, *, query_ref, owner, context, trace=None):
    """Independently reconcile paid attempts, returned transitions and failures."""
    require(rows and rows[0].get("kind") == "query_begin", "query-begin journal record missing")
    begin = rows[0]
    require(begin["owner_hash"] == owner and begin["context_id"] == context["id"]
            and canonical(begin["ancestry"]) == canonical(context["ancestry"])
            and begin["budget_controls"] == context["remaining"], "journal ownership/context/budget mismatch")
    require(begin["task_config"] == TASK, "journal task differs from pinned task")
    attempts, completions, errors = {}, {}, {}
    active, ended, rejected, call_wall, resets, reset_done = None, False, 0, 0., 0, 0
    for index, row in enumerate(rows):
        kind = row["kind"]
        require(row["id"] == f"{query_ref}:{index}" and row["query_id"] == query_ref,
                "journal ID/sequence mismatch")
        require(row.get("controls") == int(kind == "control_attempt"), "journal control charge mismatch")
        require(not ended, "record appended after query end")
        if kind == "control_attempt":
            require(active is None and not errors, "overlapping or postfailure control attempt")
            finite_array(row["pre"]["state"], (19,)); finite_array(row["pre"]["warmstart"], (9,))
            action = finite_array(row["action"], (6,))
            require(np.all(np.abs(action) <= 1.), "attempt action outside native bounds")
            attempts[row["id"]] = row; active = row["id"]
        elif kind in ("control_complete", "control_error"):
            identifier = row["attempt_id"]
            require(identifier == active and identifier in attempts, "unmatched/duplicate completed attempt")
            if kind == "control_complete":
                completions[identifier] = row
                s = finite_array(row["post"]["state"], (19,)).astype(float)
                o = finite_array(row["observation"], (17,))
                a = np.asarray(attempts[identifier]["action"], dtype=float)
                same(o, observations(s), "journal post-observation mismatch", atol=0, rtol=0)
                parts = cost_components(s[None], a[None])[0]
                same([row["cost_components"][p] for p in PARTS], parts, "journal cost components mismatch")
                same(row["raw_task_cost"], parts.sum(), "journal raw cost mismatch")
                same(row["info"]["task_cost"], parts.sum(), "journal info raw cost mismatch")
                require(row["terminated"] == bool(not healthy(s))
                        and type(row["terminated"]) is bool and type(row["cutoff"]) is bool,
                        "journal health/termination mismatch")
                if "fall_cost" in row["info"]:
                    same(row["info"]["fall_cost"], parts[-1], "journal fall-cost mismatch")
                require(type(row["call_wall_seconds"]) in (int,float) and row["call_wall_seconds"] >= 0,
                        "invalid simulator-call wall time")
                call_wall += row["call_wall_seconds"]
            else:
                require(type(row["call_returned"]) is bool, "error call-returned flag missing")
                errors[identifier] = row
            active = None
        elif kind == "control_rejected":
            rejected += 1
        elif kind == "reset_attempt":
            require(not attempts and resets == 0,"repeated/postcontrol simulator reset")
            requested = row["options"].get("options",{})
            require(snapshot_hash(requested["state"],requested["warmstart"]) == context["snapshot_digest"],
                    "reset requested a different state/warmstart")
            if "seed" in row["options"]:
                require(row["options"]["seed"] == context["ancestry"]["reset_seed"],"reset seed differs from context root")
            resets += 1
        elif kind == "reset_complete":
            require(resets == 1 and reset_done == 0 and not attempts,"unmatched simulator reset")
            require(snapshot_hash(row["state"],row["warmstart"]) == context["snapshot_digest"],"reset lost exact snapshot")
            same(row["observation"],observations(row["state"]),"reset observation mismatch",atol=0,rtol=0)
            reset_done += 1
        elif kind == "query_end":
            ended = True
            if row.get("status") == "complete":
                require(active is None and not errors and not rejected, "failed/unresolved query marked complete")
        elif kind != "query_begin" or index != 0:
            raise ValueError("unknown/duplicate query journal record kind")
    require(len(attempts) <= context["remaining"], "query exceeds full reserved horizon")
    complete = bool(ended and rows[-1].get("status") == "complete" and not errors and active is None)
    if trace is not None:
        n = trace["executed_steps"]
        require(complete and len(attempts) == len(completions) == n, "successful raw trace/journal attempt counts disagree")
        for index, (identifier, attempt) in enumerate(attempts.items()):
            done = completions[identifier]
            for a, b, label in ((attempt["pre"]["state"], trace["states"][index], "prestate"),
                (attempt["pre"]["warmstart"],trace["warmstarts"][index],"prewarm"),
                (attempt["action"],trace["actions"][index],"action"),
                (done["post"]["state"],trace["states"][index+1],"poststate"),
                (done["post"]["warmstart"],trace["warmstarts"][index+1],"postwarm"),
                (done["observation"],trace["observations"][index+1],"observation")):
                same(a,b,"journal/raw "+label+" mismatch",atol=0,rtol=0)
            same(done["raw_task_cost"],trace["raw_costs"][index],"journal/raw cost mismatch")
            require(done["terminated"] == bool(index == n-1 and trace["terminated"]), "nonfinal/omitted fall")
    return dict(charged_requested_controls=len(attempts), completed_controls=len(completions),
        error_controls=len(errors), inflight_or_interrupted_controls=len(attempts)-len(completions)-len(errors),
        confirmed_returned_controls=len(completions)+sum(r["call_returned"] for r in errors.values()),
        unknown_partial_error_controls=sum(not r["call_returned"] for r in errors.values()),
        rejected_before_call=rejected, query_ended=ended, complete=complete,
        simulator_call_wall_seconds=call_wall, records=len(rows),
        reset_attempts=resets,reset_completions=reset_done)


def score_panel(refs, owner, contexts, results, gamma):
    require(len(refs) == len(contexts) and len(set(refs)) == len(refs), "panel query references incomplete/duplicated")
    normalized = []
    for ref, context in zip(refs, contexts):
        query = results[ref]
        require(query["owner_hash"] == owner and query["context_id"] == context["id"]
                and query["remaining"] == context["remaining"] and query["gamma"] == gamma,
                "panel reference belongs to a different owner/context/objective")
        normalized.append(query["cost"]/max(1.,absorbing_scale(gamma,context["remaining"])))
    normalized = np.asarray(normalized)
    roles = {role:float(normalized[[c["role"] == role for c in contexts]].mean())
             for role in sorted({c["role"] for c in contexts})}
    clusters = {}
    for root in sorted({c["ancestry"]["reset_seed"] for c in contexts}):
        values = [float(normalized[[c["ancestry"]["reset_seed"] == root and c["role"] == role
                   for c in contexts]].mean()) for role in ("root","guide")]
        clusters[str(root)] = float(np.mean(values))
    return dict(score=float(np.mean(list(roles.values()))), role_scores=roles,
                normalized_returns=normalized, root_clusters=clusters)


def select_feasible(scores, root_scores, tolerance=0.):
    scores,roots=finite_array(scores),finite_array(root_scores)
    require(scores.ndim==1 and len(scores)>0 and roots.shape==scores.shape,"invalid feasible-selection bank")
    eligible=roots<=roots[0]+tolerance
    return dict(selected_index=int(np.argmin(np.where(eligible,scores,np.inf))),
                unconstrained_selected_index=int(np.argmin(scores)),feasible_flags=eligible)


def canonical_contexts(catalog):
    """Independently reconstruct exact physical-context aliases from all60 slots."""
    slots = catalog["contexts"]
    require(len(slots) == 60 and sum(s["active"] is True for s in slots) == 59, "catalog slots/failure omitted")
    groups, inactive = {}, []
    for slot in slots:
        if slot["active"] is not True:
            require(slot["id"] == "pair1:reset31:planner20031:t900" and slot["fall_before_handover"] is True,
                    "unexpected unavailable provenance slot")
            inactive.append(slot); continue
        d = snapshot_hash(slot["state"], slot["warmstart"])
        require(d == slot["snapshot_digest"], "catalog snapshot hash mismatch")
        groups.setdefault(d, []).append(slot)
    output = dict(train=[], heldout=[], inactive=inactive, all_slots=slots, active_slots=59, duplicate_aliases=10)
    for d, aliases in groups.items():
        aliases.sort(key=lambda s:s["id"]); first = aliases[0]
        root = first["ancestry"]["reset_seed"]
        split = "train" if root in range(30,35) else "heldout" if root in range(35,40) else None
        require(split is not None and all(a["split"] == split and a["role"] == first["role"]
                and a["remaining"] == first["remaining"] and a["ancestry"]["reset_seed"] == root for a in aliases),
                "alias crosses root/role/horizon/split")
        require(first["remaining"] in (100,500,1000) and bool(healthy(first["state"])), "invalid active curriculum snapshot")
        same(first["state"][0],.008*(1000-first["remaining"]),"catalog clock mismatch",atol=1e-10,rtol=0)
        output[split].append(dict(id="canonical:"+d,snapshot_digest=d,state=first["state"],
            warmstart=first["warmstart"],remaining=first["remaining"],role=first["role"],split=split,
            aliases=[a["id"] for a in aliases],ancestry=dict(reset_seed=root,
                source_context_ids=[a["id"] for a in aliases],source_ancestry=[a["ancestry"] for a in aliases])))
    for split, count, horizon, roots in (("train",24,10900,set(range(30,35))), ("heldout",25,11000,set(range(35,40)))):
        output[split].sort(key=lambda c:(c["ancestry"]["reset_seed"],-c["remaining"],c["id"]))
        require(len(output[split]) == count and sum(c["remaining"] for c in output[split]) == horizon
                and {c["ancestry"]["reset_seed"] for c in output[split]} == roots, "canonical context composition mismatch")
    return output


class MissingEvidence(Exception):
    pass


class VerificationJournal:
    """Create-only/fsynced attempt ledger BEFORE each NumPy forward batch."""
    def __init__(self,path,cap=1836144):
        self.path=Path(path);self.cap=cap;self.rows=0;self.seq=0;self.attempts=0
        self.file=self.path.open("x",encoding="utf-8")
        descriptor=os.open(self.path.parent,os.O_RDONLY)
        try:os.fsync(descriptor)
        finally:os.close(descriptor)
        self.emit("audit_begin",cap_rows=cap,new_physics_controls=0)

    def emit(self,kind,**values):
        row=dict(sequence=self.seq,kind=kind,**values)
        self.file.write(json.dumps(row,separators=(",",":"),allow_nan=False)+"\n")
        self.file.flush();os.fsync(self.file.fileno());self.seq+=1

    def reserve(self,owner,label,obs):
        require(self.rows+len(obs)<=self.cap,"durable independent audit inference cap exhausted")
        identifier=f"v{self.attempts:07d}"
        a=np.ascontiguousarray(obs,dtype=np.float64)
        self.emit("verification_attempt",verification_id=identifier,owner_hash=owner,
            stage=label,requested_rows=len(a),cumulative_requested_rows=self.rows+len(a),
            input_dtype=str(a.dtype),input_shape=list(a.shape),
            input_sha256=digest(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()))
        self.rows+=len(a);self.attempts+=1
        return identifier

    def close(self,status):
        self.emit("audit_end",status=status,requested_rows=self.rows,attempts=self.attempts)
        self.file.close()


class Auditor:
    def __init__(self, root, verify_actions, source_root=None, manifest=None,verification_journal=None):
        self.root = Path(root).resolve(); self.verify_actions = verify_actions
        self.source_root = None if source_root is None else Path(source_root).resolve()
        self.manifest = None if manifest is None else Path(manifest).resolve()
        self.errors, self.pending, self.inputs = [], [], {}
        self.actors, self.parameters, self.results, self.query_journals = {}, {}, {}, {}
        self.seen_journal_refs = set()
        self.audit_inference_rows, self.action_max_error = 0, 0.
        self.accounting = Counter(); self.query_details, self.seed_metrics = [], []
        self.round_metrics, self.science_failures = [], []
        self.config, self.contexts, self.events, self.inferences = {}, {}, [], []
        self.calibrations, self.inference_attempts, self.inference_completed = {}, {}, {}
        self.can_verify_actions=False
        self.verification_journal=verification_journal;self.action_failed=False
        self.initial_owners=set()

    def inventory_paid_work(self):
        """Count durable paid-attempt records even when later validation fails."""
        observed=Counter()
        files=list((self.root/"queries").glob("*/journal.jsonl"))
        for path in files+[self.root/"events.jsonl",self.root/"inferences.jsonl"]:
            if not path.is_file():continue
            for index,line in enumerate(path.read_bytes().splitlines(),1):
                try:row=json.loads(line)
                except (ValueError,UnicodeDecodeError):
                    observed["unparsed_journal_lines"]+=1;continue
                if not isinstance(row,dict):
                    observed["unparsed_journal_lines"]+=1;continue
                kind=row.get("kind")
                if path.name=="journal.jsonl":
                    observed["observed_requested_control_slots"]+=int(kind=="control_attempt")
                    observed["observed_control_completion_records"]+=int(kind=="control_complete")
                    observed["observed_control_error_records"]+=int(kind=="control_error")
                    observed["observed_reset_attempts"]+=int(kind=="reset_attempt")
                elif path.name=="inferences.jsonl":
                    observed["observed_inference_attempt_rows"]+=int(kind=="inference_attempt")
                    observed["observed_inference_completion_records"]+=int(kind=="inference_complete")
                    observed["observed_inference_error_records"]+=int(kind=="inference_error")
                elif kind=="query_reserved":
                    observed["observed_reserved_queries"]+=1
                    if type(row.get("remaining")) is int and row["remaining"]>=0:
                        observed["observed_reserved_controls"]+=row["remaining"]
                    else:observed["unknown_reservation_horizons"]+=1
        self.accounting.update(observed)
        self.accounting["charged_requested_controls"]=observed["observed_requested_control_slots"]
        self.accounting["reserved_controls"]=observed["observed_reserved_controls"]
        self.accounting["reserved_queries"]=observed["observed_reserved_queries"]
        self.accounting["inference_attempts"]=observed["observed_inference_attempt_rows"]
        require(observed["observed_requested_control_slots"]<=1832400
                and observed["observed_reserved_controls"]<=1832400
                and observed["observed_reserved_queries"]<=4044
                and observed["observed_inference_attempt_rows"]<=1836144,"observed paid-work hard cap exceeded")

    def input_obj(self, relative):
        expected = self.preflight["verified_files"].get(relative)
        require(expected is not None,"input was not in frozen preflight: "+relative)
        local = safe_path(self.root,relative)
        if local.is_file():
            return json.loads(self.data(relative,expected))
        if self.source_root is None:
            raise MissingEvidence("provide --source-root for frozen input "+relative)
        path = safe_path(self.source_root,relative)
        data = path.read_bytes(); require(digest(data) == expected,"frozen packet input SHA mismatch")
        self.inputs["packet:"+relative] = digest(data)
        return json.loads(data)

    def guard(self, label, function):
        try:
            return function()
        except (MissingEvidence,FileNotFoundError) as e:
            self.pending.append(f"{label}: {e}")
        except Exception as e:
            self.errors.append(f"{label}: {type(e).__name__}: {e}")
        return None

    def data(self, relative, expected=None):
        path = safe_path(self.root,relative)
        if not path.is_file():
            raise MissingEvidence(relative)
        raw = path.read_bytes(); h = digest(raw)
        if expected is not None:
            require(h == expected,"artifact SHA mismatch: "+relative)
        self.inputs[relative] = h
        return raw

    def obj(self, relative):
        return json.loads(self.data(relative))

    def arrays(self, relative, expected=None):
        with np.load(io.BytesIO(self.data(relative,expected)),allow_pickle=False) as archive:
            return {name:np.array(archive[name],copy=True) for name in archive.files}

    def actor(self, owner):
        if owner not in self.actors:
            metadata = self.obj(f"actors/{owner}.json")
            arrays = self.arrays(f"actors/{owner}.npz")
            require(metadata["raw_path"]==f"actors/{owner}.npz"
                    and metadata["raw_sha256"]==self.inputs[f"actors/{owner}.npz"],"actor NPZ metadata/hash mismatch")
            require(metadata["class_identity"] == CLASS_ID and metadata["owner_hash"] == owner,
                    "actor class/owner metadata mismatch")
            require(metadata["parameter_names"] == list(PARAMETERS), "actor parameter flatten order mismatch")
            theta = validate_actor(arrays,hidden=self.config["hidden_dim"])
            a=np.ascontiguousarray(theta)
            require(metadata["parameter_sha256"]==digest(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()),"flattened parameter digest mismatch")
            require(metadata["parameter_count"]==len(theta) and metadata["hidden_dim"]==256
                    and metadata["policy_type"]=="tanh_mlp" and metadata["observation_transform"]=="identity"
                    and metadata["parameter_shapes"]==[list(arrays[n].shape) for n in PARAMETERS],
                    "actor architecture/transform metadata differs from tensors")
            require(owner_hash(arrays,metadata["class_identity"]) == owner,"complete actor tensor/buffer hash mismatch")
            self.actors[owner],self.parameters[owner] = arrays,theta
        return self.actors[owner]

    def verify(self, owner, obs, actions, label):
        arrays = self.actor(owner)
        if not self.verify_actions or not self.can_verify_actions or self.action_failed or self.errors:
            return
        require(self.verification_journal is not None,"action verification requires durable create-only inference journal")
        for start in range(0,len(obs),256):
            x=np.asarray(obs[start:start+256],dtype=np.float64)
            require(self.verification_journal.rows+len(x)<=self.config["max_inference_attempts"],
                    "independent verification exceeds fixed assay inference cap")
            identifier=self.verification_journal.reserve(owner,label,x)
            self.audit_inference_rows=self.verification_journal.rows
            forward_started=time.perf_counter()
            try:
                predicted=numpy_actions(arrays,x,batch_size=256)
                error=float(np.max(abs(predicted-np.asarray(actions[start:start+len(x)]))))
                self.action_max_error=max(self.action_max_error,error)
                self.verification_journal.emit("verification_complete",verification_id=identifier,
                    requested_rows=len(x),maximum_absolute_error=error,within_tolerance=error<=2e-6,
                    forward_wall_seconds=time.perf_counter()-forward_started,
                    predicted_actions_sha256=digest(np.ascontiguousarray(predicted).tobytes()))
            except BaseException as e:
                self.action_failed=True
                self.verification_journal.emit("verification_error",verification_id=identifier,
                    requested_rows=len(x),error_type=type(e).__name__,error=str(e))
                raise
            if error>2e-6:
                self.action_failed=True
                raise ValueError(label+": recorded action is not the named actor action")

    def load_config(self):
        self.config = self.obj("config.json")
        self.preflight = self.obj("preflight.json")
        fixed = dict(schema="walker-owned-signal-v1",seeds=[0,1,2],rounds=3,directions=8,hidden_dim=256,
            gamma=.999,total_steps=1000,proposal_radius=.249975,max_step_norm=.25,
            selection_min_improvement=.002,selection_root_regression_tolerance=0,
            action_rms_min=.01,action_rms_max=.20,min_effective_samples=12,saturation_threshold=.98,
            max_saturation_fraction=.1,max_reserved_controls=1832400,max_queries=4044,
            max_inference_attempts=1836144,expected_train_roots=list(range(30,35)),expected_held_roots=list(range(35,40)))
        require(all(self.config.get(k) == v and not (type(v) in (int,float) and type(self.config.get(k)) is bool)
                    for k,v in fixed.items()),"unreviewed fixed configuration")
        v = self.config["verification"]
        require(v["final_mixture_gain"] == .01 and v["increment_gain"] == .002
                and v["min_positive_increments"] == 2 and v["root_regression_tolerance"] == 0
                and v["min_improved_root_clusters"] == 4 and v["final_root_gain"] == .002,
                "unreviewed final signal criterion")
        gate = self.input_obj(self.config["qualification_gate_path"])
        require(gate.get("ready") is True and gate.get("phase") == "PASS", "prior acquisition gate did not pass")
        require(all(gate.get("proof",{}).get(k) is True for k in ("source_identity_verified","raw_stream_binding_verified",
                "runtime_metadata_verified","fully_expanded_config_verified")),"prior acquisition certificate lacks source/raw/runtime proof")
        require(self.input_obj(self.config["task_path"]) == TASK,"actual cost/task configuration differs")
        catalog = self.input_obj(self.config["catalog_path"])
        expected = canonical_contexts(catalog)
        actual = self.obj("contexts.json")
        require(canonical(actual) == canonical(expected),"canonical context roster differs from all60 original slots")
        self.contexts = actual
        self.context_by_id = {c["id"]:c for split in ("train","heldout") for c in actual[split]}

    def verify_source_packet(self):
        if self.source_root is None or self.manifest is None:
            self.pending.append("independent frozen source verification needs --source-root and --manifest")
            return
        raw = self.manifest.read_bytes(); manifest = json.loads(raw)
        require(digest(raw) == self.preflight["manifest_sha256"],"frozen manifest differs from execution preflight")
        require(manifest["schema"] == "walker-owned-signal-manifest-v1"
                and manifest["files"] == self.preflight["verified_files"],"frozen file set differs from execution preflight")
        for relative, expected in manifest["files"].items():
            require(digest(safe_path(self.source_root,relative).read_bytes()) == expected,"source/input changed since freeze: "+relative)
        auditor_relative="scripts/diagnostics/audit_walker_owned_signal.py"
        require(manifest["files"].get(auditor_relative)==digest(Path(__file__).read_bytes()),"auditor itself is not the reviewed frozen version")
        self.inputs["frozen_manifest"] = digest(raw)
        require(self.preflight["packages"] == self.config["runtime"]["packages"],"execution package versions differ")
        require(self.preflight["backend"] == "mujoco_cpu" and self.preflight["frame_skip"] == 4
                and self.preflight["control_dt"] == .008,"execution backend differs")
        terminal=self.obj("status.json")
        self.can_verify_actions=terminal.get("phase")=="COMPLETED_TENTATIVE_BLIND_CHECK"
        if self.verify_actions and not self.can_verify_actions:
            self.pending.append("independent action inference waits for completed closed assay")

    def load_events(self):
        self.events, partial = jsonl(safe_path(self.root,"events.jsonl"))
        if partial:
            self.pending.append("incomplete final global event line")
        require([r["event_seq"] for r in self.events] == list(range(len(self.events))),"global event sequence mismatch")
        reservations = [r for r in self.events if r["kind"] == "query_reserved"]
        self.reservations = {}
        total = 0
        for index,event in enumerate(reservations,1):
            ref = event["query_ref"]
            require(ref not in self.reservations and type(event["remaining"]) is int and event["remaining"] >= 1,
                    "duplicate/invalid query reservation")
            total += event["remaining"]
            require(event["cumulative_queries"] == index and event["cumulative_reserved_controls"] == total,
                    "reservation cumulative counters disagree")
            self.reservations[ref] = event
        require(total <= self.config["max_reserved_controls"] and len(reservations) <= self.config["max_queries"],"reserved query/control cap exceeded")
        self.accounting["reserved_controls"]=total;self.accounting["reserved_queries"]=len(reservations)
        freezes = [e for e in self.events if e["kind"] == "all_chains_frozen"]
        require(len(freezes) <= 1,"multiple immutable chain-freeze boundaries")
        self.freeze = freezes[0] if freezes else None

    def load_queries(self):
        paths = sorted((self.root/"queries").glob("*/journal.jsonl"))
        for path in paths:
            relative = str(path.relative_to(self.root)); ref = path.parent.name
            def one():
                rows, partial = jsonl(path)
                self.inputs[relative] = digest(path.read_bytes())
                # Charge observed attempted slots before outcome/owner/trace
                # validation. A corrupt result must never erase paid work.
                self.seen_journal_refs.add(ref)
                require(rows,"empty query journal")
                owner,context_id = rows[0]["owner_hash"],rows[0]["context_id"]
                context = self.context_by_id[context_id]
                require(ref in self.reservations and self.reservations[ref]["remaining"] == context["remaining"],"unreserved query attempt")
                result_path = path.parent/"result.json"
                result,trace = None,None
                if result_path.exists():
                    result = self.obj(str(result_path.relative_to(self.root)))
                    require(result["query_ref"] == ref and result["owner_hash"] == owner
                            and result["context_id"] == context_id
                            and canonical(result["context_ancestry"]) == canonical(context["ancestry"]),"result/journal ownership mismatch")
                    require(result["gamma"] == self.config["gamma"] and result["phase"] in ("selection_complete","blind_verification"),"wrong query phase/objective")
                    require(result["actor_path"] == f"actors/{owner}.npz", "query actor path does not name exact owner")
                    self.data(result["actor_path"],result["actor_sha256"])
                    require(result["journal_path"] == relative,"query journal path mismatch")
                    self.data(relative,result["journal_sha256"])
                    trace = audit_trace(self.arrays(result["raw_path"],result["raw_sha256"]),result,context)
                    require(result["reservation_event_seq"] == self.reservations[ref]["event_seq"],"query reserved-sequence mismatch")
                    require(result["ledger"]["policy_controls"] == trace["executed_steps"] and all(v == 0 for k,v in result["ledger"].items() if k.startswith("teacher_")),"foreign controller/query ledger mismatch")
                    expected_split = "heldout" if result["phase"] == "blind_verification" else "train"
                    require(context["split"] == expected_split,"selection/held context leak")
                    if expected_split == "heldout":
                        require(self.freeze is not None and self.reservations[ref]["global_event_seq"] > self.freeze["global_event_seq"],"held query preceded immutable chain freeze")
                    self.results[ref] = dict(result,cost=trace["cost"])
                stats = audit_journal(rows,query_ref=ref,owner=owner,context=context,trace=trace)
                if trace is not None:
                    require(stats["reset_attempts"]==stats["reset_completions"]==1,"successful query lacks exactly one recorded snapshot restoration")
                if partial or not stats["complete"]:
                    self.pending.append("unfinished/failed query "+ref)
                self.query_journals[ref] = dict(rows=rows,stats=stats,context=context,owner_hash=owner)
                self.accounting.update({k:v for k,v in stats.items() if k != "charged_requested_controls"
                                       and type(v) in (int,float) and type(v) is not bool})
                if result:
                    for key,value in result["journal_stats"].items():
                        if key in stats:
                            require(value == stats[key],"reported journal counter mismatch: "+key)
                attempts = [r for r in rows if r["kind"] == "control_attempt"]
                obs = observations(np.asarray([r["pre"]["state"] for r in attempts])) if attempts else np.empty((0,17))
                actions = np.asarray([r["action"] for r in attempts]).reshape(-1,6)
                self.verify(owner,obs,actions,"query "+ref)
                reservation = self.reservations[ref]
                self.query_details.append(dict(query_ref=ref,owner_hash=owner,context_id=context_id,
                    seed=reservation.get("seed"),round=reservation.get("round"),phase=reservation.get("phase"),
                    role=context["role"],root_cluster=context["ancestry"]["reset_seed"],
                    reservation_global_event_seq=reservation["global_event_seq"],remaining=context["remaining"],
                    cost=None if result is None else result["cost"],
                    executed_cost=None if result is None else result["executed_cost"],
                    absorbing_cost=None if result is None else result["absorbing_cost"],
                    absorbing_steps=None if result is None else result["absorbing_steps"],
                    executed_steps=None if trace is None else trace["executed_steps"],
                    survived=None if trace is None else not trace["terminated"],
                    terminated=None if trace is None else trace["terminated"],
                    velocity_mae=None if trace is None or not trace["executed_steps"] else float(np.mean(abs(trace["states"][1:,10]-1.5))),
                    minhealthyheight_margin=None if trace is None else float(np.min(np.minimum(
                        trace["states"][healthy(trace["states"]),2]-.8,2.-trace["states"][healthy(trace["states"]),2]))),
                    minhealthyangle_margin=None if trace is None else float(np.min(1.-abs(trace["states"][healthy(trace["states"]),3]))),
                    final_height_margin=None if trace is None else float(min(trace["states"][-1,2]-.8,2.-trace["states"][-1,2])),
                    final_angle_margin=None if trace is None else float(1.-abs(trace["states"][-1,3])),
                    **stats))
            self.guard(relative,one)
        absent = set(self.reservations)-self.seen_journal_refs
        if absent:
            self.pending.append("reserved queries without auditable journals: "+",".join(sorted(absent)))
        for result in (self.root/"queries").glob("*/result.json"):
            require(result.parent.name in self.query_journals,"query result without journal")

    def load_inferences(self):
        self.inferences,partial = jsonl(safe_path(self.root,"inferences.jsonl"))
        if partial:
            self.pending.append("incomplete final inference record")
        require([r["event_seq"] for r in self.inferences] == list(range(len(self.inferences))),"inference file sequence mismatch")
        combined = sorted(r["global_event_seq"] for r in self.events+self.inferences)
        require(combined == list(range(len(combined))),"shared global event clock has gaps/duplicates")
        errors = {}
        for row in self.inferences:
            kind = row["kind"]; identifier = row["inference_id"]
            if kind == "inference_attempt":
                require(identifier == f"i{len(self.inference_attempts):09d}","inference reservation ID mismatch")
                self.inference_attempts[identifier] = row
            elif kind in ("inference_complete","inference_error"):
                require(identifier == row["attempt_id"] and identifier in self.inference_attempts
                        and identifier not in self.inference_completed and identifier not in errors,"unmatched/duplicate inference completion")
                attempt = self.inference_attempts[identifier]
                require(row["global_event_seq"] > attempt["global_event_seq"],"inference completes before attempt")
                for key in ("owner_hash","phase","context_id","query_ref","step_index","input_sha256","input_dtype","input_shape"):
                    require(row[key] == attempt[key],"inference owner/input metadata changed")
                (self.inference_completed if kind == "inference_complete" else errors)[identifier] = row
            else:
                raise ValueError("unknown inference journal record kind")
            context = self.context_by_id[row["context_id"]]
            if context["split"] == "heldout":
                require(row["phase"] == "blind_verification" and self.freeze is not None
                        and row["global_event_seq"] > self.freeze["global_event_seq"],"held inference occurred before whole-chain freeze")
        for key,value in dict(inference_attempts=len(self.inference_attempts),inference_completions=len(self.inference_completed),
            inference_errors=len(errors),inflight_inferences=len(self.inference_attempts)-len(self.inference_completed)-len(errors)).items():
            self.accounting[key]=value
        require(len(self.inference_attempts) <= self.config["max_inference_attempts"],"main inference cap exceeded")
        if errors or len(self.inference_completed) != len(self.inference_attempts):
            self.pending.append("failed/incomplete inference attempts retained")
        self.referenced_inferences = set()

    def check_inference(self, identifier, owner, context, obs, action, phase, query_ref, step_index):
        require(identifier in self.inference_completed,"recorded action lacks completed inference")
        row = self.inference_completed[identifier]
        def array_hash(value):
            a = np.ascontiguousarray(value); return digest(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())
        require(row["owner_hash"] == owner and row["context_id"] == context["id"]
                and row["phase"] == phase and row["query_ref"] == query_ref and row["step_index"] == step_index,
                "action inference owner/context/phase mismatch")
        require(row["input_dtype"] == "float64" and row["input_shape"] == [17]
                and row["input_sha256"] == array_hash(np.asarray(obs,dtype=np.float64)),"inference input digest mismatch")
        require(row["output_dtype"] == "float32" and row["output_shape"] == [6]
                and row["output_sha256"] == array_hash(np.asarray(action,dtype=np.float32)),"inference output digest mismatch")
        require(identifier not in self.referenced_inferences,"inference reused as a new recorded action")
        self.referenced_inferences.add(identifier)

    def load_calibrations(self):
        train = self.contexts["train"]
        ids = [c["id"] for c in train]
        obs = observations(np.asarray([c["state"] for c in train]))
        for path in sorted((self.root/"calibration").glob("*.json")):
            def one():
                meta = self.obj(str(path.relative_to(self.root))); owner = meta["owner_hash"]
                require(path.stem == owner and meta["context_ids"] == ids,"calibration roster/owner mismatch")
                raw = self.arrays(meta["raw_path"],meta["raw_sha256"])
                same(raw["observations"],obs,"calibration observations differ from selection states",atol=0,rtol=0)
                actions = finite_array(raw["actions"],(24,6))
                require(np.all(abs(actions) <= 1.) and len(meta["inference_ids"]) == 24,"invalid calibration actions/inference count")
                for i,identifier in enumerate(meta["inference_ids"]):
                    self.check_inference(identifier,owner,train[i],obs[i],actions[i],"calibration",None,None)
                self.verify(owner,obs,actions,"calibration "+owner)
                self.calibrations[owner] = actions
            self.guard(str(path.relative_to(self.root)),one)
        self.accounting["calibration_rows"] = 24*len(self.calibrations)
        for ref,entry in self.query_journals.items():
            def one_query():
                attempts = [r for r in entry["rows"] if r["kind"] == "control_attempt"]
                result = self.results.get(ref)
                if result is None:
                    return
                ids = result["inference_ids"]
                require(len(ids) == len(attempts),"query has missing/additional action inferences")
                for i,(identifier,attempt) in enumerate(zip(ids,attempts)):
                    self.check_inference(identifier,entry["owner_hash"],entry["context"],observations(attempt["pre"]["state"]),
                        attempt["action"],result["phase"],ref,i)
                    require(self.inference_completed[identifier]["global_event_seq"] > self.reservations[ref]["global_event_seq"],"action inference preceded query reservation")
            self.guard("query inference "+ref,one_query)
        unreferenced = set(self.inference_completed)-self.referenced_inferences
        if unreferenced:
            self.pending.append(f"{len(unreferenced)} completed inference rows lack saved raw action evidence")

    def panel(self, row, contexts):
        require(row["context_ids"] == [c["id"] for c in contexts],"panel context order differs")
        computed = score_panel(row["query_refs"],row["owner_hash"],contexts,self.results,self.config["gamma"])
        same(row["score"],computed["score"],"reported panel mixture score mismatch",atol=1e-11)
        require(set(row["role_scores"]) == set(computed["role_scores"]),"panel roles mismatch")
        for role,value in computed["role_scores"].items():
            same(row["role_scores"][role],value,"reported role mean mismatch",atol=1e-11)
        for key in ("normalized_returns","complete_returns"):
            if key in row:
                expected = computed["normalized_returns"] if key == "normalized_returns" else [self.results[r]["cost"] for r in row["query_refs"]]
                same(row[key],expected,"reported percontext returns mismatch",atol=1e-9)
        return computed

    def samples(self, seed, index, base):
        prefix = f"banks/seed{seed}_round{index}_samples"
        meta = self.obj(prefix+".json"); raw = self.arrays(meta["raw_path"],meta["raw_sha256"])
        require(meta["seed"] == seed and meta["round"] == index and meta["base_owner"] == base
                and meta["proposal_seed"] == seed*100000+index and meta["radius"] == self.config["proposal_radius"],"sample bank identity/proposal seed differs")
        require(meta["context_ids"] == [c["id"] for c in self.contexts["train"]],"proposal action calibration context mismatch")
        self.actor(base); theta = self.parameters[base]
        same(raw["base_parameters"],theta,"bank base differs from named actor",atol=0,rtol=0)
        directions = finite_array(raw["directions"],(8,len(theta)))
        rng = np.random.default_rng(seed*100000+index)
        require(canonical(meta["rng_pre"]) == canonical(rng.bit_generator.state),"proposal RNG prestate mismatch")
        pieces=[]
        for name in PARAMETERS:
            block = rng.standard_normal((8,self.actors[base][name].size))
            pieces.append(block/np.linalg.norm(block,axis=1)[:,None]/math.sqrt(6.))
        expected_directions = np.concatenate(pieces,axis=1)
        same(directions,expected_directions,"proposal directions not frozen private block-balanced draw",atol=1e-15,rtol=1e-14)
        require(canonical(meta["rng_post"]) == canonical(rng.bit_generator.state),"proposal RNG poststate mismatch")
        intended = np.empty((16,len(theta)))
        intended[0::2] = theta+self.config["proposal_radius"]*directions
        intended[1::2] = theta-self.config["proposal_radius"]*directions
        same(raw["intended_parameters"],intended,"antithetic intended parameter mismatch",atol=0,rtol=0)
        actual = intended.astype(np.float32).astype(np.float64)
        same(raw["sample_parameters"],actual,"serialized float32 sample mismatch",atol=0,rtol=0)
        owners = meta["owner_hashes"]; require(len(owners) == 16,"sample owner bank incomplete")
        for owner,parameters in zip(owners,actual):
            self.actor(owner); same(self.parameters[owner],parameters,"sample parameters differ from owner",atol=0,rtol=0)
        norms = np.linalg.norm(actual-theta[None],axis=1)
        require(np.all(norms <= self.config["max_step_norm"]),"sample violates common parameter bound")
        same(raw["step_norms"],norms,"sample norm accounting mismatch")
        require(base in self.calibrations and all(o in self.calibrations for o in owners),"sample calibration evidence missing")
        base_actions = self.calibrations[base]; actions = np.stack([self.calibrations[o] for o in owners])
        same(raw["base_actions"],base_actions,"base calibration differs",atol=0,rtol=0)
        same(raw["sample_actions"],actions,"sample calibration differs",atol=0,rtol=0)
        rms = np.sqrt(np.mean((actions-base_actions[None])**2,axis=(1,2)))
        sat = np.mean(abs(actions)>=self.config["saturation_threshold"],axis=(1,2))
        effective = (rms>=self.config["action_rms_min"]) & (rms<=self.config["action_rms_max"])
        passed = bool(effective.sum()>=self.config["min_effective_samples"] and np.all(sat<=self.config["max_saturation_fraction"]))
        same(raw["action_rms"],rms,"action-RMS distribution statistic mismatch")
        same(raw["saturation_fractions"],sat,"peractor saturation statistic mismatch")
        dist = meta["distribution"]
        require(dist["passes"] is passed and dist["effective_count"] == int(effective.sum()),"distribution gate differs from raw actor actions")
        same(dist["action_rms"],rms,"JSON action-RMS mismatch")
        same(dist["saturation_fractions"],sat,"JSON saturation mismatch")
        return dict(owners=owners,theta=theta,directions=directions,passed=passed,
                    action_rms=rms,saturation_fractions=sat,metadata=meta)

    def train_round(self, seed, index, base, path):
        row = self.obj(path); bank = self.samples(seed,index,base)
        require(row["seed"] == seed and row["round"] == index and row["base_owner"] == base,
                "round ancestry does not follow tentative chain")
        if not bank["passed"]:
            require(row["status"] == "DISTRIBUTION_FAILURE" and row["roster"] == []
                    and row["tentative_changed"] is False and row["checkpoint_owner"] == base,
                    "unsupported proposal distribution was used for actor search")
            self.round_metrics.append(dict(seed=seed,round=index,base_owner=base,distribution_passed=False))
            return base,True
        require(row["status"] == "TENTATIVE" and row["phase"] == "selection_complete", "valid distribution lacks complete selection bank")
        esmeta = self.obj(f"banks/seed{seed}_round{index}_es.json")
        esraw = self.arrays(esmeta["raw_path"],esmeta["raw_sha256"])
        esowner = esmeta["owner_hash"]; self.actor(esowner)
        labels = [("base",-1,base)]+[("sample",i,o) for i,o in enumerate(bank["owners"])]+[("es",-1,esowner)]
        expected,lookup = [],{}
        for kind,ordinal,owner in labels:
            if owner in lookup:
                expected[lookup[owner]]["duplicate_aliases"].append(dict(kind=kind,index=ordinal))
            else:
                lookup[owner]=len(expected)
                expected.append(dict(kind=kind,index=ordinal,owner_hash=owner,duplicate_aliases=[]))
        roster = row["roster"]
        require(len(roster)==len(expected),"sample/base/ES roster incomplete or expanded")
        computed=[]
        for item,wanted in zip(roster,expected):
            require(all(item.get(k)==v for k,v in wanted.items()),"owner alias or proposal order incorrect")
            require(item["phase"] == "selection_complete","nonselection data ranks tentative changes")
            computed.append(self.panel(item,self.contexts["train"]))
        sample_scores = np.asarray([computed[lookup[o]]["score"] for o in bank["owners"]])
        same(esraw["sample_scores"],sample_scores,"ES lacks independently verified fullbank scores",atol=1e-11)
        operational_sample_scores=np.asarray(esraw["sample_scores"],dtype=float)
        same(operational_sample_scores,[roster[lookup[o]]["score"] for o in bank["owners"]],
             "ES score inputs differ from actual measured roster scores",atol=0,rtol=0)
        combination = -np.mean((operational_sample_scores[0::2]-operational_sample_scores[1::2])[:,None]*bank["directions"],axis=0)
        norm = float(np.linalg.norm(combination))
        intended = bank["theta"] if norm == 0 else bank["theta"]+self.config["proposal_radius"]*combination/norm
        actual = intended.astype(np.float32).astype(np.float64)
        for got,want,label in ((esraw["base_parameters"],bank["theta"],"base"),
             (esraw["combination"],combination,"combination"),
             (esraw["es_parameters"],actual,"serialized proposal")):
            same(got,want,"ES "+label+" mismatch",atol=1e-12,rtol=1e-10)
        same(self.parameters[esowner],actual,"ES actor differs from own proposal",atol=0,rtol=0)
        stepnorm = float(np.linalg.norm(actual-bank["theta"]))
        require(stepnorm<=self.config["max_step_norm"],"synthesized ES proposal violates same parameter cap")
        same(esmeta["step_norm"],stepnorm,"ES reported step norm differs")
        require(esmeta["zero_combination"] is (norm==0),"ES zero-combination flag mismatch")
        duplicate = next(("base" if kind=="base" else f"sample:{ordinal}"
                          for kind,ordinal,owner in labels[:-1] if owner==esowner),None)
        require(esmeta["duplicate_of"]==duplicate,"duplicate ES actor not explicitly identified")
        same(esraw["paired_differences"],operational_sample_scores[0::2]-operational_sample_scores[1::2],
             "ES paired cost differences mismatch",atol=0,rtol=0)
        same(esraw["es_actions"],self.calibrations[esowner],"ES calibration action mismatch",atol=0,rtol=0)
        # Operational numbers are independently checked above; use their exact
        # serialized values to reconstruct ties and float32 optimizer rounding.
        scores = np.asarray([p["score"] for p in roster])
        roots = np.asarray([p["role_scores"]["root"] for p in roster])
        selection=select_feasible(scores,roots,self.config["selection_root_regression_tolerance"])
        feasible=selection["feasible_flags"];selected=selection["selected_index"]
        base_score,best = computed[0],computed[selected]
        gain = float(scores[0]-scores[selected])
        regression = float(roots[selected]-roots[0])
        changed = bool(gain>=self.config["selection_min_improvement"] and regression<=self.config["selection_root_regression_tolerance"])
        next_owner = roster[selected]["owner_hash"] if changed else base
        require(row["selected_index"]==selected and row["selected_owner"]==roster[selected]["owner_hash"]
                and row["selected_kind"]==roster[selected]["kind"] and row["tentative_changed"] is changed
                and row["checkpoint_owner"]==next_owner,"argmin/trainmargin/root-nonregression selection mismatch")
        require(row.get("feasible_flags")==feasible.tolist(),"feasible-before-argmin eligibility mismatch")
        require(row.get("unconstrained_selected_index")==selection["unconstrained_selected_index"],"unconstrained diagnostic argmin mismatch")
        for item,eligible in zip(roster,feasible):
            self.actor(item["owner_hash"])
            distance=float(np.linalg.norm(self.parameters[item["owner_hash"]]-bank["theta"]))
            require(distance<=self.config["max_step_norm"] and item["eligible"] is bool(eligible),"roster eligibility differs from same bounded geometry/root constraint")
            same(item["step_norm"],distance,"roster serialized step norm differs")
        same(row["train_improvement"],gain,"reported train improvement differs",atol=1e-11)
        same(row["root_regression"],regression,"reported root regression differs",atol=1e-11)
        self.round_metrics.append(dict(seed=seed,round=index,base_owner=base,selected_owner=roster[selected]["owner_hash"],
            selected_kind=roster[selected]["kind"],checkpoint_owner=next_owner,tentative_changed=changed,
            base_score=base_score["score"],best_score=best["score"],es_score=computed[lookup[esowner]]["score"],
            sampled_best_score=float(sample_scores.min()),checkpoint_score=best["score"] if changed else base_score["score"],
            sampled_best_feasible_score=min((p["score"] for i,p in enumerate(computed)
                if roster[i]["kind"]=="sample" and feasible[i]),default=None),
            es_feasible=bool(feasible[lookup[esowner]]),selected_index=selected,
            unconstrained_selected_index=selection["unconstrained_selected_index"],
            unconstrained_best_score=float(scores.min()),feasible_flags=feasible.tolist(),
            train_improvement=gain,root_regression=regression,distribution_passed=True,
            action_rms=bank["action_rms"].tolist(),saturation_fractions=bank["saturation_fractions"].tolist()))
        return next_owner,False

    def chains_and_held(self):
        if self.freeze is None:
            raise MissingEvidence("all three chains have not been frozen")
        frozen = self.obj("chains_frozen.json")
        self.data("chains_frozen.json",self.freeze["freeze_sha256"])
        require(frozen["event_seq"] == self.freeze["event_seq"] and frozen["global_event_seq"] == self.freeze["global_event_seq"],"whole-chain immutable boundary mismatch")
        require([c["seed"] for c in frozen["chains"]] == self.config["seeds"],"whole-chain freeze omitted/replaced seed")
        seen_held_owners=set()
        for frozen_chain in frozen["chains"]:
            def one_seed():
                seed = frozen_chain["seed"]; self.data(frozen_chain["path"],frozen_chain["sha256"])
                chain = self.obj(frozen_chain["path"])
                require(chain["seed"]==seed and chain["tentative"] is True,"trusted/deployed or misidentified chain")
                points=chain["checkpoints"]
                require(points and points[0]["index"]==0,"initial checkpoint missing")
                base=points[0]["owner_hash"]; self.actor(base)
                require(base not in self.initial_owners,"training seeds reuse identical initial actor")
                self.initial_owners.add(base)
                failed=False
                for index,path in enumerate(chain["rounds"]):
                    require(index<3 and path==f"rounds/seed{seed}_round{index}.json","round omitted/reordered/expanded")
                    base,failed = self.train_round(seed,index,base,path)
                    if failed:
                        require(index==len(chain["rounds"])-1,"redraw/search after distribution failure")
                        break
                    require(points[index+1]["owner_hash"]==base and points[index+1]["index"]==index+1,"checkpoint does not follow selected chain")
                require(chain["distribution_failed"] is failed,"chain distribution failure flag differs")
                for point in points:
                    data = self.data(point["checkpoint_path"],point["checkpoint_sha256"])
                    record=json.loads(data)
                    require(record["owner_hash"]==point["owner_hash"] and record["index"]==point["index"]
                            and record["seed"]==seed and record["status"]=="tentative_not_deployed","checkpoint identity differs")
                    if point["index"]==0:
                        require(record["initialization_seed"]==seed,"initialization source seed differs")
                if failed:
                    self.science_failures.append(f"seed{seed}: distribution failure")
                    self.seed_metrics.append(dict(seed=seed,distribution_failed=True,passed=False,held=[]))
                    return
                require(len(chain["rounds"])==3 and len(points)==4,"full three-round chain missing")
                panels=[]
                for index,point in enumerate(points):
                    row=self.obj(f"held/seed{seed}_checkpoint{index}.json")
                    require(row["owner_hash"]==point["owner_hash"] and row["seed"]==seed
                            and row["checkpoint_index"]==index and row["phase"]=="blind_verification","held checkpoint identity/phase differs")
                    require(row["cache_alias"] is (point["owner_hash"] in seen_held_owners),"identical held checkpoint cache alias omitted or fabricated")
                    seen_held_owners.add(point["owner_hash"])
                    p=self.panel(row,self.contexts["heldout"])
                    panels.append(dict(index=index,owner_hash=point["owner_hash"],score=p["score"],role_scores=p["role_scores"],
                        root_clusters=p["root_clusters"],normalized_returns=p["normalized_returns"].tolist(),
                        query_refs=row["query_refs"],cache_alias=row["cache_alias"]))
                gain=panels[0]["score"]-panels[-1]["score"]
                rootgain=panels[0]["role_scores"]["root"]-panels[-1]["role_scores"]["root"]
                increments=[a["score"]-b["score"] for a,b in zip(panels,panels[1:])]
                cluster_gains={k:panels[0]["root_clusters"][k]-panels[-1]["root_clusters"][k] for k in panels[0]["root_clusters"]}
                v=self.config["verification"]
                criteria=dict(final_mixture_gain=gain>=v["final_mixture_gain"],
                    repeatable_increment_gain=sum(g>=v["increment_gain"] for g in increments)>=v["min_positive_increments"],
                    final_root_nonregression=rootgain>=-v["root_regression_tolerance"],
                    final_root_gain=rootgain>=v["final_root_gain"],
                    improved_root_clusters=sum(g>0 for g in cluster_gains.values())>=v["min_improved_root_clusters"])
                passed=all(criteria.values())
                if not passed:self.science_failures.append(f"seed{seed}: held development criteria failed")
                self.seed_metrics.append(dict(seed=seed,distribution_failed=False,passed=passed,held=panels,
                    final_mixture_gain=gain,final_root_gain=rootgain,increment_gains=increments,
                    root_cluster_gains=cluster_gains,criteria=criteria))
            self.guard("seed chain "+str(frozen_chain["seed"]),one_seed)

    def verify_event_artifacts(self):
        kinds={"query_complete":("result_path","result_sha256"),
            "tentative_step":("record_path","record_sha256"),"chain_frozen":("chain_path","chain_sha256"),
            "held_panel_complete":("record_path","record_sha256"),"calibration_complete":("raw_path","raw_sha256"),
            "all_chains_frozen":("freeze_path","freeze_sha256")}
        completed={}; cache_hits=0
        for event in self.events:
            kind=event["kind"]
            if kind in kinds:
                pathkey,hashkey=kinds[kind]; self.data(event[pathkey],event[hashkey])
            if kind=="query_complete":
                ref=event["query_ref"]
                require(ref not in completed and ref in self.results,"query-complete event missing/duplicates raw result")
                result=self.results[ref]
                require(event["executed_controls"]==result["executed_steps"] and all(event[k]==result[k] for k in
                        ("owner_hash","context_id","phase","seed","round")),"query event/result identity or counts mismatch")
                require(event["global_event_seq"]>self.reservations[ref]["global_event_seq"],"query completes before reservation")
                completed[ref]=event
            elif kind=="query_cache_hit":
                cache_hits+=1; result=self.results[event["query_ref"]]
                require(all(event[k]==result[k] for k in ("owner_hash","context_id","phase")),"query cache aliases a different continuation/objective")
            if kind in ("tentative_step","chain_frozen"):
                require(self.freeze is not None and event["global_event_seq"]<self.freeze["global_event_seq"],"chain selected or changed after whole-chain freeze")
        require(set(completed)==set(self.results),"raw successful result omitted from global completed ledger")
        keys={}
        for ref,result in self.results.items():
            key=(result["owner_hash"],result["snapshot_digest"],result["remaining"],result["gamma"],result["phase"])
            require(key not in keys,"identical actor/context/objective executed again instead of explicit cache alias")
            keys[key]=ref
            reservation=self.reservations[ref]
            require(all(result[k]==reservation[k] for k in ("owner_hash","context_id","phase","seed","round")),"query reservation belongs to different owner/phase")
            require(result["reservation_global_event_seq"]==reservation["global_event_seq"],"query shared-clock reservation mismatch")
        self.accounting["query_cache_hits"]=cache_hits
        self.accounting["unique_completed_queries"]=len(self.results)
        if self.freeze:
            require({e["seed"] for e in self.events if e["kind"]=="chain_frozen"}==set(self.config["seeds"]),"not all three seeds had independent immutable chain records")
            for row in self.inferences:
                if row["phase"]=="calibration":
                    require(row["global_event_seq"]<self.freeze["global_event_seq"],"new calibration after whole-chain freeze")

    def status_and_totals(self):
        self.status=self.obj("status.json")
        require(self.status.get("trusted_deployment") is False and self.status.get("verdict")=="NOT_ASSIGNED",
                "runner assigned a trusted/deployed or self-certified actor")
        budget=self.status["budget"]
        for key,actual in (("reserved_controls",self.accounting["reserved_controls"]),
            ("reserved_queries",self.accounting["reserved_queries"]),("inference_attempts",self.accounting["inference_attempts"])):
            require(budget[key]==actual,"status/global paid-work counter mismatch: "+key)
        require(self.status["completed_queries"]==len(self.results)
                and self.status["completed_controls"]==sum(q["executed_steps"] for q in self.results.values())
                and self.status["inference_completions"]==len(self.inference_completed),"status successful-return totals differ from raw evidence")
        require(self.accounting["charged_requested_controls"]<=self.accounting["reserved_controls"],"physical attempts exceed nonreallocatable reservation cap")
        phase=self.status["phase"]
        self.finished=phase=="COMPLETED_TENTATIVE_BLIND_CHECK"
        if phase=="FAILED_PRESERVED_NO_RETRY":
            self.errors.append("runner stopped on failure; no retry or complete actor-signal evidence")
        elif not self.finished:
            self.pending.append("assay has not reached its closed terminal status")
        if self.finished:
            require(sum(e["kind"]=="assay_complete" for e in self.events)==1
                    and not any(e["kind"]=="assay_failure" for e in self.events),"terminal marker/global completion mismatch")
        self.query_details.sort(key=lambda r:r["reservation_global_event_seq"])
        cumulative=0; byseed={}; byphase={}
        for row in self.query_details:
            cumulative+=row["charged_requested_controls"]
            row["cumulative_charged_controls"]=cumulative
            for group,key in ((byseed,str(row["seed"])),(byphase,str(row["phase"]))):
                if key not in group:group[key]=dict(queries=0,requested_controls=0,completed_controls=0,reserved_controls=0)
                group[key]["queries"]+=1
                group[key]["requested_controls"]+=row["charged_requested_controls"]
                group[key]["completed_controls"]+=row["completed_controls"]
                group[key]["reserved_controls"]+=row["remaining"]
        self.by_seed,self.by_phase=byseed,byphase

    def run(self):
        started=time.perf_counter(); self.finished=False; self.status={}; self.by_seed={}; self.by_phase={}
        self.guard("unconditional durable paid-work inventory",self.inventory_paid_work)
        self.guard("configuration and raw context inputs",self.load_config)
        if self.config and self.contexts:
            self.guard("immutable source packet",self.verify_source_packet)
            self.guard("global query reservations",self.load_events)
            if hasattr(self,"reservations"):
                self.guard("all attempted query journals",self.load_queries)
                self.guard("all inference attempts",self.load_inferences)
                if hasattr(self,"referenced_inferences"):
                    self.guard("calibration/action inference bindings",self.load_calibrations)
                self.guard("global artifact hashes and aliases",self.verify_event_artifacts)
                self.guard("frozen tentative chains and held development panels",self.chains_and_held)
                self.guard("terminal status and cumulative paid work",self.status_and_totals)
        signal_passed=(len(self.seed_metrics)==3 and all(s["passed"] for s in self.seed_metrics))
        if self.finished and len(self.seed_metrics)!=3:
            self.pending.append("all three seed verdicts are not independently available")
        if not self.verify_actions:
            self.pending.append("--verify-actions required for final owner-action certificate")
        elif not self.can_verify_actions:
            self.pending.append("action inference withheld until independent frozen-source verification succeeds")
        phase="FAIL" if self.errors else "PENDING" if self.pending or not self.finished else "PASS" if signal_passed else "FAIL"
        arithmetic_valid=not self.errors
        return dict(schema="walker-owned-signal-independent-audit-v1",phase=phase,ready=phase=="PASS",
            evidence_arithmetic_valid=arithmetic_valid,signal_criterion_passed=signal_passed,
            action_verification_requested=self.verify_actions,action_verification_rows=self.audit_inference_rows,
            maximum_action_absolute_error=self.action_max_error,action_tolerance=2e-6,
            reasons=self.errors+self.pending+self.science_failures,
            errors=self.errors,pending=self.pending,scientific_negative=self.science_failures,
            work=dict(self.accounting),costs_by_seed=self.by_seed,costs_by_phase=self.by_phase,
            queries=self.query_details,rounds=self.round_metrics,seeds=self.seed_metrics,
            input_sha256=self.inputs,run_directory=str(self.root),
            audit_numpy_version=np.__version__,
            wall_seconds=dict(audit=time.perf_counter()-started,run=self.status.get("wall_seconds"),
                recorded_simulator_calls=self.accounting["simulator_call_wall_seconds"],
                runner_timing=self.status.get("timing"),runner_timing_fields_overlap=True),
            scope="Bounded actor-owned learning-signal and root-disjoint development-transfer assay only",
            limitations=["No independent MuJoCo integration or attestation of hidden/unlogged process calls.",
                "Inference verification uses saved tensors and observed raw inputs; all its rows are charged separately.",
                "Exact actor/context cache aliases are repeated use of the same evidence, not independent observations.",
                "Five held reset clusters per seed are a development panel, not standalone walking competence.",
                "No new MPPI feedback, standalone GPS policy success, exploration efficiency or critic result is established.",
                "Earlier acquisition/qualification/interruption costs remain separate; this assay does not reset them."])


def audit(run_dir, *, verify_actions=False, source_root=None, manifest=None,audit_inference_journal=None):
    total_started=time.perf_counter()
    if manifest is None and (Path(run_dir)/"frozen_manifest.json").is_file():
        manifest=Path(run_dir)/"frozen_manifest.json"
    require(not verify_actions or audit_inference_journal is not None,
            "--verify-actions/API verification needs a NEW durable audit inference journal")
    journal=VerificationJournal(audit_inference_journal) if verify_actions else None
    result=None
    try:
        result=Auditor(run_dir,verify_actions,source_root,manifest,journal).run()
        return result
    finally:
        if journal is not None:
            journal.close("interrupted_or_exception" if result is None else result["phase"])
            if result is not None:
                rows,partial=jsonl(journal.path)
                attempts=[r for r in rows if r["kind"]=="verification_attempt"]
                complete=[r for r in rows if r["kind"]=="verification_complete"]
                errors=[r for r in rows if r["kind"]=="verification_error"]
                result["audit_inference_journal"]=dict(path=str(journal.path.resolve()),sha256=digest(journal.path.read_bytes()),
                    requested_rows=sum(r["requested_rows"] for r in attempts),completed_rows=sum(r["requested_rows"] for r in complete),
                    error_rows=sum(r["requested_rows"] for r in errors),incomplete_final_line=partial,
                    inflight_rows=sum(r["requested_rows"] for r in attempts)-sum(r["requested_rows"] for r in complete+errors))
                result["wall_seconds"]["numpy_forward"]=sum(r.get("forward_wall_seconds",0.) for r in complete+errors)
        if result is not None:
            result["wall_seconds"]["audit_total_including_durable_journal"]=time.perf_counter()-total_started


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    parser.add_argument("--source-root",type=Path,help="exact frozen execution source/input packet")
    parser.add_argument("--manifest",type=Path,help="defaults to run_dir/frozen_manifest.json")
    parser.add_argument("--verify-actions",action="store_true",help="charge independent NumPy forward rows; required for final PASS")
    parser.add_argument("--out",type=Path,required=True,help="NEW JSON result path, never overwritten")
    args=parser.parse_args()
    require(not args.out.exists(),"audit output already exists; avoid an uncharged repeated forward audit")
    journal=args.out.with_suffix(".inferences.jsonl") if args.verify_actions else None
    require(journal is None or not journal.exists(),"audit inference journal already exists; interrupted verification cannot silently repeat")
    result=audit(args.run_dir,verify_actions=args.verify_actions,source_root=args.source_root,manifest=args.manifest,
                 audit_inference_journal=journal)
    with args.out.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps(dict(phase=result["phase"],ready=result["ready"],work=result["work"],
                         action_verification_rows=result["action_verification_rows"],output=str(args.out))))


if __name__=="__main__":
    main()
