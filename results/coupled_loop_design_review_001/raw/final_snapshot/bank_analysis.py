"""Saved-only design review. No simulator, network or policy fitting imports."""
from pathlib import Path
import csv
import hashlib
import json
import os
import resource
import sys
import time
import traceback

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"
import h5py
import numpy as np

BASE = Path(__file__).resolve().parent
START, CPU = time.monotonic(), time.process_time()
resource.setrlimit(resource.RLIMIT_CPU, (120, 121))
CONFIG = json.loads((BASE / "config.json").read_text())
OUT = BASE / "banks_and_fit"
OUT.mkdir(exist_ok=False)
BINDINGS = {}


def bound(spec):
    path = Path(spec["path"]).resolve()
    if str(path) not in BINDINGS:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != spec["bytes"] or digest != spec["sha256"]:
            raise ValueError("Input binding changed: " + str(path))
        BINDINGS[str(path)] = dict(bytes=path.stat().st_size, sha256=digest)
    return path


def read(spec):
    return json.loads(bound(spec).read_text())


def weights(c, d, alpha, lam):
    valid = np.isfinite(c)
    score = np.full(len(c), np.inf)
    score[valid] = c[valid] + alpha * d[valid]
    w = np.exp(-(score - np.min(score[valid])) / lam)
    return w / np.sum(w)


def regression(acc):
    n = acc["rows"]
    vn = acc["nn"] - acc["n"]**2/n
    vd = acc["dd"] - acc["d"]**2/n
    cv = acc["nd"] - acc["n"]*acc["d"]/n
    return dict(rows=n, slope=cv/vn if vn > 0 else None,
                r2=cv**2/(vn*vd) if vn > 0 and vd > 0 else None,
                within_bank_slope=acc["center_nd"]/acc["center_nn"] if acc["center_nn"] > 0 else None,
                within_bank_r2=acc["center_nd"]**2/(acc["center_nn"]*acc["center_dd"]) if acc["center_nn"] > 0 and acc["center_dd"] > 0 else None)


def add(acc, n, d):
    nn, dd = n - n.mean(), d - d.mean()
    for k, v in dict(rows=len(n), n=n.sum(), d=d.sum(), nn=np.sum(n*n), dd=np.sum(d*d), nd=np.sum(n*d), center_nn=np.sum(nn*nn), center_dd=np.sum(dd*dd), center_nd=np.sum(nn*dd)).items():
        acc[k] = acc.get(k, 0) + float(v)


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mse(diag):
    with np.load(bound(diag["data"]), allow_pickle=False) as data:
        target = data["actions"].copy()
    with np.load(bound(diag["predictions"]), allow_pickle=False) as pred:
        prediction = pred["predictions"].copy()
    measured = float(np.mean((prediction.astype(np.float64) - target)**2))
    return measured


try:
    rows, fits, accumulators = [], [], {}
    for stream in CONFIG["snapshot_roster"]:
        snapshot = read(stream["snapshot"])
        for block in snapshot["rounds"]:
            if block["status"] != "COMPLETED":
                continue
            r, s = block["round"], stream["stream"]
            for fit_row in block["fits"]:
                fit = read(fit_row["result"])
                row = dict(stream=s, round=r, arm=fit_row["arm"], updates=fit["updates"], cumulative_updates=fit["cumulative_updates"])
                for phase in ("before", "after"):
                    for role in ("own", "held", "anchor", "cross"):
                        row[f"{phase}_{role}_mse"] = mse(fit[f"diagnostics_{phase}"][role])
                fits.append(row)
            for episode in block["acquisitions"]:
                manifest = read(episode["manifest"])
                source = bound(manifest["bank_stats"])
                with h5py.File(source, "r") as banks:
                    for bank_id, g in banks.items():
                        c, d = g["C"][:], g["D"][:]
                        length, survived = g["lengths"][:], g["full_survival"][:]
                        valid = np.isfinite(c)
                        if not valid.any() or np.any(length[valid] <= 0) or not np.isfinite(d[valid]).all():
                            raise ValueError("Undefined saved executable-prefix bank")
                        alpha, lam = float(g.attrs["alpha"]), float(g.attrs["lambda_value"])
                        horizon = len(g["U_before"])
                        original = weights(c, d, alpha, lam)
                        mean_d = np.zeros(len(d))
                        mean_d[valid] = d[valid] / length[valid]
                        mean_w = weights(c, mean_d, alpha, lam)
                        matched_w = weights(c, horizon * mean_d, alpha, lam)
                        saved_w = g["weights"][:]
                        ratio = float(np.std(alpha*d[valid]) / np.std(c[valid])) if np.std(c[valid]) > 0 else None
                        row = dict(stream=s, round=r, arm=episode["arm"], slot=episode["slot"], bank=bank_id,
                                   source=str(source), control=int(g.attrs["control"]), inner=int(g.attrs["inner"]),
                                   candidates=len(c), finite_candidates=int(valid.sum()), alpha=alpha, temperature=lam,
                                   prefix_min=int(length[valid].min()), prefix_max=int(length[valid].max()),
                                   mean_prefix=float(length[valid].mean()), std_C=float(np.std(c[valid])), std_alphaD=float(np.std(alpha*d[valid])),
                                   strength_ratio=ratio, original_weight_max_error=float(np.max(np.abs(original-saved_w))),
                                   survival_fraction=float(survived.mean()), original_survivor_mass=float(original[survived].sum()),
                                   mean_survivor_mass=float(mean_w[survived].sum()), scaled_mean_survivor_mass=float(matched_w[survived].sum()),
                                   mean_survivor_mass_change=float(mean_w[survived].sum()-original[survived].sum()),
                                   scaled_mean_survivor_mass_change=float(matched_w[survived].sum()-original[survived].sum()),
                                   original_top=int(np.argmax(original)), mean_top=int(np.argmax(mean_w)), scaled_mean_top=int(np.argmax(matched_w)),
                                   mean_top_changed=int(np.argmax(original)!=np.argmax(mean_w)), scaled_mean_top_changed=int(np.argmax(original)!=np.argmax(matched_w)))
                        rows.append(row)
                        for key in (episode["arm"], f"{s}:{r}:{episode['arm']}"):
                            add(accumulators.setdefault(key, {}), length[valid].astype(float), d[valid])
                print(json.dumps(dict(stage="bank_file", stream=s, round=r, arm=episode["arm"], slot=episode["slot"], banks=len(rows), cpu_seconds=time.process_time()-CPU)), flush=True)
    csv_write(OUT / "per_bank.csv", rows)
    csv_write(OUT / "fit_mse.csv", fits)
    aggregates = []
    for key in sorted({(x["stream"], x["round"], x["arm"]) for x in rows}):
        selected = [x for x in rows if (x["stream"],x["round"],x["arm"]) == key]
        ratios = [x["strength_ratio"] for x in selected if x["strength_ratio"] is not None]
        aggregate = dict(stream=key[0], round=key[1], arm=key[2], banks=len(selected),
                         ratio_median=float(np.median(ratios)), ratio_p05=float(np.quantile(ratios,.05)), ratio_p95=float(np.quantile(ratios,.95)),
                         mean_top_changed=sum(x["mean_top_changed"] for x in selected), scaled_mean_top_changed=sum(x["scaled_mean_top_changed"] for x in selected),
                         mean_survivor_mass_change=float(np.mean([x["mean_survivor_mass_change"] for x in selected])),
                         scaled_mean_survivor_mass_change=float(np.mean([x["scaled_mean_survivor_mass_change"] for x in selected])))
        aggregates.append(aggregate)
    csv_write(OUT / "per_cell.csv", aggregates)
    regressions = {key: regression(value) for key,value in accumulators.items()}
    (OUT / "D_length_regressions.json").write_text(json.dumps(regressions, indent=2, allow_nan=False)+"\n")

    l2base = Path(CONFIG["original_run"]).parent / "2026-11-03_local_coupling_learning_l2_001"
    l2 = json.loads((l2base/"worker/result.json").read_text())
    outcomes = {}
    for entry in l2["acquisitions"]:
        m = read(entry["manifest"])
        with h5py.File(bound(m["trace"]), "r") as raw:
            n = len(raw["actions"])
            native = float(raw["costs"][:].sum())
            terminal = float(raw["native_fall_cost"][:].sum())
        outcomes[(m["arm"],m["slot"])] = dict(controls=n, native=native, terminal=terminal, tail=5*(300-n), complete=native+5*(300-n))
    pairs = []
    for slot in range(8):
        a, b = outcomes[("independent",slot)], outcomes[("weak",slot)]
        pairs.append(dict(slot=slot, independent_controls=a["controls"], coupled_controls=b["controls"],
                          independent_cost=a["complete"], coupled_cost=b["complete"], gain=a["complete"]-b["complete"],
                          terminal_gain=a["terminal"]-b["terminal"], absorbing_tail_gain=a["tail"]-b["tail"]))
    csv_write(OUT / "earlier_slot_costs.csv", pairs)
    positive = [x for x in rows if x["arm"] == "positive"]
    summary = dict(banks=len(rows), positive_banks=len(positive), candidate_rows=sum(x["finite_candidates"] for x in rows),
                   regressions=regressions, maximum_original_weight_error=max(x["original_weight_max_error"] for x in rows),
                   same_alpha=dict(changed=sum(x["mean_top_changed"] for x in positive), denominator=len(positive), survivor_mass_change=float(np.mean([x["mean_survivor_mass_change"] for x in positive]))),
                   scale_matched=dict(changed=sum(x["scaled_mean_top_changed"] for x in positive), denominator=len(positive), survivor_mass_change=float(np.mean([x["scaled_mean_survivor_mass_change"] for x in positive]))),
                   strength_ratio_median=float(np.median([x["strength_ratio"] for x in positive if x["strength_ratio"] is not None])),
                   earlier_total_gain=sum(x["gain"] for x in pairs), earlier_avoided_fall_slots=[x for x in pairs if x["independent_controls"]<300 and x["coupled_controls"]==300])
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    result = dict(status="COMPLETED", cpu_seconds=time.process_time()-CPU, wall_seconds=time.monotonic()-START, files=len(BINDINGS), model_controls=0, native_controls=0, neural_forwards=0, policy_updates=0)
except BaseException as error:
    result = dict(status="FAILED_SAVED_ANALYSIS", error=repr(error), traceback=traceback.format_exc(), cpu_seconds=time.process_time()-CPU, wall_seconds=time.monotonic()-START, model_controls=0, native_controls=0, neural_forwards=0, policy_updates=0)
finally:
    (OUT / "inputs.json").write_text(json.dumps(BINDINGS,indent=2)+"\n")
    (OUT / "receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
sys.exit(result["status"]!="COMPLETED")
