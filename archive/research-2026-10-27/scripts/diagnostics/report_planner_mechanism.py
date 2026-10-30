"""Report the bounded frozen-model planner mechanism diagnostic.

Reads only this study's trusted, locally generated result dictionaries.  It
does not load controller/model inputs, simulate, fit, or change a live study.
Discovery seeds select the endpoint oracle; confirmation seeds only assess it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


KINDS = ("executed_controls", "candidate_controls", "memory_controls",
         "verification_controls", "planner_calls")
LABELS = {"coupled_mlp": "Frozen coupled MLP", "paired_ridge": "Frozen paired ridge"}
COLORS = {"coupled_mlp": "#3969ac", "paired_ridge": "#e68310"}


def _json(path):
    return json.loads(path.read_text())


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def _normalizer(gamma, remaining):
    return 5. * (remaining if gamma == 1 else -np.expm1(remaining * np.log(gamma)) / (1-gamma))


def _paired(candidate, baseline, name):
    if candidate.get("seeds") != baseline.get("seeds"):
        raise ValueError(f"{name}: paired continuation seeds differ")


def _costs(packet, repeats, offset=0):
    values = np.asarray(packet["full_costs"], dtype=float)
    if values.ndim != 1 or len(values) < offset+repeats or not np.isfinite(values).all():
        raise ValueError("Require complete finite branch returns")
    if len(packet["seeds"]) != len(values):
        raise ValueError("Branch seed/return length differs")
    return values[offset:offset+repeats]


def _stats(rows):
    """Equal-weight root means; bootstrap ROOTS, never independent repeats."""
    values = np.asarray([r["mean_difference"] for r in rows], dtype=float)
    if not len(values):
        return dict(roots=0, mean_difference=None, root_cluster_bootstrap_95=None)
    interval = None
    if len(values) >= 2:
        rng = np.random.default_rng(20261026)
        means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
        interval = np.quantile(means, [.025, .975]).tolist()
    return dict(roots=len(values), mean_difference=float(values.mean()),
                root_means=values.tolist(), roots_with_lower_cost=int(np.sum(values < 0)),
                root_cluster_bootstrap_95=interval,
                interval_scope="Descriptive root-cluster bootstrap; few roots, one frozen model per cohort; no training-seed inference")


def _difference(root, first, second, repeats, offset=0):
    a, b = _costs(first, repeats, offset), _costs(second, repeats, offset)
    delta = a-b
    return dict(root=int(root), candidate_mean=float(a.mean()), baseline_mean=float(b.mean()),
                mean_difference=float(delta.mean()), paired_differences=delta.tolist(),
                candidate_costs=a.tolist(), baseline_costs=b.tolist(),
                paired_difference_standard_error=(float(delta.std(ddof=1)/np.sqrt(len(delta)))
                                                  if len(delta) > 1 else None))


class _Reader:
    def __init__(self, directory):
        self.directory = directory
        self.cache, self.work, self.files = {}, {}, {}

    def get(self, phase, stem, required=True):
        relative = f"{phase}/{stem}.pkl"
        if relative in self.cache:
            return self.cache[relative]
        path = self.directory / relative
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return None
        # Fixed result paths from the authorized experiment, never arbitrary
        # user-supplied pickles or controller checkpoints.
        data = pickle.loads(path.read_bytes())
        if not isinstance(data, dict):
            raise ValueError(f"{relative}: expected result dictionary")
        ledger = data.get("ledger", {})
        for key, count in ledger.items():
            if isinstance(count, bool) or int(count) != count or count < 0:
                raise ValueError(f"{relative}: invalid work count")
            target = self.work.setdefault(phase, {})
            target[key] = target.get(key, 0) + int(count)
        self.files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.cache[relative] = data
        return data


def _cohort(reader, cohort, selection, discovery, confirmation):
    name = cohort["name"]
    cfg = cohort["config"]
    gamma = float(cfg.get("gamma", .999))
    roots = cohort.get("roots", [])
    result = dict(name=name, requested_roots=list(roots), roots=[], early_terminations=[],
                  endpoint_oracle=[], first_action_oracle=[], selected_plan_handoff=[],
                  unqualified_guided_controller=[], handoff_minus_replanning=[],
                  endpoint_vs_replanning=[], prediction_pairs=[], alternative_contrasts=[],
                  frozen_nominee_endpoint=[], frozen_nominee_replanning=[])
    for root in roots:
        stem = f"{name}_root_{root}"
        cap = reader.get("captures", stem)
        if cap.get("root") != root:
            raise ValueError("Capture root mismatch")
        if cap.get("status") != "captured":
            result["early_terminations"].append(dict(root=root, status=cap.get("status"),
                trace={k: cap.get("prefix_trace", {}).get(k) for k in
                       ("steps", "cost", "duration_seconds", "velocity_mae", "terminated")},
                no_backfill=cap.get("no_backfill")))
            continue
        bank = cap["bank"]
        pick = selection[str(root)]
        baseline, oracle = int(pick["baseline_index"]), int(pick["oracle_index"])
        if baseline != int(bank["baseline_index"]):
            raise ValueError("Selection baseline differs from captured local winner")
        candidate_indices = [int(i) for i in pick["candidate_indices"]]
        if baseline not in candidate_indices or oracle not in candidate_indices:
            raise ValueError("Oracle/baseline outside preselected candidate set")
        endpoints, replan = {}, {}
        for index in candidate_indices:
            key = f"{stem}_candidate_{index}"
            endpoints[index] = reader.get("endpoints", key)
            replan[index] = reader.get("replanning", key)
            if endpoints[index]["candidate_index"] != index or replan[index]["candidate_index"] != index:
                raise ValueError("Branch candidate index mismatch")
            _costs(endpoints[index], confirmation, discovery)
            if len(endpoints[index]["seeds"]) != discovery+confirmation:
                raise ValueError("Endpoint discovery/confirmation split length differs from protocol")
            if (len(set(endpoints[index]["seeds"])) != discovery+confirmation
                    or len(set(replan[index]["seeds"])) != confirmation
                    or set(endpoints[index]["seeds"]) & set(replan[index]["seeds"])):
                raise ValueError("Discovery, endpoint confirmation and replanning streams must be disjoint")
            observed = float(_costs(endpoints[index], discovery).mean())
            recorded = float(pick["discovery_means"][str(index)])
            if not np.isclose(observed, recorded, rtol=1e-10, atol=1e-8):
                raise ValueError("Discovery means do not reproduce oracle selection")
        min_discovery = min(float(_costs(v, discovery).mean()) for v in endpoints.values())
        if not np.isclose(float(_costs(endpoints[oracle], discovery).mean()), min_discovery):
            raise ValueError("Oracle was not selected from discovery returns")
        ebase, rbase = endpoints[baseline], replan[baseline]
        for packet in endpoints.values():
            _paired(packet, ebase, "endpoint")
        for packet in replan.values():
            _paired(packet, rbase, "replanning")
        primary = _difference(root, endpoints[oracle], ebase, confirmation, discovery)
        oracle_plan, baseline_plan = np.asarray(bank["plans"][oracle]), np.asarray(bank["plans"][baseline])
        first_equal = bool(np.array_equal(oracle_plan[0], baseline_plan[0]))
        first_l2 = float(np.linalg.norm(oracle_plan[0]-baseline_plan[0]))
        replanning_exact = bool(np.array_equal(_costs(replan[oracle], confirmation),
                                              _costs(rbase, confirmation)))
        if first_equal:
            for a, b in zip(replan[oracle].get("traces", []), rbase.get("traces", [])):
                replanning_exact &= all(np.array_equal(a[key], b[key]) for key in ("states", "actions"))
        raw_post_u = (cap.get("raw_post") or {}).get("U")
        shifted_oracle = np.concatenate([oracle_plan[1:], oracle_plan[-1:]])
        nominal_change = (float(np.linalg.norm(np.asarray(raw_post_u)-shifted_oracle))
                          if raw_post_u is not None else None)
        primary.update(oracle_index=oracle, baseline_index=baseline,
                       oracle_differs_from_baseline=oracle != baseline,
                       oracle_first_action_L2_vs_baseline=first_l2,
                       oracle_first_action_exactly_equals_baseline=first_equal,
                       identical_first_action_replanning_exact=(replanning_exact if first_equal else None),
                       raw_post_vs_shifted_oracle_nominal_frobenius=nominal_change)
        result["endpoint_oracle"].append(primary)
        result["first_action_oracle"].append(_difference(root, replan[oracle], rbase, confirmation))
        handoffs = {i: reader.get("handoff", f"{stem}_candidate_{i}")
                    for i in dict.fromkeys((baseline, oracle))}
        for packet in handoffs.values():
            _paired(packet, rbase, "handoff")
        result["selected_plan_handoff"].append(
            _difference(root, handoffs[oracle], handoffs[baseline], confirmation))
        result["handoff_minus_replanning"].append(
            _difference(root, handoffs[oracle], replan[oracle], confirmation))
        # Same current state but deliberately different continuation semantics.
        result["endpoint_vs_replanning"].append(dict(root=root,
            endpoint_candidate_advantage=primary["mean_difference"],
            replan_candidate_advantage=result["first_action_oracle"][-1]["mean_difference"],
            handoff_candidate_advantage=result["selected_plan_handoff"][-1]["mean_difference"]))
        guided = reader.get("guided", stem)
        if guided["candidate_index"] != bank["selected_index"]:
            raise ValueError("Guided diagnostic must start with actual bounded selected action")
        _paired(guided, rbase, "unqualified guided controller")
        gd = _difference(root, guided, rbase, confirmation)
        gd.update(diagnostic_alpha=guided.get("diagnostic_alpha"),
                  unqualified_diagnostic=guided.get("unqualified_diagnostic"),
                  guidance_fractions=[t.get("guidance_fraction") for t in guided.get("traces", [])],
                  action_change_fractions=[t.get("action_change_fraction") for t in guided.get("traces", [])])
        result["unqualified_guided_controller"].append(gd)
        local = np.asarray(bank["local_costs"], dtype=float)
        tails = bank.get("predicted_values")
        tails = np.full(len(local), np.nan) if tails is None else np.asarray(tails, dtype=float)
        supported = bank.get("supported")
        supported = np.zeros(len(local), dtype=bool) if supported is None else np.asarray(supported, dtype=bool)
        healthy = np.asarray(bank["healthy"], dtype=bool)
        correction_cap = float(bank.get("diagnostic_correction_cap", cap.get("diagnostic_alpha", .1)
            * 3 * cfg.get("max_tail_shift_temperatures", 2)))
        score = local + gamma ** int(bank["H"]) * tails
        score = np.where(healthy, score, local)
        scale = _normalizer(gamma, int(cap["remaining"]))
        candidates = []
        for index in candidate_indices:
            end = endpoints[index]
            measured = _costs(end, confirmation, discovery)
            discrepancy = float(end["endpoint_max_abs_discrepancy"])
            comparable = bool(not end["fell_in_prefix"]
                              and end.get("prefix_trace", {}).get("steps") == int(bank["H"]))
            batch, replay = end.get("batch_prediction"), end.get("replay_prediction")
            role = next((r for r in cap["selection"]["roles"] if int(r["index"]) == index), {})
            first_action_l2 = float(np.linalg.norm(np.asarray(bank["plans"][index, 0])
                                                 -np.asarray(bank["plans"][baseline, 0])))
            first_action_equal = bool(np.array_equal(bank["plans"][index, 0], bank["plans"][baseline, 0]))
            item = dict(index=index, role=role.get("role", "unknown"),
                decision_type=role.get("decision_type"), selection_fallback=role.get("fallback"),
                first_action_L2_vs_baseline=first_action_l2,
                first_action_exactly_equals_baseline=first_action_equal,
                local_cost=float(local[index]), predicted_unbounded_total=_number(score[index]),
                supported=bool(supported[index]), healthy=bool(healthy[index]),
                local_gap=float(local[index]-local[baseline]),
                bounded_score=float(bank["scores"][index]),
                correction=float(bank["corrections"][index]), correction_cap=correction_cap,
                can_strictly_beat_baseline_with_cap=bool(index != baseline and healthy[index] and supported[index]
                    and healthy[baseline] and supported[baseline]
                    and local[index]-local[baseline] < correction_cap),
                confirmation_costs=measured.tolist(), confirmation_mean=float(measured.mean()),
                discovery_mean=float(_costs(end, discovery).mean()),
                endpoint_max_abs_discrepancy=discrepancy,
                endpoint_max_abs_discrepancy_raw=discrepancy,
                endpoint_replay_comparable=comparable,
                valid_endpoint_max_abs_discrepancy=discrepancy if comparable else None,
                endpoint_replay_comparison_reason=("same healthy H-step endpoint" if comparable else
                    "first-fall stop is not comparable with the padded H-step batch state" if end["fell_in_prefix"] else
                    "prefix did not execute the complete captured H-step plan"),
                batch_prediction=batch, replay_prediction=replay,
                original_batch_prediction=end.get("original_batch_prediction"),
                replay_prediction_shift=(_number(replay["cost"]-batch["cost"])
                    if comparable and batch and replay and batch.get("cost") is not None and replay.get("cost") is not None else None),
                replay_support_changed=(bool(batch["supported"] != replay["supported"])
                                        if comparable and batch and replay else None),
                replay_prefix_cost=float(end["prefix_cost"]),
                batch_local_minus_replayed_prefix=float(local[index]-end["prefix_cost"]),
                fell_in_prefix=bool(end["fell_in_prefix"]))
            candidates.append(item)
            if index != baseline:
                result["alternative_contrasts"].append(dict(root=int(root), index=index,
                    baseline_index=baseline, role=item["role"], decision_type=item["decision_type"],
                    supported=bool(supported[index] and supported[baseline]),
                    candidate_healthy=bool(healthy[index]), baseline_healthy=bool(healthy[baseline]),
                    first_action_L2_vs_baseline=first_action_l2,
                    first_action_exactly_equals_baseline=first_action_equal,
                    endpoint=_difference(root, end, ebase, confirmation, discovery),
                    replanning=_difference(root, replan[index], rbase, confirmation),
                    selection_scope="Every preselected nonbaseline plan, without filtering by any return"))
            if index != baseline and np.isfinite(score[index]) and np.isfinite(score[baseline]):
                delta = measured-_costs(ebase, confirmation, discovery)
                predicted = float(score[index]-score[baseline])
                observed = float(delta.mean())
                base_comparable = bool(not ebase["fell_in_prefix"]
                    and ebase.get("prefix_trace", {}).get("steps") == int(bank["H"]))
                if healthy[index] and healthy[baseline] and comparable and base_comparable:
                    pair_kind = "learned_healthy"
                elif (not healthy[index] and not healthy[baseline]
                      and end["fell_in_prefix"] and ebase["fell_in_prefix"]):
                    pair_kind = "known_fall"
                elif bool(healthy[index]) == comparable and bool(healthy[baseline]) == base_comparable:
                    pair_kind = "mixed_healthy_known_fall"
                else:
                    pair_kind = "replay_health_mismatch"
                result["prediction_pairs"].append(dict(root=root, index=index,
                    pair_kind=pair_kind,
                    predicted_difference=predicted, observed_mean_difference=observed,
                    observed_paired_differences=delta.tolist(), normalizer=scale,
                    normalized_error=(predicted-observed)/scale,
                    supported=bool(supported[index] and supported[baseline]),
                    sign_correct=bool(np.sign(predicted) == np.sign(observed))))
        all_gaps = local-local[baseline]
        # This secondary selector has no measured-return input.  It uses only
        # the model/local scores captured before any endpoint query, including
        # the local baseline and respecting the original candidate order.
        finite_nominees = [i for i in candidate_indices if np.isfinite(score[i])]
        nominee = min(finite_nominees, key=lambda i: score[i]) if finite_nominees else None
        if nominee is not None:
            nominee_metadata = dict(nominee_index=nominee, baseline_index=baseline,
                differs_from_baseline=nominee != baseline,
                role=next((r.get("role") for r in cap["selection"]["roles"] if int(r["index"]) == nominee), None),
                decision_type=next((r.get("decision_type") for r in cap["selection"]["roles"] if int(r["index"]) == nominee), None),
                selected_from="Original captured unbounded full score among preselected measured plans including baseline; no return labels",
                first_action_L2_vs_baseline=float(np.linalg.norm(bank["plans"][nominee, 0]-bank["plans"][baseline, 0])),
                first_action_exactly_equals_baseline=bool(np.array_equal(bank["plans"][nominee, 0], bank["plans"][baseline, 0])))
            nd = _difference(root, endpoints[nominee], ebase, confirmation, discovery)
            nd.update(nominee_metadata)
            result["frozen_nominee_endpoint"].append(nd)
            nd = _difference(root, replan[nominee], rbase, confirmation)
            nd.update(nominee_metadata)
            result["frozen_nominee_replanning"].append(nd)
        rootrow = dict(root=root, capture_step=cap["capture_step"], remaining=cap["remaining"],
            baseline_index=baseline, oracle_index=oracle, selected_index=int(bank["selected_index"]),
            oracle_first_action_L2_vs_baseline=first_l2,
            oracle_first_action_exactly_equals_baseline=first_equal,
            identical_first_action_replanning_exact=(replanning_exact if first_equal else None),
            raw_post_vs_shifted_oracle_nominal_frobenius=nominal_change,
            frozen_unbounded_nominee_index=nominee,
            frozen_unbounded_nominee_selection="Original batch scores only; baseline included; no return labels",
            selection=cap["selection"], baseline_supported=bool(supported[baseline]),
            baseline_healthy=bool(healthy[baseline]), baseline_fell_in_replayed_prefix=bool(ebase["fell_in_prefix"]),
            bank_candidates=len(local), healthy_candidates=int(healthy.sum()),
            supported_healthy_candidates=int((supported & healthy).sum()),
            fallback_reason=bank.get("fallback_reason"), guidance_used=bool(bank["guidance_used"]),
            action_changed=bool(bank["action_changed"]), correction_cap=correction_cap,
            supported_healthy_alternatives_within_cap=int(np.sum((np.arange(len(local)) != baseline)
                & healthy & supported & (all_gaps < correction_cap))) if supported[baseline] and healthy[baseline] else 0,
            candidates=candidates)
        result["roots"].append(rootrow)
    for key in ("endpoint_oracle", "first_action_oracle", "selected_plan_handoff",
                "unqualified_guided_controller", "handoff_minus_replanning",
                "frozen_nominee_endpoint", "frozen_nominee_replanning"):
        result[key+"_summary"] = _stats(result[key])
    pairs = result["prediction_pairs"]
    learned = [r for r in pairs if r["pair_kind"] == "learned_healthy"]
    for label, subset in (("all", pairs), ("learned", learned),
            ("supported", [r for r in learned if r["supported"]]),
            ("known_fall", [r for r in pairs if r["pair_kind"] == "known_fall"]),
            ("mixed", [r for r in pairs if r["pair_kind"] == "mixed_healthy_known_fall"]),
            ("replay_health_mismatch", [r for r in pairs if r["pair_kind"] == "replay_health_mismatch"])):
        result[f"prediction_{label}"] = dict(pairs=len(subset),
            normalized_difference_rmse=(float(np.sqrt(np.mean([r["normalized_error"]**2 for r in subset]))) if subset else None),
            sign_accuracy=(float(np.mean([r["sign_correct"] for r in subset])) if subset else None),
            scope="Descriptive selected pairs; multiple pairs share roots; no independent pair confidence interval")
    all_candidates = [p for r in result["roots"] for p in r["candidates"]]
    comparable = [p for p in all_candidates if p["endpoint_replay_comparable"]]
    result["endpoint_replay_summary"] = dict(
        captured_roots=len(result["roots"]),
        healthy_baseline_roots=sum(r["baseline_healthy"] for r in result["roots"]),
        fallen_baseline_roots=sum(not r["baseline_healthy"] for r in result["roots"]),
        measured_candidates=len(all_candidates), healthy_completed_prefixes=len(comparable),
        fallen_prefixes=sum(p["fell_in_prefix"] for p in all_candidates),
        noncomparable_prefixes=len(all_candidates)-len(comparable),
        healthy_replay_max_abs_discrepancy=(max(p["valid_endpoint_max_abs_discrepancy"]
            for p in comparable) if comparable else None),
        healthy_replay_max_abs_prediction_shift=(max(abs(p["replay_prediction_shift"])
            for p in comparable if p["replay_prediction_shift"] is not None)
            if any(p["replay_prediction_shift"] is not None for p in comparable) else None),
        interpretation="Only healthy complete H-step prefixes compare the same endpoint. First-fall stopped states and padded H-step batch states are different-time states; their preserved raw differences are not replay errors.")
    return result


def _fmt(value, digits=2):
    return "unavailable" if value is None else f"{value:.{digits}f}"


def _markdown(report):
    lines = ["# Frozen planner mechanism diagnostic", "",
        "This is a bounded read-only mechanism experiment. No model was trained or promoted. "
        "Both frozen models were tested using the common paired-endpoint candidate-bank rule; "
        "the coupled MLP cohort does not reproduce its original learned-mean controller.", "",
        "Lower finite discounted cost is better. Differences below are candidate minus local-memory baseline. "
        "The endpoint oracle is selected only on discovery repeats and assessed on separate confirmation repeats. "
        "It is the best of the preselected plans on those discovery measurements, not a globally optimal planner.", "",
        "| Frozen cohort | Captured / requested roots | Endpoint oracle Δ | First action + replan Δ | Single-plan handoff Δ | Unqualified α=.1 controller Δ |",
        "|---|---:|---:|---:|---:|---:|"]
    for c in report["cohorts"]:
        values = [_fmt(c[k+"_summary"]["mean_difference"]) for k in
                  ("endpoint_oracle", "first_action_oracle", "selected_plan_handoff", "unqualified_guided_controller")]
        lines.append(f"| {LABELS.get(c['name'], c['name'])} | {len(c['roots'])}/{len(c['requested_roots'])} | " + " | ".join(values)+" |")
    lines += ["", "## Per-root confirmation evidence", "",
        "Each row retains its paired confirmation returns in comparison.json. Early terminations are retained below without replacing roots.", "",
        "| Cohort / root | Baseline index → discovery oracle | Endpoint Δ | Replanning Δ | Handoff Δ | Guided controller Δ |",
        "|---|---|---:|---:|---:|---:|"]
    for c in report["cohorts"]:
        for j, root in enumerate(c["roots"]):
            values = [_fmt(c[k][j]["mean_difference"]) for k in
                      ("endpoint_oracle", "first_action_oracle", "selected_plan_handoff", "unqualified_guided_controller")]
            lines.append(f"| {c['name']} / {root['root']} | {root['baseline_index']} → {root['oracle_index']} | "+" | ".join(values)+" |")
        for early in c["early_terminations"]:
            lines.append(f"| {c['name']} / {early['root']} | {early['status']}; no backfill | — | — | — | — |")
    lines += ["", "## All preselected alternatives and frozen-score nominee", "",
        "If discovery selects the local baseline, the primary oracle-versus-baseline contrast is exactly zero "
        "and cannot diagnose execution transfer. The following secondary evidence retains every preselected "
        "alternative without selecting or filtering on confirmation performance. Endpoint and replanning "
        "differences each use their own paired confirmation streams.", "",
        "| Cohort / root / candidate | Pre-query role | Endpoint Δ ± paired SE | First-action replanning Δ ± paired SE | First-action L2 difference |",
        "|---|---|---:|---:|---:|"]
    for c in report["cohorts"]:
        for row in c["alternative_contrasts"]:
            e, r = row["endpoint"], row["replanning"]
            lines.append(f"| {c['name']} / {row['root']} / {row['index']} | "
                f"{row['role']} ({row['decision_type'] or 'preselected'}) | "
                f"{_fmt(e['mean_difference'])} ± {_fmt(e['paired_difference_standard_error'])} | "
                f"{_fmt(r['mean_difference'])} ± {_fmt(r['paired_difference_standard_error'])} | "
                f"{_fmt(row['first_action_L2_vs_baseline'], 6)} |")
    lines += ["", "The frozen-score nominee is selected solely by the original captured "
        "local-plus-unbounded-value score among the measured candidate set **including the baseline**. "
        "It is a model-prediction contrast, not the discovery oracle, the production bounded winner, "
        "or a hindsight-best confirmation plan. Unsupported/exploratory choices retain their recorded status. "
        "No alternative handoff branches were added for this analysis.", "",
        "| Cohort | Frozen-score nominee changed / roots | Nominee endpoint Δ | Nominee first-action replanning Δ |",
        "|---|---:|---:|---:|"]
    for c in report["cohorts"]:
        rows = c["frozen_nominee_endpoint"]
        lines.append(f"| {c['name']} | {sum(r['differs_from_baseline'] for r in rows)}/{len(rows)} | "
            f"{_fmt(c['frozen_nominee_endpoint_summary']['mean_difference'])} | "
            f"{_fmt(c['frozen_nominee_replanning_summary']['mean_difference'])} |")
    lines += ["", "| Cohort / root | Frozen-score nominee (baseline) | Recorded decision type | Endpoint Δ | Replanning Δ |",
        "|---|---|---|---:|---:|"]
    for c in report["cohorts"]:
        for e, r in zip(c["frozen_nominee_endpoint"], c["frozen_nominee_replanning"]):
            lines.append(f"| {c['name']} / {e['root']} | {e['nominee_index']} ({e['baseline_index']}) | "
                f"{e['role']} ({e['decision_type'] or 'preselected'}) | "
                f"{_fmt(e['mean_difference'])} | {_fmt(r['mean_difference'])} |")
    lines += ["", "## Prediction, support and score reach", "",
        "Predicted differences use the original captured batch endpoint and local score, "
        "not a retrospectively substituted replay endpoint. Measured differences use complete H-plan execution "
        "then canonical-restart fixed-reference continuation. Normalization uses the finite absorbing-cost "
        "scale at the captured remaining horizon; it is not a physical return bound.", "",
        "Only pairs with two healthy completed H-step endpoints count as learned-value ranking evidence. "
        "Pairs whose two plans fall have known finite costs and do not test the learned model. "
        "Mixed healthy/fallen pairs and replay-health mismatches remain separate in comparison.json.", "",
        "| Cohort | Healthy learned pairs | Healthy Δ RMSE (normalized) | Supported Δ RMSE | Supported sign accuracy | Bounded action changed / roots |",
        "|---|---:|---:|---:|---:|---:|"]
    for c in report["cohorts"]:
        lines.append(f"| {c['name']} | {c['prediction_learned']['pairs']} | {_fmt(c['prediction_learned']['normalized_difference_rmse'], 3)} | "
            f"{_fmt(c['prediction_supported']['normalized_difference_rmse'], 3)} | "
            f"{_fmt(c['prediction_supported']['sign_accuracy'], 3)} | "
            f"{sum(r['action_changed'] for r in c['roots'])}/{len(c['roots'])} |")
    lines += ["", "| Cohort | Known fallen-cost pairs | Known-cost Δ RMSE | Mixed healthy/fallen pairs | Replay-health mismatch pairs |",
        "|---|---:|---:|---:|---:|"]
    for c in report["cohorts"]:
        lines.append(f"| {c['name']} | {c['prediction_known_fall']['pairs']} | "
            f"{_fmt(c['prediction_known_fall']['normalized_difference_rmse'], 3)} | "
            f"{c['prediction_mixed']['pairs']} | {c['prediction_replay_health_mismatch']['pairs']} |")
    lines += ["", "| Cohort | Healthy / fallen baseline roots | Healthy comparable prefixes | Fallen prefixes | Healthy replay maximum state discrepancy |",
        "|---|---:|---:|---:|---:|"]
    for c in report["cohorts"]:
        s = c["endpoint_replay_summary"]
        lines.append(f"| {c['name']} | {s['healthy_baseline_roots']} / {s['fallen_baseline_roots']} | "
            f"{s['healthy_completed_prefixes']} | {s['fallen_prefixes']} | "
            f"{_fmt(s['healthy_replay_max_abs_discrepancy'], 8)} |")
    lines += ["", "Per-root data additionally report every preselected candidate's support, local score gap, "
        "correction bound, whether that bound could change the local winner, original/recomputed batch "
        "predictions, sequential replay endpoint discrepancy and replay prediction/support shift. "
        "First-fall replay stops are compared with padded H-step batch states only as preserved raw differences: "
        "they are different-time states, not valid replay-error measurements. Only healthy complete H-step "
        "prefixes enter the replay maxima. An exploratory unbounded model alternative is explicitly "
        "distinguished from the actual bounded model choice.", "",
        "Oracle-versus-baseline first-action distance/equality and the shifted-oracle-versus-raw-post nominal "
        "distance are retained per root. Identical first actions with identical future nominal/RNG imply "
        "identical production replanning trajectories; a different H-step plan need not provide a different first action.", "",
        "## Interpretations and limits", "",
        "- Endpoint improvement with weak first-action improvement identifies a mismatch in these measured "
        "continuation experiments; it does not prove a universal replanning failure. H-step execution, "
        "one-action replanning and nominal handoff are deliberately different controllers.",
        "- Endpoint discovery, endpoint confirmation and first-action confirmation use disjoint future RNG streams. "
        "Candidate-versus-baseline comparisons are paired within each controller semantics; "
        "the endpoint-versus-replanning contrast compares their respective paired advantages, not identical stochastic trajectories.",
        "- Handoff changes only the initial shifted selected nominal; later controls use production raw nominal updates. "
        "The α=.1 diagnostic instead uses the frozen value on subsequent decisions. Neither was accepted, "
        "promoted or deployed in either training study.",
        "- Root-cluster bootstrap intervals in comparison.json resample roots, not individual seeds or candidate pairs. "
        "With very few roots they are descriptive and weak; they do not establish training-seed robustness. "
        "With one root no interval is reported.",
        "- The confirmation minimum over all plans is hindsight; it is never used as the primary oracle or "
        "reported as an unbiased available choice. The candidate set is small and selected before continuation labels.",
        "- No additional reset panel, retuning, fitting, or source change to the active training runs follows from this report.", "",
        "## Executed diagnostic work", "",
        "Counts include every loaded result file once, including captures and all repeated rejected/counterfactual branches. "
        "Executed physical controls, candidate rollout controls, archive resimulation, mean verification and planner calls "
        "remain separate. Analytic absorbing costs are not physical controls.", "",
        "| Phase | Executed | Candidate | Memory | Verification | Planner calls |", "|---|---:|---:|---:|---:|---:|"]
    for phase, ledger in report["work_by_phase"].items():
        lines.append(f"| {phase} | "+" | ".join(str(ledger.get(k, 0)) for k in KINDS)+" |")
    lines += ["| **Total** | "+" | ".join(str(report['work_total'].get(k, 0)) for k in KINDS)+" |", "",
        "![Mechanism comparison](mechanism_diagnostics.png)", "",
        "Complete root/candidate/repeat values, root-cluster intervals, source checks and result-file hashes: [comparison.json](comparison.json)."]
    return "\n".join(lines)+"\n"


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
    axes = axes.ravel()
    for a in (axes[0], axes[2], axes[3]):
        a.axhline(0, color="#666666", lw=.8)
        a.grid(axis="y", alpha=.18)
    for ci, c in enumerate(report["cohorts"]):
        color = COLORS.get(c["name"], f"C{ci}")
        label = LABELS.get(c["name"], c["name"])
        rows = c["endpoint_oracle"]
        x = np.arange(len(rows)) + ci*.12
        for j, row in enumerate(rows):
            axes[0].scatter(np.full(len(row["paired_differences"]), x[j]), row["paired_differences"],
                            s=15, color=color, alpha=.25)
        axes[0].plot(x, [r["mean_difference"] for r in rows], "o-", color=color, label=label)
        learned_pairs = [p for p in c["prediction_pairs"] if p["pair_kind"] == "learned_healthy"]
        for pair in learned_pairs:
            axes[1].scatter(pair["predicted_difference"], pair["observed_mean_difference"],
                marker="o" if pair["supported"] else "x", color=color, alpha=.8,
                label=label if pair is learned_pairs[0] else None)
        for k, key in enumerate(("endpoint_oracle", "first_action_oracle", "selected_plan_handoff")):
            vals = [r["mean_difference"] for r in c[key]]
            if vals:
                axes[2].scatter(np.full(len(vals), k+(ci-.5)*.15), vals, color=color, alpha=.5, s=25)
                axes[2].plot(k+(ci-.5)*.15, np.mean(vals), "D", color=color, ms=7,
                             label=label+" · discovery oracle" if k == 0 else None)
        for k, key in enumerate(("frozen_nominee_endpoint", "frozen_nominee_replanning")):
            vals = [r["mean_difference"] for r in c[key]]
            if vals:
                axes[2].scatter(np.full(len(vals), k+(ci-.5)*.15+.04), vals,
                    marker="v", facecolors="none", edgecolors=color, alpha=.6, s=35)
                axes[2].plot(k+(ci-.5)*.15+.04, np.mean(vals), "v", color=color, ms=8,
                    label=label+" · frozen-score nominee" if k == 0 else None)
        for j, row in enumerate(c["unqualified_guided_controller"]):
            axes[3].scatter(np.full(len(row["paired_differences"]), j+ci*.12), row["paired_differences"],
                            s=15, color=color, alpha=.25)
        axes[3].plot(np.arange(len(c["unqualified_guided_controller"]))+ci*.12,
            [r["mean_difference"] for r in c["unqualified_guided_controller"]], "o-", color=color, label=label)
    axes[0].set(title="1. Discovery-selected endpoint oracle: fresh confirmation", xlabel="Root position within cohort", ylabel="Oracle − local baseline cost")
    axes[1].set(title="2. Learned ranking: two healthy endpoints only", xlabel="Predicted candidate − baseline cost", ylabel="Measured candidate − baseline cost")
    bounds = axes[1].get_xlim()+axes[1].get_ylim()
    lo, hi = min(bounds), max(bounds)
    axes[1].plot([lo, hi], [lo, hi], "--", color="#777777", lw=1)
    axes[1].axhline(0, color="#aaaaaa", lw=.6); axes[1].axvline(0, color="#aaaaaa", lw=.6)
    axes[1].text(.02, .98, "o supported; × unsupported\nKnown fallen-cost pairs excluded\nMultiple pairs share a root", transform=axes[1].transAxes, va="top", fontsize=9)
    axes[2].set(title="3. Transfer: discovery oracle / frozen-score nominee", xticks=[0, 1, 2],
        xticklabels=["Execute H + reference", "First action + replan", "Single-plan handoff"], ylabel="Selected plan − corresponding baseline cost")
    axes[3].set(title="4. Frozen α=.1 controller (UNQUALIFIED diagnostic)", xlabel="Root position within cohort", ylabel="Guided diagnostic − memory replan cost")
    for a in axes:
        handles, _ = a.get_legend_handles_labels()
        if handles:
            a.legend(fontsize=8, loc="best")
    fig.suptitle("Frozen-model mechanism probes · lower is better · no training or promotion\n"
        "Dots: paired confirmation repeats / root means; no independent training-seed claim", fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build(study_dir):
    study = Path(study_dir)
    protocol = _json(study / "protocol.json")
    selection = _json(study / "selection.json")
    discovery = int(protocol.get("discovery_repeats", 2))
    confirmation = int(protocol.get("confirmation_repeats", 4))
    if discovery < 1 or confirmation < 1:
        raise ValueError("Discovery and confirmation counts must be positive")
    cohorts = protocol["cohorts"]
    if isinstance(cohorts, dict):
        cohorts = [dict(value, name=name) for name, value in cohorts.items()]
    reader = _Reader(study)
    outcomes = [_cohort(reader, c, selection[c["name"]], discovery, confirmation) for c in cohorts]
    # Do not silently omit files or their work, even if an early-fall branch
    # or duplicate intervention was scheduled but not needed by the tables.
    for phase in ("captures", "endpoints", "replanning", "handoff", "guided"):
        for path in sorted((study/phase).glob("*.pkl")):
            reader.get(phase, path.stem)
    total = {}
    for ledger in reader.work.values():
        for key, value in ledger.items():
            total[key] = total.get(key, 0)+value
    status = _json(study/"status.json") if (study/"status.json").exists() else None
    if status is not None:
        recorded = status.get("work", {}).get("total")
        if recorded is not None and recorded != total:
            raise ValueError("Reconstructed work differs from the runner's completed ledger")
        if status.get("status") == "complete" and status.get("source_unchanged") is not True:
            raise ValueError("Completed diagnostic lacks unchanged-source verification")
    archive_checks = []
    for relative, digest in protocol.get("source_sha256", {}).items():
        path = study/"source"/relative
        ok = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
        archive_checks.append(dict(path=relative, matches=ok))
    if any(not c["matches"] for c in archive_checks):
        raise ValueError("Archived diagnostic source changed")
    inputs = _json(study/"input_manifest.json") if (study/"input_manifest.json").exists() else None
    checks = []
    if inputs:
        for name, cohort in inputs.get("cohorts", {}).items():
            for filename, digest in cohort.get("files", {}).items():
                path = study/"inputs"/name/filename
                ok = path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == digest
                checks.append(dict(path=str(path.relative_to(study)), matches=ok))
        if any(not c["matches"] for c in checks):
            raise ValueError("Frozen diagnostic input hashes changed")
    report = dict(schema_version=1, protocol=protocol, cohorts=outcomes,
        discovery_repeats=discovery, confirmation_repeats=confirmation,
        controller_semantics="Common PairedEndpointPlanner bank/reranking for both cohorts, not the original coupled learned weighted mean",
        no_training=True, no_promotion=True, input_checks=checks,
        input_unchanged=all(c["matches"] for c in checks) if checks else None,
        diagnostic_status=status, archived_source_checks=archive_checks,
        archived_source_unchanged=all(c["matches"] for c in archive_checks) if archive_checks else None,
        work_by_phase=reader.work, work_total=total, result_sha256=reader.files,
        uncertainty="Equal-weight roots; descriptive root-cluster bootstrap with few roots; confirmation repeats are paired within root",
        oracle_selection="Lowest discovery mean among preselected plans; no confirmation outcome used to select the oracle")
    report = _plain(report)
    (study/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (study/"comparison.md").write_text(_markdown(report))
    _plot(report, study/"mechanism_diagnostics.png")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.study_dir)
    print(json.dumps({"cohorts": len(report["cohorts"]), "input_unchanged": report["input_unchanged"],
                      "comparison": str(args.study_dir/"comparison.md")}))


if __name__ == "__main__":
    main()
