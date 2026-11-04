"""Saved-only paired query/actor plots and the already selected joint objective.

This is a display supplement, outside the frozen scientific worker. It changes
no decision rule and performs no simulation, network inference or fitting.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def supplement(run, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((run / "config.json").read_text())
    phases, displays, queries, accounting, optimizer, read_bytes = [], [], [], [], [], 0

    def load(path):
        nonlocal read_bytes
        read_bytes += path.stat().st_size
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}

    for stream in range(3):
        root = run / f"stream_{stream}"
        summary = json.loads((root / "result.json").read_text())
        ledger = root / "ledger.jsonl"
        by_phase = {}
        read_bytes += ledger.stat().st_size
        with ledger.open() as handle:
            for line in handle:
                event = json.loads(line)
                if event["event"] == "complete":
                    counts = by_phase.setdefault(event["phase"], {})
                    for key, value in event["counts"].items():
                        counts[key] = counts.get(key, 0) + value
        total = {}
        for counts in by_phase.values():
            for key, value in counts.items():
                total[key] = total.get(key, 0) + value
        if total != json.loads((root / "work.json").read_text())["completed"]:
            raise ValueError("Per-phase accounting does not join worker work")
        accounting.append(dict(stream=stream, exact_original_phase_counts=by_phase))
        for row in summary["rounds"]:
            r, arm = row["round"], row["arm"]
            cell = root / f"r{r:02d}_{arm}"
            if r:
                before = after = 0.0
                for slot in range(2):
                    its = json.loads((cell / f"search_{slot}/iterations.json").read_text())
                    info = json.loads((cell / f"search_{slot}/selection.json").read_text())
                    before += its[0]["J"]
                    after += info["selected"]["J"]
                    optimizer.append(
                        dict(
                            stream=stream,
                            arm=arm,
                            round=r,
                            slot=slot,
                            selected_iteration=info["selected"]["iteration"],
                            raw_best_iteration=info["raw_best"]["iteration"],
                            raw_best_rejected=info["raw_best_rejected"],
                            retained_action_rms_change=info["selected_change_rms"],
                            objective_relative_gain=info["objective_relative_gain"],
                            walking_nominals=sum(v["walking"] for v in its),
                            evaluated_nominals=len(its),
                            walking_samples=sum(v["candidate_walks"] for v in its),
                            surviving_samples=sum(v["candidate_survives"] for v in its),
                            samples=len(its) * 256,
                            best_later_nominal_C=min(v["C"] for v in its[1:]),
                            best_later_nominal_J=min(v["J"] for v in its[1:]),
                            first_update_rms=its[0]["displacement"],
                            first_ESS=its[0]["ESS"],
                            first_temperature=its[0]["temperature"],
                        )
                    )
                native = sum(t["cost"] for t in row["teacher"])
                phases.append(
                    dict(
                        stream=stream,
                        arm=arm,
                        round=r,
                        before_search=before,
                        after_search=after,
                        after_fit=native + row["alpha"] * 6000 * row["post_mse"],
                        note="Post-fit value uses recorded physical-target predictions; existing FP32 batch rounding applies.",
                    )
                )
            for query_row in row["queries"]:
                details = dict(stream=stream, arm=arm, round=r, **query_row)
                if query_row["available"]:
                    anchor = query_row["anchor"]
                    selected = load(cell / f"query_{anchor}/selected.npz")
                    states = selected["posts"]
                    healthy = (
                        (states[:, 2] > 0.8)
                        & (states[:, 2] < 2)
                        & (states[:, 3] > -1)
                        & (states[:, 3] < 1)
                    )
                    bad = np.flatnonzero(~healthy)
                    details.update(
                        survived=bool(healthy.all()),
                        executable_controls=int(bad[0] + 1) if len(bad) else len(states),
                        horizon=1000,
                        query_role="Post-fit common task-only PI diagnostic; not an optimal expert",
                    )
                queries.append(details)
            query_path = cell / "query_0/selected.npz"
            if not query_path.exists():
                continue
            actor = load(cell / f"eval_{cfg['primary_resets'][0]}.npz")
            query = load(query_path)
            context = json.loads((cell / "query_0/selection.json").read_text())["initial_context"]
            initial = np.asarray(context["state"])
            if not np.array_equal(initial, actor["states"][0]):
                raise ValueError("Paired plot initial context differs")
            q = query["posts"]
            healthy = (q[:, 2] > 0.8) & (q[:, 2] < 2) & (q[:, 3] > -1) & (q[:, 3] < 1)
            bad = np.flatnonzero(~healthy)
            length = int(bad[0] + 1) if len(bad) else len(q)
            q = np.concatenate((initial[None], q[:length]))
            fig, axes = plt.subplots(7, 1, figsize=(10, 11), sharex=True)
            for label, states, color in (
                ("native actor", actor["states"], "black"),
                ("task-only PI query (model)", q, "tab:blue"),
            ):
                t = np.arange(len(states)) * 0.008
                for j, ax in enumerate(axes):
                    ax.plot(t, states[:, j + 3], label=label, color=color, lw=0.7)
            for j, ax in enumerate(axes):
                ax.set_ylabel(f"q{j} (rad)")
                ax.set_xlim(0, 8)
            axes[0].legend(fontsize=8)
            axes[-1].set_xlabel("Same-start elapsed seconds; each path ends at its own fall")
            fig.suptitle(
                f"Paired query/actor: stream {stream}, {arm}, round {r}; no phase alignment"
            )
            fig.tight_layout()
            name = f"paired_s{stream}_{arm}_r{r:02d}.png"
            fig.savefig(out / name, dpi=100)
            plt.close(fig)
            displays.append(
                dict(
                    stream=stream,
                    arm=arm,
                    round=r,
                    file=name,
                    actor_controls=len(actor["actions"]),
                    query_controls=length,
                    query_fell=bool(len(bad)),
                )
            )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for stream, ax in enumerate(axes):
        for arm, color in (("zero", "tab:blue"), ("fixed", "tab:orange")):
            values = sorted(
                (v for v in phases if v["stream"] == stream and v["arm"] == arm),
                key=lambda v: v["round"],
            )
            x = []
            y = []
            for v in values:
                x.extend([v["round"] - 0.25, v["round"], v["round"] + 0.25])
                y.extend([v["before_search"], v["after_search"], v["after_fit"]])
            ax.plot(x, y, "o-", color=color, label=arm, ms=3)
        ax.set_title(f"Stream {stream}")
        ax.set_xlabel("Round: before search → after search → after fit")
        ax.set_ylabel("C + alpha D on the teacher paths")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "block_objective.png", dpi=140)
    plt.close(fig)
    result = dict(
        display_only=True,
        decision_rules_unchanged=True,
        new_physics=0,
        new_inference=0,
        new_fitting=0,
        logical_array_read_bytes=read_bytes,
        phases=phases,
        query_diagnostics=queries,
        acquisition_diagnostics=optimizer,
        accounting_by_phase=accounting,
        phase_accounting_note="Counts join the exact worker ledger. Initial replay/loading is shared. For round0 only, the first 2000-row reconstruction is in initial, and the second 2000-row reconstruction is labelled r0_zero_student_queries because the phase changes at evaluation. Subsequent rounds have explicit per-arm search/fit/evaluation/query phases. Initial reconstruction therefore costs 2000 rows per arm, not additional diagnostic work.",
        paired_displays=displays,
        distinction="The original report's first fresh student versus source-teacher atlas is unpaired. These query0/actor displays share an exact initial context. Query curves are model rollouts, cropped at first fall.",
    )
    (out / "supplement.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    r = supplement(a.run, a.out)
    print(
        json.dumps(
            dict(displays=len(r["paired_displays"]), new_physics=0, new_inference=0, new_fitting=0)
        )
    )


if __name__ == "__main__":
    main()
