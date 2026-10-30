"""Plots and a candid report drawn only from retained raw run events."""
from pathlib import Path
import json
import numpy as np


def render_report(path,result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path=Path(path);destination=path/"reports"
    destination.mkdir(exist_ok=True)
    version=1
    while any(destination.glob(f"report_{version:03d}*")):version+=1
    stem=f"report_{version:03d}"
    updates=[r for r in result["events"] if r["kind"]=="actor_update"]
    if updates:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for branch in sorted({r["branch"] for r in updates}):
            for seed in sorted({r["seed"] for r in updates}):
                rows=[r for r in updates if r["branch"]==branch and r["seed"]==seed]
                scoped=all("charged_arm_work" in r for r in rows)
                x=[r["charged_arm_work"]["controls"] if scoped else r["cumulative_controls"] for r in rows]
                costs=[];captures=[]
                for row in rows:
                    chosen=[t for t in row["trials"] if t["family"]==row["selected_teacher_family"]]
                    costs.append(chosen[0]["cost"] if chosen else row["base"]["mean_cost"])
                    captures.append(chosen[0]["capture"] if chosen else row["base"]["capture_rate"])
                for ax,values in zip(axes,(costs,captures)):ax.plot(x,values,marker=".",label=f"{branch} seed {seed}")
        axes[0].set_ylabel("Complete actor qualification cost");axes[1].set_ylabel("Actor qualification capture fraction")
        scoped=all("charged_arm_work" in r for r in updates)
        for ax in axes:
            ax.set_xlabel("Charged arm physics controls, including shared inputs" if scoped else
                "Cumulative unique run controls (both arms)")
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7);fig.tight_layout()
        with (destination/f"{stem}_learning_curves.png").open("xb") as stream:fig.savefig(stream,format="png")
        plt.close(fig)
    counts={k:v for k,v in result.items() if k!="events"}
    with (destination/f"{stem}.json").open("x") as stream:stream.write(json.dumps(counts,indent=2)+"\n")
    verdict=[]
    for name in ("validate.json","qualification.json","acquisition.json","development.json"):
        if (path/name).exists():
            data=json.loads((path/name).read_text())
            verdict.append(f"- {name}: {data.get('failure') or data.get('verdict') or 'see retained stage outcome'}")
    text=("# Bidirectional MPPI–GPS evidence\n\n"+
        f"Charged requested controls: {result['unique_requested_controls']:,}. "
        f"Completed controls: {result['completed_controls']:,}. Unknown: {result['unknown_controls']:,}.\n\n"+
        "\n".join(verdict)+"\n\nMechanism tests, task qualification, actor-only learning and guided control are separate outcomes. "
        "No efficiency or standalone success follows from code execution alone.\n")
    with (destination/f"{stem}.md").open("x") as stream:stream.write(text)
    return destination/f"{stem}.md"
