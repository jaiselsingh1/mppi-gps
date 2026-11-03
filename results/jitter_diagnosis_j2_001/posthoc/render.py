"""Post-hoc view of six audited clean-data endpoints; no predictive admission."""
from pathlib import Path
import hashlib
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
started = time.monotonic()
payload = (root / "summary.json").read_bytes()
value = json.loads(payload)
assert value["post_hoc"] and not value["primary_confirmation"]
figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for axis, field, title, utility in (
    (axes[0], "offline_error", "Held reference-state action MSE", value["offline_utility"]),
    (axes[1], "diagnostic_error", "Post-fit on-policy action MSE", value["diagnostic_utility"]),
):
    for row in value["records"]:
        axis.scatter(row[field], row["walking"], s=40)
        axis.annotate(str(row["seed"] - 3600003000), (row[field], row["walking"]),
                      xytext=(4, 3), textcoords="offset points")
    axis.set_xlabel(title)
    axis.set_ylabel("Walking successes / 32 fresh resets")
    axis.set_ylim(0, 34)
    axis.grid(alpha=.2)
    axis.set_title(f"Utility = −Spearman(error, walking): {utility:.3f}")
figure.suptitle("Six clean-data children — exploratory within-condition analysis\n"
                 "J2 stopped at qualification; frozen primary comparison was not run")
figure.savefig(root / "within_clean.png", dpi=160)
plt.close(figure)
output = (root / "within_clean.png").read_bytes()
(root / "render_receipt.json").write_text(json.dumps(dict(
    post_hoc=True, primary_confirmation=False, read_bytes=len(payload),
    source_input_sha256=hashlib.sha256(payload).hexdigest(),
    output_bytes=len(output), output_sha256=hashlib.sha256(output).hexdigest(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    wall_seconds=time.monotonic()-started, new_scientific_work=0,
), indent=2) + "\n")
