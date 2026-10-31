# MPPI–policy learning on MuJoCo Walker

This repository studies whether independent MPPI discovery can teach a global torque policy, and whether feedback from learning can improve later trajectory optimization. The protected MPPI proposal bank remains policy independent.

The cloning and coupling studies remain negative for standalone walking. The [v3.1 matched data comparison](docs/loop_v3_results.md) tested recorded demonstrations; independent data improved survival without walking, while guided data lost that survival benefit. [L1 and H4.1](docs/walker_experiments_l1_h4_results.md) separately tested fresh student feature pipelines and feedback-selected agreement-weight backoff. All six H4 teachers walked, but all three children regressed against their incoming policy. No confirmation or second round ran. H3.2 and H4.2 remain untested. These torque-policy results do not establish a failed reproduction of the paper's state-to-state controller.

The [K1 compatibility diagnostic](docs/action_compatibility_k1_results.md) compares successful MPPI, random-action and qualified-reference trajectories against untrained, cloned and independently trained policies. An existing SAC reference walks **16/16 primary + 16/16 confirmation**, despite much worse MPPI action agreement than the failing clones. No new policy was trained. The result separates agreement from competence without establishing that trajectory diversity caused cloning failure.

## Setup

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Run commands from the checkout root:

```sh
uv sync --locked --extra test
uv run --locked --extra test python -m pytest -q
uv run python scripts/check_repository.py
```

CPU operation is sufficient for tests, saved-data fitting and metadata inspection. Optional GPU support is separate. See the [development guide](docs/development.md) for debugging and validation.

## Commands

```sh
# Read tracked results and verify capsule copies; no simulator or training.
uv run python scripts/train_mppi_gps.py inspect --run results/adaptive_coupling_h4_1 --verify
uv run python scripts/train_mppi_gps.py inspect --run results/student_learnability_l1 --json

# Discover the existing study interfaces without launching work.
uv run python scripts/train_mppi_gps.py study guided-data --help
uv run python scripts/train_mppi_gps.py study learnability --help
uv run python scripts/train_mppi_gps.py study backoff --help
uv run python scripts/train_mppi_gps.py study compatibility --help
uv run python scripts/train_mppi_gps.py --help
```

The public CLI retains `baseline`, `train`, `evaluate` and `render`. Study recipes live in [configs/studies](configs/studies); historical input paths must be supplied explicitly. Preparation creates a new frozen source/configuration/input binding, and workers execute that frozen source. Choose fresh output directories; see [safe debugging and replay](docs/development.md#current-source-and-frozen-workers). Cloning does not fetch bulk evidence or authorize a new experiment.

## Reading order

1. [Research status and document map](docs/README.md), then [methods and interfaces](docs/methods.md).
2. [K1 comparison and verdict](docs/action_compatibility_k1_results.md), [L1/H4.1 results](docs/walker_experiments_l1_h4_results.md) and [v3.1 comparison](docs/loop_v3_results.md).
3. [Exact result capsules](results/README.md), [evidence availability](docs/evidence_availability.md) and [machine-readable index](docs/evidence_index.json).
4. [Code map](docs/code_map.md), [debugging guide](docs/development.md), [primary-source review](docs/loop_v3_prior_art.md) and [historical record](docs/history.md).

Frozen runs, checkpoints, ledgers, raw arrays and historical source are preserved. The tracked capsules contain exact small evidence copies; H4.1 full raw evidence is **local only**. No beneficial loop closure, population-wide success or efficiency result is claimed.
