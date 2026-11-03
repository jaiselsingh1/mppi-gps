# MPPI–policy learning on MuJoCo Walker

The goal is a cheap standalone walking policy taught by competent MPPI search, with feedback from learning improving subsequent acquisition. The central open question is **pre-update teachability**: can a score predict what the student will execute after a fixed learning update?

Start with the [coordinated investigation map](docs/walker_coordination_review.md). MPPI-taught competent walking and beneficial coupling remain unestablished. [J1](docs/similarity_j1_results.md) lacked outcome variation for its protected prediction comparison. [J2](docs/jitter_diagnosis_j2_results.md) produced three competent clean-reference clones but stopped at its frozen qualification gate; its six-student post-fit association is exploratory. The `sac-continuation` prototypes and their missing controls/audits are reviewed separately. These torque-policy studies are not a reproduction of CIO–GPS’s state-to-state controller.

Frozen runs, failed results, configurations, checkpoints and historical sources remain preserved. No merge or new experiment is part of the current coordination/cleanup.

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
# Supported commands and study interfaces; help launches no experiment.
uv run python scripts/train_mppi_gps.py --help
uv run python scripts/train_mppi_gps.py study jitter-diagnosis --help
uv run python scripts/train_mppi_gps.py walker-run --help
uv run python scripts/train_mppi_gps.py walker-replay --help

# Read-only provenance, source inventory and saved-bank headers.
uv run python scripts/train_mppi_gps.py inspect --run results/jitter_diagnosis_j2_001 --verify
uv run python scripts/train_mppi_gps.py review inventory --root .
uv run python scripts/train_mppi_gps.py review sample-bank --bank runs/2026-11-03_sample_bank_001

# Tests and saved-report replay from an existing clean export.
uv run python scripts/train_mppi_gps.py validate-export --export /path/to/clean-export
```

The sample bank is local external evidence; its command reports missing files explicitly when unavailable. Historical script paths remain only for documented compatibility. No command example above launches research.

## Reading order

1. [Investigation map and next research decision](docs/walker_coordination_review.md).
2. [Methods](docs/methods.md), [study reports and protocols](docs/README.md), and [primary-source review](docs/loop_v3_prior_art.md).
3. [Supported code and cleanup map](docs/code_map.md), then [debugging](docs/development.md).
4. [Result capsules](results/README.md), [external evidence availability](docs/evidence_availability.md), [machine-readable index](docs/evidence_index.json), and [history](docs/history.md).

Large arrays, ledgers, checkpoints and videos remain outside Git. H4.1 full raw evidence is local only. Checksum preservation and engineering tests do not substitute for scientific audit.
