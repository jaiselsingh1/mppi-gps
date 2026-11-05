# MPPI–policy learning on MuJoCo Walker

The goal is a cheap standalone walking policy taught by competent MPPI search, with feedback from learning improving subsequent acquisition. The central open question is **pre-update teachability**: can a score predict what the student will execute after a fixed learning update?

The requested [Acrobot controller-loop port](docs/acrobot_controller_loop.md) stopped at its teacher prerequisite: [0/10 fresh starts balanced](docs/acrobot_controller_teacher_result.md). All sixteen teacher episodes and the saved-evidence review completed. No Acrobot policy was fitted or alternating loop launched; the preserved controller-loop implementation remains the route after teacher qualification.

The subsequent [near-upright and energy probes](docs/acrobot_energy_discovery_result.md)
also stopped at teacher control: **0/4 near-upright holds and 0/6 energy swing-ups**.
All ten complete traces, the interrupted recording, saved-review correction and
complete costs are preserved; the independent saved review passed.

The [transient–attractor formulation and Acrobot study](docs/acrobot_attractor_results.md)
now establish a local upright LQR controller with **128/128** fresh local captures.
Adaptive MPPI achieved ESS32 but held from **0/4** starts; its two swing-ups reached
3.836/3.902 m yet made **0/2 basin entries**. The teacher gate remains unmet and no
learning loop ran. Original failures and the saved-only lock recovery are preserved.

The [eight-round result](docs/local_coupling_multiround_l3_1_result.md) completed its planned alternation: final standalone walking was 0/192, with retention in two coupled episodes. Its final audit failed and remains uncertified. The [saved-evidence design review](docs/coupled_loop_design_review.md) finds real changes to coupled teacher experience, but fixed historical restarts and a static-refitting alpha-zero control. It proposes a current-student corrective-data comparison; no follow-up is launched.

The [shared pilot brief and coordination log](docs/coordination/coupled_pilot_brief.md) connect the `sac-continuation` work. Our [evidence corrections and ownership response](docs/coordination/coupled_pilot_response.md) preserve contrary findings and distinguish proposed diagnostics from demonstrated learnability.

The authorized [saved-bank calibration](docs/coupling_calibration_t1_result.md) selected mean-prefix coupling alpha31.7349. The [four-state controller check](docs/coupling_controller_t0_result.md) completed all16 comparisons: coupling reduced policy disagreement but caused a paired teacher fall. The declared destabilization rule stopped progression before fitting. These are 300-control teacher outcomes, not standalone student walking.

The completed, independently audited [L2.0 result](docs/local_coupling_learning_l2_0_result.md) tested the actual H48 MPPI controller and matched policy learning. Alpha `.12` preserved local walking and improved teacher survival/cost, but all children still walked `0/16` on both fresh panels. Coupled data improved retention; the frozen walking criterion correctly prevented round two. [Protocol](docs/local_coupling_learning_l2_0_protocol.md), [evidence and curves](results/local_coupling_learning_l2_001/README.md), and [accounting](docs/local_coupling_learning_l2_0_accounting.md) preserve the entire result. Other branches remain unchanged.

The [A/B/C review](docs/full_pi_abc_coordination.md) and failed six-round H1000 acquisition remain evidence. P1.1's 20-round H1000 experiment stays held; it is not resumed by L2.0. Historical failures, raw sources and accounting remain preserved.

The original [P1 replay failure](docs/trajectory_pi_p1_result.md) remains preserved. [Origin-host and worker-native revalidation passed](docs/trajectory_pi_replay_resolution_result.md), producing separately bound native baselines for P1.1. Those results resolve its replay route, not the scientific similarity hypothesis. Its implementation and protocol remain intact. Recurrence stays stopped.

Start with the [coordinated investigation map](docs/walker_coordination_review.md). MPPI-taught competent walking and beneficial coupling remain unestablished. [J1](docs/similarity_j1_results.md) lacked outcome variation for its protected prediction comparison. [J2](docs/jitter_diagnosis_j2_results.md) produced three competent clean-reference clones but stopped at its frozen qualification gate; its six-student post-fit association is exploratory. The `sac-continuation` prototypes and their missing controls/audits are reviewed separately. These torque-policy studies are not a reproduction of CIO–GPS’s state-to-state controller.

Frozen runs, failed results, configurations, checkpoints and historical sources remain preserved. Branch boundaries remain intact. L2.0 authorizes only its frozen controller/learning/conditional-feedback roster, with no automatic third round or other intervention.

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
uv run python scripts/train_mppi_gps.py study trajectory-pi --help
uv run python scripts/train_mppi_gps.py study local-coupling-loop --help
uv run python scripts/train_mppi_gps.py walker-run --help
uv run python scripts/train_mppi_gps.py walker-replay --help

# Read-only provenance, source inventory and saved-bank headers.
uv run python scripts/train_mppi_gps.py inspect --run results/jitter_diagnosis_j2_001 --verify
uv run python scripts/train_mppi_gps.py inspect --run results/trajectory_pi_p1_001 --verify
uv run python scripts/train_mppi_gps.py review inventory --root .
uv run python scripts/train_mppi_gps.py review sample-bank --bank runs/2026-11-03_sample_bank_001

# Tests and saved-report replay from an existing clean export.
uv run python scripts/train_mppi_gps.py validate-export --export /path/to/clean-export
```

The sample bank is local external evidence; its command reports missing files explicitly when unavailable. Historical script paths remain only for documented compatibility. No command example above launches research.

## Reading order

1. [L2.0 end-to-end result](docs/local_coupling_learning_l2_0_result.md), its [frozen protocol](docs/local_coupling_learning_l2_0_protocol.md), then the [coordinated investigation map](docs/walker_coordination_review.md). The earlier [whole-trajectory PI question](docs/full_trajectory_pi_similarity.md) and [paper/optimizer review](docs/cio_optimizer_gap_review.md) remain preserved.
2. [Methods](docs/methods.md), [study reports and protocols](docs/README.md), and [primary-source review](docs/loop_v3_prior_art.md).
3. [Supported code and cleanup map](docs/code_map.md), then [debugging](docs/development.md).
4. [Result capsules](results/README.md), [external evidence availability](docs/evidence_availability.md), [machine-readable index](docs/evidence_index.json), and [history](docs/history.md).

Large arrays, ledgers, checkpoints and videos remain outside Git. H4.1 full raw evidence is local only. Checksum preservation and engineering tests do not substitute for scientific audit.
