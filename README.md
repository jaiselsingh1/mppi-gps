# MPPI–policy learning on MuJoCo Walker

This repository studies whether independent MPPI discovery can teach a global torque policy, and whether feedback from learning can improve later trajectory optimization. The protected MPPI proposal bank remains policy independent.

The completed MPPI-to-student and coupling studies remain negative for competent standalone walking. The [v3.1 matched data comparison](docs/loop_v3_results.md) tested recorded demonstrations; independent data improved survival without walking, while guided data lost that survival benefit. [L1 and H4.1](docs/walker_experiments_l1_h4_results.md) separately tested fresh student feature pipelines and feedback-selected agreement-weight backoff. All six H4 teachers walked, but all three children regressed against their incoming policy. No confirmation or second round ran. H3.2 and H4.2 remain untested. These torque-policy results do not establish a failed reproduction of the paper's state-to-state controller.

The [K1 compatibility diagnostic](docs/action_compatibility_k1_results.md) compares successful MPPI, random-action and qualified-reference trajectories against untrained, cloned and independently trained policies. An existing SAC reference walks **16/16 primary + 16/16 confirmation**, despite much worse MPPI action agreement than the failing clones. No new policy was trained. The result separates agreement from competence without establishing that trajectory diversity caused cloning failure.

The [W1 width contrast](docs/student_width_w1_results.md) is also closed: widening 128 to 256 tanh units reduced training error by about 48%, increased held-trajectory error by about 24%, and produced no walking in any of three paired seed comparisons. All teachers remained successful. The wider policies fell sooner; W1 confirmation was not triggered.

The [W2 update-budget contrast](docs/student_update_w2_results.md) confirmed that 2k checkpoints retain more controls and have lower complete cost than their continued 10k counterparts, across three fresh streams and a separate confirmation panel. Every endpoint still has **0/16 walking and full survival** on each panel. This is partial retention, not competent walking. W2 launched no follow-up.

The [S1 source comparison](docs/student_source_s1_results.md) completed six matched fits, then the host rebooted during confirmation. Its original evidence is preserved; the full comparison remains incomplete. [S1.1 is complete and independently audited](docs/student_source_s1_1_results.md): the reference walked 32/32; MPPI-data children walked 0/96; reference-data children walked 5/96, all from one training seed. Neither source produced a competent student, and the source-benefit criterion failed. All 320 episodes were retained with closed off-host backups. No new fitting or coupling round ran. The [fixed-reference K1 trajectory review](docs/action_compatibility_k1_results.md#fixed-reference-trajectory-spread-review) distinguishes within-source score spread from the established group ordering. The preceding [D1/D2 diagnosis](docs/transfer_diagnosis_d1d2_results.md) remains complete and audited.

[K1.1 is complete and independently audited](docs/trajectory_review_k1_1_results.md): some close full-trajectory agreement scores conceal large, opposing local differences. All 34 trajectories and 304 ordered bins reconcile. The [capsule](results/trajectory_review_k1_1/README.md) preserves both audit/report attempts and their sources. No training, inference, physics or coupling ran. The later S1.2 validity result is below; its learning contrast remains untested.

[K1.2 is complete and independently audited](docs/motion_diagnosis_k1_2_results.md): all 34 actual motions and 561 cross-trajectory pairs are preserved with full and first-100-control views. The [capsule](results/motion_diagnosis_k1_2/README.md) contains both 34-page motion atlases, complete tables and costs. No new inference, integration, fitting or acquisition ran; this diagnosis does not establish a transfer cause or useful coupling.

[H5.0r1 is complete and independently audited](docs/teacher_proposal_h5_0_round1_results.md). Independent teachers walked 12/12; selector-only and proposal-guided teachers each walked 11/12. The frozen competence stop ended acquisition before any child fit, standalone panel or paired-gap episode, so fixed-update learning remains unmeasured. The [result capsule](results/teacher_proposal_h5_0_round1/README.md) preserves the evidence and costs. The [frozen launched source](docs/validation/teacher_proposal_h5_0_round1_2026-11-01/prelaunch/README.md) passed 987 tests with the existing skip and expected failure. The useful-learning prerequisite remains unmet, with no loop or second round.

[C1.0 completed its fixed worker roster](docs/student_consequence_c1_results.md): all twelve teachers replayed successfully, but parents and uniform, consequence-weighted and shuffled children all failed walking and full survival. [Audit-only recovery is now resolved](docs/student_consequence_c1_audit_recovery_results.md) through a separate certificate closure; both the original and first recovery coordinator failures remain recorded. No physics, fitting or evaluation was repeated. [S1.2 V is complete and independently audited](docs/student_visitation_s1_2_v_results.md): reference reset controls walked 32/32; noninitial takeovers walked 27/72 and fully survived 46/72, versus zero in parent replays. The fixed recovery screen passed in the same two streams on both panels. This supports useful reference behavior, not useful labels or learning. The [capsule](results/student_visitation_s1_2_v/README.md) preserves both preparation and publication failures; conditional learning remains review-gated.

[J1.0](docs/similarity_j1_protocol.md) is the next authorized bounded investigation: five pre-update scores predict a shared set of fixed-update learning outcomes. It keeps walking, motion reproduction and survival separate, with disjoint development and protected groups. Its prospective protocol is frozen before execution; preparation and calibration alone are not a scientific result.

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
uv run python scripts/train_mppi_gps.py study width --help
uv run python scripts/train_mppi_gps.py study update-budget --help
uv run python scripts/train_mppi_gps.py study transfer-diagnosis --help
uv run python scripts/train_mppi_gps.py study source --help
uv run python scripts/train_mppi_gps.py study source-evaluation --help
uv run python scripts/train_mppi_gps.py study trajectory-review --help
uv run python scripts/train_mppi_gps.py study motion-diagnosis --help
uv run python scripts/train_mppi_gps.py study teacher-proposal --help
uv run python scripts/train_mppi_gps.py study teacher-proposal census
uv run python scripts/train_mppi_gps.py study student-consequence --help
uv run python scripts/train_mppi_gps.py study visitation --help
uv run python scripts/train_mppi_gps.py study similarity-investigation --help
uv run python scripts/train_mppi_gps.py --help
```

The public CLI retains `baseline`, `train`, `evaluate` and `render`. Study recipes live in [configs/studies](configs/studies); historical input paths must be supplied explicitly. Preparation creates a new frozen source/configuration/input binding, and workers execute that frozen source. Choose fresh output directories; see [safe debugging and replay](docs/development.md#current-source-and-frozen-workers). Cloning does not fetch bulk evidence or authorize a new experiment.

K1.1's [prepare/run/audit/report commands](docs/development.md#saved-only-k11-review) operate on saved K1 JSON/CSV inputs. Audit and report outputs belong outside the immutable run or capsule.

## Reading order

1. [Research status and document map](docs/README.md), then [methods and interfaces](docs/methods.md).
2. [C1.0 findings and audit failure](docs/student_consequence_c1_results.md), [execution contract](docs/student_consequence_c1_execution.md), [H5.0r1 result](docs/teacher_proposal_h5_0_round1_results.md), [K1.2 actual-motion diagnosis](docs/motion_diagnosis_k1_2_results.md), [K1.1 audited review](docs/trajectory_review_k1_1_results.md), [S1.1 fresh evaluation](docs/student_source_s1_1_results.md), [S1 interrupted comparison](docs/student_source_s1_results.md), [D1/D2 diagnosis](docs/transfer_diagnosis_d1d2_results.md), [W2 fixed-budget result](docs/student_update_w2_results.md), [W1 width result](docs/student_width_w1_results.md), [K1 comparison and verdict](docs/action_compatibility_k1_results.md), [L1/H4.1 results](docs/walker_experiments_l1_h4_results.md) and [v3.1 comparison](docs/loop_v3_results.md).
3. [Exact result capsules](results/README.md), [evidence availability](docs/evidence_availability.md) and [machine-readable index](docs/evidence_index.json).
4. [Code map](docs/code_map.md), [debugging guide](docs/development.md), [primary-source review](docs/loop_v3_prior_art.md) and [historical record](docs/history.md).

Frozen runs, checkpoints, ledgers, raw arrays and historical source are preserved. The tracked capsules contain exact small evidence copies; H4.1 full raw evidence is **local only**. No beneficial loop closure, population-wide success or efficiency result is claimed.
