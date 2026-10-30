# Preservation and reproduction

Existing `runs/` and `artifacts/` remain at their original locations and are never rewritten to improve a result. Source/configuration frozen inside each run is authoritative.

- `archive/research-2026-10-27/`: the original pre-cleanup checkout, with its 792-file preservation manifest and explicit legacy launcher.
- `archive/research-2026-10-28/`: 48 subsequent experimental configurations and the previous long documentation. `preservation.json` records original paths, exact sizes and SHA-256 values. `runs`/`artifacts` links resolve to the original root evidence.
- [Pre-consolidation algorithm document](../artifacts/2026-10-28_algorithm_document_consolidation_001/before.md): the complete v35–v42 formulation history and prior-work discussion, preserved byte-for-byte before the current mechanism page was consolidated.

The [MuJoCo-only retirement snapshot](../archive/non_mujoco-2026-10-28/ARCHIVE.md) preserves the128-file source/configuration/test/documentation boundary before Ending was removed from active development. Its manifest verifies every copied byte. The two non-MuJoCo environment implementations and Ending-specific learning/orchestration are archived; original runs remain unchanged.

The active configuration is `configs/bidirectional_mujoco.json`. The public entry point supports MuJoCo `status`, `validate`, `qualify` and archive verification. Historical commands in preserved documents describe their original checkout and are not commands for the current CLI. Prefer the corresponding frozen run source/configuration when inspecting or reproducing past work; no old policy or result becomes current through archival access.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps verify-archive
.venv/bin/python -m scripts.training.train_bidirectional_gps status
```

The current archive verifier checks the MuJoCo-retirement snapshot. The earlier manifests remain preserved at their original archive locations. Active qualification uses the real MuJoCo Point Mass model; its results must be reported separately from the retired scalar experiments.
