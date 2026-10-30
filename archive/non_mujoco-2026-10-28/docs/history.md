# Preservation and reproduction

Existing `runs/` and `artifacts/` remain at their original locations and are never rewritten to improve a result. Source/configuration frozen inside each run is authoritative.

- `archive/research-2026-10-27/`: the original pre-cleanup checkout, with its 792-file preservation manifest and explicit legacy launcher.
- `archive/research-2026-10-28/`: 48 subsequent experimental configurations and the previous long documentation. `preservation.json` records original paths, exact sizes and SHA-256 values. `runs`/`artifacts` links resolve to the original root evidence.
- [Pre-consolidation algorithm document](../artifacts/2026-10-28_algorithm_document_consolidation_001/before.md): the complete v35–v42 formulation history and prior-work discussion, preserved byte-for-byte before the current mechanism page was consolidated.

The active root now presents one current configuration, `configs/bidirectional_ending.json`, and one public research entry point. Historical profiles still exist as compatibility code; archived configurations must be selected explicitly from the repository root. Historical documentation preserves its original command text, so prefix configuration paths with `archive/research-2026-10-28/` when using the current checkout.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps verify-archive
.venv/bin/python -m scripts.training.train_bidirectional_gps legacy scripts.training.train_bidirectional_gps --help
```

The legacy launcher uses the preserved pre-cleanup import environment. It does not make a historical algorithm current. For exact later reproductions, prefer the corresponding frozen run source and configuration.
