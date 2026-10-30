# Evidence availability

Git contains the active source, tests, configuration, historical source snapshots, result summaries and selected figures. Generated `runs/`, `artifacts/`, checkpoints, videos and full query ledgers are preserved outside the Git checkout.

Links into `runs/` or `artifacts/` in the reports refer to that local/remote evidence store. They become available when the corresponding evidence is present; cloning the source does not download it. No evidence was deleted to prepare the cleanup commit.

The canonical experiment host is `robopil5090`:

| Path | Contents |
|---|---|
| `/home/jaisel/mppi-gps-workspace/current/mppi-gps/` | Active working code |
| `/home/jaisel/mppi-gps-workspace/runs/` | Frozen experiments, checkpoints and full raw evidence |
| `/home/jaisel/mppi-gps-workspace/history/artifacts/` | Audits and the focused policy diagnosis |

The versioned [evidence index](evidence_index.json) records representative result/configuration hashes and the full raw-HDF locators and hashes already produced by the audits. It contains metadata, not model-query arrays.

Core tests do not require historical training data:

```sh
uv sync --extra test
uv run --extra test python -m pytest -q
```

`baseline` uses the included Walker model/configuration. `train --stage fit` and `train --stage feedback` require the historical v1 evidence described in the README; `evaluate` requires a supplied checkpoint. Source snapshots remain separate from generated evidence.
