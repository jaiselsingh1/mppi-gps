# MPPI–GPS: one Walker loop

The active implementation adapts the alternating trajectory/policy optimization of the **2015 CIO paper** to MuJoCo Walker and MPPI. It fits a bounded action policy to an adapting teacher. Action fitting is intentional. The independent planner retains its proposals, RNG and task-only nominal updates.

**First pilot:** independent discovery 4/4; trained standalone policy 0/16 held successes. Guidance stayed off. See the [results and videos](docs/results.md).

Start with the [algorithm](docs/algorithm.md), [paper mapping](docs/paper_mapping.md), [code map](docs/code_map.md) and [measured results](docs/results.md).

~~~sh
# Four independent walking episodes.
.venv/bin/python scripts/train_mppi_gps.py baseline --out runs/walker_baseline_001

# Discovery, fitting, standalone qualification and up to two feedback rounds.
.venv/bin/python scripts/train_mppi_gps.py train --out runs/walker_cio_mppi_001

# Fresh actor-only evaluation.
.venv/bin/python scripts/train_mppi_gps.py evaluate --checkpoint runs/walker_cio_mppi_001/checkpoints/final.pt --out runs/walker_actor_eval_001

# Render recorded states, adding no controls.
.venv/bin/python scripts/train_mppi_gps.py render --run runs/walker_cio_mppi_001 --out runs/walker_cio_mppi_001/walker.mp4
~~~

Each experiment creates a source/configuration snapshot and runs from it. Output directories are create-only. Read result.json, stdout.log, curves.png and evidence/. An unsuccessful scientific gate is a recorded result, not a software exception.

The first pilot is ordinary walking. It makes no claim about beyond-horizon ending behavior or sample efficiency. Prior experiments remain [historical evidence](docs/history.md).

Install with uv sync; add --extra test for tests or --extra gpu for optional Warp. Current physical experiments use native CPU MuJoCo; GPU simulation needs separate qualification.
