"""A resumed run continues with the per-iteration seeds an unbroken run uses."""
import json

from scripts.training.train_walker_mdgps import TrainConfig, main

_SMALL = dict(K=8, H=4, steps=10, episodes_per_iter=2, label_every=2, workers=1, grad_steps=5,
              eval_every=100, eval_episodes=1, iterations=3)


def _rows(path):
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines() if line.strip()]


def test_resume_continues_with_the_seeds_an_unbroken_run_uses(tmp_path):
    main(tmp_path / "unbroken", TrainConfig(**_SMALL))
    broken = tmp_path / "broken"
    main(broken, TrainConfig(**_SMALL))
    # Simulate a crash after iteration 1.
    lines = (broken / "metrics.jsonl").read_text().splitlines()
    (broken / "metrics.jsonl").write_text("\n".join(lines[:2]) + "\n")
    (broken / "policy_002.pt").unlink()
    (broken / "status.json").unlink()
    main(broken, TrainConfig(**_SMALL, resume=True))
    unbroken, resumed = _rows(tmp_path / "unbroken"), _rows(broken)
    assert [row["iteration"] for row in resumed] == [0, 1, 2]
    assert resumed[2]["resumed"] and not resumed[1]["resumed"]
    for key in ("collect_mean_steps", "collect_cost_per_step", "labels"):
        assert resumed[2][key] == unbroken[2][key]
    assert json.loads((broken / "status.json").read_text())["resumed_at"] == 2
