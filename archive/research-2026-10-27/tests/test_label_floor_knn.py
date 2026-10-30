"""Label floor from nearby states: the estimators, and that the data is Step 15's."""
from dataclasses import replace

import numpy as np

from scripts.diagnostics.label_floor_knn import FloorConfig, knn_errors, nugget, record_teacher
from scripts.training.train_walker_mdgps import TrainConfig, _collect, _new_actor


def test_knn_and_nugget_recover_the_noise_variance():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, (40_000, 2))
    y = np.stack([np.sin(2.0 * x[:, 0]), x[:, 1] ** 2], axis=1) + rng.normal(0.0, 0.3, (40_000, 2))
    errors, distance, half_gap = knn_errors(x[:30_000], y[:30_000], x[30_000:], y[30_000:], ks=(1, 20))
    # 1-NN pays the noise twice; 20-NN pays it once plus a twentieth.
    np.testing.assert_allclose(errors[1].mean(), 0.18, rtol=0.1)
    np.testing.assert_allclose(errors[20].mean(), 0.09 * (1 + 1 / 20), rtol=0.1)
    np.testing.assert_allclose(nugget(distance, half_gap, 0.25), 0.09, rtol=0.1)


def test_recorded_labels_are_step15_and_step16_labels():
    cfg = FloorConfig(K=8, H=6, steps=12)
    data = record_teacher(cfg, 0)
    train = TrainConfig(labeler="teacher", K=8, H=6, steps=12, workers=1)
    job = {"cfg": train, "actor": {k: v.cpu() for k, v in _new_actor(0).state_dict().items()},
           "reset_seed": cfg.seed_base, "planner_seed": cfg.seed_base + 100_000,
           "noise_seed": cfg.seed_base + 200_000, "execute_expert": True, "probe_probability": 0.0,
           "neff_fraction": 0.2}
    step15 = _collect(job)
    np.testing.assert_array_equal(data["obs"], step15["obs"])
    np.testing.assert_array_equal(data["executed"], step15["labels"])
    step16 = _collect(job | {"cfg": replace(train, label_neff_fraction=0.2)})
    np.testing.assert_array_equal(data["smoothed"], step16["labels"])
