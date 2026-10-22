"""The GPU-labelling collector must visit exactly the states the CPU path visits."""
from dataclasses import replace

import numpy as np

from scripts.training.train_walker_mdgps import TrainConfig, _collect, _new_actor


def test_collect_only_mode_visits_the_cpu_paths_states():
    cfg = TrainConfig(K=16, H=4, steps=30, label_every=2, workers=1)
    actor = _new_actor(0)
    job = {"actor": {k: v.cpu() for k, v in actor.state_dict().items()}, "reset_seed": 7,
           "planner_seed": 11, "noise_seed": 13, "execute_expert": False,
           "probe_probability": 0.5, "neff_fraction": 0.2}
    cpu = _collect({**job, "cfg": cfg})
    collect_only = _collect({**job, "cfg": replace(cfg, gpu_estep=True)})
    np.testing.assert_array_equal(cpu["obs"], collect_only["obs"])
    assert cpu["steps"] == collect_only["steps"]
    assert len(collect_only["pending"]) == len(cpu["labels"])
    assert [stream for _, _, stream in collect_only["pending"]] == [(11, j) for j in range(len(cpu["labels"]))]
    assert sum(collect_only["probe_flags"]) == len(cpu["probes"])
