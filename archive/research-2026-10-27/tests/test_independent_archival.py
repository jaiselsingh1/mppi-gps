"""Rejected fits must remain reviewable without reconstructing their weights."""
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np

from src.gps.independent_core import Memory
from scripts.training.run_independent_search import Controller, fit


def test_fit_archives_candidate_before_any_calibration_or_qualification(tmp_path):
    (tmp_path / "controllers").mkdir()
    config = json.loads((Path(__file__).parents[1] / "configs/independent_search_pilot.json").read_text())
    saved, events = [], []
    ev = SimpleNamespace(cfg=config, out=tmp_path, save_controller=saved.append,
                         log=lambda **row: events.append(row))
    states = np.array([[-.5, 0.], [-.49, .01], [-.48, .01]])
    traces = [dict(states=states, costs=np.array([1., 1.]), returns=np.array([2., 1.]),
                   remaining_start=2, root=1, id="measured", forced=False)]
    reference = Controller(Memory([]))
    value = fit(ev, reference, "mc", traces, [], seed=0, cycle=0)
    assert len(saved) == 1
    assert saved[0].value.digest == value.digest
    assert saved[0].value.target_version == reference.version
    assert events[0]["fit_updates"] == 1
