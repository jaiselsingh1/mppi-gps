"""Small real-MuJoCo integration and controller authority checks."""
from pathlib import Path
import json
import hashlib

import numpy as np
import pytest


def test_preserved_reset_bytes():
    root = Path(__file__).resolve().parents[1]
    archive = root/'archive/pre_cio_reset-2026-10-29'
    manifest = json.loads((archive/'manifest.json').read_text())
    assert len(manifest['files']) == 110
    for row in manifest['files']:
        assert hashlib.sha256((archive/row['file']).read_bytes()).hexdigest() == row['sha256']


def test_actual_walker_bank_restore_and_meter(tmp_path):
    from src.envs.walker2d import Walker2d
    from src.utils.config import MPPIConfig
    from src.gps.mppi_gps.runtime import Meter, snapshot, restore
    from src.gps.mppi_gps.search import IndependentMPPI, CoupledSelector
    env = Walker2d(nthread=1, cost_style='target_velocity', target_velocity=1.5)
    meter = Meter(tmp_path/'evidence', max_controls=1000, max_neural_rows=100)
    meter.instrument(env)
    try:
        np.random.seed(42); env.reset()
        physical = snapshot(env)
        planner = IndependentMPPI(env, MPPIConfig(K=4,H=4,initial_iterations=1), 17)
        private = planner.snapshot()
        first = planner.propose(env.get_state(), env.get_warmstart())
        np.testing.assert_array_equal(first.observations[:,0],
            np.broadcast_to(env._get_obs(), (5,17)))
        plan, info = CoupledSelector(alpha=1.,eta_u=.01).select(first, actor=None,
            meter=meter, enabled=False)
        assert not info['influence_used']
        expected = env.batch_rollout(env.get_state(), plan[None,:1],
                                    initial_warmstart=env.get_warmstart())[0][0,0]
        env.step(plan[0])
        np.testing.assert_allclose(env.get_state(),expected,rtol=0.,atol=1e-12)
        restore(env, physical); planner.restore(private)
        repeated = planner.propose(env.get_state(),env.get_warmstart())
        assert first.bank_digest == repeated.bank_digest
        assert meter.totals()['completed_live_controls'] == 1
        assert meter.totals()['failed_queries'] == 0
    finally:
        env.close(); meter.close()


def test_authority_policy_change_revokes_without_network_call(tmp_path):
    import torch
    from src.gps.mppi_gps.policy import Policy
    from src.gps.mppi_gps.runtime import Meter
    from src.gps.mppi_gps.loop import Authority
    actor = Policy(seed=1)
    meter = Meter(tmp_path/'evidence')
    try:
        support = Authority(actor,np.zeros((2,17)),1.,meter,'same_task')
        support.promoted = True
        assert support.allowed(actor,np.zeros(17),False)
        with torch.no_grad():
            next(actor.parameters()).add_(.1)
        assert not support.allowed(actor,np.zeros(17),False)
        assert not support.valid
        assert meter.totals()['requested_neural_rows'] == 0
    finally:
        meter.close()


def test_complete_cost_gain_cannot_override_survival_loss():
    from src.gps.mppi_gps.loop import cost_improved
    baseline = dict(survived=2,mean_cost=100.)
    assert not cost_improved(dict(survived=1,mean_cost=1.),baseline,.001)
    assert cost_improved(dict(survived=2,mean_cost=99.),baseline,.001)
