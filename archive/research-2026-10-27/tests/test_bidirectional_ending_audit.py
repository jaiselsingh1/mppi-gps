"""Synthetic arithmetic/decoding tests: no simulator or neural actor imports."""
import io
import json
import pickle
import zipfile

import numpy as np
import pytest

from src.gps.bidirectional.ending_audit import (Evidence, checkpoint_numpy,
    observations, policy_identity, resolutions, transition_arithmetic)


TASK = dict(gate=12, delay=12, move=.2, effort=.02, greed=.2,
            detour=.2, detour_steps=5, penalty=20., ending_y=-.1)


def test_ending_cost_exactly_once_and_absorbing_flags():
    state = np.array([[24., .2], [25., .2], [24., -.2]])
    post, parts, costs, flags = transition_arithmetic(state, np.zeros((3, 1)), TASK)
    np.testing.assert_array_equal(post[:, 0], [25., 25., 25.])
    np.testing.assert_array_equal(parts['ending'], [20., 0., 0.])
    np.testing.assert_array_equal(costs, [20., 0., 0.])
    np.testing.assert_array_equal(flags['ending_event'], [True, False, True])
    np.testing.assert_array_equal(flags['ending_good'], [False, False, True])


def test_post_action_greed_and_gate_freeze():
    post, parts, costs, flags = transition_arithmetic(
        np.array([[0., .3], [12., -.3]]), np.array([[-1.], [1.]]), TASK)
    np.testing.assert_allclose(post, [[1., .1], [13., -.3]])
    np.testing.assert_allclose(parts['greed'], [.09, 0.])
    np.testing.assert_allclose(costs, [.11, .2])
    np.testing.assert_array_equal(flags['good_lane'], [False, True])


@pytest.mark.parametrize('state,action', [([0., 1.1], [0.]), ([.5, 0.], [0.]),
                                        ([0., 0.], [1.0001]), ([0., 0.], [float('nan')])])
def test_transition_rejects_invalid_bounds(state, action):
    with pytest.raises(ValueError): transition_arithmetic(state, action, TASK)


def test_observation_boundary_alignment():
    obs = observations(np.array([[0., .3], [25., -.2]]), 25)
    np.testing.assert_array_equal(obs[:, :3], [[.3, 0., 1.], [-.2, 1., 0.]])
    assert not obs[:, 3:].any()


def test_hash_binding_and_object_npz_rejection(tmp_path):
    path = tmp_path/'raw.npz'
    np.savez(path, values=np.array([1.]))
    ref = dict(file='raw.npz', sha256='0'*64, bytes=path.stat().st_size)
    with pytest.raises(ValueError, match='SHA'): Evidence(tmp_path).npz('.', ref)


def test_work_resolution_preserves_failed_and_unresolved_requests():
    rows = [dict(id='r0', kind='optimizer_requested', updates=1, utc='2026-01-01T00:00:00+00:00'),
            dict(id='r1', kind='optimizer_requested', updates=1, utc='2026-01-01T00:00:01+00:00'),
            dict(id='e', kind='optimizer_error', request_id='r0', updates=1, utc='2026-01-01T00:00:02+00:00')]
    pending = []
    requests, resolved = resolutions(rows, 'optimizer', 'updates', pending)
    assert sum(r['updates'] for r in requests.values()) == 2
    assert len(resolved) == 1 and len(pending) == 2


def test_restricted_checkpoint_rejects_arbitrary_pickle_global(tmp_path):
    path = tmp_path/'unsafe.pt'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('archive/byteorder', b'little')
        z.writestr('archive/data.pkl', b'cos\nsystem\n.')
    with pytest.raises(ValueError, match='prohibited'): checkpoint_numpy(path)


def test_checkpoint_policy_hash_binds_parameters():
    policy = dict(policy_type='bidirectional-family-gaussian-v1', format_version=1,
        obs_dim=8, hidden_dim=128, num_families=2, actor={'weight':np.array([0.], dtype=np.float32)}, policy_hash='0'*64)
    with pytest.raises(ValueError, match='owner hash'): policy_identity(policy)


def test_float32_native_action_uses_float64_transition_contract():
    action=np.array([.38721603],dtype=np.float32)
    state=np.array([0.,.3],dtype=np.float64)
    post,_,cost,_=transition_arithmetic(state,action,TASK)
    expected_y=.3+.2*float(action[0])
    assert post[1]==expected_y
    assert float(cost)==pytest.approx(.02*float(action[0])**2+.1*(1-expected_y),abs=2e-12)


@pytest.mark.parametrize('protocol',[2,4])
def test_restricted_checkpoint_decodes_numeric_boolean_without_torch(tmp_path,protocol):
    path=tmp_path/'numeric.pt'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('archive/byteorder',b'little')
        archive.writestr('archive/data.pkl',pickle.dumps({'accepted':np.bool_(True)},protocol=protocol))
    assert checkpoint_numpy(path)=={'accepted':True}


@pytest.mark.parametrize('protocol',[2,4])
def test_restricted_checkpoint_decodes_numeric_rng_array(tmp_path,protocol):
    path=tmp_path/'rng.pt';values=np.arange(12,dtype=np.uint64).reshape(3,4)
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('archive/byteorder',b'little')
        archive.writestr('archive/data.pkl',pickle.dumps({'rng':values},protocol=protocol))
    np.testing.assert_array_equal(checkpoint_numpy(path)['rng'],values)


def test_restricted_checkpoint_rejects_object_array(tmp_path):
    path=tmp_path/'object.pt'
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('archive/byteorder',b'little')
        archive.writestr('archive/data.pkl',pickle.dumps(np.array([None],dtype=object),protocol=2))
    with pytest.raises(ValueError,match='Unsupported serialized numeric dtype'):
        checkpoint_numpy(path)
