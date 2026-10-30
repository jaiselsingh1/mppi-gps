"""Independent saved-tensor fixtures only; no Torch/policy/simulator imports."""
from copy import deepcopy
import numpy as np
import pytest

from src.gps.bidirectional.ending_interpolation_audit import tensor_interpolation


PARAMETERS=["log_std","fc1.weight","fc1.bias","fc2.weight","fc2.bias","fc_mean.weight","fc_mean.bias"]


def _policy(offset=0.):
    arrays={name:np.array([.10234567+offset],np.float32) for name in PARAMETERS}
    arrays.update(action_low=np.array([-1.],np.float32),action_high=np.array([1.],np.float32),
        action_scale=np.array([1.],np.float32),action_bias=np.array([0.],np.float32))
    return dict(format_version=1,policy_type="fixture",obs_dim=8,hidden_dim=1,num_families=2,
        action_low=[-1.],action_high=[1.],actor=arrays,sampling_rng_state=np.array([1,2,3],np.uint8))


@pytest.mark.parametrize("alpha",[0.,.0625,.125,.25,.5,1.])
def test_independent_serialized_interpolation_all_parameters_and_buffers(alpha):
    parent=_policy();child=_policy(-.91234567);candidate=deepcopy(parent)
    for name in PARAMETERS:
        a,b=parent["actor"][name],child["actor"][name]
        candidate["actor"][name]=a.copy() if alpha==0 else b.copy() if alpha==1 else (
            a.astype(np.float64)+alpha*(b.astype(np.float64)-a.astype(np.float64))).astype(a.dtype)
    tensor_interpolation(parent,child,candidate,alpha,PARAMETERS)
    candidate["actor"]["log_std"][0]+=.01
    with pytest.raises(ValueError,match="interpolation bytes"):
        tensor_interpolation(parent,child,candidate,alpha,PARAMETERS)


def test_independent_interpolation_rejects_undeclared_buffer_or_private_rng_change():
    parent=_policy();child=_policy();candidate=deepcopy(parent)
    child["actor"]["action_scale"][0]=.9
    with pytest.raises(ValueError,match="buffer changed"):
        tensor_interpolation(parent,child,candidate,.5,PARAMETERS)
    child=_policy();candidate["sampling_rng_state"][0]=9
    with pytest.raises(ValueError,match="private RNG"):
        tensor_interpolation(parent,child,candidate,.5,PARAMETERS)
    candidate=deepcopy(parent)
    with pytest.raises(ValueError,match="roster"):
        tensor_interpolation(parent,child,candidate,.5,PARAMETERS[:-1])
