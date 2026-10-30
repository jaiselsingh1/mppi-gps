"""Actual-return promotion, reset ownership and honest evaluation summaries."""
from dataclasses import replace

import numpy as np
import pytest

from scripts.training.train_coupled_value import CoupledConfig, _promote, _summary, validate


def panel(costs, survivors=None):
    if survivors is None:
        survivors=[True]*len(costs)
    return [dict(reset_id=100+i,cost=cost,survived=survived,steps=1000 if survived else 50,
                 duration_seconds=8. if survived else .4,velocity_mae=.2,
                 guidance_fraction=.5) for i,(cost,survived) in enumerate(zip(costs,survivors))]


def test_promotion_requires_own_full_return_gain_over_both_references():
    good=panel([8.,7.,9.,6.])
    assert _promote(good,panel([11.,10.,11.,10.]),panel([12.,11.,13.,12.]),1.)
    # Beating the memory ablation alone is insufficient if incumbent is better.
    assert not _promote(good,panel([11.,10.,11.,10.]),panel([6.,5.,7.,4.]),1.)


def test_promotion_cannot_trade_lost_completions_for_lower_mean_cost():
    assert not _promote(panel([1.,1.],[True,False]),panel([10.,10.]),panel([10.,10.]),0.)


def test_promotion_rejects_unpaired_or_empty_measurements():
    for a,b,c in [(panel([1.]),panel([2.,2.]),panel([2.])),([],[],[])]:
        with pytest.raises(ValueError,match="matched"):
            _promote(a,b,c,0.)
    wrong=panel([2.]); wrong[0]["reset_id"]=999
    with pytest.raises(ValueError,match="matched"):
        _promote(panel([1.]),wrong,panel([2.]),0.)


def test_walking_gate_includes_failed_episodes_and_requires_full_horizon():
    rows=panel([100.,200.],[True,False])
    result=_summary(rows,1000)
    assert result["survival_count"] == 1 and result["survival_rate"] == .5
    assert result["mean_duration_seconds"] == 4.2 and result["finite_discounted_cost"] == 150.
    assert not result["walking_gate"]
    assert _summary(panel([1.,1.]),64)["walking_gate"] is None


@pytest.mark.parametrize("change",[
    {"seed":.5},{"seed":100},{"iterations":0},{"H":1},
    {"H":1001},{"alpha_step":0.},{"alpha_step":.6},
    {"support_radius":np.nan},{"endpoint_rmse_max":np.nan},{"velocity_noise":-1.},
])
def test_configuration_fails_before_acquisition(change):
    with pytest.raises(ValueError):
        validate(replace(CoupledConfig(),**change))
