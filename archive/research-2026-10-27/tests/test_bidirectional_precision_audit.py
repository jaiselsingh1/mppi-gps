"""Synthetic saved-record checks only; no simulator or actor calls."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from src.gps.bidirectional.ending_completion_audit import behavior_stem,is_mc_fit
from src.gps.bidirectional.ending_precision_audit import _paired,_behavior_digest


@pytest.mark.parametrize("arm",["original","precision"])
@pytest.mark.parametrize("child",["raw","guided"])
def test_precision_mc_dispatch_is_arm_and_phase_bound(arm,child):
    purpose=f"precision_{arm}_matched_child_{child}"
    assert is_mc_fit(purpose)
    assert behavior_stem(purpose,f"precision_{arm}_child_fit")==f"precision_{arm}_own_"
    with pytest.raises(ValueError):behavior_stem(purpose,"precision_foreign_child_fit")


def test_precision_decision_requires_cost_and_capture_on_identical_roots():
    base=[dict(context_id="same",cost=2.,capture=True)]
    candidate=[dict(context_id="same",cost=1.,capture=False)]
    assert not _paired(base,candidate,.001)["improved"]
    candidate[0]["capture"]=True
    assert _paired(base,candidate,.001)["improved"]
    candidate[0]["context_id"]="foreign"
    with pytest.raises(ValueError):_paired(base,candidate,.001)


def test_saved_behavior_digest_reconstructs_complete_dataclass(tmp_path):
    # Creating the record does not invoke a model, policy or optimizer.
    from src.gps.bidirectional.learning import BehaviorEpisode
    from src.gps.bidirectional.feedback import _digest
    import json
    raw=dict(observations=np.zeros((3,8)),actions=np.zeros((2,1),np.float32),
        pre_tanh=np.zeros((2,1),np.float32),old_log_probs=np.array([-.1,-.2]),
        raw_costs=np.array([.3,20.]),owner_json=np.array(json.dumps(dict(
            actor_hash="actor",controller_hash="controller"))))
    class Evidence:
        root=tmp_path
        def npz(self,base,ref):return deepcopy(raw)
    ref=dict(file="episodes/own.npz",sha256="unused",bytes=1)
    panel=[dict(context_id="context",raw_ref=ref)]
    episode=BehaviorEpisode(raw["observations"][:-1],raw["actions"],raw["pre_tanh"],
        raw["old_log_probs"],raw["raw_costs"],"context",2,"actor","controller",
        query_ref=str(tmp_path/"evidence/assay/episodes/own.npz"))
    assert _behavior_digest(Evidence(),Path("evidence/assay"),panel)==_digest([episode])
