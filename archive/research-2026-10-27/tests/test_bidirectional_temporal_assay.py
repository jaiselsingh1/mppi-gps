"""Finite schedule and raw block integrity; no physics or actor execution."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np
import pytest
from src.gps.bidirectional.audit import verify_block_proposals
from src.gps.bidirectional.search import SearchConfig
from src.gps.bidirectional.task import TaskConfig
from src.gps.bidirectional.temporal_assay import protocol


def test_declared_temporal_assay_has_exact_equal_control_expenditure():
    config=dict(task=asdict(TaskConfig()),search=asdict(SearchConfig(H=144)),backend='warp')
    result=protocol(config)
    assert result['max_controls']==235407776
    assert result['max_controls_per_condition']*2==result['max_controls']
    assert result['execution_order']==[['short_knots',0],['independent_blocks',0],['short_knots',1],['independent_blocks',1]]
    assert result['actor_rows']==result['optimizer_steps']==0
    changed=deepcopy(config);changed['task']['capture_distance']=.3
    with pytest.raises(ValueError,match='original H144'):protocol(changed)


def test_independent_raw_block_audit_rejects_intra_block_action_change():
    config=dict(K=2,smooth_candidates=2,memory_candidates=1,knot_spacing=1,independent_block_lengths=[2,4])
    actions=np.zeros((5,4,1));actions[2,:,0]=[-1.,-1.,.5,.5];actions[3,:,0]=.3
    assert verify_block_proposals(actions,config)==('independent_blocks:2','independent_blocks:4')
    actions[2,1,0]=-.8
    with pytest.raises(ValueError,match='changes inside a block'):verify_block_proposals(actions,config)


def test_legacy_bank_declares_no_block_channel():
    assert verify_block_proposals(np.zeros((2,3,1)),{'K':2})==()
