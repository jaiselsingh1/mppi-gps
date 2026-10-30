"""Pure frozen-plan guard/accounting checks. Never run the pilot here."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import numpy as np
from types import SimpleNamespace
from src.gps.bidirectional import ending_pilot as pilot
from src.gps.bidirectional.search import MemoryFragment
from src.gps.bidirectional.ending_pilot import DEFAULT,load_pilot_config,pilot_work_bound,run_ending_pilot


def test_provisional_training_progress_never_deploys_a_rejected_actor():
    from src.gps.bidirectional.ending_pilot import _next_learners
    initial,first,second=object(),object(),object()
    training,deployed=_next_learners(initial,initial,first,False,True)
    assert training is first and deployed is initial
    training,deployed=_next_learners(training,deployed,second,False,True)
    assert training is second and deployed is initial
    training,deployed=_next_learners(training,deployed,second,True,True)
    assert training is deployed is second
    training,deployed=_next_learners(initial,initial,first,False,False)
    assert training is deployed is initial


def test_finite_config_matches_checked_in_specification_and_all_paid_caps():
    root=Path(__file__).resolve().parents[1]
    assert load_pilot_config(root/'configs/bidirectional_ending_pilot_v0.json')==DEFAULT
    bound=pilot_work_bound()
    assert bound['model_controls']==797892 and bound['executed_controls']==11658
    assert bound['total_controls']==DEFAULT['max_controls']==809550
    assert bound['actor_rows']==DEFAULT['max_actor_rows']==810608
    assert bound['adam_steps']==DEFAULT['max_adam_steps']==2056
    assert bound['diagnostic_backward_passes']==2 and DEFAULT['learning']['return_scale']==25


def test_config_rejects_silent_retuning_and_unfrozen_call_has_no_query_side_effects(tmp_path):
    changed=deepcopy(DEFAULT);changed['evaluation_roots']=128
    path=tmp_path/'retuned.json';path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='no automatic retuning'):load_pilot_config(path)
    run=tmp_path/'unfrozen';run.mkdir()
    with pytest.raises(FileNotFoundError):run_ending_pilot(DEFAULT,run)
    assert not (run/'evidence').exists() and not (run/'work_ledger.jsonl').exists()


def test_query_budget_rejection_is_durable_and_prevents_selector_fallback_without_physics():
    events=[]
    archive=SimpleNamespace(work_scope='budget_fixture',budget_failed=False,
        append=lambda kind,**kw:events.append(dict(kind=kind,**kw)))
    dynamics=pilot.CountedEnding(archive,object(),0)
    with pytest.raises(RuntimeError,match='cap exhausted'):
        dynamics._begin(np.array([0.,.3]),np.array([.1]),'fixture',{})
    assert archive.budget_failed is True and events[0]['kind']=='query_rejected'
    assert events[0]['before_physics'] is True and dynamics.reserved==0


def test_memory_snapshot_digest_supports_dataclasses_without_actor_or_model_calls():
    fragment=MemoryFragment(np.zeros(8),25,np.zeros((25,1)),'actual_teacher',0)
    snapshot=dict(memory=[fragment],nominal=np.zeros((4,1)))
    assert pilot.identity(snapshot)==pilot.identity(snapshot)
    assert 'actual_teacher' in pilot.packed(snapshot)


def test_candidate_qualification_revision_keeps_learning_work_and_declares_extra_controller_trials():
    root=Path(__file__).resolve().parents[1]
    config=load_pilot_config(root/'configs/bidirectional_ending_pilot_v2.json')
    bounds=pilot_work_bound(config)
    assert bounds['total_controls']==config['max_controls']==1108942
    assert bounds['actor_rows']==config['max_actor_rows']==961008
    assert bounds['adam_steps']==2056 and bounds['diagnostic_backward_passes']==2
    assert config['learning']==DEFAULT['learning'] and config['updates']==DEFAULT['updates']

