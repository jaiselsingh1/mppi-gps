"""Executable credit, data-boundary and selection falsifiers; no engines."""
from dataclasses import fields,dataclass
from copy import deepcopy
import json
import numpy as np
import pytest
from src.gps.bidirectional.ending_pilot import CountedEnding as _ProductionEnding
from src.gps.bidirectional.return_curriculum import (StateExperience,reset_context,comparison,
    select_student,load_config,DEFAULT,GOAL_DEFAULT,SUPPORT_DEFAULT,work_bound,goal_reset_context,suffix_eligibility)
from src.gps.bidirectional.learning import BehaviorEpisode,build_mc_batch


class RawLedger:
    def __init__(self,path):
        self.path=path;self.stream=path.open('x');self.scope='curriculum-continuation-test';self.sequence=0
    def append(self,kind,**fields):
        identifier=f'raw{self.sequence}';self.sequence+=1
        row=dict(id=identifier,kind=kind,**fields)
        self.stream.write(json.dumps(row,default=lambda a:np.asarray(a).tolist())+'\n');self.stream.flush()
        return identifier


class CountedEnding(_ProductionEnding):
    """Expose this test's production transitions to the global test hook."""
    def transition(self,states,actions):
        n=int(np.prod(np.shape(states)[:-1])) if np.ndim(states)>1 else 1
        request=self.ledger.append('transition_attempt',controls=n,prestates=states,actions=actions)
        result=super().transition(states,actions)
        self.ledger.append('transition_complete',request_id=request,states=result[0],components=result[1],costs=result[2])
        return result


def test_learner_search_evidence_cannot_contain_action_or_teacher_return_labels():
    states=np.column_stack((np.arange(26),np.linspace(.3,-.12,26)))
    e=StateExperience('actual-search',states,{'raw':'complete-executed-trace'})
    assert [f.name for f in fields(e)]==['episode','states','source']
    with pytest.raises(TypeError):StateExperience('search',states,{},actions=np.ones((25,1)))
    with pytest.raises(ValueError):e.states.flags.writeable=True
    context,source=reset_context(e,11,0,'student-owned-reset')
    assert context.elapsed==11 and context.state[0]==11
    assert context.state[1]==pytest.approx(states[11,1]-.025)
    assert source['teacher_actions_used'] is False


def behavior(cost,ref,owner='behavior'):
    raw=np.zeros(25);raw[-1]=cost
    return BehaviorEpisode(np.zeros((25,8)),np.zeros((25,1)),np.zeros((25,1)),np.zeros(25),raw,
        'same-context',25,owner,'same-stochastic-controller',query_ref=ref)


def test_measured_ending_supplies_credit_to_actions_before_MPPIs_horizon():
    b=build_mc_batch([behavior(20.,'failed'),behavior(0.,'captured')],'behavior',minimum_replicates=2)
    np.testing.assert_array_equal(b['returns'][:25],np.full(25,20.))
    np.testing.assert_array_equal(b['advantages'][:25],np.full(25,20.))
    np.testing.assert_array_equal(b['advantages'][25:],np.full(25,-20.))
    assert b['returns'][0]>sum(behavior(20.,'x').raw_costs[:4])
    with pytest.raises(ValueError):build_mc_batch([behavior(20.,'a'),behavior(0.,'b',owner='updated-actor')],'behavior')


@dataclass
class TestOutcome:
    __test__=False
    context_id:str
    cost:float
    capture:bool
    complete:bool=True
    deterministic:bool=True
    remaining_steps:int=25


def rows(tag,cost,captures=16,n=16):
    return [TestOutcome(f'ordinary:{i}',cost,i<captures) for i in range(n)]


def test_student_teaching_source_selected_by_complete_outcome_not_teacher_quality():
    p=dict(parent=rows('p',2.),raw=rows('r',1.8),guided=rows('g',1.9))
    selected,criterion=select_student(p)
    assert selected=='raw' and criterion['guided_vs_parent']['passed']
    assert not criterion['guided_vs_raw']['passed']
    p['guided']=rows('g',1.7,captures=15)
    assert select_student(p)[0]=='raw' # Cheaper but capture-regressing child cannot win.


def test_every_actor_assessment_requires_complete_fresh_start_continuations():
    p=rows('a',2.);c=rows('b',1.8);c[0].remaining_steps=14
    with pytest.raises(ValueError):comparison(p,c)
    c=rows('b',1.8);c[0].context_id='wrong'
    with pytest.raises(ValueError):comparison(p,c)


def test_conservative_bound_charges_declined_handovers_and_both_learning_rounds():
    b=work_bound();raw=49*94+12*325
    assert b['teacher_pairs']==112 and b['model_controls']==112*(3*raw+25+50*25)==3000816
    assert b['actor_behavior_controls']==90624 and b['learning_curve_controls']==4800 and b['handoff_controls']==560
    assert b['executed_controls']==113184 and b['total_controls']==3114000 and b['actor_rows']==680656
    assert b['adam_steps']==512 and b['diagnostic_backward_passes']==128
    assert load_config()==DEFAULT and DEFAULT['learning']['aux_weight']==DEFAULT['learning']['bootstrap_updates']==0


def test_failed_teacher_performance_cannot_be_relabelled_success_by_actor_capture():
    p=rows('base',1.7);c=rows('guided',1.8)
    assert all(r.capture for r in c) and not comparison(p,c)['passed']
    assert comparison(rows('parent',2.),rows('child',1.9))['passed']


def test_goal_bridge_discloses_constructed_reset_and_uses_no_action_labels():
    states=np.column_stack((np.arange(26),np.linspace(.3,-.12,26)))
    e=StateExperience('discovered',states,{'raw':'measured-success'})
    for t,alpha in zip((11,8,4,0),(0.,.5,.75,1.)):
        snapshot,source=goal_reset_context(e,t,0,'constructed')
        assert snapshot.state[1]==pytest.approx(alpha*states[t,1]+(1-alpha)*states[-1,1]-.025)
        assert source['synthetic_reset_not_executed_teacher_state']
        assert not source['teacher_actions_used'] and source['alpha']==alpha
    assert goal_reset_context(e,0,1,'ordinary')[0].state[1]==pytest.approx(.325)


def test_goal_only_control_ignores_all_search_prefixes_and_endings():
    a=np.column_stack((np.arange(26),np.linspace(.3,-.12,26)))
    b=a.copy();b[1:,1]=1.
    ea=StateExperience('a',a,{'task_goal_only_control':True})
    eb=StateExperience('b',b,{'task_goal_only_control':True})
    for t in (11,8,4,0):
        sa,pa=goal_reset_context(ea,t,0,'same');sb,pb=goal_reset_context(eb,t,0,'same')
        np.testing.assert_array_equal(sa.state,sb.state)
        assert pa['observed_prefix'] is pa['observed_ending'] is None
        assert pb['ending_anchor_y']==-.1


def test_goal_revision_keeps_full_work_and_counts_only_relevant_handoff_support(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.gps.bidirectional import ending_pilot,policy,meter
    states=np.column_stack((np.arange(26),np.linspace(.3,-.12,26)))
    experience=[StateExperience(str(i),states,{}) for i in range(8)]
    visited=[];events=[]
    def episode(executor,snapshot,tag,**kw):
        visited.append(snapshot)
        return TestOutcome(snapshot.context_id,1.,True,remaining_steps=21),None,None,None
    def append(kind,**kw):events.append((kind,kw));return str(len(events))
    executor=SimpleNamespace(archive=SimpleNamespace(append=append),
        task=SimpleNamespace(observation=lambda s:np.r_[s.state,np.zeros(6)]))
    monkeypatch.setattr(ending_pilot,'_episode',episode)
    monkeypatch.setattr(policy,'policy_hash',lambda a:'unchanged-version')
    monkeypatch.setattr(meter,'attach_meter',lambda *a:None)
    anchors,eligible=suffix_eligibility(executor,object(),experience,'test',True,handoff_only=True)
    assert eligible and anchors.shape==(16,8) and len({s.context_id for s in visited})==16
    assert {s.elapsed for s in visited}=={4}
    assert visited[0].state[1]==pytest.approx(states[4,1]-.025)
    assert visited[1].state[1]==pytest.approx(states[4,1]+.025)
    assert events[0][1]['guidance_weight']==events[-1][1]['guidance_weight']==0.
    assert len(events[0][1]['source_experience'])==8
    np.testing.assert_array_equal(events[0][1]['source_experience'][0]['states'],states)
    bound=work_bound(GOAL_DEFAULT)
    assert bound['total_controls']==3114112 and bound['executed_controls']==113296
    assert bound['handoff_controls']==672 and bound['actor_rows']==680768
    assert bound['adam_steps']==512 and bound['diagnostic_backward_passes']==128
    path=tmp_path/'config.json';path.write_text(json.dumps(GOAL_DEFAULT))
    assert load_config(path)==GOAL_DEFAULT
    assert work_bound(SUPPORT_DEFAULT)==bound
    path.write_text(json.dumps(SUPPORT_DEFAULT));assert load_config(path)==SUPPORT_DEFAULT


def test_committed_prefix_then_same_actor_matches_full_outcome_but_different_continuation_does_not(tmp_path,monkeypatch):
    """75 actual cheap controls, globally charged by the test launcher."""
    from scripts.diagnostics.run_bidirectional_tests import DurableWork
    from src.gps.bidirectional.evidence import Archive
    from src.gps.bidirectional.ending_pilot import EndingContext,EndingExecutor
    from src.gps.bidirectional.return_curriculum import actor_completion
    from src.gps.bidirectional import policy
    class FixedActor:
        def act(self,obs,family,deterministic):return np.array([-.5])
    # Synthetic controller identifier; no neural network is constructed or run.
    monkeypatch.setattr(policy,'policy_hash',lambda actor:'synthetic-fixed-negative-controller')
    archive=Archive(tmp_path/'queries');archive.work_scope='committed-continuation-falsifier'
    work=DurableWork(tmp_path/'local_work.jsonl');dynamics=CountedEnding(archive,work,75)
    ledger=RawLedger(tmp_path/'raw_ending_ledger.jsonl');object.__setattr__(dynamics,'ledger',ledger)
    executor=EndingExecutor(EndingContext(dynamics),archive);initial=executor.task.reset(123)
    prefix=np.zeros((4,1));predicted=actor_completion(executor,FixedActor(),initial,prefix,'declared')
    actual=[]
    for tail_action in (-.5,.5):
        state=initial.state.copy();cost=0.
        for t in range(25):
            action=prefix[t] if t<4 else np.array([tail_action])
            state,_,c,flags=dynamics.live(state,action,dict(semantics='synthetic-test-controller'))
            cost+=float(c)
        actual.append((cost,bool(flags['ending_good'])))
    assert dynamics.reserved==75
    assert actual[0][0]==pytest.approx(predicted['cost'],abs=1e-12)
    assert actual[0][1]==predicted['capture'] is True
    assert actual[1][1] is False and actual[1][0]>actual[0][0]
    work.close();archive.close();ledger.stream.close()
