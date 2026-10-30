"""Runner seams that pure individual-component tests cannot establish."""
import json
import numpy as np
import torch

from src.gps.bidirectional.evidence import Archive,QueryExecutor
from src.gps.bidirectional.policy import FamilyGaussianActor,policy_hash
from src.gps.bidirectional.learning import PolicyLearner,LearningConfig
from src.gps.bidirectional.search import IndependentSearch,SearchConfig,GuidedSelector
from src.gps.bidirectional.trust import TrustManager
from src.gps.bidirectional.workflow import execute_episode
from src.gps.bidirectional.meter import attach_meter
from test_bidirectional_task_evidence import task,upright


def test_zero_trust_preserves_real_search_result_without_actor_forward(tmp_path):
    synthetic=task(T=4);initial=upright(synthetic)
    with Archive(tmp_path/"evidence") as archive:
        executor=QueryExecutor(synthetic,archive)
        planner=IndependentSearch(executor,SearchConfig(K=4,H=2,smooth_candidates=1,memory_candidates=1),seed=7)
        bank=planner.propose(initial)
        actor=FamilyGaussianActor(hidden_dim=16)
        with torch.no_grad():actor.fc_mean.bias.fill_(100.)
        selected=GuidedSelector(executor,actor,TrustManager()).select(bank)
        assert isinstance(bank.independent_score,float)
        assert not selected.influenced and not selected.changed
        np.testing.assert_array_equal(selected.plan,bank.independent_plan)
        assert "neural_requested" not in archive.ledger_path.read_text()


def test_complete_stochastic_execution_can_feed_owned_update_and_meter(tmp_path):
    synthetic=task(T=4);initial=upright(synthetic)
    with Archive(tmp_path/"evidence") as archive:
        executor=QueryExecutor(synthetic,archive)
        actor=FamilyGaussianActor(hidden_dim=16,seed=3)
        learner=PolicyLearner(actor,LearningConfig(epochs=1,batch_size=8,return_scale=4),seed=9)
        owner=policy_hash(actor);episodes=[]
        for i in range(2):
            trace,behavior,row=execute_episode(executor,initial,f"replicate{i}",actor=actor,
                deterministic=False,generator=learner.torch_generator)
            assert trace.complete and behavior.behavior_policy_hash==owner
            episodes.append(behavior)
        attach_meter(actor,archive,"fit")
        learning=learner.update(episodes,owner)
        assert learning["controls"]==8 and not learning["promotion_claim"]
        rows=[json.loads(s) for s in archive.ledger_path.read_text().splitlines()]
        requests=[r for r in rows if r["kind"]=="neural_requested"]
        completions=[r for r in rows if r["kind"]=="neural_complete"]
        assert len(requests)==len(completions)
        assert sum(r["rows"] for r in requests)==8+8+8
        assert len([r for r in rows if r["kind"]=="behavior_saved"])==2
        assert archive.totals()["completed_controls"]==8


def test_synthesized_mean_enforces_native_bounds_before_verification(tmp_path,monkeypatch):
    import src.gps.bidirectional.search as search
    synthetic=task(T=4);initial=upright(synthetic)
    with Archive(tmp_path/"evidence") as archive:
        executor=QueryExecutor(synthetic,archive)
        planner=IndependentSearch(executor,SearchConfig(K=4,H=2,smooth_candidates=0,memory_candidates=0),seed=7)
        monkeypatch.setattr(search.np,"einsum",lambda *args,**kwargs:np.full((2,1),np.nextafter(1.,2.)))
        planner.propose(initial)
        rows=[json.loads(s) for s in archive.ledger_path.read_text().splitlines()]
        query=next(r for r in rows if r["kind"]=="query_requested" and r["query_kind"]=="independent_mean_verification")
        with np.load(archive.path/query["inputs"]["file"]) as raw:
            np.testing.assert_array_equal(raw["actions"],np.ones((1,2,1)))


def test_unsupported_current_state_blocks_future_supported_agreement(tmp_path):
    synthetic=task(T=4);initial=upright(synthetic)
    class FutureOnlyTrust:
        def weight_for(self,*args,**kwargs):return 0.
        def weights_for_batch(self,*args,**kwargs):raise AssertionError("Current unsupported state must stop selection")
    with Archive(tmp_path/"evidence") as archive:
        executor=QueryExecutor(synthetic,archive)
        planner=IndependentSearch(executor,SearchConfig(K=4,H=2,smooth_candidates=0,memory_candidates=0),seed=7)
        bank=planner.propose(initial)
        actor=FamilyGaussianActor(hidden_dim=16)
        selected=GuidedSelector(executor,actor,FutureOnlyTrust()).select(bank)
        assert selected.reason=="unsupported_current_state" and not selected.influenced
        assert "neural_requested" not in archive.ledger_path.read_text()
