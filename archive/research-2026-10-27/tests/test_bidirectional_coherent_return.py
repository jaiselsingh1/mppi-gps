"""Pure parameter geometry and synthetic-return tests; no live query/forward."""
from dataclasses import dataclass
import numpy as np
import pytest

from src.gps.bidirectional.coherent_return import (HeadSubspace,make_subspace,coordinate_bank,
    actor_from_coordinates,score_candidate,selector_view,select_candidate)
from src.gps.bidirectional.coherent_assay import DEFAULT,work_bound,choose_controls


def test_fixed_subspace_orthonormal_immutable_and_private_rng():
    old=np.random.get_state();a=make_subspace();b=make_subspace()
    assert np.array_equal(a.weight_directions,b.weight_directions)
    assert np.allclose(a.weight_directions@a.weight_directions.T,np.eye(4),atol=1e-12)
    assert np.array_equal(old[1],np.random.get_state()[1])
    with pytest.raises(ValueError):a.weight_directions.flags.writeable=True
    with pytest.raises(ValueError):make_subspace(7)
    with pytest.raises(ValueError):HeadSubspace(np.ones((4,128)))


def test_center_first_axial_bank_reachability_and_no_rescaling():
    first=coordinate_bank(np.zeros(6),0)
    assert first.shape==(13,6) and np.array_equal(first[0],np.zeros(6))
    assert np.array_equal(first[1:3,0],[-1.,1.])
    assert np.all(np.abs(first[1:]).sum(1)==1)
    second=coordinate_bank(first[1],1)
    assert np.array_equal(second[0],first[1]) and np.max(np.abs(second).sum(1))==2
    with pytest.raises(ValueError):coordinate_bank(first[1],0)
    with pytest.raises(ValueError):coordinate_bank(np.zeros(6),3)


def test_actor_construction_changes_only_head_and_preserves_private_rng_without_forward():
    import torch
    from src.gps.bidirectional.policy import FamilyGaussianActor,policy_hash
    base=FamilyGaussianActor(seed=71);basis=make_subspace();old=base.policy_state()
    zero=actor_from_coordinates(base,basis,np.zeros(6))
    assert policy_hash(zero)==policy_hash(base)
    c=np.array([1.,-1.,0.,0.,0.,-1.]);child=actor_from_coordinates(base,basis,c)
    for name,value in child.state_dict().items():
        if name not in ("fc_mean.bias","fc_mean.weight","log_std"):assert torch.equal(value,old["actor"][name])
    assert torch.equal(child._generator.get_state(),base._generator.get_state())
    assert np.allclose(child.fc_mean.bias.detach().numpy(),old["actor"]["fc_mean.bias"].numpy()+.1)
    assert np.allclose(child.log_std.detach().numpy(),old["actor"]["log_std"].numpy()-.5)
    assert all(not p.requires_grad for p in child.parameters())


@dataclass(frozen=True)
class Measured:
    context_id:str
    cost:float
    capture:bool
    complete:bool
    actor_hash:str
    controller_hash:str
    deterministic:bool
    remaining_steps:int
    query_ids:tuple
    semantics:str="actor_suffix"
    task_hash:str="task"
    family_id:int=0


def panels(owner="actor",cost=2.):
    result={}
    for role,remaining in (("ordinary",25),("raw",19),("long",19)):
        result[role]={}
        for mode,det in (("mean",True),("stochastic",False)):
            result[role][mode]=[Measured(f"{role}:{i}",cost,True,True,owner,owner+mode,det,remaining,
                tuple(f"{role}:{mode}:{i}:{t}" for t in range(remaining))) for i in range(8)]
    return result


def test_complete_owned_modes_are_measured_separately_and_no_query_reuse():
    p=panels();s=score_candidate(p,"actor")
    assert s["roles"]["ordinary"]["joint_normalized_cost"]==2/25
    p["long"]["mean"][0]=Measured("long:0",2.,True,True,"other","othermean",True,19,tuple(f"x{i}" for i in range(19)))
    with pytest.raises(ValueError):score_candidate(p,"actor")
    p=panels();p["raw"]["mean"][0]=Measured("raw:0",2.,True,False,"actor","actormean",True,19,tuple(f"x{i}" for i in range(19)))
    with pytest.raises(ValueError):score_candidate(p,"actor")


def test_data_source_cannot_buy_root_regression_and_hidden_panels_rejected():
    center=score_candidate(panels(),"actor")
    import copy
    scores=[copy.deepcopy(center) for _ in range(13)]
    # Tempting rescue gain but ordinary mean regression: ineligible.
    scores[1]["roles"]["ordinary"]["mean"]["mean_cost"]=2.01
    scores[1]["roles"]["long"]["joint_normalized_cost"]=0.
    # Smaller rescue gain with no ordinary loss: eligible and chosen.
    scores[2]["roles"]["long"]["joint_normalized_cost"]*=.5
    with pytest.raises(ValueError):select_candidate(scores,"long",1.)
    decision=select_candidate([selector_view(s,"long") for s in scores],"long",1.)
    assert not decision["feasible_flags"][1] and decision["selected_index"]==2
    ordinary=select_candidate([selector_view(s,"ordinary") for s in scores],"ordinary",0.)
    assert ordinary["selected_index"]==0
    assert set(selector_view(scores[0],"raw")["roles"])=={"ordinary","raw"}


def test_exact_all_paid_budget_includes_parent_controls_and_fresh_confirmation():
    b=work_bound()
    assert b["training_controls"]==196560 and b["qualification_controls"]==4800 and b["final_controls"]==9600
    assert b["total_controls"]==b["actor_rows"]==DEFAULT["max_controls"]==210960
    assert b["model_controls"]==b["adam_steps"]==0 and b["training_episodes"]==9360


def qualification_panel(name,cost,det):
    return [Measured(f"ordinary:{i}",cost,True,True,name,name+str(det),det,25,(f"q{name}{det}{i}",)) for i in range(16)]


def test_strong_control_selected_independently_can_reject_long_data_winner():
    costs=dict(parent=2.,ordinary=1.7,raw025=1.8,raw1=1.75,long025=1.72,long1=1.73)
    m={n:qualification_panel(n,c,True) for n,c in costs.items()};s={n:qualification_panel(n,c,False) for n,c in costs.items()}
    chosen=choose_controls(m,s)
    assert chosen["best_control"]=="ordinary" and chosen["selected_long"] is None
    m["long025"]=qualification_panel("long025",1.6,True);s["long025"]=qualification_panel("long025",1.6,False)
    assert choose_controls(m,s)["selected_long"]=="long025"
