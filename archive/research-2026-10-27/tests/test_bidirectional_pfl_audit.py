"""Independent algebra fixtures; no simulator, policy or CARE solve."""
import numpy as np
import pytest
from src.gps.bidirectional import pfl_audit as a


def parameters():
    return dict(m1=1.,m2=1.,c1=.5,c2=.5,l1=1.,I1=.0955703125,I2=.09530365719649561,
        gravity=9.81,damping1=.05,damping2=.05,gear=2.,pivot_height=2.,armature1=0.,armature2=0.)


def test_finite_caps_keep_integration_and_zero_time_separate():
    c,b=a.fixed_config(),a.fixed_bound()
    assert c['max_controls']==b['nominal_controls']+b['selection_controls']+b['final_controls']==57600
    assert b['max_direct_zero_time_forwards']==24+144 and b['parity_mj_fullM_calls']==24
    assert len(c['gains'])==4 and len(c['nominal_seeds'])==2 and len(c['selection_seeds'])==8
    assert len(c['final_seeds'])==32 and c['final_min_admissible']==29


@pytest.mark.parametrize('x',[np.array([float('nan')]),np.array([float('inf')]),np.array(['x'])])
def test_finite_rejects_unverifiable_fields(x):
    with pytest.raises(ValueError):a.finite(x)


def test_native_mass_packing_has_diagonals_before_ancestor():
    np.testing.assert_array_equal(a.unpack_native_mass([3.,2.,.5]),[[3.,.5],[.5,2.]])
    with pytest.raises(ValueError):a.unpack_native_mass([1.,2.])


def test_gravity_is_potential_gradient_and_energy_origin_is_disclosed():
    p=parameters();q=np.array([.4,-.7]);w=np.array([3.,-2.]);m=a.mechanics(p,q,w);e=1e-6
    d=np.array([(a.mechanics(p,q+e*np.eye(2)[i],[0.,0.])['energy']-
        a.mechanics(p,q-e*np.eye(2)[i],[0.,0.])['energy'])/(2*e) for i in range(2)])
    np.testing.assert_allclose(d,m['G'],atol=1e-8,rtol=0)
    assert a.mechanics(p,[0.,0.],[0.,0.])['energy']-a.mechanics(p,[np.pi,0.],[0.,0.])['energy']==pytest.approx(39.24)


@pytest.mark.parametrize('u',[-1.,0.,1.])
def test_independent_schur_elimination_and_clipping_residual(u):
    p=parameters();t=a.pfl(p,[.4,-.7],[3.,-2.],[4.,2.,1.],58.86);f=a.native_fields(p,t,u)
    np.testing.assert_allclose(t['M']@f['acceleration']+t['h'],[0.,2*u],atol=1e-12)
    assert f['acceleration'][1]==pytest.approx(f['elbow_elimination'],abs=1e-12)
    assert f['acceleration'][1]-t['virtual_acceleration']==pytest.approx(f['clip_residual'],abs=1e-12)


def test_parity_detects_wrong_bias_without_forgiving_other_fields():
    p=parameters();t=a.pfl(p,[.4,-.7],[3.,-2.],[1.,1.,.1],58.86);f=a.native_fields(p,t,t['action'])
    n=dict(bias=t['Cw']+t['G'],passive=-t['damping'],actuator=f['torque'],qacc=f['acceleration'],energy=np.array([t['energy'],0.]))
    checks,_,_=a.parity_comparison(p,t,t['action'],n,t['M']);assert all(checks.values())
    n['bias']=n['bias']+1e-4;checks,_,_=a.parity_comparison(p,t,t['action'],n,t['M'])
    assert checks['qacc'] and not checks['bias']


def test_saved_care_is_checked_without_solving():
    residual,eig,relative=a.verify_care(np.array([[-1.]]),np.array([[0.]]),np.array([[1.]]),np.array([[2.]]),1.,np.array([[0.]]))
    assert relative==0 and eig[0]==-1 and residual[0,0]==0
    with pytest.raises(ValueError):a.verify_care(np.array([[-1.]]),np.array([[0.]]),np.array([[1.]]),np.array([[3.]]),1.,np.array([[0.]]))


def test_controller_wraps_only_local_error():
    p=parameters();K=np.array([[1.,2.,3.,4.]]);q=np.array([2*np.pi+.1,-2*np.pi+.05]);w=np.array([.2,-.1])
    u,m=a.controller_action(p,[1.,1.,.1],K,np.r_[0.,q,w],58.86)
    assert u[0]==pytest.approx(-.4) and m['mode']=='upright_lqr'
    _,m=a.controller_action(p,[1.,1.,.1],K,[0.,.4,2*np.pi+.2,.1,.2],58.86)
    assert m['mode']=='collocated_pfl' and m['q2_unwrapped']>2*np.pi


def test_pending_and_duplicate_inspection_work_remain_visible():
    req=dict(id='a',kind='pfl_native_forward_requested');done=dict(id='b',kind='pfl_native_forward_complete',request_id='a')
    assert a.close_intents([req])[2]==['a']
    assert a.close_intents([req,done])[2]==[]
    with pytest.raises(ValueError):a.close_intents([req,done,dict(done,id='c')])
    with pytest.raises(ValueError):a.close_intents([done,req])
