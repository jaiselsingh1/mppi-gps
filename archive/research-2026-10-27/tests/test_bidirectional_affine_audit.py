"""Independent affine audit arithmetic fixtures; zero simulator/actor calls."""
import copy
import hashlib
import json

import numpy as np
import pytest

from src.gps.bidirectional.affine_audit import (
    affine_bellman, audit_affine_reference, derivative_arrays, feedback_action,
    fixed_bound, fixed_config, saved_episode_rows, stencil, validate_riccati,
)


def channel(n=1):
    return np.repeat(np.eye(4)[None],n,axis=0),np.repeat(np.array([0.,0.,1.,0.])[None,:,None],n,axis=0)


def test_fixed_budget_counts_fresh_unperturbed_rows_and_actual_nominals_separately():
    b=fixed_bound()
    assert b==dict(native_qualification=48,exact_point_derivative_controls=35200,
        actual_nominal_controls=3200,selection_controls=25600,fresh_final_controls=25600,total=89648)
    assert sum(v for k,v in b.items() if k!='total')==b['total']==fixed_config()['max_controls']
    assert fixed_config()['final_min_admissible']==29


def test_stencil_is_node_major_with_exact_saved_nominal_and_clipped_action_pairs():
    s=np.array([[0.,.2,-.3,.4,-.5],[7.99,.1,.2,-.2,.3]])
    u=np.array([[-1.],[1.]])
    ss,uu,width=stencil(s,u)
    assert ss.shape==(22,5) and uu.shape==(22,1)
    blocks,controls=ss.reshape(2,11,5),uu.reshape(2,11,1)
    np.testing.assert_array_equal(blocks[:,10],s)
    np.testing.assert_array_equal(controls[:,10],u)
    np.testing.assert_array_equal(blocks[:,:,0],np.repeat(s[:,0,None],11,axis=1))
    np.testing.assert_allclose(width,[1e-5,1e-5],atol=1e-16,rtol=0)
    for i in range(2):
        for j in range(4):
            plus,minus=s[i].copy(),s[i].copy();plus[j+1]+=1e-5;minus[j+1]-=1e-5
            np.testing.assert_array_equal(blocks[i,2*j],plus)
            np.testing.assert_array_equal(blocks[i,2*j+1],minus)
    assert np.max(abs(uu))<=1


def test_derivatives_use_fresh_eleventh_row_and_actual_clipped_width():
    s=np.array([[0.,.2,-.3,.4,-.5],[7.99,.1,.2,-.2,.3]]);u=np.array([[-1.],[1.]])
    ss,uu,w=stencil(s,u)
    matrix=np.array([[1.,.2,0.,.1],[-.1,.9,.2,0.],[0.,.1,1.1,-.2],[.3,0.,-.1,.8]])
    b=np.array([.1,-.2,.3,-.4]);offset=np.array([.01,-.02,.03,-.04])
    output=np.column_stack((ss[:,0]+.01,ss[:,1:]@matrix.T+uu*b+offset))
    A,B,F=derivative_arrays(output,w)
    np.testing.assert_allclose(A,np.repeat(matrix[None],2,axis=0),atol=2e-10,rtol=0)
    np.testing.assert_allclose(B,np.repeat(b[None,:,None],2,axis=0),atol=2e-10,rtol=0)
    np.testing.assert_array_equal(F,output.reshape(2,11,5)[:,10])
    with pytest.raises(ValueError):derivative_arrays(output,np.zeros(2))


@pytest.mark.parametrize('d',[.2,-.2])
def test_independent_one_step_schur_solution_has_correct_defect_feedforward_sign(d):
    A,B=channel();defect=np.array([[0.,0.,d,0.]])
    raw=affine_bellman(A,B,defect)
    np.testing.assert_allclose(raw['K'][0,0],[0.,0.,10/11,0.],atol=1e-14,rtol=0)
    np.testing.assert_allclose(raw['k'][0],[d/1.1],atol=1e-14,rtol=0)
    np.testing.assert_allclose(raw['p'][0],[0.,0.,-d/11,0.],atol=1e-14,rtol=0)
    assert abs(raw['k'][0,0]-d)<abs(d)
    _,error=validate_riccati(raw,A,B,defect,[10.,10.,1.,1.],.1)
    assert error<1e-15


def test_two_step_independent_value_gradient_and_hessian_match_rational_solution():
    A,B=channel(2);d=np.zeros((2,4));d[:,2]=[.1,.2]
    r=affine_bellman(A,B,d)
    np.testing.assert_allclose(r['K'][:,0,2],[120/131,10/11],atol=2e-14,rtol=0)
    np.testing.assert_allclose(r['k'][:,0],[14/131,2/11],atol=2e-14,rtol=0)
    np.testing.assert_allclose(r['p'][:,2],[-1.4/131,-.2/11,0.],atol=2e-14,rtol=0)
    np.testing.assert_allclose(r['P'][:,2,2],[143/131,12/11,1.],atol=2e-14,rtol=0)


def dense_control_solution(error,A,B,d,Q,R):
    """Independent dense quadratic expansion, not a project transition."""
    n=len(A);weight=np.diag(Q);mapping=np.zeros((4,n));offset=np.asarray(error,float)
    h=R*np.eye(n);linear=np.zeros(n)
    for t in range(n):
        h+=mapping.T@weight@mapping;linear+=mapping.T@weight@offset
        offset=A[t]@offset-d[t];mapping=A[t]@mapping;mapping[:,t]+=B[t,:,0]
    h+=mapping.T@weight@mapping;linear+=mapping.T@weight@offset
    return -np.linalg.solve(h,linear)


@pytest.mark.parametrize('error',[[0.,0.,0.,0.],[.2,-.1,.3,-.2],[-.3,.2,-.1,.4]])
def test_affine_feedback_is_stationary_for_independent_dense_full_quadratic(error):
    matrix=np.array([[1.,.2,0.,.1],[-.1,.9,.2,0.],[0.,.1,1.1,-.2],[.3,0.,-.1,.8]])
    A=np.repeat(matrix[None],3,axis=0);B=np.repeat(np.array([.1,-.2,.3,-.4])[None,:,None],3,axis=0)
    d=np.array([[.01,-.02,.03,-.04],[-.02,.03,-.04,.01],[.03,-.04,.01,-.02]])
    r=affine_bellman(A,B,d);e=np.asarray(error,float);controls=[]
    for t in range(3):
        u=float(r['k'][t,0]-r['K'][t,0]@e);controls.append(u)
        e=A[t]@e+B[t,:,0]*u-d[t]
    np.testing.assert_allclose(controls,dense_control_solution(error,A,B,d,[10.,10.,1.,1.],.1),atol=2e-14,rtol=0)


@pytest.mark.parametrize('key',['P','p','K','k','H'])
def test_stored_affine_math_mutations_fail_independent_reconstruction(key):
    A,B=channel(2);d=np.zeros((2,4));d[:,2]=[.1,.2];r=affine_bellman(A,B,d)
    broken=copy.deepcopy(r);broken[key].flat[0]+=.01
    with pytest.raises(ValueError,match='Bellman'):
        validate_riccati(broken,A,B,d,[10.,10.,1.,1.],.1)


def test_zero_defects_have_zero_affine_linear_term_and_feedforward():
    A,B=channel(2);r=affine_bellman(A,B,np.zeros((2,4)))
    np.testing.assert_array_equal(r['k'],np.zeros((2,1)))
    np.testing.assert_array_equal(r['p'],np.zeros((3,4)))


def test_actual_control_adds_feedforward_before_bounded_native_error_feedback():
    reference=np.zeros((801,5));reference[:,0]=np.arange(801)*.01
    actions=np.zeros((800,1));actions[5]=.2
    K=np.zeros((800,1,4));K[5,0]=[1.,.5,.25,-.125]
    k=np.zeros((800,1));k[5]=.3
    state=reference[5].copy();state[1:]+=[2*np.pi+.1,-2*np.pi+.2,.4,-.8]
    np.testing.assert_allclose(feedback_action(state,5,reference,actions,K,k),[.1],atol=2e-14,rtol=0)
    k[5]=2.;np.testing.assert_array_equal(feedback_action(reference[5],5,reference,actions,K,k),[1.])


@pytest.mark.parametrize('angle',[np.pi,-np.pi,3*np.pi,-3*np.pi])
def test_actual_control_wraps_half_open_angle_ties(angle):
    reference=np.zeros((801,5));reference[:,0]=np.arange(801)*.01
    K=np.zeros((800,1,4));K[0,0,0]=.1
    state=reference[0].copy();state[1]=angle
    np.testing.assert_allclose(feedback_action(state,0,reference,np.zeros((800,1)),K,np.zeros((800,1))),[.1*np.pi],atol=1e-14,rtol=0)


def test_absent_run_cannot_receive_affine_pass(tmp_path):
    r=audit_affine_reference(tmp_path/'absent')
    assert r['status']=='PENDING' and not r['native_reference_pass']
    assert r['additional_physics_controls']==r['additional_actor_inference_rows']==0


def test_actual_archive_episode_saved_event_name_and_duplicate_guard():
    saved=dict(kind='episode_saved',episode_id='affine0_nominal',raw={'file':'actual.npz'})
    assert saved_episode_rows([dict(kind='episode',episode_id='untrusted'),saved])=={'affine0_nominal':saved}
    with pytest.raises(ValueError,match='Duplicate'):
        saved_episode_rows([saved,dict(saved)])


def test_composed_nonempty_component_hashes_preserve_archive_paths(tmp_path,monkeypatch):
    """Real composition must still read both ledgers after merging hash maps."""
    from src.gps.bidirectional import audit as native_audit
    from src.gps.bidirectional import hetero_audit

    consumed={};calls=[]
    for archive in ('native','heterogeneous'):
        path=tmp_path/'evidence'/archive/'ledger.jsonl';path.parent.mkdir(parents=True)
        rows=[dict(id='event-000000000',kind='archive_begin',schema='bidirectional-evidence-v1',utc='2026-10-27T00:00:00Z',wall_seconds=0.),
              dict(id='event-000000001',kind='archive_close',utc='2026-10-27T00:00:00Z',wall_seconds=.1,accounting={})]
        blob=''.join(json.dumps(row)+'\n' for row in rows).encode();path.write_bytes(blob)
        consumed[str(path.resolve())]=hashlib.sha256(blob).hexdigest()

    def component(path,**kwargs):
        assert path==tmp_path/'evidence'/path.name
        assert kwargs['source_root']==tmp_path/'source'
        calls.append(path.name)
        name=str((path/'ledger.jsonl').resolve())
        return dict(status='PASS',arithmetic_valid=True,errors=[],pending=[],input_sha256={name:consumed[name]})

    monkeypatch.setattr(native_audit,'audit_archive',component)
    monkeypatch.setattr(hetero_audit,'audit_heterogeneous',component)
    report=audit_affine_reference(tmp_path)
    assert calls==['native','heterogeneous']
    assert report['status']=='PENDING' and report['errors']==[]
    assert report['counts']['queries']==report['counts']['requested_controls']==0
    assert set(report['component_audits'])=={'native','heterogeneous'}
    assert report['input_sha256']==consumed
    assert any('Missing closed artifact: affine_result.json'==row for row in report['pending'])
    assert report['additional_physics_controls']==report['additional_actor_inference_rows']==report['additional_fits']==0


def test_corrupt_declared_affine_source_fails_even_without_execution(tmp_path):
    source=tmp_path/'source'/'bad.py';source.parent.mkdir();source.write_text('changed\n')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files=[dict(file='source/bad.py',sha256='0'*64,bytes=8)])))
    r=audit_affine_reference(tmp_path)
    assert r['status']=='FAIL' and any('source hash' in e for e in r['errors'])
    assert not r['native_reference_pass']
