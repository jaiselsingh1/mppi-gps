"""Independent exact-fraction verification of a saved local certificate."""
import pathlib,json,hashlib,resource,time
from fractions import Fraction as F
resource.setrlimit(resource.RLIMIT_CPU,(59,59));start=time.process_time();ROOT=pathlib.Path.cwd();OUT=ROOT/'runs/2026-11-05_acrobot_external_reference_001/drake_certificate_audit_001';proto=json.load(open(OUT/'protocol_r1.json'));source=ROOT/proto['source'];prep=ROOT/'runs/2026-11-05_acrobot_external_reference_001/drake_preparation_001/preparation.json'
if hashlib.sha256(source.read_bytes()).hexdigest()!=proto['source_sha256']:raise RuntimeError('Certificate source changed after frozen review binding')
d=json.load(open(prep));c=d['certificate'];par=d['model_parameters'];checks=[]
def ck(name,ok):checks.append({'check':name,'passed':bool(ok)})
def tr(a):return list(map(list,zip(*a)))
def mul(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def inv(a):
 n=len(a);z=[list(row)+[F(i==j) for j in range(n)] for i,row in enumerate(a)]
 for j in range(n):
  if z[j][j]==0:
   k=next(k for k in range(j+1,n) if z[k][j]);z[j],z[k]=z[k],z[j]
  v=z[j][j];z[j]=[x/v for x in z[j]]
  for i in range(n):
   if i!=j:
    v=z[i][j];z[i]=[x-v*y for x,y in zip(z[i],z[j])]
 return [row[n:] for row in z]
def positive(a):
 z=[r[:] for r in a]
 for j in range(len(z)):
  if z[j][j]<=0:return False
  for i in range(j+1,len(z)):
   for k in range(j+1,len(z)):z[i][k]-=z[i][j]*z[j][k]/z[j][j]
 return True
p={k:F(v) for k,v in par.items()};a=p['m2']*p['l1']*p['c2'];lo=p['I2']+p['m2']*p['c2']**2;up=p['I1']+p['m1']*p['c1']**2;M=[[up+lo+p['m2']*p['l1']**2+2*a,lo+a],[lo+a,lo]];mi=inv(M);shoulder=p['m1']*p['c1']+p['m2']*p['l1'];end=p['m2']*p['c2'];g=p['gravity'];k=[F(x) for x in d['K_physical'][0]];P=[[F(x) for x in r] for r in c['P_rational']]
force=[[g*(shoulder+end),g*end,-p['damping1'],F(0)],[g*end-k[0],g*end-k[1],-k[2],-p['damping2']-k[3]]];aa=mul(mi,force);Acl=[[F(0),F(0),F(1),F(0)],[F(0),F(0),F(0),F(1)],*aa];L=mul(tr(Acl),P);Qc=[[-L[i][j]-L[j][i] for j in range(4)] for i in range(4)]
r=F(c['radius']['rational']);gamma=F(c['gamma']['rational']);rho=F(c['level']['rational']);gram=[[F(x) for x in row] for row in c['gram_rational']];ldlL=[[F(x) for x in row] for row in c['gram_LDL_lower']];ldlD=[F(x) for x in c['gram_LDL_diagonal']];D=[[ldlD[i] if i==j else F(0) for j in range(4)] for i in range(4)]
ck('P_symmetric',P==tr(P));ck('P_positive_exact',positive(P));ck('Qc_symmetric',Qc==tr(Qc));ck('Qc_positive_exact',positive(Qc));ck('gram_exact_formula',gram==[[Qc[i][j]-(gamma if i==j else 0) for j in range(4)] for i in range(4)]);ck('gram_LDL_identity_exact',mul(mul(ldlL,D),tr(ldlL))==gram);ck('gram_LDL_diagonal_positive',all(v>0 for v in ldlD));ck('gram_independent_positive',positive(gram))
masslo=(M[0][0]*M[1][1]-M[0][1]**2)/(M[0][0]+M[1][1]);ck('mass_lower_exact',masslo==F(c['mass_eigenvalue_lower']['rational']));den=masslo-2*a*r*r;ck('perturbed_mass_lower_positive',den>0)
B=(g*(shoulder+8*end)/6+2*a+2*a*sum(abs(v) for row in aa for v in row))/den;ck('remainder_B_exact',B==F(c['remainder_B']['rational']));pb=sum(abs(P[i][j]) for i in range(4) for j in [2,3]);ck('gamma_exact',gamma==2*pb*B*r*r)
Qtarget=[[F(10 if i<2 else 1)*F(i==j)+k[i]*k[j] for j in range(4)] for i in range(4)];res=max(sum(abs(Qtarget[i][j]-Qc[i][j]) for j in range(4)) for i in range(4));ql=1-res;ck('CARE_residual_exact',res==F(c['CARE_residual_infinity_bound']['rational']));ck('Qlower_exact',ql==F(c['Qcl_lower']['rational']));ck('remainder_margin',gamma<=ql/2 and ql>0)
pi=inv(P);pinorm=max(sum(abs(v) for v in row) for row in pi);ck('ellipsoid_containment_exact',rho==r*r/pinorm and rho>0);torq=sum(abs(v) for v in k)*r;ck('torque_bound_exact',torq==F(c['torque_bound']['rational']));ck('torque_margin',torq<=F(9,5) and torq<2)
# Unambiguous assumptions checked against source/output declarations.
ck('physical_input_gain',d['input_map']=='physical torque=2*normalized action');ck('CARE_one',d['counts']['CARE_solves']==1);ck('SDP_one',d['counts']['SDP_solves']==1);ck('no_native_controls',d['counts']['native_controls']==0 and d['counts']['physics_substeps']==0)
result={'status':'PASS_EXACT_CONTINUOUS_CERTIFICATE' if all(x['passed'] for x in checks) else 'FAIL','preparation_path':str(prep),'preparation_sha256':hashlib.sha256(prep.read_bytes()).hexdigest(),'source_sha256':proto['source_sha256'],'checks':checks,'radius':float(r),'level':float(rho),'torque_upper_bound_nm':float(torq),'mass_lower_on_ball':float(den),'gamma':float(gamma),'Qcl_lower':float(ql),'P_inverse_infinity_norm':float(pinorm),'SOS_saved':d['SOS'],'claim':'Sufficient invariant attracting ellipsoid for exact nominal continuous dynamics with continuously recomputed unsaturated local feedback. Quadratic SOS Gram is exact-rational positive; nonlinear trigonometric remainder is bounded analytically.','not_certified':['50Hz sample-and-hold','RK4 numerical map','global Spong swing-up','entry into this tiny ellipsoid','parameter/state-estimation uncertainty','a large useful recovery region'],'new_CARE_solves':0,'new_SDP_solves':0,'new_physics_controls':0,'independent_cpu_seconds':time.process_time()-start}
with (OUT/'result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
