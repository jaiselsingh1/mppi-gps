"""Independent numeric witnesses; no project code imported."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# A rare low-cost region can be absent from every finite-batch sample.
p = 1e-6
q = p*np.exp(20)/(1-p+p*np.exp(20))
kl = q*np.log(q/p)+(1-q)*np.log((1-q)/(1-p))
rare = dict(prior_mass=p, target_mass=q, population_kl=kl,
            probability_no_rare_sample_in_K256=(1-p)**256,
            empirical_kl_when_missed=0, empirical_ess_when_missed=256)

# Exact Gaussian base; feasible active cost budget, symmetric tilted law.
# S(u)=(u^2-1)^2. Fixed-covariance mean projection always returns mean zero.
def moment(nu, fn):
    def density(u): return np.exp(-u*u/2-nu*(u*u-1)**2)/np.sqrt(2*np.pi)
    z=quad(density,-np.inf,np.inf,epsabs=1e-12)[0]
    return quad(lambda u:fn(u)*density(u),-np.inf,np.inf,epsabs=1e-12)[0]/z
budget=.5
nu=brentq(lambda nu:moment(nu,lambda u:(u*u-1)**2)-budget,.001,10)
trust=dict(budget=budget, nu=nu, prior_expected_cost=2.,
           tilted_expected_cost=moment(nu,lambda u:(u*u-1)**2),
           tilted_mean=moment(nu,lambda u:u), projected_mean=0.,
           cost_of_mean_plan=1., projected_prior_expected_cost=2.)
assert abs(trust['tilted_mean'])<1e-12
assert abs(trust['tilted_expected_cost']-budget)<1e-10
assert trust['projected_prior_expected_cost']>budget

# Exact parametric least squares is not pointwise interpolation.
y=np.array([-1.,1.]); theta=float(y.mean())
projection=dict(targets=y.tolist(), exact_best_constant=theta,
                residuals=(y-theta).tolist(), loss_gradient=float(2*np.mean(theta-y)))

# ESS constraint is only sufficient (Renyi-2 upper bounds Shannon KL).
w=np.array([.7,.1,.1,.1]); shannon=float(np.sum(w*np.log(4*w)))
ess=float(1/np.sum(w*w)); cap=.5
not_equivalent=dict(weights=w.tolist(), kl=shannon, epsilon=cap, ess=ess,
                    kl_cap_met=shannon<=cap,
                    ess_sufficient_condition_met=bool(ess/4>=np.exp(-cap)))
assert not_equivalent['kl_cap_met'] and not not_equivalent['ess_sufficient_condition_met']

# Clipping destroys the nominal-mean equality even at infinite temperature.
mean=.9; sigma=.3
from scipy.stats import norm
# The same fixed-point failure also holds with the implementation's clipping:
# the pushed-forward Gaussian has atoms at +/-1, where the double-well cost is 0.
boundary_mass = 2*norm.sf(1)
def clipped_cost(nu):
    density = lambda u: np.exp(-nu*(u*u-1)**2)*norm.pdf(u)
    z = boundary_mass + quad(density, -1, 1, epsabs=1e-12)[0]
    return quad(lambda u: (u*u-1)**2*density(u), -1, 1, epsabs=1e-12)[0]/z
clipped_budget = .1
clipped_nu = brentq(lambda nu: clipped_cost(nu)-clipped_budget, 0, 20)
clipped_trust = dict(bounds=[-1, 1], budget=clipped_budget, nu=clipped_nu,
                     prior_expected_cost=clipped_cost(0), tilted_expected_cost=clipped_cost(clipped_nu),
                     tilted_mean_by_symmetry=0., projected_mean=0., cost_of_mean_plan=1.)
assert clipped_trust['prior_expected_cost'] > clipped_budget
a=(-1-mean)/sigma; b=(1-mean)/sigma
clipped_mean=(-1)*norm.cdf(a)+mean*(norm.cdf(b)-norm.cdf(a))+sigma*(norm.pdf(a)-norm.pdf(b))+(1-norm.cdf(b))
clip=dict(nominal=mean,sigma=sigma,infinite_temperature_expected_label=clipped_mean,
          mean_shift=clipped_mean-mean)
result=dict(rare_region=rare,trust_fixed_point=trust,clipped_trust_fixed_point=clipped_trust,exact_projection=projection,
            ess_cap_not_equivalence=not_equivalent,clipped_full_trust=clip)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
