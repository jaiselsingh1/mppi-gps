"""Trust terms from docs/theory.md: the budget temperature, the oracle's E-step
label, and the predictive check in training."""
from dataclasses import replace

import numpy as np

from scripts.diagnostics.lqr_trust_check import estep_label
from scripts.training.train_walker_mdgps import TrainConfig, _batch_act, _collect, _new_actor, _noise_sigma
from src.envs.lqr_point_mass import LQRPointMass
from src.gps.mdgps import closed_loop_nominal, e_step
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.math import softmin_weights, temperature_for_budget


def test_budget_temperature_meets_the_budget():
    scores = np.random.default_rng(0).gamma(2.0, 3.0, 256)
    budget = float(np.min(scores) + 0.3 * (np.mean(scores) - np.min(scores)))
    lam = temperature_for_budget(scores, budget)
    np.testing.assert_allclose(np.sum(softmin_weights(scores, lam) * scores), budget, rtol=1e-6)
    assert temperature_for_budget(scores, float(np.mean(scores))) == float("inf")


def _lqr_case():
    env = LQRPointMass(nthread=1)
    cfg = MPPIConfig(K=64, H=8, noise_sigma=0.3, lam_target_neff_fraction=0.2)
    state = env.full_state(np.array([0.05, -0.03, 0.02, 0.04]))
    act = lambda s: -0.5 * env.K_lqr @ env.lqr_state(s)
    nominal, _ = closed_loop_nominal(env, state, act, 8)
    return env, cfg, state, nominal


def test_oracle_label_reproduces_e_step():
    env, cfg, state, nominal = _lqr_case()
    reference = e_step(MPPI(env, cfg, seed=3), env, state, nominal).label
    label, _ = estep_label(env, state, nominal, MPPI(env, cfg, seed=3)._sample_noise(), 0.2)
    np.testing.assert_allclose(label, reference, rtol=1e-12, atol=1e-12)


def test_full_trust_is_the_working_step_and_the_check_keeps_own_action():
    env, cfg, state, nominal = _lqr_case()
    noise = MPPI(env, cfg, seed=3)._sample_noise()
    fixed, _ = estep_label(env, state, nominal, noise, 0.2)
    # The policy as its own planner meets the budget exactly: full trust.
    trusted, _ = estep_label(env, state, nominal, noise, 0.2, planner_plan=nominal, rho_cap=0.05)
    np.testing.assert_array_equal(trusted, fixed)
    checked, info = estep_label(env, state, nominal, noise, 0.2, check=True)
    np.testing.assert_array_equal(checked, nominal[0] if info["rejected"] else fixed)


def test_predictive_check_replaces_only_labels_whose_plan_is_worse():
    cfg = TrainConfig(K=16, H=4, steps=40, label_every=2, workers=1)
    actor = _new_actor(0)
    # A nearly uniform step (n_eff/K = 0.9) averages noise, so some plans get worse.
    job = {"actor": {k: v.cpu() for k, v in actor.state_dict().items()}, "reset_seed": 8,
           "planner_seed": 11, "noise_seed": 13, "execute_expert": False,
           "probe_probability": 0.5, "neff_fraction": 0.9}
    plain = _collect({**job, "cfg": cfg})
    checked = _collect({**job, "cfg": replace(cfg, predictive_check=True)})
    worse = plain["diagnostics"][:, 3] > plain["diagnostics"][:, 2]
    assert worse.any() and not worse.all()
    np.testing.assert_array_equal(checked["labels"][~worse], plain["labels"][~worse])
    np.testing.assert_allclose(checked["labels"][worse], _batch_act(actor)(plain["obs"])[worse], atol=1e-6)


def test_sigma_schedule_sets_the_e_step_noise():
    cfg = TrainConfig(K=16, H=4, steps=20, label_every=2, workers=1, sigma_decay_iterations=25.0)
    assert _noise_sigma(cfg, 0.3, 0) == 0.3
    np.testing.assert_allclose(_noise_sigma(cfg, 0.3, 25), 0.3 / np.sqrt(2.0))
    assert _noise_sigma(replace(cfg, sigma_decay_iterations=None), 0.3, 50) == 0.3
    actor = _new_actor(0)
    job = {"actor": {k: v.cpu() for k, v in actor.state_dict().items()}, "reset_seed": 7,
           "planner_seed": 11, "noise_seed": 13, "execute_expert": False,
           "probe_probability": 0.5, "neff_fraction": 0.2, "noise_sigma": 1e-7}
    # With almost no sampling noise, the improved plan stays on the policy's own.
    out = _collect({**job, "cfg": cfg})
    np.testing.assert_allclose(out["labels"], _batch_act(actor)(out["obs"]), atol=1e-5)
