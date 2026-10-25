"""Small CPU reference for decoupled proposals and verified plan selection."""
from dataclasses import replace

import numpy as np
import torch

from src.gps.decoupled import make_proposals, synthesize_verified
from src.gps.mdgps import closed_loop_nominal
from src.gps.policy_tail import WalkerPolicyTail
from src.mppi.mppi import MPPI


class DecoupledCPUEStep:
    def __init__(self, env, cfg, tail_steps=100, states_per_batch=1):
        self.env, self.cfg, self.tail_steps = env, cfg, tail_steps

    @torch.no_grad()
    def __call__(self, states, warmstarts, actor_cuda, noise_seeds, *, proposal_mode="seeded",
                 synthesis_mode="mean_verified", policy_sigma=.3, credit_value=None, credit_gamma=.995):
        from src.gps.decoupled_gpu import discounted_policy_value_score
        env, cfg = self.env, self.cfg
        actor = actor_cuda.cpu().eval()
        def act(obs):
            mean, _ = actor(torch.as_tensor(np.array(obs, copy=True), dtype=torch.float32))
            return (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)
        tail = WalkerPolicyTail(env, act, self.tail_steps)
        low, high = [torch.as_tensor(x, dtype=torch.float64) for x in env.action_bounds]
        outputs, diagnostics = [], []
        ledger = {key: 0 for key in ("candidate_controls", "nominal_generation_controls", "mean_verification_controls")}
        for state, warm, seed in zip(states, warmstarts, noise_seeds):
            nominal, nominal_states = closed_loop_nominal(env, state,
                lambda x: act(env.rollout_states_to_obs(x)), cfg.H, warm)
            # Replaying the nominal supplies a reference cost and identical solver semantics.
            # Its extra H controls are recorded, rather than hidden in the candidate count.
            def score(plans):
                paths, costs, _ = env.batch_rollout(state, plans, initial_warmstart=warm)
                if credit_value is None:
                    scores = costs + (tail(paths, plans) if self.tail_steps else 0.)
                    endpoints = env.rollout_states_to_obs(paths[:, -1])
                    healthy = env._is_healthy(paths[..., 2], paths[..., 3]).all(axis=1)
                else:
                    scores, endpoints, healthy = discounted_policy_value_score(env,
                        torch.as_tensor(paths[..., 1:10]), torch.as_tensor(paths[..., 10:19]),
                        torch.as_tensor(plans), credit_value, credit_gamma)
                    scores, endpoints, healthy = scores.numpy(), endpoints.numpy(), healthy.numpy()
                return np.asarray(scores), np.asarray(endpoints), np.asarray(healthy)
            baseline, nominal_end, nominal_healthy = score(nominal[None])
            draws = MPPI(env, replace(cfg, noise_sigma=policy_sigma), seed=int(seed))._sample_noise()
            sampled = make_proposals(torch.as_tensor(nominal[None]), torch.as_tensor(draws[None]), low, high,
                proposal_mode=proposal_mode, policy_sigma=policy_sigma)[0]
            scores, sample_end, sample_healthy = score(sampled.numpy())
            mean_cache = {}
            def verify(plans, row_indices):
                cost, end, healthy = score(plans.numpy())
                mean_cache.update(end=end, healthy=healthy)
                return torch.as_tensor(cost)
            result = synthesize_verified(sampled[None], torch.as_tensor(scores[None]),
                torch.as_tensor(nominal[None]), torch.as_tensor(baseline), low, high, verify,
                synthesis_mode=synthesis_mode, neff_fraction=cfg.lam_target_neff_fraction)
            source = int(result.source_codes[0])
            if source == 2:
                index = int(result.anchor_indices[0]); end, healthy = sample_end[index], sample_healthy[index]
            elif source == 1:
                end, healthy = mean_cache["end"][0], mean_cache["healthy"][0]
            else:
                end, healthy = nominal_end[0], nominal_healthy[0]
            outputs.append({"labels": result.labels.numpy(), "nominal_first": nominal[None, 0],
                "nominal_cost": baseline, "plan_cost": result.costs.numpy(),
                "mean_cost": result.mean_costs.numpy(), "mean_first": result.mean_plans[:, 0].numpy(),
                "expected_candidate_cost": result.expected_candidate_cost.numpy(),
                "n_eff": result.n_eff.numpy(), "source_codes": result.source_codes.numpy(),
                "accepted": result.accepted.numpy(), "endpoint_obs": end[None],
                "endpoint_healthy": np.asarray([healthy]),
                "independent_selected": np.asarray([source == 2 and int(result.anchor_indices[0]) >= cfg.K // 2])})
            diagnostics.append(result.diagnostics())
            length = cfg.H + (self.tail_steps if credit_value is None else 0)
            ledger["candidate_controls"] += cfg.K * length
            ledger["nominal_generation_controls"] += cfg.H + length
            ledger["mean_verification_controls"] += length
        out = {key: np.concatenate([row[key] for row in outputs]) for key in outputs[0]}
        counts = ("states", "accepted", "mean_rejected_count", "mean_tied_count", "bimodal_collapse_count",
                  "medoid_rescued_collapse_count", "rescue_eligible_count", "rescued_count")
        out["diagnostics"] = {key: sum(d[key] for d in diagnostics) for key in counts}
        out["ledger"] = {"logical": ledger, "physical_capacity": dict(ledger),
                         "physical_note": "CPU reference conservatively charges requested tail slots, including absorbed suffixes."}
        return out
