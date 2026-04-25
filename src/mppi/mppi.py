"""Information-theoretic MPPI — Algorithm 2 of Williams et al. 2017
(https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf).

γ = λ (α = 0 case). Weights are computed directly from
    w_k = (1/η) exp( -(1/λ)(S_k - ρ) ),   ρ = min_k S_k
with S_k = running_cost + terminal_cost + γ·Σ_t u_t^T Σ^{-1} ε_{k,t}
         + (optional) policy-tracking cost from GPS.
"""
import numpy as np
from src.envs.base import BaseEnv
from src.utils.config import MPPIConfig
from src.utils.math import effective_sample_size

class MPPI:
    
    def __init__(self, env: BaseEnv, cfg: MPPIConfig):
        self.env = env 
        self.cfg = cfg
        self.K = cfg.K 
        self.H = cfg.H
        self.lam = cfg.lam
        self.sigma = cfg.noise_sigma

        self.nu = env.action_dim
        self.act_low, self.act_high = env.action_bounds

        self.reset()

        self._last_states = None
        self._last_actions = None
        self._last_weights = None
        self._last_costs = None
        self._last_sensordata = None

    def reset(self):
        self.U = np.zeros((self.H, self.nu))

    def plan_step(
            self,
            state: np.ndarray,
            nominal: np.ndarray | None = None,
            nominal_first: np.ndarray | None = None,
            prior_cost = None,
    ) -> tuple[np.ndarray, dict]:
        """One MPPI iteration (paper's Algorithm 2).

        state: current environment state
        nominal: optional (H, nu) full action sequence to replace self.U before
            perturbing. Centers the sampling distribution on a guiding policy's
            full rollout (GPS-style). Overrides nominal_first if both given.
        nominal_first: optional (nu,) action to overwrite U[0] only.
        prior_cost: optional callable (states, actions) -> (K,) cost in env-cost
            units. Added to S_k before the softmin. This is how GPS injects the
            λ_track · ‖a − π(s)‖² policy-tracking term.
        """
        if nominal is not None:
            self.U = np.clip(nominal.copy(), self.act_low, self.act_high)
        elif nominal_first is not None:
            self.U[0] = nominal_first

        # sample ε ~ N(0, σ² I), perturb, clamp for rollout
        eps = np.random.randn(self.K, self.H, self.nu) * self.sigma
        U_perturbed = self.U[None, :, :] + eps
        U_clipped = np.clip(U_perturbed, self.act_low, self.act_high)

        # rollouts → per-sample base cost (running + terminal)
        states, costs, sensordata = self.env.batch_rollout(state, U_clipped)

        # assemble S_k components (paper's Algorithm 2, γ=λ, Σ=σ²I):
        #    S_k = S_env + λ · Σ_t u_t · ε_{k,t}/σ² + (optional) λ_track · Σ_t ‖a-π‖²
        lam = self.lam
        is_corr = self._is_correction(eps, lam)
        track = prior_cost(states, U_clipped) if prior_cost is not None else None
        S = costs + is_corr + (track if track is not None else 0.0)

        # paper weights: ρ = min_k S_k, w_k = exp(-(S_k - ρ)/λ) / η
        weights, n_eff = self._softmin_weights(S, lam)

        # adaptive λ (not in paper; keeps n_eff in a sensible range)
        if self.cfg.adaptive_lam:
            for _ in range(5):
                if n_eff < self.cfg.n_eff_threshold:
                    lam *= 2.0
                elif n_eff > 0.75 * self.K:
                    lam *= 0.5
                else:
                    break
                lam = float(np.clip(lam, 0.01, 100.0))
                # γ=λ: is_corr depends on λ; track does not
                is_corr = self._is_correction(eps, lam)
                S = costs + is_corr + (track if track is not None else 0.0)
                weights, n_eff = self._softmin_weights(S, lam)
            self.lam = lam

        # weighted update on raw ε (not clipped U) to avoid clipping bias
        self.U = self.U + np.einsum('k, kha -> ha', weights, eps)
        self.U = np.clip(self.U, self.act_low, self.act_high)

        action = self.U[0].copy()

        # shift horizon
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2].copy()

        # stash for GPS
        self._last_states = states
        self._last_actions = U_clipped
        self._last_weights = weights
        self._last_costs = costs
        self._last_sensordata = sensordata

        # S-component diagnostics (all in the same cost units — directly comparable)
        info = {
            'cost_mean': float(np.mean(costs)),
            'cost_min': float(np.min(costs)),
            'cost_env_mean': float(np.mean(costs)),            # running + terminal
            'cost_is_mean': float(np.mean(is_corr)),           # IS term (≈0 by symmetry)
            'cost_is_std': float(np.std(is_corr)),             # IS term magnitude
            'cost_track_mean': float(np.mean(track)) if track is not None else 0.0,
            'cost_s_mean': float(np.mean(S)),                  # total S = sum of above
            'n_eff': float(n_eff),
            'lam': float(lam),
        }
        return action, info

    def _is_correction(self, eps: np.ndarray, lam: float) -> np.ndarray:
        """γ · Σ_t u_t^T Σ^{-1} ε_{k,t} with γ=λ, Σ=σ²I → (K,)."""
        return lam * np.sum(self.U[None, :, :] * eps, axis=(1, 2)) / (self.sigma ** 2)

    def _softmin_weights(self, S: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        """Paper's weight formula with min-baseline stabilization."""
        rho = np.min(S)
        unnorm = np.exp(-(S - rho) / lam)
        eta = np.sum(unnorm)
        weights = unnorm / eta
        return weights, effective_sample_size(weights)
    
    def get_rollout_data(self) -> dict:
        return {
            'states': self._last_states,
            'actions': self._last_actions,
            'weights': self._last_weights,
            'costs': self._last_costs,
            'sensordata': self._last_sensordata, 
            }
    



                
                




