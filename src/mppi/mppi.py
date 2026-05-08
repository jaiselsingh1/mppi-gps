"""Information-theoretic MPPI — Algorithm 2 of Williams et al. 2017
(https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf).

Weights are computed directly from
    w_k = (1/η) exp( -(1/λ)(S_k - ρ) ),   ρ = min_k S_k
with S_k = running_cost + terminal_cost + optional Σ_t u_t^T Σ^{-1} ε_{k,t}
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
        self.use_is_correction = cfg.use_is_correction
        if self.lam <= 0.0:
            raise ValueError(f"MPPI temperature lam must be positive, got {self.lam}.")
        if cfg.clip_actions and self.use_is_correction:
            raise ValueError("clip_actions is not compatible with use_is_correction.")

        self.nu = env.action_dim
        self.action_low, self.action_high = env.action_bounds
        self.noise_cov, self.noise_chol, self.noise_precision = self._build_noise_model(cfg)

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
            coupling = None,
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
        coupling: optional callable that can replace the MPPI score vector after
            env cost/IS/prior-cost assembly.
        """
        if nominal is not None:
            self.U = nominal.copy()
        elif nominal_first is not None:
            self.U[0] = nominal_first
        if self.cfg.clip_actions:
            self.U = np.clip(self.U, self.action_low, self.action_high)

        # When clipping is enabled, use the effective bounded perturbation for
        # the update so sampled rollouts and the nominal sequence stay feasible.
        noise = self._sample_noise()
        U_noisy = self.U[None, :, :] + noise
        if self.cfg.clip_actions:
            U_sampled = np.clip(U_noisy, self.action_low, self.action_high)
            eps = U_sampled - self.U[None, :, :]
        else:
            U_sampled = U_noisy
            eps = noise

        # rollouts → per-sample base cost (running + terminal)
        states, costs, sensordata = self.env.batch_rollout(state, U_sampled)

        # assemble S_k components:
        #    S_k = S_env + optional Σ_t u_t^T Σ^{-1} ε_{k,t}
        #          + optional λ_track · Σ_t ‖a-π‖²
        lam = self.lam
        is_corr = self._is_correction(eps) if self.use_is_correction else np.zeros(self.K)
        track = prior_cost(states, U_sampled) if prior_cost is not None else None
        S_base = costs + is_corr + (track if track is not None else 0.0)
        S, coupling_diag, fallback_score = self._apply_coupling(
            coupling,
            states,
            U_sampled,
            costs,
            S_base,
            lam,
        )

        # paper weights: ρ = min_k S_k, w_k = exp(-(S_k - ρ)/λ) / η
        weights, n_eff = self._softmin_weights(S, lam)
        weights, n_eff, S, used_coupling_fallback = self._maybe_fallback_coupling_weights(
            weights,
            n_eff,
            S,
            fallback_score,
            coupling_diag,
            lam,
        )

        # weighted update on sampled perturbations
        self.U = self.U + np.einsum('k, kha -> ha', weights, eps)
        if self.cfg.clip_actions:
            self.U = np.clip(self.U, self.action_low, self.action_high)
        action = np.clip(self.U[0].copy(), self.action_low, self.action_high)

        # shift horizon
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2].copy()
        if self.cfg.clip_actions:
            self.U = np.clip(self.U, self.action_low, self.action_high)

        # stash for GPS
        self._last_states = states
        self._last_actions = U_sampled
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
            'cost_s_mean': self._finite_mean(S),               # total S = sum of above
            'n_eff': float(n_eff),
            'lam': float(lam),
            'use_is_correction': float(self.use_is_correction),
            'coupling_active': coupling_diag['active'],
            'coupling_used_fallback': float(used_coupling_fallback),
            'coupling_feasible_fraction': coupling_diag['feasible_fraction'],
            'coupling_policy_cost_mean': coupling_diag['policy_cost_mean'],
            'coupling_policy_cost_std': coupling_diag['policy_cost_std'],
            'coupling_score_mean': coupling_diag['score_mean'],
        }
        return action, info

    def _is_correction(self, eps: np.ndarray) -> np.ndarray:
        """Σ_t u_t^T Σ^{-1} ε_{k,t} → (K,)."""
        precision_eps = np.einsum('ij,ktj->kti', self.noise_precision, eps)
        return np.sum(self.U[None, :, :] * precision_eps, axis=(1, 2))

    def _sample_noise(self) -> np.ndarray:
        standard = np.random.randn(self.K, self.H, self.nu)
        return np.einsum('khi,ji->khj', standard, self.noise_chol)

    def _build_noise_model(
            self,
            cfg: MPPIConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cfg.noise_std is not None and cfg.noise_cov is not None:
            raise ValueError("Set either noise_std or noise_cov, not both.")

        if cfg.noise_std is not None:
            std = np.asarray(cfg.noise_std, dtype=float)
            if std.shape != (self.nu,):
                raise ValueError(
                    f"MPPI noise_std must have shape {(self.nu,)}, got {std.shape}."
                )
            if np.any(std <= 0.0):
                raise ValueError("MPPI noise_std entries must be positive.")
            cov = np.diag(std ** 2)
        elif cfg.noise_cov is None:
            if self.sigma <= 0.0:
                raise ValueError(f"MPPI noise_sigma must be positive, got {self.sigma}.")
            cov = (self.sigma ** 2) * np.eye(self.nu)
        else:
            cov = np.asarray(cfg.noise_cov, dtype=float)
            if cov.shape != (self.nu, self.nu):
                raise ValueError(
                    f"MPPI noise_cov must have shape {(self.nu, self.nu)}, got {cov.shape}."
                )
            if not np.allclose(cov, cov.T):
                raise ValueError("MPPI noise_cov must be symmetric.")

        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("MPPI noise covariance must be positive definite.") from exc

        precision = np.linalg.inv(cov)
        return cov, chol, precision

    def _softmin_weights(self, S: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
        """Paper's weight formula with min-baseline stabilization."""
        finite = np.isfinite(S)
        if not np.any(finite):
            weights = np.full_like(S, 1.0 / len(S), dtype=float)
            return weights, effective_sample_size(weights)

        rho = np.min(S[finite])
        shifted = np.where(finite, S - rho, np.inf)
        unnorm = np.exp(-shifted / lam)
        eta = np.sum(unnorm)
        if not np.isfinite(eta) or eta <= 0.0:
            weights = np.zeros_like(S, dtype=float)
            weights[np.argmin(shifted)] = 1.0
            return weights, effective_sample_size(weights)

        weights = unnorm / eta
        return weights, effective_sample_size(weights)

    def _finite_mean(self, x: np.ndarray) -> float:
        finite = x[np.isfinite(x)]
        if len(finite) == 0:
            return float("inf")
        return float(np.mean(finite))

    def _apply_coupling(
            self,
            coupling,
            states: np.ndarray,
            actions: np.ndarray,
            costs: np.ndarray,
            base_score: np.ndarray,
            lam: float,
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray | None]:
        default_diag = {
            'active': 0.0,
            'feasible_fraction': 1.0,
            'policy_cost_mean': 0.0,
            'policy_cost_std': 0.0,
            'score_mean': float(np.mean(base_score)),
        }
        if coupling is None:
            return base_score, default_diag, None

        result = coupling(
            states=states,
            actions=actions,
            costs=costs,
            base_score=base_score,
            lam=lam,
        )
        score = np.asarray(result["score"], dtype=float)
        fallback_score = np.asarray(result.get("fallback_score", base_score), dtype=float)
        diag = default_diag | result.get("info", {})
        diag["min_n_eff"] = float(result.get("min_n_eff", 0.0))
        diag["max_weight"] = float(result.get("max_weight", 1.0))
        if not np.any(np.isfinite(score)):
            score = fallback_score
            diag["active"] = 0.0
        return score, diag, fallback_score

    def _maybe_fallback_coupling_weights(
            self,
            weights: np.ndarray,
            n_eff: float,
            score: np.ndarray,
            fallback_score: np.ndarray | None,
            coupling_diag: dict[str, float],
            lam: float,
    ) -> tuple[np.ndarray, float, np.ndarray, bool]:
        if fallback_score is None:
            return weights, n_eff, score, False

        min_n_eff = coupling_diag.get('min_n_eff', 0.0)
        max_weight = coupling_diag.get('max_weight', 1.0)
        should_fallback = n_eff < min_n_eff or float(np.max(weights)) > max_weight
        if not should_fallback:
            return weights, n_eff, score, False

        fallback_weights, fallback_n_eff = self._softmin_weights(fallback_score, lam)
        return fallback_weights, fallback_n_eff, fallback_score, True
    
    def get_rollout_data(self) -> dict:
        return {
            'states': self._last_states,
            'actions': self._last_actions,
            'weights': self._last_weights,
            'costs': self._last_costs,
            'sensordata': self._last_sensordata, 
            }
    



                
                
