"""Information-theoretic MPPI — Algorithm 2 of Williams et al. 2017
(https://homes.cs.washington.edu/~bboots/files/InformationTheoreticMPC.pdf).

Weights are computed directly from
    w_k = (1/η) exp( -(1/λ)(S_k - ρ) ),   ρ = min_k S_k
with S_k = running_cost + terminal_cost + optional Σ_t u_t^T Σ^{-1} ε_{k,t}
         + (optional) policy-tracking cost from GPS.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, toeplitz
from scipy.signal import butter, tf2ss

from src.envs.base import BaseEnv
from src.utils.config import MPPIConfig
from src.utils.math import effective_sample_size

class MPPI:
    
    def __init__(self, env: BaseEnv, cfg: MPPIConfig, seed: int | None = None):
        self.env = env 
        self.cfg = cfg
        self.K = cfg.K 
        self.H = cfg.H
        self.lam = cfg.lam
        self.sigma = cfg.noise_sigma
        self.noise_lowpass_cutoff_hz = cfg.noise_lowpass_cutoff_hz
        self.noise_lowpass_sample_rate_hz = cfg.noise_lowpass_sample_rate_hz
        self.noise_lowpass_order = cfg.noise_lowpass_order
        self.noise_lowpass_active = self.noise_lowpass_cutoff_hz is not None
        self.use_is_correction = cfg.use_is_correction
        self.initial_iterations = cfg.initial_iterations
        if not np.isfinite(self.lam) or self.lam <= 0.0:
            raise ValueError(f"MPPI temperature lam must be positive, got {self.lam}.")
        if not isinstance(self.H, int) or self.H < 2:
            raise ValueError(f"MPPI horizon H must be at least 2, got {self.H}.")
        if not isinstance(self.K, int) or self.K < 1:
            raise ValueError(f"MPPI sample count K must be positive, got {self.K}.")
        if type(self.initial_iterations) is not int or self.initial_iterations < 1:
            raise ValueError("MPPI initial_iterations must be a positive integer.")
        if self.use_is_correction:
            raise ValueError("use_is_correction is not compatible with bounded action clipping.")

        self.nu = env.action_dim
        self.action_low, self.action_high = env.action_bounds
        self.noise_cov, self.noise_chol, self.noise_precision = self._build_noise_model(cfg)
        self._noise_temporal_chol = self._build_temporal_noise_model(cfg)
        if self.noise_lowpass_active and hasattr(env, "_dt"):
            if not np.isclose(self.noise_lowpass_sample_rate_hz, 1.0 / env._dt):
                raise ValueError("Noise sample rate must match the environment control rate.")
        self._rng = np.random.default_rng(seed) if seed is not None else None

        self.reset()

        self._last_states = None
        self._last_actions = None
        self._last_weights = None
        self._last_costs = None
        self._last_sensordata = None

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.U = np.zeros((self.H, self.nu))
        self._initial_plan = True

    def plan_step(
            self,
            state: np.ndarray,
            nominal: np.ndarray | None = None,
            nominal_first: np.ndarray | None = None,
            prior_cost = None,
            coupling = None,
            initial_warmstart: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Optimize one control decision, then shift the horizon once.

        The first decision after reset uses cfg.initial_iterations passes from
        the same state and solver warm-start. Later decisions use one pass.
        Diagnostics and cached rollouts describe the final pass.

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
        if not np.all(np.isfinite(state)):
            raise ValueError("MPPI requires a finite initial state.")
        if nominal is not None:
            if nominal.shape != (self.H, self.nu) or not np.all(np.isfinite(nominal)):
                raise ValueError("nominal must be a finite (H, action_dim) sequence.")
            self.U = nominal.copy()
        elif nominal_first is not None:
            if nominal_first.shape != (self.nu,) or not np.all(np.isfinite(nominal_first)):
                raise ValueError("nominal_first must be a finite action vector.")
            self.U[0] = nominal_first
        self.U = np.clip(self.U, self.action_low, self.action_high)

        iterations = self.initial_iterations if self._initial_plan else 1
        for _ in range(iterations):
            action, info = self._plan_iteration(
                state, prior_cost, coupling, initial_warmstart
            )

        # Refinement passes optimize the same horizon; only an executed control
        # decision advances it. The returned action is copied before shifting.
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2].copy()
        self._initial_plan = False
        info['optimization_iterations'] = iterations
        return action, info

    def _plan_iteration(
            self,
            state: np.ndarray,
            prior_cost,
            coupling,
            initial_warmstart: np.ndarray | None,
    ) -> tuple[np.ndarray, dict]:
        """Refine the current nominal without advancing the control horizon."""

        # Use the effective bounded perturbation for the update so sampled
        # rollouts and the nominal sequence stay feasible.
        noise = self._sample_noise()
        U_noisy = self.U[None, :, :] + noise
        U_sampled = np.clip(U_noisy, self.action_low, self.action_high)
        eps = U_sampled - self.U[None, :, :]

        # rollouts → per-sample base cost (running + terminal)
        rollout_kwargs = {}
        if initial_warmstart is not None:
            rollout_kwargs["initial_warmstart"] = initial_warmstart
        states, costs, sensordata = self.env.batch_rollout(
            state, U_sampled, **rollout_kwargs
        )

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
        self.U = np.clip(self.U, self.action_low, self.action_high)
        action = self.U[0].copy()

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
            'noise_lowpass_active': float(self.noise_lowpass_active),
        }
        return action, info

    def _is_correction(self, eps: np.ndarray) -> np.ndarray:
        """Σ_t u_t^T Σ^{-1} ε_{k,t} → (K,)."""
        precision_eps = np.einsum('ij,ktj->kti', self.noise_precision, eps)
        return np.sum(self.U[None, :, :] * precision_eps, axis=(1, 2))

    def _sample_noise(self) -> np.ndarray:
        if self._rng is None:
            standard = np.random.randn(self.K, self.H, self.nu)
        else:
            standard = self._rng.standard_normal((self.K, self.H, self.nu))
        if self._noise_temporal_chol is not None:
            standard = np.einsum(
                'ht,kta->kha',
                self._noise_temporal_chol,
                standard,
                optimize=True,
            )
        return np.einsum('khi,ji->khj', standard, self.noise_chol)

    def _build_temporal_noise_model(
            self,
            cfg: MPPIConfig,
    ) -> np.ndarray | None:
        """Build a stationary Butterworth covariance over the planning horizon."""
        cutoff = cfg.noise_lowpass_cutoff_hz
        sample_rate = cfg.noise_lowpass_sample_rate_hz
        order = cfg.noise_lowpass_order

        if cutoff is None:
            if sample_rate is not None:
                raise ValueError(
                    "noise_lowpass_sample_rate_hz requires noise_lowpass_cutoff_hz."
                )
            return None
        if sample_rate is None:
            raise ValueError(
                "noise_lowpass_cutoff_hz requires noise_lowpass_sample_rate_hz."
            )
        if type(order) is not int or order != 2:
            raise ValueError("noise_lowpass_order must be the integer value 2.")

        cutoff = float(cutoff)
        sample_rate = float(sample_rate)
        if not np.isfinite(sample_rate) or sample_rate <= 0.0:
            raise ValueError("noise_lowpass_sample_rate_hz must be positive and finite.")
        if not np.isfinite(cutoff) or not 0.0 < cutoff < 0.5 * sample_rate:
            raise ValueError(
                "noise_lowpass_cutoff_hz must be positive, finite, and below Nyquist."
            )

        numerator, denominator = butter(
            order,
            cutoff,
            btype="lowpass",
            fs=sample_rate,
            output="ba",
        )
        state_a, state_b, state_c, state_d = tf2ss(numerator, denominator)
        state_covariance = solve_discrete_lyapunov(
            state_a,
            state_b @ state_b.T,
        )

        autocovariance = np.empty(self.H, dtype=float)
        autocovariance[0] = (
            state_c @ state_covariance @ state_c.T + state_d @ state_d.T
        ).item()
        previous_power = np.eye(state_a.shape[0], dtype=float)
        for lag in range(1, self.H):
            current_power = state_a @ previous_power
            autocovariance[lag] = (
                state_c @ current_power @ state_covariance @ state_c.T
                + state_c @ previous_power @ state_b @ state_d.T
            ).item()
            previous_power = current_power

        if not np.all(np.isfinite(autocovariance)) or autocovariance[0] <= 0.0:
            raise ValueError("Butterworth stationary autocovariance is invalid.")
        correlation = toeplitz(autocovariance / autocovariance[0])
        correlation = 0.5 * (correlation + correlation.T)
        np.fill_diagonal(correlation, 1.0)
        try:
            return np.linalg.cholesky(correlation)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Butterworth temporal correlation is not positive definite."
            ) from exc

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
            if not np.all(np.isfinite(std)) or np.any(std <= 0.0):
                raise ValueError("MPPI noise_std entries must be positive.")
            cov = np.diag(std ** 2)
        elif cfg.noise_cov is None:
            if not np.isfinite(self.sigma) or self.sigma <= 0.0:
                raise ValueError(f"MPPI noise_sigma must be positive, got {self.sigma}.")
            cov = (self.sigma ** 2) * np.eye(self.nu)
        else:
            cov = np.asarray(cfg.noise_cov, dtype=float)
            if cov.shape != (self.nu, self.nu):
                raise ValueError(
                    f"MPPI noise_cov must have shape {(self.nu, self.nu)}, got {cov.shape}."
                )
            if not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T):
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
            raise FloatingPointError("MPPI has no finite trajectory scores; refusing an unscored action.")

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
    



                
                
