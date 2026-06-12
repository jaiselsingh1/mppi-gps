"""dataclass configurations for mppi-gps"""
import json
from dataclasses import dataclass
from pathlib import Path

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

@dataclass 
class MPPIConfig:
    K: int = 256 # number of samples 
    H: int = 256 # planning horizon 
    lam: float = 1.0 # temperature parameter 
    # this parameter essentially helps you know how much you want to focus on specific samples vs others 
    noise_sigma: float = 0.5 # exploration noise std 
    use_is_correction: bool = False
    # Optional per-action exploration std. Builds diag(noise_std^2) internally.
    noise_std: list[float] | None = None
    # Optional full action covariance. When unset, MPPI uses noise_sigma^2 * I.
    noise_cov: list[list[float]] | None = None

    @staticmethod
    def load(env_name: str) -> "MPPIConfig":
        """Load best tuned params from configs/<env_name>_best.json."""
        path = _CONFIGS_DIR / f"{env_name}_best.json"
        params = json.loads(path.read_text())
        return MPPIConfig(**params)

@dataclass
class PolicyConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    lr: float = 1e-4
    activation: str = "relu"
    use_history: bool = False # gate the history path 
    history_len: int = 8

@dataclass
class GPSConfig:
    n_gps_iters: int = 30
    episodes_per_iter: int = 10
    steps_per_episode: int = 1000
    batch_size: int = 128
    bc_epochs_per_iter: int = 1
    bc_max_epochs: int = 0             # >0: train to plateau/target, capped here
    bc_target_loss: float = 0.0        # epoch-mean MSE early-stop threshold
    replay_max_pairs: int = 0
    eval_every: int = 5
    eval_n_episodes: int = 10
    eval_episode_len: int = 1000
    eval_mppi_baseline_episodes: int = 0
    coupling_warmup_iters: int = 5
    lambda_policy_track: float = 0.001 # cost-unit weight for policy-tracking prior
    adaptive_policy_trust: bool = True
    policy_trust_bad_cost_per_step: float = 12.0
    policy_trust_min: float = 0.0
    policy_trust_max: float = 1.0
    collection_mode: str = "gps"       # bc | gps
    coupling_mode: str = "track"       # track | filter | merge; used when collection_mode == "gps"
    # merge mode: policy nudges the post-update plan, never the score.
    # Largest blend in merge_betas whose rollout cost stays within budget wins.
    merge_betas: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)
    merge_delta_frac: float = 0.01     # accepted task-cost regression, fraction of J(U*)
    merge_delta_floor: float = 1.0     # cost floor so a near-zero J(U*) still has budget
    # TD-MPC-style sample mixing: fraction of MPPI samples centered on the
    # policy's closed-loop rollout. Task score arbitrates; no trust schedule.
    mix_fraction: float = 0.0
    # >0: replay samples from iter t get weight 0.5^((now-t)/halflife) in BC.
    bc_recency_halflife: float = 0.0
    # PLATO-style share of collection episodes driven by the policy while
    # states are labeled with the certified planner action (recovery data).
    dagger_fraction: float = 0.0
    # Mordatch'15-style in-score coupling with dual ascent: when alpha > 0
    # (and coupling_mode == "merge"), add lambda_track * sum_t ||u - pi||^2
    # to the MPPI score with lambda <- min(lambda + alpha * E||u_exec -
    # pi||^2, cap). Grows while the policy and planner disagree, stalls as
    # they converge — no hand schedule.
    track_dual_alpha: float = 0.0
    track_dual_lambda_max: float = 1.0
    # GPS-style stochastic collection: execute label + N(0, std^2) so the
    # dataset covers a tube around nominal trajectories (recovery labels).
    exec_noise_std: float = 0.0
    # >0: OU-correlate the exec noise with this time constant (control
    # steps). White actuator noise is low-pass-filtered by the plant;
    # persistent noise mimics the learner's directional drift (DART).
    exec_noise_ou_tau: float = 0.0
    # z-score policy inputs with replay-buffer stats each iteration
    normalize_obs: bool = False
    policy_coupling_min_fraction: float = 0.05
    policy_coupling_keep_fraction: float = 1.0
    policy_coupling_min_n_eff: float = 0.0
    policy_coupling_max_weight: float = 1.0
    obs_dim: int = 6
    act_dim: int = 1

    @staticmethod
    def load(env_name: str) -> "GPSConfig":
        path = _CONFIGS_DIR / f"gps_{env_name}.json"
        params = json.loads(path.read_text())
        return GPSConfig(**params)
