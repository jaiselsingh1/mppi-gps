"""dataclass configurations for mppi-gps"""
import json
from dataclasses import dataclass
from pathlib import Path

_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

@dataclass 
class MPPIConfig:
    K: int = 256 # number of samples 
    H: int = 50 # planning horizon 
    lam: float = 1.0 # temperature parameter 
    # this parameter essentially helps you know how much you want to focus on specific samples vs others 
    noise_sigma: float = 0.5 # exploration noise std 
    noise_smoothing: float = 0.0 # temporal correlation for sampled action noise
    is_correction_scale: float = 1.0 # 1.0 is Williams et al.; 0.0 is useful for non-paper nominal priors

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
    replay_max_pairs: int = 0
    action_ema_alpha: float = 0.0
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
    coupling_mode: str = "cost"        # raw | cost | filter | hard_filter | hybrid
    policy_coupling_beta: float = 0.3
    policy_coupling_cost_slack_rel: float = 0.25
    policy_coupling_cost_slack_abs: float = 0.0
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
