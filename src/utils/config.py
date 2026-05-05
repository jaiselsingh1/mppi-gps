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
    adaptive_lam: bool = False # adapt lam in order to maintain the n_eff
    n_eff_threshold: float = 64.0 # number of samples that you want to contribute to the weighted mean

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
    eval_every: int = 5
    eval_n_episodes: int = 10
    eval_episode_len: int = 1000
    lambda_policy_track: float = 0.2   # 0 = R1 control; >0 = R2 (policy-biased MPPI)
    coupling_mode: str = "cost"        # raw | cost | filter
    policy_coupling_beta: float = 0.3
    policy_coupling_cost_slack_rel: float = 0.25
    policy_coupling_cost_slack_abs: float = 0.0
    policy_coupling_min_fraction: float = 0.05
    policy_coupling_min_n_eff: float = 8.0
    policy_coupling_max_weight: float = 0.8
    obs_dim: int = 6
    act_dim: int = 1

    @staticmethod
    def load(env_name: str) -> "GPSConfig":
        path = _CONFIGS_DIR / f"gps_{env_name}.json"
        params = json.loads(path.read_text())
        return GPSConfig(**params)
