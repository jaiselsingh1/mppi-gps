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
    # Optional temporal shaping for absolute-action perturbations.
    noise_lowpass_cutoff_hz: float | None = None
    noise_lowpass_sample_rate_hz: float | None = None
    noise_lowpass_order: int = 2
    # Total optimization passes on the first plan_step after each reset.
    initial_iterations: int = 1

    @staticmethod
    def load(env_name: str) -> "MPPIConfig":
        """Load best tuned params from configs/<env_name>_best.json."""
        path = _CONFIGS_DIR / f"{env_name}_best.json"
        params = json.loads(path.read_text())
        return MPPIConfig(**params)
