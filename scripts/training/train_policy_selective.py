"""Policy-selective MPPI distillation with the requested Walker experiment preset."""
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import tyro

from scripts.training.train_walker_mdgps import TrainConfig, main as train


@dataclass
class PolicySelectiveConfig(TrainConfig):
    """Training-only MPPI teacher; the saved tanh MLP acts alone at evaluation.

    All collected states are independent. GPU chunks bound physics memory and
    never carry plans between states. Increase gpu_states_per_batch for a larger
    physical batch, subject to the device's available memory.
    """

    H: int = 32
    tail_steps: int = 100
    iterations: int = 75
    episodes_per_iter: int = 10
    steps: int = 400
    sigma_decay_iterations: float | None = 25.0
    selection_mode: Literal["legacy", "anchor", "selective"] = "selective"
    policy_architecture: Literal["sac", "tanh"] = "tanh"
    gpu_estep: bool = True
    m_step_updates: int | None = 2000
    workers: int = 4
    research_ladder: bool = True


def main(out_dir: Path, cfg: PolicySelectiveConfig, enable_early_abort: bool = True) -> None:
    """Run a fresh experiment, or resume with the saved configuration."""
    train(out_dir, replace(cfg, enable_early_abort=cfg.enable_early_abort and enable_early_abort))


if __name__ == "__main__":
    tyro.cli(main)
