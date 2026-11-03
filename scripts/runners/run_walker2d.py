"""Compatibility facade for the native Walker runner in the research library."""

import sys
from functools import wraps
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.gps.mppi_gps import native_runner as _implementation

# PEP 562 supplies these historical exports lazily.
__all__ = [
    name
    for name in (
        "hashlib",
        "inspect",
        "json",
        "platform",
        "sys",
        "time",
        "asdict",
        "version",
        "Path",
        "mujoco",
        "np",
        "Walker2d",
        "MPPI",
        "MPPIConfig",
        "main",
    )
]


@wraps(_implementation.main)
def main(*args, **kwargs):
    # Keep historical module-level dependency overrides (including test doubles).
    return _implementation._run(*args, **kwargs, _overrides=globals())


def __getattr__(name):
    return getattr(_implementation, name)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
