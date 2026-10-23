"""Write the weight average of several policy checkpoints, in load_policy format."""
from pathlib import Path

import torch
import tyro

from src.gps.averaging import weight_average
from src.rl.sac import load_policy


def main(out: Path, checkpoints: tuple[Path, ...]) -> None:
    if out.exists():
        raise FileExistsError(out)
    states = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        states.append(payload.get("policy", payload))
    averaged = weight_average(states)
    load_policy(averaged)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, out)


if __name__ == "__main__":
    tyro.cli(main)
