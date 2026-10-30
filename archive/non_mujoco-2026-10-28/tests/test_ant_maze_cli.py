import inspect
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
import pytest
import tyro

matplotlib.use("Agg")

from scripts.runners import run_ant_maze as runner


def parsed_options(argv):
    def capture(**kwargs):
        return kwargs
    capture.__signature__ = inspect.signature(runner.main)
    capture.__annotations__ = runner.main.__annotations__.copy()
    return tyro.cli(capture, args=argv)


def test_cli_parses_standard_controller_overrides():
    options = parsed_options([
        "--episodes", "1", "--steps", "2", "--no-render",
        "--k", "4", "--h", "3", "--noise-sigma", "0.1",
        "--out-dir", "runs/ant experiment",
    ])
    assert options["render"] is False
    assert options["k"] == 4
    assert options["h"] == 3
    assert options["noise_sigma"] == 0.1
    assert options["out_dir"] == "runs/ant experiment"
    assert "backend" not in options
    assert "device" not in options


@pytest.mark.parametrize("argv", [["--backend", "warp"], ["--device", "cuda"]])
def test_archived_gpu_cli_options_are_rejected(argv):
    with pytest.raises(SystemExit) as error:
        parsed_options(argv)
    assert error.value.code == 2


def test_import_and_help_do_not_load_archived_learning_modules():
    root = Path(runner.__file__).resolve().parents[2]
    script = """
import importlib.abc
import sys

class BlockLearningImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefixes = ('scripts.training', 'src.policy', 'src.gps')
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in prefixes):
            raise AssertionError('Archived learning import: ' + fullname)

sys.meta_path.insert(0, BlockLearningImports())
from scripts.runners.run_ant_maze import main
import tyro
tyro.cli(main, args=['--help'])
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--noise-sigma" in result.stdout
    assert "--backend" not in result.stdout
    assert "--device" not in result.stdout


def test_short_headless_run_uses_standard_mppi_and_writes_report(tmp_path):
    runner.main(
        episodes=1, steps=2, seed=91, render=False,
        out_dir=str(tmp_path), k=4, h=3, noise_sigma=0.1,
    )
    report = json.loads((tmp_path / "summary.json").read_text())
    assert report["backend"] == "mujoco_cpu"
    assert report["episodes"] == 1
    assert report["steps"] == 2
    assert report["mppi"]["K"] == 4
    assert report["mppi"]["H"] == 3
    assert len(report["episodes_detail"]) == 1
    assert 0.0 <= report["mean_healthy_frac"] <= 1.0
    assert (tmp_path / "ant_maze_mppi_path.png").stat().st_size > 0
    assert not (tmp_path / "ant_maze_mppi.mp4").exists()
