import inspect
from pathlib import Path

import pytest
import tyro

from scripts.runners import run_walker2d as runner


def parsed_options(argv):
    def capture(**kwargs):
        return kwargs
    capture.__signature__ = inspect.signature(runner.main)
    capture.__annotations__ = runner.main.__annotations__.copy()
    return tyro.cli(capture, args=argv)


def test_cli_preserves_defaults_and_hyphenated_option_forms():
    root = Path(runner.__file__).resolve().parents[2]
    assert parsed_options([]) == {
        'config': root / 'configs/walker2d_best.json',
        'task_config': root / 'configs/walker2d_task.json',
        'planner_seed': 10000, 'traces_output': None, 'episodes': 5,
        'steps': 500, 'seed': 0, 'render': False,
        'output': Path('runs/walker2d_teacher/walker2d.mp4'),
        'metrics_output': None, 'log_every': 20,
    }
    assert parsed_options([
        '--config', 'controller.json', '--task-config', 'task.json',
        '--planner-seed', '500', '--traces-output', 'trace with spaces.npz',
        '--episodes=2', '--steps', '3', '--seed', '4', '--render',
        '--output', 'movie.mp4', '--metrics-output', 'metrics.json', '--log-every', '0',
    ]) == {
        'config': Path('controller.json'), 'task_config': Path('task.json'),
        'planner_seed': 500, 'traces_output': Path('trace with spaces.npz'),
        'episodes': 2, 'steps': 3, 'seed': 4, 'render': True,
        'output': Path('movie.mp4'), 'metrics_output': Path('metrics.json'), 'log_every': 0,
    }


class EnvironmentReached(Exception):
    pass


@pytest.fixture
def no_environment(monkeypatch, tmp_path):
    calls = []
    def stop(**kwargs):
        calls.append(kwargs)
        raise EnvironmentReached
    stop.__signature__ = inspect.signature(runner.Walker2d)
    monkeypatch.setattr(runner, 'Walker2d', stop)
    monkeypatch.chdir(tmp_path)
    yield calls
    assert list(tmp_path.rglob('*')) == [], 'CLI validation created output files'


@pytest.mark.parametrize('argv, message', [
    (['--episodes', '0'], 'must be positive'),
    (['--steps', '-1'], 'must be positive'),
    (['--log-every', '-1'], 'must be nonnegative'),
    (['--seed', '-1'], 'seeds must remain'),
    (['--planner-seed', '-1'], 'seeds must remain'),
    (['--seed', str(2**32 - 1), '--episodes', '2'], 'seeds must remain'),
    (['--planner-seed', str(2**32 - 1), '--episodes', '2'], 'seeds must remain'),
    (['--metrics-output', 'same.json', '--traces-output', './same.json'], 'must be distinct'),
    (['--metrics-output', str(runner._REPO_ROOT / 'configs/walker2d_best.json')], 'must be distinct'),
    (['--traces-output', str(runner._REPO_ROOT / 'configs/walker2d_task.json')], 'must be distinct'),
    (['--render', '--output', str(runner._REPO_ROOT / 'configs/walker2d_best.json')], 'must be distinct'),
])
def test_invalid_arguments_exit_two_before_environment(argv, message, no_environment, capsys):
    with pytest.raises(SystemExit) as error:
        tyro.cli(runner.main, args=argv)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert no_environment == []


def test_help_exits_before_environment(no_environment, capsys):
    with pytest.raises(SystemExit) as error:
        tyro.cli(runner.main, args=['--help'])
    assert error.value.code == 0
    assert '--planner-seed' in capsys.readouterr().out
    assert no_environment == []


def test_maximum_seeds_and_unused_video_path_pass_validation(no_environment):
    with pytest.raises(EnvironmentReached):
        tyro.cli(runner.main, args=[
            '--episodes', '1', '--seed', str(2**32 - 1),
            '--planner-seed', str(2**32 - 1), '--log-every', '0',
            '--output', str(runner._REPO_ROOT / 'configs/walker2d_best.json'),
        ])
    assert len(no_environment) == 1


def test_runtime_value_errors_are_not_converted_to_argument_errors(monkeypatch, no_environment):
    def broken_simulator(**kwargs):
        raise ValueError('simulator initialization failed')
    monkeypatch.setattr(runner, 'Walker2d', broken_simulator)
    with pytest.raises(ValueError, match='simulator initialization failed'):
        tyro.cli(runner.main, args=[])
