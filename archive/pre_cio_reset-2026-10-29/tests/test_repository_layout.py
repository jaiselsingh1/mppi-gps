"""Active MuJoCo boundary and preservation contracts; no substitute dynamics."""
import ast
import hashlib
import json
from pathlib import Path
from scripts.training.train_bidirectional_gps import ROOT, ARCHIVE, research_status, verify_archive

def test_status_does_not_promote_historical_or_unqualified_policies(tmp_path):
    config = tmp_path / 'configs'
    config.mkdir()
    (config / 'bidirectional_mujoco.json').write_text(json.dumps(dict(schema='physical')))
    for name, schema in (('physical', 'physical'), ('historical', 'ending')):
        run = tmp_path / 'runs' / name
        run.mkdir(parents=True)
        (run / 'config.json').write_text(json.dumps(dict(schema=schema)))
        (run / 'result.json').write_text(json.dumps(dict(stage='qualify',
            software_pass=True, qualification_pass=True, actor_training=False)))
    status = research_status(tmp_path)
    assert status['allowed_engine'] == 'MuJoCo'
    assert status['progression'] == ['point_mass', 'acrobot', 'walker2d']
    assert not status['policy_promoted'] and not status['beneficial_bidirectional_loop']
    assert Path(status['saved_evidence']['qualify']['run']).name == 'physical'
    assert 'Not inferred' in status['process_liveness']

def test_archive_verifier_detects_changed_or_missing_bytes(tmp_path):
    metadata = tmp_path / 'metadata'
    metadata.mkdir()
    source = tmp_path / 'source.py'
    source.write_text('preserved\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (metadata / 'snapshot.json').write_text(json.dumps(dict(
        files=[dict(file='source.py', sha256=digest)])))
    assert verify_archive(tmp_path)['passed']
    source.write_text('changed\n')
    assert verify_archive(tmp_path)['failures'] == ['source.py']
    source.unlink()
    assert verify_archive(tmp_path)['failures'] == ['source.py']

def test_active_local_imports_resolve_without_retired_workflow():
    missing = []
    for folder in ('src', 'scripts'):
        for path in (ROOT / folder).rglob('*.py'):
            package = list(path.relative_to(ROOT).with_suffix('').parts[:-1])
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        base = package[:len(package)-node.level+1]
                        modules = ['.'.join(base + (node.module or '').split('.')).rstrip('.')]
                    else:
                        modules = [node.module or '']
                for name in modules:
                    if name.split('.')[0] not in {'src', 'scripts'}:
                        continue
                    target = ROOT / name.replace('.', '/')
                    if not target.with_suffix('.py').is_file() and not (target / '__init__.py').is_file():
                        missing.append((str(path.relative_to(ROOT)), node.lineno, name))
    assert missing == []

def test_active_concrete_environment_classes_use_mujoco():
    for path in (ROOT / 'src/envs').glob('*.py'):
        if path.name in {'base.py', 'mujoco_env.py', '__init__.py'}:
            continue
        classes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)]
        assert classes, path
        environments = 0
        for node in classes:
            if any(ast.unparse(base) in {'NamedTuple', 'typing.NamedTuple'} for base in node.bases):
                continue  # Cost-component records do not implement dynamics.
            assert any(isinstance(base, ast.Name) and base.id == 'MuJoCoEnv' for base in node.bases), path
            environments += 1
        assert environments > 0, path
    assert not (ROOT / 'src/gps/bidirectional/ending.py').exists()
    assert not (ROOT / 'configs/bidirectional_ending.json').exists()

def test_retired_bytes_and_original_evidence_are_preserved():
    assert verify_archive()['passed']
    assert (ARCHIVE / 'src/gps/bidirectional/ending.py').is_file()
    for name in ('runs', 'artifacts'):
        assert (ARCHIVE / name).is_symlink()
        assert (ARCHIVE / name).resolve() == ROOT / name
    import tomllib
    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert config['tool']['pytest']['ini_options']['testpaths'] == ['tests']
