"""Import-boundary and recovery contracts; no simulator or actor execution."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.training.train_bidirectional_gps import ROOT, ARCHIVE, legacy, verify_archive, research_status


def _module_exists(name):
    path = ROOT/name.replace('.','/')
    return path.with_suffix('.py').is_file() or (path/'__init__.py').is_file()


def test_saved_status_does_not_infer_process_liveness_or_child_promotion(tmp_path):
    config=tmp_path/'configs';config.mkdir()
    (config/'bidirectional_ending.json').write_text(json.dumps(dict(schema='current')))
    run=tmp_path/'runs/full';run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(dict(schema='current')))
    (run/'study.json').write_text(json.dumps(dict(stop_reason='no_qualified_changed_child',rounds=[],
        stages=[dict(stage=0,accepted_updates=dict(raw=9,guided=12),selected_next_parent='parent',
            qualification_history=[dict(promoted=True)])])))
    value=research_status(tmp_path);full=value['saved_evidence']['full_experiment']
    assert full['state']=='producer_complete_final_result_missing'
    assert full['completed_feedback_rounds']==0
    assert full['stages'][0]['selected_next_parent']=='parent'
    assert value['process_liveness']=='Not inferred from saved files; check the launch process separately'


def test_saved_status_keeps_negative_science_separate_from_passed_audit(tmp_path):
    config=tmp_path/'configs';config.mkdir()
    (config/'bidirectional_ending.json').write_text(json.dumps(dict(schema='current')))
    for name,schema in (('current','current'),('later_other_protocol','other')):
        run=tmp_path/'runs'/name;run.mkdir(parents=True)
        (run/'config.json').write_text(json.dumps(dict(schema=schema)))
        (run/'result.json').write_text(json.dumps(dict(audit_pass=True,interface_pass=True,
            beneficial_bidirectional_loop=schema!='current',producer_failed=False)))
        (run/'launch.json').write_text(json.dumps(dict(returncode=1)))
    full=research_status(tmp_path)['saved_evidence']['full_experiment']
    assert Path(full['run']).name=='current'
    assert full['state']=='closed' and full['audit_pass'] and full['interface_pass']
    assert not full['beneficial_bidirectional_loop'] and not full['producer_failed']


def test_saved_status_tolerates_a_result_record_being_written(tmp_path):
    config=tmp_path/'configs';config.mkdir()
    (config/'bidirectional_ending.json').write_text(json.dumps(dict(schema='current')))
    run=tmp_path/'runs/current';run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(dict(schema='current')))
    (run/'result.json').write_text('{')
    full=research_status(tmp_path)['saved_evidence']['full_experiment']
    assert full['state']=='final_result_record_incomplete'
    assert 'producer_failed' not in full and 'beneficial_bidirectional_loop' not in full


def test_active_local_imports_resolve_without_historical_checkout():
    missing = []
    for folder in ('src','scripts'):
        for path in (ROOT/folder).rglob('*.py'):
            package = list(path.relative_to(ROOT).with_suffix('').parts[:-1])
            for node in ast.walk(ast.parse(path.read_text())):
                modules = []
                if isinstance(node,ast.Import):
                    modules = [item.name for item in node.names]
                elif isinstance(node,ast.ImportFrom):
                    if node.level:
                        base = package[:len(package)-node.level+1]
                        modules = ['.'.join(base+node.module.split('.')) if node.module else '.'.join(base)]
                    else:
                        modules = [node.module or '']
                for module in modules:
                    if module.split('.')[0] in {'src','scripts'} and not _module_exists(module):
                        missing.append((str(path.relative_to(ROOT)),node.lineno,module))
    assert missing == []


def test_archive_verifier_detects_corruption_and_missing_files(tmp_path):
    metadata = tmp_path/'metadata'; metadata.mkdir()
    source = tmp_path/'code.py'; source.write_text('original\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (metadata/'worktree.json').write_text(json.dumps(dict(files=[dict(file='code.py',sha256=digest)])))
    assert verify_archive(tmp_path)['passed']
    source.write_text('changed\n')
    assert verify_archive(tmp_path)['failures'] == ['code.py']
    source.unlink()
    assert verify_archive(tmp_path)['failures'] == ['code.py']


def test_legacy_launch_uses_fresh_process_and_archived_import_root(monkeypatch):
    calls = []
    def run(args,**kwargs):
        calls.append((args,kwargs))
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr('scripts.training.train_bidirectional_gps.subprocess.run',run)
    assert legacy('scripts.training.train_bidirectional_gps',['--help']) == 7
    args, options = calls[0]
    assert args[1:] == ['-m','scripts.training.train_bidirectional_gps','--help']
    assert options['cwd'] == ARCHIVE
    assert options['env']['PYTHONPATH'] == str(ARCHIVE)
    assert options['env']['PYTHONDONTWRITEBYTECODE'] == '1'


@pytest.mark.parametrize('module',['../src/gps','src.gps.mdgps','scripts.missing_module'])
def test_legacy_rejects_wrong_namespace_or_missing_module(module):
    with pytest.raises(ValueError):
        legacy(module,[])


def test_default_test_collection_excludes_archive_and_raw_runs():
    import tomllib
    config = tomllib.loads((ROOT/'pyproject.toml').read_text())
    assert config['tool']['pytest']['ini_options']['testpaths'] == ['tests']
    assert not (ROOT/'src/gps/mdgps.py').exists()
    assert not (ROOT/'src/gps/bidirectional/deployment_loop.py').exists()
    assert (ARCHIVE/'src/gps/mdgps.py').is_file()
    assert (ARCHIVE/'src/gps/bidirectional/deployment_loop.py').is_file()


def test_archive_evidence_links_point_to_original_roots():
    for name in ('runs','artifacts'):
        assert (ARCHIVE/name).is_symlink()
        assert (ARCHIVE/name).resolve() == ROOT/name
