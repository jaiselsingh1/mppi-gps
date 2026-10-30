"""One finite discovery intervention; imports never execute physics or fitting."""
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
import time

from .search import SearchConfig
from .task import TaskConfig


def protocol(config):
    base=SearchConfig(**config['search'])
    if base!=SearchConfig(H=144) or TaskConfig.from_dict(config['task'])!=TaskConfig():
        raise ValueError('This assay retains the original H144 search and full Acrobot task')
    if config['backend']!='warp':
        raise ValueError('This finite production assay uses the qualified Warp runtime')
    per_episode=(512+32+16+1)*sum(min(144,r) for r in range(1,801))+800
    return dict(schema='independent-temporal-proposals-dev001',
        reset_seeds=[110100,110101],planner_seeds=[30000,30001],
        conditions={'short_knots':asdict(base),'independent_blocks':asdict(replace(base,
            independent_block_lengths=(20,40,80,144)))},
        execution_order=[[name,i] for i in range(2) for name in ('short_knots','independent_blocks')],
        max_controls_per_episode=per_episode,max_controls_per_condition=2*per_episode,
        max_controls=4*per_episode,actor_rows=0,optimizer_steps=0,
        memory='cold independent discovery; no cross-root or cross-condition memory',
        intervention='Only 32 broad proposal trajectories change; identical random draw consumption and unchanged other channels at the same private planner state',
        development_lead='2/2 actual treatment captures; retain every control outcome and compare costs/local approach',
        stopping_rule='Execute exactly four complete episodes unless an execution or accounting error occurs. No replacement, extra search, reward change, or within-run tuning.',
        research_rule='A failed run closes this formulation only; a new diagnosed version may follow.',
        interpretation='A discovery assay, not horizon-gap qualification, actor learning, or an efficiency result')


def check_backend(config, report_path):
    """Require the already qualified production shapes, source and runtime."""
    import torch
    from . import backend
    from .evidence import file_sha256
    path=Path(report_path);e=json.loads(path.read_text());m=json.loads((path.parent/'manifest.json').read_text())
    if (not e.get('qualified') or e.get('task_hash')!=TaskConfig().task_hash
            or e.get('source_sha256')!=file_sha256(backend.__file__)):
        raise ValueError('Backend report does not qualify these unchanged physics/task sources')
    for package in ('mujoco','mujoco-warp','warp-lang','numpy'):
        if m['dependency_versions'].get(package)!=importlib.metadata.version(package):
            raise ValueError('Backend dependency changed: '+package)
    if m.get('cuda_device')!=torch.cuda.get_device_name():
        raise ValueError('Backend GPU model changed')
    # The report explicitly tests K560/K1, H144/H288. This assay uses the
    # same capacities and shorter cutoffs; no unqualified larger batch is used.
    production=e.get('production_shapes',{})
    checked={(r['worlds'],r['horizon']) for r in production.get('checks',[]) if r.get('passed')}
    if (e.get('schema')!='bidirectional-backend-qualification-v3' or not production.get('qualified')
            or not {(1,144),(560,144)}.issubset(checked) or e.get('unknown_controls')!=0):
        raise ValueError('Backend report lacks the passed production capacities used here')


def run(config,out):
    from .evidence import Archive,QueryExecutor,file_sha256
    from .task import AcrobotCaptureTask
    from .workflow import write_json,teacher_panel,summary
    out=Path(out);spec=protocol(config);started=time.perf_counter()
    manifest=json.loads((out/'manifest.json').read_text())
    if not Path(__file__).resolve().is_relative_to(out/'source') or any(
            file_sha256(out/r['file'])!=r['sha256'] for r in manifest['files']):
        raise ValueError('Execute this assay only from its immutable source packet')
    write_json(out/'protocol.json',spec)
    archives={};task=None;outcomes={name:[] for name in spec['conditions']}
    try:
        task=AcrobotCaptureTask(TaskConfig.from_dict(config['task']),backend=config['backend'])
        executors={}
        for name in spec['conditions']:
            archive=Archive(out/'evidence'/name);archives[name]=archive
            executors[name]=QueryExecutor(task,archive,scope=name,
                max_requested_controls=spec['max_controls_per_condition'])
        for name,i in spec['execution_order']:
            rows,_=teacher_panel(executors[name],SearchConfig(**spec['conditions'][name]),
                [spec['reset_seeds'][i]],f'{name}_root{i}',planner_seed=spec['planner_seeds'][i])
            outcomes[name].extend(rows)
            write_json(out/f'{name}_root{i}.json',rows[0])
        unchanged=all(file_sha256(out/r['file'])==r['sha256'] for r in manifest['files'])
        if not unchanged:raise RuntimeError('Source changed during this assay')
    except BaseException as error:
        write_json(out/'execution_failure.json',dict(error_type=type(error).__name__,error=str(error),
            completed_outcomes=outcomes,wall_seconds=time.perf_counter()-started))
        raise
    finally:
        for archive in archives.values():archive.close()
        if task is not None:task.close()
    counts={name:archive.totals() for name,archive in archives.items()}
    if sum(r['requested_controls'] for r in counts.values())!=spec['max_controls']:
        raise RuntimeError('Complete finite assay work differs from its declared schedule')
    result=dict(schema=spec['schema'],status='complete',conditions={name:dict(**summary(rows),outcomes=rows)
        for name,rows in outcomes.items()},accounting=counts,
        development_lead=sum(r['capture'] for r in outcomes['independent_blocks'])==2,
        original_task_qualified=False,standalone_learning_tested=False,efficiency_claim=False,
        actor_rows=0,optimizer_steps=0,source_unchanged=unchanged,wall_seconds=time.perf_counter()-started)
    write_json(out/'assay_result.json',result)
    return result
