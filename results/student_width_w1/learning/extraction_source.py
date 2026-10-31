"""Export saved W1 stochastic training curves; no neural or physics operations."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import tempfile
import time

ROOT = Path('/Users/jaisel/Documents/mppi-gps')
RUN = ROOT / 'runs/2026-10-31_student_width_w1_001'
OUT = ROOT / 'artifacts/2026-10-31_student_width_w1_001'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    started = time.perf_counter()
    result_bytes = (RUN / 'result.json').read_bytes()
    result = json.loads(result_bytes)
    assert result['execution_complete'] is True and result['error'] is None
    config = json.loads((RUN / 'config.json').read_text())
    manifest = json.loads((OUT / 'full_run_file_manifest.json').read_text())
    groups, sources = {}, []
    for reference in result['fits']:
        path = RUN / reference['file']
        data = path.read_bytes()
        sha256 = digest(data)
        assert sha256 == reference['sha256'] == manifest['files'][reference['file']]['sha256']
        fit = json.loads(data)
        point = result['points'][reference['point']]
        identity = (point['pair'], point['width'])
        rows = fit['learning_curve']
        assert fit['failed'] is False
        assert len(rows) == fit['completed_updates']
        assert all(math.isfinite(row[key]) for row in rows for key in ('action_mse','proximal_loss','total_loss','unclipped_gradient_norm'))
        assert all(row['action_mse'] >= 0 and row['proximal_loss'] == 0 and row['total_loss'] == row['action_mse'] for row in rows)
        assert [row['update'] for row in rows] == list(range(rows[0]['update'], rows[-1]['update'] + 1))
        source = dict(file=str(path), sha256=sha256, bytes=len(data), point=reference['point'], rows=len(rows), update_start=rows[0]['update'], update_end=rows[-1]['update'])
        sources.append(source)
        groups.setdefault(identity, []).extend((row, source) for row in rows)
    assert len(groups) == 6 and len(sources) == 18
    records = []
    for (pair, width), entries in sorted(groups.items()):
        entries.sort(key=lambda entry: entry[0]['update'])
        assert [row['update'] for row, _ in entries] == list(range(1,10001))
        for start in range(0,len(entries),100):
            chunk = entries[start:start+100]
            assert len(chunk) == 100
            rows = [row for row, _ in chunk]
            owners = {source['file'] for _, source in chunk}
            assert len(owners) == 1
            source = chunk[0][1]
            values = [row['action_mse'] for row in rows]
            records.append(dict(pair=pair,width=width,policy_seed=config['seed_pairs'][pair]['policy_seed'],fit_seed=config['seed_pairs'][pair]['fit_seed'],update_start=rows[0]['update'],update_end=rows[-1]['update'],updates=100,mean_action_mse=sum(values)/100,min_action_mse=min(values),max_action_mse=max(values),mean_unclipped_gradient_norm=sum(row['unclipped_gradient_norm'] for row in rows)/100,source_file=source['file'],source_sha256=source['sha256'],source_point=source['point']))
    assert len(records) == 600
    with (OUT / 'binned_learning_curves.csv').open('x',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='walker-w1-publication-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1,3,figsize=(14.5,4.3),constrained_layout=True)
    for pair, axis in enumerate(axes):
        for width,color in [(128,'#335c81'),(256,'#cf5c36')]:
            selected = [row for row in records if row['pair']==pair and row['width']==width]
            x = [(row['update_start']+row['update_end'])/2 for row in selected]
            axis.plot(x,[row['mean_action_mse'] for row in selected],label=f'width {width}',color=color)
            axis.fill_between(x,[row['min_action_mse'] for row in selected],[row['max_action_mse'] for row in selected],color=color,alpha=.14)
        axis.set(title=f'Paired seed {pair}',xlabel='Adam update',ylabel='Training minibatch action MSE',yscale='log')
        axis.grid(alpha=.2)
        axis.legend()
    figure.suptitle('W1: saved stochastic training loss in fixed 100-update bins\nLines: bin means; shading: minibatch min–max, not confidence intervals',fontsize=12)
    for suffix in ['png','pdf']:
        path = OUT / ('binned_learning_curves.'+suffix)
        assert not path.exists()
        figure.savefig(path,dpi=180)
    plt.close(figure)
    outputs={name:dict(bytes=(OUT/name).stat().st_size,sha256=digest((OUT/name).read_bytes())) for name in ['binned_learning_curves.csv','binned_learning_curves.png','binned_learning_curves.pdf']}
    receipt=dict(schema='walker-w1-saved-learning-bin-extraction-v1',run=str(RUN),result_sha256=digest(result_bytes),source_sha256=digest(Path(__file__).read_bytes()),source_rows=60000,source_fit_files=18,bins=600,updates_per_bin=100,binning='Within each of six pair/width series, fixed inclusive update intervals1..100,101..200,...,9901..10000. Arithmetic mean/min/max action MSE and mean unclipped gradient norm. No checkpoint selection or smoothing across source files.',scope='Source-bound rendering/aggregation of already recorded stochastic minibatch training losses. Not full-dataset train/validation losses, uncertainty intervals, new inference, optimizer arithmetic, or a research rerun.',sources=sources,outputs=outputs,new_controls=0,new_forward_rows=0,new_backward_rows=0,new_optimizer_steps=0,wall_seconds=time.perf_counter()-started)
    with (OUT/'binned_learning_curves_receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({key:value for key,value in receipt.items() if key not in ['sources']},indent=2))


if __name__ == '__main__':
    main()
