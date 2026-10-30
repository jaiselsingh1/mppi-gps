"""Saved-data figures and accounting for L1 and H4.1; no inference or physics."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.diagnostics.guided_data_comparison import sha, write


def load_run(run):
    result = json.loads((run / 'result.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    if not result['passed'] or result['error']:
        raise ValueError('A closed valid experiment is required; software errors are not scientific negatives')
    return result, config


def outcome_panel(ax, scores, palette, title):
    names = list(scores)
    maximum = max(max(score['costs']) for score in scores.values())
    label_offset = max(10., .02 * maximum)
    for i, name in enumerate(names):
        score = scores[name]
        ax.bar(i, score['mean_cost'], color=palette[name], alpha=.65)
        x = i + np.linspace(-.16, .16, score['episodes'])
        ax.scatter(x, score['costs'], color=palette[name], s=22, edgecolors='white', zorder=3)
        ax.text(i, max(score['costs']) + label_offset,
                f"{score['successes']}/{score['episodes']} walk; {score['survived']}/{score['episodes']} survive\n"
                f"{score['mean_steps']:.1f} controls", ha='center', fontsize=8)
    ax.set(xticks=range(len(names)), xticklabels=[name.replace('_', ' ') for name in names],
           ylabel='Complete 1,000-control cost', title=title)
    ax.set_ylim(0, maximum + max(60., .20 * maximum))


def plot_student(fig, axs, result, cfg, palette):
    checkpoints = cfg['checkpoints']
    for arm in ('clipped', 'full'):
        values = [result['full_mse'][arm + '_initial']] + [result['full_mse'][f'{arm}_{k}'] for k in checkpoints]
        axs[0, 0].plot([0] + checkpoints, values, 'o-', color=palette[arm], label=arm + ' inputs')
        points = [result['standalone_curves'][f'{arm}_{k}']['mean_steps'] for k in checkpoints]
        axs[0, 1].plot(checkpoints, points, 'o-', color=palette[arm], label=arm + ' inputs')
    axs[0, 0].set(title='Fresh learning: own full-data MSE', xlabel='Adam updates', ylabel='Action MSE')
    axs[0, 0].legend(frameon=False)
    parent = result['standalone_curves']['parent']['mean_steps']
    axs[0, 1].axhline(parent, linestyle=':', color=palette['parent'], label='Historical Bfree reference')
    axs[0, 1].axhline(1000, linestyle='--', color='#64748b', linewidth=1)
    axs[0, 1].set(title='Standalone curves: four paired starts', xlabel='Adam updates',
                  ylabel='Mean controls / 1,000', ylim=(0, 1050))
    axs[0, 1].legend(frameon=False, fontsize=8)
    primary = result['cases']['primary']
    outcome_panel(axs[1, 0], primary, palette, 'Fresh primary outcomes: 16 paired starts')
    names = list(primary); x = np.arange(len(names)); width = .32
    axs[1, 1].bar(x - width / 2, [primary[k]['successes'] for k in names], width, label='Walking success', color='#007c91')
    axs[1, 1].bar(x + width / 2, [primary[k]['survived'] for k in names], width, label='Full survival', color='#96adbb')
    axs[1, 1].set(title='Walking requires survival and velocity tracking', xticks=x,
                  xticklabels=names, ylim=(0, 17), ylabel='Episodes / 16')
    axs[1, 1].legend(frameon=False)
    fig.suptitle('Walker L1 — fresh clipped versus full-velocity training', fontsize=15, fontweight='bold')


def plot_coupling(fig, axs, result, cfg, palette):
    teachers = result['teacher_summaries']
    outcome_panel(axs[0, 0], teachers, palette, 'Actual updated-policy teachers: two paired starts')
    names = list(teachers)
    axs[0, 1].bar(range(len(names)), [cfg['weights'][k] for k in names], color=[palette[k] for k in names])
    axs[0, 1].set(xticks=range(len(names)), xticklabels=[k.replace('_', ' ') for k in names],
                  ylabel='Agreement weight', title='Same incoming updated actor; declared weight intervention')
    primary = result.get('cases', {}).get('primary', {})
    if primary:
        outcome_panel(axs[1, 0], primary, palette, 'Fresh standalone primary outcomes')
    else:
        axs[1, 0].axis('off')
        axs[1, 0].text(.05, .65, 'No new standalone policy was fitted', fontsize=13)
    if result.get('learning_curves'):
        for arm in ('independent', 'fixed_high', 'backoff'):
            x = [0] + cfg['fit_checkpoints']
            y = [result['full_mse']['parent_on_' + arm]] + [result['full_mse'][f'{arm}_{k}'] for k in cfg['fit_checkpoints']]
            axs[1, 1].plot(x, y, 'o-', label=arm.replace('_', ' '), color=palette[arm])
        axs[1, 1].set(title='Matched fixed-update child fits', xlabel='Additional Adam updates', ylabel='Own-data MSE')
        axs[1, 1].legend(frameon=False)
    else:
        axs[1, 1].axis('off')
        failed = [key for key, value in result['teacher_admission']['checks'].items() if not value]
        axs[1, 1].text(.02, .85, 'Teacher constraint failed; no child fitting', fontsize=12, va='top')
        axs[1, 1].text(.02, .65, '\n'.join(failed), fontsize=10, va='top')
        axs[1, 1].text(.02, .20, 'All executed attempts are retained.\nNo failed-slot substitution or weight retry.', fontsize=10)
    fig.suptitle('Walker H4.1 — feedback-selected backoff versus fixed high and zero', fontsize=14, fontweight='bold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('student', 'coupling'), required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter(); run = args.run.resolve(); out = args.out.resolve()
    result, cfg = load_run(run)
    out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='walker-experiment-plot-'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    palette = dict(parent='#64748b', clipped='#007c91', full='#c46b2a',
                   independent='#007c91', fixed_high='#7858a6', backoff='#c46b2a')
    if args.kind == 'student':
        plot_student(fig, axs, result, cfg, palette)
    else:
        plot_coupling(fig, axs, result, cfg, palette)
    fig.savefig(out / 'learning_and_outcomes.png', dpi=180)
    fig.savefig(out / 'learning_and_outcomes.pdf')
    plt.close(fig)
    write(out / 'plotted_data.json', dict(kind=args.kind, run=str(run), result_sha256=sha(run / 'result.json'),
        primary=result.get('cases', {}).get('primary'), teacher_summaries=result.get('teacher_summaries'),
        standalone_curves=result.get('standalone_curves'), full_mse=result.get('full_mse'),
        paired_gap=result.get('paired_teacher_student_gap'), weights=cfg.get('weights')))
    with (out / 'standalone_episodes.csv').open('x', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['cohort', 'arm', 'tag', 'complete_cost', 'executed_controls'])
        for cohort, scores in result.get('cases', {}).items():
            for arm, score in scores.items():
                for tag, cost, steps in zip(score['tags'], score['costs'], score['steps']):
                    writer.writerow([cohort, arm, tag, cost, steps])
    write(out / 'cost_accounting.json', dict(schema='walker-experiment-stage-accounting-v1',
        run=str(run), result_sha256=sha(run / 'result.json'), work=result['work'], caps=cfg['budgets'],
        internal_seconds=result['wall_seconds'], launch_receipt=json.loads((run / 'launch_receipt.json').read_text())
            if (run / 'launch_receipt.json').exists() else None,
        inherited_work_charged_here=False,
        scope='All new experimental work from stage ledger, inherited work separate. Plotting uses saved arrays; tests, audits, preparation and transfer are separate overhead.'))
    names = ['learning_and_outcomes.png', 'learning_and_outcomes.pdf', 'plotted_data.json',
             'standalone_episodes.csv', 'cost_accounting.json']
    write(out / 'receipt.json', dict(schema='walker-saved-experiment-report-v1',
        source_sha256=sha(Path(__file__)), outputs={name: sha(out / name) for name in names},
        result_sha256=sha(run / 'result.json'), controls=0, neural_rows=0, backward_rows=0,
        optimizer_steps=0, wall_seconds=time.perf_counter() - started))
    print(json.dumps(dict(report=str(out), kind=args.kind, status=result['status'])))


if __name__ == '__main__':
    main()
