"""PDF figures and behavior-level summaries for both Section 5 experiments."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prefix.runner import tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['R', 'C', 'delta_C', 'delta_perp']


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['id']].append(row)
    summaries = []
    for condition, group in sorted(groups.items()):
        row = {key: group[0][key] for key in ['id','method','m','alpha','k']}
        by_input = defaultdict(list)
        for item in group:
            by_input[item['index']].append(item)
        row['n'] = len(by_input)
        for field in FIELDS:
            samples = [[v[field] for v in sample if v[field] is not None] for sample in by_input.values()]
            values = np.array([np.mean(sample) for sample in samples if sample])
            n = len(values)
            if n:
                rng = np.random.default_rng(42)
                selection = rng.integers(0, n, size=(2000, n))
                interval = np.quantile(values[selection].mean(axis=1), [.025,.975])
                row[field] = dict(mean=float(values.mean()), lower=float(interval[0]), upper=float(interval[1]),
                                  n=n, undefined=row['n']-n)
            else:
                row[field] = dict(mean=None, lower=None, upper=None, n=0, undefined=row['n'])
        row['continued_after_eos'] = sum(any(v.get('continued_after_eos', False) for v in sample)
                                         for sample in by_input.values())
        summaries.append(row)
    return summaries


def load(study, limit, allow_partial):
    rows, completed, expected_units, metadata = [], [], None, None
    for index in range(limit):
        path = ROOT / f'results/section5_{study}_{index:03d}.json'
        if not path.exists():
            if allow_partial:
                continue
            raise FileNotFoundError(path)
        state = json.loads(path.read_text())
        manifest = state['manifest']
        shared = {k:v for k,v in manifest.items() if k not in ('index','activation_sha256')}
        if metadata is not None and metadata != shared:
            raise ValueError(f'Incompatible manifests: {path}')
        metadata = shared
        conditions = {c['id'] for c in manifest['conditions']}
        expected_units = (conditions if study == 'generation' else
                          {f'{layer}_{head}' for layer in manifest['layers'] for head in manifest['heads']})
        complete = set(state['units']) == expected_units
        if not complete and not allow_partial:
            raise ValueError(f'Incomplete study: {path}')
        if complete:
            completed.append(index)
        if study == 'fixed':
            # Main cross-study comparison uses the prespecified native layer/head.
            unit = state['units'].get('17_0')
            if unit is None:
                continue
            if {r['id'] for r in unit['rows']} != conditions:
                raise ValueError(f'Incomplete fixed-head conditions: {path}')
            for r in unit['rows']:
                if r['eligible']:
                    rows.append({**r, 'index': index, 'base_length': unit['base_length']})
        else:
            rows.extend({**r, 'index': index} for r in state['units'].values() if r['eligible'])
    return rows, dict(completed_examples=len(completed), requested_examples=limit,
                      complete=len(completed)==limit, manifest=metadata)


def figure(groups, length_groups, study, complete):
    by_id = {r['id']: r for r in groups}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for row, field in enumerate(['R','C']):
        for method, label in [('single','Single token'),('full','Full steering')]:
            series = sorted([g for g in groups if g['method']==method and g['m']==8], key=lambda g:g['alpha'])
            if series:
                axes[row,0].plot([g['alpha'] for g in series], [g[field]['mean'] for g in series], 'o-', label=label)
                axes[row,0].fill_between([g['alpha'] for g in series], [g[field]['lower'] for g in series],
                                          [g[field]['upper'] for g in series], alpha=.15)
        for method, label in [('single','Single token'),('full','Full steering'),('prompt','Prompt')]:
            series = sorted([g for g in groups if g['method']==method and (method=='prompt' or g['alpha']==1)], key=lambda g:g['m'])
            if series:
                axes[row,1].plot([g['m'] for g in series], [g[field]['mean'] for g in series], 'o-', label=label)
        for column in range(3):
            if 'unsteered' in by_id:
                axes[row,column].axhline(by_id['unsteered'][field]['mean'], color='black', linestyle=':', label='Unsteered')
            axes[row,column].set_ylabel(field)
            axes[row,column].grid(alpha=.2)
        if 'prompt_m8' in by_id:
            axes[row,0].axhline(by_id['prompt_m8'][field]['mean'], color='tab:green', linestyle='--', label='Prompt (m=8)')
        axes[row,0].set_xlabel('Strength α (m=8)')
        axes[row,1].set_xscale('log', base=2)
        axes[row,1].set_xticks([1,8,32,128], ['1','8','32','128'])
        axes[row,1].set_xlabel('Prompt length m (α=1)')
        length_series = sorted([g for g in length_groups if g['alpha']==1],
                               key=lambda g: g['k'] if g['k']>0 else 100000)
        if length_series:
            x = list(range(len(length_series)))
            axes[row,2].plot(x, [g[field]['mean'] for g in length_series], 'o-')
            axes[row,2].fill_between(x, [g[field]['lower'] for g in length_series],
                                     [g[field]['upper'] for g in length_series], alpha=.15)
            axes[row,2].set_xticks(x, [str(g['k']) if g['k']>0 else 'Full' for g in length_series])
            axes[row,2].set_xlabel('Steered input tokens (m=8, α=1)')
            axes[row,2].set_title(f'Matched cohort: n={length_series[0]["n"]}')
    axes[0,0].legend(fontsize=9)
    axes[0,1].legend(fontsize=9)
    fig.suptitle(f'{study.capitalize()} · layer 17 / head 0 · prediction step 128' + ('' if complete else ' · PARTIAL'))
    path = ROOT / f'figs/section5_{study}_rc.pdf'
    fig.savefig(path)
    plt.close(fig)
    print(f'Saved {path.name}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=400)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    with tee_stdout(ROOT / 'logs/section5_summary.log'):
        summary = {}
        report = ['# Section 5: concept alignment and preservation', '',
                  'R = cos(output, unsteered output); C = cos(output, fixed reference direction).',
                  'The direction is the m=8 first-step constructed value displacement, fixed across methods, prompt lengths and strengths within each example/head. It measures prompt alignment, not a separately validated safety concept.', '',
                  'Prediction step 128 uses the query at generated token 127. Fixed replay excludes token 128 and all unused prompt positions. The linear replay precedes QK normalization and RoPE and shares the original eight-token-prompt continuation. Native generation uses each method\'s own trajectory with the model\'s native operations.', '',
                  'Full steering applies the same last-input-token-constructed displacement to all input and visible generated positions. Prefix-length sweeps apply that same displacement to the last k input positions. Native runs intervene after input normalization at layer 17; head 0 is the prespecified primary measurement.', '',
                  'Numerically zero concept directions retain R and mark C and concept-dependent quantities undefined. Per-metric sample counts and undefined counts are stored explicitly.', '',
                  'Intervals are percentile 95% bootstrap intervals over behaviors (2,000 resamples, seed 42). Head observations are never counted as independent behaviors. EOS is ignored to reach 128 tokens, and continuation after EOS is counted explicitly.', '']
        for study in ['fixed','generation']:
            rows, status = load(study, args.limit, args.allow_partial)
            if not rows:
                summary[study] = status
                report += [f'## {study}: no completed measurements', '']
                continue
            groups = summarize(rows)
            # Use the same eligible behavior cohort throughout the input-length sweep.
            cohort = {r['index'] for r in rows if r.get('base_length',0) >= 64}
            length_groups = summarize([r for r in rows if r['index'] in cohort and r['m']==8
                                        and r['method'] in ('single','prefix','full')])
            summary[study] = {**status, 'groups': groups, 'input_length_matched_cohort': length_groups}
            report += [f'## {study}: {status["completed_examples"]}/{args.limit} complete examples', '',
                       '| Method (m=8, α=1) | n | R | C | ΔC | Continued after EOS |',
                       '|---|---:|---:|---:|---:|---:|']
            for g in groups:
                if g['id'] in ['unsteered','prompt_m8','single_m8_a1','full_m8_a1']:
                    report.append(f'| {g["method"]} | {g["n"]} | {g["R"]["mean"]:.5f} | {g["C"]["mean"]:.5f} | {g["delta_C"]["mean"]:.5f} | {g["continued_after_eos"]} |')
            report.append('')
            figure(groups, length_groups, study, status['complete'])
        write_json_atomic(ROOT / 'results/section5_summary.json', summary)
        (ROOT / 'results/section5_report.md').write_text('\n'.join(report) + '\n')
        print('Saved Section 5 summary and report', flush=True)


if __name__ == '__main__':
    main()
