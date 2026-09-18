"""Two requested plots: steering length at fixed m, prompt length at fixed k.

Average six head measurements within each HarmBench input first, then report
the mean and sample standard deviation across inputs. Notifications are disabled.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prefix.runner import tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def mean_std(rows, field):
    by_prompt = {}
    for row in rows:
        by_prompt.setdefault(row['index'], []).append(row[field])
    values = np.array([np.mean(by_prompt[index]) for index in sorted(by_prompt)])
    return {'n': len(values), 'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def summarize(rows):
    def vector(key):
        return np.array([r[key] for r in rows])
    error, bound, d = vector('error'), vector('bound_certified'), vector('diameter')
    tol = 2e-8 * np.maximum(1., d)
    mask = (error > tol) & (bound > tol)
    ratio = error[mask] / bound[mask]
    return {
        'rows': len(rows), 'behaviors': len({r['index'] for r in rows}),
        'violations': int(np.sum(error > bound + tol)),
        'nonzero_error_rows': int(np.sum(error > tol)),
        'median_error_over_bound': float(np.median(ratio)) if len(ratio) else None,
        'p10_error_over_bound': float(np.quantile(ratio, .1)) if len(ratio) else None,
        'p90_error_over_bound': float(np.quantile(ratio, .9)) if len(ratio) else None,
        'diameter_cap_fraction': float(np.mean(np.abs(bound - d) <= tol)),
        'max_identity_residual': float(vector('identity_residual').max()),
        'value_dominates_fraction': float(np.mean(vector('value_error')[error > tol]
                                                  > vector('weight_error')[error > tol])) if np.any(error > tol) else None,
    }


def load_rows(limit, suite='main'):
    rows = []
    prefix = 'section4' if suite == 'main' else 'section4_redundancy'
    for index in range(limit):
        path = ROOT / f'results/{prefix}_{index:03d}.json'
        state = json.loads(path.read_text())
        if state['manifest'].get('jacobian_anchor') != 'last_input_token':
            raise ValueError(f'Jacobian anchor audit not complete: {path}')
        expected = {f'{layer}_{head}/{c["id"]}' for layer in state['manifest']['layers']
                    for head in state['manifest']['heads'] for c in state['manifest']['conditions']}
        if set(state['units']) != expected:
            raise ValueError(f'Incomplete conditions: {path}')
        if any(row['index'] != index for row in state['units'].values()):
            raise ValueError(f'Misassigned behavior: {path}')
        rows.extend(state['units'].values())
    return rows


def statistics(rows):
    output = []
    for m in [1, 2, 4, 8]:
        for k in [1, 2, 4, 8]:
            group = [r for r in rows if r['family'] == 'grid' and r['m'] == m and r['k'] == k]
            output.append({'setting': 'input_only', 'm': m, 'k': k, 'g': 0,
                           'error': mean_std(group, 'error'),
                           'bound': mean_std(group, 'bound_certified')})
    for g in [0, 1, 4, 8, 16]:
        group = [r for r in rows if r['family'] == 'generated_trajectory' and r['g'] == g]
        output.append({'setting': 'input_and_generated', 'm': 4, 'k': 4, 'g': g,
                       'error': mean_std(group, 'error'),
                       'bound': mean_std(group, 'bound_certified')})
    return output


def plot_line(ax, entries, field, x, label, color):
    xs = [x(row) for row in entries]
    means = np.array([row[field]['mean'] for row in entries])
    stds = np.array([row[field]['std'] for row in entries])
    ax.plot(xs, means, '-o', color=color, label=label, linewidth=1.8, markersize=4)
    ax.fill_between(xs, means - stds, means + stds, color=color, alpha=.15)


def plots(stats):
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    n = stats[0]['error']['n']
    fixed_m = [r for r in stats if r['setting'] == 'input_only' and r['m'] == 4]
    extended = [r for r in stats if r['setting'] == 'input_and_generated']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, field, title in zip(axes, ['error', 'bound'], ['Measured error', 'Error upper bound']):
        plot_line(ax, fixed_m, field, lambda r: r['k'], 'Input positions only', '#0072B2')
        plot_line(ax, extended, field, lambda r: r['k'] + r['g'], '4 input + generated positions', '#D55E00')
        ax.set_xlabel('Number of steered positions')
        ax.set_ylabel(r'$\|o_{\mathrm{steer}}-o_{\mathrm{prompt}}\|_2$' if field == 'error' else 'Certified lemma upper bound')
        ax.set_xticks([1, 4, 8, 12, 16, 20])
        ax.axhline(0, color='gray', linewidth=.6)
        ax.grid(alpha=.18); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle(f'Fixed prompt length m = 4 · Qwen3-4B / {n} HarmBench prompts\nMean ± std across prompts; first point is single-token steering', fontsize=12)
    fig.savefig(ROOT / 'figs/section4_steering_length.pdf', bbox_inches='tight')
    plt.close(fig)

    single = [r for r in stats if r['setting'] == 'input_only' and r['k'] == 1]
    fig, ax = plt.subplots(figsize=(5.7, 4), constrained_layout=True)
    plot_line(ax, single, 'error', lambda r: r['m'], 'Single-token steering', '#0072B2')
    ax.set_xlabel('Number of prompt tokens m')
    ax.set_ylabel(r'$\|o_{\mathrm{steer}}-o_{\mathrm{prompt}}\|_2$')
    ax.set_xticks([1, 2, 4, 8]); ax.axhline(0, color='gray', linewidth=.6); ax.grid(alpha=.18)
    ax.set_title(f'Fixed single-token steering (k = 1)\nQwen3-4B / {n} prompts · mean ± std', fontsize=12)
    fig.savefig(ROOT / 'figs/section4_prompt_length.pdf', bbox_inches='tight')
    plt.close(fig)
    print('Saved the two requested PDF figures.', flush=True)


def audit(rows, extra):
    summary = {'overall': summarize(rows), 'redundancy': summarize(extra),
               'by_lemma': {str(lemma): summarize([r for r in rows if r['lemma'] == lemma]) for lemma in [2,4,6,7]}}
    summary['chain_violations'] = 0
    for row in rows + extra:
        values = [row[k] for k in ['error', 'bound_decomp', 'bound_log', 'bound_certified', 'diameter']]
        summary['chain_violations'] += any(a > b + 2e-8 * max(1.,row['diameter']) for a,b in zip(values, values[1:]))
    summary['rows_above_sampled_lower_endpoint'] = sum(r['error'] > r['bound_sampled'] + 2e-8 * max(1.,r['diameter']) for r in rows)
    summary['jacobian_interval_violations'] = sum(r['bound_sampled'] > r['bound_certified'] + 2e-8 * max(1.,r['diameter']) for r in rows)
    summary['original_lemma2_violations'] = sum(r['error'] > r['bound_lemma2'] + 2e-8 * max(1.,r['diameter']) for r in rows if r['lemma'] == 2)
    summary['max_reference_error'] = max(r.get('linear_reference_error', 0) for r in rows)
    summary['max_rematch_error'] = max(r['error'] for r in rows if r['family'] == 'rematch')
    summary['zero_drift_missing_beta_violations'] = sum(r['error'] > r['bound_without_beta'] + 2e-8 * max(1.,r['diameter']) for r in rows if r['family'] == 'generated_zero')
    preservation = ROOT / 'data/section4_measurement_hashes.json'
    if preservation.exists():
        snapshot = json.loads(preservation.read_text())
        checked = 0
        for name, hashes in snapshot['files'].items():
            state = json.loads((ROOT / 'results' / name).read_text())
            for unit, expected in hashes.items():
                row = state['units'][unit]
                current = hashlib.sha256(json.dumps({field:row[field] for field in snapshot['fields']}, sort_keys=True).encode()).hexdigest()
                if current != expected:
                    raise AssertionError(f'Measurement changed during Jacobian correction: {name}/{unit}')
                checked += 1
        summary['preserved_measurements_checked'] = checked
    if (summary['chain_violations'] or summary['jacobian_interval_violations']
            or summary['original_lemma2_violations'] or summary['redundancy']['violations']):
        raise AssertionError(f'Bound audit failed: {summary}')
    return summary


def report(summary, stats, rows):
    meta = json.loads((ROOT / 'results/section4_manifest.json').read_text())
    n = summary['overall']['behaviors']
    table = ['| Input steered k | Generated steered g | Mean error ± std | Mean bound ± std |',
             '|---:|---:|---:|---:|']
    for r in stats:
        if r['m'] != 4 or (r['setting'] == 'input_and_generated' and r['g'] == 0):
            continue
        table.append(f'| {r["k"]} | {r["g"]} | {r["error"]["mean"]:.4f} ± {r["error"]["std"]:.4f} | '
                     f'{r["bound"]["mean"]:.4f} ± {r["bound"]["std"]:.4f} |')
    prompt_table = ['| Prompt length m | Single-token steering error: mean ± std |', '|---:|---:|']
    for r in stats:
        if r['setting'] == 'input_only' and r['k'] == 1:
            prompt_table.append(f'| {r["m"]} | {r["error"]["mean"]:.4f} ± {r["error"]["std"]:.4f} |')
    early_eos = sum(151645 in json.loads((ROOT / f'checkpoints/section4_generation_{i:03d}.json').read_text())['tokens'][:-1] for i in range(n))
    summary['traces_with_early_eos'] = early_eos
    summary['comparison'] = stats
    text = f'''# Qwen3-4B / HarmBench: single- and multi-token steering

Completed **{n} HarmBench inputs**, **{summary['overall']['rows']:,} main conditions**, and **{summary['redundancy']['rows']:,} redundancy controls** on the local machine.

## The two requested figures

1. [Fixed prompt length: steering length versus actual error and bound](../figs/section4_steering_length.pdf). Prompt length m=4. Compare steering only input positions with steering four input positions plus generated positions.
2. [Fixed single-token steering: prompt length versus actual error](../figs/section4_prompt_length.pdf). k=1, g=0; prompt lengths m=1,2,4,8.

Both use **mean ± sample standard deviation across HarmBench inputs**. For each input and setting, first average its six head errors (three layers × two heads). Heads are not counted as independent input prompts. Shading is standard deviation, not a confidence interval; negative lower edges of mean-minus-std do not represent negative observed errors.

### Fixed prompt length m=4

{chr(10).join(table)}

### Fixed single-token steering k=1

{chr(10).join(prompt_table)}

All other m/k settings are retained in [the comparison CSV](section4_error_comparison.csv). These are descriptive finite-sample comparisons; increasing the number of positions is not mathematically guaranteed to increase error on every input.

## Experimental design

- Model: `{meta['model']}`, revision `{meta['revision']}`. BF16 model forward passes and float64 attention calculations. Layers 8,17,26 (0-indexed), query heads 0,16 in each, using their corresponding grouped-query K/V heads.
- Use all 400 records of the pinned HarmBench text CSV, including the supplied context. Inputs are not truncated. Apply the model's chat template with thinking disabled.
- Append the eight-token suffix `{meta['prompt_text']}`. The prompt-length settings use nested prefixes of 1,2,4,8 tokens. Thus "prompt length" on the second x-axis refers to this appended block, not to the length of the HarmBench question.
- Extract real post-input-layernorm representations from the model. Build a common state library and use its learned **linear** W_Q/W_K/W_V projections before Q/K normalization and RoPE, matching the assumptions of Section 4.
- Reference query q0 comes from the final original input position. For every input, head, and (m,k), construct r using the theorem's effective-value match plus a correction in ker(W_V) that matches aggregate attention weight. Single-token steering modifies the final input position; input-only multi-token steering adds the same r to the last k input positions.
- Keep that r fixed and compare the two attention outputs at the query from the final continuation position. The measured error is `||o_steer - o_prompt||_2`. This measures transfer of the initial match, not optimizer success or the existence of r: both methods match at q0 under the theorem's assumptions.
- For generated-position settings, keep k=4 and extend the same intervention to g=1,4,8,16 generated positions. The corresponding bound includes beta, the reference mismatch of the enlarged block.
- The common trace has 16 greedy decoding steps from the eight-token suffix. EOS does not terminate this fixed-length diagnostic: {early_eos}/{n} traces encounter EOS before step 16. Shorter prompt settings reuse the same captured generated states. This controls inherited-state and query differences; these are **fixed-state head-level comparisons**, not separate free-generation runs for each setting.
- Each head/example has its own constructed r. The experiment does not establish that one vector simultaneously matches every head, nor that the final model distributions or behaviors are equal.

## Bounds and checks

No extra diagnostic plots are produced. Full per-condition measurements and intermediate bounds remain available locally in `results/section4_000.json` through `results/section4_399.json`.

- Bound-chain violations (`E <= decomposition <= log-weight <= certified drift <= D`): **{summary['chain_violations']}**, tolerance `2e-8 * max(1,D)`.
- Original Lemma 2 violations: **{summary['original_lemma2_violations']}**.
- Rows exceeding the sampled lower endpoint of the path-J lemma formula: **{summary['rows_above_sampled_lower_endpoint']}**. If zero, the original formula is numerically supported through `E <= sampled endpoint <= actual lemma formula <= certified bound`.
- Maximum reference matching error: **{summary['max_reference_error']:.3e}**. Maximum rematching-control error: **{summary['max_rematch_error']:.3e}**.
- Removing beta creates **{summary['zero_drift_missing_beta_violations']:,} violations** in the zero-query-drift generated-position controls.
- The Jacobian coordinates are anchored at **the last input token t_n**, exactly as in the manuscript. A preservation audit checked **{summary.get('preserved_measurements_checked',0):,}** existing measurements while correcting the coordinate anchor; only Jacobian-dependent quantities were updated.

The plotted bound is a conservative evaluation of the relevant lemma. The infinity-to-2 Jacobian norm is exact by sign enumeration for at most eight coordinates; larger blocks use a column-norm upper bound and deterministic sign probes as a lower estimate. Nine path points are sampled. A valid between-point envelope adds `L/(2*(points-1))`, with `L = 2*(range(delta_P)*diameter(V_P) + range(delta_M)*diameter(V_M))`, and is capped by the sum of the two block diameters. The single-token, single-prompt Jacobian maximum is computed analytically from the sigmoid derivative. Finite sampling alone is not claimed to be an upper bound.

The additional zero-drift, rematching, query-direction, redundant-activation, and native-Q/K-normalization/RoPE controls are saved as numerical results. The native architecture control is outside the linear-key assumptions and is excluded from lemma-violation counts.

## Reproduce or resume

```bash
uv run python scripts/run_section4_bounds.py extract
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/run_section4_bounds.py analyze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/run_section4_bounds.py analyze --suite redundancy
uv run python scripts/plot_section4_bounds.py
uv run pytest -q tests/test_attention_bounds.py tests/test_run_section4_bounds.py tests/test_plot_section4_bounds.py
```

Independent CPU workers can use `--start 0 --stride 8` through `--start 7 --stride 8`. Decoding checkpoints are saved after each token. Replay conditions are appended to durable JSONL journals and consolidated into per-input JSON; interrupted runs skip completed units. Logs are under `logs/`, activation checkpoints under `checkpoints/`, and PDFs under `figs/`. **Email notifications have been removed from this experiment at the user's request.**

The report, summary, comparison CSV, manifest, and two PDF figures are retained in Git. Large activation checkpoints and per-condition metric files remain local.
'''
    (ROOT / 'results/section4_report.md').write_text(text)
    write_json_atomic(ROOT / 'results/section4_summary.json', summary)
    with (ROOT / 'results/section4_error_comparison.csv').open('w') as output:
        writer = csv.DictWriter(output, fieldnames=['setting','m','k','g','n_prompts','error_mean','error_std','bound_mean','bound_std'])
        writer.writeheader()
        for r in stats:
            writer.writerow({**{k:r[k] for k in ['setting','m','k','g']}, 'n_prompts':r['error']['n'],
                             'error_mean':r['error']['mean'], 'error_std':r['error']['std'],
                             'bound_mean':r['bound']['mean'], 'bound_std':r['bound']['std']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=400)
    args = parser.parse_args()
    with tee_stdout(ROOT / 'logs/section4_plot.log'):
        rows, extra = load_rows(args.limit), load_rows(args.limit, 'redundancy')
        summary = audit(rows, extra)
        stats = statistics(rows)
        plots(stats)
        report(summary, stats, rows)
        print(json.dumps({'overall':summary['overall'], 'comparison':stats}, indent=2), flush=True)


if __name__ == '__main__':
    main()
