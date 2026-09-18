"""Layer-specific, matched-cohort long-token plots (PDF only; no notifications)."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prefix.runner import tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = [1,2,4,8,16,32,64,128]


def mean_std(rows, field):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['index'], []).append(row[field])
    values = np.array([np.mean(v) for v in grouped.values()])
    return dict(n=len(values), mean=float(values.mean()),
                std=float(values.std(ddof=1)) if len(values)>1 else 0.)


def load_rows(limit):
    rows, manifest = [], None
    for index in range(limit):
        path = ROOT / f'results/section4_long_{index:03d}.json'
        state = json.loads(path.read_text())
        meta = state['manifest']
        expected = {f'{layer}_{head}/{c["id"]}' for layer in meta['layers']
                    for head in meta['heads'] for c in meta['conditions']}
        if set(state['units']) != expected:
            raise ValueError(f'Incomplete conditions: {path}')
        if meta.get('jacobian_anchor') != 'last_input_token':
            raise ValueError(f'Wrong anchor: {path}')
        if manifest is not None and meta != manifest:
            raise ValueError(f'Inconsistent manifest: {path}')
        manifest = meta
        if any(r['index'] != index for r in state['units'].values()):
            raise ValueError(f'Misassigned input: {path}')
        rows.extend(state['units'].values())
    return rows


def statistics(rows):
    cohort = {r['index'] for r in rows if r['base_length'] >= 128}
    groups = {}
    for row in rows:
        if not row['eligible']:
            continue
        figures = []
        if row['m']==4 and row['index'] in cohort:
            figures.append('steering')
        if row['k']==1 and row['g']==0:
            figures.append('prompt')
        for figure in figures:
            key = (row['layer'],figure,row['m'],row['k'],row['g'])
            groups.setdefault(key,[]).append(row)
    return [dict(layer=layer,figure=figure,m=m,k=k,g=g,
                 error=mean_std(group,'error'), bound=mean_std(group,'bound_certified'))
            for (layer,figure,m,k,g),group in sorted(groups.items())]


def layer_mean_statistics(rows):
    """Average the ten head-level errors per input, then summarize inputs."""
    return statistics([{**row, 'layer':-1} for row in rows])


def trends(stats):
    output = []
    for layer in sorted({r['layer'] for r in stats}):
        groups = {
            'input_only': [r for r in stats if r['layer']==layer and r['figure']=='steering' and r['g']==0],
            'input_and_generated': [r for r in stats if r['layer']==layer and r['figure']=='steering' and (r['g'] or r['k']==4)],
            'prompt_length': [r for r in stats if r['layer']==layer and r['figure']=='prompt'],
        }
        for setting,rows in groups.items():
            if len(rows)<2:
                continue
            rows = sorted(rows,key=lambda r:r['m'] if setting=='prompt_length' else r['k']+r['g'])
            values = np.array([r['error']['mean'] for r in rows])
            output.append(dict(layer=layer+1,setting=setting,
                               first=rows[0]['error'],last=rows[-1]['error'],
                               last_minus_first=float(values[-1]-values[0]),
                               monotone_nondecreasing=bool(np.all(np.diff(values)>=0))))
    return output


COLORS = {'input':'#2673B8', 'mixed':'#D97721', 'prompt':'#7B4FA3'}
STYLE = {
    'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':11,
    'axes.labelsize':10, 'legend.fontsize':8, 'pdf.fonttype':42,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.edgecolor':'#999999', 'axes.linewidth':.7,
    'xtick.color':'#444444', 'ytick.color':'#444444',
    'figure.facecolor':'white', 'savefig.facecolor':'white',
}


def line(ax, rows, field, x, label, color, log_floor=None, band_floor=None):
    rows = sorted(rows,key=x)
    xs = [x(r) for r in rows]
    means = np.array([r[field]['mean'] for r in rows])
    stds = np.array([r[field]['std'] for r in rows])
    bound = field=='bound'
    displayed_means = means if log_floor is None else np.ma.masked_less_equal(means,0)
    floor = log_floor if log_floor is not None else band_floor
    lower = means-stds if floor is None else np.maximum(means-stds,floor)
    upper = means+stds if log_floor is None else np.maximum(means+stds,log_floor)
    ax.plot(xs,displayed_means,linestyle='--' if bound else '-', marker='s' if bound else 'o',
            label=label,color=color,linewidth=1.7,markersize=4,
            markerfacecolor='white' if bound else color,zorder=3)
    ax.fill_between(xs,lower,upper,color=color,alpha=.07 if bound else .14,
                    linewidth=0,zorder=1)


def format_axis(ax, xlabel):
    ax.set_xscale('log',base=2)
    ax.set_xticks(LENGTHS,labels=[str(x) for x in LENGTHS])
    ax.set_xlabel(xlabel,labelpad=8)
    ax.grid(axis='y',color='#E6E8EB',linewidth=.65)
    ax.set_axisbelow(True)
    ax.margins(x=.04)
    ax.tick_params(length=3,width=.6)


def combined_figure(stats, layer):
    """One figure: overlaid error/bound at left, prompt-length error at right."""
    steering = [r for r in stats if r['layer']==layer and r['figure']=='steering']
    prompt = [r for r in stats if r['layer']==layer and r['figure']=='prompt']
    label = 'Five-layer mean' if layer==-1 else f'Layer {layer+1}'
    with plt.rc_context(STYLE):
        fig,(left,right) = plt.subplots(1,2,figsize=(11.2,4.5),gridspec_kw={'width_ratios':[1.25,1]})
        moments = [r[field] for r in steering for field in ['error','bound']]
        positive = [v for r in moments for v in [r['mean'],r['mean']-r['std']] if v>0]
        low = min(positive)/1.8
        high = max(r['mean']+r['std'] for r in moments)
        left.set_yscale('log')
        left.set_ylim(low,high*(high/low)**.34)
        for rows,color,name in [
            ([r for r in steering if r['g']==0],COLORS['input'],'input only'),
            ([r for r in steering if r['g'] or r['k']==4],COLORS['mixed'],'4 input + generated'),
        ]:
            for field,kind in [('error','Measured'),('bound','Bound')]:
                line(left,rows,field,lambda r:r['k']+r['g'],f'{kind} · {name}',color,log_floor=low)
        format_axis(left,'Steered positions')
        left.set_ylabel('Raw L₂ error / bound (log scale)',labelpad=8)
        left.text(.5,-.30,'(a)',transform=left.transAxes,ha='center',va='top',fontsize=11)
        left.legend(loc='upper left',bbox_to_anchor=(0,1.01),frameon=False,handlelength=2.6,
                    labelspacing=.45,borderaxespad=.4)
        line(right,prompt,'error',lambda r:r['m'],'Measured · single-token steering',COLORS['prompt'],band_floor=0)
        format_axis(right,'Appended prompt tokens')
        right.set_ylabel('Raw L₂ error',labelpad=8)
        right.text(.5,-.30,'(b)',transform=right.transAxes,ha='center',va='top',fontsize=11)
        right.legend(loc='upper left',frameon=False,handlelength=2.6)
        low,high = right.get_ylim()
        right.set_ylim(0,high+.13*(high-low))
        fig.suptitle(f'Qwen3-4B  /  {label}',x=.085,ha='left',y=.99,fontsize=14,fontweight='bold')
        fig.text(.085,.015,'Mean ± std · raw L₂ differences · left: log scale; bands clipped at positive axis floor',
                 fontsize=9,color='#666666')
        fig.subplots_adjust(left=.085,right=.985,bottom=.28,top=.90,wspace=.32)
    return fig


def figure_name(layer):
    return 'section4_layer_mean_combined.pdf' if layer==-1 else f'section4_layer{layer+1:02d}_combined.pdf'


def plots(stats, mean_stats=()):
    combined = list(stats)+list(mean_stats)
    for layer in sorted({r['layer'] for r in combined}):
        fig = combined_figure(combined,layer)
        path = ROOT/'figs'/figure_name(layer)
        fig.savefig(path,bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {path.name}',flush=True)


def audit(rows):
    valid = [r for r in rows if r['eligible']]
    chain = ['error','bound_decomp','bound_log','bound_certified','diameter']
    counts = {'chain_violations':0,'jacobian_interval_violations':0,'original_lemma2_violations':0,
              'rows_above_sampled_lower_endpoint':0}
    for r in valid:
        tol = 2e-8*max(1.,r['diameter'])
        counts['chain_violations'] += any(r[a]>r[b]+tol for a,b in zip(chain,chain[1:]))
        counts['jacobian_interval_violations'] += r['bound_sampled']>r['bound_certified']+tol
        counts['original_lemma2_violations'] += r['lemma']==2 and r['error']>r['bound_lemma2']+tol
        counts['rows_above_sampled_lower_endpoint'] += r['error']>r['bound_sampled']+tol
    if any(counts[k] for k in ['chain_violations','jacobian_interval_violations','original_lemma2_violations']):
        raise AssertionError(counts)
    return dict(**counts,rows=len(rows),eligible_rows=len(valid),ineligible_rows=len(rows)-len(valid),
                max_identity_residual=max(r['identity_residual'] for r in valid),
                cohort_indices=sorted({r['index'] for r in valid if r['base_length']>=128}))


def report(rows,stats,summary,mean_stats=()):
    table = ['| Layer | Setting | m | k | g | n | Error mean ± std | Bound mean ± std |',
             '|---:|:---|---:|---:|---:|---:|---:|---:|']
    for r in stats:
        table.append(f'| {r["layer"]+1} | {r["figure"]} | {r["m"]} | {r["k"]} | {r["g"]} | {r["error"]["n"]} | '
                     f'{r["error"]["mean"]:.4f} ± {r["error"]["std"]:.4f} | '
                     f'{r["bound"]["mean"]:.4f} ± {r["bound"]["std"]:.4f} |')
    mean_table = ['| Setting | m | k | g | n | Error mean ± std | Bound mean ± std |',
                  '|:---|---:|---:|---:|---:|---:|---:|']
    for r in mean_stats:
        mean_table.append(f'| {r["figure"]} | {r["m"]} | {r["k"]} | {r["g"]} | {r["error"]["n"]} | '
                          f'{r["error"]["mean"]:.4f} ± {r["error"]["std"]:.4f} | '
                          f'{r["bound"]["mean"]:.4f} ± {r["bound"]["std"]:.4f} |')
    links = '\n'.join(f'- Layer {layer+1}: [combined figure](../figs/{figure_name(layer)}).'
                      for layer in sorted({r['layer'] for r in stats}))
    links += f'\n- Five-layer mean: [combined figure](../figs/{figure_name(-1)}).'
    n = len({r['index'] for r in rows})
    early_eos = 0
    for path in sorted((ROOT/'checkpoints').glob('section4_long_decode_*.json')):
        state = json.loads(path.read_text())
        early_eos += sum(151645 in tokens[:-1] for index,tokens in zip(state['indices'],state['tokens']) if index<n)
    summary['traces_with_early_eos'] = early_eos
    records = json.loads((ROOT/'data/section4_harmbench.json').read_text())
    summary['cohort_categories'] = dict(Counter(records[i]['FunctionalCategory'] for i in summary['cohort_indices']))
    summary['layer_trends'] = trends(stats)
    trend_text = '\n'.join(f'- Layer {r["layer"]}, {r["setting"]}: endpoint mean error '
                           f'{r["first"]["mean"]:.4f} → {r["last"]["mean"]:.4f}; '
                           f"monotone nondecreasing across all counts: {r['monotone_nondecreasing']}."
                           for r in summary['layer_trends'])
    body = f'''# Long-token steering comparison by layer

Qwen3-4B, {n} HarmBench inputs; layers 3, 9, 18, 27, 34 (one-based). Two query heads (0 and 16) per layer. Each layer is reported separately: average the two head measurements within an input, then compute mean and sample standard deviation across inputs. Shading is standard deviation, not confidence intervals. Negative mean-minus-std values are not negative observed errors.

## Figures

{links}

Each layer has one two-panel PDF: the left panel overlays measured error (solid) and the certified theoretical upper bound (dashed); the right panel shows prompt-length error in purple. The left y-axis is logarithmic and the right y-axis is linear. Standard-deviation bands are clipped at the positive axis floor on the left and at zero on the right for display only; raw means and standard deviations are unchanged. A sixth PDF averages the five layers. Open [the plotting notebook](../notebooks/section4_attention_bounds.ipynb) to adjust and regenerate the figures from the tracked summary without rerunning the model. The x-axis uses base-2 logarithmic spacing with explicitly labeled token counts 1, 2, 4, 8, 16, 32, 64, 128.

- Fixed appended prompt length m=4: input-only steering uses the last k input positions. Mixed steering uses four input positions plus g generated positions, with total k+g=4,8,16,32,64,128. Both curves use the **same {len(summary['cohort_indices'])} inputs with at least 128 input tokens**, including the model chat template. No short inputs are padded or silently clamped. The fixed subset's functional categories are {summary['cohort_categories']}; its conclusions do not automatically generalize to the full dataset.
- Fixed single-token steering k=1: vary the appended prompt length m over the eight token counts, using all {n} inputs. This x-axis is the appended instruction block length, not the HarmBench question length or number of input examples.

## Design and interpretation

Use the same pinned Qwen3-4B revision and HarmBench records as [the initial study](section4_report.md). Capture post-input-layernorm representations using BF16 forward passes, then replay each head's learned linear W_Q/W_K/W_V in float64, before Q/K normalization and RoPE. These are conditional attention-level comparisons under the lemma assumptions, not full-model free-generation steering interventions.

The reference q0 is from the last original input token. Construct a separate r for each input, layer, head, prompt length, and input steering count: match the effective value, then adjust aggregate attention mass in the null space of W_V. The same r is added to every selected position. Mixed input/generated conditions retain the r constructed from the four input positions; their bounds include the enlarged block's reference mismatch beta. Existence at q0 is imposed by construction; this experiment tests transfer error, not how easily an optimizer finds r or whether one vector matches all heads.

Extend the original eight-token prompt's common greedy trace from 16 to **128 steps**, preserving its first 16 tokens and checkpointing each additional token. All long-study conditions use the query at the final continuation position. Therefore even the short-count points are recomputed consistently for this target query; they must not be spliced with the initial study's 16-step-query results. EOS does not stop this diagnostic; {early_eos}/{n} traces encounter EOS before the final step.

Long prompt blocks are nested prefixes of one extended safety instruction. Their causal representations come from the original input plus 128 prompt tokens. Generated states come from the common eight-token-prompt trace. Combining these into a fixed-state library holds the query and generated representations constant across m; it is not a claim that separate prompts produce identical states in the full architecture. Every layer shares input identities, selected positions, and trace token IDs.

Actual error is ||o_steer - o_prompt||_2, without division by the output norm, head dimension, or diameter. Softmax and the model input layer normalization are part of constructing the attention output, not a normalization of this error metric. The five-layer mean first averages the ten head-level scalar errors (five layers × two heads) for each input, then computes mean ± sample std across inputs; it does not average the five standard deviations or take the norm after averaging vectors. Bounds are averaged identically. Layers with larger raw error magnitudes contribute more to the resulting mean. The bound follows the same anchored-at-last-input-token calculation as the initial study, using certified between-grid Jacobian envelopes and the diameter cap. Blocks larger than eight coordinates use conservative column-norm upper bounds; sampled sign probes provide a lower estimate. Large-count bounds may be loose or saturate at the diameter. Increasing error is a hypothesis, not an enforced constraint, and lengths also change instruction content.

## Numerical audit

- Completed conditions: {summary['rows']:,}; eligible: {summary['eligible_rows']:,}; explicitly ineligible input-length conditions: {summary['ineligible_rows']:,}.
- Bound-chain violations: {summary['chain_violations']}; original Lemma 2 violations: {summary['original_lemma2_violations']}; Jacobian interval violations: {summary['jacobian_interval_violations']}.
- Rows above the sampled lower endpoint: {summary['rows_above_sampled_lower_endpoint']}.
- Maximum decomposition identity residual: {summary['max_identity_residual']:.3e}.

## Observed trends

{trend_text}

These are descriptive mean differences, without a significance claim. A lower single-token endpoint does not imply monotonicity at intermediate counts.

## Five-layer mean ± std

{chr(10).join(mean_table)}

## Per-layer mean ± std

{chr(10).join(table)}

## Reproduce or resume

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/run_section4_long.py extract
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/run_section4_long.py analyze
uv run python scripts/plot_section4_long.py
```

Decode checkpoints persist after every token; analysis journals persist after every condition and consolidate to JSON per input. Completed conditions are skipped on restart. Use analyze --start N --stride W for W independent workers. Logs contain logs only. Notifications are disabled; no email is sent. Large checkpoints and raw conditions remain local; manifests, summary, CSV, report, and PDF figures are retained in Git.
'''
    (ROOT/'results/section4_long_report.md').write_text(body)
    write_json_atomic(ROOT/'results/section4_long_summary.json',{**summary,'comparison':stats,'layer_mean_comparison':list(mean_stats)})
    fields = ['layer','figure','m','k','g','n','error_mean','error_std','bound_mean','bound_std']
    with (ROOT/'results/section4_long_comparison.csv').open('w') as output:
        writer = csv.DictWriter(output,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for r in stats:
            writer.writerow(dict(layer=r['layer']+1,figure=r['figure'],m=r['m'],k=r['k'],g=r['g'],
                                 n=r['error']['n'],error_mean=r['error']['mean'],error_std=r['error']['std'],
                                 bound_mean=r['bound']['mean'],bound_std=r['bound']['std']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=400)
    args = parser.parse_args()
    with tee_stdout(ROOT/'logs/section4_long_plot.log'):
        rows = load_rows(args.limit)
        summary = audit(rows)
        stats = statistics(rows)
        mean_stats = layer_mean_statistics(rows)
        plots(stats,mean_stats)
        report(rows,stats,summary,mean_stats)
        print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
