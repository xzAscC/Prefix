"""Audit complete native-attention runs and save three publication PDF figures."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from prefix.native_attention import LENGTHS, conditions
from prefix.runner import tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'input':'#2673B8','mixed':'#D97721','prompt':'#7B4FA3',
          'null':'#2673B8','sensitive':'#D97721','random':'#7B4FA3'}
STYLE = {'font.size':11, 'axes.labelsize':12, 'legend.fontsize':10,
         'pdf.fonttype':42, 'axes.spines.top':False, 'axes.spines.right':False}


def moments(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError('no data')
    return dict(n=len(values),mean=float(values.mean()),std=float(values.std(ddof=1)) if len(values)>1 else 0.)


def check_coverage(rows, indices, heads, configs):
    expected = {(i,h,c['id']) for i in indices for h in heads for c in configs}
    actual = [(r['identity']['index'],r['identity']['head'],r['identity']['condition']['id']) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError(f'incomplete or duplicate coverage: {len(actual)} / {len(expected)}')


def token_statistics(rows):
    groups = {}
    for row in rows:
        condition = row['identity']['condition']
        group = groups.setdefault(condition['id'], dict(condition=condition, examples={}))
        example = group['examples'].setdefault(row['identity']['index'], [])
        error = dict(zip(row['queries'],row['errors']))
        example.append({**{name:float(np.mean([error[q] for q in row['query_sets'][name]]))
                          for name in ['common','exposed','all']},
                        'dimension':row['diagnostic_dimension'],'reference':row['fit']['final_error'],
                        'expanded':row['expanded']})
    output = []
    for group in groups.values():
        averaged = [{key:float(np.mean([r[key] for r in rows])) for key in rows[0]}
                    for rows in group['examples'].values()]
        original = [row for row in averaged if not row['expanded']]
        output.append({**group['condition'], **{name:moments([row[name] for row in averaged])
                       for name in ['common','exposed','all','dimension','reference']},
                       'original_only':moments([row['common'] for row in original]) if original else None})
    return sorted(output,key=lambda row:(row['m'],row['b'],row['g']))


def drift_statistics(rows):
    groups = {}
    for row in rows:
        meta = row['identity']
        for unit in row['units'].values():
            key = (meta['condition']['id'],unit['direction'],unit['scale'])
            examples = groups.setdefault(key,{})
            examples.setdefault(meta['index'],[]).append(unit)
    fields = ['rho','query_change_norm','native_error','linear_error','linear_residual_aware_bound']
    return [dict(condition=key[0],direction=key[1],scale=key[2],
                 **{field:moments([np.mean([r[field] for r in group]) for group in examples.values()])
                    for field in fields}) for key,examples in sorted(groups.items())]


def line(ax, rows, x, field, label, color, band=True, linestyle='-'):
    rows = sorted(rows,key=x)
    xs = np.array([x(row) for row in rows])
    means = np.array([row[field]['mean'] for row in rows])
    stds = np.array([row[field]['std'] for row in rows])
    ax.plot(xs,means,marker='o',markersize=4,linewidth=1.7,color=color,label=label,linestyle=linestyle)
    if band:
        ax.fill_between(xs,np.maximum(means-stds,1e-12 if ax.get_yscale()=='log' else 0),
                        means+stds,color=color,alpha=.12,linewidth=0)


def token_axis(ax, label):
    ax.set_xscale('log',base=2)
    ax.set_xticks(LENGTHS,labels=[str(n) for n in LENGTHS])
    ax.set_xlabel(label)
    ax.grid(axis='y',alpha=.2)


def save(fig, name):
    path = ROOT / f'figs/{name}.pdf'
    fig.savefig(path,bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}',flush=True)


def figures(stats, drift, controls, n, suffix=''):
    with plt.rc_context(STYLE):
        fig,axes = plt.subplots(1,2,figsize=(11.2,3.5))
        left,right = axes
        left.set_yscale('log')
        input_rows = [r for r in stats if r['m']==4 and r['g']==0]
        mixed = [r for r in stats if r['m']==4 and r['b']==1]
        for rows,name,label in [(input_rows,'input','Input only'),(mixed,'mixed','1 input + generated')]:
            line(left,rows,lambda r:r['b']+r['g'],'common',label,COLORS[name])
        prompt = [r for r in stats if r['b']==1 and r['g']==0]
        line(right,prompt,lambda r:r['m'],'common','Single-token steering',COLORS['prompt'])
        token_axis(left,'Steered tokens'); token_axis(right,'Prompt tokens')
        left.set_ylabel('Attention-output L₂ error'); right.set_ylabel('Attention-output L₂ error')
        left.legend(frameon=False); right.legend(frameon=False)
        for ax,label in zip(axes,['(a)','(b)']):
            ax.text(.5,-.29,label,transform=ax.transAxes,ha='center')
        fig.suptitle(f'Qwen3-4B · layer 27 · native attention · {n} examples',fontsize=13)
        fig.text(.5,-.065,'Mean ± sample std across examples; 128 common held-out queries; r fitted at a separate query.',ha='center',fontsize=9)
        fig.tight_layout()
        save(fig,f'section4_native_token_error{suffix}')

        fig,axes = plt.subplots(2,2,figsize=(11.2,7))
        configs = [('m1_b1_g0','1 prompt, 1 input'),('m4_b1_g0','4 prompt, 1 input'),
                   ('m4_b4_g0','4 prompt, 4 input'),('m4_b1_g3','4 prompt, 1 input + 3 generated')]
        for ax,(condition,title) in zip(axes.flat,configs):
            for direction in ['null','sensitive','random']:
                rows = [r for r in drift if r['condition']==condition and r['direction']==direction]
                # Put roundoff-level null distances exactly at zero for display.
                line(ax,rows,lambda r:0 if r['rho']['mean'] < 1e-9 else r['rho']['mean'],
                     'native_error',direction,COLORS[direction],band=False)
            ax.set_xscale('symlog',linthresh=.01)
            ax.set_xlabel('Distance ρ to original linear U⊥')
            ax.set_ylabel('Native attention L₂ error')
            ax.set_title(title,fontsize=11)
            ax.grid(alpha=.2)
        axes[0,0].legend(frameon=False)
        fig.suptitle('Fixed optimized r: equal-norm query perturbations',fontsize=13)
        fig.text(.5,-.01,'U⊥ is a linear-theory diagnostic. Native QK normalization / RoPE need not preserve matching, even at ρ = 0.',ha='center',fontsize=9)
        fig.tight_layout()
        save(fig,f'section4_native_distance_error{suffix}')

        fig,axes = plt.subplots(1,3,figsize=(14,3.6))
        for rows,name,label in [(input_rows,'input','Input only'),(mixed,'mixed','1 input + generated')]:
            line(axes[0],rows,lambda r:r['b']+r['g'],'dimension',label,COLORS[name])
        line(axes[1],prompt,lambda r:r['m'],'dimension','Prompt length',COLORS['prompt'])
        for control,label,color in [('redundant','Redundant directions',COLORS['input']),('independent','Independent directions',COLORS['mixed'])]:
            group = [r for r in controls if r['control']==control]
            line(axes[2],group,lambda r:r['count'],'dimension',label,color)
        for ax,label in zip(axes,['Steered tokens','Prompt tokens','Controlled prompt-key count']):
            token_axis(ax,label)
            ax.set_ylim(-3,131)
            ax.set_ylabel('Dimension of original linear U⊥')
            ax.legend(frameon=False)
        fig.suptitle('Linear key-space diagnostic; fixed value-null direction within each example/head',fontsize=13)
        fig.text(.5,-.01,'The controlled panel adds key-space directions, not newly generated text. These dimensions are not a native-attention matching guarantee.',ha='center',fontsize=9)
        fig.tight_layout()
        save(fig,f'section4_native_subspace{suffix}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indices',type=int,nargs='+')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    with tee_stdout(ROOT/'logs/section4_native_plot.log'), contextlib.redirect_stderr(sys.stdout):
        cohort = json.loads((ROOT/'data/section4_native_cohort.json').read_text())
        indices = args.indices or [r['index'] for r in cohort['examples']]
        if args.indices and not args.smoke:
            raise ValueError('partial figures require --smoke; formal plots require the entire cohort')
        config = cohort['config']
        rows,drift = [],[]
        for index in indices:
            for head in config['heads']:
                for c in conditions():
                    name = f'{index:03d}_h{head:02d}_{c["id"]}'
                    row = json.loads((ROOT/f'results/section4_native_{name}.json').read_text())
                    if not row['complete'] or row['identity']['cohort'] != cohort['fingerprint']:
                        raise ValueError('incomplete or inconsistent result')
                    queries = row['query_sets']
                    if len(queries['common']) != 128 or queries['reference'] in row['queries']:
                        raise ValueError('held-out query coverage failed')
                    if not set(queries['common']) <= set(row['queries']) or len(row['errors']) != len(row['queries']):
                        raise ValueError('missing query measurements')
                    if not np.isfinite(row['errors']).all() or min(row['errors']) < 0:
                        raise ValueError('nonfinite/negative errors')
                    error = dict(zip(row['queries'],row['errors']))
                    if max((error[q] for q in queries['pre_prompt']),default=0.) > 1e-9:
                        raise ValueError('pre-intervention causal control failed')
                    rows.append(row)
                    if (c['m'],c['b'],c['g']) in [(1,1,0),(4,1,0),(4,4,0),(4,1,3)]:
                        d = json.loads((ROOT/f'results/section4_native_drift_{name}.json').read_text())
                        expected = {f'{direction}_{scale:g}' for direction in ['null','sensitive','random'] for scale in config['drift_scales']}
                        if d['identity'] != row['identity'] or not d['complete'] or set(d['units']) != expected:
                            raise ValueError('query-drift coverage failed')
                        if any(r['linear_error'] > r['linear_residual_aware_bound']+1e-7 for r in d['units'].values()):
                            raise ValueError('linear replay certificate violated')
                        drift.append(d)
        check_coverage(rows,indices,config['heads'],conditions())
        stats = token_statistics(rows)
        drift_stats = drift_statistics(drift)
        grouped = {}
        for row in rows:
            if row['identity']['condition']['id'] != 'm1_b1_g0':
                continue
            for control in row['controlled_dimensions']:
                key = (control['control'],control['count'])
                grouped.setdefault(key,{}).setdefault(row['identity']['index'],[]).append(control['dimension'])
        controls = [dict(control=name,count=count,dimension=moments([np.mean(v) for v in group.values()]))
                    for (name,count),group in sorted(grouped.items())]
        suffix = '_smoke' if args.smoke else ''
        fit_errors = [r['fit']['final_error'] for r in rows]
        summary = dict(cohort_fingerprint=cohort['fingerprint'],examples=len(indices),indices=indices,
            heads=config['heads'],layer=config['layer'],conditions=len(rows),drift_units=sum(len(r['units']) for r in drift),
            query_measurements=sum(len(r['errors']) for r in rows),common_queries_per_condition=128,
            expanded_examples=sum(r['expanded'] for r in cohort['examples'] if r['index'] in indices),
            max_reference_error=max(fit_errors),median_reference_error=float(np.median(fit_errors)),
            unconverged_fits=sum(not r['fit']['converged'] for r in rows),
            reference_error_above_1e_5=sum(e>1e-5 for e in fit_errors),
            analysis_hashes=sorted({r['identity']['analysis_sha256'] for r in rows}),
            token_statistics=stats,drift_statistics=drift_stats,controlled_dimensions=controls)
        if len(summary['analysis_hashes']) != 1:
            raise ValueError('mixed runner implementations')
        write_json_atomic(ROOT/f'results/section4_native_summary{suffix}.json',summary)
        figures(stats,drift_stats,controls,len(indices),suffix)
        with (ROOT/f'results/section4_native_token_statistics{suffix}.csv').open('w',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['m','input_steered','generated_steered','n','common_query_error_mean','common_query_error_std',
                             'all_query_error_mean','exposed_query_error_mean','linear_dimension_mean','reference_error_mean'])
            for r in stats:
                writer.writerow([r['m'],r['b'],r['g'],r['common']['n'],r['common']['mean'],r['common']['std'],
                                 r['all']['mean'],r['exposed']['mean'],r['dimension']['mean'],r['reference']['mean']])
        report = f'''# Native attention experiments at Qwen3-4B layer 27

## Coverage and numerical audit

- Examples: {len(indices)} ({summary['expanded_examples']} explicitly expanded inputs).
- Heads: 0 and 16, fitted separately; layer index 26. This is a head-level study, before the output projection.
- Completed token configurations: {len(rows)}; query evaluations: {summary['query_measurements']:,}.
- Query-drift measurements: {summary['drift_units']:,}.
- Reference-fit error: median {summary['median_reference_error']:.3g}, maximum {max(fit_errors):.3g}; {summary['unconverged_fits']} fits did not reach the configured tolerance.
- Pre-prompt causal controls and residual-aware linear replay certificates passed their numerical checks.

## Design

The cohort takes all 95 original HarmBench inputs with at least 128 input tokens and the five longest remaining inputs. The latter retain their original requests and receive the explicit neutral introductory context recorded in `data/section4_native_cohort.json`. This selected contextual cohort is not representative of all 400 behaviors. Summary JSON also gives the 95 original-only sensitivity results.

Frozen representations come from the pinned Qwen3-4B revision. A common greedy trace uses the original 8-token instruction suffix and is extended to 256 tokens; EOS positions are recorded, but EOS does not truncate this fixed-length diagnostic. The prompt library contains 128 nested instruction tokens. These are controlled fixed-state calculations, not separately generated trajectories for each condition.

Every evaluation uses Qwen3 QK RMS normalization, RoPE, and an actual causal mask. The prompt arm has its own compact positions for input + prompt + continuation; the steering arm has compact positions for input + continuation. The same unmodified reading representation is used in both arms, with its appropriate position ID. The library is frozen; we do not claim equality of the hidden states that independent full-model runs would produce.

For each example/head/configuration, L-BFGS optimizes a shared displacement r against the native head output at generated position 256. Optimization takes place in the realizable row space of [W_K; W_V] and saves optimizer state after every step. r is then fixed. All prompt positions, steered positions, and the fitting query are excluded from evaluation. The primary curves use the same 128 held-out positions (generated positions 128–255) for every configuration. All other eligible queries are also measured; pre-prompt queries have zero causal exposure and are reported separately rather than diluting the primary curves.

Input steering affects the last b input tokens. Mixed steering always affects one input token plus the first g generated tokens. The steering-count sweep fixes m=4; the prompt-count sweep fixes b=1, g=0. Error is the unnormalized L2 norm of the head-output difference. Average across queries first, then the two heads within each behavior, then report the behavior mean and sample standard deviation.

## Three figures and theory boundary

1. `figs/section4_native_token_error{suffix}.pdf`: native attention error versus steered and prompt token counts.
2. `figs/section4_native_distance_error{suffix}.pdf`: equal-norm changes of the raw projected query, within the original linear U-perp, within its orthogonal complement, or random; measured native error versus actual distance rho.
3. `figs/section4_native_subspace{suffix}.pdf`: dimension of the original linear-theory diagnostic, including controlled redundant versus independent key directions. These controls operate in key space, not on new text.

The original U uses raw linear key differences and one fixed value-null direction z for each example/head. With r numerically fitted to native attention, this U is a diagnostic, not a guaranteed matching subspace. Native QK normalization and RoPE are not covered by the linear matching theorem. We therefore do not label the original theorem's bounds as native-attention upper bounds. The summary separately records errors and residual-aware certificates for the corresponding linear replay, including its nonzero reference mismatch.

Results are empirical; increasing errors or decreasing dimensions are not imposed on the measurements.
'''
        (ROOT/f'results/section4_native_report{suffix}.md').write_text(report)
        print(json.dumps({k:v for k,v in summary.items() if not isinstance(v,list)},indent=2),flush=True)


if __name__ == '__main__':
    main()
