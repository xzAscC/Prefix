"""Audit Lemma 8 replay and render behavior-level mean/std comparisons as PDFs."""
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

ROOT=Path(__file__).resolve().parents[1]


def mean_std(rows,field):
    grouped=defaultdict(list)
    for r in rows:
        grouped[r['index']].append(r[field])
    values=np.array([np.mean(v) for v in grouped.values()])
    return dict(n=len(values),mean=float(values.mean()) if len(values) else None,
                std=float(values.std(ddof=1)) if len(values)>1 else 0.)


def low_share_statistics(rows,epsilon):
    selected=[{**r,'epsilon_bound':epsilon*(r['value_shift_norm']+r['diameter_short'])}
              for r in rows if r['alpha']!=0 and r['extra_count']>0
              and max(r['w_short'],r['w_long'])<=epsilon]
    return dict(epsilon=epsilon,conditions=len(selected),error=mean_std(selected,'error'),
                bound=mean_std(selected,'epsilon_bound'),
                violations=sum(r['error']>r['epsilon_bound']+1e-8*max(1,r['diameter_short'],r['value_shift_norm']) for r in selected))


def load(limit,partial):
    rows=[];complete=0;shared_manifest=None;ineligible=0
    for index in range(limit):
        path=ROOT/f'results/lemma8_{index:03d}.json'
        if not path.exists():
            if partial:
                continue
            raise FileNotFoundError(path)
        state=json.loads(path.read_text());meta=state['manifest']
        shared={k:v for k,v in meta.items() if k not in ('index','activation_sha256')}
        if shared_manifest is not None and shared!=shared_manifest:
            raise ValueError('Mixed Lemma 8 manifests')
        shared_manifest=shared
        expected={f'{l}_{h}' for l in meta['source']['layers'] for h in meta['source']['heads']}
        done=set(state['units'])==expected
        if not done and not partial:
            raise ValueError(f'Incomplete heads: {path}')
        complete+=done
        for unit in state['units'].values():
            if {r['id'] for r in unit['rows']}!={c['id'] for c in meta['conditions']}:
                raise ValueError(f'Incomplete conditions: {path}')
            for r in unit['rows']:
                if not r['eligible']:
                    ineligible+=1
                    continue
                tolerance=1e-8*max(1,r['diameter_short'],r['value_shift_norm'])
                if not all(np.isfinite(r[k]) for k in ['error','bound_shares','bound_score','bound_epsilon','identity_residual']):
                    raise ValueError('Nonfinite evidence')
                if (r['error']>r['bound_shares']+tolerance or r['bound_shares']>r['bound_score']+tolerance
                    or r['bound_shares']>r['bound_epsilon']+tolerance or r['identity_residual']>tolerance):
                    raise ValueError(f'Lemma 8 violation: {path}/{r["id"]}')
                for e,bound in r['epsilon_bounds'].items():
                    eligible=max(r['w_short'],r['w_long'])<=float(e)
                    if eligible != (bound is not None) or (eligible and r['error']>bound+tolerance):
                        raise ValueError('Invalid conditional epsilon bound')
                rows.append({**r,'index':index,'layer':unit['layer'],'head':unit['head'],'base_length':unit['base_length']})
    return rows,dict(complete=complete==limit,completed_examples=complete,requested_examples=limit,
                     eligible_conditions=len(rows),ineligible_conditions=ineligible,violations=0,manifest=shared_manifest)


def plot(rows,layer,complete):
    cohort={r['index'] for r in rows if r['base_length']>=64}
    panels=[('Extra generated tokens','g',lambda r:r['m']==8 and r['k']==1 and r['alpha']==1),
            ('Strength α','alpha',lambda r:r['m']==8 and r['k']==1 and r['g']==127),
            ('Prompt tokens used to construct r','m',lambda r:r['k']==1 and r['g']==127 and r['alpha']==1),
            ('Short input support size','k',lambda r:r['m']==8 and r['g']==127 and r['alpha']==1 and r['index'] in cohort)]
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,(label,field,select) in zip(axes.flat,panels):
        groups=defaultdict(list)
        for r in rows:
            if r['schedule']=='generated' and select(r):
                groups[r[field]].append(r)
        x=sorted(groups)
        if not x:
            ax.text(.5,.5,'No eligible behaviors yet',ha='center',transform=ax.transAxes)
        for metric,name in [('error','Measured difference'),('bound_shares','Share bound'),('bound_score','Score bound')]:
            stats=[mean_std(groups[v],metric) for v in x]
            y=np.array([s['mean'] for s in stats]);sd=np.array([s['std'] for s in stats])
            ax.plot(x,y,'o-',label=name,markersize=4)
            ax.fill_between(x,np.maximum(0,y-sd),y+sd,alpha=.12)
        ax.set_xlabel(label);ax.set_ylabel('Attention output L2 distance');ax.grid(alpha=.2)
        if field in ('g','m','k'):
            ax.set_xscale('symlog',base=2,linthresh=1)
            ax.set_xticks(x,[str(v) for v in x])
    axes[0,0].legend(fontsize=9)
    title='All layers' if layer==-1 else f'Layer {layer} (zero-based)'
    fig.suptitle(f'Lemma 8 · {title} · prediction step 128'+('' if complete else ' · PARTIAL'))
    path=ROOT/f'figs/lemma8_{"mean" if layer==-1 else layer}.pdf'
    fig.savefig(path);plt.close(fig)
    print(f'Saved {path.name}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=400)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    with tee_stdout(ROOT/'logs/lemma8_summary.log'):
        rows,status=load(args.limit,args.allow_partial)
        if not rows:
            raise ValueError('No completed Lemma 8 measurements')
        layers=sorted({r['layer'] for r in rows})
        stats={}
        for layer in [-1,*layers]:
            subset=rows if layer==-1 else [r for r in rows if r['layer']==layer]
            stats[str(layer)]={str(e):low_share_statistics(subset,e) for e in [.001,.01,.05,.1]}
            plot(subset,layer,status['complete'])
        write_json_atomic(ROOT/'results/lemma8_summary.json',{**status,'low_share':stats})
        report=['# Lemma 8: prefix versus longer steering','',
                f'{status["completed_examples"]}/{args.limit} complete examples; {len(rows)} eligible head/condition measurements; zero violations.', '',
                'Both schedules share the query, original keys/values and the same scaled displacement. The long support contains the short support. Replays use the Section 4 linear-head convention before QK normalization and RoPE, at the query predicting token 128; no independent-generation claim follows.', '',
                'The first bound uses the diameter of ALL short-schedule values, including modified and unmodified positions. Attention shares are normalized over ALL visible tokens. The score bound uses the absolute added score, not query drift. The same displacement is constructed at the last input token and is reused when support changes.', '',
                'Figures show mean ± sample standard deviation after averaging heads (and layers in the mean figure) within each behavior. Input-support comparisons use a fixed cohort with at least 64 input tokens. The conditional table excludes zero-strength and empty-extra-set cases.', '',
                '| ε (both shares ≤ ε) | Eligible conditions | Behaviors | Mean difference | Mean ε bound | Violations |',
                '|---|---:|---:|---:|---:|---:|']
        for item in stats['-1'].values():
            report.append(f'| {item["epsilon"]} | {item["conditions"]} | {item["error"]["n"]} | {item["error"]["mean"]} | {item["bound"]["mean"]} | {item["violations"]} |')
        (ROOT/'results/lemma8_report.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':
    main()
