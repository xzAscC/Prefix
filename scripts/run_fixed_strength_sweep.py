"""Native frozen-context strength sweep; uses the existing 100-example cohort.

Extract once with --phase extract. Then run --phase run --mode single-ratio.
The readout follows 128 actual generated tokens, excluding every intervention.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from prefix.duration_strength import calibrated_strength, sample_summary
from prefix.fixed_attention_strength import FixedNativeAttention
from prefix.runner import tee_stdout, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'lemma5_qwen3_native'
TAG = 'lemma5_qwen3_fixed_ratio'
FACTORS = [.001, .01, .1, 1., 10.]
LENGTHS = [4,8,16,32,64,128]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    return json.loads((ROOT/f'results/{SOURCE}_manifest.json').read_text())


def extract(limit):
    from transformers import AutoModelForCausalLM
    m = source_manifest()
    model = AutoModelForCausalLM.from_pretrained(m['model'], revision=m['revision'],
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    attn = model.model.layers[m['layer_index']].self_attn
    for i in range(limit):
        path = ROOT/f'checkpoints/{TAG}_states_{i:03d}.pt'
        metadata_path = path.with_suffix('.json')
        source_path = ROOT/f'results/{SOURCE}_{i:03d}.json'
        identity = dict(source_sha256=digest(source_path), model=m['model'], revision=m['revision'],
                        layer_index=m['layer_index'], head=m['head'], readout='after 128 generated tokens')
        if metadata_path.exists():
            saved = json.loads(metadata_path.read_text())
            if saved['identity'] != identity or saved['states_sha256'] != digest(path):
                raise ValueError('state extraction identity mismatch')
            print(f'Extract {i+1}/100 already complete', flush=True)
            continue
        row = json.loads(source_path.read_text())
        ids = row['prompt_ids'] + row['baseline_tokens']
        captured = {}
        def capture(_module, args, kwargs):
            captured['hidden'] = kwargs['hidden_states'][0].detach().cpu()
            captured['embeddings'] = tuple(x.detach().cpu() for x in kwargs['position_embeddings'])
        def output(_module, args):
            d = attn.head_dim
            captured['native_bf16_output'] = args[0][0,-1,m['head']*d:(m['head']+1)*d].detach().float().cpu()
        handles = [attn.register_forward_pre_hook(capture, with_kwargs=True),
                   attn.o_proj.register_forward_pre_hook(output)]
        try:
            with torch.inference_mode():
                model(torch.tensor([ids], device='cuda'), use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        captured.update(prompt_length=len(row['prompt_ids']), token_ids=ids,
                        identity=identity, behavior_id=row['example']['behavior_id'])
        temp = path.with_suffix('.tmp')
        torch.save(captured, temp)
        os.replace(temp, path)
        write_json_atomic(metadata_path, dict(identity=identity, states_sha256=digest(path)))
        print(f'Extracted {i+1}/100; context positions={len(ids)}', flush=True)


def run(limit, mode):
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    m = source_manifest()
    folder = Path(hf_hub_download(m['model'], 'config.json', revision=m['revision'], local_files_only=True)).parent
    cfg = AutoConfig.from_pretrained(folder, local_files_only=True)
    cfg._attn_implementation = 'eager'
    # Load the selected module's genuine pretrained parameters without a second
    # full-model allocation. BF16 weights match extraction, then upcast to FP32.
    attention = Qwen3Attention(cfg, m['layer_index']).to(dtype=torch.bfloat16)
    prefix = f'model.layers.{m["layer_index"]}.self_attn.'
    weights = {}
    for file in folder.glob('*.safetensors'):
        with safe_open(file, framework='pt', device='cpu') as f:
            for name in f.keys():
                if name.startswith(prefix):
                    weights[name[len(prefix):]] = f.get_tensor(name)
    attention.load_state_dict(weights, strict=True)
    attention = attention.cuda().eval()
    direction_path = ROOT/f'checkpoints/{SOURCE}_direction.json'
    d = json.loads(direction_path.read_text())
    direction = torch.tensor(d['direction'], device='cuda') * d['mean_norm']
    manifest = dict(version=1, source=m, mode=mode, factors=FACTORS, lengths=LENGTHS,
        distributed_strength=.1, n_examples=100,
        query='last token of prompt plus 128-token unsteered continuation; no token 129 generated',
        native_attention='actual Qwen3Attention forward including QK norm, RoPE, causal mask',
        arithmetic='BF16 model representations and weights upcast; native attention evaluated in FP32',
        comparison='frozen pre-intervention representations and identical unsteered readout query',
        direction_sha256=digest(direction_path),
        source_code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/prefix/fixed_attention_strength.py']})
    mp = ROOT/f'results/{TAG}_manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('sweep manifest mismatch')
    write_json_atomic(mp, manifest)
    for i in range(limit):
        path = ROOT/f'results/{TAG}_{i:03d}.json'
        state_path = ROOT/f'checkpoints/{TAG}_states_{i:03d}.pt'
        identity = dict(manifest=manifest, states_sha256=digest(state_path), index=i)
        state = json.loads(path.read_text()) if path.exists() else dict(identity=identity, conditions=[])
        if state['identity'] != identity:
            raise ValueError('example identity mismatch')
        if len(state['conditions']) == 30:
            print(f'Sweep {i+1}/100 already complete', flush=True)
            continue
        saved = torch.load(state_path, weights_only=True, map_location='cuda')
        study = FixedNativeAttention(attention, saved['hidden'], saved['embeddings'],
                                     saved['prompt_length'], m['head'])
        if 'baseline_output' not in state:
            base, pi = study.output(direction, 0, 0.)
            same, _ = study.output(direction, 1, .1)
            state.update(behavior_id=saved['behavior_id'], baseline_output=base.cpu().tolist(),
                         attention=pi.cpu().tolist(), native_bf16_output=saved['native_bf16_output'].cpu().tolist(),
                         uncalibrated_output=same.cpu().tolist(), targets={},
                         bf16_fp32_relative_difference=float((base-saved['native_bf16_output']).norm()/base.norm().clamp_min(1e-30)))
            write_json_atomic(path, state)
        base = torch.tensor(state['baseline_output'],device='cuda')
        pi = state['attention']
        same = torch.tensor(state['uncalibrated_output'],device='cuda')
        seen = {(c['length'],c['factor']) for c in state['conditions']}
        for k in LENGTHS:
            if str(k) not in state['targets']:
                target, _ = study.output(direction, k, .1)
                state['targets'][str(k)] = target.cpu().tolist()
                write_json_atomic(path, state)
            target = torch.tensor(state['targets'][str(k)],device='cuda')
            beta = calibrated_strength(pi, saved['prompt_length'], k, .1)
            effect = float((target-base).norm())
            for factor in FACTORS:
                if (k,factor) in seen:
                    continue
                actual, _ = study.output(direction, 1, factor*beta)
                error = float((target-actual).norm())
                state['conditions'].append(dict(length=k, factor=factor, beta=beta,
                    single_strength=factor*beta, output_l2=error,
                    uncalibrated_l2=float((target-same).norm()), target_effect_l2=effect,
                    error_over_target_effect=error/effect if effect>1e-12 else None,
                    target_output=target.cpu().tolist(), single_output=actual.cpu().tolist(),
                    uncalibrated_output=same.cpu().tolist()))
                write_json_atomic(path, state)
        state['complete'] = True
        write_json_atomic(path, state)
        print(f'Completed fixed-query sweep {i+1}/100 (30 conditions)', flush=True)
        del study


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = [json.loads((ROOT/f'results/{TAG}_{i:03d}.json').read_text()) for i in range(100)]
    assert len({r['behavior_id'] for r in records}) == 100
    table = []
    for k in LENGTHS:
        for factor in FACTORS:
            cells = [next(c for c in r['conditions'] if c['length']==k and c['factor']==factor) for r in records]
            for c in cells:
                np.testing.assert_allclose(np.linalg.norm(np.asarray(c['target_output'])-c['single_output']),
                                           c['output_l2'], rtol=1e-6, atol=1e-7)
            row = dict(length=k, factor=factor)
            for key in ['output_l2','uncalibrated_l2','target_effect_l2','beta','error_over_target_effect']:
                stats = sample_summary([c[key] for c in cells if c[key] is not None])
                row.update({f'{key}_{field}':stats[field] for field in ['mean','std']})
            row['improved_count'] = sum(c['output_l2'] < c['uncalibrated_l2'] for c in cells)
            table.append(row)
    with (ROOT/f'results/{TAG}_table.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(table)
    parity = sample_summary([r['bf16_fp32_relative_difference'] for r in records])
    write_json_atomic(ROOT/f'results/{TAG}_summary.json', dict(complete=True, examples=100,
        conditions=3000, raw_outputs_verified=True, bf16_fp32_relative_difference=parity, table=table))
    with plt.rc_context({'font.family':'STIXGeneral','pdf.fonttype':42}):
        fig, axes = plt.subplots(2,3,figsize=(9,5.1),constrained_layout=True)
        for ax,k in zip(axes.flat,LENGTHS):
            subset = [r for r in table if r['length']==k]
            mean = np.array([r['output_l2_mean'] for r in subset])
            std = np.array([r['output_l2_std'] for r in subset])
            ax.plot(FACTORS,mean,'o-',color='#237f95',ms=3,label='Scaled single token')
            ax.fill_between(FACTORS,mean-std,mean+std,color='#237f95',alpha=.18)
            control, spread = subset[0]['uncalibrated_l2_mean'],subset[0]['uncalibrated_l2_std']
            ax.axhline(control,color='#cc8241',ls='--',label='Same strength as multi-token')
            ax.fill_between(FACTORS,control-spread,control+spread,color='#cc8241',alpha=.10)
            ax.axvline(1,color='.6',lw=.7,ls=':')
            ax.set(xscale='log',title=f'k = {k}'+(' (full)' if k==128 else ''),
                   xlabel=r'Single-token strength / $\beta$',ylabel='Attention-output L2 error')
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.2)
        axes[0,0].legend(frameon=False,fontsize=7)
        fig.savefig(ROOT/f'figs/{TAG}.pdf',bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(dict(completed=100,conditions=3000,fp32_parity=parity,table=table),indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['extract','run','summarize'],required=True)
    parser.add_argument('--mode',choices=['single-ratio'])
    parser.add_argument('--limit',type=int,default=100)
    args = parser.parse_args()
    if args.phase=='run' and args.mode is None:
        parser.error('run requires an explicit --mode')
    torch.set_num_threads(4)
    with tee_stdout(ROOT/f'logs/{TAG}_{args.phase}.log'), contextlib.redirect_stderr(sys.stdout):
        if args.phase=='extract': extract(args.limit)
        elif args.phase=='run': run(args.limit,args.mode)
        else: summarize()
