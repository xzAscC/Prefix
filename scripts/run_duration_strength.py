"""Run Lemma 5's native output diagnostic on Olmo 3 7B and 100 HarmBench cases.

uv run python scripts/run_duration_strength.py --limit 100
The default 128-token diagnostic continues after EOS, recording first EOS.
This is an independent-generation measurement, not fixed-state theorem replay.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import numpy as np
import torch

from prefix.data import load_llm_lat, llm_lat_revision, HARMBENCH_URL
from prefix.duration_strength import calibrated_strength, decode, sample_summary, select_cohort
from prefix.runner import tee_stdout, write_json_atomic
from prefix.steering import dim_direction, mean_hidden_norm

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'allenai/Olmo-3-7B-Think'
REVISION = 'd97e442d7cc678210054dbcc9b440894d62c89a4'
LENGTHS = [4, 8, 16, 32, 64, 128]


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def prompt_ids(tokenizer, text):
    return tokenizer.apply_chat_template([{'role': 'user', 'content': text}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)


def prepare_data(tag):
    path = ROOT / f'data/{tag}.json'
    if path.exists():
        return json.loads(path.read_text())
    # Legacy project caches omit BehaviorID and ContextString. Preserve both by
    # reading the pinned upstream CSV into this experiment's separate cache.
    source_path = ROOT/f'data/{tag}_source.csv'
    if not source_path.exists():
        with urlopen(HARMBENCH_URL, timeout=60) as response:
            source_path.write_bytes(response.read())
    source = source_path.read_text()
    rows = list(csv.DictReader(io.StringIO(source)))
    if len(rows) != 400:
        raise ValueError('expected 400 source behaviors')
    cohort = select_cohort(rows)
    datasets = ['LLM-LAT/benign-dataset', 'LLM-LAT/harmful-dataset']
    data = dict(cohort=cohort, harmbench_source=HARMBENCH_URL, seed=42,
                source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                positive=load_llm_lat(datasets[0], 100),
                negative=load_llm_lat(datasets[1], 100),
                direction_revisions={d:llm_lat_revision(d) for d in datasets})
    write_json_atomic(path, data)
    print('Saved 100-example cohort and 100+100 direction-training prompts', flush=True)
    return data


def build_direction(model, tokenizer, data, layer, tag, identity):
    path = ROOT / f'checkpoints/{tag}_direction.json'
    state = json.loads(path.read_text()) if path.exists() else dict(identity=identity, hiddens=[])
    if state['identity'] != identity:
        raise ValueError('direction identity mismatch')
    texts = data['positive'] + data['negative']
    captured = []
    def capture(_module, args, kwargs):
        h = kwargs.get('hidden_states', args[0] if args else None)
        captured[:] = [h[0,-1].detach().float().cpu()]
    handle = model.model.layers[layer].self_attn.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        for i in range(len(state['hiddens']), len(texts)):
            ids = prompt_ids(tokenizer, texts[i])
            with torch.inference_mode():
                model(torch.tensor([ids], device='cuda'), use_cache=False, logits_to_keep=1)
            state['hiddens'].append(captured[0].tolist())
            write_json_atomic(path, state)
            if (i+1) % 20 == 0:
                print(f'Direction activations {i+1}/{len(texts)}', flush=True)
    finally:
        handle.remove()
    h = torch.tensor(state['hiddens'])
    direction = dim_direction(h[:100], h[100:])
    norm = mean_hidden_norm(h)
    state.update(direction=direction.tolist(), mean_norm=norm, complete=True,
                 definition='unit mean(benign)-mean(harmful); native normalized attention input')
    write_json_atomic(path, state)
    return direction * norm, state


def summarize(tag, manifest):
    rows = []
    for i in range(100):
        path = ROOT / f'results/{tag}_{i:03d}.json'
        if path.exists():
            row = json.loads(path.read_text())
            if row['manifest'] != manifest:
                raise ValueError('result manifest mismatch')
            rows.append(row)
    summary = dict(manifest=manifest, completed=len(rows), expected=100,
                   complete=len(rows)==100, conditions=[])
    for k in LENGTHS:
        values = [next(c for c in r['conditions'] if c['length']==k) for r in rows]
        if not values:
            continue
        summary['conditions'].append(dict(length=k, **{key:sample_summary([v[key] for v in values])
            for key in ['output_l2', 'relative_l2', 'uncalibrated_l2', 'beta',
                        'trajectory_mean_l2', 'distributed_norm', 'baseline_to_distributed_l2']},
            improved_count=sum(v['output_l2'] < v['uncalibrated_l2'] for v in values)))
    write_json_atomic(ROOT/f'results/{tag}_summary.json', summary)
    return summary


def run(args):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data = prepare_data(args.tag)
    manifest = dict(version=1, model=MODEL, revision=REVISION, layer_index=args.layer,
        head=args.head, alpha=args.alpha, lengths=LENGTHS, target=128, n_examples=100,
        seed=42, data_sha256=digest(data), torch=torch.__version__, transformers=transformers.__version__,
        dtype='bfloat16', decoding='greedy, 128 tokens regardless of EOS; first EOS recorded',
        intervention='attention input before native QK normalization and RoPE',
        positions='last prompt token plus k-1 generated-token positions',
        output='selected head before o_proj at query predicting generated token 128',
        calibration='per-example oracle: unsteered step-128 attention mass; beta=alpha*mass(S)/mass(j)',
        comparison='independent native generation; changing queries and representations',
        std='sample standard deviation across 100 example-wise L2 distances',
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), ROOT/'src/prefix/duration_strength.py']})
    identity = dict(model=MODEL, revision=REVISION, layer=args.layer, data_sha256=digest(data))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    direction, direction_record = build_direction(model, tokenizer, data, args.layer, args.tag, identity)
    manifest['direction_sha256'] = digest(direction_record)
    write_json_atomic(ROOT/f'results/{args.tag}_manifest.json', manifest)
    print(f'Model loaded; layer index={args.layer}, head={args.head}, alpha={args.alpha}', flush=True)
    for i, example in enumerate(data['cohort'][:args.limit]):
        result_path = ROOT/f'results/{args.tag}_{i:03d}.json'
        if result_path.exists():
            if json.loads(result_path.read_text())['manifest'] != manifest:
                raise ValueError('completed example manifest mismatch')
            print(f'Example {i+1}/100 already complete; skipping', flush=True)
            continue
        start = time.monotonic()
        ids = prompt_ids(tokenizer, example['text'])
        local = dict(manifest=manifest, example=example)
        def progress(state):
            t = len(state['tokens'][0])
            if t % 32 == 0:
                print(f'Example {i+1}/100: {len(state["tokens"])} arms, token {t}/128', flush=True)
        baseline = decode(model, ids, direction, [dict(id='unsteered', length=0, strength=0.)],
            args.layer, args.head, ROOT/f'checkpoints/{args.tag}_{i:03d}_baseline.json',
            local, on_step=progress, capture_attention=True)
        weights = baseline['attention'][0]
        arms = []
        for k in LENGTHS:
            beta = calibrated_strength(weights, len(ids), k, args.alpha)
            arms.extend([dict(id=f'distributed_{k}', length=k, strength=args.alpha),
                         dict(id=f'single_for_{k}', length=1, strength=beta)])
        arms.append(dict(id='single_uncalibrated', length=1, strength=args.alpha))
        state = decode(model, ids, direction, arms, args.layer, args.head,
            ROOT/f'checkpoints/{args.tag}_{i:03d}_arms.json', local, on_step=progress)
        outputs = np.asarray(state['outputs'], dtype=np.float64)
        conditions = []
        for j, k in enumerate(LENGTHS):
            target, actual = outputs[2*j:2*j+2]
            error = float(np.linalg.norm(target-actual))
            norm = float(np.linalg.norm(target))
            conditions.append(dict(length=k, beta=arms[2*j+1]['strength'], output_l2=error,
                relative_l2=error/max(norm, 1e-30), distributed_norm=norm,
                uncalibrated_l2=float(np.linalg.norm(target-outputs[-1])),
                baseline_to_distributed_l2=float(np.linalg.norm(target-baseline['outputs'][0])),
                trajectory_mean_l2=float(np.asarray(state['trajectory_l2'])[:,j].mean())))
        result = dict(manifest=manifest, index=i, example=example, prompt_ids=ids,
            conditions=conditions, arms=arms, final_outputs=state['outputs'],
            generated_tokens=state['tokens'], generated_texts=tokenizer.batch_decode(state['tokens']),
            first_eos=state['first_eos'], continued_after_eos=[v is not None and v<128 for v in state['first_eos']],
            baseline_tokens=baseline['tokens'][0], baseline_output=baseline['outputs'][0],
            calibration_attention=weights, trajectory_l2=state['trajectory_l2'],
            elapsed_seconds=time.monotonic()-start)
        write_json_atomic(result_path, result)
        summary = summarize(args.tag, manifest)
        print(f'Completed {summary["completed"]}/100; example took {result["elapsed_seconds"]:.1f}s', flush=True)
    summary = summarize(args.tag, manifest)
    print(json.dumps(summary['conditions'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--layer', type=int, default=17, help='zero-based; default is displayed layer 18')
    parser.add_argument('--head', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--tag', default='lemma5_olmo3_7b_native')
    args = parser.parse_args()
    if not 1 <= args.limit <= 100 or args.alpha <= 0:
        parser.error('limit must be 1..100 and alpha positive')
    with tee_stdout(ROOT/f'logs/{args.tag}.log'), contextlib.redirect_stderr(sys.stdout):
        run(args)


if __name__ == '__main__':
    main()
