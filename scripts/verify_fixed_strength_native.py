"""Verify frozen native-attention readouts against the captured real model."""
import contextlib
import json
from pathlib import Path
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from prefix.duration_strength import sample_summary
from prefix.runner import tee_stdout, write_json_atomic
from run_fixed_strength_sweep import ROOT, TAG, source_manifest


def verify():
    torch.set_num_threads(4)
    m = source_manifest()
    folder = Path(hf_hub_download(m['model'], 'config.json', revision=m['revision'], local_files_only=True)).parent
    cfg = AutoConfig.from_pretrained(folder, local_files_only=True)
    cfg._attn_implementation = 'sdpa'
    attn = Qwen3Attention(cfg, m['layer_index']).to(dtype=torch.bfloat16)
    prefix = f'model.layers.{m["layer_index"]}.self_attn.'
    weights = {}
    for file in folder.glob('*.safetensors'):
        with safe_open(file, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.startswith(prefix): weights[key[len(prefix):]] = f.get_tensor(key)
    attn.load_state_dict(weights, strict=True)
    attn = attn.cuda().eval()
    rows = []
    for i in range(100):
        saved = torch.load(ROOT/f'checkpoints/{TAG}_states_{i:03d}.pt',weights_only=True,map_location='cuda')
        result = json.loads((ROOT/f'results/{TAG}_{i:03d}.json').read_text())
        captured = []
        d = attn.head_dim
        handle = attn.o_proj.register_forward_pre_hook(
            lambda _m, a: captured.append(a[0][0,-1,m['head']*d:(m['head']+1)*d].detach().float()))
        with torch.inference_mode():
            attn(saved['hidden'][None], saved['embeddings'], None)
        handle.remove()
        delta = captured[0]-saved['native_bf16_output']
        rows.append(dict(index=i, max_absolute_difference=float(delta.abs().max()),
                         l2_difference=float(delta.norm()),
                         fp32_relative_difference=result['bf16_fp32_relative_difference']))
        write_json_atomic(ROOT/f'results/{TAG}_native_verification.json',dict(
            completed=len(rows), rows=rows,
            comparison='BF16 native SDPA module replay vs captured full-model head output'))
    max_error = max(r['max_absolute_difference'] for r in rows)
    assert max_error == 0., f'Native output mismatch: {max_error}'
    report = dict(complete=True, examples=100, bit_exact_bf16_replay=True,
                  maximum_absolute_difference=max_error,
                  bf16_to_fp32_relative_difference=sample_summary([r['fp32_relative_difference'] for r in rows]),
                  rows=rows)
    write_json_atomic(ROOT/f'results/{TAG}_native_verification.json', report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)


if __name__ == '__main__':
    with tee_stdout(ROOT/f'logs/{TAG}_verification.log'), contextlib.redirect_stderr(sys.stdout):
        verify()
