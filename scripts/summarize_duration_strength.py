"""Audit saved native head outputs, then export the mean/std table and PDF."""
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

from prefix.duration_strength import calibrated_strength, sample_summary
from prefix.runner import tee_stdout, write_json_atomic
from run_duration_strength import ROOT, LENGTHS, summarize


def audit(tag):
    manifest = json.loads((ROOT/f'results/{tag}_manifest.json').read_text())
    records = [json.loads((ROOT/f'results/{tag}_{i:03d}.json').read_text()) for i in range(100)]
    assert len({r['example']['behavior_id'] for r in records}) == 100
    for i, row in enumerate(records):
        assert row['index'] == i and row['manifest'] == manifest
        assert len(row['generated_tokens']) == 13
        assert all(len(t)==128 for t in row['generated_tokens'])
        assert len(row['baseline_tokens']) == 128
        outputs = np.asarray(row['final_outputs'], dtype=np.float64)
        assert outputs.shape == (13, 128) and np.isfinite(outputs).all()
        assert len(row['calibration_attention']) == len(row['prompt_ids']) + 127
        assert len(row['trajectory_l2']) == 128
        for j, k in enumerate(LENGTHS):
            c = row['conditions'][j]
            actual = np.linalg.norm(outputs[2*j]-outputs[2*j+1])
            np.testing.assert_allclose(c['output_l2'], actual, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(row['trajectory_l2'][-1][j], actual, rtol=1e-6, atol=1e-7)
            beta = calibrated_strength(row['calibration_attention'], len(row['prompt_ids']), k, manifest['alpha'])
            np.testing.assert_allclose(c['beta'], beta, rtol=1e-12)
    summary = summarize(tag, manifest)
    assert summary['complete']
    table = []
    for j, k in enumerate(LENGTHS):
        out = dict(length=k)
        for metric in ['output_l2', 'relative_l2', 'uncalibrated_l2', 'beta', 'trajectory_mean_l2']:
            stats = sample_summary([r['conditions'][j][metric] for r in records])
            for name in ['mean', 'std']:
                out[f'{metric}_{name}'] = stats[name]
        out['improved_count'] = sum(r['conditions'][j]['output_l2'] < r['conditions'][j]['uncalibrated_l2'] for r in records)
        out['distributed_early_eos_count'] = sum(r['continued_after_eos'][2*j] for r in records)
        out['single_early_eos_count'] = sum(r['continued_after_eos'][2*j+1] for r in records)
        out['identical_generated_context_count'] = sum(
            r['generated_tokens'][2*j][:127] == r['generated_tokens'][2*j+1][:127]
            for r in records)
        pre_eos = []
        for row in records:
            stop = min(row['first_eos'][2*j] or 128, row['first_eos'][2*j+1] or 128)
            pre_eos.append(float(np.asarray(row['trajectory_l2'])[:stop,j].mean()))
        stats = sample_summary(pre_eos)
        out['pre_eos_trajectory_l2_mean'] = stats['mean']
        out['pre_eos_trajectory_l2_std'] = stats['std']
        table.append(out)
    with (ROOT/f'results/{tag}_table.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    write_json_atomic(ROOT/f'results/{tag}_audit.json', dict(
        complete=True, examples=100, conditions=600, native_trajectories=1400,
        generated_tokens=1400*128, raw_output_recomputation_passed=True,
        unique_behavior_ids=True, lengths=LENGTHS, table=table))
    with plt.rc_context({'font.family':'STIXGeneral', 'pdf.fonttype':42}):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
        x = np.arange(6)
        for metric, label, color, offset in [
            ('output_l2','Calibrated single token','#237F95',-.06),
            ('uncalibrated_l2','Uncalibrated single token','#CC8241',.06)]:
            axes[0].errorbar(x+offset, [t[f'{metric}_mean'] for t in table],
                yerr=[t[f'{metric}_std'] for t in table], fmt='o-', lw=1.1,
                markersize=3.5, capsize=2, color=color, label=label)
        axes[1].errorbar(x, [t['relative_l2_mean'] for t in table],
            yerr=[t['relative_l2_std'] for t in table], fmt='o-', lw=1.1,
            markersize=3.5, capsize=2, color='#237F95')
        axes[0].set_ylabel(r'$\|o_{S,\alpha}-o_{j,\beta}\|_2$')
        axes[1].set_ylabel(r'Relative output difference')
        axes[0].legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.02))
        for ax in axes:
            ax.set_xticks(x, ['4','8','16','32','64','128\n(full)'])
            ax.set_xlabel('Number of distributed steering positions')
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y', alpha=.2)
        fig.savefig(ROOT/f'figs/{tag}.pdf', bbox_inches='tight')
        plt.close(fig)
    lines = [
        '# Native duration-strength experiment', '',
        'Completed 100 unique HarmBench examples, six duration comparisons per example.', '',
        f'Model: `{manifest["model"]}` at revision `{manifest["revision"]}`. '
        f'Layer index {manifest["layer_index"]} (displayed layer {manifest["layer_index"]+1}), '
        f'head {manifest["head"]}, BF16, greedy generation.', '',
        'The direction is the unit difference of 100 benign and 100 harmful '
        'last-prompt-token normalized attention inputs (LLM-LAT), scaled by their mean norm. '
        'Distributed strength alpha=0.1. Steering starts at the last prompt token; '
        'length k includes this position plus k-1 generated-token positions.', '',
        'For each example, the unsteered query predicting token 128 supplies the native '
        'attention weights for beta=alpha*mass(S)/mass(j). This is per-example oracle '
        'calibration using a complete reference trajectory, not a deployable shared strength.', '',
        'Each arm then generates independently. Output o is captured directly before '
        'o_proj at the query predicting token 128. Mean and sample std (ddof=1) are '
        'computed over 100 example-wise L2 distances. Relative error divides by '
        'the distributed-arm output norm. Native QK normalization and RoPE remain active.', '',
        '| k | Output L2 mean ± std | Relative L2 mean ± std | Uncalibrated L2 mean ± std | Improved / 100 |',
        '|---|---|---|---|---|',
    ]
    for t in table:
        lines.append(f'| {t["length"]} | {t["output_l2_mean"]:.6f} ± {t["output_l2_std"]:.6f} '
            f'| {t["relative_l2_mean"]:.4f} ± {t["relative_l2_std"]:.4f} '
            f'| {t["uncalibrated_l2_mean"]:.6f} ± {t["uncalibrated_l2_std"]:.6f} '
            f'| {t["improved_count"]} |')
    lines += ['', 'All arms produce exactly 128 tokens. EOS is recorded but does not stop '
        'this fixed-length diagnostic. Later outputs may therefore be post-EOS continuations. '
        'Early-EOS and identical-generated-context counts are in the CSV/audit. The differences combine direct steering '
        'effects with differing generated tokens, queries, and representations; they are '
        'not a fixed-query validation of the lemma and do not measure refusal rates. '
        'The CSV also gives per-example time-averaged errors over all 128 queries and '
        'over the aligned prefix ending at the first EOS in either compared arm, '
        'with sample std across examples.', '',
        'Raw tokens, decoded text, final head vectors, calibration attention weights, '
        'per-step differences, provenance, and per-token resume states are stored locally '
        'under the same experiment prefix. No synthetic outputs are used.', '',
        f'Reproduce: `uv run python scripts/run_duration_strength.py --tag {tag}` followed by '
        f'`uv run python scripts/summarize_duration_strength.py --tag {tag}`.',
    ]
    (ROOT/f'results/{tag}_report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='lemma5_qwen3_native')
    args = parser.parse_args()
    with tee_stdout(ROOT/f'logs/{args.tag}_audit.log'), contextlib.redirect_stderr(sys.stdout):
        audit(args.tag)
