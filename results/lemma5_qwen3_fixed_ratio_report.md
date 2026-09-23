# Fixed-query native attention strength sweep

Completed 100 unique HarmBench examples and 3,000 conditions (six durations times five strength ratios).

Qwen3-4B, pinned revision 1cfa9a7208912126459214e8b04321603b3df60c, displayed layer 18 (index 17), head 0. The cohort, safety direction and unsteered 128-token continuations are the same as the preceding native-generation study.

For each example, the original model is run on the prompt plus its 128 saved generated tokens. The normalized attention inputs and rotary embeddings are captured. The common readout is at the final generated token, after all steered positions. This forward reads the existing context; it does not generate a 129th token. For k positions, steering covers the last prompt token and k-1 generated-token positions, so the query representation is never directly steered, even for k=128.

The distributed strength stays at 0.1. Unsteered native attention weights at this query determine beta=0.1*mass(S)/mass(j). The single-token arm uses c*beta for c in {0.001, 0.01, 0.1, 1, 10}. A separate single-token arm at strength 0.1 is the uncalibrated control. All arms share the same pre-intervention representations and query.

Outputs are captured before o_proj from the actual Transformers Qwen3Attention module, retaining QK normalization, RoPE and the causal mask. Local calculations use FP32 to retain small perturbations; measured representations and original BF16 weights are upcast without refitting or synthetic output construction. A separate BF16 native-module replay reproduces the full-model captured head output bit-for-bit for all 100 examples (maximum absolute difference 0). FP32 unsteered outputs differ from BF16 outputs by mean relative L2 0.02210 (sample std 0.01431); precision is therefore part of this experiment definition.

All means and sample standard deviations (ddof=1) are computed across the same 100 examples. Every one of the 3,000 reported L2 errors was verified from its saved raw vectors.

## Results at c=1

| k | L2 mean ± std | Uncalibrated L2 mean ± std | Mean error reduction | Improved examples / 100 |
|---|---|---|---|---|
| 4 | 0.002327 ± 0.005783 | 0.003155 ± 0.006101 | 26.25% | 73 |
| 8 | 0.004679 ± 0.008137 | 0.005919 ± 0.009350 | 20.95% | 64 |
| 16 | 0.009967 ± 0.015033 | 0.011274 ± 0.015310 | 11.59% | 50 |
| 32 | 0.019342 ± 0.023729 | 0.020451 ± 0.023440 | 5.42% | 38 |
| 64 | 0.042197 ± 0.044616 | 0.043681 ± 0.043690 | 3.40% | 46 |
| 128 | 0.131826 ± 0.096355 | 0.133818 ± 0.097285 | 1.49% | 49 |

Among the five tested ratios, c=1 has the lowest mean error at each duration. Mean improvement over uncalibrated single-token steering decreases from 26.25% at k=4 to 1.49% at k=128. Improvement is not uniform across examples. The remaining error can still be large relative to the target intervention effect (mean error/effect rises from 0.482 at k=4 to 0.959 at k=128). These results support a setting-dependent local reduction in error, not near-exact equivalence across durations.

The study freezes states rather than running independent steered generation. The native model includes QK normalization and RoPE, beyond the simplified theoretical attention model; the experiment is not an exact theorem verification. Saved reference continuations may include post-EOS tokens, as recorded in the preceding study. No safety success rate or capability-preservation claim is measured here.

## Reproduce

```bash
uv run python scripts/run_fixed_strength_sweep.py --phase extract
uv run python scripts/run_fixed_strength_sweep.py --phase run --mode single-ratio
uv run python scripts/run_fixed_strength_sweep.py --phase summarize
uv run python scripts/verify_fixed_strength_native.py
```

Extraction states and per-condition output vectors are saved locally with the lemma5_qwen3_fixed_ratio prefix in checkpoints/ and results/. Extraction and sweep phases resume completed work. Aggregate data, the six-panel PDF, provenance and native-output verification are committed.
