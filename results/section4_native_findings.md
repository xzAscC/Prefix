# Observed results: 100-example native attention run

The three PDF figures use Qwen3-4B layer 27, heads 0 and 16 fitted separately, with native QK normalization, RoPE, and causal masking on frozen representations. They are head-output experiments, not full-model generation or behavioral evaluations. See `section4_native_report.md` for the protocol and `section4_native_summary.json` for all numerical aggregates.

| Sweep | Mean held-out L2 error, first → last configuration |
| --- | --- |
| 1 → 128 steered input tokens, 4 prompt tokens | 0.1244 → 0.0754 |
| 1 input → 1 input + 127 generated tokens, 4 prompt tokens | 0.1244 → 0.0992 |
| 1 → 128 prompt tokens, one steered input token | 0.0420 → 0.2065 |

Steering-count errors are not monotonically increasing. Prompt-count errors increase overall, with a decrease at 8 tokens. The original 95 inputs without added context give corresponding endpoint pairs of 0.1267 → 0.0769, 0.1267 → 0.1015, and 0.0431 → 0.2094. These are descriptive comparisons, not significance tests.

At the largest query perturbation, directions inside the original linear U-perp have mean native errors of 0.0763, 0.2284, 0.1792, and 0.1992 across the four tested configurations, despite distances rho at numerical zero. Thus this diagnostic subspace does not guarantee native matching. The original linear dimensions decrease to zero with increasing independent keys; the redundant-key control retains dimension 126.

All 4,400 conditions and 14,400 drift units completed, with 2,636,592 held-out query measurements. Of the reference fits, 117/4,400 did not reach the 1e-6 tolerance; the median residual is 4.98e-7 and the maximum is 0.1774. All fits remain in the aggregates. The median displacement norm is 8.85 and the maximum is 16,511.75; unconstrained optimization can produce very large displacements. The results do not establish an optimum or a uniform small-perturbation regime.

The fixed-length traces contain 12,368/12,800 common positions at or after EOS. This is a substantial limitation: the curves characterize fixed-state attention replay and must not be presented as ordinary EOS-terminated generation. The common queries, token selection, and five explicit input expansions are recorded in the cohort manifest.

Validation includes 21 focused tests, full coverage and causal-control checks, linear replay certificate checks, comparison against the actual pretrained Qwen3 attention module, and a CPU/CUDA replay comparison (maximum component difference 3.56e-15). All 621 results completed before remote migration were preserved byte-for-byte. The remote jobs have ended and their allocations have been released; no email notifications were used for the final run.

## Reproduction and resume

Preparation reuses the existing Section 4 activation caches, weights, dataset, and long-run manifest. Model weights and those activation caches are not included in these aggregate artifacts. With these prerequisites and the project's uv environment:

```bash
uv run --no-sync python scripts/run_section4_native.py prepare
uv run --no-sync python scripts/run_section4_native.py extract
uv run --no-sync python scripts/run_section4_native.py analyze --device cpu --workers 8
uv run --no-sync python scripts/plot_section4_native.py
```

Extraction requires a GPU by default; analysis also supports CUDA. Completed conditions are skipped on restart, unfinished fits resume their saved optimizer state, and every drift condition is checkpointed. Run plotting only after full coverage is available. The figures show means and sample standard deviations across behaviors; standard-deviation bands crossing zero extend below the visible logarithmic plotting range.
