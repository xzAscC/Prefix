# Layer 27 dimension rerun with 100 dataset prompts

Panel (a) of `notebooks/section4_layer27_summary.ipynb` now reads `section4_prompt_diversity_summary.json`. Panels (b) and (c) retain their previous data and protocols.

## Design

- Qwen3-4B, pinned revision in the manifest, layer index 26; heads 0 and 16.
- The same 100 selected HarmBench records supply 100 different appended prompts. Each prompt places the original request before its context. Seven short texts receive the exact supplementary neutral text recorded in `data/section4_prompt_diversity_manifest.json`.
- Original inputs are cyclically paired with the next record's prompt, avoiding self-pairing. This is 100 paired input/prompt examples, not a 100-by-100 crossed design.
- Distinct token prefixes at lengths 1, 2, 4, 8, 16, 32, 64, 128: **22, 33, 61, 82, 92, 100, 100, 100**. Full 128-token prompt libraries are all different; short prefixes can coincide.
- Newly captured prompt states come from BF16 model forwards. Original input/generated states and reference queries remain frozen. The original linear-theory directions and numerical rank are computed in float64; this is a dimension diagnostic, not a native-attention matching guarantee.
- The displayed sweeps cover 15 unique configurations, 100 examples, and two heads: **3,000 completed conditions**. The shared baseline appears in both displayed curves but is computed once per example/head.

## Result

Every configuration has sample std **0**, including across its 200 individual head-level records. Prompt-length dimensions are **126, 125, 123, 119, 111, 95, 63, 0**. Mixed-steering dimensions are **123, 122, 120, 116, 108, 92, 60, 0**.

These are fresh singular-value computations on the varied prompt states, not dimensions assigned from token counts. The measured constraint ranks coincide across these examples, so different vectors can have identical integer dimensions. This does not establish orthogonality or imply that actual attention errors have zero variance.

Relative rank thresholds 1e-5, 1e-6, 1e-7, and 1e-8 give the same means and standard deviations here. The primary threshold remains 1e-7. Per-condition checkpoints retain singular values and dimensions at every threshold; the aggregate sensitivity audit is included separately.

## Execution and reproduction

Two examples were tested locally first. The remote GPU attempt spent 12 minutes 29 seconds loading the environment and was cancelled without new extraction results; its allocation was released. Following the user's request, all 100 prompt libraries were extracted on the local RTX 4090 and all ranks were computed on the local CPU. The 60 completed smoke-test results were preserved byte-for-byte. Six focused tests passed, including both notebook launch directories and rank behavior on independent versus redundant directions. The notebook was executed and its PDF/SVG rendering inspected. No email notifications were sent.

With the existing native-run cohort, dataset, model cache, weights, and frozen activation libraries:

```bash
uv run --no-sync python scripts/run_section4_prompt_diversity.py prepare
uv run --no-sync python scripts/run_section4_prompt_diversity.py extract --device cuda
uv run --no-sync python scripts/run_section4_prompt_diversity.py analyze --device cpu
```

Then execute `notebooks/section4_layer27_summary.ipynb` to save `figs/section4_layer27_summary.pdf` and display its SVG preview. Extraction resumes per example; dimension calculations checkpoint per example/head/configuration. Completed compatible units are reused and manifest mismatches are rejected.
