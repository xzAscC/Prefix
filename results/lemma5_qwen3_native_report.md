# Native duration-strength experiment

Completed 100 unique HarmBench examples, six duration comparisons per example.

Model: `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`. Layer index 17 (displayed layer 18), head 0, BF16, greedy generation.

The direction is the unit difference of 100 benign and 100 harmful last-prompt-token normalized attention inputs (LLM-LAT), scaled by their mean norm. Distributed strength alpha=0.1. Steering starts at the last prompt token; length k includes this position plus k-1 generated-token positions.

For each example, the unsteered query predicting token 128 supplies the native attention weights for beta=alpha*mass(S)/mass(j). This is per-example oracle calibration using a complete reference trajectory, not a deployable shared strength.

Each arm then generates independently. Output o is captured directly before o_proj at the query predicting token 128. Mean and sample std (ddof=1) are computed over 100 example-wise L2 distances. Relative error divides by the distributed-arm output norm. Native QK normalization and RoPE remain active.

| k | Output L2 mean ± std | Relative L2 mean ± std | Uncalibrated L2 mean ± std | Improved / 100 |
|---|---|---|---|---|
| 4 | 0.606256 ± 0.676862 | 0.7904 ± 0.7894 | 0.423940 ± 0.589612 | 41 |
| 8 | 0.644973 ± 0.776433 | 0.7111 ± 0.6023 | 0.441955 ± 0.617439 | 44 |
| 16 | 0.720934 ± 0.791750 | 0.9192 ± 0.8345 | 0.473842 ± 0.593630 | 41 |
| 32 | 0.718798 ± 0.754196 | 1.0048 ± 0.9083 | 0.553544 ± 0.611671 | 45 |
| 64 | 0.840181 ± 0.891051 | 1.1832 ± 1.3394 | 0.605375 ± 0.598715 | 40 |
| 128 | 0.686140 ± 0.641210 | 0.9839 ± 0.8684 | 0.606593 ± 0.575747 | 42 |

All arms produce exactly 128 tokens. EOS is recorded but does not stop this fixed-length diagnostic. Later outputs may therefore be post-EOS continuations. Early-EOS and identical-generated-context counts are in the CSV/audit. The differences combine direct steering effects with differing generated tokens, queries, and representations; they are not a fixed-query validation of the lemma and do not measure refusal rates. The CSV also gives per-example time-averaged errors over all 128 queries and over the aligned prefix ending at the first EOS in either compared arm, with sample std across examples.

Raw tokens, decoded text, final head vectors, calibration attention weights, per-step differences, provenance, and per-token resume states are stored locally under the same experiment prefix. No synthetic outputs are used.

Reproduce: `uv run python scripts/run_duration_strength.py --tag lemma5_qwen3_native` followed by `uv run python scripts/summarize_duration_strength.py --tag lemma5_qwen3_native`.
