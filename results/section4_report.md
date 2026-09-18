# Qwen3-4B / HarmBench: single- and multi-token steering

Completed **400 HarmBench inputs**, **218,400 main conditions**, and **7,200 redundancy controls** on the local machine.

## The two requested figures

1. [Fixed prompt length: steering length versus actual error and bound](../figs/section4_steering_length.pdf). Prompt length m=4. Compare steering only input positions with steering four input positions plus generated positions.
2. [Fixed single-token steering: prompt length versus actual error](../figs/section4_prompt_length.pdf). k=1, g=0; prompt lengths m=1,2,4,8.

Both use **mean ± sample standard deviation across HarmBench inputs**. For each input and setting, first average its six head errors (three layers × two heads). Heads are not counted as independent input prompts. Shading is standard deviation, not a confidence interval; negative lower edges of mean-minus-std do not represent negative observed errors.

### Fixed prompt length m=4

| Input steered k | Generated steered g | Mean error ± std | Mean bound ± std |
|---:|---:|---:|---:|
| 1 | 0 | 0.4846 ± 0.5588 | 14.7939 ± 1.4801 |
| 2 | 0 | 0.6231 ± 0.6542 | 15.3412 ± 1.3884 |
| 4 | 0 | 0.6885 ± 0.7059 | 15.9384 ± 1.4217 |
| 8 | 0 | 0.7753 ± 0.7723 | 16.3029 ± 1.4078 |
| 4 | 1 | 0.9694 ± 0.9454 | 16.3801 ± 1.3747 |
| 4 | 4 | 1.1141 ± 0.9764 | 17.7160 ± 1.4404 |
| 4 | 8 | 1.2972 ± 1.0188 | 19.0761 ± 1.8044 |
| 4 | 16 | 1.8571 ± 1.1406 | 20.8646 ± 2.2391 |

### Fixed single-token steering k=1

| Prompt length m | Single-token steering error: mean ± std |
|---:|---:|
| 1 | 0.1549 ± 0.2557 |
| 2 | 0.3381 ± 0.5247 |
| 4 | 0.4846 ± 0.5588 |
| 8 | 0.6212 ± 0.5413 |

All other m/k settings are retained in [the comparison CSV](section4_error_comparison.csv). These are descriptive finite-sample comparisons; increasing the number of positions is not mathematically guaranteed to increase error on every input.

## Experimental design

- Model: `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`. BF16 model forward passes and float64 attention calculations. Layers 8,17,26 (0-indexed), query heads 0,16 in each, using their corresponding grouped-query K/V heads.
- Use all 400 records of the pinned HarmBench text CSV, including the supplied context. Inputs are not truncated. Apply the model's chat template with thinking disabled.
- Append the eight-token suffix ` Please respond safely and helpfully. Do`. The prompt-length settings use nested prefixes of 1,2,4,8 tokens. Thus "prompt length" on the second x-axis refers to this appended block, not to the length of the HarmBench question.
- Extract real post-input-layernorm representations from the model. Build a common state library and use its learned **linear** W_Q/W_K/W_V projections before Q/K normalization and RoPE, matching the assumptions of Section 4.
- Reference query q0 comes from the final original input position. For every input, head, and (m,k), construct r using the theorem's effective-value match plus a correction in ker(W_V) that matches aggregate attention weight. Single-token steering modifies the final input position; input-only multi-token steering adds the same r to the last k input positions.
- Keep that r fixed and compare the two attention outputs at the query from the final continuation position. The measured error is `||o_steer - o_prompt||_2`. This measures transfer of the initial match, not optimizer success or the existence of r: both methods match at q0 under the theorem's assumptions.
- For generated-position settings, keep k=4 and extend the same intervention to g=1,4,8,16 generated positions. The corresponding bound includes beta, the reference mismatch of the enlarged block.
- The common trace has 16 greedy decoding steps from the eight-token suffix. EOS does not terminate this fixed-length diagnostic: 27/400 traces encounter EOS before step 16. Shorter prompt settings reuse the same captured generated states. This controls inherited-state and query differences; these are **fixed-state head-level comparisons**, not separate free-generation runs for each setting.
- Each head/example has its own constructed r. The experiment does not establish that one vector simultaneously matches every head, nor that the final model distributions or behaviors are equal.

## Bounds and checks

No extra diagnostic plots are produced. Full per-condition measurements and intermediate bounds remain available locally in `results/section4_000.json` through `results/section4_399.json`.

- Bound-chain violations (`E <= decomposition <= log-weight <= certified drift <= D`): **0**, tolerance `2e-8 * max(1,D)`.
- Original Lemma 2 violations: **0**.
- Rows exceeding the sampled lower endpoint of the path-J lemma formula: **0**. If zero, the original formula is numerically supported through `E <= sampled endpoint <= actual lemma formula <= certified bound`.
- Maximum reference matching error: **2.321e-13**. Maximum rematching-control error: **1.069e-13**.
- Removing beta creates **8,403 violations** in the zero-query-drift generated-position controls.
- The Jacobian coordinates are anchored at **the last input token t_n**, exactly as in the manuscript. A preservation audit checked **166,016** existing measurements while correcting the coordinate anchor; only Jacobian-dependent quantities were updated.

The plotted bound is a conservative evaluation of the relevant lemma. The infinity-to-2 Jacobian norm is exact by sign enumeration for at most eight coordinates; larger blocks use a column-norm upper bound and deterministic sign probes as a lower estimate. Nine path points are sampled. A valid between-point envelope adds `L/(2*(points-1))`, with `L = 2*(range(delta_P)*diameter(V_P) + range(delta_M)*diameter(V_M))`, and is capped by the sum of the two block diameters. The single-token, single-prompt Jacobian maximum is computed analytically from the sigmoid derivative. Finite sampling alone is not claimed to be an upper bound.

The additional zero-drift, rematching, query-direction, redundant-activation, and native-Q/K-normalization/RoPE controls are saved as numerical results. The native architecture control is outside the linear-key assumptions and is excluded from lemma-violation counts.

## Reproduce or resume

```bash
uv run python scripts/run_section4_bounds.py extract
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/run_section4_bounds.py analyze
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/run_section4_bounds.py analyze --suite redundancy
uv run python scripts/plot_section4_bounds.py
uv run pytest -q tests/test_attention_bounds.py tests/test_run_section4_bounds.py tests/test_plot_section4_bounds.py
```

Independent CPU workers can use `--start 0 --stride 8` through `--start 7 --stride 8`. Decoding checkpoints are saved after each token. Replay conditions are appended to durable JSONL journals and consolidated into per-input JSON; interrupted runs skip completed units. Logs are under `logs/`, activation checkpoints under `checkpoints/`, and PDFs under `figs/`. **Email notifications have been removed from this experiment at the user's request.**

The report, summary, comparison CSV, manifest, and two PDF figures are retained in Git. Large activation checkpoints and per-condition metric files remain local.
