# Lemma 8: prefix versus longer steering

400/400 complete examples; 181000 eligible head/condition measurements; zero violations.

Both schedules share the query, original keys/values and the same scaled displacement. The long support contains the short support. Replays use the Section 4 linear-head convention before QK normalization and RoPE, at the query predicting token 128; no independent-generation claim follows.

The shared continuation is already past EOS in 391/400 examples at this prediction step; these are fixed-state diagnostics, not natural response-end measurements.

The first bound uses the diameter of ALL short-schedule values, including modified and unmodified positions. Attention shares are normalized over ALL visible tokens. The score bound uses the absolute added score, not query drift. The same displacement is constructed at the last input token and is reused when support changes.

Y axes use a symmetric-log scale with a linear region below 0.1 to retain exact-zero controls. Figures show mean ± sample standard deviation after averaging heads (and layers in the mean figure) within each behavior. Input-support comparisons use a fixed cohort with at least 64 input tokens. The conditional table excludes zero-strength and empty-extra-set cases.

| ε (both shares ≤ ε) | Eligible conditions | Behaviors | Mean difference | Mean ε bound | Violations |
|---|---:|---:|---:|---:|---:|
| 0.001 | 22968 | 400 | 0.001097453775411629 | 0.1560176562186952 | 0 |
| 0.01 | 29138 | 400 | 0.011542895865136021 | 1.3328163612990267 | 0 |
| 0.05 | 40247 | 400 | 0.05340519883726578 | 5.2194628683368 | 0 |
| 0.1 | 45547 | 400 | 0.10563596779538306 | 9.588893260014148 | 0 |

## Main schedule comparisons (m=8, α=1)

| Short support → long support | Behaviors | Mean difference | Mean share bound | Mean score bound |
|---|---:|---:|---:|---:|
| All input → all input + 127 generated | 400 | 5.62131 | 12.9456 | 120.113 |
| 1 input → all input + 127 generated | 400 | 12.4561 | 14.0905 | 118.953 |
| 1 input → same input + 127 generated | 400 | 8.70176 | 15.1918 | 113.464 |

All bounds are checked with tolerance 1e-8 × max(1, D_short, value-shift norm). Token count alone does not imply a monotone difference; value and weight contributions can cancel.
