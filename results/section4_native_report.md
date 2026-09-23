# Native attention experiments at Qwen3-4B layer 27

## Coverage and numerical audit

- Examples: 100 (5 explicitly expanded inputs).
- Heads: 0 and 16, fitted separately; layer index 26. This is a head-level study, before the output projection.
- Completed token configurations: 4400; query evaluations: 2,636,592.
- Query-drift measurements: 14,400.
- Reference-fit error: median 4.98e-07, maximum 0.177; 117 fits did not reach the configured tolerance.
- Pre-prompt causal controls and residual-aware linear replay certificates passed their numerical checks.
- EOS occurs at or before the first common query in 95/100 traces and at or before the fitting query in 97/100 traces. These fixed-length, post-EOS diagnostics must not be described as ordinary EOS-terminated generation.
- Displacement norm: median 8.85, maximum 1.65e+04. Numerical rank uses relative singular-value tolerance 1e-07.

## Design

The cohort takes all 95 original HarmBench inputs with at least 128 input tokens and the five longest remaining inputs. The latter retain their original requests and receive the explicit neutral introductory context recorded in `data/section4_native_cohort.json`. This selected contextual cohort is not representative of all 400 behaviors. Summary JSON also gives the 95 original-only sensitivity results.

Frozen representations come from BF16 forward passes of the pinned Qwen3-4B revision; attention replay and optimization use float64. A common greedy trace uses the original 8-token instruction suffix and is extended to 256 tokens; EOS positions are recorded, but EOS does not truncate this fixed-length diagnostic. The prompt library contains 128 nested instruction tokens. These are controlled fixed-state calculations, not separately generated trajectories for each condition.

Every evaluation uses Qwen3 QK RMS normalization, RoPE, and an actual causal mask. The prompt arm has its own compact positions for input + prompt + continuation; the steering arm has compact positions for input + continuation. The same unmodified reading representation is used in both arms, with its appropriate position ID. The library is frozen; we do not claim equality of the hidden states that independent full-model runs would produce.

For each example/head/configuration, L-BFGS optimizes a shared displacement r against the native head output at generated position 256. Optimization takes place in the realizable row space of [W_K; W_V] and saves optimizer state after every step. r is then fixed. All prompt positions, steered positions, and the fitting query are excluded from evaluation. The primary curves use the same 128 held-out positions (generated positions 128–255) for every configuration. All other eligible queries are also measured; pre-prompt queries have zero causal exposure and are reported separately rather than diluting the primary curves.

Input steering affects the last b input tokens. Mixed steering always affects one input token plus the first g generated tokens. The steering-count sweep fixes m=4; the prompt-count sweep fixes b=1, g=0. Error is the unnormalized L2 norm of the head-output difference. Average across queries first, then the two heads within each behavior, then report the behavior mean and sample standard deviation.

## Three figures and theory boundary

1. `figs/section4_native_token_error.pdf`: native attention error versus steered and prompt token counts.
2. `figs/section4_native_distance_error.pdf`: equal-norm changes of the raw projected query, within the original linear U-perp, within its orthogonal complement, or random; measured native error versus actual distance rho.
3. `figs/section4_native_subspace.pdf`: dimension of the original linear-theory diagnostic, including controlled redundant versus independent key directions. These controls operate in key space, not on new text.

The original U uses raw linear key differences and one fixed value-null direction z for each example/head. With r numerically fitted to native attention, this U is a diagnostic, not a guaranteed matching subspace. Native QK normalization and RoPE are not covered by the linear matching theorem. We therefore do not label the original theorem's bounds as native-attention upper bounds. The summary separately records errors and residual-aware certificates for the corresponding linear replay, including its nonzero reference mismatch.

Results are empirical; increasing errors or decreasing dimensions are not imposed on the measurements.
