# Long-token steering comparison by layer

Qwen3-4B, 400 HarmBench inputs; layers 3, 9, 18, 27, 34 (one-based). Two query heads (0 and 16) per layer. Each layer is reported separately: average the two head measurements within an input, then compute mean and sample standard deviation across inputs. Shading is standard deviation, not confidence intervals. Negative mean-minus-std values are not negative observed errors.

## Figures

- Layer 3: [combined figure](../figs/section4_layer03_combined.pdf).
- Layer 9: [combined figure](../figs/section4_layer09_combined.pdf).
- Layer 18: [combined figure](../figs/section4_layer18_combined.pdf).
- Layer 27: [combined figure](../figs/section4_layer27_combined.pdf).
- Layer 34: [combined figure](../figs/section4_layer34_combined.pdf).
- Five-layer mean: [combined figure](../figs/section4_layer_mean_combined.pdf).

Each layer has one two-panel PDF: the left panel overlays measured error (solid) and the certified theoretical upper bound (dashed); the right panel shows prompt-length error in purple. A sixth PDF averages the five layers. Open [the plotting notebook](../notebooks/section4_attention_bounds.ipynb) to adjust and regenerate the figures from the tracked summary without rerunning the model. The x-axis uses base-2 logarithmic spacing with explicitly labeled token counts 1, 2, 4, 8, 16, 32, 64, 128.

- Fixed appended prompt length m=4: input-only steering uses the last k input positions. Mixed steering uses four input positions plus g generated positions, with total k+g=4,8,16,32,64,128. Both curves use the **same 95 inputs with at least 128 input tokens**, including the model chat template. No short inputs are padded or silently clamped. The fixed subset's functional categories are {'contextual': 95}; its conclusions do not automatically generalize to the full dataset.
- Fixed single-token steering k=1: vary the appended prompt length m over the eight token counts, using all 400 inputs. This x-axis is the appended instruction block length, not the HarmBench question length or number of input examples.

## Design and interpretation

Use the same pinned Qwen3-4B revision and HarmBench records as [the initial study](section4_report.md). Capture post-input-layernorm representations using BF16 forward passes, then replay each head's learned linear W_Q/W_K/W_V in float64, before Q/K normalization and RoPE. These are conditional attention-level comparisons under the lemma assumptions, not full-model free-generation steering interventions.

The reference q0 is from the last original input token. Construct a separate r for each input, layer, head, prompt length, and input steering count: match the effective value, then adjust aggregate attention mass in the null space of W_V. The same r is added to every selected position. Mixed input/generated conditions retain the r constructed from the four input positions; their bounds include the enlarged block's reference mismatch beta. Existence at q0 is imposed by construction; this experiment tests transfer error, not how easily an optimizer finds r or whether one vector matches all heads.

Extend the original eight-token prompt's common greedy trace from 16 to **128 steps**, preserving its first 16 tokens and checkpointing each additional token. All long-study conditions use the query at the final continuation position. Therefore even the short-count points are recomputed consistently for this target query; they must not be spliced with the initial study's 16-step-query results. EOS does not stop this diagnostic; 391/400 traces encounter EOS before the final step.

Long prompt blocks are nested prefixes of one extended safety instruction. Their causal representations come from the original input plus 128 prompt tokens. Generated states come from the common eight-token-prompt trace. Combining these into a fixed-state library holds the query and generated representations constant across m; it is not a claim that separate prompts produce identical states in the full architecture. Every layer shares input identities, selected positions, and trace token IDs.

Actual error is ||o_steer - o_prompt||_2, without division by the output norm, head dimension, or diameter. Softmax and the model input layer normalization are part of constructing the attention output, not a normalization of this error metric. The five-layer mean first averages the ten head-level scalar errors (five layers × two heads) for each input, then computes mean ± sample std across inputs; it does not average the five standard deviations or take the norm after averaging vectors. Bounds are averaged identically. Layers with larger raw error magnitudes contribute more to the resulting mean. The bound follows the same anchored-at-last-input-token calculation as the initial study, using certified between-grid Jacobian envelopes and the diameter cap. Blocks larger than eight coordinates use conservative column-norm upper bounds; sampled sign probes provide a lower estimate. Large-count bounds may be loose or saturate at the diameter. Increasing error is a hypothesis, not an enforced constraint, and lengths also change instruction content.

## Numerical audit

- Completed conditions: 80,000; eligible: 71,500; explicitly ineligible input-length conditions: 8,500.
- Bound-chain violations: 0; original Lemma 2 violations: 0; Jacobian interval violations: 0.
- Rows above the sampled lower endpoint: 0.
- Maximum decomposition identity residual: 5.925e-13.

## Observed trends

- Layer 3, input_only: endpoint mean error 0.0011 → 0.0002; monotone nondecreasing across all counts: False.
- Layer 3, input_and_generated: endpoint mean error 0.0009 → 0.1526; monotone nondecreasing across all counts: True.
- Layer 3, prompt_length: endpoint mean error 0.0007 → 0.0263; monotone nondecreasing across all counts: True.
- Layer 9, input_only: endpoint mean error 0.0133 → 0.0091; monotone nondecreasing across all counts: False.
- Layer 9, input_and_generated: endpoint mean error 0.0112 → 0.9298; monotone nondecreasing across all counts: True.
- Layer 9, prompt_length: endpoint mean error 0.0043 → 0.2512; monotone nondecreasing across all counts: True.
- Layer 18, input_only: endpoint mean error 0.0562 → 0.0434; monotone nondecreasing across all counts: False.
- Layer 18, input_and_generated: endpoint mean error 0.0506 → 0.6248; monotone nondecreasing across all counts: False.
- Layer 18, prompt_length: endpoint mean error 0.0270 → 1.6134; monotone nondecreasing across all counts: True.
- Layer 27, input_only: endpoint mean error 0.1928 → 1.0889; monotone nondecreasing across all counts: False.
- Layer 27, input_and_generated: endpoint mean error 0.7889 → 6.0410; monotone nondecreasing across all counts: True.
- Layer 27, prompt_length: endpoint mean error 0.2227 → 1.4316; monotone nondecreasing across all counts: True.
- Layer 34, input_only: endpoint mean error 1.1207 → 1.7354; monotone nondecreasing across all counts: False.
- Layer 34, input_and_generated: endpoint mean error 1.7379 → 1.7379; monotone nondecreasing across all counts: False.
- Layer 34, prompt_length: endpoint mean error 1.5633 → 7.2530; monotone nondecreasing across all counts: True.

These are descriptive mean differences, without a significance claim. A lower single-token endpoint does not imply monotonicity at intermediate counts.

## Five-layer mean ± std

| Setting | m | k | g | n | Error mean ± std | Bound mean ± std |
|:---|---:|---:|---:|---:|---:|---:|
| prompt | 1 | 1 | 0 | 400 | 0.3636 ± 1.5100 | 33.0631 ± 4.8953 |
| prompt | 2 | 1 | 0 | 400 | 0.4522 ± 1.7553 | 36.7238 ± 3.2534 |
| prompt | 4 | 1 | 0 | 400 | 0.6837 ± 2.1503 | 38.4403 ± 3.0919 |
| prompt | 8 | 1 | 0 | 400 | 0.9943 ± 2.6057 | 39.5916 ± 3.0006 |
| prompt | 16 | 1 | 0 | 400 | 1.0821 ± 2.6077 | 40.4565 ± 3.0344 |
| prompt | 32 | 1 | 0 | 400 | 1.2141 ± 2.6497 | 41.5557 ± 3.0138 |
| prompt | 64 | 1 | 0 | 400 | 1.4387 ± 2.8379 | 42.8982 ± 2.9456 |
| prompt | 128 | 1 | 0 | 400 | 2.1151 ± 3.5182 | 44.2840 ± 3.0733 |
| steering | 4 | 1 | 0 | 95 | 0.2768 ± 1.2460 | 41.6643 ± 2.5501 |
| steering | 4 | 2 | 0 | 95 | 0.4298 ± 1.3967 | 41.9003 ± 2.4999 |
| steering | 4 | 4 | 0 | 95 | 0.5179 ± 1.4810 | 42.3365 ± 2.5040 |
| steering | 4 | 4 | 4 | 95 | 0.5874 ± 1.4861 | 43.2231 ± 2.5447 |
| steering | 4 | 4 | 12 | 95 | 0.6723 ± 1.4991 | 44.5393 ± 2.5733 |
| steering | 4 | 4 | 28 | 95 | 0.9792 ± 1.5536 | 45.5759 ± 2.6431 |
| steering | 4 | 4 | 60 | 95 | 1.3592 ± 1.5538 | 47.1300 ± 2.9742 |
| steering | 4 | 4 | 124 | 95 | 1.8972 ± 1.6516 | 48.7532 ± 2.9196 |
| steering | 4 | 8 | 0 | 95 | 0.5577 ± 1.4885 | 42.4494 ± 2.5401 |
| steering | 4 | 16 | 0 | 95 | 0.5989 ± 1.5087 | 43.4379 ± 2.9258 |
| steering | 4 | 32 | 0 | 95 | 0.5832 ± 1.4771 | 43.9553 ± 3.0097 |
| steering | 4 | 64 | 0 | 95 | 0.5730 ± 1.4783 | 45.1538 ± 3.2261 |
| steering | 4 | 128 | 0 | 95 | 0.5754 ± 1.4793 | 46.3111 ± 3.0838 |

## Per-layer mean ± std

| Layer | Setting | m | k | g | n | Error mean ± std | Bound mean ± std |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 3 | prompt | 1 | 1 | 0 | 400 | 0.0007 ± 0.0004 | 0.0620 ± 0.0361 |
| 3 | prompt | 2 | 1 | 0 | 400 | 0.0012 ± 0.0007 | 0.0974 ± 0.0563 |
| 3 | prompt | 4 | 1 | 0 | 400 | 0.0023 ± 0.0013 | 0.1431 ± 0.0820 |
| 3 | prompt | 8 | 1 | 0 | 400 | 0.0042 ± 0.0024 | 0.1958 ± 0.1118 |
| 3 | prompt | 16 | 1 | 0 | 400 | 0.0073 ± 0.0042 | 0.2520 ± 0.1438 |
| 3 | prompt | 32 | 1 | 0 | 400 | 0.0115 ± 0.0068 | 0.3090 ± 0.1776 |
| 3 | prompt | 64 | 1 | 0 | 400 | 0.0185 ± 0.0107 | 0.3691 ± 0.2119 |
| 3 | prompt | 128 | 1 | 0 | 400 | 0.0263 ± 0.0151 | 0.4298 ± 0.2318 |
| 3 | steering | 4 | 1 | 0 | 95 | 0.0011 ± 0.0007 | 0.1697 ± 0.0998 |
| 3 | steering | 4 | 2 | 0 | 95 | 0.0010 ± 0.0006 | 0.1210 ± 0.0689 |
| 3 | steering | 4 | 4 | 0 | 95 | 0.0009 ± 0.0007 | 0.0812 ± 0.0442 |
| 3 | steering | 4 | 4 | 4 | 95 | 0.0072 ± 0.0024 | 0.2255 ± 0.0461 |
| 3 | steering | 4 | 4 | 12 | 95 | 0.0190 ± 0.0064 | 0.3267 ± 0.0482 |
| 3 | steering | 4 | 4 | 28 | 95 | 0.0417 ± 0.0139 | 0.4039 ± 0.0505 |
| 3 | steering | 4 | 4 | 60 | 95 | 0.0827 ± 0.0256 | 0.4793 ± 0.0556 |
| 3 | steering | 4 | 4 | 124 | 95 | 0.1526 ± 0.0430 | 0.5714 ± 0.0664 |
| 3 | steering | 4 | 8 | 0 | 95 | 0.0009 ± 0.0006 | 0.0548 ± 0.0263 |
| 3 | steering | 4 | 16 | 0 | 95 | 0.0006 ± 0.0004 | 0.0385 ± 0.0152 |
| 3 | steering | 4 | 32 | 0 | 95 | 0.0004 ± 0.0003 | 0.0297 ± 0.0090 |
| 3 | steering | 4 | 64 | 0 | 95 | 0.0002 ± 0.0002 | 0.0261 ± 0.0062 |
| 3 | steering | 4 | 128 | 0 | 95 | 0.0002 ± 0.0002 | 0.0246 ± 0.0051 |
| 9 | prompt | 1 | 1 | 0 | 400 | 0.0043 ± 0.0017 | 0.6572 ± 0.2444 |
| 9 | prompt | 2 | 1 | 0 | 400 | 0.0102 ± 0.0042 | 1.1818 ± 0.4751 |
| 9 | prompt | 4 | 1 | 0 | 400 | 0.0271 ± 0.0110 | 1.8867 ± 0.7559 |
| 9 | prompt | 8 | 1 | 0 | 400 | 0.0507 ± 0.0204 | 2.4079 ± 0.9016 |
| 9 | prompt | 16 | 1 | 0 | 400 | 0.0759 ± 0.0329 | 2.9995 ± 1.0604 |
| 9 | prompt | 32 | 1 | 0 | 400 | 0.1264 ± 0.0530 | 3.7747 ± 1.2196 |
| 9 | prompt | 64 | 1 | 0 | 400 | 0.1931 ± 0.0701 | 4.5913 ± 1.3119 |
| 9 | prompt | 128 | 1 | 0 | 400 | 0.2512 ± 0.0832 | 5.2827 ± 1.3303 |
| 9 | steering | 4 | 1 | 0 | 95 | 0.0133 ± 0.0053 | 1.8831 ± 0.7320 |
| 9 | steering | 4 | 2 | 0 | 95 | 0.0120 ± 0.0051 | 1.7146 ± 0.6369 |
| 9 | steering | 4 | 4 | 0 | 95 | 0.0112 ± 0.0057 | 1.5774 ± 0.5892 |
| 9 | steering | 4 | 4 | 4 | 95 | 0.0603 ± 0.0270 | 2.6757 ± 0.6492 |
| 9 | steering | 4 | 4 | 12 | 95 | 0.1653 ± 0.0663 | 3.7101 ± 0.8109 |
| 9 | steering | 4 | 4 | 28 | 95 | 0.3223 ± 0.1136 | 4.5292 ± 0.8803 |
| 9 | steering | 4 | 4 | 60 | 95 | 0.5752 ± 0.1686 | 5.4173 ± 0.9484 |
| 9 | steering | 4 | 4 | 124 | 95 | 0.9298 ± 0.2193 | 6.4791 ± 1.0329 |
| 9 | steering | 4 | 8 | 0 | 95 | 0.0109 ± 0.0064 | 1.5741 ± 0.5376 |
| 9 | steering | 4 | 16 | 0 | 95 | 0.0099 ± 0.0062 | 2.0485 ± 0.5866 |
| 9 | steering | 4 | 32 | 0 | 95 | 0.0092 ± 0.0061 | 2.2681 ± 0.5513 |
| 9 | steering | 4 | 64 | 0 | 95 | 0.0088 ± 0.0059 | 2.6370 ± 0.6386 |
| 9 | steering | 4 | 128 | 0 | 95 | 0.0091 ± 0.0059 | 3.1884 ± 0.8519 |
| 18 | prompt | 1 | 1 | 0 | 400 | 0.0270 ± 0.0135 | 8.8314 ± 2.5074 |
| 18 | prompt | 2 | 1 | 0 | 400 | 0.0477 ± 0.0267 | 12.2792 ± 3.4828 |
| 18 | prompt | 4 | 1 | 0 | 400 | 0.0818 ± 0.0466 | 16.9031 ± 3.6440 |
| 18 | prompt | 8 | 1 | 0 | 400 | 0.1514 ± 0.0946 | 19.5890 ± 4.7886 |
| 18 | prompt | 16 | 1 | 0 | 400 | 0.2783 ± 0.1936 | 23.1220 ± 5.0316 |
| 18 | prompt | 32 | 1 | 0 | 400 | 0.5783 ± 0.4653 | 27.0755 ± 5.3252 |
| 18 | prompt | 64 | 1 | 0 | 400 | 0.9930 ± 0.8703 | 30.8023 ± 6.2242 |
| 18 | prompt | 128 | 1 | 0 | 400 | 1.6134 ± 1.4867 | 36.7851 ± 7.4999 |
| 18 | steering | 4 | 1 | 0 | 95 | 0.0562 ± 0.0293 | 18.0325 ± 3.6786 |
| 18 | steering | 4 | 2 | 0 | 95 | 0.0527 ± 0.0267 | 18.1035 ± 3.7550 |
| 18 | steering | 4 | 4 | 0 | 95 | 0.0506 ± 0.0254 | 19.2728 ± 3.9761 |
| 18 | steering | 4 | 4 | 4 | 95 | 0.0432 ± 0.0171 | 21.9558 ± 3.7685 |
| 18 | steering | 4 | 4 | 12 | 95 | 0.0590 ± 0.0248 | 27.0977 ± 5.8045 |
| 18 | steering | 4 | 4 | 28 | 95 | 0.1304 ± 0.0673 | 30.9036 ± 6.2200 |
| 18 | steering | 4 | 4 | 60 | 95 | 0.3003 ± 0.1868 | 37.2907 ± 7.6862 |
| 18 | steering | 4 | 4 | 124 | 95 | 0.6248 ± 0.3935 | 43.9843 ± 6.1217 |
| 18 | steering | 4 | 8 | 0 | 95 | 0.0475 ± 0.0240 | 19.7865 ± 3.9865 |
| 18 | steering | 4 | 16 | 0 | 95 | 0.0449 ± 0.0218 | 24.6362 ± 6.3740 |
| 18 | steering | 4 | 32 | 0 | 95 | 0.0420 ± 0.0191 | 27.5751 ± 7.1237 |
| 18 | steering | 4 | 64 | 0 | 95 | 0.0420 ± 0.0176 | 33.2521 ± 7.6811 |
| 18 | steering | 4 | 128 | 0 | 95 | 0.0434 ± 0.0181 | 38.4523 ± 8.2158 |
| 27 | prompt | 1 | 1 | 0 | 400 | 0.2227 ± 0.7001 | 18.0511 ± 7.1019 |
| 27 | prompt | 2 | 1 | 0 | 400 | 0.4625 ± 1.1650 | 24.1472 ± 4.7651 |
| 27 | prompt | 4 | 1 | 0 | 400 | 0.6591 ± 1.2831 | 26.9036 ± 2.7250 |
| 27 | prompt | 8 | 1 | 0 | 400 | 0.9589 ± 1.3509 | 28.0405 ± 1.5868 |
| 27 | prompt | 16 | 1 | 0 | 400 | 1.2428 ± 1.5848 | 28.1809 ± 1.4693 |
| 27 | prompt | 32 | 1 | 0 | 400 | 1.3025 ± 1.6075 | 28.2679 ± 1.3687 |
| 27 | prompt | 64 | 1 | 0 | 400 | 1.4008 ± 1.6401 | 28.4817 ± 1.2552 |
| 27 | prompt | 128 | 1 | 0 | 400 | 1.4316 ± 1.6533 | 28.6034 ± 1.2229 |
| 27 | steering | 4 | 1 | 0 | 95 | 0.1928 ± 0.3680 | 27.9021 ± 2.8778 |
| 27 | steering | 4 | 2 | 0 | 95 | 0.7080 ± 1.2850 | 29.1184 ± 2.0530 |
| 27 | steering | 4 | 4 | 0 | 95 | 0.7889 ± 1.3555 | 30.3614 ± 1.3574 |
| 27 | steering | 4 | 4 | 4 | 95 | 1.0884 ± 1.7506 | 30.8690 ± 1.4981 |
| 27 | steering | 4 | 4 | 12 | 95 | 1.3801 ± 2.1969 | 31.1723 ± 1.4997 |
| 27 | steering | 4 | 4 | 28 | 95 | 2.6636 ± 3.6162 | 31.6534 ± 1.6229 |
| 27 | steering | 4 | 4 | 60 | 95 | 4.1000 ± 4.0206 | 32.0714 ± 1.6578 |
| 27 | steering | 4 | 4 | 124 | 95 | 6.0410 ± 3.7422 | 32.3281 ± 1.5623 |
| 27 | steering | 4 | 8 | 0 | 95 | 0.9915 ± 1.7641 | 30.4419 ± 1.3892 |
| 27 | steering | 4 | 16 | 0 | 95 | 1.2032 ± 1.9120 | 30.0767 ± 1.4567 |
| 27 | steering | 4 | 32 | 0 | 95 | 1.1288 ± 1.6110 | 29.4881 ± 1.2751 |
| 27 | steering | 4 | 64 | 0 | 95 | 1.0784 ± 1.5852 | 29.4384 ± 1.2993 |
| 27 | steering | 4 | 128 | 0 | 95 | 1.0889 ± 1.6269 | 29.4030 ± 1.2579 |
| 34 | prompt | 1 | 1 | 0 | 400 | 1.5633 ± 7.5141 | 137.7140 ± 23.6004 |
| 34 | prompt | 2 | 1 | 0 | 400 | 1.7393 ± 8.7449 | 145.9134 ± 15.1807 |
| 34 | prompt | 4 | 1 | 0 | 400 | 2.6481 ± 10.6734 | 146.3648 ± 14.3935 |
| 34 | prompt | 8 | 1 | 0 | 400 | 3.8062 ± 13.0000 | 147.7246 ± 13.3579 |
| 34 | prompt | 16 | 1 | 0 | 400 | 3.8062 ± 13.0000 | 147.7283 ± 13.3555 |
| 34 | prompt | 32 | 1 | 0 | 400 | 4.0520 ± 13.1706 | 148.3513 ± 12.9747 |
| 34 | prompt | 64 | 1 | 0 | 400 | 4.5881 ± 14.0884 | 150.2468 ± 12.0711 |
| 34 | prompt | 128 | 1 | 0 | 400 | 7.2530 ± 17.5207 | 150.3187 ± 12.0327 |
| 34 | steering | 4 | 1 | 0 | 95 | 1.1207 ± 6.2383 | 160.3341 ± 13.1783 |
| 34 | steering | 4 | 2 | 0 | 95 | 1.3753 ± 6.6622 | 160.4442 ± 13.0712 |
| 34 | steering | 4 | 4 | 0 | 95 | 1.7379 ± 6.9765 | 160.3896 ± 13.1542 |
| 34 | steering | 4 | 4 | 4 | 95 | 1.7379 ± 6.9765 | 160.3896 ± 13.1542 |
| 34 | steering | 4 | 4 | 12 | 95 | 1.7379 ± 6.9765 | 160.3896 ± 13.1542 |
| 34 | steering | 4 | 4 | 28 | 95 | 1.7379 ± 6.9765 | 160.3896 ± 13.1542 |
| 34 | steering | 4 | 4 | 60 | 95 | 1.7379 ± 6.9765 | 160.3914 ± 13.1525 |
| 34 | steering | 4 | 4 | 124 | 95 | 1.7379 ± 6.9765 | 160.4030 ± 13.1507 |
| 34 | steering | 4 | 8 | 0 | 95 | 1.7379 ± 6.9765 | 160.3896 ± 13.1542 |
| 34 | steering | 4 | 16 | 0 | 95 | 1.7357 ± 6.9770 | 160.3896 ± 13.1542 |
| 34 | steering | 4 | 32 | 0 | 95 | 1.7354 ± 6.9771 | 160.4155 ± 13.1624 |
| 34 | steering | 4 | 64 | 0 | 95 | 1.7354 ± 6.9771 | 160.4155 ± 13.1624 |
| 34 | steering | 4 | 128 | 0 | 95 | 1.7354 ± 6.9771 | 160.4873 ± 13.1317 |

## Reproduce or resume

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/run_section4_long.py extract
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run python scripts/run_section4_long.py analyze
uv run python scripts/plot_section4_long.py
```

Decode checkpoints persist after every token; analysis journals persist after every condition and consolidate to JSON per input. Completed conditions are skipped on restart. Use analyze --start N --stride W for W independent workers. Logs contain logs only. Notifications are disabled; no email is sent. Large checkpoints and raw conditions remain local; manifests, summary, CSV, report, and PDF figures are retained in Git.
