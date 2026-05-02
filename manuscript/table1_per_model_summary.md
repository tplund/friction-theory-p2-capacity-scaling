# Table 1 — Per-model summary: Zorbetik accuracy and per-token CR

Accuracy values from ICL condition on the Zorbetik domain (Qwen2.5 ladder n=50 cloze / 47 application; Qwen3-8B, Llama-3.3-70B, Qwen3-235B, DeepSeek-V3 n=20 each).

CR statistics are **medians** over the per-token CR values of application responses.  Values are given on a log10 scale for CR ≥ 1e4 (formatted as `1e<exponent>`), because logprob resolution differs by platform.  Absolute CR magnitudes are therefore only comparable within the same platform (Qwen2.5 rows: unsloth local; Qwen3-8B & Llama-3.3-70B: Fireworks; Qwen3-235B & DeepSeek-V3: Together).

| Model | Size (B) | Active (B) | Cloze % | Application % | med_cr | first5_cr | last5_cr | n_resp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 0.5 | 0.5 | 56 | 2 | 13.6 | 8.66 | 21.6 | 47 |
| Qwen2.5-1.5B | 1.5 | 1.5 | 90 | 17 | 19.2 | 12.2 | 27.0 | 47 |
| Qwen2.5-3B | 3 | 3 | 58 | 32 | 71.6 | 19.6 | 148.4 | 47 |
| Qwen2.5-7B | 7 | 7 | 80 | 40 | 81.8 | 23.4 | 433.3 | 47 |
| Qwen3-8B | 8 | 8 | 90 | 65 | 1e8.3 | 1e5.5 | 1e8.1 | 20 |
| Qwen2.5-14B | 14 | 14 | 96 | 64 | 71.8 | 28.0 | 1123.1 | 47 |
| Llama-3.3-70B | 70 | 70 | 90 | 85 | 1e6.3 | 64.1 | 1e8.0 | 20 |
| Qwen3-235B | 235 | 22 | 90 | 70 | 1e6.7 | 1e4.9 | 1e7.3 | 20 |
| DeepSeek-V3 | 671 | 37 | 90 | 85 | 1e5.2 | 1131.1 | 1e6.2 | 20 |

**Footnotes:**

1. Accuracy numbers are taken from the paper spec and match a recount of `precal_eval_all.jsonl` within ±1pp on the Qwen2.5 ladder (differences within rounding).
2. CR medians for Fireworks- and Together-served models (Qwen3-8B, Llama-3.3-70B, Qwen3-235B, DeepSeek-V3) are ~4–14 orders of magnitude larger than for the locally-served Qwen2.5 ladder because those platforms return logprobs clipped to -1e-5 on confident tokens, inflating the CR denominator.  We therefore report log10 of the median instead of raw CR for those rows.
3. `first5_cr` / `last5_cr` are the medians of the first / last 5 per-token CR values within each application response, then aggregated across responses.  On the dense Qwen2.5 ladder the last5 values are systematically higher than first5 at every size from 3B upward, consistent with the §2.8 observation that application friction accumulates during the reasoning chain rather than spiking on the first token.
