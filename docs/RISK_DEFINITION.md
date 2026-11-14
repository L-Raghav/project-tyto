# Heuristic Risk Definition

> **Disclaimer:** Weights are arbitrary and not derived from human quality labels. Risk scores are heuristic proxies only.

## Overview

Because no human translation-quality annotations exist yet for the OPUS OpenSubtitles v2018 English–Tamil corpus (32,417 subtitle pairs across 28 documents), Project Tyto MVP computes an unsupervised **heuristic risk score** in `[0, 100]` to prioritize pairs for inspection and pattern mining.

## Formula

$$\text{risk\_score} = \sum_{i=1}^{7} w_i \cdot s_i$$

where $s_i \in [0, 1]$ is either a percentile-rank-normalized continuous signal or a binary indicator, and $\sum_{i=1}^{7} w_i = 100$.

| # | Signal Column | Description | Type | Weight ($w_i$) |
|---|---|---|---|---|
| 1 | `sig_labse_cos_inv_rank` | Inverted percentile rank of `labse_cos` (`1 - rank_pct(labse_cos)`); `0.5` if `labse_cos` is `NaN` | Continuous (rank 0–1) | **30** |
| 2 | `sig_len_ratio_extreme` | `1` if `len_ratio < p5` or `len_ratio > p95` across the corpus, else `0` | Binary | **15** |
| 3 | `sig_ta_spaced_punct` | `1` if `ta_spaced_punct > 0` (spaced punctuation in Tamil, unusual for most films), else `0` | Binary | **10** |
| 4 | `sig_ta_latin_heavy` | `1` if `ta_latin_share > 0.1` (Latin-heavy Tamil text), else `0` | Binary | **10** |
| 5 | `sig_digit_mismatch` | `1` if English contains digits (`\d+`) and `digit_match == 0`, else `0` | Binary | **10** |
| 6 | `sig_punct_parity` | `1` if `abs(punct(en) - punct(ta)) > 1`, else `0` | Binary | **10** |
| 7 | `sig_idiom_low_sim` | `1` if `idiom_present == 1` (illustrative 40-idiom list) AND `labse_cos < median(labse_cos)`, else `0` | Binary | **15** |

**Total weight:** $30 + 15 + 10 + 10 + 10 + 10 + 15 = 100$.

## Risk Tiers

Tiers are assigned by corpus percentile of `risk_score`:

- **High**: top 10% (`risk_score >= p90`)
- **Medium**: next 20% (`p70 <= risk_score < p90`)
- **Low**: bottom 70% (`risk_score < p70`)

## Caveats

1. Weights are arbitrary and not derived from human quality labels. Risk scores are heuristic proxies only.
2. Idiom matching uses an illustrative 40-phrase English list pending native-speaker review.
3. Spaced punctuation and Latin-heavy Tamil text reflect observable corpus surface differences rather than confirmed translation errors.
