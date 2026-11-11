# Project Tyto — Data Findings & Hypotheses

This document separates verified empirical **facts**, exploratory **observations**,
and working **hypotheses** discovered while inspecting OPUS OpenSubtitles data.

---

## 1. OPUS v2024 Tamil Corruption (FACT — Excluded)

During early exploration of OPUS OpenSubtitles `v2024` Tamil subtitles, we found
widespread missing Tamil dependent vowel signs (Unicode range `U+0BBE`–`U+0BCC`):

- Healthy Tamil text typically has a **~0.22–0.26** share of characters falling in
  the Tamil vowel-sign range (`U+0BBE`–`U+0BCC`).
- In `v2024`, roughly **~98.6 %** of documents exhibit severe vowel-sign
  deficiency (corrupted rendering / stripped diacritics). Running
  `scripts/survey_v2024_ta_integrity.py` directly on `data/raw/opus/v2024/ta.zip`
  surveyed **2,211** documents (up to 10,000 lines each) and found **2,193 (99.2 %)**
  below the 0.10 threshold, with a mean vowel-sign share of **0.0114**.
- **Decision:** `v2024` data was excluded completely from the MVP pipeline.
  See `scripts/survey_v2024_ta_integrity.py` for the standalone integrity check.

---

## 2. OPUS v2018 Integrity Evidence (FACT)

- Source archive: `data/raw/opus/v2018/en-ta.txt.zip` (Moses format: `.en`, `.ta`, `.ids`).
- Volume: **32,417** aligned subtitle pairs across **28** distinct documents (IMDb film IDs).
- Integrity gate: All **28** documents pass the minimum **0.10** Tamil vowel-sign
  share threshold enforced by `src/tyto/mvp/load.py`.

---

## 3. Spaced-Punctuation Observation (OBSERVATION — Not a Conclusion)

Across the 28 films in `v2018`, the frequency of spaces before/around punctuation
(`" , "`, `" . "`, `" ! "`, `" ? "`) varies sharply by film:

| IMDb Film ID | Observed Spaced-Punctuation Rate |
|--------------|----------------------------------|
| `770828`     | 82.5 %                           |
| `5255710`    | 21.7 %                           |
| `816692`     | 16.2 %                           |
| Most other films | < 4 %                        |

This disparity suggests mixed subtitle provenance (e.g., different tokenization,
OCR, or subtitling pipelines across sources), though the exact upstream cause is
not established.

---

## 4. Machine-Translation Hypothesis (OBSERVATION — Not Confirmed)

Inspecting sample Tamil subtitle lines shows instances where English proper nouns
appear transliterated without expected Tamil grammatical case endings (for example,
pairs corresponding to `"Do you know Boston?"`).

- **Status:** Qualitative **observation**, not a confirmed diagnosis. Determining
  what fraction of the corpus is machine-translated vs. human-translated requires
  native-speaker linguistic annotation.

---

## 5. Series Group Hypothesis (HYPOTHESIS)

Six IMDb identifiers in the `v2018` corpus share closely clustered numerical IDs:

`6261062`, `6261064`, `6261068`, `6261070`, `6261074`, `6261076`

- **Hypothesis:** These six documents may belong to the same episodic series and
  are tagged `series_group = "series_A"` in the processed dataset so downstream
  grouping/cross-validation experiments can account for potential shared-series
  leakage.
- **Status:** Working **hypothesis** based on IMDb ID adjacency; not externally
  verified against IMDb metadata.
