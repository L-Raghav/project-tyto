# Project Tyto MVP — 5-Minute Review Script & Q&A

## 5-Minute Review Script

### 1. What the Project Is (~45s)
Project Tyto is an exploratory data-mining and quality-triage system for
English-to-Tamil film subtitles. Rather than claiming automated translation grading,
it surfaces structural anomalies, recurring feature patterns, and heuristic risk
signals across subtitle pairs so human reviewers know where to look first.

### 2. What We Built (~90s)
We built an end-to-end reproducible Python pipeline (`python -m tyto mvp ...`)
covering:
- **Ingestion & Integrity Gate:** Reads the OPUS OpenSubtitles v2018 Moses zip
  (32,417 pairs across 28 films) and verifies Tamil script integrity (vowel-sign
  share >= 0.10 per document) after discovering widespread corruption in v2024.
- **Feature Extraction:** Computes cheap surface/structural features (word/char
  counts, length ratio, Tamil vowel-sign share, Latin-script ratio, spaced
  punctuation, digit match, illustrative idiom/hedge hits) alongside optional
  LaBSE bilingual cosine similarity on GPU.
- **Heuristic Risk Scoring:** Combines weak observable signals into a transparent
  risk proxy (`Low`, `Medium`, `High`) with clear disclosure that no human labels
  exist.
- **Pattern Mining & Predictor:** Runs FP-Growth association rules, KMeans
  clustering, and IsolationForest anomaly detection, plus a GroupKFold-validated
  classifier trained to predict the heuristic proxy from English-side features.
- **Delivery:** Interactive Streamlit app and UTF-8-BOM Power BI star-schema CSVs
  with a step-by-step build guide PDF.

### 3. Key Findings (~60s)
- **Data quality discovery:** OPUS v2024 Tamil subtitles have missing vowel signs
  (U+0BBE–U+0BCC) in ~98.6 % of documents, making v2024 unusable; v2018 passes
  integrity checks across all 28 films.
- **Spaced-punctuation variation:** Spaced punctuation rates vary widely across
  films (e.g., 82.5 % in IMDb `770828`, 21.7 % in `5255710`, 16.2 % in `816692`,
  while most films are under 4 %), suggesting mixed subtitle provenance.
- **Clusters, anomalies & rules:** KMeans selected **k = 8** clusters (silhouette = **0.2126**), IsolationForest surfaced **30** top structural anomalies, and FP-Growth mined **15** association rules (top rule `en_caps_high, en_words=4-7 -> film_id=2081437, ta_latin_share=low`, lift = **18.396**, confidence = **0.502**).
- **Data quality survey confirmation:** Running `scripts/survey_v2024_ta_integrity.py` on `v2024/ta.zip` measured **2,193 / 2,211 (99.2 %)** documents below the 0.10 vowel-sign share threshold (mean **0.0114**).

### 4. Limitations (~45s)
- Only **28 films** (small, non-representative sample).
- **No human ground-truth labels**; risk scores are **heuristic proxies**.
- Predictor AUROC measures proxy predictability, not human quality judgement.
- Illustrative idiom (40) and Tamil hedge (5) lists need native-speaker review.

---

## 10 Hardest Likely Questions (Honest Answers)

1. **Why are there only 28 films?**
   OPUS v2018 Moses format for English–Tamil contains 28 distinct documents
   (32,417 pairs). We excluded v2024 because ~98.6 % of its Tamil documents are
   corrupted (missing vowel signs).

2. **Is the accuracy real?**
   The cross-validated AUROC is computed cleanly with `GroupKFold` by film on real
   data, so the metric itself is real — **but** it measures how well the model
   predicts our *heuristic risk proxy*, not human translation quality.

3. **Why use heuristic labels?**
   No human annotations exist yet. Heuristic risk scores serve as transparent weak
   proxies for exploratory triage and pipeline validation until native Tamil
   speakers label a gold set.

4. **How was LaBSE used?**
   LaBSE bilingual sentence embeddings provide cosine similarity between English
   and Tamil lines as one signal among many (alongside length ratio, script share,
   punctuation, and digit parity).

5. **Why not use machine translation (MT) or an LLM to score quality directly?**
   That is beyond the MVP scope. This phase focuses on interpretable data mining
   and surface/alignment diagnostics rather than generative LLM evaluation.

6. **Can you predict quality for new films?**
   The GroupKFold model generalizes across unseen films to predict the *heuristic
   risk proxy* from English features, not true human-perceived quality.

7. **What does the spaced-punctuation finding mean?**
   It is an empirical observation of sharp stylistic/formatting differences across
   films (82.5 % in IMDb `770828` vs. <4 % in most films), pointing to mixed
   provenance rather than a single confirmed cause.

8. **Why English→Tamil specifically?**
   It is the course project focus; Tamil is an underserved language with
   morphological and script characteristics (e.g., vowel-sign encoding) that
   expose real corpus-engineering pitfalls.

9. **What would it take to validate the system properly?**
   A structured annotation pass by native Tamil speakers on a stratified sample of
   pairs (covering low/medium/high proxy tiers and multiple films) to establish
   gold quality labels and measure correlation.

10. **Is the Tamil text machine-translated?**
    It is plausible for portions of the corpus based on observations such as
    English proper nouns appearing transliterated without Tamil case endings
    (e.g., `"Do you know Boston?"`), but this is an unconfirmed observation, not a
    proven conclusion.
