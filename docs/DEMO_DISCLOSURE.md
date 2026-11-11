# Project Tyto MVP — Demo Disclosure

This document describes the provenance and status of every data item used in the
MVP demonstration. It is intended for reviewers and collaborators.

| Item | Status | Notes |
|------|--------|-------|
| Subtitle pair data | REAL | OPUS OpenSubtitles v2018, Moses-format zip; 32,417 pairs, 28 distinct films |
| Cheap features (lengths, ratios, punct) | REAL | Computed from actual text using deterministic rules |
| LaBSE similarity scores | REAL (if model loaded) or MISSING (if failed) | GPU required; falls back gracefully if CUDA unavailable |
| Risk scores | HEURISTIC PROXY | Weighted combination of cheap signals; **no human quality labels exist** |
| Idiom list (40 phrases) | ILLUSTRATIVE | To be reviewed and extended by a native Tamil speaker before production use |
| Tamil hedge list (5 words) | ILLUSTRATIVE | To be reviewed and extended by a native Tamil speaker before production use |
| Demo example sentences (12) | ILLUSTRATIVE | Labelled as such; outputs are real model outputs on real data |
| Association rules | REAL | Derived from actual feature data via FP-Growth (mlxtend) |
| Cluster assignments | REAL | KMeans on actual feature vectors |
| Anomaly detection | REAL | IsolationForest on actual feature vectors |
| Predictor model | REAL (trained on heuristic proxy) | GroupKFold AUROC from real cross-validation; predicts proxy, not human judgement |
| Power BI CSVs | REAL | Derived from all of the above |
| Human quality labels | NOT DONE | No annotation exercise has been completed |
| v2024 Tamil data | NOT USED | Corrupted: vowel-sign characters (U+0BBE–U+0BCC) absent in ~98.6 % of documents; healthy Tamil should show a ~0.22–0.26 vowel-sign share |

---

> **Important:** Risk scores are **heuristic proxies** constructed from observable
> text features. They are not validated against human translation-quality judgements
> and must not be presented as such.
