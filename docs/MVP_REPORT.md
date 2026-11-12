# Project Tyto MVP — Phase 1 Report

## Executive Summary

Project Tyto is an exploratory data mining and quality-triage system for
English-to-Tamil subtitle pairs drawn from OPUS OpenSubtitles v2018 (32,417 pairs
across 28 films). Because no human quality annotations exist yet, risk scores are
computed as **heuristic proxies** from observable text and alignment features, and
all downstream pattern mining, clustering, anomaly detection, and predictive
modelling operate on those proxy labels.

---

## Stage Status Overview

| Stage | Description | Status | Real Artifacts / Notes |
|-------|-------------|--------|------------------------|
| Stage 1 | Setup & dependencies (`reportlab>=2.7`, `tyto.mvp` scaffold) | DONE | `pyproject.toml`, `src/tyto/mvp/__init__.py` |
| Stage 2 | OPUS v2018 loader & integrity gate | DONE | `data/processed/pairs_v2018.parquet` (32,417 pairs, 28 films; all pass >=0.10 vowel-sign gate) |
| Stage 3 | Feature extraction (cheap features + LaBSE) | DONE | `artifacts/features.parquet` (32,417 rows, 14 idiom hits, 397 hedge hits, LaBSE on CUDA, mean cos=0.7916) |
| Stage 4 | Heuristic risk scoring | DONE | `artifacts/risk.parquet` (Low=22,692, Medium=6,483, High=3,242; heuristic proxy only) |
| Stage 5 | Pattern mining (FP-Growth, KMeans, IsolationForest) | DONE | `artifacts/association_rules.parquet` (15 rules, top lift=18.396), `artifacts/clusters.parquet` (k=8, sil=0.2126), `artifacts/anomalies.parquet` (30 pairs), `artifacts/film_stats.parquet` (28 films) |
| Stage 6 | GroupKFold risk-proxy predictor | DONE | `models/` + `artifacts/predictor_metrics.json` (LightGBM AUROC=0.6994, PR-AUC=0.3863, Recall@10%=0.2370; LogReg AUROC=0.6990; Baseline AUROC=0.5723) |
| Stage 7 | Streamlit interactive review app | DONE | `src/tyto/app/mvp_app.py` (6-page interactive triage app) |
| Stage 8 | Power BI CSV export + Build Guide PDF | DONE | `exports/powerbi/*.csv` (7 UTF-8 BOM CSVs, 32,417 pairs, 28 films), `DATA_DICTIONARY.md`, `RELATIONSHIPS.md`, `PowerBI_Build_Guide.pdf` (6 pages, 13 headings + title page) |
| Stage 9 | Docs: report, disclosure, talking points, runbook, findings, v2024 survey | DONE | `docs/*.md`, `scripts/survey_v2024_ta_integrity.py` (2,193/2,211 = 99.2% v2024 docs <0.10 vowel-share, mean 0.0114) |
| Stage 10 | Quality gates (`pytest`, `ruff`, full pipeline run) | IN PROGRESS | Unit tests + ruff checks underway; final full-pipeline run pending |

---

## Pipeline Summary & Measured Outputs

- **Dataset (`data/processed/pairs_v2018.parquet`):** OPUS OpenSubtitles v2018 (`data/raw/opus/v2018/en-ta.txt.zip`), **32,417** aligned pairs across **28** films.
- **v2024 Integrity Survey (`scripts/survey_v2024_ta_integrity.py`):** Surveyed **2,211** documents in `data/raw/opus/v2024/ta.zip`; **2,193 (99.2 %)** fall below the 0.10 Tamil vowel-sign share threshold (mean **0.0114** vs. healthy ~0.22–0.26).
- **v2018 Integrity Gate:** All **28** v2018 documents pass the minimum 0.10 Tamil vowel-sign share threshold (0.1793–0.2106).
- **Features & LaBSE (`artifacts/features.parquet`):** Computed on CUDA (`mean labse_cos = 0.7916`); **14** pairs matched the illustrative English idiom list and **397** pairs matched the illustrative Tamil hedge list.
- **Heuristic Risk Tiers (`artifacts/risk.parquet`):** Mean = **18.23**, p70 = **23.89**, p90 = **33.07**; Low = **22,692**, Medium = **6,483**, High = **3,242** (top 10 % heuristic proxy).
- **Pattern Mining (`artifacts/`):** KMeans selected **k = 8** (silhouette = **0.2126**); FP-Growth produced **15** rules (top rule `en_caps_high, en_words=4-7 -> film_id=2081437, ta_latin_share=low`, lift = **18.396**, support = **0.0104**, confidence = **0.5022**); IsolationForest flagged **30** top anomalies across **28** films.
- **Predictor Metrics (`artifacts/predictor_metrics.json`, 5-fold GroupKFold across 23 film-groups, 32,417 rows, 20.0 % positive rate at p80 risk threshold 27.29):**
  - **LightGBM:** AUROC = **0.6994** (95 % CI [0.6921, 0.7062]), PR-AUC = **0.3863** ([0.3760, 0.3967]), Recall@10% = **0.2370**
  - **Logistic Regression:** AUROC = **0.6990** ([0.6920, 0.7066]), PR-AUC = **0.3722** ([0.3626, 0.3825]), Recall@10% = **0.2238**
  - **Baseline:** AUROC = **0.5723** ([0.5661, 0.5790]), PR-AUC = **0.2289** ([0.2259, 0.2321]), Recall@10% = **0.1180**
  - _Disclosure:_ Evaluates prediction of the heuristic proxy on English-side features, not human quality judgement.
- **Power BI Export & Build Guide PDF (`exports/powerbi/PowerBI_Build_Guide.pdf`):** Reopened PDF verified **6 pages** and **13 section headings** (`1. What We Are Building` through `13. Glossary`, plus Title Page). Exported 7 UTF-8 BOM CSVs (`fact_pairs.csv` = 32,417 rows with 3,242 High-risk, `dim_film.csv` = 28, `dim_cluster.csv` = 8, `dim_risk_tier.csv` = 3, `fact_rules.csv` = 15, `fact_anomalies.csv` = 30, `fact_model_metrics.csv` = 3).

---

## Limitations

1. **Small sample size:** 28 documents is not a representative sample of general English-to-Tamil subtitle translation.
2. **No human quality labels:** No native Tamil speaker annotation has been performed yet.
3. **Risk scores are heuristic proxies:** The risk score is a weighted combination of observable surface/alignment signals, not ground-truth translation quality.
4. **Pilot is not a validated accuracy claim:** Cross-validated AUROC/PR-AUC reflects how well the classifier predicts the heuristic proxy given English-side signals, not human quality judgement.
5. **Possible machine-translation artifacts (observations, not conclusions):** Some Tamil lines show patterns consistent with machine translation or transliteration without Tamil case endings, and spaced-punctuation rates vary sharply across films. These remain empirical observations pending linguistic review.
6. **Illustrative lexicons:** The 40-phrase English idiom list and 5-word Tamil hedge list are illustrative placeholders awaiting native-speaker review.
