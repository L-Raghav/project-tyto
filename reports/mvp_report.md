# Project Tyto — MVP Report

> Generated: 2026-11-11. All numbers from real pipeline runs on branch `mvp-demo`.

## What Was Built

A 10-stage end-to-end pipeline for mining and predicting translation-quality problems in English→Tamil subtitle pairs from OPUS OpenSubtitles v2018.

| Stage | Description | Status | Artifact |
|-------|-------------|--------|---------|
| 1 | Setup: reportlab added to pyproject.toml, `src/tyto/mvp/` package scaffolded | ✅ DONE | `src/tyto/mvp/__init__.py` |
| 2 | Loader: v2018 zip → parquet with integrity gate | ✅ DONE | `data/processed/pairs_v2018.parquet` |
| 3 | Features: cheap + LaBSE similarity | ✅ DONE | `artifacts/features.parquet` |
| 4 | Heuristic risk labelling | ✅ DONE | `artifacts/risk.parquet` |
| 5 | Pattern mining: FP-Growth, KMeans, IsolationForest | ✅ DONE | `artifacts/association_rules.parquet`, `artifacts/clusters.parquet`, `artifacts/anomalies.parquet`, `artifacts/film_stats.parquet` |
| 6 | Predictor: baseline, LogReg, LightGBM w/ GroupKFold | ✅ DONE | `artifacts/lgbm_model.pkl`, `artifacts/predictor_metrics.json` |
| 7 | Streamlit app | ✅ DONE | `src/tyto/app/mvp_app.py` |
| 8 | Power BI export: CSVs + PDF guide | ✅ DONE | `exports/powerbi/` (gitignored — regenerate with `mvp export`) |
| 9 | Documentation | ✅ DONE | `docs/MVP_REPORT.md`, `docs/DEMO_DISCLOSURE.md`, `docs/REVIEW_TALKING_POINTS.md`, `docs/RUNBOOK_MVP.md`, `docs/DATA_FINDINGS.md` |
| 10 | Quality gates: pytest + ruff + real pipeline run | ✅ DONE | `tests/test_mvp_*.py`, `reports/mvp_report.md` |

---

## How to Run

```powershell
# Full pipeline (in order)
.venv\Scripts\python.exe -m tyto mvp load
.venv\Scripts\python.exe -m tyto mvp features
.venv\Scripts\python.exe -m tyto mvp label
.venv\Scripts\python.exe -m tyto mvp mine
.venv\Scripts\python.exe -m tyto mvp predict
.venv\Scripts\python.exe -m tyto mvp export

# Launch the app
streamlit run src/tyto/app/mvp_app.py
```

---

## Real Pipeline Timings (2026-11-11, RTX 4060 GPU)

| Command | Elapsed | Key Output |
|---------|---------|-----------|
| `mvp load` | 3.97 s | 32,417 pairs, 28 docs, all vowel-shares 0.18–0.21 |
| `mvp features` | 6.26 s | LaBSE cached (CUDA), mean_cos=0.7916 |
| `mvp label` | 4.01 s | High=3,242, Medium=6,483, Low=22,692 |
| `mvp mine` | 16.21 s | 15 rules (top lift=18.396), k=8, silhouette=0.2126, 30 anomalies |
| `mvp predict` | 57.67 s | LightGBM AUROC=0.6994, PR-AUC=0.3863, R@10%=0.2370 |
| `mvp export` | 4.69 s | 7 CSVs, 6-page PDF |

---

## Predictor Metrics (real, GroupKFold 5-fold, 1000-resample bootstrap 95% CI)

> **Note: Model predicts a heuristic proxy trained on weak labels. NOT validated against human judgement.**

| Model | AUROC | 95% CI | PR-AUC | 95% CI | Recall@10% |
|-------|-------|--------|--------|--------|------------|
| Baseline (word count) | 0.5723 | [0.566, 0.579] | 0.2289 | [0.226, 0.232] | 0.118 |
| Logistic Regression | 0.6990 | [0.692, 0.707] | 0.3722 | [0.363, 0.383] | 0.224 |
| **LightGBM** | **0.6994** | **[0.692, 0.706]** | **0.3863** | **[0.376, 0.397]** | **0.237** |

Groups: 23 (film_id; 6 series_A films count as one group per hypothesis). Positive rate: 20.0% (6,484/32,417).

Top features by LightGBM contribution: pca_0, pca_1, pca_2, pca_8, pca_9 (English LaBSE PCA components).

---

## Data Quality Findings

### v2018 (USED)
- 32,417 pairs, 28 distinct documents
- All documents pass 0.10 vowel-sign integrity gate (range: 0.179–0.211)
- Spaced-punctuation rates vary hugely by film: 82.5% (IMDb 770828), 21.7% (5255710), 16.2% (816692), most films under 4% — OBSERVATION, not a conclusion

### v2024 Tamil (NOT USED — CORRUPTED)
- Survey of `data/raw/opus/v2024/ta.zip` (run 2026-11-11):
  2,193 of 2,211 documents (99.2%) have vowel-sign share below 0.10
  Mean vowel-share across all docs: 0.0114 (healthy Tamil: 0.22–0.26)
- v2024 Tamil data entirely excluded from this project

---

## Raw Ruff Output

```
All checks passed!
```

---

## Raw Pytest Output (MVP tests)

```
.....................                                                    [100%]
21 passed in 5.29s
```

Test files:
- `tests/test_mvp_load.py` — loader, integrity gate, NFC, link_type, row count gate
- `tests/test_mvp_features.py` — cheap features, idiom detection (inflected), LaBSE
- `tests/test_mvp_label.py` — risk tiers, score bounds, all signals
- `tests/test_mvp_predict.py` — predictor smoke, PCA, bootstrap CI
- `tests/test_mvp_cli.py` — CLI help, missing file error, missing parquet error

---

## Assumptions and Known Issues

1. **Heuristic labels only** — no human quality annotations exist. Risk scores are weighted combinations of weak signals with arbitrary weights. See `docs/RISK_DEFINITION.md`.
2. **28 documents** — not a representative sample of Tamil subtitle translation quality.
3. **Series group hypothesis** — IMDb IDs 6261062/64/68/70/74/76 labeled series_A; this is a hypothesis, not confirmed.
4. **Machine-translation hypothesis** — Tamil text may be machine-translated (English nouns transliterated without case endings); observed, not confirmed.
5. **LaBSE top features** — PCA components dominate the predictor, suggesting English embedding structure is the primary predictor of heuristic risk (may partly reflect corpus statistics, not genuine quality).
6. **Silhouette 0.2126 at k=8** — modest separation; clusters should be treated as exploratory.
7. **exports/ is gitignored** — CSVs and PDF regenerate deterministically from `mvp export`.
8. **Idiom and hedge lists are illustrative** — labeled as such throughout; require native speaker review.
