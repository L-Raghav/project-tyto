# MVP Progress Checklist

> Updated by agents after each stage. All numbers from real code output.

| Stage | Description | Status | Notes |
|-------|-------------|--------|-------|
| 1 | Setup: install deps, declare in pyproject.toml | ✅ DONE | reportlab>=2.7 added to pyproject.toml; src/tyto/mvp/__init__.py created |
| 2 | Loader: `mvp load` → pairs_v2018.parquet | ✅ DONE | 32,417 pairs, 28 docs; vowel_share 0.1793–0.2106 (all pass >=0.10 gate) |
| 3 | Features: `mvp features` → features.parquet | ✅ DONE | 32,417 rows, 14 idiom hits, LaBSE ok (cuda), mean_cos=0.7916 |
| 4 | Heuristic risk: `mvp label` → risk.parquet | ✅ DONE | 32,417 rows, mean=18.23, p70=23.89, p90=33.07; Low=22692, Medium=6483, High=3242; docs/RISK_DEFINITION.md written |
| 5 | Mining: `mvp mine` → rules/clusters/anomalies | ✅ DONE | 15 rules (top_lift=18.396), KMeans best_k=8 (sil=0.2126), 30 anomalies, 28 films |
| 6 | Predictor: `mvp predict` → model + metrics | ✅ DONE | GroupKFold (5 folds, 23 groups) on 32,417 rows (20.0% pos): LightGBM AUROC=0.6994 [0.6921, 0.7062], PR-AUC=0.3863 [0.3760, 0.3967], R@10%=0.2370; LogReg AUROC=0.6990, PR-AUC=0.3722; Baseline AUROC=0.5723, PR-AUC=0.2289 (heuristic proxy only) |
| 7 | App: Streamlit mvp_app.py | ✅ DONE | 6-page Streamlit app at `src/tyto/app/mvp_app.py` (Overview, Film Explorer, Pair Explorer, Patterns, Predictor, About & Limits) |
| 8 | Power BI export: CSVs + PDF guide | ✅ DONE | 7 UTF-8 BOM CSVs (`exports/powerbi/`, 32,417 pairs, 3,242 High-risk, 28 films, 8 clusters, 15 rules, 30 anomalies, 3 model rows) + `DATA_DICTIONARY.md` + `RELATIONSHIPS.md` + `PowerBI_Build_Guide.pdf` (6 pages, 13 headings) |
| 9 | Docs: MVP_REPORT, DEMO_DISCLOSURE, etc. | ✅ DONE | `docs/DEMO_DISCLOSURE.md`, `docs/MVP_REPORT.md`, `docs/REVIEW_TALKING_POINTS.md`, `docs/RUNBOOK_MVP.md`, `docs/DATA_FINDINGS.md`, `scripts/survey_v2024_ta_integrity.py` (2,193/2,211 = 99.2% v2024 docs <0.10 vowel-share, mean 0.0114) |
| 10 | Quality gates: pytest + ruff + real pipeline run | ⏳ PENDING | |

## Blockers
_None yet_

## Real Timings (from Stage 10 run)
_To be filled after Stage 10_
