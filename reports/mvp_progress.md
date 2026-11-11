# MVP Progress Checklist

> Updated 2026-11-11 after Stage 10 real pipeline run. All numbers from real code output.

| Stage | Description | Status | Notes |
|-------|-------------|--------|-------|
| 1 | Setup: install deps, declare in pyproject.toml | ✅ DONE | reportlab added; src/tyto/mvp/ scaffolded |
| 2 | Loader: `mvp load` → pairs_v2018.parquet | ✅ DONE | 32,417 pairs, 28 docs, vowel-share gate passed |
| 3 | Features: `mvp features` → features.parquet | ✅ DONE | LaBSE on CUDA cached, mean_cos=0.7916 |
| 4 | Heuristic risk: `mvp label` → risk.parquet | ✅ DONE | High=3,242 Medium=6,483 Low=22,692 |
| 5 | Mining: `mvp mine` → rules/clusters/anomalies | ✅ DONE | 15 rules top_lift=18.396; k=8 silhouette=0.2126; 30 anomalies |
| 6 | Predictor: `mvp predict` → model + metrics | ✅ DONE | LightGBM AUROC=0.6994 PR-AUC=0.3863 R@10%=0.237 |
| 7 | App: Streamlit mvp_app.py | ✅ DONE | src/tyto/app/mvp_app.py |
| 8 | Power BI export: CSVs + PDF guide | ✅ DONE | 7 CSVs, 6-page PDF (exports/ gitignored; regenerate with mvp export) |
| 9 | Docs: MVP_REPORT, DEMO_DISCLOSURE, etc. | ✅ DONE | docs/ written + updated with real numbers |
| 10 | Quality gates: pytest + ruff + real pipeline run | ✅ DONE | ruff=clean, pytest=21/21, timings in reports/mvp_report.md |

## Real Timings (Stage 10 run, 2026-11-11, RTX 4060)
| Command | Elapsed |
|---------|---------|
| mvp load | 3.97 s |
| mvp features | 6.26 s (LaBSE cached) |
| mvp label | 4.01 s |
| mvp mine | 16.21 s |
| mvp predict | 57.67 s |
| mvp export | 4.69 s |

## v2024 Corruption Survey (real output)
2,193 / 2,211 documents (99.2%) below 0.10 vowel-sign share; mean = 0.0114
