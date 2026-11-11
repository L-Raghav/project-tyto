# MVP Progress Checklist

> Updated by agents after each stage. All numbers from real code output.

| Stage | Description | Status | Notes |
|-------|-------------|--------|-------|
| 1 | Setup: install deps, declare in pyproject.toml | ✅ DONE | reportlab>=2.7 added to pyproject.toml; src/tyto/mvp/__init__.py created |
| 2 | Loader: `mvp load` → pairs_v2018.parquet | ✅ DONE | 32,417 pairs, 28 docs; vowel_share 0.1793–0.2106 (all pass >=0.10 gate) |
| 3 | Features: `mvp features` → features.parquet | ⏳ PENDING | |
| 4 | Heuristic risk: `mvp label` → risk.parquet | ⏳ PENDING | |
| 5 | Mining: `mvp mine` → rules/clusters/anomalies | ⏳ PENDING | |
| 6 | Predictor: `mvp predict` → model + metrics | ⏳ PENDING | |
| 7 | App: Streamlit mvp_app.py | ⏳ PENDING | |
| 8 | Power BI export: CSVs + PDF guide | ⏳ PENDING | |
| 9 | Docs: MVP_REPORT, DEMO_DISCLOSURE, etc. | ⏳ PENDING | |
| 10 | Quality gates: pytest + ruff + real pipeline run | ⏳ PENDING | |

## Blockers
_None yet_

## Real Timings (from Stage 10 run)
_To be filled after Stage 10_
