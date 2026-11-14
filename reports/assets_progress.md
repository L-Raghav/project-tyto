# Report Assets Progress Checklist

- [x] **0. Setup**: Branch `mvp-report-assets` created from `mvp-demo`, sibling folder `../tyto-report-assets` created.
- [x] **A. Remove Two Banners**: Removed global banner and Predictor-page warning banner from `src/tyto/app/mvp_app.py`; verified `ruff` and `pytest`; committed (`64ad5ab`).
- [x] **B.1 Dependencies**: Added `report` optional dependency group (`playwright`, `pygments`, `matplotlib`) to `pyproject.toml`, installed into `.venv`, and installed Playwright Chromium.
- [x] **B.2 Script**: Implemented `scripts/make_report_assets.py` (`--out`, `--only`).
- [x] **B.3 `01_app/`**: Captured all 6 Streamlit pages (1440x900 viewport + full page), Pair Explorer limited to 10 rows, Predictor predefined example (`[3]`), and Predictor live input (`14` PNGs).
- [x] **B.4 `02_code/`**: Rendered chunked code listings (<=55 lines, >=1600px wide, Pygments HtmlFormatter + Playwright) across all 7 target modules (`86` PNGs).
- [x] **B.5 `03_output/`**: Captured real CLI/test/git/v2024 survey outputs and artifact snapshots, secret-checked and <=40 lines per image (`18` PNGs).
- [x] **B.6 `04_charts/`**: Generated 200 dpi matplotlib charts from real artifacts (`5` PNGs).
- [x] **B.7 Docs & QA**: Generated `MANIFEST.md` and `CAPTIONS.md`, verified all 123 PNGs (>=1600px wide, >15KB, non-blank), visually inspected representative PNGs, updated `reports/assets_report.md`, verified `ruff` and `pytest`.
