# Report Assets Progress Checklist

- [x] **0. Setup**: Branch `mvp-report-assets` created from `mvp-demo`, sibling folder `../tyto-report-assets` created.
- [x] **A. Remove Two Banners**: Removed global banner and Predictor-page warning banner from `src/tyto/app/mvp_app.py`; verified `ruff` and `pytest`; wrote `reports/assets_report.md`; ready to commit.
- [ ] **B.1 Dependencies**: Add `report` optional dependency group (`playwright`, `pygments`, `matplotlib`) to `pyproject.toml`, install, and install Playwright Chromium.
- [ ] **B.2 Script**: Implement `scripts/make_report_assets.py` (`--out`, `--only`).
- [ ] **B.3 `01_app/`**: Capture real Streamlit app pages (1440x900 full page + viewport, Predictor predefined + live input, Pair Explorer 10 rows).
- [ ] **B.4 `02_code/`**: Render chunked code listings (<=55 lines, >=1600px wide, Pygments HtmlFormatter + Playwright).
- [ ] **B.5 `03_output/`**: Capture real CLI/test/git/artifact outputs (secret-checked, <=40 lines per image).
- [ ] **B.6 `04_charts/`**: Generate 200 dpi matplotlib charts from real artifacts.
- [ ] **B.7 Docs & QA**: Generate `MANIFEST.md` and `CAPTIONS.md`, verify PNG dimensions (>=1600px) and size (>15KB), visually inspect images, update `reports/assets_report.md`, run final checks, commit, push `mvp-report-assets`.
