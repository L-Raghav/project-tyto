# Project Tyto — Report Assets & Banner Removal Report

## Part A: Removal of Disclosure Banners

### Changes Made
Removed exactly the two requested disclosure banners from `src/tyto/app/mvp_app.py`:
1. Removed `PERSISTENT_BANNER` constant and the global `st.warning(PERSISTENT_BANNER)` call in `main()`.
2. Removed `st.warning("This model predicts a heuristic proxy. NOT validated against human judgement.")` in `_page_predictor()`.

Searched `src/`, `tests/`, and `docs/` for banner references and partial phrases (`HEURISTIC PROXY`, `NOT validated`, `not human-validated`, `No ground-truth quality labels`, `28 documents`, `PERSISTENT_BANNER`, `banner`). No tests asserted on the banner text and no `docs/` files claimed the app shows a banner; About & Limits page and all `docs/` disclosures were left untouched.

### Verification (Part A)

#### `.venv\Scripts\python.exe -m ruff check .` (exit code 0)
```
All checks passed!
```

#### `.venv\Scripts\python.exe -m pytest -q` (exit code 0)
```
........................................................................ [ 55%]
.........................................................                [100%]
```
_(Note: `pyproject.toml` sets `addopts = "-ra -q"`, so passing `-q` yields quiet `-qq` output; 129 tests ran and exited with code 0.)_

---

## Part B: Report Assets Generation

### What Was Built
- Added `[project.optional-dependencies] report` (`playwright>=1.49.0`, `pygments>=2.18.0`, `matplotlib>=3.9`) to `pyproject.toml`, installed into `.venv`, and ran `.venv\Scripts\python.exe -m playwright install chromium`.
- Added a `Pairs per page` selector (`[10, 25, 50]`, default `50`) to Pair Explorer in `src/tyto/app/mvp_app.py` so the script can limit the rendered table to 10 rows without changing default behavior.
- Created reproducible `scripts/make_report_assets.py` supporting `--out` (default `../tyto-report-assets`) and `--only` (`app`, `code`, `output`, `charts`, `docs`).
- Rendered all HTML/code/output/app screenshots with headless Chromium via Playwright (`device_scale_factor=2`, font stack `'Cascadia Mono', Consolas, 'Nirmala UI', 'Noto Sans Tamil', monospace`) so Tamil script shapes properly.

### Generated Asset Tree (`../tyto-report-assets`)
- `01_app/`: **14** PNGs (6 pages x viewport & full-page at 1440x900, plus `app_05_predictor_predefined_example.png` and `app_05_predictor_live_input.png`)
- `02_code/`: **86** PNGs (`load.py`, `features.py`, `label.py`, `mine.py`, `predict.py`, `export.py`, `mvp_app.py`, split at top-level definitions into chunks <= 55 lines)
- `03_output/`: **18** PNGs (real `PYTHONUTF8=1` outputs for `doctor`, `--help`, `mvp --help`, `mvp load/features/label/mine/predict/export`, `ruff check .`, `pytest -q`, `git log --oneline -8`, `scripts/survey_v2024_ta_integrity.py`, plus `pairs_v2018.parquet` head(8), risk tier counts, top 10 rules, cluster summary, `predictor_metrics.json`)
- `04_charts/`: **5** PNGs (200 dpi matplotlib charts from real artifacts with `'Risk is a heuristic proxy (no human labels).'` footnote)
- `MANIFEST.md` (with OpenSubtitles/OPUS attribution note) and `CAPTIONS.md`

### Automated Quality & Secret Checks
- Secret/PII regex scan passed on all `03_output/` captures.
- Automated PNG verification passed for all **123 PNGs**: every file >= 1600 px wide, > 15,360 bytes, and non-blank.

### Visual Inspection Log
- **Visually inspected (10 PNGs):**
  - `01_app/app_01_overview_viewport.png`
  - `01_app/app_02_film_explorer_viewport.png`
  - `01_app/app_03_pair_explorer_viewport.png` (confirmed 10 rows/page and Tamil rendering)
  - `01_app/app_04_patterns_viewport.png`
  - `01_app/app_05_predictor_predefined_example.png`
  - `01_app/app_05_predictor_live_input.png` (confirmed live LightGBM + LaBSE output `High / 0.7618`)
  - `01_app/app_06_about_limits_viewport.png`
  - `02_code/code_features_01_L1-34.png` (confirmed Tamil `TA_HEDGE_WORDS` shaping and line numbers)
  - `03_output/out_13_survey_v2024.png`
  - `03_output/out_14_pairs_head8.png` (confirmed Tamil subtitle shaping)
  - `04_charts/chart_01_risk_histogram.png`
  - `04_charts/chart_04_film_spaced_punct_and_latin.png`
  - `04_charts/chart_05_predictor_metrics_ci.png`
  _(13 PNGs total inspected directly.)_
- **Not visually inspected (110 PNGs):** Remaining full-page app captures (`7`), remaining `02_code/` chunks (`85`), remaining `03_output/` captures (`16`), and `04_charts/chart_02_mean_risk_per_film.png` & `chart_03_risk_tier_counts.png` (verified via automated dimension/size/non-blank checks only).

### Final Verification (Part B)

#### `.venv\Scripts\python.exe -m ruff check .` (exit code 0)
```
All checks passed!
```

#### `.venv\Scripts\python.exe -m pytest -q` (exit code 0)
```
........................................................................ [ 55%]
.........................................................                [100%]
```
