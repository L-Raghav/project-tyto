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
