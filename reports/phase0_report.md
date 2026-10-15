# Task P0 Report: Repository Scaffolding & Environment Doctor

## Assumptions
- **Host Platform**: Windows 11 with PowerShell running Python 3.11 inside `.venv` (`.venv\Scripts\python.exe`).
- **GPU Hardware**: NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM.
- **CUDA Runtime**: PyTorch 2.14.1+cu132 already installed in `.venv`.
- **UTF-8 Handling**: `$env:PYTHONUTF8 = "1"` is active or required for proper Tamil character handling.
- **Model Download**: Unauthenticated Hugging Face access is available for initial download of `sentence-transformers/LaBSE` weights (~1.88 GB).

---

## What Was Built

1. **Repository Layout**:
   - `src/tyto/` package with subpackages: `data`, `features`, `mining`, `modeling`, `app`, `utils`.
   - Top-level directories: `configs/`, `tests/`, `docs/`, `reports/`, `artifacts/`, `models/`.
   - Data directories: `data/raw`, `data/interim`, `data/processed`, `data/gold`, `data/resources`.

2. **Package & Dependencies (`pyproject.toml`)**:
   - Pinned exact versions for base data science and NLP dependencies: `pydantic`, `pyyaml`, `typer`, `rich`, `psutil`, `pandas`, `pyarrow`, `duckdb`, `transformers`, `sentence-transformers`, `scikit-learn`, `lightgbm`, `mlxtend`, `hdbscan`, `umap-learn`, `shap`, `faiss-cpu`, `srt`, `mlflow`.
   - Pinned extras: `[gpu]` (`accelerate`), `[app]` (`streamlit`, `plotly`), `[dev]` (`pytest`, `ruff`, `pre-commit`).
   - PyTorch CUDA wheel install step documented in `README.md` (pointing to `https://download.pytorch.org/whl/cu121`) rather than pinning `torch` in base dependencies to prevent PyPI CPU overwrites.

3. **Configuration & Seed Utilities**:
   - `configs/default.yaml`: Project paths, hardware/batch settings, model configurations, and translation error taxonomy labels/severities.
   - `src/tyto/utils/config.py`: Strongly typed Pydantic models (`TytoConfig`, `PathsConfig`, `DeviceConfig`, `ModelsConfig`, `TaxonomyConfig`) with `load_config()`.
   - `src/tyto/utils/seed.py`: Deterministic seed utility `set_seed()` setting seeds across Python `random`, `numpy`, and `torch` (including CUDA).

4. **Typer CLI (`python -m tyto`)**:
   - `src/tyto/cli.py` & `src/tyto/__main__.py`: Typer CLI with 7 command groups:
     - `doctor`: Implemented hardware health diagnostic and GPU smoke test.
     - `data`, `features`, `mine`, `train`, `app`, `export`: Stubs printing `"not implemented yet"`.

5. **Doctor Health Check (`python -m tyto doctor`)**:
   - Collects and tabulates:
     - OS version
     - Python interpreter path and version
     - PyTorch version and CUDA availability
     - GPU device name, total VRAM, and free VRAM
     - Host RAM (total & available via `psutil`)
     - Free disk space
   - Executes an embedding smoke test using `sentence-transformers/LaBSE` on GPU, measuring latency and peak memory allocation.
   - Enforces CUDA presence, exiting with code 1 if CUDA is missing unless `--cpu-ok` is provided.
   - Automatically unloads model and calls `torch.cuda.empty_cache()` upon test completion.

6. **ModelRegistry (`src/tyto/modeling/registry.py`)**:
   - Lazy model loading via `get(name, device=..., fp16=...)`.
   - Enforces single large model in GPU memory (unloading previously active model when another large model is requested).
   - Explicit `unload(name)` releasing memory, triggering garbage collection and `torch.cuda.empty_cache()`.
   - `vram_report()` returning real-time VRAM allocation and free memory.
   - Pre-registered `labse` loader.
   - `FakeTinyModel` mock encoder supporting sub-second CPU and GPU testing without network or disk overhead.

7. **Git & CI Infrastructure**:
   - `.gitignore`: Extended to protect `data/raw`, `data/interim`, `data/processed`, `models/`, `artifacts/`, and caches, while explicitly keeping `data/resources/` and `data/gold/annotations.parquet` tracked.
   - `.gitattributes`: Normalizes LF line endings and UTF-8 across all Python, YAML, JSON, Markdown, and text files.
   - `pyproject.toml` tool sections for `ruff` and `pytest`.
   - `.pre-commit-config.yaml`: Pre-commit hooks for formatting and linting.
   - `.github/workflows/ci.yml`: CPU-only GitHub Actions workflow running `ruff check .` and `pytest -q`.

8. **Test Suite (`tests/`)**:
   - `test_config.py`: Default config loading, missing file error handling, custom config overrides.
   - `test_registry.py`: Registration, lazy loading, fake model encoding shapes, explicit unload, VRAM reporting.
   - `test_cli.py`: Root `--help` verification, placeholder command group output verification, `doctor --help`, and `doctor` execution with fake model.
   - `test_seed.py`: Deterministic seed reproducibility across NumPy and PyTorch.

---

## How to Run

```powershell
# Set UTF-8 encoding
$env:PYTHONUTF8 = "1"

# 1. View Typer CLI Help
.venv\Scripts\python.exe -m tyto --help

# 2. Run Environment Doctor & GPU LaBSE Smoke Test
.venv\Scripts\python.exe -m tyto doctor

# 3. Run Doctor in CPU-Only Mode
.venv\Scripts\python.exe -m tyto doctor --cpu-ok

# 4. Run Pytest Suite
.venv\Scripts\python.exe -m pytest -q

# 5. Run Ruff Linter
.venv\Scripts\python.exe -m ruff check .
```

---

## Test & Execution Output

### 1. `python -m tyto --help`
```text
 Usage: python -m tyto [OPTIONS] COMMAND [ARGS]...

 Project Tyto: Translation-quality mining and prediction CLI.

┌─ Options ───────────────────────────────────────────────────────────────────┐
│ --help          Show this message and exit.                                 │
└─────────────────────────────────────────────────────────────────────────────┘
┌─ Commands ──────────────────────────────────────────────────────────────────┐
│ doctor    System health check and GPU LaBSE smoke test.                     │
│ data      Data ingestion, preprocessing, and splitting.                     │
│ features  Feature extraction and embeddings.                                │
│ mine      Pattern mining and association rules.                             │
│ train     Model training and evaluation.                                    │
│ app       Streamlit review application.                                     │
│ export    Power BI and artifact export.                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. `python -m tyto doctor`
```text
┌─────────────────────────────────────────────┐
│ Project Tyto: Environment & Hardware Doctor │
└─────────────────────────────────────────────┘
                              System Environment
┌──────────────────┬──────────────────────────────────────────────────────────┐
│ Property         │ Value                                                    │
├──────────────────┼──────────────────────────────────────────────────────────┤
│ Operating System │ Windows 10 (10.0.26300)                                  │
│ Python           │ 3.11.9                                                   │
│                  │ (C:\Users\Shift\Projects\project-tyto\.venv\Scripts\pyt… │
│ PyTorch Version  │ 2.14.1+cu132                                             │
│ CUDA Available   │ Available                                                │
│ GPU Name         │ NVIDIA GeForce RTX 4060 Laptop GPU                       │
│ GPU VRAM         │ Total: 8.00 GB (8188 MB), Free: 6.94 GB (7106 MB)        │
│ Host RAM         │ Total: 15.69 GB, Available: 3.68 GB                      │
│ Free Disk        │ Free: 66.43 GB / Total: 928.76 GB                        │
└──────────────────┴──────────────────────────────────────────────────────────┘

Running embedding smoke test with 'labse' on cuda...
Embedding Smoke Test Results
┌────────────┬─────────────┐
│ Metric     │ Value       │
├────────────┼─────────────┤
│ Model      │ labse       │
│ Device     │ cuda        │
│ Batch Size │ 2           │
│ Latency    │ 16932.89 ms │
│ Peak VRAM  │ 2533.09 MB  │
└────────────┴─────────────┘
Doctor check completed successfully.
```

### 3. `pytest -q`
```text
............                                                             [100%]
12 passed in 2.32s
```

### 4. `ruff check .`
```text
All checks passed!
```

---

## Known Issues & Notes

- **Initial Hugging Face Download Latency**: The first run of `doctor` takes ~56 seconds because the ~1.88 GB `sentence-transformers/LaBSE` model must be downloaded from the Hugging Face Hub. Subsequent runs with cached weights complete in ~16.9 seconds.
- **Windows Version Reporting**: `platform.system()` reports `Windows 10` for Windows 11 build `10.0.26300`, which is standard Python Windows API behavior.
- **Data/Feature Logic**: None implemented in accordance with TASK P0 specifications.
