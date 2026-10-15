# Project Tyto

Mines and predicts translation-quality problems (idiom literalisation, tone or intensity shift, omission, register mismatch) in English-Tamil movie subtitle pairs from OPUS OpenSubtitles.

---

## Environment & Requirements

- **Operating System**: Windows 11 (PowerShell)
- **Python**: 3.11 (`.venv\Scripts\python.exe`)
- **Hardware**: NVIDIA GeForce RTX 4060 (8 GB VRAM), 16 GB RAM
- **Character Encoding**: UTF-8 throughout. Set the environment variable:
  ```powershell
  $env:PYTHONUTF8 = "1"
  ```

---

## Installation

### 1. PyTorch with CUDA Wheels
PyTorch CUDA wheels are published on the official PyTorch wheel repository and should be installed before other dependencies:
```powershell
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

*(For CPU-only environments, use `--index-url https://download.pytorch.org/whl/cpu`)*

### 2. Install Project Tyto
Install in editable mode along with optional extras (`dev` for testing and linting, `app` for Streamlit/Plotly):
```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,app]"
```

---

## CLI Usage

Project Tyto provides a Typer CLI interface accessed via `python -m tyto`:

```powershell
# Display help and available command groups
.venv\Scripts\python.exe -m tyto --help

# Run environment diagnostics and GPU embedding smoke test
.venv\Scripts\python.exe -m tyto doctor

# Run doctor in CPU-only mode
.venv\Scripts\python.exe -m tyto doctor --cpu-ok
```

### Command Groups
- `doctor`: Hardware environment verification and LaBSE GPU smoke benchmark.
- `data`: Corpus ingestion, subtitle parsing, normalization, and split management. *(coming soon)*
- `features`: Multilingual embedding extraction and linguistic feature calculation. *(coming soon)*
- `mine`: Pattern mining, association rules, and subgroup discovery. *(coming soon)*
- `train`: Model training, hyperparameter optimization, and evaluation. *(coming soon)*
- `app`: Streamlit interactive subtitle inspection application. *(coming soon)*
- `export`: Power BI tabular exports and reporting artifacts. *(coming soon)*

---

## Testing & Code Quality

Run tests with `pytest`:
```powershell
.venv\Scripts\python.exe -m pytest -q
```

Run linter checks with `ruff`:
```powershell
.venv\Scripts\python.exe -m ruff check .
```

---

## Credits & Attribution

- **OpenSubtitles**: Subtitle corpus created and maintained by the OpenSubtitles community ([opensubtitles.org](https://www.opensubtitles.org)).
- **OPUS**: Subtitle data collection and parallel alignment provided by the OPUS corpus ([opus.nlpl.eu](https://opus.nlpl.eu)).
