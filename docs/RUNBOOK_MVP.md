# Project Tyto MVP — Runbook

Run all commands from the repository root (`project-tyto/`).
On Windows PowerShell, use `.venv\Scripts\python.exe` (or forward slashes in Git Bash as shown below). Never run bare `python`, `pip`, or `pytest` (global Python is 3.13; the project venv is Python 3.11 with CUDA PyTorch).

```bash
# Stage 2: Load OPUS v2018 data & run Tamil vowel-sign integrity gate
.venv/Scripts/python.exe -m tyto mvp load

# Stage 3: Extract cheap features + LaBSE similarity (GPU recommended for LaBSE)
.venv/Scripts/python.exe -m tyto mvp features

# Stage 4: Compute heuristic risk scores
.venv/Scripts/python.exe -m tyto mvp label

# Stage 5: Run pattern mining (FP-Growth rules, KMeans clusters, IsolationForest anomalies)
.venv/Scripts/python.exe -m tyto mvp mine

# Stage 6: Train GroupKFold predictor on heuristic risk proxy
.venv/Scripts/python.exe -m tyto mvp predict

# Stage 8: Export Power BI CSVs (UTF-8 BOM) + PowerBI_Build_Guide.pdf
.venv/Scripts/python.exe -m tyto mvp export

# Stage 7: Run interactive Streamlit review app
.venv/Scripts/streamlit.exe run src/tyto/app/mvp_app.py
```

### Windows PowerShell Note

In PowerShell, replace `/` with `\` if preferred (both work in PowerShell for `.venv\Scripts\python.exe`), and chain commands with `;` rather than `&&`:

```powershell
.venv\Scripts\python.exe -m tyto mvp load; .venv\Scripts\python.exe -m tyto mvp features
```
