PROJECT: Project Tyto (repo: project-tyto). Mines and predicts translation-quality problems (idiom literalisation, tone or intensity shift, omission, register mismatch) in English-Tamil movie subtitle pairs from OPUS OpenSubtitles. It is a Data Mining Techniques course project and a public portfolio project, so rigor, reproducibility and honesty matter more than feature count.

ENVIRONMENT: Windows 11 + PowerShell, Python 3.11, NVIDIA RTX 4060 8 GB VRAM, 16 GB RAM. Every command, script and path must work on Windows: use pathlib, no bash-only scripts, no Makefile; use a Typer CLI (python -m tyto <command>). Prefer packages with Windows wheels. Avoid bitsandbytes, triton and torch.compile. DataLoader num_workers=0 by default. Open every file with encoding utf-8 and document PYTHONUTF8=1; Tamil text must survive every read and write.

STACK: torch (CUDA), transformers, sentence-transformers, pandas, pyarrow, duckdb, scikit-learn, lightgbm, mlxtend, hdbscan, umap-learn, shap, faiss-cpu, plotly, streamlit, typer, pydantic, pyyaml, pytest, ruff, mlflow, srt. Pin versions in pyproject.toml. Check each library's docs and each model card for current usage; never rely on memory for model names or APIs.

ARCHITECTURE: package under src/tyto/ with subpackages data, features, mining, modeling, app, utils. Config in configs/*.yaml loaded with pydantic. Every stage reads and writes Parquet so stages rerun independently. Expensive outputs (embeddings, MT, sentiment) are cached, keyed by hash(text, model_id, model_version).

GPU RULES: 8 GB VRAM. Load models through a ModelRegistry with lazy load, explicit unload and torch.cuda.empty_cache(). Never keep more than one large model on the GPU. fp16 for inference. Batch size backs off automatically on CUDA OOM. Everything must also run on CPU (slowly) for tests and demo mode.

DATA RULES: never commit raw corpus or model weights; data/, models/ and artifacts/ are gitignored. Scripts download from OPUS. Credit OpenSubtitles and OPUS in the README. Split train, validation and test by FILM, never by line, to prevent leakage from adjacent and repeated dialogue.

TEXT RULES: normalise Unicode to NFC; strip subtitle markup, speaker dashes, music symbols and sound descriptions like [laughs]; keep a raw column untouched; handle Tamil script, romanised Tamil and code-mixed lines explicitly (flag them, never silently drop).

TAXONOMY: OK, LOCALISED_OK, IDIOM_LITERAL, TONE_SOFTENED, TONE_INTENSIFIED, OMISSION, ADDITION, MISTRANSLATION, REGISTER_MISMATCH, MISALIGNED. Severity 0, 1, 2. y_issue = any error tag except OK, LOCALISED_OK, MISALIGNED. y_severe = severity 2.

HONESTY RULES: never hardcode or fabricate results. Every number in the README, reports and app must come from a generated artefact. If a step cannot run (no GPU, no network, gated model), implement a CPU or stub path, say so in the report and list what the human must run. State assumptions at the top of each report instead of guessing silently.

QUALITY BAR: type hints, docstrings on public functions, small pure functions, logging instead of print, fixed and configurable seeds, pytest tests with tiny fixtures that run on CPU in under 60 seconds, ruff clean.

WORKFLOW: for each task write reports/<task>_report.md with what was built, how to run it, test output, assumptions and known issues. Do not add features beyond the task. Do not refactor unrelated files.