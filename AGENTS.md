# Project Tyto: agent rules

- Read PROJECT_BRIEF.md first and follow it on every task.
- Platform is Windows 11. Always run Python through the project venv: `.venv\Scripts\python.exe` (never the global Python).
- Install packages with `.venv\Scripts\python.exe -m pip install ...`.
- Work only on the current git branch. Never commit, push or merge. I do that myself.
- Never delete files outside this repo. Ask before installing anything system-wide.
- Never fabricate results. Every number must come from a generated artefact.
- A Python 3.11 venv already exists at .venv with CUDA torch installed. Do not install, upgrade or change torch, and do not create another environment.
- .gitignore already exists (GitHub's Python template plus project folders). Extend it; never replace it. Keep data/resources/ and data/gold/annotations.parquet tracked.
- Ask me before any command that installs something outside this repo.
- Never run bare `python`, `pip` or `pytest`. Always call `.venv\Scripts\python.exe -m ...` (for example `.venv\Scripts\python.exe -m pytest -q`). The global Python is 3.13 and is wrong for this project.
