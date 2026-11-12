"""Generate reproducible report screenshots, code listings, CLI outputs, and charts.

Usage (from repo root):
    .venv\\Scripts\\python.exe scripts/make_report_assets.py
    .venv\\Scripts\\python.exe scripts/make_report_assets.py --out ../tyto-report-assets --only charts
"""  # noqa: E501

from __future__ import annotations

import argparse
import ast
import html
import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from playwright.sync_api import Page, sync_playwright  # noqa: E402
from pygments import highlight  # noqa: E402
from pygments.formatters import HtmlFormatter  # noqa: E402
from pygments.lexers import PythonLexer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data"

FONT_STACK = "'Cascadia Mono', Consolas, 'Nirmala UI', 'Noto Sans Tamil', monospace"
FOOTNOTE_TEXT = "Risk is a heuristic proxy (no human labels)."
MAX_CODE_LINES = 55
MAX_OUTPUT_LINES = 40
MIN_WIDTH_PX = 1600
MIN_BYTES = 15 * 1024

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    (
        "GitHub/API token",
        re.compile(r"(?:ghp_|github_pat_|sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})"),
    ),
    (
        "credential assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|api_key|secret_key|access_token|auth_token|private_key)"
            r"\s*[:=]\s*\S+"
        ),
    ),
]


@dataclass
class AssetRecord:
    """Metadata for a generated PNG asset used to build MANIFEST.md and CAPTIONS.md."""

    rel_path: str
    shows: str
    source: str
    chapter: str
    caption: str
    generated_at: str


def _now_iso() -> str:
    return datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _check_secrets(text: str, label: str) -> None:
    """Abort immediately if any captured output contains secret-like strings or emails."""
    for kind, pattern in SECRET_PATTERNS:
        m = pattern.search(text)
        if m:
            raise RuntimeError(
                f"Secret/PII check failed in '{label}' ({kind}): matched '{m.group(0)[:20]}...'"
            )


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Read (width, height) from PNG IHDR header without Pillow."""
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a valid PNG: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _is_blank_png(path: Path) -> bool:
    """Return True if image has near-zero variance (solid/blank)."""
    arr = mpimg.imread(str(path))
    return float(np.std(arr)) < 1e-4


def _verify_pngs(out_dir: Path) -> list[str]:
    """Verify every PNG is >=1600px wide, >15KB, and non-blank. Return list of failures."""
    failures: list[str] = []
    for p in sorted(out_dir.glob("*/*.png")):
        size = p.stat().st_size
        w, h = _png_dimensions(p)
        blank = _is_blank_png(p)
        if w < MIN_WIDTH_PX or size <= MIN_BYTES or blank:
            failures.append(f"{p.relative_to(out_dir)}: {w}x{h} px, {size} bytes, blank={blank}")
    return failures


def _render_html_to_png(
    page: Page,
    html_str: str,
    out_path: Path,
    base_width: int = 1100,
) -> None:
    """Render HTML in headless Chromium (device_scale_factor=2) and save full-page PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.set_viewport_size({"width": base_width, "height": 800})
    page.set_content(html_str, wait_until="load")
    scroll_w = page.evaluate("() => Math.ceil(document.documentElement.scrollWidth)")
    target_w = max(base_width, int(scroll_w) + 24)
    if target_w != base_width:
        page.set_viewport_size({"width": target_w, "height": 800})
    page.wait_for_timeout(150)
    page.screenshot(path=str(out_path), full_page=True)


# ---------------------------------------------------------------------------
# 1. Streamlit App Screenshots (01_app/)
# ---------------------------------------------------------------------------


def _find_free_port(preferred: int = 8599) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred)) != 0:
            return preferred
    raise RuntimeError(f"Port {preferred} is already in use; please free it before running.")


def _wait_for_health(port: int, timeout_s: float = 45.0) -> None:
    url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"Streamlit server did not become healthy at {url} within {timeout_s}s")


def _wait_streamlit_render(page: Page, extra_ms: int = 1800) -> None:
    page.locator('[data-testid="stAppViewContainer"]').wait_for(state="visible", timeout=20000)
    try:
        page.locator('[data-testid="stStatusWidget"]').wait_for(state="hidden", timeout=20000)
    except Exception:
        pass
    try:
        page.locator(".stSkeleton").first.wait_for(state="detached", timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(extra_ms)


def _select_sidebar_page(page: Page, label: str) -> None:
    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar.get_by_text(label, exact=True).click()
    _wait_streamlit_render(page)


def _select_streamlit_selectbox(
    page: Page, container_locator: Any, label_text: str, option_text: str
) -> None:
    """Select an option in a Streamlit BaseWeb selectbox."""
    combo = container_locator.get_by_role("combobox", name=label_text).first
    combo.scroll_into_view_if_needed()
    combo.click()
    page.wait_for_timeout(350)
    page.get_by_role("option").filter(has_text=option_text).first.click()
    _wait_streamlit_render(page, extra_ms=1400)


def generate_app_assets(out_dir: Path, records: list[AssetRecord]) -> None:
    """Start headless Streamlit on port 8599 and capture all 6 pages + Predictor states."""
    app_dir = out_dir / "01_app"
    app_dir.mkdir(parents=True, exist_ok=True)
    port = _find_free_port(8599)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/tyto/app/mvp_app.py",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(port, timeout_s=45.0)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                device_scale_factor=2,
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
            _wait_streamlit_render(page, extra_ms=2200)

            pages_meta = [
                (
                    "Overview",
                    "app_01_overview",
                    "Streamlit Overview page showing KPIs, mean risk by film, and v2018 integrity.",
                    "System Overview & Corpus Statistics",
                    "Overview page of the Project Tyto Streamlit MVP showing 32,417 aligned "
                    "pairs across 28 films (High: 3,242; Medium: 6,483; Low: 22,692), per-film "
                    "mean heuristic risk scores, and v2018 Tamil vowel-sign verification.",
                ),
                (
                    "Film Explorer",
                    "app_02_film_explorer",
                    "Streamlit Film Explorer page with per-film metrics, donut chart, and pairs.",
                    "Exploratory Analysis",
                    "Film Explorer page displaying per-film metrics, risk-tier donut "
                    "distribution, and top high-risk English-Tamil subtitle pair cards.",
                ),
                (
                    "Pair Explorer",
                    "app_03_pair_explorer",
                    "Streamlit Pair Explorer page (10 rows/page) with sidebar filters and cards.",
                    "Exploratory Analysis",
                    "Pair Explorer page configured to 10 rows per page, showing English-Tamil "
                    "subtitle pair cards with risk scores, LaBSE cosine similarity, and flags.",
                ),
                (
                    "Patterns",
                    "app_04_patterns",
                    "Streamlit Patterns page with FP-Growth rules, KMeans clusters, and anomalies.",
                    "Pattern Mining",
                    "Patterns page presenting 15 FP-Growth association rules, KMeans cluster "
                    "summary (k=8), cluster exemplar cards, and top 30 IsolationForest anomalies.",
                ),
                (
                    "Predictor",
                    "app_05_predictor",
                    "Streamlit Predictor page with GroupKFold CV metrics and interactive tabs.",
                    "Predictive Modelling",
                    "Source-Only Predictor page displaying 5-fold GroupKFold metrics (LightGBM "
                    "AUROC 0.6994, LogisticRegression 0.6990, Baseline 0.5723) and live UI.",
                ),
                (
                    "About & Limits",
                    "app_06_about_limits",
                    "Streamlit About & Limits page detailing real vs illustrative items & limits.",
                    "Limitations & Ethics",
                    "About & Limits page detailing real corpus artifacts, illustrative lexicons, "
                    "absence of human quality labels, spaced-punctuation rates, and v2024 notes.",
                ),
            ]

            for nav_label, stem, shows, chapter, caption in pages_meta:
                _select_sidebar_page(page, nav_label)
                if nav_label == "Pair Explorer":
                    sidebar = page.locator('[data-testid="stSidebar"]')
                    _select_streamlit_selectbox(page, sidebar, "Pairs per page", "10")
                    page.get_by_text("Showing rows 1–10", exact=False).wait_for(
                        state="visible", timeout=10000
                    )
                    _wait_streamlit_render(page, extra_ms=1000)

                vp_path = app_dir / f"{stem}_viewport.png"
                full_path = app_dir / f"{stem}_full.png"
                page.screenshot(path=str(vp_path), full_page=False)
                page.screenshot(path=str(full_path), full_page=True)

                ts = _now_iso()
                records.append(
                    AssetRecord(
                        f"01_app/{vp_path.name}",
                        f"{shows} (1440x900 viewport)",
                        "src/tyto/app/mvp_app.py (port 8599)",
                        chapter,
                        f"{caption} (Viewport capture at 1440x900.)",
                        ts,
                    )
                )
                records.append(
                    AssetRecord(
                        f"01_app/{full_path.name}",
                        f"{shows} (full page)",
                        "src/tyto/app/mvp_app.py (port 8599)",
                        chapter,
                        caption,
                        ts,
                    )
                )

                if nav_label == "Predictor":
                    main_area = page.locator('[data-testid="stMain"]')
                    _select_streamlit_selectbox(
                        page, main_area, "Choose an ILLUSTRATIVE input sentence", "[3]"
                    )
                    pre_path = app_dir / "app_05_predictor_predefined_example.png"
                    page.screenshot(path=str(pre_path), full_page=True)
                    records.append(
                        AssetRecord(
                            f"01_app/{pre_path.name}",
                            "Predictor page with illustrative example #3 selected.",
                            "src/tyto/app/mvp_app.py + data/resources/demo_examples.json",
                            "Predictive Modelling",
                            "Predictor page with predefined illustrative example [3] "
                            "(anger_intense) selected, displaying stored risk tier, "
                            "probability, and feature contribution chart.",
                            _now_iso(),
                        )
                    )

                    textarea = main_area.locator("textarea").first
                    textarea.fill("He kicked the bucket at 10 PM, didn't he? What a tragedy!")
                    textarea.blur()
                    _wait_streamlit_render(page, extra_ms=800)
                    btn = main_area.get_by_role("button", name="Predict Risk Proxy")
                    btn.scroll_into_view_if_needed()
                    btn.click()
                    main_area.get_by_text(
                        "Real model output (predicting heuristic proxy", exact=False
                    ).wait_for(state="visible", timeout=120000)
                    _wait_streamlit_render(page, extra_ms=2000)
                    live_path = app_dir / "app_05_predictor_live_input.png"
                    page.screenshot(path=str(live_path), full_page=True)
                    records.append(
                        AssetRecord(
                            f"01_app/{live_path.name}",
                            "Predictor page with live free-text sentence scored by LightGBM.",
                            "src/tyto/app/mvp_app.py + artifacts/lgbm_model.pkl",
                            "Predictive Modelling",
                            "Live Predictor result for a custom English sentence "
                            "('He kicked the bucket at 10 PM, didn't he? What a tragedy!'), "
                            "showing predicted risk tier, probability, and feature impacts.",
                            _now_iso(),
                        )
                    )

            browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# 2. Code Listings (02_code/)
# ---------------------------------------------------------------------------


def _split_code_chunks(path: Path, max_lines: int = MAX_CODE_LINES) -> list[tuple[int, int, str]]:
    """Split a Python file at top-level definitions into chunks of <= max_lines."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []

    tree = ast.parse(text)
    def_starts: list[int] = sorted(
        {
            node.lineno
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
    )

    boundaries = [1] + [ln for ln in def_starts if ln > 1] + [n + 1]
    segments: list[tuple[int, int]] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1] - 1
        if s <= e:
            segments.append((s, e))

    chunks: list[tuple[int, int, str]] = []
    cur_s: int | None = None
    cur_e: int | None = None

    def _flush() -> None:
        nonlocal cur_s, cur_e
        if cur_s is not None and cur_e is not None:
            chunk_text = "\n".join(lines[cur_s - 1 : cur_e])
            chunks.append((cur_s, cur_e, chunk_text))
            cur_s, cur_e = None, None

    for s, e in segments:
        seg_len = e - s + 1
        if seg_len <= max_lines:
            if cur_s is None:
                cur_s, cur_e = s, e
            elif e - cur_s + 1 <= max_lines:
                cur_e = e
            else:
                _flush()
                cur_s, cur_e = s, e
        else:
            _flush()
            pos = s
            while pos <= e:
                limit = min(pos + max_lines - 1, e)
                if limit < e:
                    blank_candidates = [
                        idx for idx in range(limit, pos + 10, -1) if not lines[idx - 1].strip()
                    ]
                    if blank_candidates:
                        limit = blank_candidates[0]
                chunks.append((pos, limit, "\n".join(lines[pos - 1 : limit])))
                pos = limit + 1

    _flush()
    return chunks


def generate_code_assets(out_dir: Path, records: list[AssetRecord]) -> None:
    """Render chunked Python source files via Pygments HtmlFormatter + Playwright."""
    code_dir = out_dir / "02_code"
    code_dir.mkdir(parents=True, exist_ok=True)

    target_files = [
        REPO_ROOT / "src/tyto/mvp/load.py",
        REPO_ROOT / "src/tyto/mvp/features.py",
        REPO_ROOT / "src/tyto/mvp/label.py",
        REPO_ROOT / "src/tyto/mvp/mine.py",
        REPO_ROOT / "src/tyto/mvp/predict.py",
        REPO_ROOT / "src/tyto/mvp/export.py",
        REPO_ROOT / "src/tyto/app/mvp_app.py",
    ]

    lexer = PythonLexer()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1100, "height": 800},
            device_scale_factor=2,
        )
        page = context.new_page()

        for fpath in target_files:
            rel_src = fpath.relative_to(REPO_ROOT).as_posix()
            mod_name = fpath.stem
            chunks = _split_code_chunks(fpath, MAX_CODE_LINES)
            for idx, (start_ln, end_ln, chunk_code) in enumerate(chunks, start=1):
                formatter = HtmlFormatter(
                    style="friendly",
                    linenos="table",
                    linenostart=start_ln,
                    cssclass="highlight",
                )
                pyg_css = formatter.get_style_defs(".highlight")
                body_html = highlight(chunk_code, lexer, formatter)
                doc_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 20px;
    background: #ffffff;
    font-family: {FONT_STACK};
    color: #1f2328;
  }}
  .card {{
    border: 1px solid #d0d7de;
    border-radius: 8px;
    overflow: hidden;
    background: #ffffff;
  }}
  .caption-bar {{
    background: #f6f8fa;
    border-bottom: 1px solid #d0d7de;
    padding: 10px 16px;
    font-size: 14px;
    font-weight: 600;
    color: #24292f;
    display: flex;
    justify-content: space-between;
  }}
  .code-wrap {{
    padding: 10px 14px;
    font-size: 13.5px;
    line-height: 1.45;
  }}
  table.highlighttable {{
    border-collapse: collapse;
    width: 100%;
  }}
  td.linenos {{
    padding-right: 12px;
    border-right: 1px solid #eaeef2;
    color: #6e7781;
    user-select: none;
    vertical-align: top;
    text-align: right;
  }}
  td.code {{
    padding-left: 14px;
    vertical-align: top;
    width: 100%;
  }}
  pre {{
    margin: 0;
    font-family: {FONT_STACK};
    white-space: pre;
  }}
  {pyg_css}
</style>
</head>
<body>
  <div class="card">
    <div class="caption-bar">
      <span>{html.escape(rel_src)}</span>
      <span>Lines L{start_ln}–L{end_ln} ({end_ln - start_ln + 1} lines)</span>
    </div>
    <div class="code-wrap">{body_html}</div>
  </div>
</body>
</html>"""
                fname = f"code_{mod_name}_{idx:02d}_L{start_ln}-{end_ln}.png"
                out_path = code_dir / fname
                _render_html_to_png(page, doc_html, out_path, base_width=1100)
                records.append(
                    AssetRecord(
                        f"02_code/{fname}",
                        f"Source listing of {rel_src} lines {start_ln}-{end_ln}.",
                        f"{rel_src}:L{start_ln}-L{end_ln}",
                        "Implementation & Code Appendix",
                        f"Syntax-highlighted code for {rel_src} (lines {start_ln}–{end_ln}).",
                        _now_iso(),
                    )
                )

        browser.close()


# ---------------------------------------------------------------------------
# 3. Command Outputs & Artifact Snapshots (03_output/)
# ---------------------------------------------------------------------------


def _truncate_output(text: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    """Truncate output to at most max_lines with an explicit marker."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    kept = lines[: max_lines - 1]
    remaining = len(lines) - (max_lines - 1)
    kept.append(f"... ({remaining} more lines)")
    return "\n".join(kept)


def _run_capture(cmd: list[str], display_cmd: str) -> tuple[str, int]:
    """Run command with PYTHONUTF8=1 and return combined stdout+stderr and returncode."""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = proc.stdout or ""
    if proc.stderr:
        combined = f"{combined}\n{proc.stderr}" if combined else proc.stderr
    _check_secrets(combined, display_cmd)
    return combined.rstrip(), proc.returncode


def _render_terminal_html(prompt_cmd: str, body: str, exit_code: int | None = None) -> str:
    status_badge = ""
    if exit_code is not None:
        color = "#1a7f37" if exit_code == 0 else "#cf222e"
        status_badge = f"<span style='color:{color};font-weight:600;'>exit={exit_code}</span>"
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{
    margin: 0;
    padding: 22px;
    background: #ffffff;
    font-family: {FONT_STACK};
    color: #1f2328;
  }}
  .term {{
    border: 1px solid #d0d7de;
    border-radius: 8px;
    overflow: hidden;
    background: #fbfcfd;
  }}
  .prompt {{
    background: #f3f5f7;
    border-bottom: 1px solid #d0d7de;
    padding: 12px 16px;
    font-size: 14px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .prompt-cmd {{
    font-weight: 700;
    color: #0969da;
  }}
  pre.out {{
    margin: 0;
    padding: 16px 18px;
    font-family: {FONT_STACK};
    font-size: 13.5px;
    line-height: 1.48;
    white-space: pre-wrap;
    word-break: break-word;
    background: #ffffff;
  }}
</style>
</head>
<body>
  <div class="term">
    <div class="prompt">
      <span class="prompt-cmd">PS&gt; {html.escape(prompt_cmd)}</span>
      {status_badge}
    </div>
    <pre class="out">{html.escape(body)}</pre>
  </div>
</body>
</html>"""


def generate_output_assets(out_dir: Path, records: list[AssetRecord]) -> None:
    """Run real CLI/test/git/survey commands and artifact snapshots, then screenshot."""
    output_dir = out_dir / "03_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str], str, str, str]] = [
        (
            "out_01_doctor.png",
            [sys.executable, "-m", "tyto", "doctor"],
            "python -m tyto doctor",
            "Environment and CUDA/dependency check via `python -m tyto doctor`.",
            "Environment & Reproducibility",
        ),
        (
            "out_02_tyto_help.png",
            [sys.executable, "-m", "tyto", "--help"],
            "python -m tyto --help",
            "Top-level Tyto CLI help output.",
            "System Overview & CLI",
        ),
        (
            "out_03_mvp_help.png",
            [sys.executable, "-m", "tyto", "mvp", "--help"],
            "python -m tyto mvp --help",
            "MVP subcommand group help output.",
            "System Overview & CLI",
        ),
        (
            "out_04_mvp_load.png",
            [sys.executable, "-m", "tyto", "mvp", "load"],
            "python -m tyto mvp load",
            "OPUS v2018 corpus loading and Tamil vowel-sign integrity gate output.",
            "Data Ingestion & Quality",
        ),
        (
            "out_05_mvp_features.png",
            [sys.executable, "-m", "tyto", "mvp", "features"],
            "python -m tyto mvp features",
            "Surface feature extraction and LaBSE cosine similarity run output.",
            "Feature Engineering",
        ),
        (
            "out_06_mvp_label.png",
            [sys.executable, "-m", "tyto", "mvp", "label"],
            "python -m tyto mvp label",
            "Heuristic risk scoring and percentile tiering output.",
            "Heuristic Risk Scoring",
        ),
        (
            "out_07_mvp_mine.png",
            [sys.executable, "-m", "tyto", "mvp", "mine"],
            "python -m tyto mvp mine",
            "Pattern mining (FP-Growth, KMeans, IsolationForest) run output.",
            "Pattern Mining",
        ),
        (
            "out_08_mvp_predict.png",
            [sys.executable, "-m", "tyto", "mvp", "predict"],
            "python -m tyto mvp predict",
            "GroupKFold source-only risk-proxy predictor training and CV output.",
            "Predictive Modelling",
        ),
        (
            "out_09_mvp_export.png",
            [sys.executable, "-m", "tyto", "mvp", "export"],
            "python -m tyto mvp export",
            "Power BI UTF-8 BOM CSV and Build Guide PDF export output.",
            "Reporting & Export",
        ),
        (
            "out_10_ruff.png",
            [sys.executable, "-m", "ruff", "check", "."],
            "python -m ruff check .",
            "Ruff static lint check output.",
            "Software Quality & Testing",
        ),
        (
            "out_11_pytest.png",
            [sys.executable, "-m", "pytest", "-q"],
            "python -m pytest -q",
            "Pytest unit test suite output.",
            "Software Quality & Testing",
        ),
        (
            "out_12_git_log.png",
            ["git", "log", "--oneline", "-8"],
            "git log --oneline -8",
            "Recent git commit history on the active branch.",
            "Reproducibility & Provenance",
        ),
    ]

    v2024_zip = DATA_DIR / "raw" / "opus" / "v2024" / "ta.zip"
    if v2024_zip.exists():
        commands.append(
            (
                "out_13_survey_v2024.png",
                [sys.executable, "scripts/survey_v2024_ta_integrity.py"],
                "python scripts/survey_v2024_ta_integrity.py",
                "OPUS v2024 Tamil vowel-sign integrity survey output.",
                "Data Ingestion & Quality",
            )
        )

    snapshots: list[tuple[str, str, str, str, str]] = []

    pairs_df = pd.read_parquet(DATA_DIR / "processed" / "pairs_v2018.parquet")
    head8 = pairs_df[["pair_id", "film_id", "link_type", "en", "ta"]].head(8).copy()
    head8["en"] = head8["en"].astype(str).str.slice(0, 42)
    head8["ta"] = head8["ta"].astype(str).str.slice(0, 42)
    snapshots.append(
        (
            "out_14_pairs_head8.png",
            "head(8) of data/processed/pairs_v2018.parquet",
            head8.to_string(index=False),
            "First 8 aligned English-Tamil subtitle pairs from pairs_v2018.parquet.",
            "Data Ingestion & Quality",
        )
    )

    risk_df = pd.read_parquet(ARTIFACTS_DIR / "risk.parquet")
    tier_counts = (
        risk_df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0)
    )
    tier_tbl = pd.DataFrame(
        {
            "risk_tier": tier_counts.index,
            "count": tier_counts.values,
            "pct": [f"{(c / len(risk_df)) * 100:.2f}%" for c in tier_counts.values],
        }
    )
    tier_text = (
        f"Total pairs: {len(risk_df):,} | "
        f"mean={risk_df['risk_score'].mean():.2f}, "
        f"p70={np.percentile(risk_df['risk_score'], 70):.2f}, "
        f"p90={np.percentile(risk_df['risk_score'], 90):.2f}\n\n"
        + tier_tbl.to_string(index=False)
    )
    snapshots.append(
        (
            "out_15_risk_tier_counts.png",
            "Risk tier counts from artifacts/risk.parquet",
            tier_text,
            "Heuristic risk tier distribution (Low, Medium, High) from risk.parquet.",
            "Heuristic Risk Scoring",
        )
    )

    rules_df = pd.read_parquet(ARTIFACTS_DIR / "association_rules.parquet")
    top_rules = (
        rules_df[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).copy()
    )
    for col in ("support", "confidence", "lift"):
        top_rules[col] = top_rules[col].map(lambda v: f"{float(v):.4f}")
    snapshots.append(
        (
            "out_16_top10_rules.png",
            "Top 10 FP-Growth association rules (artifacts/association_rules.parquet)",
            top_rules.to_string(index=False),
            "Top 10 FP-Growth association rules ranked by lift.",
            "Pattern Mining",
        )
    )

    cl_df = pd.read_parquet(ARTIFACTS_DIR / "clusters.parquet")
    cl_summary = (
        cl_df.groupby("cluster_id", as_index=False)
        .agg(
            size=("pair_id", "count"),
            mean_risk=("risk_score", "mean"),
            mean_en_words=("en_words", "mean"),
            mean_ta_chars=("ta_chars", "mean"),
            mean_labse_cos=("labse_cos", "mean"),
            spaced_punct=("ta_spaced_punct", "mean"),
            latin_share=("ta_latin_share", "mean"),
        )
        .sort_values("cluster_id")
    )
    fmt_cols = (
        "mean_risk",
        "mean_en_words",
        "mean_ta_chars",
        "mean_labse_cos",
        "spaced_punct",
        "latin_share",
    )
    for col in fmt_cols:
        cl_summary[col] = cl_summary[col].map(lambda v: f"{float(v):.3f}")
    snapshots.append(
        (
            "out_17_cluster_summary.png",
            "KMeans cluster summary (artifacts/clusters.parquet)",
            cl_summary.to_string(index=False),
            "Per-cluster summary statistics across all KMeans clusters.",
            "Pattern Mining",
        )
    )

    metrics_obj = json.loads((ARTIFACTS_DIR / "predictor_metrics.json").read_text(encoding="utf-8"))
    metrics_pretty = json.dumps(metrics_obj, indent=2)
    snapshots.append(
        (
            "out_18_predictor_metrics.png",
            "artifacts/predictor_metrics.json (pretty-printed)",
            metrics_pretty,
            "Pretty-printed GroupKFold cross-validation metrics from predictor_metrics.json.",
            "Predictive Modelling",
        )
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1150, "height": 700},
            device_scale_factor=2,
        )
        page = context.new_page()

        for fname, cmd, display_cmd, shows, chapter in commands:
            raw_out, rc = _run_capture(cmd, display_cmd)
            if fname == "out_13_survey_v2024.png":
                lines = raw_out.splitlines()
                if len(lines) > MAX_OUTPUT_LINES:
                    head_lines = lines[:18]
                    tail_lines = lines[-15:]
                    omitted = len(lines) - len(head_lines) - len(tail_lines)
                    truncated = "\n".join(
                        head_lines + [f"... ({omitted} more lines) ..."] + tail_lines
                    )
                else:
                    truncated = raw_out
            else:
                truncated = _truncate_output(raw_out, MAX_OUTPUT_LINES)

            _check_secrets(truncated, fname)
            html_doc = _render_terminal_html(display_cmd, truncated, exit_code=rc)
            out_path = output_dir / fname
            _render_html_to_png(page, html_doc, out_path, base_width=1150)
            records.append(
                AssetRecord(
                    f"03_output/{fname}",
                    shows,
                    display_cmd,
                    chapter,
                    f"Real captured terminal output of `{display_cmd}` (exit code {rc}).",
                    _now_iso(),
                )
            )

        for fname, prompt_lbl, raw_text, shows, chapter in snapshots:
            _check_secrets(raw_text, fname)
            truncated = _truncate_output(raw_text, MAX_OUTPUT_LINES)
            html_doc = _render_terminal_html(prompt_lbl, truncated, exit_code=None)
            out_path = output_dir / fname
            _render_html_to_png(page, html_doc, out_path, base_width=1150)
            records.append(
                AssetRecord(
                    f"03_output/{fname}",
                    shows,
                    prompt_lbl,
                    chapter,
                    f"Rendered snapshot of `{prompt_lbl}`.",
                    _now_iso(),
                )
            )

        browser.close()


# ---------------------------------------------------------------------------
# 4. Matplotlib Charts (04_charts/)
# ---------------------------------------------------------------------------


def _style_chart(ax: plt.Axes, fig: plt.Figure, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.5,
        0.015,
        FOOTNOTE_TEXT,
        ha="center",
        fontsize=9,
        color="#555555",
        style="italic",
    )


def generate_chart_assets(out_dir: Path, records: list[AssetRecord]) -> None:
    """Generate 200-dpi matplotlib figures directly from artifacts/*.parquet & JSON."""
    charts_dir = out_dir / "04_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    risk_df = pd.read_parquet(ARTIFACTS_DIR / "risk.parquet")
    film_df = pd.read_parquet(ARTIFACTS_DIR / "film_stats.parquet")
    metrics = json.loads((ARTIFACTS_DIR / "predictor_metrics.json").read_text(encoding="utf-8"))

    # 1. Risk score histogram with tier thresholds
    scores = risk_df["risk_score"].astype(float).to_numpy()
    p70 = float(np.percentile(scores, 70))
    p90 = float(np.percentile(scores, 90))
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=200, facecolor="white")
    ax.hist(scores, bins=40, color="#3b82f6", edgecolor="white", alpha=0.88)
    ax.axvline(
        p70, color="#f59e0b", linestyle="--", linewidth=2, label=f"p70 (Medium >= {p70:.2f})"
    )
    ax.axvline(p90, color="#dc2626", linestyle="--", linewidth=2, label=f"p90 (High >= {p90:.2f})")
    ax.legend(frameon=True)
    _style_chart(
        ax,
        fig,
        f"Heuristic Risk Score Distribution (n={len(risk_df):,}, mean={scores.mean():.2f})",
        "Heuristic Risk Score (0-100)",
        "Subtitle Pair Count",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p1 = charts_dir / "chart_01_risk_histogram.png"
    fig.savefig(p1, dpi=200, facecolor="white")
    plt.close(fig)
    records.append(
        AssetRecord(
            f"04_charts/{p1.name}",
            "Histogram of heuristic risk_score with p70 and p90 tier thresholds.",
            "artifacts/risk.parquet",
            "Heuristic Risk Scoring",
            f"Distribution of heuristic risk scores across {len(risk_df):,} pairs "
            f"(mean {scores.mean():.2f}), with dashed thresholds at p70 ({p70:.2f}) "
            f"and p90 ({p90:.2f}).",
            _now_iso(),
        )
    )

    # 2. Mean risk per film
    film_sorted = film_df.sort_values("mean_risk", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=200, facecolor="white")
    ax.bar(film_sorted["film_id"].astype(str), film_sorted["mean_risk"], color="#2563eb", alpha=0.9)
    ax.tick_params(axis="x", rotation=55, labelsize=8.5)
    _style_chart(
        ax,
        fig,
        f"Mean Heuristic Risk Score by Film ({len(film_sorted)} Films)",
        "Film IMDb ID",
        "Mean Heuristic Risk Score (0-100)",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p2 = charts_dir / "chart_02_mean_risk_per_film.png"
    fig.savefig(p2, dpi=200, facecolor="white")
    plt.close(fig)
    top_film = film_sorted.iloc[0]
    bot_film = film_sorted.iloc[-1]
    records.append(
        AssetRecord(
            f"04_charts/{p2.name}",
            "Bar chart of mean heuristic risk score per film.",
            "artifacts/film_stats.parquet",
            "Exploratory Analysis",
            f"Mean heuristic risk score across all {len(film_sorted)} films, ranging from "
            f"{float(bot_film['mean_risk']):.2f} ({bot_film['film_id']}) to "
            f"{float(top_film['mean_risk']):.2f} ({top_film['film_id']}).",
            _now_iso(),
        )
    )

    # 3. Tier counts
    tier_order = ["Low", "Medium", "High"]
    tier_counts = [int((risk_df["risk_tier"] == t).sum()) for t in tier_order]
    tier_colors = ["#16a34a", "#f59e0b", "#dc2626"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=200, facecolor="white")
    bars = ax.bar(tier_order, tier_counts, color=tier_colors, width=0.55)
    for bar, cnt in zip(bars, tier_counts, strict=True):
        pct = cnt / len(risk_df) * 100.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + len(risk_df) * 0.012,
            f"{cnt:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, max(tier_counts) * 1.18)
    _style_chart(
        ax,
        fig,
        f"Heuristic Risk Tier Counts (Total = {len(risk_df):,} Pairs)",
        "Heuristic Risk Tier",
        "Subtitle Pair Count",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p3 = charts_dir / "chart_03_risk_tier_counts.png"
    fig.savefig(p3, dpi=200, facecolor="white")
    plt.close(fig)
    records.append(
        AssetRecord(
            f"04_charts/{p3.name}",
            "Bar chart of Low, Medium, and High heuristic risk tier counts.",
            "artifacts/risk.parquet",
            "Heuristic Risk Scoring",
            f"Heuristic risk tier counts across {len(risk_df):,} pairs: Low = "
            f"{tier_counts[0]:,}, Medium = {tier_counts[1]:,}, High = {tier_counts[2]:,}.",
            _now_iso(),
        )
    )

    # 4. Per-film spaced-punctuation and Latin-letter rates
    f_punct = film_df.sort_values("spaced_punct_rate", ascending=False).reset_index(drop=True)
    x = np.arange(len(f_punct))
    width = 0.42
    fig, ax = plt.subplots(figsize=(11, 6.0), dpi=200, facecolor="white")
    ax.bar(
        x - width / 2,
        f_punct["spaced_punct_rate"] * 100,
        width,
        label="Spaced Punctuation Rate (%)",
        color="#7c3aed",
    )
    ax.bar(
        x + width / 2,
        f_punct["latin_share"] * 100,
        width,
        label="Tamil Latin-Letter Share (%)",
        color="#0ea5e9",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(f_punct["film_id"].astype(str), rotation=55, fontsize=8.5)
    ax.legend(frameon=True)
    _style_chart(
        ax,
        fig,
        "Per-Film Spaced-Punctuation Rate and Tamil Latin-Letter Share",
        "Film IMDb ID (sorted by spaced-punctuation rate)",
        "Percentage (%)",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p4 = charts_dir / "chart_04_film_spaced_punct_and_latin.png"
    fig.savefig(p4, dpi=200, facecolor="white")
    plt.close(fig)
    max_sp = f_punct.iloc[0]
    records.append(
        AssetRecord(
            f"04_charts/{p4.name}",
            "Grouped bar chart of per-film spaced-punctuation rate and Latin-letter share.",
            "artifacts/film_stats.parquet",
            "Data Findings",
            f"Per-film spaced-punctuation rate and Latin-letter share across {len(f_punct)} "
            f"films (highest spaced-punctuation rate: {max_sp['film_id']} at "
            f"{float(max_sp['spaced_punct_rate']) * 100:.2f}%).",
            _now_iso(),
        )
    )

    # 5. Predictor metrics with 95% confidence intervals
    model_names = [m for m in ("baseline", "logistic", "lgbm") if m in metrics]
    if model_names:
        display_labels = {
            "baseline": "Baseline\n(en_words)",
            "logistic": "Logistic\nRegression",
            "lgbm": "LightGBM",
        }
        auroc_vals = [float(metrics[m]["auroc"]) for m in model_names]
        auroc_lo = [
            float(metrics[m]["auroc"]) - float(metrics[m]["auroc_ci"][0]) for m in model_names
        ]
        auroc_hi = [
            float(metrics[m]["auroc_ci"][1]) - float(metrics[m]["auroc"]) for m in model_names
        ]
        prauc_vals = [float(metrics[m]["pr_auc"]) for m in model_names]
        prauc_lo = [
            float(metrics[m]["pr_auc"]) - float(metrics[m]["pr_auc_ci"][0]) for m in model_names
        ]
        prauc_hi = [
            float(metrics[m]["pr_auc_ci"][1]) - float(metrics[m]["pr_auc"]) for m in model_names
        ]

        x = np.arange(len(model_names))
        w = 0.34
        fig, ax = plt.subplots(figsize=(9.5, 5.8), dpi=200, facecolor="white")
        b1 = ax.bar(
            x - w / 2,
            auroc_vals,
            w,
            yerr=[auroc_lo, auroc_hi],
            capsize=5,
            label="AUROC (95% CI)",
            color="#2563eb",
            alpha=0.9,
        )
        b2 = ax.bar(
            x + w / 2,
            prauc_vals,
            w,
            yerr=[prauc_lo, prauc_hi],
            capsize=5,
            label="PR-AUC (95% CI)",
            color="#059669",
            alpha=0.9,
        )
        for bar, val in zip(b1, auroc_vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )
        for bar, val in zip(b2, prauc_vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([display_labels.get(m, m) for m in model_names], fontsize=10)
        ax.set_ylim(0, 0.85)
        ax.legend(frameon=True)
        _style_chart(
            ax,
            fig,
            f"Source-Only Predictor GroupKFold CV "
            f"({metrics['n_folds']} folds, {metrics['n_groups']} groups)",
            "Model",
            "Score",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        p5 = charts_dir / "chart_05_predictor_metrics_ci.png"
        fig.savefig(p5, dpi=200, facecolor="white")
        plt.close(fig)
        records.append(
            AssetRecord(
                f"04_charts/{p5.name}",
                "GroupKFold AUROC and PR-AUC with 95% CIs for all predictor models.",
                "artifacts/predictor_metrics.json",
                "Predictive Modelling",
                f"5-fold GroupKFold CV across {metrics['n_samples']:,} pairs: "
                f"LightGBM AUROC {metrics['lgbm']['auroc']:.4f} "
                f"(PR-AUC {metrics['lgbm']['pr_auc']:.4f}), "
                f"LogisticRegression AUROC {metrics['logistic']['auroc']:.4f} "
                f"(PR-AUC {metrics['logistic']['pr_auc']:.4f}), "
                f"Baseline AUROC {metrics['baseline']['auroc']:.4f} "
                f"(PR-AUC {metrics['baseline']['pr_auc']:.4f}).",
                _now_iso(),
            )
        )


# ---------------------------------------------------------------------------
# 5. MANIFEST.md & CAPTIONS.md
# ---------------------------------------------------------------------------


def write_manifest_and_captions(out_dir: Path, records: list[AssetRecord]) -> None:
    """Write MANIFEST.md and CAPTIONS.md in out_dir."""
    manifest_lines = [
        "# Project Tyto Report Assets — Manifest",
        "",
        f"> Generated at `{_now_iso()}` by `scripts/make_report_assets.py`.",
        "> **Attribution:** Any screenshot or table displaying subtitle text comes from "
        "**OpenSubtitles** via **OPUS** and must be credited (`opensubtitles.org`) in the report.",
        "",
        "| File | What It Shows | Source Command / File & Lines | Generated At "
        "| Suggested Report Chapter |",
        "|------|---------------|-------------------------------|--------------|"
        "--------------------------|",
    ]
    for r in records:
        manifest_lines.append(
            f"| `{r.rel_path}` | {r.shows} | `{r.source}` | {r.generated_at} | {r.chapter} |"
        )
    (out_dir / "MANIFEST.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    caption_lines = [
        "# Project Tyto Report Assets — Figure & Listing Captions",
        "",
        f"> Generated at `{_now_iso()}`.",
        "",
    ]
    for r in records:
        caption_lines.append(f"- **`{r.rel_path}`**: {r.caption}")
    (out_dir / "CAPTIONS.md").write_text("\n".join(caption_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


def _normalize_only(raw_only: list[str] | None) -> set[str]:
    if not raw_only:
        return {"app", "code", "output", "charts", "docs"}
    mapping = {
        "app": "app",
        "01_app": "app",
        "code": "code",
        "02_code": "code",
        "output": "output",
        "03_output": "output",
        "charts": "charts",
        "04_charts": "charts",
        "docs": "docs",
        "manifest": "docs",
    }
    selected: set[str] = set()
    for item in raw_only:
        for part in item.split(","):
            key = part.strip().lower()
            if key not in mapping:
                raise ValueError(f"Unknown --only section: {part!r}")
            selected.add(mapping[key])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible Project Tyto report assets."
    )
    parser.add_argument(
        "--out",
        default="../tyto-report-assets",
        help="Output directory (default: ../tyto-report-assets)",
    )
    parser.add_argument(
        "--only",
        action="append",
        help="Sections to run: app, code, output, charts, docs (repeatable/comma-separated).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sections = _normalize_only(args.only)

    records: list[AssetRecord] = []
    if "app" in sections:
        print("[1/4] Generating 01_app/ screenshots...")
        generate_app_assets(out_dir, records)
    if "code" in sections:
        print("[2/4] Generating 02_code/ listings...")
        generate_code_assets(out_dir, records)
    if "output" in sections:
        print("[3/4] Generating 03_output/ command & artifact captures...")
        generate_output_assets(out_dir, records)
    if "charts" in sections:
        print("[4/4] Generating 04_charts/ matplotlib figures...")
        generate_chart_assets(out_dir, records)
    if "docs" in sections and records:
        print("[docs] Writing MANIFEST.md and CAPTIONS.md...")
        write_manifest_and_captions(out_dir, records)

    failures = _verify_pngs(out_dir)
    if failures:
        print("PNG quality check FAILED for:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    total_pngs = len(list(out_dir.glob("*/*.png")))
    print(
        f"Done. Verified {total_pngs} PNG(s) in {out_dir} "
        f"(all >= {MIN_WIDTH_PX}px wide, > {MIN_BYTES} bytes, non-blank)."
    )


if __name__ == "__main__":
    main()
