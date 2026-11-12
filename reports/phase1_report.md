# Phase 1 Report: Data Ingestion, Cleaning, QC, and Audit

**Project:** Project Tyto - English-Tamil movie subtitle translation-quality mining
**Date:** 2026-10-15 (updated 2026-10-26)
**Branch:** (current branch, uncommitted)

---

## Assumptions

1. **OPUS version**: `2016` is the latest available en-ta release verified on the OPUS site. The version is **configurable** in `configs/default.yaml` (`opus.version`).
2. **PYTHONUTF8=1** must be set in the shell before any CLI command so Tamil text survives all Windows file I/O.
3. The data pipeline requires `opustools`, `datasketch`, `requests`, `matplotlib`, and `tabulate` - all declared in `pyproject.toml` and installed in the venv.
4. LaBSE scoring is **stage-2 only** and is gated behind `--labse`; it requires the GPU. Tests use `FakeTinyModel` instead.
5. Near-dedup operates on English text only (MinHash over 4-char shingles).

---

## What Was Built

### 1. Data Download - `src/tyto/data/download.py`
- `download_opus(cfg)` fetches `OPUS-OpenSubtitles/<version>/xml/en-ta.zip` from the OPUS CDN.
- **Resume-safe**: sends HTTP `Range` headers if a partial file exists.
- **MD5 checksum**: downloads `.md5` sidecar and verifies; removes corrupt file and raises on mismatch.
- CLI: `python -m tyto data download [--version 2016] [--force]`

### 2. OPUS XML Parser - `src/tyto/data/parse.py`
- Reads the XML zip, streams through the alignment file grouped by document.
- Extracts per-sentence text, timing (seconds), alignment type, film metadata.
- Calls `clean_text()` for `en`/`ta` columns; keeps `en_raw`/`ta_raw` untouched.
- Writes `data/processed/pairs.parquet` with the canonical schema.
- CLI: `python -m tyto data parse [--zip path] [--max-pairs N]`

### 3. Text Cleaning - `src/tyto/data/clean.py`
- NFC normalisation, ZWJ/ZWNJ/BOM removal (critical for Tamil), markup tags, sound descriptions, music symbols, speaker dashes, double dashes.
- `script_ratios(text)` and `compute_script_flags(ta_text)` for language ID.

### 4. Script-based Language ID
Flags: `WRONG_LANG`, `ROMANISED_TA`, `CODE_MIXED` - flag only, never delete.

### 5. QC Filters - `src/tyto/data/qc.py`
- Length filters (token count + char-length ratio), script flags, exact dedup with `repeat_count`, near-dedup (MinHash LSH), optional LaBSE alignment score.
- **Stop-gate**: raises `RuntimeError` if usable pairs < `qc.min_usable_pairs`.
- CLI: `python -m tyto data qc [--labse]`

### 6. SRT Aligner - `src/tyto/data/srt_align.py`
- Aligns two SRT files by temporal IoU, merges split lines, returns `PAIRS_SCHEMA` DataFrame.
- CLI: `python -m tyto data srt-align en.srt ta.srt`

### 7. Data Audit - `src/tyto/data/audit.py`
- Generates `reports/data_audit.md` plus 6 matplotlib figures (pair counts, lines-per-film, length distributions, script ratios, flag share, LaBSE scores).
- CLI: `python -m tyto data audit`

### 8. DuckDB Helper - `src/tyto/data/db.py`
- `PairsDB(parquet_path)` context manager with `query()`, `film_summary()`, `flag_summary()`, `sample_pairs()`, `summary_stats()`, `usable_count()`.

---

## How to Run

```powershell
$env:PYTHONUTF8=1
.venv\Scripts\python.exe -m tyto data download
.venv\Scripts\python.exe -m tyto data parse
.venv\Scripts\python.exe -m tyto data qc
.venv\Scripts\python.exe -m tyto data audit
```

---

## Idempotency

Re-running any stage on the same input data is safe and produces the same output:

- **download**: checks MD5 on the existing zip before downloading; skips if checksum passes.
  A partial or corrupt file is removed and re-downloaded.
- **parse**: overwrites `pairs.parquet` with `pq.write_table(...)`; same zip + version -> identical schema and row content.
- **qc**: reads the current parquet, re-applies all filters, and overwrites in place; deterministic (fixed seed via config, MinHash uses stable permutations).
- **audit**: overwrites `reports/data_audit.md` and all figure files; confirmed byte-for-byte identical on consecutive runs against the same parquet (MD5 verified: `1EB3E82E687B33DDBEECD99A9EF14EFA` both runs).

---

## Stop-Gate Behaviour

When `count_usable(df) < cfg.qc.min_usable_pairs`, the CLI `data qc` command:

1. **Prints the pair count** to stdout, for example:
   ```
   STOP GATE: only 1 usable pairs remain (threshold: 99,999).
   Review data_audit.md for options before continuing.
   ```
2. **Exits with code 1** (non-zero), so it is detectable in shell scripts:
   ```powershell
   .venv\Scripts\python.exe -m tyto data qc
   if ($LASTEXITCODE -ne 0) { Write-Host "Pipeline halted at stop-gate." }
   ```

Both behaviours were verified against a synthetic 1-row parquet with `min_usable_pairs=99999`:
- Exit code confirmed: **1**
- Message confirmed: contains `STOP GATE`, the actual pair count, and the threshold.

---

## Test Output

```
95 passed in 4.48s
```

| Test file | Tests | Purpose |
|---|---|---|
| `test_clean.py` | 27 | TEXT RULES: markup, ZWJ, music, dashes, NFC, script ratios |
| `test_qc.py` | 16 | QC filters, dedup, stop-gate, usable_count |
| `test_srt_align.py` | 16 | IoU, alignment, schema, edge cases |
| `test_db.py` | 12 | DuckDB helper: count, sample, summary, flag counts |
| `test_parse.py` | 12 | OPUS XML parse fixture: schema, text, timing, film_id |

---

## Dependencies Added (Phase 1)

All declared in `pyproject.toml` and installed in the venv:

| Package | Version | Purpose |
|---|---|---|
| `opustools` | >=1.6.6 | OPUS corpus API (informational listing) |
| `datasketch` | >=1.6.5 | MinHash LSH for near-dedup |
| `requests` | >=2.32.0 | HTTP download with Range-header resume |
| `matplotlib` | >=3.9 | Figures in data_audit.md |
| `tabulate` | >=0.9.0 | pandas .to_markdown() for audit tables |

---

## Known Issues

1. OPUS 2016 en-ta coverage may be small; the stop-gate will fire if under 20,000 usable pairs survive QC.
2. Near-dedup performance: for >500K pairs, MinHash LSH may be slow.
3. LaBSE batch-size back-off is configured in `cfg.device.backoff_factor` but not yet wired into `apply_labse_flags`.
