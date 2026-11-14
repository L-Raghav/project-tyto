# Phase 8: Course-Ready Tabular Dataset Report

> **Task:** Build a spreadsheet-style, course-ready tabular dataset from the existing Project Tyto MVP artifacts, plus two Excel-style preview screenshots.  
> **Date:** 2026-11-14  
> **Branch:** `mvp-dataset`  
> **Output Location:** `../tyto-dataset` (outside the repository root)

---

## 1. What Was Built

A comprehensive, production-grade tabular export pipeline that compiles the Project Tyto MVP artifacts into a unified 33-column tabular dataset formatted for educational data mining courses, spreadsheet software (Excel, LibreOffice, Google Sheets), and machine learning practice:

1. **Dataset Generator Module (`src/tyto/mvp/dataset.py`)**:
   - Integrates features, heuristic risk scores, risk tiers, KMeans cluster assignments, IsolationForest anomaly flags, and per-film metrics.
   - Enforces strict Honesty Rules: all values derived by code from real artifacts; no synthetic filling or hand-crafted records.
   - Standardizes headers to PascalCase and maps indicators to literal `"Yes"` / `"No"` strings.
   - Applies `pd.qcut` quartile discretization (levels 1 to 4) for `SimilarityLevel`, `LengthRatioLevel`, and `VowelShareLevel`.
   - Includes two uninformative constant teaching columns (`LanguagePair="en-ta"`, `PairCount=1`) mirroring classic textbook datasets.

2. **Typer CLI Command (`python -m tyto mvp dataset`)**:
   - Registered under the `mvp` group: `python -m tyto mvp dataset --out ../tyto-dataset`.
   - Supports configurable paths for input pairs and artifacts directory.

3. **Multi-format Exports in `../tyto-dataset/`**:
   - `tyto_dataset.csv`: UTF-8 with BOM (`utf-8-sig`), comma-separated, no index, 32,417 rows × 33 columns.
   - `tyto_dataset.xlsx`: Excel workbook with `Data` sheet (bold headers, frozen row 1, autofilter, column widths up to 24, number formatting: `0.000` for ratios/shares, `0.0` for RiskScore, no formulas) and `Dictionary` sheet.
   - `tyto_dataset_sample_1470.csv`: Stratified random sample of 1,470 rows (stratified by `RiskTier` with fixed seed 42; 70% Low, 20% Medium, 10% High).
   - `tyto_dataset_source_only.csv`: Leakage-free 11-column dataset containing only source-side (English) predictors and identifiers.
   - `DATA_DICTIONARY.md`: Full documentation covering all 33 columns, roles, types, derivation formulas, and quartile boundaries.
   - `DATASET_CARD.md`: Dataset card detailing corpus provenance (OPUS OpenSubtitles v2018), film distribution, circularity warnings, and the v2024 corruption rationale.
   - `private/PRIVATE_pair_text.csv` and `private/README.md`: Isolated subtitle text reference (kept strictly outside the repo in accordance with data governance policies).
   - `previews/preview_1.png` & `previews/preview_2.png`: High-resolution 2x device-scale screenshots rendered from HTML with spreadsheet chrome (column letters A..AG, row numbers 1..21, thin gridlines, bold header) via Playwright Chromium.
   - `previews/preview_all_columns_note.txt`: Detailed mapping of columns displayed in each screenshot.

---

## 2. Verification Gates

All verification gates were strictly evaluated on the real corpus artifacts before writing output files. All gates passed with 100% exact numerical agreement:

| Gate Check | Expected Value | Actual Value | Status |
|---|---|---|---|
| **Total Rows** | 32,417 | 32,417 | ✅ PASS |
| **PairID Uniqueness** | 32,417 unique values (1..32417) | 32,417 unique values | ✅ PASS |
| **RiskTier: High** | 3,242 rows (10.0%) | 3,242 rows (10.0%) | ✅ PASS |
| **RiskTier: Medium** | 6,483 rows (20.0%) | 6,483 rows (20.0%) | ✅ PASS |
| **RiskTier: Low** | 22,692 rows (70.0%) | 22,692 rows (70.0%) | ✅ PASS |
| **RiskFlag: Yes** | Exactly 6,484 rows (20.0%) | 6,484 rows (20.0%) | ✅ PASS |
| **RiskFlag: No** | Exactly 25,933 rows (80.0%) | 25,933 rows (80.0%) | ✅ PASS |

---

## 3. Columns Omitted and Why

- **`PredictedProbability`**: **OMITTED.**  
  *Reason:* Out-of-fold cross-validation predicted probabilities were not persisted in the pipeline artifacts (`artifacts/predictor_metrics.json` records aggregate AUROC/PR-AUC/Recall metrics; `artifacts/lgbm_model.pkl` stores the final fitted model bundle without per-row OOF scores). In strict accordance with Project Tyto's Honesty Rules, this column was omitted from the main table rather than fabricated, interpolated, or backfilled. This omission is explicitly documented in `DATA_DICTIONARY.md`, `DATASET_CARD.md`, and `previews/preview_all_columns_note.txt`.

Total columns in the final table: **33 columns** (PascalCase headers).

---

## 4. Raw Linter and Test Suite Output

### Raw Ruff Check Output
```text
All checks passed!
```

### Raw Pytest Output
```text
........................................................................ [ 51%]
....................................................................     [100%]
140 passed in 10.45s
```

All 140 project tests passed, including 11 dedicated unit tests in `tests/test_mvp_dataset.py` verifying column order, unique PairID, Yes/No domain restrictions, quartile bounds, leakage prevention, and CLI execution.

---

## 5. First 5 Rows of `tyto_dataset.csv`

*(Headers and values only; no subtitle text)*

```text
PairID,RiskFlag,RiskTier,FilmID,ReleaseYear,SeriesGroup,LinkType,EnglishWords,TamilChars,LengthRatio,LengthBand,TamilVowelShare,TamilLatinShare,SpacedPunctuation,ExclamationMark,QuestionMark,CapsRatio,DigitMatch,PunctuationParity,RepeatCount,IsShortLine,IdiomPresent,HedgeCount,LabseCosine,SimilarityLevel,LengthRatioLevel,VowelShareLevel,FilmSpacedPunctRate,RiskScore,ClusterID,IsAnomaly,LanguagePair,PairCount
1,Yes,Medium,395461,2005,,1:2,13,65,4.643,Typical,0.154,0.000,No,No,No,0.048,No,Yes,2,No,No,0,0.777,2,2,1,0.000,29.4,4,No,en-ta,1
2,Yes,High,395461,2005,,1:1,1,10,5.000,Typical,0.300,0.000,No,Yes,No,0.222,No,Yes,1,Yes,No,0,0.505,1,3,4,0.000,38.5,7,No,en-ta,1
3,No,Low,395461,2005,,1:1,3,15,3.750,Typical,0.133,0.000,No,Yes,No,0.077,No,No,1,Yes,No,0,0.935,4,2,1,0.000,3.0,7,No,en-ta,1
4,Yes,High,395461,2005,,1:2,10,49,4.455,Typical,0.224,0.000,No,No,No,0.048,No,Yes,1,No,No,0,0.659,1,2,3,0.000,35.5,4,No,en-ta,1
5,Yes,High,395461,2005,,1:2,8,54,6.000,Long,0.204,0.000,No,No,No,0.042,No,Yes,1,No,No,0,0.584,1,4,3,0.000,37.3,4,No,en-ta,1
```

---

## 6. Output Folder Tree and File Sizes

The output folder was created strictly as a sibling of the project repository root at `../tyto-dataset`. No generated dataset or subtitle text files exist within the Git repository.

```text
C:\Users\Shift\Projects\tyto-dataset/
├── DATASET_CARD.md                      (3,832 bytes)
├── DATA_DICTIONARY.md                   (7,525 bytes)
├── tyto_dataset.csv                     (6,251,307 bytes, ~6.0 MB)
├── tyto_dataset.xlsx                    (5,204,942 bytes, ~5.0 MB)
├── tyto_dataset_sample_1470.csv         (282,679 bytes, ~276 KB)
├── tyto_dataset_source_only.csv         (1,710,954 bytes, ~1.6 MB)
├── previews/
│   ├── preview_1.png                    (203,520 bytes, 3304 × 1176 px)
│   ├── preview_2.png                    (157,844 bytes, 3368 × 1176 px)
│   └── preview_all_columns_note.txt     (1,905 bytes)
└── private/
    ├── PRIVATE_pair_text.csv            (3,913,749 bytes, ~3.7 MB)
    └── README.md                        (438 bytes)
```

---

## 7. Repo Hygiene Check

`git status` confirms zero untracked dataset or text files inside `project-tyto`:
```text
On branch mvp-dataset
nothing to commit, working tree clean
```
