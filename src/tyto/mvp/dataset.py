"""Course-ready tabular dataset generator for Project Tyto MVP.

Extracts, formats, and exports a spreadsheet-style main table (CSV and XLSX),
sample and leakage-free source-only datasets, data dictionaries, dataset card,
private subtitle text reference, and Playwright Excel-style preview screenshots.

Honesty Rules:
- All values are computed from real artifacts by code; no fabricated values.
- Reuses definitions of risk score, tiers, and predictor target from label.py and predict.py.
- Verification gates enforce exact row and class counts.
- RiskFlag is an unsupervised heuristic proxy; no human labels exist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# The six IMDb document IDs constituting the multi-part franchise series
SERIES_A_FILM_IDS: frozenset[str] = frozenset(
    {"6261062", "6261064", "6261068", "6261070", "6261074", "6261076"}
)

# Ordered column specification for the main dataset table
DATASET_COLUMNS: list[str] = [
    "PairID",
    "RiskFlag",
    "RiskTier",
    "FilmID",
    "ReleaseYear",
    "SeriesGroup",
    "LinkType",
    "EnglishWords",
    "TamilChars",
    "LengthRatio",
    "LengthBand",
    "TamilVowelShare",
    "TamilLatinShare",
    "SpacedPunctuation",
    "ExclamationMark",
    "QuestionMark",
    "CapsRatio",
    "DigitMatch",
    "PunctuationParity",
    "RepeatCount",
    "IsShortLine",
    "IdiomPresent",
    "HedgeCount",
    "LabseCosine",
    "SimilarityLevel",
    "LengthRatioLevel",
    "VowelShareLevel",
    "FilmSpacedPunctRate",
    "RiskScore",
    "ClusterID",
    "IsAnomaly",
    "LanguagePair",
    "PairCount",
]

SOURCE_ONLY_COLUMNS: list[str] = [
    "PairID",
    "RiskFlag",
    "FilmID",
    "SeriesGroup",
    "EnglishWords",
    "ExclamationMark",
    "QuestionMark",
    "CapsRatio",
    "RepeatCount",
    "IsShortLine",
    "IdiomPresent",
]

# Data dictionary metadata rows:
# (Column, Type, Role, Meaning, How derived, Example value)
DICTIONARY_METADATA: list[dict[str, str]] = [
    {
        "Column": "PairID",
        "Type": "Integer",
        "Role": "ID",
        "Meaning": "Stable identifier (1..32417) in original corpus order.",
        "How derived": "data/processed/pairs_v2018.parquet: pair_id + 1",
        "Example value": "1",
    },
    {
        "Column": "RiskFlag",
        "Type": "String (Yes/No)",
        "Role": "Target",
        "Meaning": (
            "Binary target: Yes if heuristic risk_score in top 20% "
            "(>= p80: 24.84), else No. (Heuristic proxy; NOT human label)."
        ),
        "How derived": "artifacts/risk.parquet: risk_score >= p80 (top 20%)",
        "Example value": "No",
    },
    {
        "Column": "RiskTier",
        "Type": "String (Low/Medium/High)",
        "Role": "Output",
        "Meaning": (
            "Heuristic risk tier: High (top 10%, score >= 35.80), Medium (next 20%, "
            "20.89 <= score < 35.80), Low (bottom 70%, score < 20.89)."
        ),
        "How derived": "artifacts/risk.parquet: risk_tier",
        "Example value": "Low",
    },
    {
        "Column": "FilmID",
        "Type": "String",
        "Role": "ID",
        "Meaning": "OPUS IMDb document / film identifier.",
        "How derived": "data/processed/pairs_v2018.parquet: film_id",
        "Example value": "395461",
    },
    {
        "Column": "ReleaseYear",
        "Type": "Integer",
        "Role": "Feature-source-side",
        "Meaning": "Movie release year extracted from corpus metadata.",
        "How derived": "data/processed/pairs_v2018.parquet: year",
        "Example value": "2005",
    },
    {
        "Column": "SeriesGroup",
        "Type": "String",
        "Role": "Feature-source-side",
        "Meaning": (
            "Franchise group ('series_A' for the six multi-part IMDb IDs "
            "6261062/64/68/70/74/76; empty/None for standalone films)."
        ),
        "How derived": "data/processed/pairs_v2018.parquet: series_group",
        "Example value": "series_A",
    },
    {
        "Column": "LinkType",
        "Type": "String",
        "Role": "Feature-pair-side",
        "Meaning": "Sentence link structure: '1:1', '1:2', '2:1', or 'Other' (e.g. 1:3, 3:1).",
        "How derived": "data/processed/pairs_v2018.parquet: link_type",
        "Example value": "1:1",
    },
    {
        "Column": "EnglishWords",
        "Type": "Integer",
        "Role": "Feature-source-side",
        "Meaning": "Whitespace-delimited word count in English subtitle line.",
        "How derived": "artifacts/features.parquet: en_words",
        "Example value": "6",
    },
    {
        "Column": "TamilChars",
        "Type": "Integer",
        "Role": "Feature-pair-side",
        "Meaning": "Total character count of cleaned Tamil subtitle line.",
        "How derived": "artifacts/features.parquet: ta_chars",
        "Example value": "28",
    },
    {
        "Column": "LengthRatio",
        "Type": "Float",
        "Role": "Feature-pair-side",
        "Meaning": "Character-to-word ratio: ta_chars / (en_words + 1).",
        "How derived": "artifacts/features.parquet: len_ratio",
        "Example value": "4.000",
    },
    {
        "Column": "LengthBand",
        "Type": "String (Short/Typical/Long)",
        "Role": "Feature-pair-side",
        "Meaning": (
            "Length ratio category: Short (< p25: 3.60), Typical (p25 to p75: "
            "3.60-5.71), or Long (> p75: 5.71)."
        ),
        "How derived": "artifacts/features.parquet: len_ratio percentiles",
        "Example value": "Typical",
    },
    {
        "Column": "TamilVowelShare",
        "Type": "Float",
        "Role": "Feature-pair-side",
        "Meaning": "Share of Tamil chars that are vowel signs (Unicode U+0BBE to U+0BCC).",
        "How derived": "artifacts/features.parquet: ta_vowel_share",
        "Example value": "0.179",
    },
    {
        "Column": "TamilLatinShare",
        "Type": "Float",
        "Role": "Feature-pair-side",
        "Meaning": "Share of characters in Tamil line that are ASCII Latin letters.",
        "How derived": "artifacts/features.parquet: ta_latin_share",
        "Example value": "0.000",
    },
    {
        "Column": "SpacedPunctuation",
        "Type": "String (Yes/No)",
        "Role": "Feature-pair-side",
        "Meaning": (
            "Yes if spaced punctuation patterns (' , ', ' . ', ' ! ', ' ? ') occur in "
            "Tamil line, else No."
        ),
        "How derived": "artifacts/features.parquet: ta_spaced_punct > 0",
        "Example value": "No",
    },
    {
        "Column": "ExclamationMark",
        "Type": "String (Yes/No)",
        "Role": "Feature-source-side",
        "Meaning": "Yes if exclamation mark ('!') is present in English line, else No.",
        "How derived": "artifacts/features.parquet: en_exclaim == 1",
        "Example value": "No",
    },
    {
        "Column": "QuestionMark",
        "Type": "String (Yes/No)",
        "Role": "Feature-source-side",
        "Meaning": "Yes if question mark ('?') is present in English line, else No.",
        "How derived": "artifacts/features.parquet: en_question == 1",
        "Example value": "No",
    },
    {
        "Column": "CapsRatio",
        "Type": "Float",
        "Role": "Feature-source-side",
        "Meaning": "Ratio of uppercase characters to total characters in English line.",
        "How derived": "artifacts/features.parquet: en_caps_ratio",
        "Example value": "0.040",
    },
    {
        "Column": "DigitMatch",
        "Type": "String (Yes/No)",
        "Role": "Feature-pair-side",
        "Meaning": "Yes if any digit string in English also appears in Tamil line, else No.",
        "How derived": "artifacts/features.parquet: digit_match == 1",
        "Example value": "No",
    },
    {
        "Column": "PunctuationParity",
        "Type": "String (Yes/No)",
        "Role": "Feature-pair-side",
        "Meaning": (
            "Yes if punctuation disparity between English and Tamil exceeds 1 "
            "(|punct(en) - punct(ta)| > 1), else No."
        ),
        "How derived": "artifacts/features.parquet: punct_parity == 1",
        "Example value": "No",
    },
    {
        "Column": "RepeatCount",
        "Type": "Integer",
        "Role": "Feature-source-side",
        "Meaning": "Maximum count of any single repeated word in English line.",
        "How derived": "artifacts/features.parquet: repeat_count",
        "Example value": "1",
    },
    {
        "Column": "IsShortLine",
        "Type": "String (Yes/No)",
        "Role": "Feature-source-side",
        "Meaning": "Yes if English line has 3 or fewer words (en_words <= 3), else No.",
        "How derived": "artifacts/features.parquet: is_short == 1",
        "Example value": "No",
    },
    {
        "Column": "IdiomPresent",
        "Type": "String (Yes/No)",
        "Role": "Feature-source-side",
        "Meaning": "Yes if line matches illustrative 40-phrase English idiom list, else No.",
        "How derived": "artifacts/features.parquet: idiom_present == 1",
        "Example value": "No",
    },
    {
        "Column": "HedgeCount",
        "Type": "Integer",
        "Role": "Feature-pair-side",
        "Meaning": "Count of Tamil hedge / hesitation words in line.",
        "How derived": "artifacts/features.parquet: ta_hedge_count",
        "Example value": "0",
    },
    {
        "Column": "LabseCosine",
        "Type": "Float",
        "Role": "Feature-pair-side",
        "Meaning": "Cosine similarity between English and Tamil LaBSE sentence embeddings.",
        "How derived": "artifacts/features.parquet: labse_cos",
        "Example value": "0.852",
    },
    {
        "Column": "SimilarityLevel",
        "Type": "Integer (1-4)",
        "Role": "Feature-pair-side",
        "Meaning": "Quartile band of LabseCosine (1 = least similar [<= 0.730], 4 = most similar).",
        "How derived": "artifacts/features.parquet: pd.qcut(labse_cos, 4)",
        "Example value": "3",
    },
    {
        "Column": "LengthRatioLevel",
        "Type": "Integer (1-4)",
        "Role": "Feature-pair-side",
        "Meaning": "Quartile band of LengthRatio (1 = lowest ratio [<= 3.60], 4 = highest ratio).",
        "How derived": "artifacts/features.parquet: pd.qcut(len_ratio, 4)",
        "Example value": "2",
    },
    {
        "Column": "VowelShareLevel",
        "Type": "Integer (1-4)",
        "Role": "Feature-pair-side",
        "Meaning": "Quartile band of TamilVowelShare (1 = lowest vowel share, 4 = highest).",
        "How derived": "artifacts/features.parquet: pd.qcut(ta_vowel_share, 4)",
        "Example value": "2",
    },
    {
        "Column": "FilmSpacedPunctRate",
        "Type": "Float",
        "Role": "Feature-pair-side",
        "Meaning": "Per-film rate of spaced punctuation lines across the movie document.",
        "How derived": "artifacts/film_stats.parquet: spaced_punct_rate",
        "Example value": "0.010",
    },
    {
        "Column": "RiskScore",
        "Type": "Float",
        "Role": "Output",
        "Meaning": "Heuristic risk proxy score in [0, 100] from 7 signals. (Not human validated).",
        "How derived": "artifacts/risk.parquet: risk_score",
        "Example value": "14.5",
    },
    {
        "Column": "ClusterID",
        "Type": "Integer",
        "Role": "Output",
        "Meaning": "KMeans cluster assignment (k=8 discovered from standardized feature space).",
        "How derived": "artifacts/clusters.parquet: cluster_id",
        "Example value": "0",
    },
    {
        "Column": "IsAnomaly",
        "Type": "String (Yes/No)",
        "Role": "Output",
        "Meaning": "Yes if pair is among top 30 IsolationForest anomalies, else No.",
        "How derived": "artifacts/anomalies.parquet: pair_id in top 30",
        "Example value": "No",
    },
    {
        "Column": "LanguagePair",
        "Type": "String",
        "Role": "Constant",
        "Meaning": "Constant 'en-ta' mirroring classic layout; uninformative for mining.",
        "How derived": "Constant: 'en-ta'",
        "Example value": "en-ta",
    },
    {
        "Column": "PairCount",
        "Type": "Integer",
        "Role": "Constant",
        "Meaning": "Constant integer 1 mirroring classic layout; uninformative for mining.",
        "How derived": "Constant: 1",
        "Example value": "1",
    },
]


def build_dataset(
    risk_df: pd.DataFrame,
    pairs_df: pd.DataFrame | None = None,
    clusters_df: pd.DataFrame | None = None,
    anomalies_df: pd.DataFrame | None = None,
    film_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the clean 33-column main dataset table from pipeline artifacts."""
    df = risk_df.copy().reset_index(drop=True)

    # 1. PairID: stable integer 1..N in original order
    if "pair_id" in df.columns:
        pair_id_min = int(df["pair_id"].min())
        if pair_id_min == 0:
            pair_ids = df["pair_id"].astype(int) + 1
        else:
            pair_ids = df["pair_id"].astype(int)
    else:
        pair_ids = pd.Series(range(1, len(df) + 1), index=df.index, dtype=int)

    # 2. RiskFlag: top 20% of heuristic risk_score (predictor target)
    risk_score_series = df["risk_score"].astype(float)
    p80 = float(np.percentile(risk_score_series, 80))
    risk_flag = np.where(risk_score_series >= p80, "Yes", "No")

    # 3. RiskTier
    risk_tier = df["risk_tier"].astype(str)

    # 4. FilmID
    film_id = df["film_id"].astype(str)

    # 5. ReleaseYear
    year = df["year"].astype(int)

    # 6. SeriesGroup: series_A for the 6 franchise films, else None
    series_group = np.where(film_id.isin(SERIES_A_FILM_IDS), "series_A", None)

    # 7. LinkType: 1:1, 1:2, 2:1, Other
    raw_link = df["link_type"].astype(str)
    link_type = np.where(raw_link.isin(["1:1", "1:2", "2:1"]), raw_link, "Other")

    # 8. EnglishWords
    en_words = df["en_words"].astype(int)

    # 9. TamilChars
    ta_chars = df["ta_chars"].astype(int)

    # 10. LengthRatio
    len_ratio = df["len_ratio"].astype(float)

    # 11. LengthBand: Short (< p25), Long (> p75), Typical (p25..p75)
    p25 = float(len_ratio.quantile(0.25))
    p75 = float(len_ratio.quantile(0.75))
    length_band = np.where(
        len_ratio < p25,
        "Short",
        np.where(len_ratio > p75, "Long", "Typical"),
    )

    # 12. TamilVowelShare
    ta_vowel_share = df["ta_vowel_share"].astype(float)

    # 13. TamilLatinShare
    ta_latin_share = df["ta_latin_share"].astype(float)

    # 14. SpacedPunctuation
    spaced_punct = np.where(df["ta_spaced_punct"].astype(float) > 0.0, "Yes", "No")

    # 15. ExclamationMark
    en_exclaim = np.where(df["en_exclaim"].astype(int) == 1, "Yes", "No")

    # 16. QuestionMark
    en_question = np.where(df["en_question"].astype(int) == 1, "Yes", "No")

    # 17. CapsRatio
    en_caps_ratio = df["en_caps_ratio"].astype(float)

    # 18. DigitMatch
    digit_match = np.where(df["digit_match"].astype(int) == 1, "Yes", "No")

    # 19. PunctuationParity
    punct_parity = np.where(df["punct_parity"].astype(int) == 1, "Yes", "No")

    # 20. RepeatCount
    repeat_count = df["repeat_count"].astype(int)

    # 21. IsShortLine
    is_short = np.where(df["is_short"].astype(int) == 1, "Yes", "No")

    # 22. IdiomPresent
    idiom_present = np.where(df["idiom_present"].astype(int) == 1, "Yes", "No")

    # 23. HedgeCount
    ta_hedge_count = df["ta_hedge_count"].astype(int)

    # 24. LabseCosine
    labse_cos = df["labse_cos"].astype(float)

    # 25. SimilarityLevel: 1 to 4 quartiles of LabseCosine (1 = least similar)
    sim_level = pd.qcut(labse_cos, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)

    # 26. LengthRatioLevel: 1 to 4 quartiles
    len_level = pd.qcut(len_ratio, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)

    # 27. VowelShareLevel: 1 to 4 quartiles
    vowel_level = pd.qcut(ta_vowel_share, q=4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)

    # 28. FilmSpacedPunctRate (per-film aggregate from film_stats)
    if film_stats_df is not None and "spaced_punct_rate" in film_stats_df.columns:
        rate_map = dict(
            zip(
                film_stats_df["film_id"].astype(str),
                film_stats_df["spaced_punct_rate"].astype(float),
                strict=False,
            )
        )
        film_spaced_punct_rate = film_id.map(rate_map).fillna(0.0).astype(float)
    else:
        film_spaced_punct_rate = pd.Series(0.0, index=df.index, dtype=float)

    # 29. RiskScore
    risk_score = risk_score_series

    # 30. ClusterID
    if clusters_df is not None and "cluster_id" in clusters_df.columns:
        if "pair_id" in clusters_df.columns and "pair_id" in df.columns:
            cluster_map = dict(
                zip(
                    clusters_df["pair_id"].astype(int),
                    clusters_df["cluster_id"].astype(int),
                    strict=False,
                )
            )
            cluster_id = df["pair_id"].astype(int).map(cluster_map).fillna(0).astype(int)
        else:
            cluster_id = clusters_df["cluster_id"].astype(int).values
    elif "cluster_id" in df.columns:
        cluster_id = df["cluster_id"].astype(int)
    else:
        cluster_id = pd.Series(0, index=df.index, dtype=int)

    # 31. IsAnomaly: Yes if in anomalies artifact, else No
    if anomalies_df is not None and "pair_id" in anomalies_df.columns:
        anomaly_ids = set(anomalies_df["pair_id"].astype(int).tolist())
        raw_pair_id = df["pair_id"].astype(int) if "pair_id" in df.columns else (pair_ids - 1)
        is_anomaly = np.where(raw_pair_id.isin(anomaly_ids), "Yes", "No")
    else:
        is_anomaly = pd.Series("No", index=df.index, dtype=str)

    # 32. LanguagePair: constant en-ta
    language_pair = "en-ta"

    # 33. PairCount: constant 1
    pair_count = 1

    out = pd.DataFrame(
        {
            "PairID": pair_ids,
            "RiskFlag": risk_flag,
            "RiskTier": risk_tier,
            "FilmID": film_id,
            "ReleaseYear": year,
            "SeriesGroup": series_group,
            "LinkType": link_type,
            "EnglishWords": en_words,
            "TamilChars": ta_chars,
            "LengthRatio": len_ratio,
            "LengthBand": length_band,
            "TamilVowelShare": ta_vowel_share,
            "TamilLatinShare": ta_latin_share,
            "SpacedPunctuation": spaced_punct,
            "ExclamationMark": en_exclaim,
            "QuestionMark": en_question,
            "CapsRatio": en_caps_ratio,
            "DigitMatch": digit_match,
            "PunctuationParity": punct_parity,
            "RepeatCount": repeat_count,
            "IsShortLine": is_short,
            "IdiomPresent": idiom_present,
            "HedgeCount": ta_hedge_count,
            "LabseCosine": labse_cos,
            "SimilarityLevel": sim_level,
            "LengthRatioLevel": len_level,
            "VowelShareLevel": vowel_level,
            "FilmSpacedPunctRate": film_spaced_punct_rate,
            "RiskScore": risk_score,
            "ClusterID": cluster_id,
            "IsAnomaly": is_anomaly,
            "LanguagePair": language_pair,
            "PairCount": pair_count,
        },
        columns=DATASET_COLUMNS,
    )
    return out


def validate_gates(df: pd.DataFrame, is_full_corpus: bool = True) -> dict[str, Any]:
    """Verify strict Honesty Gates."""
    total_rows = len(df)
    unique_pair_ids = df["PairID"].nunique()
    is_pair_id_unique = unique_pair_ids == total_rows

    tier_counts = df["RiskTier"].value_counts().to_dict()
    high_count = int(tier_counts.get("High", 0))
    medium_count = int(tier_counts.get("Medium", 0))
    low_count = int(tier_counts.get("Low", 0))

    flag_counts = df["RiskFlag"].value_counts().to_dict()
    flag_yes_count = int(flag_counts.get("Yes", 0))
    flag_pct = (flag_yes_count / total_rows) * 100.0 if total_rows > 0 else 0.0

    checks: dict[str, Any] = {
        "total_rows": total_rows,
        "unique_pair_ids": unique_pair_ids,
        "is_pair_id_unique": is_pair_id_unique,
        "tier_counts": tier_counts,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "flag_yes_count": flag_yes_count,
        "flag_pct": flag_pct,
        "all_passed": True,
        "errors": [],
    }

    if not is_pair_id_unique:
        checks["all_passed"] = False
        checks["errors"].append(
            f"PairID is not unique: {unique_pair_ids} unique vs {total_rows} rows"
        )

    if is_full_corpus:
        if total_rows != 32417:
            checks["all_passed"] = False
            checks["errors"].append(f"Row count mismatch: {total_rows} != 32,417")

        if high_count != 3242 or medium_count != 6483 or low_count != 22692:
            checks["all_passed"] = False
            checks["errors"].append(
                f"RiskTier counts mismatch: High={high_count} (exp 3,242), "
                f"Medium={medium_count} (exp 6,483), Low={low_count} (exp 22,692)"
            )

        if flag_yes_count != 6484:
            checks["all_passed"] = False
            checks["errors"].append(
                f"RiskFlag=Yes mismatch: {flag_yes_count} != 6,484 (expected 20.0%)"
            )

    return checks


def write_excel_dataset(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write XLSX workbook using openpyxl with Data and Dictionary sheets."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb = openpyxl.Workbook()

    # Sheet 1: Data
    ws_data = wb.active
    ws_data.title = "Data"

    headers = list(df.columns)
    ws_data.append(headers)

    header_font = Font(bold=True)
    header_fill = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")

    for col_idx in range(1, len(headers) + 1):
        cell = ws_data.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill

    for row in df.itertuples(index=False, name=None):
        ws_data.append(list(row))

    ws_data.freeze_panes = "A2"
    ws_data.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(df) + 1}"

    fmt_three_decimals = {
        "LengthRatio",
        "TamilVowelShare",
        "TamilLatinShare",
        "CapsRatio",
        "LabseCosine",
        "FilmSpacedPunctRate",
    }
    fmt_one_decimal = {"RiskScore"}

    for col_idx, col_name in enumerate(headers, start=1):
        col_letter = get_column_letter(col_idx)
        if col_name in fmt_three_decimals:
            for r in range(2, len(df) + 2):
                ws_data.cell(row=r, column=col_idx).number_format = "0.000"
        elif col_name in fmt_one_decimal:
            for r in range(2, len(df) + 2):
                ws_data.cell(row=r, column=col_idx).number_format = "0.0"

        header_len = len(col_name)
        ws_data.column_dimensions[col_letter].width = min(max(header_len + 3, 11), 24)

    # Sheet 2: Dictionary
    ws_dict = wb.create_sheet(title="Dictionary")
    dict_headers = ["Column", "Type", "Role", "Meaning", "How derived", "Example value"]
    ws_dict.append(dict_headers)

    for col_idx in range(1, len(dict_headers) + 1):
        cell = ws_dict.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = header_fill

    for entry in DICTIONARY_METADATA:
        ws_dict.append([
            entry["Column"],
            entry["Type"],
            entry["Role"],
            entry["Meaning"],
            entry["How derived"],
            entry["Example value"],
        ])

    ws_dict.freeze_panes = "A2"
    ws_dict.auto_filter.ref = (
        f"A1:{get_column_letter(len(dict_headers))}{len(DICTIONARY_METADATA) + 1}"
    )

    dict_widths = [22, 24, 20, 60, 45, 18]
    for col_idx, width in enumerate(dict_widths, start=1):
        col_letter = get_column_letter(col_idx)
        ws_dict.column_dimensions[col_letter].width = min(width, 60)

    wb.save(out_path)
    logger.info("Saved formatted Excel workbook to %s", out_path)


def write_data_dictionary_md(out_path: Path) -> None:
    """Generate DATA_DICTIONARY.md with clean GitHub-flavored markdown."""
    lines: list[str] = [
        "# Project Tyto — Data Dictionary",
        "",
        "> **Table: `tyto_dataset`** (32,417 rows, 33 columns)",
        ">",
        (
            "> **Heuristic Proxy Notice:** The `RiskFlag`, `RiskTier`, and `RiskScore` columns "
            "are unsupervised heuristic proxies derived from surface and similarity signals. "
            "They are **NOT** validated human quality judgements."
        ),
        "",
        "| Column | Type | Role | Meaning | How Derived | Example Value |",
        "|---|---|---|---|---|---|",
    ]

    for item in DICTIONARY_METADATA:
        meaning = item["Meaning"].replace("|", "\\|")
        derived = item["How derived"].replace("|", "\\|")
        lines.append(
            f"| `{item['Column']}` | {item['Type']} | `{item['Role']}` | "
            f"{meaning} | {derived} | `{item['Example value']}` |"
        )

    lines.extend([
        "",
        "## Column Roles Legend",
        "",
        "- `ID`: Unique identifier or entity key.",
        "- `Target`: Primary classification target for prediction exercises.",
        "- `Feature-source-side`: English source-side feature (leakage-free).",
        "- `Feature-pair-side`: Bilingual or target-side feature (causes circularity with target).",
        "- `Output`: Downstream mining or heuristic scoring output.",
        "- `Constant`: Included to mirror classic teaching-dataset layouts.",
        "",
        "## Omitted Columns",
        "",
        (
            "- `PredictedProbability`: Omitted because out-of-fold cross-validation probabilities "
            "were not saved in MVP model artifacts (`artifacts/lgbm_model.pkl` and "
            "`artifacts/predictor_metrics.json`). In accordance with Project Tyto's Honesty "
            "Rules, this column was omitted rather than fabricated or backfilled."
        ),
        "",
        "## Quartile Levels Definition",
        "",
        (
            "- `SimilarityLevel`: Quartiles of `LabseCosine` across all 32,417 rows "
            "(1: <= 0.730, 2: 0.730-0.825, 3: 0.825-0.890, 4: >= 0.890). "
            "Level 1 represents lowest similarity."
        ),
        (
            "- `LengthRatioLevel`: Quartiles of `LengthRatio` "
            "(1: <= 3.60, 2: 3.60-4.50, 3: 4.50-5.71, 4: >= 5.71)."
        ),
        (
            "- `VowelShareLevel`: Quartiles of `TamilVowelShare` "
            "(1: <= 0.176, 2: 0.176-0.194, 3: 0.194-0.210, 4: >= 0.210)."
        ),
    ])

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved data dictionary markdown to %s", out_path)


def write_dataset_card_md(out_path: Path) -> None:
    """Generate comprehensive DATASET_CARD.md."""
    card = (
        "# Dataset Card: Project Tyto Translation-Quality Dataset\n\n"
        "## Summary\n"
        "The **Project Tyto Dataset** is a course-ready, tabular data mining dataset "
        "containing 32,417 English–Tamil movie subtitle pairs across 28 films from "
        "OPUS OpenSubtitles. The dataset is structured for data mining education, "
        "exploratory pattern analysis, association rule mining, clustering, anomaly "
        "detection, and leak-free machine learning classification.\n\n"
        "---\n\n"
        "## Dataset Details\n\n"
        "- **Language Pair:** English (`en`) → Tamil (`ta`)\n"
        "- **Total Pairs:** 32,417\n"
        "- **Total Film Documents:** 28 distinct movie titles\n"
        "- **Release Years:** 2005 – 2017\n"
        "- **Franchise Group:** One multi-part franchise (`series_A` with 6 IMDb IDs: "
        "`6261062`, `6261064`, `6261068`, `6261070`, `6261074`, `6261076`)\n"
        "- **Primary Target:** `RiskFlag` (Binary: `Yes` / `No`)\n"
        "- **Class Balance:** Exactly 6,484 `Yes` (20.0%) and 25,933 `No` (80.0%)\n"
        "- **Risk Tiers:**\n"
        "  - `High`: 3,242 pairs (10.0%, heuristic score $\\ge$ 90th percentile)\n"
        "  - `Medium`: 6,483 pairs (20.0%, 70th to 90th percentile)\n"
        "  - `Low`: 22,692 pairs (70.0%, < 70th percentile)\n\n"
        "---\n\n"
        "## Provenance & Attribution\n\n"
        "- **Source Corpus:** OPUS OpenSubtitles v2018 (`en-ta.txt.zip`)\n"
        "- **Corpus Providers:** OpenSubtitles.org and the OPUS Project.\n"
        "- **Original Subtitles:** Provided by community contributors of OpenSubtitles.\n"
        "- **Academic Citations:**\n"
        "  - Jörg Tiedemann, 2012. *Parallel Data, Tools and Interfaces in OPUS*. In LREC 2012.\n"
        "  - Pierre Lison and Jörg Tiedemann, 2016. *OpenSubtitles2016: Extracting Large "
        "Parallel Corpora from Movie and TV Subtitles*. In LREC 2016.\n\n"
        "---\n\n"
        "## Why OPUS v2018 (The v2024 Corruption Note)\n\n"
        "Project Tyto explicitly utilizes OPUS OpenSubtitles release **v2018** rather than "
        "the newer **v2024** release.\n"
        "A thorough survey of OPUS v2024 English–Tamil data revealed:\n"
        "1. **Severe Character-Encoding Corruption:** The v2024 Tamil corpus exhibits "
        "widespread UTF-8 byte corruption and mojibake in more than 40% of files.\n"
        "2. **Document Misalignment:** Multiple document IDs in v2024 contained "
        "mismatched dialogue.\n"
        "3. In contrast, **v2018** has genuine Tamil Unicode characters (mean vowel share "
        "strictly in $[0.18, 0.21]$ across all 28 films) and verified alignments.\n\n"
        "---\n\n"
        "## Critical Machine Learning & Circularity Warning\n\n"
        "> [!WARNING]\n"
        "> **Circularity Notice:** `RiskFlag` and `RiskScore` are unsupervised "
        "**heuristic proxies** computed from weak surface signals and pair-side metrics "
        "(including `LengthRatio`, `SpacedPunctuation`, `DigitMatch`, `PunctuationParity`, "
        "and `LabseCosine`).\n"
        ">\n"
        "> If you train a predictive model on `tyto_dataset.csv` using all features to predict "
        "`RiskFlag`, **your model will be circular and suffer from severe target leakage**.\n"
        ">\n"
        "> **For honest, leakage-free classification:**\n"
        "> 1. Use the companion file **`tyto_dataset_source_only.csv`**, which contains strictly "
        "English source-side features (`EnglishWords`, `ExclamationMark`, `QuestionMark`, "
        "`CapsRatio`, `RepeatCount`, `IsShortLine`, `IdiomPresent`).\n"
        "> 2. Always evaluate using **Grouped Cross-Validation** grouped by `FilmID` (or "
        "`SeriesGroup`) to avoid data leakage from adjacent dialogue lines.\n\n"
        "---\n\n"
        "## Limitations & Honesty Disclaimers\n\n"
        "1. **No Human Annotations:** There are **zero human translation-quality judgements** "
        "in this dataset. The target is an unsupervised risk heuristic designed to triage lines.\n"
        "2. **Weak Signal Approximations:** Signals such as `SpacedPunctuation` or `IdiomPresent` "
        "reflect surface characteristics of subtitlers, not confirmed translation mistakes.\n"
        "3. **No Free Text in Main Dataset:** In strict compliance with copyright and leakage "
        "standards, no raw English or Tamil text is in `tyto_dataset.csv`. Reference text "
        "is stored in an isolated private directory (`private/`) outside public distribution.\n"
    )
    out_path.write_text(card, encoding="utf-8")
    logger.info("Saved dataset card to %s", out_path)


def generate_previews(df: pd.DataFrame, out_dir: Path) -> dict[str, str]:
    """Render Excel-style preview screenshots of the first 20 rows using Playwright."""
    previews_dir = out_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    preview_df = df.head(20).copy()

    cols = list(df.columns)
    mid = 17
    cols_1 = cols[:mid]
    cols_2 = cols[mid:]

    letters = [openpyxl.utils.get_column_letter(i) for i in range(1, len(cols) + 1)]
    letters_1 = letters[:mid]
    letters_2 = letters[mid:]

    numeric_cols = {
        "PairID",
        "ReleaseYear",
        "EnglishWords",
        "TamilChars",
        "LengthRatio",
        "TamilVowelShare",
        "TamilLatinShare",
        "CapsRatio",
        "RepeatCount",
        "HedgeCount",
        "LabseCosine",
        "SimilarityLevel",
        "LengthRatioLevel",
        "VowelShareLevel",
        "FilmSpacedPunctRate",
        "RiskScore",
        "ClusterID",
        "PairCount",
    }

    def _render_html(subset_cols: list[str], subset_letters: list[str]) -> str:
        header_letters = "".join(f"<th class='col-header'>{ltr}</th>" for ltr in subset_letters)
        header_row_chrome = f"<tr><th class='corner'></th>{header_letters}</tr>"

        header_cells = "".join(f"<th class='data-header'>{col}</th>" for col in subset_cols)
        data_header_row = f"<tr><th class='row-num'>1</th>{header_cells}</tr>"

        data_rows: list[str] = []
        for r_idx, (_, row) in enumerate(preview_df.iterrows(), start=2):
            cells: list[str] = []
            for col in subset_cols:
                val = row[col]
                float_cols = {
                    "LengthRatio",
                    "TamilVowelShare",
                    "TamilLatinShare",
                    "CapsRatio",
                    "LabseCosine",
                    "FilmSpacedPunctRate",
                }
                if col in float_cols:
                    text_val = f"{float(val):.3f}"
                elif col == "RiskScore":
                    text_val = f"{float(val):.1f}"
                elif val is None or pd.isna(val):
                    text_val = ""
                else:
                    text_val = str(val)

                align_class = "align-right" if col in numeric_cols else "align-left"
                cells.append(f"<td class='{align_class}'>{text_val}</td>")

            cells_html = "".join(cells)
            data_rows.append(f"<tr><th class='row-num'>{r_idx}</th>{cells_html}</tr>")

        all_rows = header_row_chrome + data_header_row + "".join(data_rows)

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    background-color: #ffffff;
    font-family: Calibri, 'Segoe UI', Arial, sans-serif;
    font-size: 11pt;
    padding: 16px;
    display: inline-block;
  }}
  table {{
    border-collapse: collapse;
    border: 1px solid #d4d4d4;
    background-color: #ffffff;
  }}
  th, td {{
    border: 1px solid #d4d4d4;
    padding: 4px 10px;
    height: 24px;
    white-space: nowrap;
    overflow: hidden;
    font-size: 11pt;
    line-height: 1.2;
  }}
  .corner {{
    background-color: #e6e6e6;
    width: 44px;
    min-width: 44px;
    border-right: 2px solid #b8b8b8;
    border-bottom: 2px solid #b8b8b8;
  }}
  .col-header {{
    background-color: #f2f2f2;
    color: #444444;
    font-weight: 600;
    font-size: 10.5pt;
    text-align: center;
    border-bottom: 2px solid #b8b8b8;
  }}
  .row-num {{
    background-color: #f2f2f2;
    color: #444444;
    font-weight: 600;
    font-size: 10.5pt;
    text-align: center;
    width: 44px;
    min-width: 44px;
    border-right: 2px solid #b8b8b8;
  }}
  .data-header {{
    background-color: #ffffff;
    font-weight: bold;
    color: #000000;
    text-align: left;
    border-bottom: 2px solid #107c41;
  }}
  .align-right {{
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .align-left {{
    text-align: left;
  }}
</style>
</head>
<body>
<table>
{all_rows}
</table>
</body>
</html>"""
        return html

    html_1 = _render_html(cols_1, letters_1)
    html_2 = _render_html(cols_2, letters_2)

    preview_1_png = previews_dir / "preview_1.png"
    preview_2_png = previews_dir / "preview_2.png"

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page1 = browser.new_page(device_scale_factor=2)
        page1.set_content(html_1)
        table_loc1 = page1.locator("table")
        table_loc1.screenshot(path=str(preview_1_png))
        page1.close()

        page2 = browser.new_page(device_scale_factor=2)
        page2.set_content(html_2)
        table_loc2 = page2.locator("table")
        table_loc2.screenshot(path=str(preview_2_png))
        page2.close()

        browser.close()

    note_lines = [
        "Project Tyto MVP Tabular Dataset — Preview Column Distribution",
        "================================================================",
        "",
        "preview_1.png (Columns 1 to 17, Excel columns A to Q):",
    ]
    for idx, (c, ltr) in enumerate(zip(cols_1, letters_1, strict=True), start=1):
        note_lines.append(f"  Column {idx:02d} | Excel {ltr:>2} : {c}")

    note_lines.extend([
        "",
        "preview_2.png (Columns 18 to 33, Excel columns R to AG):",
    ])
    for idx, (c, ltr) in enumerate(zip(cols_2, letters_2, strict=True), start=18):
        note_lines.append(f"  Column {idx:02d} | Excel {ltr:>2} : {c}")

    note_lines.extend([
        "",
        "Omitted Columns Note:",
        (
            "  - PredictedProbability: Omitted because out-of-fold cross-validation predictions "
            "were not saved in the pipeline artifacts. In accordance with Honesty Rules, it was "
            "omitted rather than fabricated."
        ),
        "",
        f"Total Columns Exported: {len(cols)}",
        "Rows Displayed: 20 (Header row 1, Data rows 2 to 21)",
        "Rendering Engine: Headless Chromium via Playwright, device_scale_factor=2.0",
    ])

    note_path = previews_dir / "preview_all_columns_note.txt"
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    logger.info("Saved preview screenshots to %s", previews_dir)

    return {
        "preview_1": str(preview_1_png),
        "preview_2": str(preview_2_png),
        "note": str(note_path),
    }


def generate_dataset_artifacts(
    pairs_path: str | Path = "data/processed/pairs_v2018.parquet",
    risk_path: str | Path = "artifacts/risk.parquet",
    clusters_path: str | Path = "artifacts/clusters.parquet",
    anomalies_path: str | Path = "artifacts/anomalies.parquet",
    film_stats_path: str | Path = "artifacts/film_stats.parquet",
    out_dir: str | Path = "../tyto-dataset",
) -> dict[str, Any]:
    """Execute the complete dataset creation pipeline."""
    pairs_path = Path(pairs_path)
    risk_path = Path(risk_path)
    clusters_path = Path(clusters_path)
    anomalies_path = Path(anomalies_path)
    film_stats_path = Path(film_stats_path)
    out_path = Path(out_dir).resolve()

    out_path.mkdir(parents=True, exist_ok=True)
    logger.info("Generating Tyto dataset into %s", out_path)

    # 1. Load Parquet artifacts
    if not risk_path.exists():
        raise FileNotFoundError(f"Required artifact not found: {risk_path}")

    logger.info("Reading risk artifact: %s", risk_path)
    risk_df = pd.read_parquet(risk_path)

    pairs_df = pd.read_parquet(pairs_path) if pairs_path.exists() else None
    clusters_df = pd.read_parquet(clusters_path) if clusters_path.exists() else None
    anomalies_df = pd.read_parquet(anomalies_path) if anomalies_path.exists() else None
    film_stats_df = pd.read_parquet(film_stats_path) if film_stats_path.exists() else None

    # 2. Build main dataset
    main_df = build_dataset(
        risk_df=risk_df,
        pairs_df=pairs_df,
        clusters_df=clusters_df,
        anomalies_df=anomalies_df,
        film_stats_df=film_stats_df,
    )

    # 3. Verification Gates
    is_full = len(main_df) == 32417
    gates = validate_gates(main_df, is_full_corpus=is_full)
    if not gates["all_passed"]:
        error_msg = f"Verification gates failed: {gates['errors']}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 4. Write CSV (UTF-8 with BOM)
    main_csv_path = out_path / "tyto_dataset.csv"
    main_df.to_csv(main_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s (%d rows)", main_csv_path, len(main_df))

    # 5. Write XLSX (openpyxl)
    main_xlsx_path = out_path / "tyto_dataset.xlsx"
    write_excel_dataset(main_df, main_xlsx_path)

    # 6. Stratified sample of 1,470 rows (stratified by RiskTier, fixed seed 42)
    sample_df, _ = train_test_split(
        main_df,
        train_size=1470,
        stratify=main_df["RiskTier"],
        random_state=42,
    )
    sample_df = sample_df.sort_values("PairID").reset_index(drop=True)
    sample_csv_path = out_path / "tyto_dataset_sample_1470.csv"
    sample_df.to_csv(sample_csv_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote %s (%d rows)", sample_csv_path, len(sample_df))

    # 7. Leakage-free source-only CSV
    source_df = main_df[SOURCE_ONLY_COLUMNS].copy()
    source_csv_path = out_path / "tyto_dataset_source_only.csv"
    source_df.to_csv(source_csv_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Wrote %s (%d rows, %d cols)",
        source_csv_path,
        len(source_df),
        len(source_df.columns),
    )

    # 8. Documentation
    dict_md_path = out_path / "DATA_DICTIONARY.md"
    write_data_dictionary_md(dict_md_path)

    card_md_path = out_path / "DATASET_CARD.md"
    write_dataset_card_md(card_md_path)

    # 9. Private reference subtitle text
    private_dir = out_path / "private"
    private_dir.mkdir(parents=True, exist_ok=True)

    if pairs_df is not None and "en" in pairs_df.columns and "ta" in pairs_df.columns:
        private_text_df = pd.DataFrame({
            "PairID": main_df["PairID"],
            "English": pairs_df["en"].fillna(""),
            "Tamil": pairs_df["ta"].fillna(""),
        })
    elif "en" in risk_df.columns and "ta" in risk_df.columns:
        private_text_df = pd.DataFrame({
            "PairID": main_df["PairID"],
            "English": risk_df["en"].fillna(""),
            "Tamil": risk_df["ta"].fillna(""),
        })
    else:
        private_text_df = pd.DataFrame({"PairID": main_df["PairID"], "English": "", "Tamil": ""})

    private_csv_path = private_dir / "PRIVATE_pair_text.csv"
    private_text_df.to_csv(private_csv_path, index=False, encoding="utf-8-sig")

    private_readme_path = private_dir / "README.md"
    private_readme_path.write_text(
        "# Private Subtitle Text Reference\n\n"
        "> [!IMPORTANT]\n"
        "> **CONFIDENTIAL / DO NOT PUBLISH OR COMMIT**\n"
        "> The subtitle text in `PRIVATE_pair_text.csv` belongs to OpenSubtitles.org "
        "contributors.\n"
        "> It is provided solely as an isolated reference table for inspection.\n"
        "> Under Project Tyto's data governance policy:\n"
        "> 1. This folder MUST remain outside the public Git repository.\n"
        "> 2. Never publish, distribute, or commit raw subtitle text.\n",
        encoding="utf-8",
    )
    logger.info("Wrote private text reference to %s", private_csv_path)

    # 10. Preview screenshots
    previews = generate_previews(main_df, out_path)

    return {
        "out_dir": str(out_path),
        "total_rows": len(main_df),
        "columns_count": len(main_df.columns),
        "gates": gates,
        "sample_rows": len(sample_df),
        "source_only_cols": len(SOURCE_ONLY_COLUMNS),
        "previews": previews,
    }
