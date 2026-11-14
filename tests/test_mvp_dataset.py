"""Tests for the MVP course-ready tabular dataset generator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from tyto.cli import app
from tyto.mvp.dataset import (
    DATASET_COLUMNS,
    SERIES_A_FILM_IDS,
    SOURCE_ONLY_COLUMNS,
    build_dataset,
    validate_gates,
    write_data_dictionary_md,
    write_dataset_card_md,
    write_excel_dataset,
)


@pytest.fixture
def synthetic_risk_df() -> pd.DataFrame:
    """Create a synthetic 30-row risk dataframe without loading real corpus or GPU."""
    n = 30
    rng = np.random.default_rng(42)

    film_ids = ["395461"] * 10 + ["6261062"] * 10 + ["1032751"] * 10
    link_types = ["1:1"] * 20 + ["1:2"] * 5 + ["2:1"] * 3 + ["1:3"] * 2

    # Scores from 5.0 to 80.0
    scores = np.linspace(5.0, 80.0, n)
    p70 = float(np.percentile(scores, 70))
    p90 = float(np.percentile(scores, 90))
    tiers = np.where(scores >= p90, "High", np.where(scores >= p70, "Medium", "Low"))

    df = pd.DataFrame({
        "pair_id": np.arange(n),
        "film_id": film_ids,
        "year": 2012,
        "series_group": np.where([f in SERIES_A_FILM_IDS for f in film_ids], "series_A", ""),
        "link_type": link_types,
        "en": [f"English sentence {i}" for i in range(n)],
        "ta": [f"தமிழ் வாக்கியம் {i}" for i in range(n)],
        "en_words": rng.integers(1, 20, size=n),
        "ta_chars": rng.integers(10, 80, size=n),
        "len_ratio": rng.uniform(1.5, 9.0, size=n),
        "ta_vowel_share": rng.uniform(0.12, 0.25, size=n),
        "ta_latin_share": np.where(np.arange(n) % 5 == 0, 0.15, 0.0),
        "ta_spaced_punct": np.where(np.arange(n) % 6 == 0, 0.05, 0.0),
        "en_exclaim": np.where(np.arange(n) % 4 == 0, 1, 0),
        "en_question": np.where(np.arange(n) % 3 == 0, 1, 0),
        "en_caps_ratio": rng.uniform(0.0, 0.3, size=n),
        "digit_match": np.where(np.arange(n) % 7 == 0, 1, 0),
        "punct_parity": np.where(np.arange(n) % 8 == 0, 1, 0),
        "repeat_count": rng.integers(1, 4, size=n),
        "is_short": np.where(np.arange(n) % 2 == 0, 1, 0),
        "idiom_present": np.where(np.arange(n) % 10 == 0, 1, 0),
        "ta_hedge_count": np.where(np.arange(n) % 9 == 0, 1, 0),
        "labse_cos": np.linspace(0.40, 0.98, n),
        "risk_score": scores,
        "risk_tier": tiers,
        "cluster_id": np.arange(n) % 4,
    })
    return df


@pytest.fixture
def synthetic_film_stats_df() -> pd.DataFrame:
    """Create a synthetic 3-film statistics dataframe."""
    return pd.DataFrame({
        "film_id": ["395461", "6261062", "1032751"],
        "pairs": [10, 10, 10],
        "mean_risk": [25.0, 45.0, 30.0],
        "spaced_punct_rate": [0.012, 0.005, 0.020],
        "latin_share": [0.001, 0.000, 0.002],
        "mean_len_ratio": [4.2, 4.8, 3.9],
    })


@pytest.fixture
def synthetic_anomalies_df() -> pd.DataFrame:
    """Create a synthetic anomalies dataframe with 2 anomaly pairs."""
    return pd.DataFrame({
        "pair_id": [3, 17],
        "anomaly_score": [-0.15, -0.22],
    })


def test_column_set_and_order(
    synthetic_risk_df: pd.DataFrame,
    synthetic_film_stats_df: pd.DataFrame,
    synthetic_anomalies_df: pd.DataFrame,
) -> None:
    """Verify that build_dataset produces exactly the 33 DATASET_COLUMNS in order."""
    df = build_dataset(
        risk_df=synthetic_risk_df,
        clusters_df=synthetic_risk_df[["pair_id", "cluster_id"]],
        anomalies_df=synthetic_anomalies_df,
        film_stats_df=synthetic_film_stats_df,
    )
    assert list(df.columns) == DATASET_COLUMNS
    assert len(df.columns) == 33
    assert len(df) == len(synthetic_risk_df)


def test_unique_pair_id(
    synthetic_risk_df: pd.DataFrame,
) -> None:
    """Verify PairID is 1-indexed, unique, and strictly positive integers."""
    df = build_dataset(risk_df=synthetic_risk_df)
    assert df["PairID"].is_unique
    assert df["PairID"].min() == 1
    assert df["PairID"].max() == len(synthetic_risk_df)
    assert df["PairID"].iloc[0] == 1
    assert df["PairID"].iloc[-1] == len(synthetic_risk_df)


def test_yes_no_columns_strictly_boolean_strings(
    synthetic_risk_df: pd.DataFrame,
    synthetic_anomalies_df: pd.DataFrame,
) -> None:
    """Verify all indicator columns contain only literal 'Yes' or 'No' strings."""
    df = build_dataset(
        risk_df=synthetic_risk_df,
        anomalies_df=synthetic_anomalies_df,
    )
    yes_no_cols = [
        "RiskFlag",
        "SpacedPunctuation",
        "ExclamationMark",
        "QuestionMark",
        "DigitMatch",
        "PunctuationParity",
        "IsShortLine",
        "IdiomPresent",
        "IsAnomaly",
    ]
    for col in yes_no_cols:
        unique_vals = set(df[col].unique())
        assert unique_vals.issubset({"Yes", "No"}), f"{col} has invalid values: {unique_vals}"


def test_quartile_levels_range(
    synthetic_risk_df: pd.DataFrame,
) -> None:
    """Verify quartile levels take values strictly in {1, 2, 3, 4}."""
    df = build_dataset(risk_df=synthetic_risk_df)
    for col in ["SimilarityLevel", "LengthRatioLevel", "VowelShareLevel"]:
        vals = set(df[col].unique())
        assert vals.issubset({1, 2, 3, 4}), f"{col} has invalid levels: {vals}"


def test_constant_columns(
    synthetic_risk_df: pd.DataFrame,
) -> None:
    """Verify constant columns are present with expected values."""
    df = build_dataset(risk_df=synthetic_risk_df)
    assert (df["LanguagePair"] == "en-ta").all()
    assert (df["PairCount"] == 1).all()


def test_source_only_no_pair_side_columns() -> None:
    """Verify source-only file contains no pair-side leakage features."""
    pair_side_forbidden = {
        "TamilChars",
        "LengthRatio",
        "LengthBand",
        "TamilVowelShare",
        "TamilLatinShare",
        "SpacedPunctuation",
        "DigitMatch",
        "PunctuationParity",
        "HedgeCount",
        "LabseCosine",
        "SimilarityLevel",
        "LengthRatioLevel",
        "VowelShareLevel",
        "FilmSpacedPunctRate",
        "RiskScore",
        "ClusterID",
        "IsAnomaly",
    }
    overlap = set(SOURCE_ONLY_COLUMNS).intersection(pair_side_forbidden)
    assert not overlap, f"Forbidden pair-side columns found in source-only spec: {overlap}"
    assert "RiskFlag" in SOURCE_ONLY_COLUMNS
    assert "EnglishWords" in SOURCE_ONLY_COLUMNS
    assert "IdiomPresent" in SOURCE_ONLY_COLUMNS


def test_link_type_mapping(synthetic_risk_df: pd.DataFrame) -> None:
    """Verify link_type maps non-standard alignments to 'Other'."""
    df = build_dataset(risk_df=synthetic_risk_df)
    assert set(df["LinkType"].unique()).issubset({"1:1", "1:2", "2:1", "Other"})
    assert "Other" in df["LinkType"].values


def test_series_group_franchise_mapping(synthetic_risk_df: pd.DataFrame) -> None:
    """Verify SeriesGroup is 'series_A' for franchise film IDs and None/empty otherwise."""
    df = build_dataset(risk_df=synthetic_risk_df)
    franchise_rows = df[df["FilmID"] == "6261062"]
    other_rows = df[df["FilmID"] == "395461"]

    assert (franchise_rows["SeriesGroup"] == "series_A").all()
    assert other_rows["SeriesGroup"].isna().all()


def test_validate_gates_synthetic(synthetic_risk_df: pd.DataFrame) -> None:
    """Verify validate_gates passes on synthetic data when is_full_corpus=False."""
    df = build_dataset(risk_df=synthetic_risk_df)
    gates = validate_gates(df, is_full_corpus=False)
    assert gates["is_pair_id_unique"] is True
    assert gates["all_passed"] is True
    assert len(gates["errors"]) == 0


def test_excel_and_docs_generation(
    synthetic_risk_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verify openpyxl writer and documentation files generate without errors."""
    df = build_dataset(risk_df=synthetic_risk_df)
    xlsx_path = tmp_path / "test_dataset.xlsx"
    write_excel_dataset(df, xlsx_path)
    assert xlsx_path.exists()
    assert xlsx_path.stat().st_size > 0

    dict_md = tmp_path / "DATA_DICTIONARY.md"
    write_data_dictionary_md(dict_md)
    assert dict_md.exists()
    assert "RiskFlag" in dict_md.read_text(encoding="utf-8")

    card_md = tmp_path / "DATASET_CARD.md"
    write_dataset_card_md(card_md)
    assert card_md.exists()
    assert "Circularity Notice" in card_md.read_text(encoding="utf-8")


def test_cli_runner_mvp_dataset() -> None:
    """Verify that 'python -m tyto mvp dataset --help' succeeds via CliRunner."""
    runner = CliRunner()
    result = runner.invoke(app, ["mvp", "dataset", "--help"])
    assert result.exit_code == 0
    assert "Build spreadsheet-style course-ready tabular dataset" in result.output
    assert "--out" in result.output
