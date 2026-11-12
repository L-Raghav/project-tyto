"""Data audit report generator for Phase 1.

Reads ``data/processed/pairs.parquet`` via DuckDB and generates:
- ``reports/data_audit.md``  – Markdown report with statistics and 30-row sample table.
- ``artifacts/fig_pair_counts.png``
- ``artifacts/fig_lines_per_film.png``
- ``artifacts/fig_length_dist.png``
- ``artifacts/fig_script_ratios.png``
- ``artifacts/fig_flag_share.png``
- ``artifacts/fig_labse_scores.png``  (if labse_score column present)

All numbers in the report come from generated artefacts – no fabrication.

Environment: PYTHONUTF8=1
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tyto.utils.config import TytoConfig

logger = logging.getLogger(__name__)


def _save_fig(fig: object, path: Path) -> None:
    """Save a matplotlib figure to *path*, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")  # type: ignore[attr-defined]
    logger.info("Saved figure: %s", path)


def generate_audit(
    cfg: TytoConfig,
    parquet_path: Path | None = None,
    report_path: Path | None = None,
    artifacts_dir: Path | None = None,
    *,
    seed: int = 42,
) -> Path:
    """Generate the Phase 1 data audit report and figures.

    Args:
        cfg:           TytoConfig instance.
        parquet_path:  Path to pairs.parquet (defaults to cfg path).
        report_path:   Destination for the Markdown report (defaults to cfg).
        artifacts_dir: Directory for figure files (defaults to cfg).
        seed:          Random seed for sample table.

    Returns:
        Path to the written Markdown report.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from tyto.data.clean import script_ratios
    from tyto.data.db import PairsDB

    parquet_path = parquet_path or (Path(cfg.paths.processed_dir) / "pairs.parquet")
    report_path = report_path or (Path(cfg.paths.reports_dir) / "data_audit.md")
    artifacts_dir = artifacts_dir or Path(cfg.paths.artifacts_dir)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with PairsDB(parquet_path) as db:
        stats = db.summary_stats()
        film_df = db.film_summary()
        flag_df = db.flag_summary()
        sample_df = db.sample_pairs(n=30, seed=seed)
        len_df = db.length_distribution()
        all_df = db.query("SELECT ta, qc_flags FROM pairs")

    # ── Figure 1: Pair and film counts ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["Total pairs", "Usable pairs", "Films"],
           [stats["total_pairs"], stats["usable_pairs"], stats["n_films"]],
           color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title("Pair and Film Counts")
    ax.set_ylabel("Count")
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * bar.get_height(),
                f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=9)
    fig_pair_path = artifacts_dir / "fig_pair_counts.png"
    _save_fig(fig, fig_pair_path)
    plt.close(fig)

    # ── Figure 2: Lines per film distribution ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3))
    film_df["n_pairs"].hist(bins=40, ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title("Lines per Film Distribution")
    ax.set_xlabel("Pairs per film")
    ax.set_ylabel("Film count")
    fig_film_path = artifacts_dir / "fig_lines_per_film.png"
    _save_fig(fig, fig_film_path)
    plt.close(fig)

    # ── Figure 3: Length distributions ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    len_df["en_len"].clip(upper=300).hist(bins=60, ax=axes[0], color="#4C72B0", edgecolor="white")
    axes[0].set_title("English character length")
    axes[0].set_xlabel("Characters")
    len_df["ta_len"].clip(upper=300).hist(bins=60, ax=axes[1], color="#55A868", edgecolor="white")
    axes[1].set_title("Tamil character length")
    axes[1].set_xlabel("Characters")
    fig_len_path = artifacts_dir / "fig_length_dist.png"
    _save_fig(fig, fig_len_path)
    plt.close(fig)

    # ── Figure 4: Script ratios (Tamil block ratio distribution) ─────────────
    logger.info("Computing script ratios for %d rows …", len(all_df))
    ratios = all_df["ta"].fillna("").apply(lambda t: script_ratios(t)[0])
    fig, ax = plt.subplots(figsize=(7, 3))
    ratios.hist(bins=50, ax=ax, color="#DD8452", edgecolor="white")
    label_min = f"min ({cfg.qc.tamil_ratio_min})"
    label_cm = f"code-mix ({cfg.qc.tamil_ratio_code_mix})"
    ax.axvline(cfg.qc.tamil_ratio_min, color="red", linestyle="--", label=label_min)
    ax.axvline(
        cfg.qc.tamil_ratio_code_mix, color="orange", linestyle="--", label=label_cm
    )
    ax.set_title("Tamil Unicode Block Ratio Distribution")
    ax.set_xlabel("Tamil character ratio")
    ax.set_ylabel("Sentence count")
    ax.legend()
    fig_script_path = artifacts_dir / "fig_script_ratios.png"
    _save_fig(fig, fig_script_path)
    plt.close(fig)

    romanised_n = int((all_df["qc_flags"].str.contains("ROMANISED_TA", na=False)).sum())
    romanised_pct = 100.0 * romanised_n / max(len(all_df), 1)

    # ── Figure 5: Flag share ─────────────────────────────────────────────────
    fig_flag_path = artifacts_dir / "fig_flag_share.png"
    if not flag_df.empty:
        fig, ax = plt.subplots(figsize=(8, max(3, len(flag_df) * 0.4)))
        colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
        flag_df.sort_values("n_rows").plot.barh(
            x="flag", y="n_rows", ax=ax, color=colors[: len(flag_df)], legend=False
        )
        ax.set_title("QC Flag Counts")
        ax.set_xlabel("Rows flagged")
        _save_fig(fig, fig_flag_path)
        plt.close(fig)

    # ── Figure 6: LaBSE score histogram (if column present) ──────────────────
    fig_labse_path = artifacts_dir / "fig_labse_scores.png"
    labse_available = False
    try:
        with PairsDB(parquet_path) as db:
            labse_sql = (
                "SELECT labse_score FROM pairs "
                "WHERE labse_score IS NOT NULL LIMIT 50000"
            )
            labse_df = db.query(labse_sql)
        if not labse_df.empty and labse_df["labse_score"].notna().any():
            labse_available = True
            fig, ax = plt.subplots(figsize=(7, 3))
            labse_df["labse_score"].hist(bins=60, ax=ax, color="#8172B3", edgecolor="white")
            ax.axvline(cfg.qc.labse_min_score, color="red", linestyle="--",
                       label=f"min ({cfg.qc.labse_min_score})")
            ax.set_title("LaBSE Alignment Score Distribution")
            ax.set_xlabel("Cosine similarity")
            ax.set_ylabel("Count")
            ax.legend()
            _save_fig(fig, fig_labse_path)
            plt.close(fig)
    except Exception as exc:
        logger.info("No LaBSE scores in parquet (stage 2 not yet run): %s", exc)

    # ── Markdown report ──────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("# Phase 1 Data Audit\n")
    lines.append("**Assumptions**\n")
    lines.append(
        "- Source: OPUS OpenSubtitles en-ta v" + cfg.opus.version + ".\n"
        "- All numbers in this report are derived from `data/processed/pairs.parquet`.\n"
        "- QC thresholds are read from `configs/default.yaml`.\n"
        "- LaBSE stage-2 scoring is optional; run `python -m tyto data qc --labse` to enable.\n"
    )
    lines.append("\n---\n")

    lines.append("## 1. Summary Statistics\n")
    lines.append("| Metric | Value |\n|---|---|\n")
    lines.append(f"| Total pairs | {stats['total_pairs']:,} |\n")
    lines.append(f"| Usable pairs | {stats['usable_pairs']:,} |\n")
    lines.append(f"| Usable % | {stats['usable_pct']:.1f}% |\n")
    lines.append(f"| Distinct films | {stats['n_films']:,} |\n")
    lines.append(f"| Romanised-Tamil rate | {romanised_pct:.2f}% ({romanised_n:,} rows) |\n")

    lines.append("\n## 2. Figures\n")
    lines.append(f"![Pair and film counts]({fig_pair_path})\n\n")
    lines.append(f"![Lines per film distribution]({fig_film_path})\n\n")
    lines.append(f"![Length distributions]({fig_len_path})\n\n")
    lines.append(f"![Script ratios]({fig_script_path})\n\n")
    if not flag_df.empty:
        lines.append(f"![QC flag share]({fig_flag_path})\n\n")
    if labse_available:
        lines.append(f"![LaBSE score histogram]({fig_labse_path})\n\n")
    else:
        lines.append("*LaBSE histogram: not yet available (run stage-2 scoring first).*\n\n")

    lines.append("## 3. QC Flag Counts\n")
    if flag_df.empty:
        lines.append("No QC flags set.\n")
    else:
        lines.append(flag_df.to_markdown(index=False) + "\n")

    lines.append("\n## 4. Top Films by Pair Count\n")
    lines.append(film_df.head(20).to_markdown(index=False) + "\n")

    lines.append("\n## 5. Random Sample (30 pairs)\n")
    # Truncate long cells for readability.
    display_sample = sample_df.copy()
    for col in ["en_raw", "ta_raw", "en", "ta"]:
        if col in display_sample.columns:
            display_sample[col] = display_sample[col].str[:120]
    lines.append(display_sample.to_markdown(index=False) + "\n")

    lines.append("\n## 6. Stop-Gate Status\n")
    threshold = cfg.qc.min_usable_pairs
    if stats["usable_pairs"] >= threshold:
        lines.append(
            f"✅ **PASS**: {stats['usable_pairs']:,} usable pairs ≥ threshold {threshold:,}.\n"
        )
    else:
        lines.append(
            f"🚫 **FAIL**: Only {stats['usable_pairs']:,} usable pairs < threshold {threshold:,}.\n"
            "Options:\n"
            "- Lower `qc.min_usable_pairs` in `configs/default.yaml`.\n"
            "- Relax QC thresholds (e.g. `max_length_ratio`, `min_en_tokens`).\n"
            "- Add supplementary SRT data with `python -m tyto data srt-align`.\n"
        )

    lines.append("\n---\n*Generated by `python -m tyto data audit`.*\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    logger.info("Data audit report written to %s", report_path)
    return report_path
