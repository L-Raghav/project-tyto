"""Power BI CSV export and Build Guide PDF generator for Project Tyto MVP (Stage 8)."""

from __future__ import annotations

import json
import logging
import re
import zlib
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

_RISK_TIER_MAP: dict[str, int] = {"Low": 1, "Medium": 2, "High": 3}

_RISK_TIER_ROWS: list[dict[str, Any]] = [
    {
        "risk_tier_id": 1,
        "tier_name": "Low",
        "tier_description": (
            "Heuristic risk score below the 70th corpus percentile. "
            "No human quality label; lower priority for manual review."
        ),
    },
    {
        "risk_tier_id": 2,
        "tier_name": "Medium",
        "tier_description": (
            "Heuristic risk score from the 70th up to the 90th corpus percentile. "
            "Moderate weak-signal flags; secondary review tier."
        ),
    },
    {
        "risk_tier_id": 3,
        "tier_name": "High",
        "tier_description": (
            "Heuristic risk score at or above the 90th corpus percentile. "
            "Highest concentration of weak signals; top triage priority."
        ),
    },
]

_DATA_DICT_ROWS: list[tuple[str, str, str, str]] = [
    ("pair_id", "fact_pairs.csv", "Integer", "Primary key of subtitle pair (0..N-1)."),
    ("film_id", "fact_pairs.csv", "Integer", "FK referencing dim_film[film_id_key]."),
    ("cluster_id", "fact_pairs.csv", "Integer", "FK referencing dim_cluster[cluster_id]."),
    (
        "risk_tier_id",
        "fact_pairs.csv",
        "Integer",
        "FK referencing dim_risk_tier[risk_tier_id] (1=Low, 2=Medium, 3=High).",
    ),
    ("en", "fact_pairs.csv", "Text", "Cleaned English subtitle line."),
    ("ta", "fact_pairs.csv", "Text", "Cleaned Tamil subtitle line."),
    (
        "risk_score",
        "fact_pairs.csv",
        "Float",
        "Heuristic risk proxy score (0-100); not a human judgement.",
    ),
    (
        "labse_cos",
        "fact_pairs.csv",
        "Float",
        "LaBSE bilingual embedding cosine similarity (nullable if unavailable).",
    ),
    ("en_words", "fact_pairs.csv", "Integer", "Whitespace-delimited English word count."),
    ("ta_chars", "fact_pairs.csv", "Integer", "Character count of the Tamil line."),
    ("len_ratio", "fact_pairs.csv", "Float", "ta_chars / (en_words + 1), capped at 999."),
    (
        "ta_vowel_share",
        "fact_pairs.csv",
        "Float",
        "Share of Tamil dependent vowel signs (U+0BBE-U+0BCC) in ta.",
    ),
    ("ta_latin_share", "fact_pairs.csv", "Float", "Share of ASCII Latin letters in ta."),
    (
        "ta_spaced_punct",
        "fact_pairs.csv",
        "Float",
        "Rate of spaced punctuation patterns (' , ', ' . ', ' ! ', ' ? ') in ta.",
    ),
    ("en_exclaim", "fact_pairs.csv", "Integer", "Binary indicator (1/0): '!' present in en."),
    ("en_question", "fact_pairs.csv", "Integer", "Binary indicator (1/0): '?' present in en."),
    (
        "idiom_present",
        "fact_pairs.csv",
        "Integer",
        "Binary indicator (1/0): match in illustrative 40-phrase English idiom list.",
    ),
    ("is_short", "fact_pairs.csv", "Integer", "Binary indicator (1/0): en_words <= 3."),
    (
        "is_anomaly",
        "fact_pairs.csv",
        "Integer",
        "Binary indicator (1/0): pair appears in IsolationForest top anomalies.",
    ),
    ("film_id_key", "dim_film.csv", "Integer", "Surrogate integer PK for dim_film."),
    ("film_id", "dim_film.csv", "Text", "OPUS IMDb document ID (e.g. 395461)."),
    ("year", "dim_film.csv", "Text", "Release year from OPUS document path."),
    (
        "series_group",
        "dim_film.csv",
        "Text",
        "Optional hypothesis grouping (series_A for 6 clustered IMDb IDs, else empty).",
    ),
    ("pairs_count", "dim_film.csv", "Integer", "Total aligned subtitle pairs in this film."),
    ("mean_risk", "dim_film.csv", "Float", "Mean heuristic risk_score across film pairs."),
    (
        "spaced_punct_rate",
        "dim_film.csv",
        "Float",
        "Fraction of pairs in this film with ta_spaced_punct > 0.",
    ),
    ("latin_share", "dim_film.csv", "Float", "Mean ta_latin_share across film pairs."),
    ("mean_len_ratio", "dim_film.csv", "Float", "Mean len_ratio across film pairs."),
    ("cluster_id", "dim_cluster.csv", "Integer", "PK of KMeans cluster (0..K-1)."),
    ("cluster_size", "dim_cluster.csv", "Integer", "Number of pairs assigned to cluster."),
    ("mean_risk", "dim_cluster.csv", "Float", "Mean heuristic risk_score in cluster."),
    ("mean_labse_cos", "dim_cluster.csv", "Float", "Mean labse_cos in cluster."),
    ("mean_len_ratio", "dim_cluster.csv", "Float", "Mean len_ratio in cluster."),
    (
        "description",
        "dim_cluster.csv",
        "Text",
        "Concise summary of cluster size, risk level, length ratio, and similarity.",
    ),
    ("risk_tier_id", "dim_risk_tier.csv", "Integer", "PK for risk tier (1/2/3)."),
    ("tier_name", "dim_risk_tier.csv", "Text", "Tier name (Low, Medium, High)."),
    ("tier_description", "dim_risk_tier.csv", "Text", "Percentile rule and proxy disclosure."),
    ("rule_id", "fact_rules.csv", "Integer", "Surrogate PK (1..R) for FP-Growth rules."),
    ("antecedents", "fact_rules.csv", "Text", "Comma-separated antecedent items."),
    ("consequents", "fact_rules.csv", "Text", "Comma-separated consequent items."),
    ("support", "fact_rules.csv", "Float", "Joint occurrence fraction in corpus."),
    ("confidence", "fact_rules.csv", "Float", "Conditional probability P(consequent|antecedent)."),
    ("lift", "fact_rules.csv", "Float", "Co-occurrence ratio over independence (>1 positive)."),
    (
        "anomaly_rank",
        "fact_anomalies.csv",
        "Integer",
        "Rank ordered by lowest IsolationForest score (1 = most anomalous).",
    ),
    ("pair_id", "fact_anomalies.csv", "Integer", "FK referencing fact_pairs[pair_id]."),
    ("film_id", "fact_anomalies.csv", "Integer", "FK referencing dim_film[film_id_key]."),
    ("en", "fact_anomalies.csv", "Text", "Cleaned English subtitle line."),
    ("ta", "fact_anomalies.csv", "Text", "Cleaned Tamil subtitle line."),
    ("risk_score", "fact_anomalies.csv", "Float", "Heuristic risk proxy score."),
    (
        "anomaly_score",
        "fact_anomalies.csv",
        "Float",
        "IsolationForest decision function score (lower = more anomalous).",
    ),
    ("model_name", "fact_model_metrics.csv", "Text", "Predictor or baseline identifier."),
    ("auroc", "fact_model_metrics.csv", "Float", "Mean GroupKFold AUROC on heuristic proxy."),
    ("pr_auc", "fact_model_metrics.csv", "Float", "Mean GroupKFold Precision-Recall AUC."),
    ("auroc_ci_lo", "fact_model_metrics.csv", "Float", "Lower CI bound for AUROC (if computed)."),
    ("auroc_ci_hi", "fact_model_metrics.csv", "Float", "Upper CI bound for AUROC (if computed)."),
    ("pr_auc_ci_lo", "fact_model_metrics.csv", "Float", "Lower CI bound for PR-AUC (if computed)."),
    ("pr_auc_ci_hi", "fact_model_metrics.csv", "Float", "Upper CI bound for PR-AUC (if computed)."),
    ("recall_at_10", "fact_model_metrics.csv", "Float", "Recall in top 10% predicted scores."),
    ("note", "fact_model_metrics.csv", "Text", "Evaluation scope and heuristic-proxy disclosure."),
]


def _resolve_powerbi_dir(out_dir: str | Path) -> Path:
    """Resolve exports/powerbi directory whether caller passes 'exports' or 'exports/powerbi'."""
    out_path = Path(out_dir)
    if out_path.name.lower() == "powerbi":
        return out_path
    return out_path / "powerbi"


def _ascii_safe(text: str) -> str:
    """Ensure English-only ASCII text for ReportLab PDF generation (never emit Tamil script)."""
    return text.encode("ascii", errors="replace").decode("ascii")


def _describe_cluster(row: pd.Series, overall_risk: float, overall_len: float) -> str:
    """Build a concise English description for a cluster based on real aggregate stats."""
    risk_rel = (
        "higher-risk"
        if row["mean_risk"] >= overall_risk * 1.1
        else ("lower-risk" if row["mean_risk"] <= overall_risk * 0.9 else "moderate-risk")
    )
    len_rel = (
        "longer Tamil"
        if row["mean_len_ratio"] >= overall_len * 1.15
        else (
            "shorter Tamil"
            if row["mean_len_ratio"] <= overall_len * 0.85
            else "typical length ratio"
        )
    )
    cos_val = row.get("mean_labse_cos")
    cos_txt = f"mean LaBSE cos {float(cos_val):.3f}" if pd.notna(cos_val) else "LaBSE unavailable"
    return (
        f"Cluster {int(row['cluster_id'])}: {int(row['cluster_size']):,} pairs, "
        f"{risk_rel} (mean risk {float(row['mean_risk']):.1f}), "
        f"{len_rel} (ratio {float(row['mean_len_ratio']):.2f}), {cos_txt}."
    )


def _normalize_metrics_rows(metrics_data: Any) -> list[dict[str, Any]]:
    """Convert predictor_metrics.json (dict or list) into normalized rows for fact_model_metrics."""
    rows: list[dict[str, Any]] = []
    default_note = (
        str(metrics_data.get("note"))
        if isinstance(metrics_data, dict) and metrics_data.get("note")
        else "Evaluated via GroupKFold by film on heuristic risk proxy (no human labels)."
    )

    def _extract_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "model_name": str(payload.get("model_name", name)),
            "auroc": payload.get("auroc", payload.get("roc_auc")),
            "pr_auc": payload.get("pr_auc", payload.get("average_precision")),
            "auroc_ci_lo": payload.get("auroc_ci_lo"),
            "auroc_ci_hi": payload.get("auroc_ci_hi"),
            "pr_auc_ci_lo": payload.get("pr_auc_ci_lo"),
            "pr_auc_ci_hi": payload.get("pr_auc_ci_hi"),
            "recall_at_10": payload.get("recall_at_10", payload.get("recall@10")),
            "note": str(payload.get("note", default_note)),
        }

    if isinstance(metrics_data, list):
        for idx, item in enumerate(metrics_data):
            if isinstance(item, dict):
                rows.append(_extract_row(f"model_{idx}", item))
    elif isinstance(metrics_data, dict):
        if "models" in metrics_data and isinstance(metrics_data["models"], dict):
            for name, payload in metrics_data["models"].items():
                if isinstance(payload, dict):
                    rows.append(_extract_row(str(name), payload))
        elif "models" in metrics_data and isinstance(metrics_data["models"], list):
            for idx, item in enumerate(metrics_data["models"]):
                if isinstance(item, dict):
                    rows.append(_extract_row(f"model_{idx}", item))
        elif any(k in metrics_data for k in ("auroc", "roc_auc", "pr_auc")):
            rows.append(
                _extract_row(str(metrics_data.get("model_name", "lightgbm")), metrics_data)
            )
        else:
            for name, payload in metrics_data.items():
                if isinstance(payload, dict):
                    rows.append(_extract_row(str(name), payload))

    if not rows:
        rows.append(
            {
                "model_name": "unavailable",
                "auroc": None,
                "pr_auc": None,
                "auroc_ci_lo": None,
                "auroc_ci_hi": None,
                "pr_auc_ci_lo": None,
                "pr_auc_ci_hi": None,
                "recall_at_10": None,
                "note": "No predictor metrics found in predictor_metrics.json.",
            }
        )
    return rows


def _write_data_dictionary(powerbi_dir: Path) -> Path:
    """Write exports/powerbi/DATA_DICTIONARY.md covering every exported CSV column."""
    lines = [
        "# Power BI Data Dictionary",
        "",
        "All CSV files are encoded as UTF-8 with BOM (`utf-8-sig`) so Microsoft Power BI",
        "Desktop and Excel detect Tamil Unicode characters automatically.",
        "",
        "> **Important Disclosure:** `risk_score` and `risk_tier_id` represent",
        "> **heuristic proxies** computed from weak observable signals. No human quality",
        "> annotations exist yet.",
        "",
        "| Column | File | Type | Description |",
        "|--------|------|------|-------------|",
    ]
    for col, fname, dtype, desc in _DATA_DICT_ROWS:
        lines.append(f"| `{col}` | `{fname}` | {dtype} | {desc} |")
    lines.append("")

    out_path = powerbi_dir / "DATA_DICTIONARY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_relationships_md(powerbi_dir: Path) -> Path:
    """Write exports/powerbi/RELATIONSHIPS.md listing foreign key relationships."""
    content = "\n".join(
        [
            "# Power BI Star-Schema Relationships",
            "",
            "Configure the following relationships in Power BI Desktop (**Model** view):",
            "",
            "1. `fact_pairs[film_id]` -> `dim_film[film_id_key]`",
            "   - Cardinality: Many-to-One (`*:1`)",
            "   - Cross-filter direction: Single (`dim_film` filters `fact_pairs`)",
            "   - Active: Yes",
            "",
            "2. `fact_pairs[cluster_id]` -> `dim_cluster[cluster_id]`",
            "   - Cardinality: Many-to-One (`*:1`)",
            "   - Cross-filter direction: Single (`dim_cluster` filters `fact_pairs`)",
            "   - Active: Yes",
            "",
            "3. `fact_pairs[risk_tier_id]` -> `dim_risk_tier[risk_tier_id]`",
            "   - Cardinality: Many-to-One (`*:1`)",
            "   - Cross-filter direction: Single (`dim_risk_tier` filters `fact_pairs`)",
            "   - Active: Yes",
            "",
            "4. `fact_anomalies[pair_id]` -> `fact_pairs[pair_id]`",
            "   - Cardinality: Many-to-One or One-to-One (`1:1` / `*:1`)",
            "   - Cross-filter direction: Single (`fact_pairs` filters `fact_anomalies`)",
            "   - Active: Yes",
            "",
            "5. `fact_anomalies[film_id]` -> `dim_film[film_id_key]`",
            "   - Cardinality: Many-to-One (`*:1`)",
            "   - Cross-filter direction: Single",
            "   - Active: **No** (mark inactive so `dim_film -> fact_anomalies` has no",
            "     ambiguous path alongside relationship #4)",
            "",
            "> **Standalone tables:** `fact_rules.csv` and `fact_model_metrics.csv` are",
            "> corpus-level summary tables displayed without row-level joins.",
            "",
        ]
    )
    out_path = powerbi_dir / "RELATIONSHIPS.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


class _BookmarkDocTemplate(BaseDocTemplate):
    """DocTemplate that records PDF outline bookmarks for every heading."""

    def __init__(self, filename: str, **kwargs: Any) -> None:
        super().__init__(filename, **kwargs)
        self.recorded_headings: list[str] = []

    def afterFlowable(self, flowable: Any) -> None:
        if isinstance(flowable, Paragraph):
            style_name = getattr(flowable.style, "name", "")
            text = _ascii_safe(flowable.getPlainText()).strip()
            if style_name == "SectionHeading" and text:
                key = f"sec_{len(self.recorded_headings)}"
                self.canv.bookmarkPage(key)
                self.canv.addOutlineEntry(text, key, level=0, closed=False)
                self.recorded_headings.append(text)


def _build_powerbi_pdf(
    pdf_path: Path,
    stats: dict[str, Any],
) -> tuple[Path, list[str]]:
    """Generate exports/powerbi/PowerBI_Build_Guide.pdf (English only, no Tamil script)."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "GuideTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#1B365D"),
        spaceAfter=16,
    )
    subtitle_style = ParagraphStyle(
        "GuideSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1B365D"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "GuideBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        spaceAfter=6,
    )
    code_style = ParagraphStyle(
        "GuideCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8.2,
        leading=11,
        backColor=colors.HexColor("#F7FAFC"),
        borderPadding=6,
        spaceAfter=8,
    )
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
    )

    def _p(text: str, style: ParagraphStyle = body_style) -> Paragraph:
        return Paragraph(_ascii_safe(text), style)

    def _tbl(data: list[list[str]], col_widths: list[float]) -> Table:
        wrapped = [[_p(cell, cell_style) for cell in row] for row in data]
        t = Table(wrapped, colWidths=col_widths, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B365D")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#F7FAFC")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return t

    doc = _BookmarkDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="Project Tyto: Power BI Dashboard Build Guide",
        author="Project Tyto",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    story: list[Any] = []

    # 1. Title Page
    story.append(Spacer(1, 1.8 * inch))
    story.append(_p("Project Tyto: Power BI Dashboard Build Guide", title_style))
    story.append(
        _p(
            "Step-by-step instructions for building the English-to-Tamil subtitle "
            "quality triage dashboard from exported CSV artifacts.",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        _p(
            f"Corpus Snapshot: {stats['total_pairs']:,} subtitle pairs across "
            f"{stats['films_count']} films | High-Risk Heuristic Tier: "
            f"{stats['high_risk_pairs']:,} pairs | Association Rules: {stats['rules_count']}",
            body_style,
        )
    )
    story.append(
        _p(
            "Disclosure: Risk scores and tiers in this dataset are heuristic proxies derived "
            "from observable surface and alignment signals. No human quality labels exist yet.",
            body_style,
        )
    )
    story.append(PageBreak())

    # 2. What We Are Building
    story.append(_p("1. What We Are Building", h1_style))
    story.append(
        _p(
            "This dashboard gives reviewers and project stakeholders an interactive Power BI "
            "view over the Project Tyto MVP dataset (OPUS OpenSubtitles v2018 English-Tamil). "
            "It connects pair-level features, heuristic risk tiers, KMeans clusters, "
            "IsolationForest anomalies, FP-Growth association rules, and GroupKFold metrics."
        )
    )
    story.append(
        _p(
            "Important Scope Note: Every risk score and risk tier in the dashboard is a "
            "heuristic proxy, not a ground-truth human translation grade. The dashboard helps "
            "human reviewers filter by film, risk tier, and similarity score."
        )
    )
    story.append(PageBreak())

    # 3. Prerequisites
    story.append(_p("2. Prerequisites", h1_style))
    story.append(
        _p(
            "- Microsoft Power BI Desktop (Windows-only application; free for local authoring "
            "and viewing). Download directly from microsoft.com or the Microsoft Store."
        )
    )
    story.append(
        _p(
            "- All seven CSV files inside exports/powerbi/ generated by running: "
            ".venv\\Scripts\\python.exe -m tyto mvp export"
        )
    )
    story.append(
        _p(
            "- Windows font support for Tamil script (Nirmala UI is pre-installed on Windows "
            "10 and Windows 11) for rendering the 'ta' column inside table visuals."
        )
    )

    # 4. Files and Every Column
    story.append(_p("3. Exported Files and Columns", h1_style))
    files_table = [
        ["File", "Role", "Columns"],
        [
            "fact_pairs.csv",
            "Fact (1 row/pair)",
            "pair_id, film_id, cluster_id, risk_tier_id, en, ta, risk_score, labse_cos, "
            "en_words, ta_chars, len_ratio, ta_vowel_share, ta_latin_share, ta_spaced_punct, "
            "en_exclaim, en_question, idiom_present, is_short, is_anomaly",
        ],
        [
            "dim_film.csv",
            "Dimension (films)",
            "film_id_key, film_id, year, series_group, pairs_count, mean_risk, "
            "spaced_punct_rate, latin_share, mean_len_ratio",
        ],
        [
            "dim_cluster.csv",
            "Dimension (clusters)",
            "cluster_id, cluster_size, mean_risk, mean_labse_cos, mean_len_ratio, description",
        ],
        [
            "dim_risk_tier.csv",
            "Dimension (tiers)",
            "risk_tier_id, tier_name, tier_description",
        ],
        [
            "fact_rules.csv",
            "Fact (FP-Growth)",
            "rule_id, antecedents, consequents, support, confidence, lift",
        ],
        [
            "fact_anomalies.csv",
            "Fact (top anomalies)",
            "anomaly_rank, pair_id, film_id, en, ta, risk_score, anomaly_score",
        ],
        [
            "fact_model_metrics.csv",
            "Summary (CV metrics)",
            "model_name, auroc, pr_auc, auroc_ci_lo, auroc_ci_hi, "
            "pr_auc_ci_lo, pr_auc_ci_hi, recall_at_10, note",
        ],
    ]
    story.append(_tbl(files_table, [1.4 * inch, 1.3 * inch, 4.4 * inch]))

    # 5. Importing CSVs Step by Step
    story.append(_p("4. Importing CSVs Step by Step", h1_style))
    story.append(
        _p("Repeat these steps for each of the seven CSV files in exports/powerbi/:")
    )
    import_steps = [
        "1. Open Power BI Desktop and select Home > Get Data > Text/CSV (verify).",
        "2. Browse to exports/powerbi/ and choose the CSV file (start with fact_pairs.csv).",
        "3. In the preview dialog, confirm File Origin is set to 65001: Unicode (UTF-8).",
        "4. Confirm Delimiter is Comma and Data Type Detection uses the full dataset (verify).",
        "5. Click Transform Data (verify) if needed; confirm integer keys "
        "(pair_id, film_id, film_id_key, cluster_id, risk_tier_id, rule_id, anomaly_rank) "
        "are Whole Number and First Row as Headers is applied.",
        "6. Click Close & Apply (verify) on the Home ribbon.",
    ]
    for step in import_steps:
        story.append(_p(step))

    # 6. Creating Relationships
    story.append(_p("5. Creating Relationships", h1_style))
    story.append(
        _p(
            "Open Model view on the left navigation rail (verify) and verify or create "
            "these links (matching exports/powerbi/RELATIONSHIPS.md):"
        )
    )
    rel_table = [
        ["From Table[Column]", "To Table[Column]", "Cardinality", "Active"],
        ["fact_pairs[film_id]", "dim_film[film_id_key]", "Many-to-One (*:1)", "Yes"],
        ["fact_pairs[cluster_id]", "dim_cluster[cluster_id]", "Many-to-One (*:1)", "Yes"],
        ["fact_pairs[risk_tier_id]", "dim_risk_tier[risk_tier_id]", "Many-to-One (*:1)", "Yes"],
        ["fact_anomalies[pair_id]", "fact_pairs[pair_id]", "One-to-One / *:1", "Yes"],
        ["fact_anomalies[film_id]", "dim_film[film_id_key]", "Many-to-One (*:1)", "No (inactive)"],
    ]
    story.append(_tbl(rel_table, [2.1 * inch, 2.1 * inch, 1.5 * inch, 1.4 * inch]))

    # 7. DAX Measures
    story.append(_p("6. DAX Measures (Exact Formulas)", h1_style))
    story.append(
        _p(
            "In Report view, select fact_pairs in the Data pane, click Table tools > "
            "New measure (verify), and paste these exact five formulas one by one:"
        )
    )
    dax_text = (
        "Total Pairs = COUNTROWS(fact_pairs)\n"
        "Flagged Pairs = CALCULATE(COUNTROWS(fact_pairs), fact_pairs[risk_tier_id] = 3) "
        "-- (High=3)\n"
        "Flag Rate = DIVIDE([Flagged Pairs], [Total Pairs], 0)\n"
        "Average Risk = AVERAGE(fact_pairs[risk_score])\n"
        "Pairs per Film = DIVIDE([Total Pairs], DISTINCTCOUNT(fact_pairs[film_id]), 0)"
    )
    story.append(Paragraph(_ascii_safe(dax_text).replace("\n", "<br/>"), code_style))

    # 8. Page Layout Wireframe
    story.append(_p("7. Page Layout Wireframe", h1_style))
    wireframe = (
        "+-----------------------------------------------------------------------+\n"
        "| PROJECT TYTO: SUBTITLE RISK TRIAGE (HEURISTIC PROXY)  [Film Slicer v] |\n"
        "+-----------------+-----------------+-----------------+-----------------+\n"
        "|   Total Pairs   |  Flagged Pairs  |    Flag Rate    |  Average Risk   |\n"
        "+-----------------+-----------------+-----------------+-----------------+\n"
        "|  Risk by Film (Bar Chart)         |  Risk Tier Donut (Low/Med/High)   |\n"
        "|  X: dim_film[film_id]             |  Legend: dim_risk_tier[tier_name] |\n"
        "|  Y: Average of risk_score         |  Values: [Total Pairs]            |\n"
        "+-----------------------------------+-----------------------------------+\n"
        "|  LaBSE Cosine vs Risk Score       |  Top Association Rules Table      |\n"
        "|  (Scatter colored by tier_name)   |  antecedents, consequents, lift   |\n"
        "+-----------------------------------+-----------------------------------+\n"
        "|  Subtitle Pairs Inspection Table (en, ta [Nirmala UI], risk, tier)    |\n"
        "+-----------------------------------------------------------------------+"
    )
    wf_html = _ascii_safe(wireframe).replace("\n", "<br/>").replace(" ", "&nbsp;")
    story.append(Paragraph(wf_html, code_style))

    # 9. Visuals Specification
    story.append(_p("8. Visual Specifications", h1_style))
    visuals_table = [
        ["Visual", "Type", "Fields", "Sort & Formatting"],
        [
            "KPI Cards (x4)",
            "Card (new) / Card",
            "[Total Pairs], [Flagged Pairs], [Flag Rate], [Average Risk]",
            "Format Flag Rate as Percentage (1 decimal); Average Risk to 1 decimal.",
        ],
        [
            "Risk by Film",
            "Clustered column chart",
            "X-axis: dim_film[film_id]; Y-axis: Average of fact_pairs[risk_score]",
            "Sort descending by Average Risk; enable data labels (verify).",
        ],
        [
            "Risk-Tier Donut",
            "Donut chart",
            "Legend: dim_risk_tier[tier_name]; Values: [Total Pairs]",
            "Colors: High=#D64550 (red), Medium=#F2C80F (yellow), Low=#2E8B57 (green).",
        ],
        [
            "LaBSE vs Risk Scatter",
            "Scatter chart",
            "X: fact_pairs[labse_cos]; Y: fact_pairs[risk_score]; "
            "Legend: dim_risk_tier[tier_name]",
            "If labse_cos is null, skip or note; disable aggregation on Details: pair_id.",
        ],
        [
            "Pairs Table",
            "Table",
            "fact_pairs[en], fact_pairs[ta], fact_pairs[risk_score], "
            "dim_risk_tier[tier_name]",
            "Sort desc by risk_score. Values font: Nirmala UI. "
            "Conditional bg on tier_name: High=red, Medium=yellow, Low=green.",
        ],
        [
            "Film Slicer",
            "Slicer",
            "dim_film[film_id]",
            "Dropdown style (Slicer settings > Options > Style = Dropdown, verify).",
        ],
        [
            "Rules Table",
            "Table",
            "fact_rules[antecedents], fact_rules[consequents], "
            "fact_rules[lift], fact_rules[confidence]",
            "Sort descending by lift; format confidence and lift to 3 decimals.",
        ],
    ]
    story.append(_tbl(visuals_table, [1.25 * inch, 1.15 * inch, 2.25 * inch, 2.45 * inch]))

    # 10. Tamil Font Tips
    story.append(_p("9. Tamil Font Rendering Tips", h1_style))
    story.append(
        _p(
            "- In Pairs Table and Anomalies Table visuals, open Format your visual > Visual > "
            "Values > Font (verify) and select 'Nirmala UI' (Windows built-in complex-script "
            "font with full Tamil glyph shaping)."
        )
    )
    story.append(
        _p(
            "- Enable Word wrap on the 'en' and 'ta' columns so long subtitle lines remain "
            "readable without horizontal clipping."
        )
    )
    story.append(
        _p(
            "- Never re-save the CSVs from Excel without UTF-8 BOM; Excel legacy CSV export "
            "strips non-ASCII Tamil characters."
        )
    )

    # 11. Reconciliation Checklist (Real numbers from exported CSVs)
    story.append(_p("10. Reconciliation Checklist (Actual Export Counts)", h1_style))
    story.append(
        _p(
            "After loading all seven CSVs into Power BI and creating the DAX measures, confirm "
            "your KPI cards and table counts match these exact numbers read from disk:"
        )
    )
    recon_table = [
        ["Check Item", "Source CSV", "Expected Value (Actual)"],
        ["Total pairs", "fact_pairs.csv", f"{stats['total_pairs']:,}"],
        ["High-risk pairs (risk_tier_id = 3)", "fact_pairs.csv", f"{stats['high_risk_pairs']:,}"],
        ["Distinct films", "dim_film.csv", f"{stats['films_count']:,}"],
        ["Association rules", "fact_rules.csv", f"{stats['rules_count']:,}"],
        ["Top anomalies", "fact_anomalies.csv", f"{stats['anomalies_count']:,}"],
        ["Clusters", "dim_cluster.csv", f"{stats['clusters_count']:,}"],
    ]
    story.append(_tbl(recon_table, [2.7 * inch, 2.0 * inch, 2.4 * inch]))

    # 12. Troubleshooting
    story.append(_p("11. Troubleshooting", h1_style))
    troubleshooting_items = [
        "- Tamil text shows boxes or question marks: Re-import CSV with File Origin = "
        "65001: Unicode (UTF-8) and set table Values font to Nirmala UI.",
        "- Ambiguous relationship warning between dim_film and fact_anomalies: Mark "
        "fact_anomalies[film_id] -> dim_film[film_id_key] Inactive while keeping "
        "fact_anomalies[pair_id] -> fact_pairs[pair_id] Active.",
        "- Scatter chart collapses into one point: Drag fact_pairs[pair_id] into "
        "Values/Details (verify) and set summarization to Don't summarize.",
        "- Flag Rate shows 0% or blank: Verify risk_tier_id is Whole Number and "
        "Flagged Pairs filters risk_tier_id = 3.",
    ]
    for item in troubleshooting_items:
        story.append(_p(item))

    # 13. 10-Step Tick-Box Checklist
    story.append(_p("12. Helper 10-Step Build Checklist", h1_style))
    checklist_items = [
        "[ ] 1. Verify all 7 CSV files exist inside exports/powerbi/.",
        "[ ] 2. Open Power BI Desktop and import all 7 CSVs with File Origin 65001 (UTF-8).",
        "[ ] 3. Confirm integer keys have Whole Number type and no errors in Power Query.",
        "[ ] 4. Create/verify the star-schema relationships in Model view.",
        "[ ] 5. Add all 5 DAX measures (Total Pairs, Flagged Pairs, Flag Rate, Average Risk, "
        "Pairs per Film).",
        "[ ] 6. Place the 4 KPI cards across the top of the report canvas.",
        "[ ] 7. Add the Film slicer (dim_film[film_id]) and Risk-tier donut chart.",
        "[ ] 8. Add the Risk-by-film bar chart and LaBSE vs. Risk scatter chart.",
        "[ ] 9. Add the Pairs table (Nirmala UI font + tier formatting) and Rules table.",
        "[ ] 10. Verify counts against Section 10 Reconciliation Checklist and save as .pbix.",
    ]
    for item in checklist_items:
        story.append(_p(item))

    # 14. Glossary
    story.append(_p("13. Glossary", h1_style))
    glossary_table = [
        ["Term", "Definition"],
        [
            "Heuristic Proxy",
            "Rule-based risk score built from weak signals; not human ground truth.",
        ],
        [
            "LaBSE Cosine",
            "Cosine similarity between English and Tamil LaBSE sentence embeddings (-1 to 1).",
        ],
        [
            "FP-Growth",
            "Frequent-pattern mining algorithm producing rules (support, confidence, lift).",
        ],
        [
            "Lift",
            "How much more often antecedent and consequent co-occur than by chance.",
        ],
        [
            "Vowel-Sign Share",
            "Fraction of Tamil chars in U+0BBE-U+0BCC; used to detect script corruption.",
        ],
        [
            "BOM (UTF-8-SIG)",
            "Byte Order Mark (EF BB BF) ensuring Windows apps detect UTF-8 automatically.",
        ],
    ]
    story.append(_tbl(glossary_table, [1.8 * inch, 5.3 * inch]))

    doc.build(story)
    return pdf_path, doc.recorded_headings


def inspect_pdf_file(pdf_path: str | Path) -> dict[str, Any]:
    """Reopen a generated PDF on disk to count pages and extract headings/bookmarks."""
    pdf_bytes = Path(pdf_path).read_bytes()

    page_matches = re.findall(rb"/Type\s*/Page\b(?!s)", pdf_bytes)
    page_count = len(page_matches)

    outline_titles: list[str] = []
    for raw in re.findall(rb"/Title\s*\(((?:\\.|[^\\)])*)\)", pdf_bytes):
        decoded = (
            raw.decode("latin-1", errors="replace")
            .replace(r"\(", "(")
            .replace(r"\)", ")")
            .replace(r"\\", "\\")
            .strip()
        )
        if decoded and decoded != "Project Tyto: Power BI Dashboard Build Guide":
            outline_titles.append(decoded)

    if not outline_titles:
        for stream_match in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", pdf_bytes, re.DOTALL):
            raw_stream = stream_match.group(1)
            try:
                text = zlib.decompress(raw_stream).decode("latin-1", errors="ignore")
            except zlib.error:
                continue
            for m in re.findall(r"\((\d+\.\s+[^)]+)\)\s*Tj", text):
                outline_titles.append(m)

    return {
        "page_count": page_count,
        "headings": outline_titles,
    }


def export_powerbi(
    pairs_path: str | Path = "data/processed/pairs_v2018.parquet",
    artifacts_dir: str | Path = "artifacts",
    out_dir: str | Path = "exports",
) -> dict[str, Any]:
    """Export Power BI star-schema CSVs (UTF-8 BOM), docs, and Build Guide PDF."""
    pairs_path = Path(pairs_path)
    artifacts_dir = Path(artifacts_dir)
    powerbi_dir = _resolve_powerbi_dir(out_dir)
    powerbi_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        pairs_path,
        artifacts_dir / "risk.parquet",
        artifacts_dir / "clusters.parquet",
        artifacts_dir / "anomalies.parquet",
        artifacts_dir / "film_stats.parquet",
        artifacts_dir / "association_rules.parquet",
        artifacts_dir / "predictor_metrics.json",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts for Power BI export: " + ", ".join(missing)
        )

    logger.info("Reading pipeline artifacts from %s and %s", pairs_path, artifacts_dir)
    pairs_df = pd.read_parquet(pairs_path)
    risk_df = pd.read_parquet(artifacts_dir / "risk.parquet")
    clusters_df = pd.read_parquet(artifacts_dir / "clusters.parquet")
    anomalies_df = pd.read_parquet(artifacts_dir / "anomalies.parquet")
    film_stats_df = pd.read_parquet(artifacts_dir / "film_stats.parquet")
    rules_df = pd.read_parquet(artifacts_dir / "association_rules.parquet")
    metrics_raw = (artifacts_dir / "predictor_metrics.json").read_text(encoding="utf-8")
    metrics_data = json.loads(metrics_raw)

    # 1. dim_film.csv
    film_meta = (
        pairs_df[["film_id", "year", "series_group"]]
        .assign(film_id=lambda d: d["film_id"].astype(str))
        .drop_duplicates(subset=["film_id"])
    )
    dim_film = film_stats_df.copy()
    dim_film["film_id"] = dim_film["film_id"].astype(str)
    dim_film = dim_film.merge(film_meta, on="film_id", how="left")
    dim_film = dim_film.sort_values("film_id").reset_index(drop=True)
    dim_film["film_id_key"] = range(1, len(dim_film) + 1)
    if "pairs" in dim_film.columns and "pairs_count" not in dim_film.columns:
        dim_film = dim_film.rename(columns={"pairs": "pairs_count"})
    dim_film["year"] = dim_film["year"].fillna("").astype(str)
    dim_film["series_group"] = dim_film["series_group"].fillna("").astype(str)
    dim_film_cols = [
        "film_id_key",
        "film_id",
        "year",
        "series_group",
        "pairs_count",
        "mean_risk",
        "spaced_punct_rate",
        "latin_share",
        "mean_len_ratio",
    ]
    dim_film = dim_film[dim_film_cols]
    film_key_map = dict(zip(dim_film["film_id"], dim_film["film_id_key"], strict=True))

    # 2. dim_risk_tier.csv
    dim_risk_tier = pd.DataFrame(_RISK_TIER_ROWS)

    # 3. dim_cluster.csv
    overall_risk = float(clusters_df["risk_score"].mean()) if len(clusters_df) else 0.0
    overall_len = float(clusters_df["len_ratio"].mean()) if len(clusters_df) else 0.0
    dim_cluster = (
        clusters_df.groupby("cluster_id", as_index=False)
        .agg(
            cluster_size=("pair_id", "count"),
            mean_risk=("risk_score", "mean"),
            mean_labse_cos=("labse_cos", "mean"),
            mean_len_ratio=("len_ratio", "mean"),
        )
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
    dim_cluster["cluster_id"] = dim_cluster["cluster_id"].astype(int)
    dim_cluster["description"] = dim_cluster.apply(
        lambda row: _describe_cluster(row, overall_risk, overall_len),
        axis=1,
    )

    # 4. fact_pairs.csv
    anomaly_pair_ids = set(anomalies_df["pair_id"].astype(int).tolist())
    cluster_lookup = clusters_df[["pair_id", "cluster_id"]].drop_duplicates(subset=["pair_id"])
    fact = risk_df.merge(cluster_lookup, on="pair_id", how="left", suffixes=("", "_clust"))
    if "cluster_id_clust" in fact.columns:
        fact["cluster_id"] = fact["cluster_id_clust"].fillna(fact["cluster_id"])

    fact["pair_id"] = fact["pair_id"].astype(int)
    fact["film_id_str"] = fact["film_id"].astype(str)
    fact["film_id"] = fact["film_id_str"].map(film_key_map)
    if fact["film_id"].isna().any():
        raise ValueError("Null foreign key detected in fact_pairs[film_id]")
    fact["film_id"] = fact["film_id"].astype(int)

    if fact["cluster_id"].isna().any():
        raise ValueError("Null foreign key detected in fact_pairs[cluster_id]")
    fact["cluster_id"] = fact["cluster_id"].astype(int)

    fact["risk_tier_id"] = fact["risk_tier"].map(_RISK_TIER_MAP)
    if fact["risk_tier_id"].isna().any():
        raise ValueError("Null foreign key detected in fact_pairs[risk_tier_id]")
    fact["risk_tier_id"] = fact["risk_tier_id"].astype(int)

    fact["is_anomaly"] = fact["pair_id"].isin(anomaly_pair_ids).astype(int)

    fact_pairs_cols = [
        "pair_id",
        "film_id",
        "cluster_id",
        "risk_tier_id",
        "en",
        "ta",
        "risk_score",
        "labse_cos",
        "en_words",
        "ta_chars",
        "len_ratio",
        "ta_vowel_share",
        "ta_latin_share",
        "ta_spaced_punct",
        "en_exclaim",
        "en_question",
        "idiom_present",
        "is_short",
        "is_anomaly",
    ]
    fact_pairs = fact[fact_pairs_cols].copy()

    # 5. fact_rules.csv
    rule_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    if len(rules_df) > 0:
        fact_rules = rules_df[rule_cols].copy().reset_index(drop=True)
        fact_rules.insert(0, "rule_id", range(1, len(fact_rules) + 1))
    else:
        fact_rules = pd.DataFrame(columns=["rule_id", *rule_cols])

    # 6. fact_anomalies.csv
    fact_anom = anomalies_df.sort_values("anomaly_score", ascending=True).reset_index(drop=True)
    fact_anom["anomaly_rank"] = range(1, len(fact_anom) + 1)
    fact_anom["pair_id"] = fact_anom["pair_id"].astype(int)
    fact_anom["film_id"] = fact_anom["film_id"].astype(str).map(film_key_map)
    if fact_anom["film_id"].isna().any():
        raise ValueError("Null foreign key detected in fact_anomalies[film_id]")
    fact_anom["film_id"] = fact_anom["film_id"].astype(int)
    fact_anomalies = fact_anom[
        ["anomaly_rank", "pair_id", "film_id", "en", "ta", "risk_score", "anomaly_score"]
    ].copy()

    # 7. fact_model_metrics.csv
    fact_model_metrics = pd.DataFrame(_normalize_metrics_rows(metrics_data))

    # Write CSVs with UTF-8 BOM
    csv_targets: list[tuple[str, pd.DataFrame]] = [
        ("fact_pairs.csv", fact_pairs),
        ("dim_film.csv", dim_film),
        ("dim_cluster.csv", dim_cluster),
        ("dim_risk_tier.csv", dim_risk_tier),
        ("fact_rules.csv", fact_rules),
        ("fact_anomalies.csv", fact_anomalies),
        ("fact_model_metrics.csv", fact_model_metrics),
    ]
    for filename, table in csv_targets:
        table.to_csv(powerbi_dir / filename, index=False, encoding="utf-8-sig")

    _write_data_dictionary(powerbi_dir)
    _write_relationships_md(powerbi_dir)

    # Read back ACTUAL numbers from exported CSVs for reconciliation checklist
    reloaded_pairs = pd.read_csv(powerbi_dir / "fact_pairs.csv", encoding="utf-8-sig")
    reloaded_films = pd.read_csv(powerbi_dir / "dim_film.csv", encoding="utf-8-sig")
    reloaded_rules = pd.read_csv(powerbi_dir / "fact_rules.csv", encoding="utf-8-sig")
    reloaded_anoms = pd.read_csv(powerbi_dir / "fact_anomalies.csv", encoding="utf-8-sig")
    reloaded_clusters = pd.read_csv(powerbi_dir / "dim_cluster.csv", encoding="utf-8-sig")

    reconciliation_stats = {
        "total_pairs": int(len(reloaded_pairs)),
        "high_risk_pairs": int((reloaded_pairs["risk_tier_id"] == 3).sum()),
        "films_count": int(reloaded_films["film_id"].nunique()),
        "rules_count": int(len(reloaded_rules)),
        "anomalies_count": int(len(reloaded_anoms)),
        "clusters_count": int(len(reloaded_clusters)),
    }

    pdf_path = powerbi_dir / "PowerBI_Build_Guide.pdf"
    _, recorded_headings = _build_powerbi_pdf(pdf_path, reconciliation_stats)

    pdf_info = inspect_pdf_file(pdf_path)
    headings = pdf_info["headings"] or recorded_headings
    page_count = pdf_info["page_count"]

    print(
        f"Power BI export complete -> {powerbi_dir}\n"
        f"  fact_pairs.csv       : {reconciliation_stats['total_pairs']:,} rows "
        f"(High risk={reconciliation_stats['high_risk_pairs']:,})\n"
        f"  dim_film.csv         : {reconciliation_stats['films_count']} films\n"
        f"  dim_cluster.csv      : {reconciliation_stats['clusters_count']} clusters\n"
        f"  fact_rules.csv       : {reconciliation_stats['rules_count']} rules\n"
        f"  fact_anomalies.csv   : {reconciliation_stats['anomalies_count']} anomalies\n"
        f"  PowerBI_Build_Guide.pdf : {page_count} pages, {len(headings)} headings"
    )

    return {
        "powerbi_dir": str(powerbi_dir),
        "pdf_path": str(pdf_path),
        "pdf_pages": page_count,
        "pdf_headings": headings,
        **reconciliation_stats,
    }
