"""Project Tyto MVP — Streamlit Explorer & Predictor App (Stage 7).

Run from repo root:
    .venv\\Scripts\\python.exe -m streamlit run src/tyto/app/mvp_app.py
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data"

TAMIL_STYLE = "font-family: 'Nirmala UI', 'Noto Sans Tamil', sans-serif; font-size: 1.02rem;"

_VOWEL_LO = 0x0BBE
_VOWEL_HI = 0x0BCC


def _require_artifact(path: Path, mvp_cmd: str) -> bool:
    """Check if an artifact exists; render exact st.error instruction if missing."""
    if not path.exists():
        st.error(f"Missing artifact: {path}. Run: python -m tyto mvp {mvp_cmd}")
        return False
    return True


@st.cache_data(show_spinner=False)
def _load_parquet(path_str: str) -> pd.DataFrame:
    """Load a Parquet file with Streamlit caching."""
    return pd.read_parquet(path_str)


@st.cache_data(show_spinner=False)
def _load_json(path_str: str) -> Any:
    """Load a UTF-8 JSON file with Streamlit caching."""
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _compute_v2018_doc_vowel_stats(pairs_path_str: str) -> pd.DataFrame:
    """Compute per-document Tamil vowel-sign share from pairs_v2018.parquet."""
    df = pd.read_parquet(pairs_path_str, columns=["film_id", "en_doc", "ta"])

    def _doc_vowel_share(texts: pd.Series) -> float:
        total_ta = 0
        vowel_cnt = 0
        for s in texts.fillna("").astype(str):
            total_ta += len(s)
            vowel_cnt += sum(1 for ch in s if _VOWEL_LO <= ord(ch) <= _VOWEL_HI)
        return vowel_cnt / max(total_ta, 1)

    rows: list[dict[str, Any]] = []
    for (film_id, en_doc), grp in df.groupby(["film_id", "en_doc"], sort=True):
        rows.append(
            {
                "film_id": str(film_id),
                "en_doc": str(en_doc),
                "pairs": int(len(grp)),
                "vowel_share": round(_doc_vowel_share(grp["ta"]), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("vowel_share").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def _get_live_labse_embedder() -> object | None:
    """Try loading SentenceTransformer LaBSE once for interactive predictor tab."""
    try:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = SentenceTransformer(
                "sentence-transformers/LaBSE", device=device, local_files_only=True
            )
        except Exception:
            model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
        if device == "cuda":
            model.half()
        return model
    except Exception:
        return None


def _render_pair_card(en_text: str, ta_text: str, meta_line: str) -> None:
    """Render an English | Tamil pair side by side with Tamil font styling."""
    st.caption(meta_line)
    col_en, col_ta = st.columns(2)
    with col_en:
        st.markdown(
            f"<div style='padding:8px; border:1px solid #ddd; border-radius:6px;'>"
            f"<b>EN:</b> {html.escape(en_text)}</div>",
            unsafe_allow_html=True,
        )
    with col_ta:
        st.markdown(
            f"<div style='padding:8px; border:1px solid #ddd; border-radius:6px; {TAMIL_STYLE}'>"
            f"<b>TA:</b> {html.escape(ta_text)}</div>",
            unsafe_allow_html=True,
        )


def _render_contrib_chart(top_features: list[dict[str, Any]], title: str) -> None:
    """Render a Plotly horizontal bar chart of SHAP-style feature contributions."""
    if not top_features:
        return
    fdf = pd.DataFrame(top_features)
    fdf["direction"] = fdf["contribution"].apply(
        lambda v: "Increases Risk" if v >= 0 else "Decreases Risk"
    )
    fig = px.bar(
        fdf.iloc[::-1],
        x="contribution",
        y="feature",
        color="direction",
        orientation="h",
        title=title,
        color_discrete_map={"Increases Risk": "#d62728", "Decreases Risk": "#1f77b4"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _page_overview() -> None:
    """Page 1: Corpus overview, per-film stats, v2018 integrity, and mean risk chart."""
    st.header("Overview")

    risk_path = ARTIFACTS_DIR / "risk.parquet"
    film_stats_path = ARTIFACTS_DIR / "film_stats.parquet"
    pairs_path = DATA_DIR / "processed" / "pairs_v2018.parquet"

    if not _require_artifact(risk_path, "label"):
        return
    if not _require_artifact(film_stats_path, "mine"):
        return

    risk_df = _load_parquet(str(risk_path))
    film_df = _load_parquet(str(film_stats_path))

    tier_counts = risk_df["risk_tier"].value_counts().to_dict()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Pairs", f"{len(risk_df):,}")
    c2.metric("Films / Docs", f"{risk_df['film_id'].nunique()}")
    c3.metric("High Risk", f"{int(tier_counts.get('High', 0)):,}")
    c4.metric("Medium Risk", f"{int(tier_counts.get('Medium', 0)):,}")
    c5.metric("Low Risk", f"{int(tier_counts.get('Low', 0)):,}")

    st.subheader("Mean Heuristic Risk Score by Film")
    fig = px.bar(
        film_df.sort_values("mean_risk", ascending=False),
        x="film_id",
        y="mean_risk",
        hover_data=["pairs", "spaced_punct_rate", "latin_share", "mean_len_ratio"],
        labels={"film_id": "Film IMDb ID", "mean_risk": "Mean Heuristic Risk (0-100)"},
    )
    fig.update_layout(xaxis_type="category", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-Film Statistics")
    cols = [
        "film_id",
        "pairs",
        "mean_risk",
        "spaced_punct_rate",
        "latin_share",
        "mean_len_ratio",
    ]
    st.dataframe(film_df[cols], use_container_width=True, hide_index=True)

    st.subheader("Data Quality Findings: v2018 Tamil Integrity & v2024 Exclusion")
    if _require_artifact(pairs_path, "load"):
        doc_stats = _compute_v2018_doc_vowel_stats(str(pairs_path))
        min_vs = float(doc_stats["vowel_share"].min())
        max_vs = float(doc_stats["vowel_share"].max())
        mean_vs = float(doc_stats["vowel_share"].mean())
        st.markdown(
            f"- **OPUS v2018 Tamil vowel-sign share (U+0BBE–U+0BCC)** across "
            f"**{len(doc_stats)} documents**: min = `{min_vs:.4f}`, "
            f"mean = `{mean_vs:.4f}`, max = `{max_vs:.4f}` (all pass the `>= 0.10` gate)."
        )
        with st.expander("View per-document v2018 vowel-sign share table"):
            st.dataframe(doc_stats, use_container_width=True, hide_index=True)

    st.info(
        "OPUS v2024 Tamil data was observed to have severe vowel-sign stripping/corruption "
        "across almost all documents and is excluded from this MVP."
    )


def _page_film_explorer() -> None:
    """Page 2: Per-film metrics, risk tier donut chart, and high-risk samples."""
    st.header("Film Explorer")

    risk_path = ARTIFACTS_DIR / "risk.parquet"
    film_stats_path = ARTIFACTS_DIR / "film_stats.parquet"
    if not _require_artifact(risk_path, "label") or not _require_artifact(film_stats_path, "mine"):
        return

    risk_df = _load_parquet(str(risk_path))
    film_df = _load_parquet(str(film_stats_path))

    film_ids = film_df["film_id"].astype(str).tolist()
    selected_film = st.selectbox("Select Film ID", film_ids)

    row = film_df[film_df["film_id"].astype(str) == selected_film].iloc[0]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Pairs", f"{int(row['pairs']):,}")
    m2.metric("Mean Risk", f"{float(row['mean_risk']):.2f}")
    m3.metric("Spaced Punct Rate", f"{float(row['spaced_punct_rate']):.3%}")
    m4.metric("Latin Share", f"{float(row['latin_share']):.4f}")
    m5.metric("Mean Len Ratio", f"{float(row['mean_len_ratio']):.2f}")

    sub = risk_df[risk_df["film_id"].astype(str) == selected_film]
    tier_df = sub["risk_tier"].value_counts().rename_axis("risk_tier").reset_index(name="count")
    fig = px.pie(
        tier_df,
        names="risk_tier",
        values="count",
        hole=0.45,
        title=f"Risk Tier Distribution — Film {selected_film}",
        color="risk_tier",
        color_discrete_map={"High": "#d62728", "Medium": "#ff7f0e", "Low": "#2ca02c"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample High-Risk Pairs")
    high_sub = sub[sub["risk_tier"] == "High"].sort_values("risk_score", ascending=False).head(15)
    if high_sub.empty:
        st.info("No High-risk pairs found for this film.")
    else:
        for _, prow in high_sub.iterrows():
            meta = (
                f"pair_id={int(prow['pair_id'])} | risk_score={float(prow['risk_score']):.2f} | "
                f"len_ratio={float(prow['len_ratio']):.2f} | "
                f"labse_cos={float(prow['labse_cos']):.3f}"
                if pd.notna(prow["labse_cos"])
                else f"pair_id={int(prow['pair_id'])} | risk_score={float(prow['risk_score']):.2f}"
            )
            _render_pair_card(str(prow["en"]), str(prow["ta"]), meta)


def _page_pair_explorer() -> None:
    """Page 3: Filterable, paginated English-Tamil subtitle pair explorer."""
    st.header("Pair Explorer")

    risk_path = ARTIFACTS_DIR / "risk.parquet"
    if not _require_artifact(risk_path, "label"):
        return

    df = _load_parquet(str(risk_path))

    with st.sidebar:
        st.subheader("Pair Explorer Filters")
        films = ["All"] + sorted(df["film_id"].astype(str).unique().tolist())
        film_choice = st.selectbox("Film ID", films)
        tiers = st.multiselect("Risk Tier", ["High", "Medium", "Low"], default=["High", "Medium"])
        lr_min = float(df["len_ratio"].min())
        lr_max = float(min(df["len_ratio"].quantile(0.995), df["len_ratio"].max()))
        if lr_max <= lr_min:
            lr_max = float(df["len_ratio"].max())
        lr_range = st.slider(
            "len_ratio range",
            min_value=lr_min,
            max_value=float(df["len_ratio"].max()),
            value=(lr_min, lr_max),
        )
        short_choice = st.selectbox("is_short (<=3 EN words)", ["All", "Yes (1)", "No (0)"])

    mask = (df["len_ratio"] >= lr_range[0]) & (df["len_ratio"] <= lr_range[1])
    if film_choice != "All":
        mask &= df["film_id"].astype(str) == film_choice
    if tiers:
        mask &= df["risk_tier"].isin(tiers)
    if short_choice == "Yes (1)":
        mask &= df["is_short"] == 1
    elif short_choice == "No (0)":
        mask &= df["is_short"] == 0

    filtered = df[mask].sort_values("risk_score", ascending=False)
    total = len(filtered)
    st.write(f"Matching pairs: **{total:,}**")
    if total == 0:
        return

    page_size = 50
    n_pages = max(1, math.ceil(total / page_size))
    page_num = int(st.number_input("Page (50 pairs/page)", min_value=1, max_value=n_pages, value=1))
    start = (page_num - 1) * page_size
    end = min(start + page_size, total)
    st.caption(f"Showing rows {start + 1}–{end} of {total:,} (Page {page_num}/{n_pages})")

    flag_cols = [
        "en_words",
        "ta_chars",
        "len_ratio",
        "ta_vowel_share",
        "ta_latin_share",
        "ta_spaced_punct",
        "en_exclaim",
        "en_question",
        "en_caps_ratio",
        "digit_match",
        "punct_parity",
        "repeat_count",
        "is_short",
        "ta_hedge_count",
        "idiom_present",
        "labse_cos",
        "sig_labse_cos_inv_rank",
        "sig_len_ratio_extreme",
        "sig_ta_spaced_punct",
        "sig_ta_latin_heavy",
        "sig_digit_mismatch",
        "sig_punct_parity",
        "sig_idiom_low_sim",
    ]

    page_df = filtered.iloc[start:end]
    for _, row in page_df.iterrows():
        cos_str = f"{float(row['labse_cos']):.3f}" if pd.notna(row["labse_cos"]) else "NaN"
        meta = (
            f"pair_id={int(row['pair_id'])} | film={row['film_id']} | "
            f"tier={row['risk_tier']} | risk_score={float(row['risk_score']):.2f} | "
            f"labse_cos={cos_str}"
        )
        _render_pair_card(str(row["en"]), str(row["ta"]), meta)
        active_flags = {
            c: (round(float(row[c]), 4) if isinstance(row[c], float) else row[c])
            for c in flag_cols
            if c in row.index
        }
        with st.expander("All features & flags"):
            st.json(active_flags)


def _page_patterns() -> None:
    """Page 4: FP-Growth association rules, KMeans clusters, and IsolationForest anomalies."""
    st.header("Patterns")

    rules_path = ARTIFACTS_DIR / "association_rules.parquet"
    clusters_path = ARTIFACTS_DIR / "clusters.parquet"
    anomalies_path = ARTIFACTS_DIR / "anomalies.parquet"

    if _require_artifact(rules_path, "mine"):
        st.subheader("Association Rules (FP-Growth)")
        st.info("Association is not causation.")
        rules_df = _load_parquet(str(rules_path))
        display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        st.dataframe(rules_df[display_cols], use_container_width=True, hide_index=True)

    if _require_artifact(clusters_path, "mine"):
        st.subheader("Cluster Summary (KMeans)")
        cl_df = _load_parquet(str(clusters_path))
        summary = (
            cl_df.groupby("cluster_id", as_index=False)
            .agg(
                size=("pair_id", "count"),
                mean_risk=("risk_score", "mean"),
                mean_en_words=("en_words", "mean"),
                mean_ta_chars=("ta_chars", "mean"),
                mean_len_ratio=("len_ratio", "mean"),
                mean_labse_cos=("labse_cos", "mean"),
                mean_spaced_punct=("ta_spaced_punct", "mean"),
                mean_latin_share=("ta_latin_share", "mean"),
                idiom_rate=("idiom_present", "mean"),
            )
            .sort_values("cluster_id")
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("**2 Exemplar Pairs per Cluster (closest to centroid)**")
        for cid in summary["cluster_id"].tolist():
            sub_cl = cl_df[cl_df["cluster_id"] == cid]
            exemplars = sub_cl[sub_cl["is_exemplar"]].sort_values("dist_to_centroid").head(2)
            if exemplars.empty:
                exemplars = sub_cl.sort_values("dist_to_centroid").head(2)
            with st.expander(f"Cluster {cid} exemplars (size={len(sub_cl):,})"):
                for _, erow in exemplars.iterrows():
                    meta = (
                        f"pair_id={int(erow['pair_id'])} | film={erow['film_id']} | "
                        f"risk={float(erow['risk_score']):.2f} ({erow['risk_tier']})"
                    )
                    _render_pair_card(str(erow["en"]), str(erow["ta"]), meta)

    if _require_artifact(anomalies_path, "mine"):
        st.subheader("Top 30 Anomalous Pairs (IsolationForest)")
        anom_df = _load_parquet(str(anomalies_path))
        cols = [
            "pair_id",
            "film_id",
            "anomaly_score",
            "risk_score",
            "risk_tier",
            "en_words",
            "ta_chars",
            "len_ratio",
            "en",
            "ta",
        ]
        keep = [c for c in cols if c in anom_df.columns]
        st.dataframe(anom_df[keep].head(30), use_container_width=True, hide_index=True)


def _page_predictor() -> None:
    """Page 5: Precomputed illustrative demo predictions + live LightGBM predictor."""
    st.header("Source-Only Predictor (English → Heuristic Risk Proxy)")

    metrics_path = ARTIFACTS_DIR / "predictor_metrics.json"
    fi_path = ARTIFACTS_DIR / "feature_importance.json"
    demo_preds_path = ARTIFACTS_DIR / "demo_predictions.json"
    model_path = ARTIFACTS_DIR / "lgbm_model.pkl"

    if _require_artifact(metrics_path, "predict"):
        metrics = _load_json(str(metrics_path))
        st.subheader("GroupKFold Cross-Validation Metrics (Source-Only Features)")
        rows = []
        for name in ("baseline", "logistic", "lgbm"):
            if name in metrics:
                m = metrics[name]
                rows.append(
                    {
                        "model": name,
                        "auroc": m["auroc"],
                        "auroc_95ci": f"[{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}]",
                        "pr_auc": m["pr_auc"],
                        "pr_auc_95ci": f"[{m['pr_auc_ci'][0]:.4f}, {m['pr_auc_ci'][1]:.4f}]",
                        "recall_at_10": m["recall_at_10"],
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if _require_artifact(fi_path, "predict"):
        fi_data = _load_json(str(fi_path))
        with st.expander("Global Feature Importance (LightGBM pred_contrib / SHAP-style)"):
            fi_df = pd.DataFrame(fi_data.get("features", [])).head(15)
            st.dataframe(fi_df, use_container_width=True, hide_index=True)

    st.subheader("12 Predefined Examples (ILLUSTRATIVE inputs — not from real subtitle data)")
    if _require_artifact(demo_preds_path, "predict"):
        demo_preds = _load_json(str(demo_preds_path))
        labels = [f"[{d['id']}] ({d['type']}) {d['text']}" for d in demo_preds]
        choice = st.selectbox("Choose an ILLUSTRATIVE input sentence", labels)
        selected_idx = labels.index(choice)
        rec = demo_preds[selected_idx]

        st.caption(rec.get("label", "ILLUSTRATIVE inputs — not from real subtitle data"))
        c1, c2 = st.columns(2)
        c1.metric("Predicted Risk Tier", rec["risk_tier"])
        c2.metric("Predicted Probability (Top 20% Proxy)", f"{float(rec['risk_prob']):.4f}")
        _render_contrib_chart(rec.get("top_features", []), "Top Contributing Features (Stored)")

    st.subheader("Live Predictor (Real Model Output)")
    if _require_artifact(model_path, "predict"):
        user_text = st.text_area(
            "Enter an English subtitle line:",
            value="Don't beat around the bush — tell me what happened at 10 PM!",
            height=90,
        )
        if st.button("Predict Risk Proxy", type="primary"):
            if not user_text.strip():
                st.warning("Enter non-empty English text.")
            else:
                from tyto.mvp.predict import predict_texts

                embedder = _get_live_labse_embedder()
                pred = predict_texts(
                    [user_text.strip()],
                    model_path=model_path,
                    use_labse=embedder is not None,
                    embedder=embedder,
                )[0]
                st.success("Real model output (predicting heuristic proxy, not human judgement)")
                c1, c2 = st.columns(2)
                c1.metric("Predicted Risk Tier", pred["risk_tier"])
                c2.metric("Predicted Probability", f"{float(pred['risk_prob']):.4f}")
                _render_contrib_chart(
                    pred.get("top_features", []), "Top Contributing Features (Live SHAP-style)"
                )


def _page_about_limits() -> None:
    """Page 6: Honest accounting of real vs illustrative vs missing components and limits."""
    st.header("About & Limits")

    risk_path = ARTIFACTS_DIR / "risk.parquet"
    film_stats_path = ARTIFACTS_DIR / "film_stats.parquet"
    if not _require_artifact(risk_path, "label") or not _require_artifact(film_stats_path, "mine"):
        return

    risk_df = _load_parquet(str(risk_path))
    film_df = _load_parquet(str(film_stats_path))

    total_pairs = len(risk_df)
    n_films = int(risk_df["film_id"].nunique())
    overall_spaced = float((risk_df["ta_spaced_punct"].astype(float) > 0).mean())
    films_with_spaced = int((film_df["spaced_punct_rate"] > 0.05).sum())
    max_spaced_row = film_df.sort_values("spaced_punct_rate", ascending=False).iloc[0]

    st.subheader("Real vs Illustrative vs Missing")
    st.markdown(
        "\n".join(
            [
                "- **What is REAL:**",
                f"  - OPUS OpenSubtitles **v2018** English–Tamil corpus: **{total_pairs:,}** "
                f"pairs across **{n_films}** documents.",
                "  - Surface features, LaBSE cosine similarity, KMeans clustering, FP-Growth "
                "association rules, IsolationForest anomalies, and source-only LightGBM / "
                "LogisticRegression CV metrics.",
                "- **What is ILLUSTRATIVE (needs native-speaker review):**",
                "  - The 40-phrase English idiom list (`EN_IDIOMS`), 5-word Tamil hedge list "
                "(`TA_HEDGE_WORDS`), 15-word emotion cue list (`EMOTION_CUES`), and 12 "
                "predefined demo sentences.",
                "  - The heuristic risk score weights (arbitrary 0–100 combination of "
                "weak signals).",
                "- **What is MISSING:**",
                "  - **No human quality labels** (`data/gold/annotations.parquet` is not "
                "yet populated for this MVP).",
                "  - **28 documents only** — not a representative sample of Tamil cinema "
                "or OpenSubtitles.",
            ]
        )
    )

    st.subheader("Spaced-Punctuation Observation (Real Artifact Numbers)")
    st.markdown(
        f"- Across all **{total_pairs:,}** pairs, **{overall_spaced:.2%}** contain spaced "
        "punctuation (`' , '`, `' . '`, `' ! '`, or `' ? '`) in Tamil text.\n"
        f"- **{films_with_spaced} / {n_films}** films have a spaced-punctuation rate above `5%`; "
        f"highest observed film is `{max_spaced_row['film_id']}` at "
        f"`{float(max_spaced_row['spaced_punct_rate']):.2%}` "
        f"({int(max_spaced_row['pairs']):,} pairs).\n"
        "- This is an observable stylistic/formatting difference across films, not conclusive "
        "proof of origin."
    )

    st.subheader("OPUS v2024 Corruption Note")
    st.markdown(
        "- OPUS OpenSubtitles v2024 Tamil subtitles were inspected and found to suffer from "
        "severe Tamil vowel-sign stripping (`U+0BBE–U+0BCC` deficiency across almost all "
        "documents), rendering them unsuitable. Only v2018 (where every document passes "
        "`vowel_share >= 0.10`) is used."
    )


def main() -> None:
    """Entrypoint for the Project Tyto MVP Streamlit application."""
    st.set_page_config(page_title="Project Tyto — MVP", layout="wide")
    st.title("Project Tyto — English–Tamil Subtitle Quality Mining MVP")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Film Explorer",
            "Pair Explorer",
            "Patterns",
            "Predictor",
            "About & Limits",
        ],
    )

    if page == "Overview":
        _page_overview()
    elif page == "Film Explorer":
        _page_film_explorer()
    elif page == "Pair Explorer":
        _page_pair_explorer()
    elif page == "Patterns":
        _page_patterns()
    elif page == "Predictor":
        _page_predictor()
    else:
        _page_about_limits()


if __name__ == "__main__":
    main()
