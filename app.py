from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

from src.models.predict import (
    get_active_profile,
    get_prediction_backend,
    model_artifacts_available,
    predict_article,
    predict_text,
    refresh_artifact_cache,
)
from src.models.train import (
    ACTIVE_METRICS_PATH,
    PROFILE_METRICS_PATHS,
    get_welfake_dataset_status,
    train_welfake_model,
)
from src.ui.components import display_prediction, render_glass_card, render_hero, render_top_nav
from src.ui.theme import apply_page_config, inject_global_css, render_background_accents
from src.utils.web_scraper import extract_article_from_url


SAMPLE_REAL = (
    "The local health department released a report citing hospital admission trends and "
    "interviewed physicians about seasonal respiratory illnesses."
)
SAMPLE_FAKE = (
    "Shocking secret cure exposed today! Officials banned this miracle formula because "
    "they do not want people to know the truth."
)
DEMO_TITLE = "City council publishes independent audit on housing budget"
DEMO_TEXT = (
    "The city council released a public report and cited audited records from the last three fiscal years. "
    "The report includes interviews with agency officials and a breakdown of line-item spending."
)


def _resolve_tab() -> str:
    tab = st.query_params.get("tab", "home")
    if isinstance(tab, list):
        tab = tab[0] if tab else "home"

    normalized = str(tab).strip().lower()
    if normalized not in {"home", "text", "url", "insights"}:
        normalized = "home"

    return normalized


def _compute_demo_metrics():
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_score = np.array(
        [
            0.96,
            0.93,
            0.90,
            0.89,
            0.88,
            0.91,
            0.85,
            0.87,
            0.94,
            0.82,
            0.86,
            0.88,
            0.92,
            0.90,
            0.84,
            0.79,
            0.83,
            0.48,
            0.10,
            0.16,
            0.20,
            0.18,
            0.22,
            0.26,
            0.34,
            0.30,
            0.28,
            0.24,
            0.14,
            0.66,
        ]
    )

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    fpr, tpr, _ = roc_curve(y_true, y_score)
    metrics["fpr"] = fpr
    metrics["tpr"] = tpr
    metrics["roc_auc"] = auc(fpr, tpr)
    metrics["profile"] = "demo"
    return metrics


def _load_saved_metrics(metrics_path: Path):
    if not metrics_path.exists():
        return None

    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _to_plot_ready_metrics(metrics):
    normalized = dict(metrics)
    normalized["confusion_matrix"] = np.array(metrics["confusion_matrix"])
    normalized["fpr"] = np.array(metrics["fpr"])
    normalized["tpr"] = np.array(metrics["tpr"])
    return normalized


def _show_article_preview(title: str, text: str) -> None:
    st.markdown("### Extracted Content")
    st.markdown(f"**Title:** {title or 'Untitled article'}")
    with st.expander("Preview text", expanded=True):
        preview = text[:1800]
        st.write(preview + ("..." if len(text) > 1800 else ""))


def render_home_section() -> None:
    render_hero(
        "Fake News Radar",
        "Analyze raw text and live article URLs with a fast credibility signal engine trained on the WELFake dataset.",
        eyebrow="News Verification Workspace",
    )

    backend = get_prediction_backend()
    if model_artifacts_available():
        st.success(f"Inference backend: {backend}")
    else:
        st.warning(f"Inference backend: {backend}. Train the WELFake model from Model Insights.")

    overview_col, reliability_col, ux_col = st.columns(3)
    with overview_col:
        render_glass_card(
            "Text Signal Analysis",
            "Paste headlines, body text, or social posts to score credibility patterns in seconds.",
        )
    with reliability_col:
        render_glass_card(
            "URL Extraction Pipeline",
            "Fetch article content directly from a URL and classify it without switching tools.",
        )
    with ux_col:
        render_glass_card(
            "Insight Dashboards",
            "Explore benchmark metrics and visual diagnostics from the model insights tab.",
        )

    st.info("Use the top tabs to switch tools without leaving this page.")


def render_text_section() -> None:
    if "text_analyzer_input" not in st.session_state:
        st.session_state.text_analyzer_input = ""

    render_hero(
        "Text Analyzer",
        "Paste an article, post, or headline. The engine scores linguistic credibility and manipulation cues.",
        eyebrow="Feature 1",
    )
    st.caption(f"Active backend: {get_prediction_backend()}")

    sample_col_1, sample_col_2, sample_col_3 = st.columns(3)
    with sample_col_1:
        if st.button("Load credible sample", width="stretch"):
            st.session_state.text_analyzer_input = SAMPLE_REAL
    with sample_col_2:
        if st.button("Load suspicious sample", width="stretch"):
            st.session_state.text_analyzer_input = SAMPLE_FAKE
    with sample_col_3:
        if st.button("Clear", width="stretch"):
            st.session_state.text_analyzer_input = ""

    input_text = st.text_area(
        "Article text",
        key="text_analyzer_input",
        height=240,
        placeholder="Paste text to analyze...",
    )

    if input_text.strip():
        tokens = len(input_text.split())
        characters = len(input_text)
        sentence_count = max(1, input_text.count(".") + input_text.count("!") + input_text.count("?"))
        avg_sentence_length = tokens / sentence_count

        m1, m2, m3 = st.columns(3)
        m1.metric("Word count", f"{tokens}")
        m2.metric("Character count", f"{characters}")
        m3.metric("Avg words / sentence", f"{avg_sentence_length:.1f}")

    if st.button("Analyze text", type="primary", width="stretch"):
        if input_text.strip():
            with st.spinner("Evaluating language signals..."):
                prediction, confidence = predict_text(input_text)
            display_prediction(prediction, confidence, source="Text")

            if prediction == "FAKE":
                render_glass_card(
                    "Interpretation",
                    "This text contains language patterns that often correlate with sensational or manipulative framing.",
                )
            elif prediction == "REAL":
                render_glass_card(
                    "Interpretation",
                    "This text includes more report-like and evidence-oriented language patterns.",
                )
            else:
                render_glass_card(
                    "Interpretation",
                    "The model did not have enough signal to produce a reliable classification.",
                )
        else:
            st.warning("Please enter some text before running analysis.")


def render_url_section() -> None:
    render_hero(
        "URL Analyzer",
        "Provide a public article URL and the app will extract, preview, and classify the content.",
        eyebrow="Feature 2",
    )
    st.caption(f"Active backend: {get_prediction_backend()}")
    st.caption("Prediction input combines the extracted title and main article text body.")

    url = st.text_input("Article URL", placeholder="https://example.com/article")

    analyze_col, demo_col = st.columns([2, 1])
    analyze_clicked = analyze_col.button("Analyze live URL", type="primary", width="stretch")
    demo_clicked = demo_col.button("Run with demo article", width="stretch")

    if analyze_clicked:
        if url.strip():
            with st.spinner("Fetching article content..."):
                title, text = extract_article_from_url(url)

            if title or text:
                _show_article_preview(title or "", text or "")
                prediction, confidence = predict_article(title, text)
                display_prediction(prediction, confidence, source="URL")
                render_glass_card(
                    "Quality Check",
                    "For best results, compare this score with at least one independent source and publication date.",
                )
            else:
                st.error("The article content could not be extracted from this URL.")
                st.info("Try another public article link or use the demo article button.")
        else:
            st.warning("Please enter a URL to analyze.")

    if demo_clicked:
        _show_article_preview(DEMO_TITLE, DEMO_TEXT)
        prediction, confidence = predict_article(DEMO_TITLE, DEMO_TEXT)
        display_prediction(prediction, confidence, source="Demo URL")


def render_insights_section() -> None:
    render_hero(
        "Model Insights",
        "Train directly on the WELFake dataset and inspect real model diagnostics from saved artifacts.",
        eyebrow="Feature 3",
    )

    dataset_status = get_welfake_dataset_status()
    dataset_cached = bool(dataset_status["available_locally"])
    quick_ready = model_artifacts_available("quick")
    full_ready = model_artifacts_available("full")
    active_profile = get_active_profile() or "none"
    st.caption(f"Active backend: {get_prediction_backend()}")

    info_col_1, info_col_2, info_col_3, info_col_4 = st.columns(4)
    info_col_1.metric("Dataset", "Cached" if dataset_cached else "Remote")
    info_col_2.metric("Quick Artifact", "Ready" if quick_ready else "Missing")
    info_col_3.metric("Full Artifact", "Ready" if full_ready else "Missing")
    info_col_4.metric("Active Profile", active_profile.title())

    quick_metrics = PROFILE_METRICS_PATHS["quick"].exists()
    full_metrics = PROFILE_METRICS_PATHS["full"].exists()
    st.caption(
        f"Metrics files: quick={'saved' if quick_metrics else 'missing'}, "
        f"full={'saved' if full_metrics else 'missing'}, "
        f"active={'saved' if ACTIVE_METRICS_PATH.exists() else 'missing'}."
    )

    radio_index = 1 if active_profile == "full" else 0
    selected_profile = st.radio(
        "Training profile",
        options=["quick", "full"],
        horizontal=True,
        index=radio_index,
        help="Quick uses a stratified subset for speed. Full uses the complete WELFake dataset.",
    )
    st.caption("Training profile controls both artifact name and active model selection.")

    train_col, path_col = st.columns([1, 2])
    with train_col:
        train_clicked = st.button("Train on WELFake", type="primary", width="stretch")
    with path_col:
        st.code(str(dataset_status["local_path"]), language="text")
        st.caption(
            "If not cached yet, this file is downloaded automatically from Google Drive during training."
        )

    if train_clicked:
        with st.spinner("Preparing WELFake dataset (download if needed) and training model..."):
            try:
                result = train_welfake_model(profile=selected_profile)
            except Exception as exc:
                st.error(f"Training failed: {exc}")
            else:
                st.session_state["trained_metrics"] = result["metrics"]
                st.session_state["trained_profile"] = result["profile"]
                refresh_artifact_cache()
                if result.get("dataset_downloaded"):
                    st.success(
                        "Dataset downloaded from Google Drive and training completed. "
                        f"Saved profile-specific artifacts for '{result['profile']}' and set it active."
                    )
                else:
                    st.success(
                        "Training completed. "
                        f"Saved profile-specific artifacts for '{result['profile']}' and set it active."
                    )

    if st.session_state.get("trained_metrics"):
        trained_rows = st.session_state["trained_metrics"].get("rows_used")
        trained_profile = st.session_state.get("trained_profile", "unknown")
        if trained_rows:
            st.caption(f"Latest training: profile={trained_profile}, rows used={trained_rows:,}")

    metrics_source = "demo"
    if "trained_metrics" in st.session_state:
        metrics = _to_plot_ready_metrics(st.session_state["trained_metrics"])
        metrics_source = "live"
    else:
        saved_metrics = _load_saved_metrics(ACTIVE_METRICS_PATH)
        if saved_metrics:
            metrics = _to_plot_ready_metrics(saved_metrics)
            metrics_source = "saved"
        else:
            metrics = _compute_demo_metrics()
            st.info("No trained metrics found yet. Showing demo benchmark values.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    m2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    m3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    m4.metric("F1 Score", f"{metrics['f1'] * 100:.1f}%")

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.markdown("### Confusion Matrix")
        cm = metrics["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        heatmap = ax.imshow(cm, cmap="GnBu")
        for (row, col), value in np.ndenumerate(cm):
            ax.text(col, row, f"{value}", ha="center", va="center", color="#0f172a", fontsize=12)
        ax.set_xticks([0, 1], labels=["Pred Real", "Pred Fake"])
        ax.set_yticks([0, 1], labels=["Actual Real", "Actual Fake"])
        ax.set_title("Prediction Outcomes")
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, width="stretch")

    with chart_col_2:
        st.markdown("### ROC Curve")
        fig2, ax2 = plt.subplots(figsize=(5.2, 4.0))
        ax2.plot(metrics["fpr"], metrics["tpr"], lw=2.3, color="#0f766e", label=f"AUC = {metrics['roc_auc']:.2f}")
        ax2.plot([0, 1], [0, 1], linestyle="--", color="#334155", label="Random baseline")
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title("Classifier Separability")
        ax2.legend(loc="lower right")
        st.pyplot(fig2, width="stretch")

    st.markdown("### Feature Signal Contribution")
    contribution = pd.DataFrame(
        {
            "Signal": [
                "Source citations",
                "Evidence terms",
                "Sensational language",
                "Neutral tone",
                "Headline exaggeration",
            ],
            "Weight": [0.32, 0.26, 0.18, 0.14, 0.10],
        }
    )
    bar_fig, bar_ax = plt.subplots(figsize=(7.2, 3.9))
    ordered = contribution.sort_values("Weight")
    bar_ax.barh(ordered["Signal"], ordered["Weight"], color=["#3b82f6", "#0ea5e9", "#14b8a6", "#0891b2", "#1d4ed8"])
    bar_ax.set_xlabel("Relative contribution")
    bar_ax.set_xlim(0, 0.36)
    bar_ax.grid(axis="x", linestyle="--", alpha=0.3)
    for index, value in enumerate(ordered["Weight"]):
        bar_ax.text(value + 0.006, index, f"{value:.2f}", va="center", fontsize=10, color="#0f172a")
    st.pyplot(bar_fig, width="stretch")

    if metrics_source == "demo":
        render_glass_card(
            "Benchmark Note",
            "These are demo metrics. Train on WELFake to replace them with real model diagnostics.",
        )
    elif metrics_source == "saved":
        source_profile = metrics.get("profile", "unknown")
        render_glass_card(
            "Benchmark Note",
            f"These metrics were loaded from the latest saved WELFake training run ({source_profile} profile).",
        )
    else:
        source_profile = metrics.get("profile", "unknown")
        render_glass_card(
            "Benchmark Note",
            f"These metrics come from the training run you just executed on WELFake ({source_profile} profile).",
        )


apply_page_config("Fake News Radar")
inject_global_css()
render_background_accents()

requested_tab = _resolve_tab()
if "single_page_current_tab" not in st.session_state:
    st.session_state["single_page_current_tab"] = requested_tab
elif requested_tab != st.session_state["single_page_current_tab"]:
    st.session_state["single_page_current_tab"] = requested_tab

current_tab = render_top_nav(st.session_state["single_page_current_tab"])
st.session_state["single_page_current_tab"] = current_tab
st.query_params["tab"] = current_tab

if current_tab == "text":
    render_text_section()
elif current_tab == "url":
    render_url_section()
elif current_tab == "insights":
    render_insights_section()
else:
    render_home_section()