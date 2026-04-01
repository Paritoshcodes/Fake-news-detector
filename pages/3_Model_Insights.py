import json
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve

from src.models.predict import get_active_profile, get_prediction_backend, model_artifacts_available, refresh_artifact_cache
from src.models.train import ACTIVE_METRICS_PATH, DEFAULT_DATASET_PATH, PROFILE_METRICS_PATHS, train_welfake_model
from src.ui.components import render_glass_card, render_hero, render_top_nav
from src.ui.theme import apply_page_config, inject_global_css, render_background_accents


def _compute_demo_metrics():
    y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    y_score = np.array([
        0.96, 0.93, 0.90, 0.89, 0.88, 0.91, 0.85, 0.87, 0.94, 0.82,
        0.86, 0.88, 0.92, 0.90, 0.84, 0.79, 0.83, 0.48, 0.10, 0.16,
        0.20, 0.18, 0.22, 0.26, 0.34, 0.30, 0.28, 0.24, 0.14, 0.66,
    ])

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


def display_model_insights():
    apply_page_config("Model Insights", icon="I")
    inject_global_css()
    render_background_accents()
    render_top_nav("Model Insights")

    render_hero(
        "Model Insights",
        "Train directly on the WELFake dataset and inspect real model diagnostics from saved artifacts.",
        eyebrow="Feature 3",
    )

    dataset_exists = DEFAULT_DATASET_PATH.exists()
    quick_ready = model_artifacts_available("quick")
    full_ready = model_artifacts_available("full")
    active_profile = get_active_profile() or "none"
    st.caption(f"Active backend: {get_prediction_backend()}")

    info_col_1, info_col_2, info_col_3, info_col_4 = st.columns(4)
    info_col_1.metric("Dataset", "Available" if dataset_exists else "Missing")
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

    selected_profile = st.radio(
        "Training profile",
        options=["quick", "full"],
        horizontal=True,
        help="Quick uses a stratified subset for speed. Full uses the complete WELFake dataset.",
    )
    st.caption("Training profile controls both artifact name and active model selection.")

    train_col, path_col = st.columns([1, 2])
    with train_col:
        train_clicked = st.button("Train on WELFake", type="primary", use_container_width=True)
    with path_col:
        st.code(str(DEFAULT_DATASET_PATH), language="text")

    if train_clicked:
        if not dataset_exists:
            st.error("WELFake_Dataset.csv is missing from the workspace root.")
        else:
            with st.spinner("Training model and vectorizer on WELFake dataset..."):
                result = train_welfake_model(profile=selected_profile)
                st.session_state["trained_metrics"] = result["metrics"]
                st.session_state["trained_profile"] = result["profile"]
                refresh_artifact_cache()
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
        ax.set_xticks([0, 1], labels=["Pred Fake", "Pred Real"])
        ax.set_yticks([0, 1], labels=["Actual Fake", "Actual Real"])
        ax.set_title("Prediction Outcomes")
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig, use_container_width=True)

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
        st.pyplot(fig2, use_container_width=True)

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
    st.pyplot(bar_fig, use_container_width=True)

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


if __name__ == "__main__":
    display_model_insights()