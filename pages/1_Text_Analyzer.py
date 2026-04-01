import streamlit as st

from src.models.predict import get_prediction_backend, predict_text
from src.ui.components import display_prediction, render_glass_card, render_hero, render_top_nav
from src.ui.theme import apply_page_config, inject_global_css, render_background_accents

SAMPLE_REAL = (
    "The local health department released a report citing hospital admission trends and "
    "interviewed physicians about seasonal respiratory illnesses."
)
SAMPLE_FAKE = (
    "Shocking secret cure exposed today! Officials banned this miracle formula because "
    "they do not want people to know the truth."
)

apply_page_config("Text Analyzer", icon="T")
inject_global_css()
render_background_accents()
render_top_nav("Text Analyzer")

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
    if st.button("Load credible sample", use_container_width=True):
        st.session_state.text_analyzer_input = SAMPLE_REAL
with sample_col_2:
    if st.button("Load suspicious sample", use_container_width=True):
        st.session_state.text_analyzer_input = SAMPLE_FAKE
with sample_col_3:
    if st.button("Clear", use_container_width=True):
        st.session_state.text_analyzer_input = ""

input_text = st.text_area(
    "Article text",
    key="text_analyzer_input",
    height=240,
    placeholder="Paste text to analyze...")

if input_text.strip():
    tokens = len(input_text.split())
    characters = len(input_text)
    sentence_count = max(1, input_text.count(".") + input_text.count("!") + input_text.count("?"))
    avg_sentence_length = tokens / sentence_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Word count", f"{tokens}")
    m2.metric("Character count", f"{characters}")
    m3.metric("Avg words / sentence", f"{avg_sentence_length:.1f}")

if st.button("Analyze text", type="primary", use_container_width=True):
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