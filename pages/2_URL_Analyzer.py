import streamlit as st

from src.models.predict import get_prediction_backend, predict_article
from src.utils.web_scraper import extract_article_from_url
from src.ui.components import display_prediction, render_glass_card, render_hero, render_top_nav
from src.ui.theme import apply_page_config, inject_global_css, render_background_accents

DEMO_TITLE = "City council publishes independent audit on housing budget"
DEMO_TEXT = (
    "The city council released a public report and cited audited records from the last three fiscal years. "
    "The report includes interviews with agency officials and a breakdown of line-item spending."
)

apply_page_config("URL Analyzer", icon="U")
inject_global_css()
render_background_accents()
render_top_nav("URL Analyzer")

render_hero(
    "URL Analyzer",
    "Provide a public article URL and the app will extract, preview, and classify the content.",
    eyebrow="Feature 2",
)
st.caption(f"Active backend: {get_prediction_backend()}")
st.caption("Prediction input combines the extracted title and main article text body.")

url = st.text_input("Article URL", placeholder="https://example.com/article")


def _show_article_preview(title: str, text: str) -> None:
    st.markdown("### Extracted Content")
    st.markdown(f"**Title:** {title or 'Untitled article'}")
    with st.expander("Preview text", expanded=True):
        preview = text[:1800]
        st.write(preview + ("..." if len(text) > 1800 else ""))


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