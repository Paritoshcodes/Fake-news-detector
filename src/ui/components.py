import streamlit as st


def render_top_nav(active_page: str) -> str:
    items = [
        ("Home", "home"),
        ("Text Analyzer", "text"),
        ("URL Analyzer", "url"),
        ("Model Insights", "insights"),
    ]

    normalized_active = str(active_page).strip().lower()
    active_key_aliases = {
        "home": "home",
        "text analyzer": "text",
        "text": "text",
        "url analyzer": "url",
        "url": "url",
        "model insights": "insights",
        "insights": "insights",
    }
    active_key = active_key_aliases.get(normalized_active, "home")

    nav_links = []
    for label, key in items:
        active_class = " active" if key == active_key else ""
        nav_links.append(f'<a class="top-nav-item{active_class}" href="/?tab={key}">{label}</a>')

    st.markdown(
        f"""
        <nav class="top-nav" aria-label="Primary Navigation">
            {"".join(nav_links)}
        </nav>
        """,
        unsafe_allow_html=True,
    )

    return active_key


def render_hero(title: str, subtitle: str, eyebrow: str = "Fake News Radar") -> None:
    st.markdown(
        f"""
        <section class="hero-card reveal-up">
            <p class="eyebrow">{eyebrow}</p>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_glass_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <section class="glass-card reveal-up">
            <p class="glass-title">{title}</p>
            <p>{body}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def display_prediction(prediction: str, confidence: float, source: str = "Text") -> None:
    normalized_prediction = str(prediction or "UNKNOWN").upper()
    safe_confidence = float(max(0.0, min(100.0, confidence or 0.0)))

    if normalized_prediction == "REAL":
        tone_class = "tone-real"
        signal = "Credibility signals are stronger than manipulation cues."
    elif normalized_prediction == "FAKE":
        tone_class = "tone-fake"
        signal = "Manipulation signals outweigh trustworthy language patterns."
    else:
        tone_class = "tone-neutral"
        signal = "There was not enough signal to classify this input confidently."

    st.markdown(
        f"""
        <section class="prediction-card {tone_class} reveal-up">
            <p class="prediction-source">Source: {source}</p>
            <p class="prediction-value">{normalized_prediction}</p>
            <p>{signal}</p>
            <p class="confidence-label">Confidence score: {safe_confidence:.2f}%</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.progress(safe_confidence / 100.0)


def display_prediction_results(prediction: str, confidence: float, source: str = "Text") -> None:
    display_prediction(prediction, confidence, source=source)


def display_error(message: str) -> None:
    st.error(message)


def display_info(message: str) -> None:
    st.info(message)