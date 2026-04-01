import streamlit as st


def apply_page_config(title: str, icon: str = "N") -> None:
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;700&family=Manrope:wght@400;500;600;700&display=swap');

            :root {
                --brand-deep: #0b3b4a;
                --brand-mid: #115e59;
                --brand-bright: #1d4ed8;
                --ink: #10222b;
                --card-bg: rgba(255, 255, 255, 0.88);
                --line: rgba(17, 94, 89, 0.18);
            }

            .stApp {
                background:
                    radial-gradient(circle at 8% 4%, rgba(59, 130, 246, 0.20), transparent 36%),
                    radial-gradient(circle at 92% 0%, rgba(20, 184, 166, 0.22), transparent 32%),
                    linear-gradient(160deg, #f3fbff 0%, #f7faf8 42%, #f3f8ff 100%);
                color: var(--ink);
            }

            html, body, [class*="css"] {
                font-family: 'Manrope', sans-serif;
            }

            h1, h2, h3, h4 {
                font-family: 'Fraunces', serif;
                color: #082734;
                letter-spacing: -0.01em;
            }

            [data-testid="stSidebarCollapsedControl"] {
                display: none;
            }

            .top-nav {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
                flex-wrap: nowrap;
                width: fit-content;
                margin: 0 auto 1rem;
                padding: 0.45rem;
                border-radius: 16px;
                border: 1px solid rgba(17, 94, 89, 0.24);
                background: rgba(255, 255, 255, 0.72);
                box-shadow: 0 10px 22px rgba(2, 132, 199, 0.1);
                backdrop-filter: blur(4px);
            }

            .top-nav-item {
                text-decoration: none;
                color: #123040;
                font-weight: 700;
                font-size: 0.9rem;
                padding: 0.52rem 0.9rem;
                border-radius: 999px;
                border: 1px solid rgba(29, 78, 216, 0.14);
                transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
            }

            .top-nav-item:hover {
                transform: translateY(-1px);
                box-shadow: 0 8px 14px rgba(29, 78, 216, 0.14);
                background: rgba(255, 255, 255, 0.95);
            }

            .top-nav-item.active {
                background: linear-gradient(130deg, #2563eb, #1d4ed8);
                color: #ffffff;
                border-color: transparent;
                box-shadow: 0 8px 16px rgba(29, 78, 216, 0.24);
            }

            .hero-card {
                border: 1px solid var(--line);
                background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(230, 246, 255, 0.86));
                border-radius: 24px;
                padding: 1.4rem 1.6rem;
                box-shadow: 0 18px 40px rgba(12, 74, 110, 0.12);
                margin-bottom: 1rem;
                overflow: hidden;
                position: relative;
            }

            .hero-card h1 {
                margin: 0;
                font-size: clamp(1.8rem, 2.4vw, 2.8rem);
            }

            .hero-card p {
                margin: 0.4rem 0 0;
                font-size: 1rem;
                color: #16404e;
                max-width: 68ch;
            }

            .eyebrow {
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-weight: 700;
                color: #1e3a8a;
                font-size: 0.75rem;
                margin-bottom: 0.35rem;
            }

            .glass-card {
                border-radius: 18px;
                border: 1px solid var(--line);
                background: var(--card-bg);
                box-shadow: 0 12px 26px rgba(12, 74, 110, 0.08);
                backdrop-filter: blur(6px);
                padding: 1rem 1.1rem;
                margin-bottom: 0.8rem;
            }

            .glass-title {
                margin: 0;
                font-size: 1.1rem;
                font-family: 'Fraunces', serif;
                color: #0b2a3a;
            }

            .glass-card p {
                margin: 0.35rem 0 0;
                color: #264653;
            }

            .prediction-card {
                border-radius: 20px;
                border: 1px solid var(--line);
                background: rgba(255, 255, 255, 0.92);
                padding: 1.05rem 1.2rem;
                margin: 0.8rem 0 0.65rem;
                box-shadow: 0 16px 28px rgba(15, 23, 42, 0.12);
            }

            .prediction-value {
                margin: 0.15rem 0;
                font-size: 2rem;
                font-family: 'Fraunces', serif;
                color: #082734;
                letter-spacing: -0.01em;
            }

            .prediction-source,
            .confidence-label {
                margin: 0;
                font-size: 0.92rem;
                color: #274657;
            }

            .tone-real {
                border-left: 6px solid #0f766e;
            }

            .tone-fake {
                border-left: 6px solid #be123c;
            }

            .tone-neutral {
                border-left: 6px solid #1d4ed8;
            }

            .stButton > button {
                border: none;
                border-radius: 14px;
                background: linear-gradient(120deg, var(--brand-mid) 0%, var(--brand-bright) 100%);
                color: #ffffff;
                font-weight: 700;
                padding: 0.58rem 1.1rem;
                transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
                box-shadow: 0 10px 20px rgba(17, 94, 89, 0.24);
            }

            .stButton > button:hover {
                transform: translateY(-2px);
                filter: brightness(1.05);
                box-shadow: 0 14px 26px rgba(29, 78, 216, 0.28);
            }

            div[data-baseweb="textarea"] textarea,
            div[data-baseweb="input"] input {
                border-radius: 14px !important;
                border: 1px solid rgba(30, 64, 175, 0.16) !important;
                background: rgba(255, 255, 255, 0.93) !important;
            }

            div[data-testid="stMetric"] {
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.5rem 0.7rem;
                background: rgba(255, 255, 255, 0.88);
            }

            .bg-orb {
                position: fixed;
                border-radius: 999px;
                filter: blur(40px);
                pointer-events: none;
                opacity: 0.35;
                z-index: -1;
                animation: drift 14s ease-in-out infinite;
            }

            .orb-a {
                width: 220px;
                height: 220px;
                background: #22d3ee;
                top: 12%;
                right: -30px;
            }

            .orb-b {
                width: 190px;
                height: 190px;
                background: #2dd4bf;
                bottom: 10%;
                left: -35px;
                animation-delay: 2s;
            }

            .reveal-up {
                animation: revealUp 0.65s ease both;
            }

            @keyframes revealUp {
                from {
                    opacity: 0;
                    transform: translateY(14px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes drift {
                0%, 100% {
                    transform: translateY(0px) scale(1);
                }
                50% {
                    transform: translateY(-14px) scale(1.05);
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_background_accents() -> None:
    st.markdown(
        """
        <div class="bg-orb orb-a"></div>
        <div class="bg-orb orb-b"></div>
        """,
        unsafe_allow_html=True,
    )