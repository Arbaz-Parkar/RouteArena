"""Shared visual styling and application framing for RouteArena."""

from __future__ import annotations

import plotly.io as pio
import streamlit as st

_STYLES = """
<style>
:root {
    --ra-navy: #102338;
    --ra-ink: #172B4D;
    --ra-muted: #607186;
    --ra-teal: #0F766E;
    --ra-border: #DCE5EE;
    --ra-surface: #FFFFFF;
}

.stApp {
    color: var(--ra-ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: #F3F6FA;
}

[data-testid="stHeader"] {
    background: rgba(243, 246, 250, 0.94);
}

[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    padding-top: 0.8rem;
    padding-bottom: 3rem;
}

.stApp h1 {
    color: var(--ra-navy);
    font-size: clamp(2rem, 3vw, 2.65rem);
    font-weight: 750;
    letter-spacing: -0.04em;
    margin-bottom: 0.35rem;
}

.stApp h2, .stApp h3 {
    color: var(--ra-navy);
    letter-spacing: -0.02em;
}

.stApp [data-testid="stWidgetLabel"] p {
    color: #30465E;
    font-weight: 600;
}

.stApp [data-testid="stCaptionContainer"] {
    color: var(--ra-muted);
}

[data-testid="stSidebar"] {
    min-width: 17.5rem;
    max-width: 17.5rem;
    background: var(--ra-navy);
    border-right: 1px solid #1D3852;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #C5D2E0;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #DDE7F0;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 0.55rem;
    margin-bottom: 0.18rem;
    padding: 0.4rem 0.45rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #DDE7F0;
    font-weight: 500;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: rgba(45, 212, 191, 0.14);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: #FFFFFF;
    font-weight: 700;
}

.ra-sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.2rem 0 1.6rem;
}

.ra-sidebar-mark {
    display: grid;
    width: 2.65rem;
    height: 2.65rem;
    place-items: center;
    border: 1px solid rgba(153, 246, 228, 0.42);
    border-radius: 0.8rem;
    background: rgba(15, 118, 110, 0.35);
    color: #CCFBF1;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.ra-sidebar-name {
    color: #FFFFFF;
    font-size: 1.03rem;
    font-weight: 750;
    line-height: 1.2;
}

.ra-sidebar-subtitle {
    margin-top: 0.2rem;
    color: #A9BBD0;
    font-size: 0.67rem;
    font-weight: 650;
    letter-spacing: 0.11em;
    text-transform: uppercase;
}

.ra-app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 0 0 1.25rem;
    padding: 0.8rem 1rem;
    border: 1px solid var(--ra-border);
    border-radius: 0.9rem;
    background: #FFFFFF;
    box-shadow: 0 2px 8px rgba(16, 35, 56, 0.035);
}

.ra-app-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.ra-app-mark {
    display: grid;
    width: 2.8rem;
    height: 2.8rem;
    place-items: center;
    border-radius: 0.8rem;
    background: var(--ra-teal);
    color: #FFFFFF;
    font-size: 0.84rem;
    font-weight: 800;
    letter-spacing: 0.04em;
}

.ra-app-name {
    color: #102338 !important;
    font-size: 1.32rem;
    font-weight: 800;
    line-height: 1.2;
}

.ra-app-context {
    margin-top: 0.18rem;
    color: var(--ra-muted);
    font-size: 0.76rem;
}

.ra-local-badge {
    padding: 0.4rem 0.7rem;
    border: 1px solid #C8E8E2;
    border-radius: 999px;
    background: #EAF7F4;
    color: #17675F;
    font-size: 0.72rem;
    font-weight: 700;
    white-space: nowrap;
}

[data-testid="stMetric"] {
    min-height: 7rem;
    padding: 1rem 1.1rem;
    border: 1px solid var(--ra-border);
    border-radius: 0.85rem;
    background: var(--ra-surface);
    box-shadow: 0 2px 8px rgba(16, 35, 56, 0.035);
}

[data-testid="stMetricLabel"] p {
    color: var(--ra-muted);
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: var(--ra-navy);
    font-weight: 750;
}

.stButton > button {
    min-height: 2.65rem;
    border-radius: 0.6rem;
    font-weight: 650;
    transition: filter 120ms ease, box-shadow 120ms ease;
}

.stButton > button[kind="primary"] {
    border-color: var(--ra-teal);
    background: var(--ra-teal);
    color: #FFFFFF;
}

.stButton > button:hover {
    filter: brightness(0.96);
    box-shadow: 0 3px 10px rgba(15, 118, 110, 0.14);
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
    border-radius: 0.55rem;
}

[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    overflow: hidden;
    border: 1px solid var(--ra-border);
    border-radius: 0.85rem;
    background: var(--ra-surface);
}

[data-testid="stAlert"] {
    border-radius: 0.7rem;
}

.ra-app-footer {
    margin-top: 2.5rem;
    padding-top: 0.9rem;
    border-top: 1px solid var(--ra-border);
    color: var(--ra-muted);
    font-size: 0.75rem;
}

@media (max-width: 700px) {
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 1rem 2rem;
    }

    .ra-app-header {
        margin-bottom: 1.2rem;
        padding: 0.7rem;
    }

    .ra-local-badge {
        font-size: 0.65rem;
    }
}
</style>
"""


def apply_theme() -> None:
    """Apply shared Streamlit styles and a consistent Plotly chart template."""
    pio.templates.default = "plotly_white"
    st.markdown(_STYLES, unsafe_allow_html=True)


def render_app_header() -> None:
    """Render consistent application branding and the local-demo status."""
    st.sidebar.markdown(
        """
        <div class="ra-sidebar-brand">
            <div class="ra-sidebar-mark">RA</div>
            <div>
                <div class="ra-sidebar-name">RouteArena</div>
                <div class="ra-sidebar-subtitle">Optimization studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="ra-app-header">
            <div class="ra-app-brand">
                <div class="ra-app-mark">RA</div>
                <div>
                    <div class="ra-app-name" style="color:#102338;font-size:1.32rem;font-weight:800;">
                        RouteArena
                    </div>
                    <div class="ra-app-context">Operations research · delivery planning</div>
                </div>
            </div>
            <div class="ra-local-badge">Local demonstration</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_app_footer() -> None:
    """Show the project’s scope reminder at the end of each page."""
    st.markdown(
        '<div class="ra-app-footer">RouteArena · Educational decision-support demo · '
        "Synthetic inputs and model assumptions are identified in each workflow.</div>",
        unsafe_allow_html=True,
    )
