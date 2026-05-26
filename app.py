from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from notebook_model_artifact import register_notebook_model_class


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "Churnzero_data" / "CZ_raw"
PROCESSED_DATA_DIR = BASE_DIR / "Churnzero_data" / "CZ_processed"
ARTIFACT_DIR = BASE_DIR / "retention_flow_outputs" / "artifacts"
CHART_DIR = BASE_DIR / "retention_flow_outputs" / "charts"
ASSET_DIR = BASE_DIR / "assets"
GITHUB_REPO_URL = "https://github.com/Shrutika009/RetentionFlow"

ACCENT = "#0EA5E9"
ACCENT_2 = "#6366F1"
DANGER = "#E11D48"
WARNING = "#D97706"
SUCCESS = "#059669"
RISK_RED_SCALE = ["#7F1D1D", "#B91C1C", "#E11D48", "#FB7185", "#FCA5A5"]
RISK_TIER_COLORS = {
    "Low": "#D8BFA5",
    "Medium": "#B9824F",
    "High": "#7B4A2D",
    "Critical": "#6B5040",
}
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#D8E3F2"
SURFACE = "#FFFFFF"
SOFT = "#F1F6FD"
BG = "#F6F9FF"


st.set_page_config(
    page_title="RetentionFlow - ChurnZero",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        --bg: #F6F9FF;
        --surface: #FFFFFF;
        --soft: #F1F6FD;
        --border: #D8E3F2;
        --accent: #0EA5E9;
        --accent2: #6366F1;
        --danger: #E11D48;
        --warning: #D97706;
        --success: #059669;
        --text: #0F172A;
        --muted: #64748B;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
        opacity: 1 !important;
        filter: none !important;
        backdrop-filter: none !important;
        -webkit-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    body * {
        font-family: "Raleway", sans-serif !important;
    }

    .material-icons,
    .material-icons-rounded,
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-symbols-sharp,
    [class*="material-symbols"],
    [class*="material-icons"] {
        font-family: "Material Symbols Rounded", "Material Icons" !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
    }

    [data-testid="stHeader"], [data-testid="stToolbar"], header {
        background: transparent !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stDecoration"],
    [data-testid="manage-app-button"],
    .stDeployButton,
    [class*="stDeployButton"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] *,
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] *,
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"],
    button[title="Close sidebar"],
    button[title="Open sidebar"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
        box-shadow: 14px 0 34px rgba(15, 23, 42, 0.06);
    }

    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .block-container {
        max-width: 100% !important;
        padding: 1.4rem 2rem 2.5rem 2rem !important;
    }

    [data-testid="stAppViewContainer"] *,
    [data-testid="stMain"] *,
    [data-testid="stVerticalBlock"] *,
    [data-testid="stHorizontalBlock"] * {
        opacity: 1 !important;
        filter: none !important;
        backdrop-filter: none !important;
        -webkit-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text) !important;
        letter-spacing: 0 !important;
    }

    .wordmark {
        font-size: 1.55rem;
        font-weight: 900;
        letter-spacing: 0;
        color: var(--text);
        line-height: 1.05;
    }

    .wordmark span {
        color: var(--accent);
    }

    .sidebar-subtitle {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin: 0.4rem 0 1rem 0;
    }

    .sidebar-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 0.85rem 0 0.25rem 0;
    }

    .github-link {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.55rem;
        width: 100%;
        min-height: 42px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: #FFFFFF;
        color: var(--text) !important;
        font-size: 0.82rem;
        font-weight: 900;
        text-decoration: none !important;
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
        transition: border-color 140ms ease, box-shadow 140ms ease, transform 140ms ease;
    }

    .github-link:hover {
        border-color: var(--accent);
        box-shadow: 0 16px 32px rgba(14, 165, 233, 0.14);
        transform: translateY(-1px);
        color: var(--accent) !important;
    }

    .github-link svg {
        width: 1.1rem;
        height: 1.1rem;
        fill: currentColor;
        flex: 0 0 auto;
    }

    .sidebar-source-link {
        margin: 0.9rem 0 1.1rem;
    }

    .homepage-source-link {
        width: min(280px, 100%);
        margin-left: auto;
    }

    .homepage-source-link .source-url {
        display: block;
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 700;
        line-height: 1.15;
    }

    .footer-source-link {
        width: min(260px, 100%);
        margin: 0.45rem auto 0;
    }

    @media (max-width: 760px) {
        .homepage-source-link {
            width: 100%;
            margin-left: 0;
            margin-top: 0.8rem;
        }
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stTextInput"] > div,
    [data-testid="stTextInput"] > div > div,
    [data-baseweb="input"] input,
    [data-testid="stTextInput"] input,
    .stNumberInput input,
    .stTextInput input {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        min-height: 40px !important;
    }

    [data-baseweb="select"] > div:hover,
    [data-baseweb="input"] > div:hover,
    [data-testid="stTextInput"] > div:hover,
    [data-testid="stTextInput"] > div > div:hover,
    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="input"] > div:focus-within,
    [data-testid="stTextInput"] > div:focus-within,
    [data-testid="stTextInput"] > div > div:focus-within,
    [data-baseweb="input"] input:hover,
    [data-baseweb="input"] input:focus,
    [data-testid="stTextInput"] input:hover,
    [data-testid="stTextInput"] input:focus,
    .stTextInput input:hover,
    .stTextInput input:focus,
    input:hover,
    input:focus,
    input[aria-invalid="true"],
    input[aria-invalid="true"]:focus {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        box-shadow: none !important;
        outline: none !important;
    }

    input::selection,
    [data-baseweb="input"] input::selection,
    [data-testid="stTextInput"] input::selection,
    .stTextInput input::selection,
    textarea::selection {
        background: #BFDBFE !important;
        color: var(--text) !important;
    }

    input::-moz-selection,
    [data-baseweb="input"] input::-moz-selection,
    [data-testid="stTextInput"] input::-moz-selection,
    .stTextInput input::-moz-selection,
    textarea::-moz-selection {
        background: #BFDBFE !important;
        color: var(--text) !important;
    }

    input:-webkit-autofill,
    input:-webkit-autofill:hover,
    input:-webkit-autofill:focus {
        -webkit-box-shadow: 0 0 0 1000px #FFFFFF inset !important;
        -webkit-text-fill-color: var(--text) !important;
        caret-color: var(--accent) !important;
    }

    [data-baseweb="input"],
    [data-baseweb="input"] *,
    [data-testid="stTextInput"],
    [data-testid="stTextInput"] * {
        caret-color: var(--accent) !important;
    }

    [data-testid="InputInstructions"],
    [data-testid="InputInstructions"] *,
    div[class*="InputInstructions"],
    div[class*="InputInstructions"] * {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        min-height: 0 !important;
        opacity: 0 !important;
    }

    [data-baseweb="select"] div,
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="input"] input::placeholder,
    .stNumberInput input::placeholder,
    .stTextInput input::placeholder {
        color: var(--text) !important;
        opacity: 1 !important;
    }

    [data-baseweb="select"] svg,
    [data-baseweb="select"] svg * {
        fill: var(--text) !important;
        color: var(--text) !important;
        opacity: 1 !important;
    }

    [data-testid="stWidgetLabel"],
    label {
        color: var(--text) !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: var(--text) !important;
        font-weight: 800 !important;
        opacity: 1 !important;
    }

    .stNumberInput,
    .stSelectbox {
        margin-bottom: 0.55rem;
    }

    [data-baseweb="popover"],
    [role="listbox"],
    [role="option"],
    [role="option"] * {
        background: var(--surface) !important;
        color: var(--text) !important;
    }

    [role="option"]:hover,
    [role="option"][aria-selected="true"] {
        background: #FFFFFF !important;
    }

    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.1rem;
        box-shadow: 0 16px 32px rgba(15, 23, 42, 0.05);
        min-height: 112px;
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .kpi-value {
        font-size: 1.8rem;
        font-weight: 900;
        color: var(--text);
        line-height: 1.1;
    }

    .kpi-sub {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.3rem;
    }

    .section-header {
        color: var(--text);
        font-size: 1.02rem;
        font-weight: 800;
        margin: 0.25rem 0 0.75rem 0;
    }

    .predictor-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.1rem 1.2rem;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
        margin-top: 1rem;
    }

    .predictor-section {
        color: var(--text) !important;
        font-size: 1.45rem;
        font-weight: 900;
        line-height: 1.15;
        margin: 0.2rem 0 1rem;
        padding-left: 0.85rem;
        border-left: 5px solid var(--accent);
        letter-spacing: 0 !important;
        font-family: "Raleway", sans-serif !important;
    }

    .predictor-divider {
        border-top: 1px solid var(--border);
        margin: 1.1rem 0 0.9rem;
    }

    .stNumberInput input,
    .stTextInput input {
        min-height: 40px !important;
    }

    .info-box {
        background: rgba(14, 165, 233, 0.08);
        border: 1px solid rgba(14, 165, 233, 0.22);
        border-radius: 10px;
        color: var(--muted);
        padding: 0.85rem 1rem;
        font-size: 0.88rem;
        margin-bottom: 1rem;
    }

    .overview-signal {
        background:
            radial-gradient(circle at 92% 12%, rgba(14, 165, 233, 0.16), transparent 30%),
            linear-gradient(180deg, #FFFFFF, #F8FBFF);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        min-height: 112px;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
    }

    .overview-signal-label {
        color: var(--muted);
        font-size: 0.68rem;
        font-weight: 900;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .overview-signal-value {
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 900;
        line-height: 1.25;
    }

    .overview-signal-sub {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 650;
        margin-top: 0.35rem;
    }

    .overview-chart-band {
        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(241,246,253,0.62));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 20px 46px rgba(15, 23, 42, 0.06);
    }

    [data-testid="stTabs"] button {
        color: var(--muted) !important;
        font-weight: 700 !important;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }

    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"] {
        background: var(--surface) !important;
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="StyledFullScreenButton"],
    [data-testid="StyledFullScreenButton"] *,
    button[title="View fullscreen"],
    button[aria-label="View fullscreen"],
    button[title="Fullscreen"],
    button[aria-label="Fullscreen"],
    .modebar,
    .modebar-container,
    .modebar-group,
    a.modebar-btn {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    [data-testid="stPlotlyChart"] svg text,
    [data-testid="stPlotlyChart"] .gtitle,
    [data-testid="stPlotlyChart"] .xtitle,
    [data-testid="stPlotlyChart"] .ytitle,
    [data-testid="stPlotlyChart"] .xaxislayer-above text,
    [data-testid="stPlotlyChart"] .yaxislayer-above text,
    [data-testid="stPlotlyChart"] .legend text,
    [data-testid="stPlotlyChart"] .colorbar text,
    [data-testid="stPlotlyChart"] .annotation-text {
        fill: var(--text) !important;
        color: var(--text) !important;
        opacity: 1 !important;
    }

    .stButton > button,
    [data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #0B1F3A, #123B6D) !important;
        color: white !important;
        border: 0 !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        box-shadow: 0 12px 26px rgba(11, 31, 58, 0.22) !important;
    }

    .stButton > button:hover,
    [data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(135deg, #123B6D, #0EA5E9) !important;
        color: #FFFFFF !important;
    }

    [data-testid="stFormSubmitButton"] button,
    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:focus,
    [data-testid="stFormSubmitButton"] button:disabled,
    [data-testid="stForm"] button[kind="primary"],
    [data-testid="stForm"] button[kind="primaryFormSubmit"],
    [data-testid="stForm"] button[type="submit"],
    [data-testid="stForm"] button[kind="primary"]:hover,
    [data-testid="stForm"] button[kind="primaryFormSubmit"]:hover,
    [data-testid="stForm"] button[type="submit"]:hover,
    [data-testid="stForm"] button[kind="primary"]:focus,
    [data-testid="stForm"] button[kind="primaryFormSubmit"]:focus,
    [data-testid="stForm"] button[type="submit"]:focus {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: #FFFFFF !important;
        border: 0 !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        box-shadow: none !important;
        opacity: 1 !important;
        min-height: 44px !important;
    }

    [data-testid="stForm"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 1.15rem 1.25rem 1.25rem !important;
        box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
    }

    .predictor-shell {
        max-width: 1120px;
        margin: 0 auto;
    }

    .predictor-note {
        background: rgba(225, 29, 72, 0.10);
        border: 1px solid rgba(225, 29, 72, 0.34);
        border-left: 5px solid var(--danger);
        border-radius: 8px;
        color: #7F1D1D;
        font-weight: 800;
        padding: 0.75rem 0.9rem;
        margin: 1rem 0 0;
        width: 100%;
        max-width: none;
    }

    .predictor-processing {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12), rgba(99, 102, 241, 0.10));
        border: 1px solid rgba(14, 165, 233, 0.32);
        border-left: 5px solid var(--accent);
        border-radius: 8px;
        color: var(--text);
        font-weight: 900;
        padding: 0.8rem 0.95rem;
        margin: 1rem 0 0;
        width: 100%;
        animation: processingPulse 1.15s ease-in-out infinite alternate;
    }

    @keyframes processingPulse {
        from { box-shadow: 0 0 0 rgba(14, 165, 233, 0.0); }
        to { box-shadow: 0 0 24px rgba(14, 165, 233, 0.22); }
    }

    .white-table-wrap {
        overflow: auto;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #FFFFFF;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.04);
    }

    .white-data-table {
        width: 100%;
        border-collapse: collapse;
        background: #FFFFFF;
        color: var(--text);
        font-family: "Raleway", sans-serif !important;
        font-size: 0.8rem;
    }

    .white-data-table thead tr,
    .white-data-table tbody tr,
    .white-data-table th,
    .white-data-table td {
        background: #FFFFFF !important;
        color: var(--text) !important;
    }

    .white-data-table th {
        position: sticky;
        top: 0;
        z-index: 1;
        font-weight: 900;
        border-bottom: 1px solid var(--border);
    }

    .white-data-table th,
    .white-data-table td {
        padding: 0.65rem 0.75rem;
        border-right: 1px solid var(--border);
        border-bottom: 1px solid var(--border);
        text-align: left;
        white-space: nowrap;
    }

    .risk-meter {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 14px 30px rgba(15, 23, 42, 0.04);
    }

    .risk-meter-label {
        color: var(--text);
        font-size: 0.84rem;
        font-weight: 900;
        margin-bottom: 0.7rem;
    }

    .risk-meter-track {
        height: 16px;
        border-radius: 999px;
        background: #EAF4FF;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    .risk-meter-fill {
        height: 100%;
        border-radius: 999px;
    }

    .risk-meter-scale {
        display: flex;
        justify-content: space-between;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 800;
        margin-top: 0.55rem;
    }

    hr {
        border-color: var(--border) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


CHART_ALIASES = {
    "pr_curve.png": "pr_curves.png",
    "threshold_optimization.png": "threshold_optimisation.png",
}


def github_source_link(container_class: str = "") -> str:
    class_attr = f" {container_class}" if container_class else ""
    return f"""
    <div class="github-link-wrap{class_attr}">
        <a class="github-link" href="{GITHUB_REPO_URL}" target="_blank" rel="noopener noreferrer" aria-label="View Source Code on GitHub">
            <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.56 7.56 0 0 1 8 3.86c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/>
            </svg>
            <span>
                View Source Code
                <span class="source-url">github.com/Shrutika009/RetentionFlow</span>
            </span>
        </a>
    </div>
    """


def dashboard_chart(filename: str, caption: str | None = None) -> None:
    candidates = [filename]
    if filename in CHART_ALIASES:
        candidates.append(CHART_ALIASES[filename])

    for folder in (ASSET_DIR, CHART_DIR):
        for candidate in candidates:
            path = folder / candidate
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)
                return
    st.info(f"Missing dashboard chart: {filename}")


def notebook_chart(filename: str, caption: str | None = None) -> None:
    dashboard_chart(filename, caption)


def style_light_chart(fig, height: int = 360):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=TEXT, family="Raleway, sans-serif"),
        title=dict(font=dict(color=TEXT, size=18), x=0.02, xanchor="left"),
        xaxis=dict(title_font=dict(color=TEXT), tickfont=dict(color=TEXT)),
        yaxis=dict(title_font=dict(color=TEXT), tickfont=dict(color=TEXT)),
        margin=dict(l=24, r=24, t=58, b=36),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=TEXT, bordercolor=BORDER),
        legend=dict(font=dict(color=TEXT)),
        coloraxis_colorbar=dict(
            title=dict(font=dict(color=TEXT)),
            tickfont=dict(color=TEXT),
        ),
    )
    fig.update_xaxes(
        gridcolor="#E6EEF8",
        zerolinecolor="#D8E3F2",
        linecolor="#D8E3F2",
        tickfont=dict(color=TEXT),
        tickfont_color=TEXT,
        title_font=dict(color=TEXT),
        title_font_color=TEXT,
    )
    fig.update_yaxes(
        gridcolor="#E6EEF8",
        zerolinecolor="#D8E3F2",
        linecolor="#D8E3F2",
        tickfont=dict(color=TEXT),
        tickfont_color=TEXT,
        title_font=dict(color=TEXT),
        title_font_color=TEXT,
    )
    return fig


def white_table(data: pd.DataFrame, height: int = 360) -> None:
    display = data.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].round(4)
    table_html = display.to_html(index=False, escape=True, classes="white-data-table")
    st.markdown(
        f"""
        <div class="white-table-wrap" style="max-height:{height}px;">
            {table_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def enable_enter_to_next_field() -> None:
    components.html(
        """
        <script>
        (() => {
            const doc = window.parent.document;
            if (doc.__retentionFlowEnterNavigationInstalled) return;
            doc.__retentionFlowEnterNavigationInstalled = true;

            const isVisible = (element) => {
                if (!element || element.disabled) return false;
                const style = window.parent.getComputedStyle(element);
                return style.display !== "none"
                    && style.visibility !== "hidden"
                    && style.opacity !== "0"
                    && (element.offsetWidth > 0 || element.offsetHeight > 0 || element.getClientRects().length > 0);
            };

            const isTextEntry = (element) => {
                if (!element) return false;
                const tag = element.tagName.toLowerCase();
                const role = (element.getAttribute("role") || "").toLowerCase();
                const type = (element.getAttribute("type") || "text").toLowerCase();
                return tag === "textarea"
                    || role === "combobox"
                    || (tag === "input" && !["button", "submit", "reset", "checkbox", "radio", "hidden"].includes(type));
            };

            const focusables = () => {
                const main = doc.querySelector('[data-testid="stMain"]') || doc;
                const selectors = [
                    '[data-testid="stTextInput"] input',
                    '[data-baseweb="select"] input[role="combobox"]',
                    '[data-baseweb="select"] [role="combobox"]',
                    'textarea',
                    'button'
                ].join(',');

                return Array.from(main.querySelectorAll(selectors))
                    .filter(isVisible)
                    .filter((element) => {
                        const text = (element.innerText || element.value || element.getAttribute("aria-label") || "").trim();
                        return !text.includes("Deploy") && !text.includes("Stop");
                    });
            };

            const moveNext = (current) => {
                const fields = focusables();
                const index = fields.indexOf(current);
                const next = fields[index + 1];
                if (!next) return false;
                next.focus({ preventScroll: true });
                next.scrollIntoView({ block: "center", behavior: "smooth" });
                return true;
            };

            doc.addEventListener("keydown", (event) => {
                if (event.key !== "Enter" || event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return;
                const target = event.target;
                if (!isTextEntry(target)) return;

                const menuOpen = !!doc.querySelector('[role="listbox"]');
                if (menuOpen && (target.getAttribute("role") || "").toLowerCase() === "combobox") {
                    setTimeout(() => moveNext(target), 80);
                    return;
                }

                if (moveNext(target)) {
                    event.preventDefault();
                    event.stopPropagation();
                }
            }, true);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


PRODUCT_FLAGS = [
    "savings_account_flag",
    "current_account_flag",
    "credit_card_flag",
    "personal_loan_flag",
    "home_loan_flag",
    "auto_loan_flag",
    "fixed_deposit_flag",
    "investment_product_flag",
    "insurance_product_flag",
    "demat_account_flag",
]


def load_model_artifacts():
    register_notebook_model_class()
    model = joblib.load(ARTIFACT_DIR / "retentionflow_best_model.pkl")
    encoders = joblib.load(ARTIFACT_DIR / "retentionflow_encoders.pkl")
    scaler = joblib.load(ARTIFACT_DIR / "retentionflow_scaler.pkl")
    with open(ARTIFACT_DIR / "retentionflow_config.json", "r", encoding="utf-8") as fh:
        config = json.load(fh)
    with open(ARTIFACT_DIR / "retentionflow_features.json", "r", encoding="utf-8") as fh:
        features = json.load(fh)
    return model, encoders, scaler, config, features


MODEL, ENCODERS, SCALER, MODEL_CONFIG, MODEL_FEATURES = load_model_artifacts()


def add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).str.strip().str.lower()
    present_flags = [col for col in PRODUCT_FLAGS if col in out.columns]

    def yes_no_to_number(series: pd.Series) -> pd.Series:
        mapped = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0})
        )
        numeric = pd.to_numeric(series, errors="coerce")
        return mapped.fillna(numeric)

    loan_flags = [col for col in ["personal_loan_flag", "home_loan_flag", "auto_loan_flag"] if col in out.columns]
    if loan_flags:
        derived_has_loan = (out[loan_flags].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0).astype(int)
    else:
        derived_has_loan = pd.Series(0, index=out.index)
    if "has_loan" in out.columns:
        out["has_loan"] = yes_no_to_number(out["has_loan"]).fillna(derived_has_loan).astype(int)
    else:
        out["has_loan"] = derived_has_loan

    derived_default = pd.Series(0, index=out.index)
    if "emi_payment_delay_count" in out.columns:
        derived_default = derived_default | (pd.to_numeric(out["emi_payment_delay_count"], errors="coerce").fillna(0) > 0)
    if "loan_default_risk_score" in out.columns:
        risk_score = pd.to_numeric(out["loan_default_risk_score"], errors="coerce").fillna(0)
        derived_default = derived_default | (risk_score > risk_score.median())
    derived_default = derived_default.astype(int)
    if "loan_default_history" in out.columns:
        out["loan_default_history"] = yes_no_to_number(out["loan_default_history"]).fillna(derived_default).astype(int)
    else:
        out["loan_default_history"] = derived_default

    mobile_norm = out["mobile_app_login_count"] / (out["mobile_app_login_count"].max() + 1)
    web_norm = out["website_login_count"] / (out["website_login_count"].max() + 1)
    out["composite_digital_score"] = (
        mobile_norm * 35
        + web_norm * 20
        + out["digital_transaction_ratio"] * 25
        + out["mobile_banking_active_flag"] * 10
        + out["paperless_statement_enabled"] * 10
    ).round(4)
    out["low_digital_activity"] = (out["composite_digital_score"] < 20).astype(int)
    out["balance_to_income_ratio"] = (out["avg_monthly_balance"] / (out["annual_income"] / 12 + 1)).round(4)
    out["clv_to_income_ratio"] = (out["customer_lifetime_value"] / (out["annual_income"] + 1)).round(4)
    out["spend_to_limit_ratio"] = (out["credit_card_spend"] / (out["credit_card_limit"] + 1)).round(4)
    out["emi_to_income_ratio"] = (out["emi_amount"] / (out["monthly_income_estimate"] + 1)).round(4)
    out["transaction_to_balance_ratio"] = (out["monthly_transaction_value"] / (out["avg_monthly_balance"] + 1)).round(4)
    out["total_products_held"] = out[present_flags].sum(axis=1) if present_flags else 0
    out["single_product_flag"] = (out["total_products_held"] == 1).astype(int)
    out["multi_product_flag"] = (out["total_products_held"] >= 3).astype(int)
    out["products_per_tenure"] = out["total_products_held"] / (out["tenure_months"] + 1)
    out["complaint_resolution_ratio"] = (out["unresolved_complaint_count"] / (out["total_complaints"] + 1)).round(4)
    out["high_complaint_flag"] = (out["total_complaints"] > out["total_complaints"].quantile(0.75)).astype(int)
    out["escalation_rate"] = (out["escalation_count"] / (out["total_complaints"] + 1)).round(4)
    out["satisfaction_nps_composite"] = (out["satisfaction_score"] * 0.5 + out["nps_score"] * 0.5).round(4)
    out["low_satisfaction_flag"] = ((out["satisfaction_score"] < 3) | (out["nps_score"] < 3)).astype(int)
    out["credit_risk_composite"] = (
        out["loan_default_risk_score"] * 0.4
        + out["credit_utilization_ratio"] * 0.3
        + out["late_credit_card_payment_count"] * 0.2
        + out["emi_payment_delay_count"] * 0.1
    ).round(4)
    out["financial_stress_flag"] = (
        (out["credit_utilization_ratio"] > 0.8)
        | (out["loan_default_risk_score"] > out["loan_default_risk_score"].quantile(0.85))
        | (out["emi_payment_delay_count"] >= 3)
    ).astype(int)
    out["credit_utilization_trend"] = (out["credit_utilization_ratio"] - out["credit_utilization_3m_avg"]).round(4)
    out["transaction_frequency_score"] = (out["monthly_transaction_count"] / (out["tenure_months"] + 1)).round(4)
    out["upi_dominance_ratio"] = (out["upi_transaction_count"] / (out["monthly_transaction_count"] + 1)).round(4)
    out["transaction_growth"] = out["total_ct_chng_q4_q1"]
    out["high_cash_withdrawal_flag"] = (out["cash_withdrawal_count"] > out["cash_withdrawal_count"].quantile(0.8)).astype(int)
    out["zero_balance_flag"] = (out["avg_monthly_balance"] <= 0).astype(int)
    out["balance_declining_flag"] = (out["balance_decline_percentage"] > 20).astype(int)
    out["login_days_x_inactive"] = out["last_login_days"] * out["account_inactive_days"]
    out["dormancy_risk_flag"] = ((out["last_login_days"] > 30) & (out["account_inactive_days"] > 60)).astype(int)
    out["recency_engagement_score"] = (
        1 / (out["last_login_days"] + 1)
        + 1 / (out["account_inactive_days"] + 1)
        + 1 / (out["last_contacted_days"] + 1)
    ).round(6)
    out["campaign_response_rate"] = (out["campaign_response_count"] / (out["campaign_received_count"] + 1)).round(4)
    out["retention_offer_response"] = (out["retention_offer_accepted"] / (out["retention_offer_received"] + 1)).round(4)
    out["competitor_aware_flag"] = (
        out["competitor_bank_offer_awareness"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"high": 1, "medium": 1, "low": 1, "aware": 1, "not aware": 0})
        .fillna(0)
        .astype(int)
    )

    def min_max_norm(series: pd.Series) -> pd.Series:
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    out["customer_value"] = (
        min_max_norm(out["customer_lifetime_value"]) * 0.4
        + min_max_norm(out["annual_income"]) * 0.25
        + min_max_norm(out["avg_monthly_balance"]) * 0.2
        + min_max_norm(out["total_products_held"]) * 0.15
    ).round(4)
    out["wealth_tier"] = pd.cut(
        out["customer_value"],
        bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["Basic", "Standard", "Premium", "Elite"],
    ).astype(str)
    out["age_tenure_interaction"] = out["age"] * out["tenure_months"]
    out["income_products_interaction"] = out["annual_income"] * out["total_products_held"]
    out["complaints_x_inactivity"] = out["total_complaints"] * out["account_inactive_days"]
    out["digital_x_balance"] = out["composite_digital_score"] * out["avg_monthly_balance"]
    out["satisfaction_x_tenure"] = out["satisfaction_score"] * out["tenure_months"]
    out["rfm_recency"] = min_max_norm(1 / (out["account_inactive_days"] + 1))
    out["rfm_frequency"] = min_max_norm(out["monthly_transaction_count"])
    out["rfm_monetary"] = min_max_norm(out["monthly_transaction_value"])
    out["rfm_score"] = (
        out["rfm_recency"] * 0.35
        + out["rfm_frequency"] * 0.35
        + out["rfm_monetary"] * 0.30
    ).round(4)
    for col in [
        "annual_income",
        "customer_lifetime_value",
        "avg_monthly_balance",
        "total_trans_amt",
        "loan_outstanding_amount",
    ]:
        out[f"log_{col}"] = np.log1p(out[col].clip(lower=0))
    for col in out.select_dtypes(include=["category"]).columns:
        out[col] = out[col].astype(str).replace("nan", "unknown")
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def model_matrix(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    if reference is not None and len(reference):
        ref = reference.drop(columns=["churn"], errors="ignore").copy()
        current = frame.drop(columns=["churn"], errors="ignore").copy()
        if "customer_id" in ref.columns and "customer_id" in current.columns:
            ref = ref.set_index("customer_id", drop=False)
            current = current.set_index("customer_id", drop=False)
            missing_ids = current.index.difference(ref.index)
            common_ids = current.index.intersection(ref.index)
            if len(common_ids):
                ref.loc[common_ids, current.columns] = current.loc[common_ids, current.columns]
            if len(missing_ids):
                ref = pd.concat([ref, current.loc[missing_ids]], axis=0)
            engineered = add_model_features(ref.reset_index(drop=True))
            engineered["customer_id"] = ref["customer_id"].to_numpy()
            matrix = engineered.set_index("customer_id").loc[current.index]
            matrix.index = frame.index
        else:
            work = pd.concat([ref, current], ignore_index=True)
            matrix = add_model_features(work).tail(len(frame)).set_index(frame.index)
    else:
        matrix = add_model_features(frame)
    for col, encoder in ENCODERS.items():
        class_map = {str(cls).lower(): cls for cls in encoder.classes_}
        fallback = encoder.classes_[0]
        matrix[col] = matrix[col].astype(str).str.lower().map(class_map).fillna(fallback)
        matrix[col] = encoder.transform(matrix[col])
    for feature in MODEL_FEATURES:
        if feature not in matrix.columns:
            matrix[feature] = 0
    matrix = matrix[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    if MODEL_CONFIG.get("model_name") == "LogisticRegression":
        scaled = SCALER.transform(matrix)
        return pd.DataFrame(scaled, columns=MODEL_FEATURES, index=frame.index)
    return matrix


def predict_churn(frame: pd.DataFrame, reference: pd.DataFrame | None = None) -> np.ndarray:
    return MODEL.predict_proba(model_matrix(frame, reference))[:, 1]


@st.cache_data(show_spinner=False)
def reference_feature_stats(reference: pd.DataFrame) -> dict:
    ref = reference.copy()
    present_flags = [col for col in PRODUCT_FLAGS if col in ref.columns]
    if present_flags:
        ref["total_products_held"] = ref[present_flags].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    else:
        ref["total_products_held"] = 0

    def numeric(col: str) -> pd.Series:
        return pd.to_numeric(ref[col], errors="coerce").fillna(0) if col in ref.columns else pd.Series([0.0])

    def minmax(col: str) -> tuple[float, float]:
        values = numeric(col)
        return float(values.min()), float(values.max())

    recency = 1 / (numeric("account_inactive_days") + 1)
    return {
        "mobile_app_login_count_max": float(numeric("mobile_app_login_count").max()),
        "website_login_count_max": float(numeric("website_login_count").max()),
        "total_complaints_q75": float(numeric("total_complaints").quantile(0.75)),
        "loan_default_risk_score_median": float(numeric("loan_default_risk_score").median()),
        "loan_default_risk_score_q85": float(numeric("loan_default_risk_score").quantile(0.85)),
        "cash_withdrawal_count_q80": float(numeric("cash_withdrawal_count").quantile(0.8)),
        "customer_lifetime_value_minmax": minmax("customer_lifetime_value"),
        "annual_income_minmax": minmax("annual_income"),
        "avg_monthly_balance_minmax": minmax("avg_monthly_balance"),
        "total_products_held_minmax": (float(ref["total_products_held"].min()), float(ref["total_products_held"].max())),
        "rfm_recency_minmax": (float(recency.min()), float(recency.max())),
        "monthly_transaction_count_minmax": minmax("monthly_transaction_count"),
        "monthly_transaction_value_minmax": minmax("monthly_transaction_value"),
    }


def add_model_features_single(frame: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = frame.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).str.strip().str.lower()

    def numeric(col: str) -> pd.Series:
        if col not in out.columns:
            out[col] = 0
        return pd.to_numeric(out[col], errors="coerce").fillna(0)

    def yes_no_to_number(series: pd.Series) -> pd.Series:
        mapped = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0})
        )
        numeric_series = pd.to_numeric(series, errors="coerce")
        return mapped.fillna(numeric_series)

    present_flags = [col for col in PRODUCT_FLAGS if col in out.columns]
    loan_flags = [col for col in ["personal_loan_flag", "home_loan_flag", "auto_loan_flag"] if col in out.columns]
    derived_has_loan = (
        out[loan_flags].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) > 0
    ).astype(int) if loan_flags else pd.Series(0, index=out.index)
    out["has_loan"] = yes_no_to_number(out["has_loan"]).fillna(derived_has_loan).astype(int) if "has_loan" in out.columns else derived_has_loan

    derived_default = pd.Series(0, index=out.index)
    if "emi_payment_delay_count" in out.columns:
        derived_default = derived_default | (numeric("emi_payment_delay_count") > 0)
    if "loan_default_risk_score" in out.columns:
        derived_default = derived_default | (numeric("loan_default_risk_score") > stats["loan_default_risk_score_median"])
    out["loan_default_history"] = (
        yes_no_to_number(out["loan_default_history"]).fillna(derived_default).astype(int)
        if "loan_default_history" in out.columns
        else derived_default.astype(int)
    )

    mobile_norm = numeric("mobile_app_login_count") / (stats["mobile_app_login_count_max"] + 1)
    web_norm = numeric("website_login_count") / (stats["website_login_count_max"] + 1)
    out["composite_digital_score"] = (
        mobile_norm * 35
        + web_norm * 20
        + numeric("digital_transaction_ratio") * 25
        + numeric("mobile_banking_active_flag") * 10
        + numeric("paperless_statement_enabled") * 10
    ).round(4)
    out["low_digital_activity"] = (out["composite_digital_score"] < 20).astype(int)
    out["balance_to_income_ratio"] = (numeric("avg_monthly_balance") / (numeric("annual_income") / 12 + 1)).round(4)
    out["clv_to_income_ratio"] = (numeric("customer_lifetime_value") / (numeric("annual_income") + 1)).round(4)
    out["spend_to_limit_ratio"] = (numeric("credit_card_spend") / (numeric("credit_card_limit") + 1)).round(4)
    out["emi_to_income_ratio"] = (numeric("emi_amount") / (numeric("monthly_income_estimate") + 1)).round(4)
    out["transaction_to_balance_ratio"] = (numeric("monthly_transaction_value") / (numeric("avg_monthly_balance") + 1)).round(4)
    out["total_products_held"] = out[present_flags].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1) if present_flags else 0
    out["single_product_flag"] = (out["total_products_held"] == 1).astype(int)
    out["multi_product_flag"] = (out["total_products_held"] >= 3).astype(int)
    out["products_per_tenure"] = out["total_products_held"] / (numeric("tenure_months") + 1)
    out["complaint_resolution_ratio"] = (numeric("unresolved_complaint_count") / (numeric("total_complaints") + 1)).round(4)
    out["high_complaint_flag"] = (numeric("total_complaints") > stats["total_complaints_q75"]).astype(int)
    out["escalation_rate"] = (numeric("escalation_count") / (numeric("total_complaints") + 1)).round(4)
    out["satisfaction_nps_composite"] = (numeric("satisfaction_score") * 0.5 + numeric("nps_score") * 0.5).round(4)
    out["low_satisfaction_flag"] = ((numeric("satisfaction_score") < 3) | (numeric("nps_score") < 3)).astype(int)
    out["credit_risk_composite"] = (
        numeric("loan_default_risk_score") * 0.4
        + numeric("credit_utilization_ratio") * 0.3
        + numeric("late_credit_card_payment_count") * 0.2
        + numeric("emi_payment_delay_count") * 0.1
    ).round(4)
    out["financial_stress_flag"] = (
        (numeric("credit_utilization_ratio") > 0.8)
        | (numeric("loan_default_risk_score") > stats["loan_default_risk_score_q85"])
        | (numeric("emi_payment_delay_count") >= 3)
    ).astype(int)
    out["credit_utilization_trend"] = (numeric("credit_utilization_ratio") - numeric("credit_utilization_3m_avg")).round(4)
    out["transaction_frequency_score"] = (numeric("monthly_transaction_count") / (numeric("tenure_months") + 1)).round(4)
    out["upi_dominance_ratio"] = (numeric("upi_transaction_count") / (numeric("monthly_transaction_count") + 1)).round(4)
    out["transaction_growth"] = numeric("total_ct_chng_q4_q1")
    out["high_cash_withdrawal_flag"] = (numeric("cash_withdrawal_count") > stats["cash_withdrawal_count_q80"]).astype(int)
    out["zero_balance_flag"] = (numeric("avg_monthly_balance") <= 0).astype(int)
    out["balance_declining_flag"] = (numeric("balance_decline_percentage") > 20).astype(int)
    out["login_days_x_inactive"] = numeric("last_login_days") * numeric("account_inactive_days")
    out["dormancy_risk_flag"] = ((numeric("last_login_days") > 30) & (numeric("account_inactive_days") > 60)).astype(int)
    out["recency_engagement_score"] = (
        1 / (numeric("last_login_days") + 1)
        + 1 / (numeric("account_inactive_days") + 1)
        + 1 / (numeric("last_contacted_days") + 1)
    ).round(6)
    out["campaign_response_rate"] = (numeric("campaign_response_count") / (numeric("campaign_received_count") + 1)).round(4)
    out["retention_offer_response"] = (numeric("retention_offer_accepted") / (numeric("retention_offer_received") + 1)).round(4)
    out["competitor_aware_flag"] = (
        out.get("competitor_bank_offer_awareness", pd.Series("", index=out.index))
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"high": 1, "medium": 1, "low": 1, "aware": 1, "not aware": 0})
        .fillna(0)
        .astype(int)
    )

    def min_max_norm_value(series: pd.Series, key: str) -> pd.Series:
        ref_min, ref_max = stats[key]
        values = pd.to_numeric(series, errors="coerce").fillna(0)
        min_value = np.minimum(values, ref_min)
        max_value = np.maximum(values, ref_max)
        return (values - min_value) / (max_value - min_value + 1e-9)

    out["customer_value"] = (
        min_max_norm_value(numeric("customer_lifetime_value"), "customer_lifetime_value_minmax") * 0.4
        + min_max_norm_value(numeric("annual_income"), "annual_income_minmax") * 0.25
        + min_max_norm_value(numeric("avg_monthly_balance"), "avg_monthly_balance_minmax") * 0.2
        + min_max_norm_value(out["total_products_held"], "total_products_held_minmax") * 0.15
    ).round(4)
    out["wealth_tier"] = pd.cut(
        out["customer_value"],
        bins=[0, 0.25, 0.50, 0.75, 1.01],
        labels=["Basic", "Standard", "Premium", "Elite"],
    ).astype(str)
    out["age_tenure_interaction"] = numeric("age") * numeric("tenure_months")
    out["income_products_interaction"] = numeric("annual_income") * out["total_products_held"]
    out["complaints_x_inactivity"] = numeric("total_complaints") * numeric("account_inactive_days")
    out["digital_x_balance"] = out["composite_digital_score"] * numeric("avg_monthly_balance")
    out["satisfaction_x_tenure"] = numeric("satisfaction_score") * numeric("tenure_months")
    out["rfm_recency"] = min_max_norm_value(1 / (numeric("account_inactive_days") + 1), "rfm_recency_minmax")
    out["rfm_frequency"] = min_max_norm_value(numeric("monthly_transaction_count"), "monthly_transaction_count_minmax")
    out["rfm_monetary"] = min_max_norm_value(numeric("monthly_transaction_value"), "monthly_transaction_value_minmax")
    out["rfm_score"] = (out["rfm_recency"] * 0.35 + out["rfm_frequency"] * 0.35 + out["rfm_monetary"] * 0.30).round(4)
    for col in ["annual_income", "customer_lifetime_value", "avg_monthly_balance", "total_trans_amt", "loan_outstanding_amount"]:
        out[f"log_{col}"] = np.log1p(numeric(col).clip(lower=0))
    for col in out.select_dtypes(include=["category"]).columns:
        out[col] = out[col].astype(str).replace("nan", "unknown")
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def manual_model_matrix(model_row: dict) -> pd.DataFrame:
    matrix = add_model_features_single(pd.DataFrame([model_row]), reference_feature_stats(MODEL_REFERENCE))
    for col, encoder in ENCODERS.items():
        class_map = {str(cls).lower(): cls for cls in encoder.classes_}
        fallback = encoder.classes_[0]
        matrix[col] = matrix[col].astype(str).str.lower().map(class_map).fillna(fallback)
        matrix[col] = encoder.transform(matrix[col])
    for feature in MODEL_FEATURES:
        if feature not in matrix.columns:
            matrix[feature] = 0
    matrix = matrix[MODEL_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    if MODEL_CONFIG.get("model_name") == "LogisticRegression":
        scaled = SCALER.transform(matrix)
        return pd.DataFrame(scaled, columns=MODEL_FEATURES, index=matrix.index)
    return matrix


@st.cache_data(show_spinner=False)
def cached_manual_prediction(model_row_json: str) -> tuple[float, pd.DataFrame]:
    model_row = json.loads(model_row_json)
    matrix = manual_model_matrix(model_row)
    score = float(MODEL.predict_proba(matrix)[:, 1][0])
    engineered = add_model_features_single(pd.DataFrame([model_row]), reference_feature_stats(MODEL_REFERENCE)).iloc[0]
    drivers = top_model_drivers(pd.Series(model_row), 12, engineered_row=engineered)
    return score, drivers


@st.cache_data(show_spinner=False)
def cached_filtered_predictions(data_scope: str, customer_ids: tuple[int, ...]) -> pd.DataFrame:
    if not customer_ids:
        return pd.DataFrame(columns=["customer_id", "ml_churn_probability"])
    source = (
        pd.concat([df_train, df_test], ignore_index=True)
        if data_scope == "Full Dataset"
        else (df_train if data_scope == "Training Data" else df_test)
    )
    subset = source[source["customer_id"].isin(customer_ids)].copy()
    if "notebook_churn_probability" in subset.columns and subset["notebook_churn_probability"].notna().any():
        subset["ml_churn_probability"] = pd.to_numeric(subset["notebook_churn_probability"], errors="coerce")
    else:
        subset["ml_churn_probability"] = np.nan
    missing = subset["ml_churn_probability"].isna()
    if missing.any():
        subset.loc[missing, "ml_churn_probability"] = predict_churn(subset.loc[missing], MODEL_REFERENCE)
    return subset[["customer_id", "ml_churn_probability"]]


def model_risk_label(probability: float) -> str:
    return "High" if probability >= float(MODEL_CONFIG["threshold"]) else "Low"


def model_tier(probability: float) -> tuple[str, str]:
    threshold = float(MODEL_CONFIG["threshold"])
    if probability < threshold:
        return "Low Risk", SUCCESS
    if probability >= 0.80:
        return "Critical Risk", DANGER
    if probability >= 0.60:
        return "High Risk", WARNING
    return "Medium Risk", "#CA8A04"


def format_probability(probability: float) -> str:
    if probability <= 0:
        return "0%"
    if probability < 0.001:
        return f"{probability:.4%}"
    if probability < 0.01:
        return f"{probability:.3%}"
    return f"{probability:.1%}"


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "shap_feature_importance.csv")


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    comparison = pd.read_csv(ARTIFACT_DIR / "model_comparison_results.csv", index_col=0).reset_index()
    comparison = comparison.rename(columns={"index": "model"})
    return comparison


@st.cache_data(show_spinner=False)
def load_roi_calculator() -> pd.DataFrame:
    path = BASE_DIR / "retention_flow_outputs" / "retentionflow_roi_calculator.csv"
    if not path.exists():
        return pd.DataFrame()
    roi = pd.read_csv(path)
    roi.columns = roi.columns.str.strip()
    return roi


@st.cache_data(show_spinner=False)
def load_customer_cloning() -> pd.DataFrame:
    path = BASE_DIR / "retention_flow_outputs" / "retentionflow_customer_cloning.csv"
    if not path.exists():
        return pd.DataFrame()
    cloning = pd.read_csv(path)
    cloning.columns = cloning.columns.str.strip()
    return cloning


def parse_retained_neighbors(value) -> list[int]:
    if pd.isna(value):
        return []
    return [int(match) for match in re.findall(r"\d+", str(value))]


def merge_customer_profile(base: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    if not len(base) or "customer_id" not in base.columns:
        return base
    profile_cols = [
        "customer_id",
        "customer_segment",
        "city_tier",
        "gender",
        "loyalty_program_member",
        "tenure_months",
        "customer_lifetime_value",
    ]
    profile_cols = [col for col in profile_cols if col in profile.columns]
    if len(profile_cols) <= 1:
        return base
    keep = profile[profile_cols].drop_duplicates("customer_id")
    return base.merge(keep, on="customer_id", how="left", suffixes=("", "_profile"))


def roi_priority_table(roi: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    result = merge_customer_profile(roi.copy(), profile)
    for col in ["expected_recovery", "intervention_cost", "roi", "churn_probability", "customer_lifetime_value", "net_benefit"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)
    if "customer_lifetime_value" not in result.columns:
        result["customer_lifetime_value"] = result.get("customer_value", 0)
    if "net_benefit" not in result.columns and {"expected_recovery", "intervention_cost"}.issubset(result.columns):
        result["net_benefit"] = result["expected_recovery"] - result["intervention_cost"]
    if "roi_category" not in result.columns and "roi" in result.columns:
        result["roi_category"] = pd.cut(
            result["roi"],
            bins=[-np.inf, 0, 10, 20, np.inf],
            labels=["Negative ROI", "Low ROI", "Medium ROI", "High ROI"],
        ).astype(str)
    return result


def customer_profile_metrics(customer_ids: list[int], profile: pd.DataFrame) -> pd.Series:
    if not customer_ids or "customer_id" not in profile.columns:
        return pd.Series(dtype=float)
    rows = profile[profile["customer_id"].isin(customer_ids)]
    metrics = {
        "Digital activity": ["total_digital_logins", "mobile_app_login_count", "website_login_count"],
        "Transactions": ["monthly_transaction_count", "total_trans_count", "upi_transaction_count"],
        "Product holding": ["number_of_products", "total_products_held"],
        "Balance": ["avg_monthly_balance", "current_balance"],
        "Loan exposure": ["loan_outstanding_amount", "emi_amount"],
    }
    values = {}
    for label, columns in metrics.items():
        available = [col for col in columns if col in rows.columns]
        values[label] = float(rows[available].mean(numeric_only=True).mean()) if available and len(rows) else 0.0
    return pd.Series(values)


def normalize_for_radar(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for col in columns:
        max_value = result[col].max()
        result[col] = 0 if max_value == 0 or pd.isna(max_value) else result[col] / max_value
    return result


@st.cache_data(show_spinner=False)
def feature_baseline_medians(reference: pd.DataFrame) -> pd.Series:
    return add_model_features(reference).median(numeric_only=True)


def top_model_drivers(row: pd.Series, limit: int = 8, engineered_row: pd.Series | None = None) -> pd.DataFrame:
    importance = load_feature_importance()
    engineered = engineered_row if engineered_row is not None else add_model_features(pd.DataFrame([row])).iloc[0]
    baseline = feature_baseline_medians(MODEL_REFERENCE)
    importance_lookup = importance.set_index("feature")["mean_abs_shap"].to_dict()
    rows = []
    for feature in importance["feature"]:
        if feature not in engineered.index:
            continue
        value = engineered[feature]
        model_importance = round(float(importance_lookup.get(feature, 0)), 4)
        if not pd.api.types.is_number(value):
            rows.append(
                {
                    "feature": feature,
                    "input_value": str(value),
                    "dataset_median": "-",
                    "model_importance": model_importance,
                }
            )
            continue
        base = float(baseline.get(feature, 0))
        if abs(float(value) - base) > 1e-9:
            rows.append(
                {
                    "feature": feature,
                    "input_value": round(float(value), 4),
                    "dataset_median": round(base, 4),
                    "model_importance": model_importance,
                }
            )
        if len(rows) >= limit:
            break
    result = pd.DataFrame(rows)
    if "input_value" in result.columns:
        result["input_value"] = result["input_value"].astype(str)
    return result


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = PROCESSED_DATA_DIR / "train_cleaned.csv"
    test_path = PROCESSED_DATA_DIR / "test_cleaned.csv"
    df = pd.read_csv(train_path if train_path.exists() else RAW_DATA_DIR / "ChurnZero_dataset_v1.csv")
    df_test = pd.read_csv(test_path if test_path.exists() else RAW_DATA_DIR / "ChurnZero_test_v1.csv")
    df.columns = df.columns.str.strip()
    df_test.columns = df_test.columns.str.strip()

    def attach_notebook_scoring(frame: pd.DataFrame, scoring_filename: str, scope_label: str) -> pd.DataFrame:
        frame = frame.copy()
        present_flags = [col for col in PRODUCT_FLAGS if col in frame.columns]
        frame["total_products_held"] = frame[present_flags].sum(axis=1) if present_flags else 0
        frame["single_product_flag"] = (frame["total_products_held"] == 1).astype(int)
        scoring_path = BASE_DIR / "retention_flow_outputs" / scoring_filename
        if scoring_path.exists():
            scoring = pd.read_csv(scoring_path)
            scoring.columns = scoring.columns.str.strip()
            scoring_cols = [
                "customer_id",
                "churn_probability",
                "churn_prediction",
                "customer_value",
                "wealth_tier",
                "priority_score",
                "revenue_at_risk",
                "recoverable_revenue",
                "net_benefit",
                "recommended_action",
                "recommendation_reason",
                "expected_impact",
                "recommended_channel",
                "num_risk_factors",
            ]
            scoring_cols = [col for col in scoring_cols if col in scoring.columns]
            frame = frame.merge(scoring[scoring_cols], on="customer_id", how="left")
            frame = frame.rename(
                columns={
                    "churn_probability": "notebook_churn_probability",
                    "churn_prediction": "notebook_churn_prediction",
                }
            )
        if "customer_value" not in frame.columns:
            engineered = add_model_features(frame)
            frame["customer_value"] = engineered["customer_value"]
            frame["wealth_tier"] = engineered["wealth_tier"]
        for column in ["revenue_at_risk", "recoverable_revenue", "net_benefit", "priority_score", "num_risk_factors"]:
            if column not in frame.columns:
                frame[column] = 0.0
        frame["data_scope"] = scope_label
        frame["age_band"] = pd.cut(
            frame["age"],
            bins=[25, 35, 45, 55, 65, 75],
            labels=["26-35", "36-45", "46-55", "56-65", "66-75"],
        )
        frame["income_band_calc"] = pd.cut(
            frame["annual_income"],
            bins=[0, 50_000, 100_000, 150_000, 200_000, 300_000],
            labels=["<50K", "50-100K", "100-150K", "150-200K", "200K+"],
        )
        return frame

    df = attach_notebook_scoring(df, "retentionflow_train_scoring.csv", "Training Data")
    df_test = attach_notebook_scoring(df_test, "ChurnZero_RetentionFlow_FullTestScoring.csv", "Test Data")
    return df, df_test

df_train, df_test = load_data()
MODEL_REFERENCE = df_train.copy()
train_probabilities = pd.to_numeric(
    df_train.get("notebook_churn_probability", pd.Series(dtype=float)),
    errors="coerce",
).dropna()
if train_probabilities.empty:
    train_probabilities = pd.Series(predict_churn(df_train, MODEL_REFERENCE))
MODEL_PROBABILITY_CUTS = tuple(np.quantile(train_probabilities, [0.50, 0.75, 0.90]))


with st.sidebar:
    st.markdown('<div class="wordmark">Retention<span>Flow</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-subtitle">CHURNZERO INTELLIGENCE SYSTEM</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sidebar-label">Data Scope</div>', unsafe_allow_html=True)
    data_scope = st.selectbox("Data Scope", ["Training Data", "Test Data", "Full Dataset"], label_visibility="collapsed")
    if data_scope == "Training Data":
        df = df_train.copy()
    elif data_scope == "Test Data":
        df = df_test.copy()
    else:
        df = pd.concat([df_train, df_test], ignore_index=True, sort=False)

    st.markdown('<div class="sidebar-label">Region</div>', unsafe_allow_html=True)
    sel_region = st.selectbox(
        "Region",
        ["All"] + sorted(df["region"].dropna().unique().tolist()),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">City Tier</div>', unsafe_allow_html=True)
    sel_tier = st.selectbox(
        "City Tier",
        ["All"] + sorted(df["city_tier"].dropna().unique().tolist()),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Customer Segment</div>', unsafe_allow_html=True)
    sel_segment = st.selectbox(
        "Customer Segment",
        ["All"] + sorted(df["customer_segment"].dropna().unique().tolist()),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Card Category</div>', unsafe_allow_html=True)
    sel_card = st.selectbox(
        "Card Category",
        ["All"] + sorted(df["card_category"].dropna().unique().tolist()),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Gender</div>', unsafe_allow_html=True)
    sel_gender = st.selectbox(
        "Gender",
        ["All"] + sorted(df["gender"].dropna().unique().tolist()),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Risk Tier</div>', unsafe_allow_html=True)
    sel_risk = st.selectbox(
        "Risk Tier",
        ["All", "Critical", "High", "Medium", "Low"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Tenure Range</div>', unsafe_allow_html=True)
    tenure_range = st.slider(
        "Tenure Range",
        int(df["tenure_months"].min()),
        int(df["tenure_months"].max()),
        (int(df["tenure_months"].min()), int(df["tenure_months"].max())),
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-label">Age Range</div>', unsafe_allow_html=True)
    age_range = st.slider(
        "Age Range",
        int(df["age"].min()),
        int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max())),
        label_visibility="collapsed",
    )


fdf = df.copy()
if sel_region != "All":
    fdf = fdf[fdf["region"] == sel_region]
if sel_tier != "All":
    fdf = fdf[fdf["city_tier"] == sel_tier]
if sel_segment != "All":
    fdf = fdf[fdf["customer_segment"] == sel_segment]
if sel_card != "All":
    fdf = fdf[fdf["card_category"] == sel_card]
if sel_gender != "All":
    fdf = fdf[fdf["gender"] == sel_gender]
fdf = fdf[fdf["tenure_months"].between(*tenure_range)]
fdf = fdf[fdf["age"].between(*age_range)]

total_customers = len(fdf)
model_threshold = float(MODEL_CONFIG["threshold"])
fn_cost = float(MODEL_CONFIG["fn_cost"])
fp_cost = float(MODEL_CONFIG["fp_cost"])
if total_customers:
    fdf = fdf.copy()
    prediction_lookup = cached_filtered_predictions(data_scope, tuple(fdf["customer_id"].astype(int).tolist()))
    fdf = fdf.merge(prediction_lookup, on="customer_id", how="left")
    if fdf["ml_churn_probability"].isna().any():
        missing_mask = fdf["ml_churn_probability"].isna()
        fdf.loc[missing_mask, "ml_churn_probability"] = predict_churn(fdf.loc[missing_mask], MODEL_REFERENCE)
    fdf["ml_predicted_churn"] = (fdf["ml_churn_probability"] >= model_threshold).astype(int)
    fdf["risk_tier"] = fdf["ml_churn_probability"].apply(lambda prob: model_tier(float(prob))[0].replace(" Risk", ""))
    if sel_risk != "All":
        fdf = fdf[fdf["risk_tier"] == sel_risk]
        total_customers = len(fdf)
else:
    fdf = fdf.copy()
    fdf["ml_churn_probability"] = []
    fdf["ml_predicted_churn"] = []
churners = int(fdf["ml_predicted_churn"].sum()) if total_customers else 0
churn_rate = fdf["ml_churn_probability"].mean() if total_customers else 0
avg_clv = fdf["customer_lifetime_value"].mean() if total_customers else 0
avg_sat = fdf["satisfaction_score"].mean() if total_customers else 0
avg_nps = fdf["nps_score"].mean() if total_customers else 0
revenue_at_risk = fdf["revenue_at_risk"].sum() if total_customers and "revenue_at_risk" in fdf.columns else 0


st.markdown(
    f"""
    <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:1rem;flex-wrap:wrap;">
        <div>
            <div style="display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;">
                <div style="font-size:2rem;font-weight:900;color:#0F172A;">RetentionFlow</div>
                <div style="font-size:1rem;font-weight:800;color:#0EA5E9;letter-spacing:0.04em;">
                    CHURNZERO INTELLIGENCE PLATFORM
                </div>
            </div>
            <div style="color:#64748B;font-size:0.9rem;margin-bottom:0.9rem;">
                Bank churn prediction, risk scoring, revenue recovery, and retention planning.
            </div>
        </div>
        {github_source_link("homepage-source-link")}
    </div>
    """,
    unsafe_allow_html=True,
)


def kpi(col, label: str, value: str, sub: str, color: str = ACCENT) -> None:
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{color};">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi(k1, "Total Customers", f"{total_customers:,}", "Filtered dataset", ACCENT)
kpi(k2, "Predicted Churners", f"{churners:,}", f"{churn_rate:.1%} churn rate", DANGER)
kpi(k3, "Avg CLV", f"INR {avg_clv:,.0f}", "Customer lifetime value", ACCENT_2)
kpi(k4, "Avg Satisfaction", f"{avg_sat:.2f} / 5", "Service score", SUCCESS)
kpi(k5, "Avg NPS", f"{avg_nps:.1f}", "Net promoter score", WARNING)
kpi(k6, "Revenue at Risk", f"INR {revenue_at_risk / 1e6:.1f}M", "Notebook scoring output", DANGER)

st.markdown("<br>", unsafe_allow_html=True)

tab_overview, tab_eda, tab_features, tab_model, tab_risk, tab_revenue, tab_roi, tab_recs, tab_cloning, tab_survival, tab_predict, tab_exports = st.tabs(
    [
        "Overview",
        "EDA",
        "Explainability",
        "Model Quality",
        "Risk Scoring",
        "Revenue Recovery",
        "ROI Calculator",
        "Recommendations",
        "Customer Cloning",
        "Survival Analysis",
        "Churn Predictor",
        "Data Exports",
    ]
)


with tab_overview:
    st.markdown('<div class="section-header">Executive Intelligence Snapshot</div>', unsafe_allow_html=True)
    risk_counts = (
        fdf["risk_tier"].value_counts().rename_axis("risk_tier").reset_index(name="customers")
        if len(fdf)
        else pd.DataFrame(columns=["risk_tier", "customers"])
    )
    segment_value = (
        fdf.groupby("customer_segment", dropna=False)
        .agg(
            customers=("customer_id", "count"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            avg_churn_probability=("ml_churn_probability", "mean"),
        )
        .reset_index()
        .sort_values("revenue_at_risk", ascending=False)
        if len(fdf)
        else pd.DataFrame(columns=["customer_segment", "customers", "revenue_at_risk", "avg_churn_probability"])
    )

    high_risk_count = int(fdf["ml_predicted_churn"].sum()) if len(fdf) else 0
    recoverable_total = fdf["recoverable_revenue"].sum() if "recoverable_revenue" in fdf.columns and len(fdf) else 0
    net_total = fdf["net_benefit"].sum() if "net_benefit" in fdf.columns and len(fdf) else 0
    top_segment = segment_value.iloc[0]["customer_segment"] if len(segment_value) else "N/A"
    top_segment_revenue = segment_value.iloc[0]["revenue_at_risk"] if len(segment_value) else 0
    recovery_rate = recoverable_total / revenue_at_risk if revenue_at_risk else 0
    top_action = (
        fdf.groupby("recommended_action", dropna=False)["recoverable_revenue"].sum().sort_values(ascending=False).index[0]
        if "recommended_action" in fdf.columns and "recoverable_revenue" in fdf.columns and len(fdf)
        else "N/A"
    )

    o1, o2, o3, o4 = st.columns(4)
    kpi(o1, "High-Risk Customers", f"{high_risk_count:,}", "Predicted by notebook model", DANGER)
    kpi(o2, "Recoverable Revenue", f"INR {recoverable_total / 1e6:.2f}M", "Targeted action upside", SUCCESS)
    kpi(o3, "Net Benefit", f"INR {net_total / 1e6:.2f}M", "Notebook ROI output", ACCENT)
    kpi(o4, "Top Risk Segment", str(top_segment), "Highest revenue exposure", WARNING)

    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    signal_cards = [
        (
            s1,
            "Portfolio Concentration",
            f"{high_risk_count:,} customers need attention",
            f"{(high_risk_count / len(fdf)):.1%} of filtered customers" if len(fdf) else "No filtered customers",
        ),
        (
            s2,
            "Largest Exposure Segment",
            str(top_segment),
            f"INR {top_segment_revenue / 1e6:.2f}M revenue at risk",
        ),
        (
            s3,
            "Best Recovery Route",
            str(top_action),
            f"{recovery_rate:.1%} of exposed revenue is recoverable",
        ),
    ]
    for container, label, value, sub in signal_cards:
        container.markdown(
            f"""
            <div class="overview-signal">
                <div class="overview-signal-label">{label}</div>
                <div class="overview-signal-value">{value}</div>
                <div class="overview-signal-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Executive Signal Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="overview-chart-band">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([0.85, 1.1, 1.05])
    with c1:
        risk_order = ["Critical", "High", "Medium", "Low"]
        risk_counts["risk_tier"] = risk_counts["risk_tier"].astype(str)
        risk_counts["risk_order"] = risk_counts["risk_tier"].map({name: i for i, name in enumerate(risk_order)}).fillna(99)
        risk_counts = risk_counts.sort_values("risk_order")
        fig = px.pie(
            risk_counts,
            names="risk_tier",
            values="customers",
            hole=0.52,
            color="risk_tier",
            color_discrete_map=RISK_TIER_COLORS,
            category_orders={"risk_tier": ["Low", "Medium", "High", "Critical"]},
            title="Risk Tier Mix",
        )
        fig = style_light_chart(fig, height=360)
        fig.update_traces(
            textinfo="percent",
            textposition="inside",
            textfont_color=TEXT,
            marker=dict(line=dict(color="rgba(255,255,255,0)", width=0)),
            hovertemplate="<b>%{label}</b><br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.12,
                xanchor="center",
                x=0.5,
                font=dict(color=TEXT, size=12),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(
            segment_value.head(8),
            x="revenue_at_risk",
            y="customer_segment",
            orientation="h",
            color="avg_churn_probability",
            color_continuous_scale=["#FEE2E2", "#FB7185", "#E11D48", "#7F1D1D"],
            text="customers",
            title="Revenue at Risk by Segment",
        )
        fig = style_light_chart(fig, height=360)
        fig.update_layout(yaxis_title="", xaxis_title="Revenue at risk")
        fig.update_traces(textposition="outside", marker_line_color="#FFFFFF", marker_line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        action_value = (
            fdf.groupby("recommended_action", dropna=False)
            .agg(
                customers=("customer_id", "count"),
                recoverable_revenue=("recoverable_revenue", "sum"),
                net_benefit=("net_benefit", "sum"),
            )
            .reset_index()
            .sort_values("recoverable_revenue", ascending=False)
            if {"recommended_action", "recoverable_revenue", "net_benefit"}.issubset(fdf.columns) and len(fdf)
            else pd.DataFrame(columns=["recommended_action", "customers", "recoverable_revenue", "net_benefit"])
        )
        fig = px.bar(
            action_value.head(6),
            x="recoverable_revenue",
            y="recommended_action",
            orientation="h",
            color="net_benefit",
            color_continuous_scale=["#DBEAFE", "#38BDF8", "#0EA5E9", "#123B6D"],
            text="customers",
            title="Recoverable Revenue by Action",
        )
        fig = style_light_chart(fig, height=360)
        fig.update_layout(yaxis_title="", xaxis_title="Recoverable revenue")
        fig.update_traces(textposition="outside", marker_line_color="#FFFFFF", marker_line_width=1.5)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Portfolio and Business Impact Charts</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        dashboard_chart("risk_distribution.png", "Portfolio breakdown by risk tier.")
    with c2:
        dashboard_chart("revenue_recovery.png", "Recoverable revenue from retention actions.")
    with c3:
        dashboard_chart("roi_calculator.png", "ROI opportunity from targeted retention.")


with tab_eda:
    st.markdown('<div class="section-header">Notebook EDA Overview</div>', unsafe_allow_html=True)
    dashboard_chart(
        "eda_overview.png",
        "Dataset-level exploratory summary from the notebook.",
    )
    st.divider()
    st.markdown('<div class="section-header">Categorical Churn Patterns</div>', unsafe_allow_html=True)
    dashboard_chart(
        "eda_categorical_churn_rate.png",
        "Categorical churn-rate differences used for early business hypotheses.",
    )
    st.divider()
    st.markdown('<div class="section-header">Filtered Segment Snapshot</div>', unsafe_allow_html=True)
    segment_summary = (
        fdf.groupby(["customer_segment", "risk_tier"], observed=False)
        .agg(
            customers=("customer_id", "count"),
            avg_churn_probability=("ml_churn_probability", "mean"),
            revenue_at_risk=("revenue_at_risk", "sum"),
        )
        .reset_index()
        .sort_values(["revenue_at_risk", "customers"], ascending=False)
    )
    white_table(segment_summary.head(50), height=420)


with tab_features:
    st.markdown('<div class="section-header">Explainability Command Center</div>', unsafe_allow_html=True)

    numeric = fdf.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [col for col in numeric if col not in {"churn", "customer_id", "ml_predicted_churn", "ml_churn_probability"}]
    corr = fdf[numeric + ["ml_churn_probability"]].corr(numeric_only=True)["ml_churn_probability"].drop("ml_churn_probability").dropna()
    corr_df = corr.abs().sort_values(ascending=False).head(25).reset_index()
    corr_df.columns = ["feature", "absolute_correlation"]
    corr_df["direction"] = np.where(corr[corr_df["feature"]].values > 0, "Raises model probability", "Lowers model probability")

    f1, f2, f3 = st.columns(3)
    top_driver = corr_df.iloc[0]["feature"].replace("_", " ").title() if len(corr_df) else "N/A"
    kpi(f1, "Top Live Driver", top_driver, "Filtered cohort correlation", DANGER)
    kpi(f2, "Drivers Tracked", f"{len(corr_df):,}", "Numeric features ranked", ACCENT)
    kpi(f3, "Avg Model Risk", f"{churn_rate:.1%}", "Current filtered portfolio", WARNING)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        fig = px.bar(
            corr_df.head(12).sort_values("absolute_correlation"),
            x="absolute_correlation",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={
                "Raises model probability": DANGER,
                "Lowers model probability": SUCCESS,
            },
            title="Filtered Cohort Churn Drivers",
        )
        fig = style_light_chart(fig, height=420)
        fig.update_layout(xaxis_title="Absolute correlation", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Feature Summary</div>', unsafe_allow_html=True)
        white_table(corr_df.head(12), height=420)

    st.divider()
    st.markdown('<div class="section-header">Notebook Explainability Artifacts</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        dashboard_chart("shap_summary.png", "Global SHAP summary from the notebook.")
    with c2:
        dashboard_chart("feature_importance.png", "Simplified feature importance view.")
    dashboard_chart("shap_bar.png", "Mean SHAP contribution ranking.")

with tab_model:
    risk_df = fdf.copy()
    if "priority_score" not in risk_df.columns:
        risk_df["priority_score"] = (risk_df["ml_churn_probability"] * 100).round(1)
    risk_df["model_cost_exposure"] = risk_df["revenue_at_risk"] if "revenue_at_risk" in risk_df.columns else 0

    st.markdown('<div class="section-header">Model Quality and Business Threshold</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        dashboard_chart("pr_curve.png", "Precision-recall performance for the imbalanced churn problem.")
    with c2:
        dashboard_chart("threshold_optimisation.png", "Threshold selected from business cost tradeoffs.")
    st.divider()
    st.markdown('<div class="section-header">Notebook Model Benchmark</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        dashboard_chart("model_comparison.png")
    with c4:
        dashboard_chart("calibration.png")

    model_comparison = load_model_comparison()
    white_table(model_comparison, height=260)


with tab_risk:
    risk_df = fdf.copy()
    if "priority_score" not in risk_df.columns:
        risk_df["priority_score"] = (risk_df["ml_churn_probability"] * 100).round(1)
    risk_df["model_cost_exposure"] = risk_df["revenue_at_risk"] if "revenue_at_risk" in risk_df.columns else 0

    st.markdown('<div class="section-header">Risk Scoring Workbench</div>', unsafe_allow_html=True)
    tier_summary = (
        risk_df.groupby("risk_tier", dropna=False)
        .agg(
            customers=("customer_id", "count"),
            avg_churn_probability=("ml_churn_probability", "mean"),
            revenue_at_risk=("model_cost_exposure", "sum"),
            avg_priority=("priority_score", "mean"),
        )
        .reset_index()
        .sort_values("avg_priority", ascending=False)
        if len(risk_df)
        else pd.DataFrame(columns=["risk_tier", "customers", "avg_churn_probability", "revenue_at_risk", "avg_priority"])
    )

    r1, r2, r3 = st.columns([0.9, 1.1, 1])
    with r1:
        dashboard_chart("risk_distribution.png", "Notebook risk distribution.")
    with r2:
        fig = px.bar(
            tier_summary,
            x="risk_tier",
            y="revenue_at_risk",
            color="risk_tier",
            color_discrete_sequence=RISK_RED_SCALE,
            title="Revenue Exposure by Risk Tier",
        )
        fig = style_light_chart(fig, height=330)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Revenue at risk")
        st.plotly_chart(fig, use_container_width=True)
    with r3:
        st.markdown('<div class="section-header">Risk Tier Summary</div>', unsafe_allow_html=True)
        white_table(tier_summary, height=330)

    st.divider()
    st.markdown('<div class="section-header">High-Priority Customer Watchlist</div>', unsafe_allow_html=True)
    watchlist = risk_df.sort_values("priority_score", ascending=False).head(100)
    white_table(
        watchlist[
            [
                "customer_id",
                "region",
                "customer_segment",
                "card_category",
                "risk_tier",
                "priority_score",
                "ml_churn_probability",
                "customer_lifetime_value",
                "model_cost_exposure",
                "satisfaction_score",
                "total_complaints",
            ]
        ],
        height=420,
    )


with tab_recs:
    rec_df = fdf[fdf["ml_predicted_churn"] == 1].copy()

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Actionable Customers", f"{len(rec_df):,}", "Predicted by ML model", DANGER)
    kpi(c2, "Model Threshold", f"{model_threshold:.2f}", "Loaded from config", WARNING)
    kpi(c3, "Revenue at Risk", f"INR {rec_df['revenue_at_risk'].sum() / 1e6:.1f}M" if len(rec_df) and "revenue_at_risk" in rec_df.columns else "INR 0.0M", "Notebook scoring output", SUCCESS)

    st.markdown("<br>", unsafe_allow_html=True)
    if len(rec_df):
        dashboard_chart("recommendations.png")

        st.markdown('<div class="section-header">Customer Action Table</div>', unsafe_allow_html=True)
        recommendation_columns = [
            "customer_id",
            "customer_segment",
            "card_category",
            "risk_tier",
            "ml_churn_probability",
            "priority_score",
            "revenue_at_risk",
            "recoverable_revenue",
            "recommended_action",
            "recommendation_reason",
            "expected_impact",
        ]
        recommendation_columns = [column for column in recommendation_columns if column in rec_df.columns]
        white_table(
            rec_df[recommendation_columns].head(250),
            height=420,
        )
    else:
        st.info("No customers crossed the trained model threshold in the current filters.")


with tab_revenue:
    roi_table = load_roi_calculator()
    if len(roi_table):
        revenue_df = roi_priority_table(roi_table, MODEL_REFERENCE)
        filtered_ids = set(fdf["customer_id"].astype(int).tolist()) if "customer_id" in fdf.columns and len(fdf) else set()
        filtered_revenue = revenue_df[revenue_df["customer_id"].isin(filtered_ids)] if filtered_ids else revenue_df
        if len(filtered_revenue):
            revenue_df = filtered_revenue

        st.markdown('<div class="section-header">Revenue Recovery</div>', unsafe_allow_html=True)
        total_risk = revenue_df["revenue_at_risk"].sum()
        recoverable = revenue_df["recoverable_revenue"].sum() if "recoverable_revenue" in revenue_df.columns else revenue_df["expected_recovery"].sum()
        net = revenue_df["net_benefit"].sum()
        avg_recovery = revenue_df["expected_recovery"].mean()

        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "Revenue at Risk", f"INR {total_risk / 1e6:.2f}M", "Notebook scoring output", DANGER)
        kpi(c2, "Recoverable Revenue", f"INR {recoverable / 1e6:.2f}M", "Notebook recovery estimate", SUCCESS)
        kpi(c3, "Net Benefit", f"INR {net / 1e6:.2f}M", "After intervention cost", ACCENT)
        kpi(c4, "Avg Recovery / Customer", f"INR {avg_recovery:,.0f}", "Expected recovery", WARNING)

        st.markdown("<br>", unsafe_allow_html=True)
        action_summary = (
            revenue_df.groupby("recommended_action", dropna=False)
            .agg(
                customers=("customer_id", "count"),
                revenue_at_risk=("revenue_at_risk", "sum"),
                recoverable_revenue=("recoverable_revenue", "sum"),
                net_benefit=("net_benefit", "sum"),
            )
            .reset_index()
            .sort_values("net_benefit", ascending=False)
        )
        c1, c2 = st.columns(2)
        with c1:
            dashboard_chart("revenue_recovery.png", "Notebook revenue recovery artifact.")
        with c2:
            fig = px.bar(
                action_summary,
                x="net_benefit",
                y="recommended_action",
                orientation="h",
                color="recoverable_revenue",
                color_continuous_scale="Greens",
                text="customers",
                title="Recovery Value by Recommended Action",
            )
            fig = style_light_chart(fig, height=360)
            fig.update_layout(xaxis_title="Net benefit", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown('<div class="section-header">Revenue Recovery Summary</div>', unsafe_allow_html=True)
        white_table(action_summary, height=320)
    else:
        st.info("ROI calculator output was not found.")


with tab_roi:
    roi_table = load_roi_calculator()
    st.markdown('<div class="section-header">ROI Calculator</div>', unsafe_allow_html=True)
    st.caption("Operational retention prioritization from notebook-generated ROI outputs only.")
    if len(roi_table):
        roi_df = roi_priority_table(roi_table, MODEL_REFERENCE)
        filtered_ids = set(fdf["customer_id"].astype(int).tolist()) if "customer_id" in fdf.columns and len(fdf) else set()
        filtered_roi = roi_df[roi_df["customer_id"].isin(filtered_ids)] if filtered_ids else roi_df
        if len(filtered_roi):
            roi_df = filtered_roi

        f1, f2, f3 = st.columns(3)
        with f1:
            risk_filter = st.selectbox("Risk Tier", ["All"] + sorted(roi_df["risk_tier"].dropna().astype(str).unique().tolist()), key="roi_risk_tier")
            segment_filter = st.selectbox(
                "Customer Segment",
                ["All"] + sorted(roi_df.get("customer_segment", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
                key="roi_customer_segment",
            )
        with f2:
            wealth_filter = st.selectbox("Wealth Tier", ["All"] + sorted(roi_df["wealth_tier"].dropna().astype(str).unique().tolist()), key="roi_wealth_tier")
            min_probability = st.slider("Minimum Churn Probability", 0.0, 1.0, 0.0, 0.01, key="roi_min_probability")
        with f3:
            roi_category_filter = st.selectbox("ROI Category", ["All"] + sorted(roi_df["roi_category"].dropna().astype(str).unique().tolist()), key="roi_category")
            customer_search = st.text_input("Search by Customer ID", key="roi_customer_search")

        roi_filtered = roi_df.copy()
        if risk_filter != "All":
            roi_filtered = roi_filtered[roi_filtered["risk_tier"].astype(str) == risk_filter]
        if wealth_filter != "All":
            roi_filtered = roi_filtered[roi_filtered["wealth_tier"].astype(str) == wealth_filter]
        if roi_category_filter != "All":
            roi_filtered = roi_filtered[roi_filtered["roi_category"].astype(str) == roi_category_filter]
        if segment_filter != "All" and "customer_segment" in roi_filtered.columns:
            roi_filtered = roi_filtered[roi_filtered["customer_segment"].astype(str) == segment_filter]
        roi_filtered = roi_filtered[roi_filtered["churn_probability"] >= min_probability]
        if customer_search.strip():
            roi_filtered = roi_filtered[roi_filtered["customer_id"].astype(str).str.contains(customer_search.strip(), case=False, na=False)]

        total_expected = roi_filtered["expected_recovery"].sum()
        spend = roi_filtered["intervention_cost"].sum()
        net = roi_filtered["net_benefit"].sum()
        portfolio_roi = net / spend if spend else np.nan
        avg_roi = roi_filtered["roi"].mean() if len(roi_filtered) else 0
        high_roi_count = int((roi_filtered["roi"] >= 20).sum()) if len(roi_filtered) else 0

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpi(k1, "Total Expected Recovery", f"INR {total_expected / 1e6:.2f}M", "Filtered customers", SUCCESS)
        kpi(k2, "Total Retention Spend", f"INR {spend / 1e6:.2f}M", "Notebook intervention cost", WARNING)
        kpi(k3, "Net Benefit", f"INR {net / 1e6:.2f}M", "Recovery minus spend", ACCENT)
        kpi(k4, "Overall Portfolio ROI", "N/A" if pd.isna(portfolio_roi) else f"{portfolio_roi:.1f}x", "Filtered portfolio", ACCENT_2)
        kpi(k5, "Average ROI per Customer", f"{avg_roi:.1f}x", "Mean notebook ROI", SUCCESS)
        kpi(k6, "High ROI Customer Count", f"{high_roi_count:,}", "ROI >= 20x", DANGER)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            st.markdown('<div class="section-header">Top ROI Customers Table</div>', unsafe_allow_html=True)
            table_cols = [
                "customer_id",
                "churn_probability",
                "expected_recovery",
                "intervention_cost",
                "roi",
                "risk_tier",
                "wealth_tier",
                "recommended_action",
            ]
            table_cols = [col for col in table_cols if col in roi_filtered.columns]
            top_roi = roi_filtered.sort_values(["roi", "expected_recovery", "churn_probability"], ascending=False)
            white_table(top_roi[table_cols].rename(columns={"roi": "ROI"}).head(300), height=430)
            st.download_button(
                "Download Filtered ROI CSV",
                data=top_roi[table_cols].to_csv(index=False),
                file_name="retentionflow_roi_priorities.csv",
                mime="text/csv",
            )
        with c2:
            st.markdown('<div class="section-header">Dynamic Customer Recommendation</div>', unsafe_allow_html=True)
            customer_options = top_roi["customer_id"].astype(int).tolist() if len(top_roi) else []
            if customer_options:
                selected_customer = st.selectbox("Selected Customer", customer_options, key="roi_selected_customer")
                selected = top_roi[top_roi["customer_id"] == selected_customer].iloc[0]
                c21, c22 = st.columns(2)
                kpi(c21, "Churn Probability", format_probability(float(selected["churn_probability"])), str(selected.get("risk_tier", "Risk tier")), DANGER)
                kpi(c22, "Expected Recovery", f"INR {selected.get('expected_recovery', 0):,.0f}", "Notebook output", SUCCESS)
                c23, c24 = st.columns(2)
                kpi(c23, "ROI", f"{selected.get('roi', 0):.1f}x", str(selected.get("roi_category", "ROI category")), ACCENT)
                kpi(c24, "Priority Score", f"{selected.get('priority_score', 0):.2f}", "Notebook scoring", WARNING)
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <b>Recommended action:</b> {selected.get("recommended_action", "N/A")}<br>
                        <b>Revenue at risk:</b> INR {selected.get("revenue_at_risk", 0):,.0f}<br>
                        <b>Wealth tier:</b> {selected.get("wealth_tier", "N/A")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No customers match the selected ROI filters.")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            distribution = roi_filtered["roi_category"].value_counts().rename_axis("ROI Category").reset_index(name="customers")
            fig = px.bar(
                distribution,
                x="ROI Category",
                y="customers",
                color="ROI Category",
                color_discrete_sequence=[DANGER, WARNING, ACCENT, SUCCESS],
                title="ROI Distribution",
            )
            fig = style_light_chart(fig, height=360)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(
                roi_filtered,
                x="intervention_cost",
                y="expected_recovery",
                size="customer_lifetime_value",
                color="risk_tier",
                color_discrete_sequence=RISK_RED_SCALE,
                hover_data=["customer_id", "roi", "wealth_tier"],
                title="Retention Spend vs Recovery",
            )
            fig = style_light_chart(fig, height=360)
            fig.update_layout(xaxis_title="Intervention cost", yaxis_title="Expected recovery")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown('<div class="section-header">Wealth Tier ROI Breakdown</div>', unsafe_allow_html=True)
        wealth_summary = (
            roi_filtered.groupby("wealth_tier", dropna=False)
            .agg(
                total_customers=("customer_id", "count"),
                total_recovery=("expected_recovery", "sum"),
                avg_ROI=("roi", "mean"),
                avg_churn_probability=("churn_probability", "mean"),
            )
            .reset_index()
            .sort_values("total_recovery", ascending=False)
        )
        white_table(wealth_summary, height=330)
    else:
        st.info("ROI calculator output was not found.")


with tab_cloning:
    cloning = load_customer_cloning()
    st.markdown('<div class="section-header">Customer Cloning</div>', unsafe_allow_html=True)
    st.caption("Interactive customer intelligence explorer using precomputed clone_df neighbour matches only.")
    if len(cloning):
        clone_df = merge_customer_profile(cloning.copy(), MODEL_REFERENCE)
        roi_lookup = load_roi_calculator()
        if len(roi_lookup):
            roi_cols = [
                "customer_id",
                "risk_tier",
                "wealth_tier",
                "revenue_at_risk",
                "recommended_action",
                "priority_score",
            ]
            roi_cols = [col for col in roi_cols if col in roi_lookup.columns]
            clone_df = clone_df.merge(roi_lookup[roi_cols].drop_duplicates("customer_id"), on="customer_id", how="left", suffixes=("", "_roi"))
            for col in ["risk_tier", "wealth_tier"]:
                roi_col = f"{col}_roi"
                if roi_col in clone_df.columns:
                    clone_df[col] = clone_df[col].fillna(clone_df[roi_col]) if col in clone_df.columns else clone_df[roi_col]

        for col in ["gap", "churn_probability", "churner_value", "avg_retained_value", "revenue_at_risk", "priority_score"]:
            if col in clone_df.columns:
                clone_df[col] = pd.to_numeric(clone_df[col], errors="coerce")

        f1, f2, f3 = st.columns(3)
        with f1:
            customer_filter_text = st.text_input("Customer ID", key="clone_customer_search")
            risk_filter = st.selectbox("Risk Tier", ["All"] + sorted(clone_df.get("risk_tier", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()), key="clone_risk")
        with f2:
            wealth_filter = st.selectbox("Wealth Tier", ["All"] + sorted(clone_df.get("wealth_tier", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()), key="clone_wealth")
            gap_filter = st.selectbox("Gap Feature", ["All"] + sorted(clone_df["top_gap_feature"].dropna().astype(str).unique().tolist()), key="clone_gap")
        with f3:
            min_prob = float(clone_df["churn_probability"].min()) if clone_df["churn_probability"].notna().any() else 0.0
            max_prob = float(clone_df["churn_probability"].max()) if clone_df["churn_probability"].notna().any() else 1.0
            probability_range = st.slider(
                "Churn Probability Range",
                0.0,
                1.0,
                (max(0.0, min_prob), min(1.0, max_prob)),
                0.01,
                key="clone_probability_range",
            )

        filtered_clone = clone_df.copy()
        if customer_filter_text.strip():
            filtered_clone = filtered_clone[filtered_clone["customer_id"].astype(str).str.contains(customer_filter_text.strip(), case=False, na=False)]
        if risk_filter != "All" and "risk_tier" in filtered_clone.columns:
            filtered_clone = filtered_clone[filtered_clone["risk_tier"].astype(str) == risk_filter]
        if wealth_filter != "All" and "wealth_tier" in filtered_clone.columns:
            filtered_clone = filtered_clone[filtered_clone["wealth_tier"].astype(str) == wealth_filter]
        if gap_filter != "All":
            filtered_clone = filtered_clone[filtered_clone["top_gap_feature"].astype(str) == gap_filter]
        filtered_clone = filtered_clone[
            filtered_clone["churn_probability"].between(probability_range[0], probability_range[1], inclusive="both")
        ]

        top_gap_feature = (
            str(filtered_clone["top_gap_feature"].mode().iloc[0]).replace("_", " ").title()
            if len(filtered_clone) and filtered_clone["top_gap_feature"].notna().any()
            else "N/A"
        )
        selected_customer_ids = filtered_clone["customer_id"].dropna().astype(int).tolist()
        retained_ids_all = []
        for value in filtered_clone.get("similar_retained_ids", pd.Series(dtype=str)).head(100):
            retained_ids_all.extend(parse_retained_neighbors(value))

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpi(c1, "Average Behavioral Gap", f"{filtered_clone['gap'].mean():.2f}" if len(filtered_clone) else "0.00", "Precomputed notebook gap", WARNING)
        kpi(c2, "Most Common Gap Feature", top_gap_feature, "Highest recurring gap", DANGER)
        kpi(c3, "Similarity Match Strength", "Top-5", "Notebook retained neighbours", ACCENT)
        kpi(c4, "Revenue Linked", f"INR {filtered_clone.get('revenue_at_risk', pd.Series(dtype=float)).sum() / 1e6:.2f}M", "Selected customers", SUCCESS)
        kpi(c5, "Avg Retained Customer Score", f"{filtered_clone['avg_retained_value'].mean():.2f}" if len(filtered_clone) else "0.00", "Neighbour average value", ACCENT_2)
        kpi(c6, "Top Retention Opportunity", top_gap_feature, "Close this behavior gap", WARNING)
        st.markdown("<br>", unsafe_allow_html=True)

        if len(filtered_clone):
            customer_options = filtered_clone["customer_id"].dropna().astype(int).tolist()
            selected_customer = st.selectbox("Selected Customer for Comparison", customer_options, key="clone_selected_customer")
            selected_row = filtered_clone[filtered_clone["customer_id"] == selected_customer].iloc[0]
            retained_ids = parse_retained_neighbors(selected_row.get("similar_retained_ids"))

            selected_metrics = customer_profile_metrics([int(selected_customer)], MODEL_REFERENCE)
            retained_metrics = customer_profile_metrics(retained_ids, MODEL_REFERENCE)
            comparison = pd.DataFrame(
                {
                    "metric": selected_metrics.index,
                    "selected_churner": selected_metrics.values,
                    "retained_neighbour_average": retained_metrics.reindex(selected_metrics.index).fillna(0).values,
                }
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown('<div class="section-header">Customer Comparison Panel</div>', unsafe_allow_html=True)
                melted = comparison.melt("metric", var_name="profile", value_name="value")
                fig = px.bar(
                    melted,
                    x="metric",
                    y="value",
                    color="profile",
                    barmode="group",
                    color_discrete_sequence=[DANGER, SUCCESS],
                    title="Selected Churner vs Retained Neighbour Average",
                )
                fig = style_light_chart(fig, height=390)
                fig.update_layout(xaxis_title="", yaxis_title="Profile value")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown('<div class="section-header">Behavioral Radar Chart</div>', unsafe_allow_html=True)
                radar = normalize_for_radar(comparison.copy(), ["selected_churner", "retained_neighbour_average"])
                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=radar["selected_churner"],
                        theta=radar["metric"],
                        fill="toself",
                        name="Selected churner",
                        line_color=DANGER,
                    )
                )
                fig.add_trace(
                    go.Scatterpolar(
                        r=radar["retained_neighbour_average"],
                        theta=radar["metric"],
                        fill="toself",
                        name="Retained neighbours",
                        line_color=SUCCESS,
                    )
                )
                fig.update_layout(
                    template="plotly_white",
                    height=390,
                    paper_bgcolor="#FFFFFF",
                    polar=dict(bgcolor="#FFFFFF", radialaxis=dict(visible=True, range=[0, 1], gridcolor="#E6EEF8")),
                    font=dict(color=TEXT, family="Raleway, sans-serif"),
                    margin=dict(l=24, r=24, t=40, b=24),
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            c1, c2 = st.columns([1, 1])
            with c1:
                gap_view = filtered_clone.sort_values("gap", ascending=False).head(15)
                gap_melted = gap_view.melt(
                    id_vars=["customer_id", "top_gap_feature", "gap"],
                    value_vars=["churner_value", "avg_retained_value"],
                    var_name="profile",
                    value_name="value",
                )
                fig = px.bar(
                    gap_melted,
                    x="value",
                    y="top_gap_feature",
                    color="profile",
                    orientation="h",
                    barmode="group",
                    color_discrete_sequence=[DANGER, SUCCESS],
                    title="Feature Gap Visualization",
                )
                fig = style_light_chart(fig, height=420)
                fig.update_layout(xaxis_title="Feature value", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                recommendation_text = {
                    "digital": "Increase mobile and net-banking engagement with guided activation, app nudges, and digital service support.",
                    "transaction": "Trigger a relationship-manager outreach focused on transaction activity, recurring payments, and primary-bank usage.",
                    "product": "Offer product-bundle recommendations to deepen relationship breadth.",
                    "balance": "Use balance protection, savings goals, or premium advisory outreach to stabilize wallet share.",
                    "loan": "Review loan servicing, EMI support, and repayment friction before churn risk peaks.",
                }
                feature_name = str(selected_row.get("top_gap_feature", "")).lower()
                matched_key = next((key for key in recommendation_text if key in feature_name), "transaction")
                st.markdown('<div class="section-header">Dynamic Recommendation Engine</div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <b>Selected customer:</b> {int(selected_customer)}<br>
                        <b>Top behavioral gap:</b> {str(selected_row.get("top_gap_feature", "N/A")).replace("_", " ").title()}<br>
                        <b>Recommended action:</b> {recommendation_text[matched_key]}<br>
                        <b>Business interpretation:</b> Customers retained successfully tend to show stronger engagement and broader banking relationships than this selected churn-risk customer.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown('<div class="section-header">Similar Retained Customers Table</div>', unsafe_allow_html=True)
            retained_table = MODEL_REFERENCE[MODEL_REFERENCE["customer_id"].isin(retained_ids)].copy()
            if len(retained_table):
                retained_table["retained_customer_id"] = retained_table["customer_id"]
                retained_table["similarity_score"] = "Notebook ranked match"
                retained_table["churn_probability"] = retained_table.get("ml_churn_probability", retained_table.get("notebook_churn_probability", 0))
                retained_table["retained_status"] = np.where(retained_table.get("churn", 0) == 0, "Retained", "Historical churn")
                display_cols = ["retained_customer_id", "similarity_score", "churn_probability", "wealth_tier", "retained_status"]
                display_cols = [col for col in display_cols if col in retained_table.columns]
                white_table(retained_table[display_cols], height=260)
            else:
                st.info("The selected retained-neighbour IDs were not found in the loaded profile data.")

            st.divider()
            st.markdown('<div class="section-header">Filtered Clone Output</div>', unsafe_allow_html=True)
            white_table(filtered_clone.head(400), height=380)
        else:
            st.info("No clone recommendations match the selected filters.")
    else:
        st.info("Customer cloning output was not found.")


with tab_survival:
    st.markdown('<div class="section-header">Survival Analysis</div>', unsafe_allow_html=True)
    st.caption("Interactive churn lifecycle analytics using notebook-generated survival artifact plus existing cohort scoring outputs.")
    survival_df = fdf.copy()
    if len(survival_df):
        f1, f2, f3 = st.columns(3)
        with f1:
            segment_filter = st.selectbox("Customer Segment", ["All"] + sorted(survival_df["customer_segment"].dropna().astype(str).unique().tolist()), key="survival_segment")
            gender_filter = st.selectbox("Gender", ["All"] + sorted(survival_df["gender"].dropna().astype(str).unique().tolist()), key="survival_gender")
        with f2:
            city_filter = st.selectbox("City Tier", ["All"] + sorted(survival_df["city_tier"].dropna().astype(str).unique().tolist()), key="survival_city")
            wealth_filter = st.selectbox("Wealth Tier", ["All"] + sorted(survival_df.get("wealth_tier", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()), key="survival_wealth")
        with f3:
            loyalty_options = ["All"] + sorted(survival_df["loyalty_program_member"].dropna().astype(str).unique().tolist()) if "loyalty_program_member" in survival_df.columns else ["All"]
            loyalty_filter = st.selectbox("Loyalty Member", loyalty_options, key="survival_loyalty")
            tenure_min = int(survival_df["tenure_months"].min()) if "tenure_months" in survival_df.columns else 0
            tenure_max = int(survival_df["tenure_months"].max()) if "tenure_months" in survival_df.columns else 60
            tenure_range = st.slider("Tenure Range", tenure_min, tenure_max, (tenure_min, tenure_max), key="survival_tenure")

        if segment_filter != "All":
            survival_df = survival_df[survival_df["customer_segment"].astype(str) == segment_filter]
        if city_filter != "All":
            survival_df = survival_df[survival_df["city_tier"].astype(str) == city_filter]
        if gender_filter != "All":
            survival_df = survival_df[survival_df["gender"].astype(str) == gender_filter]
        if wealth_filter != "All" and "wealth_tier" in survival_df.columns:
            survival_df = survival_df[survival_df["wealth_tier"].astype(str) == wealth_filter]
        if loyalty_filter != "All" and "loyalty_program_member" in survival_df.columns:
            survival_df = survival_df[survival_df["loyalty_program_member"].astype(str) == loyalty_filter]
        if "tenure_months" in survival_df.columns:
            survival_df = survival_df[survival_df["tenure_months"].between(tenure_range[0], tenure_range[1], inclusive="both")]

    if len(survival_df):
        survival_df["tenure_bucket"] = pd.cut(
            survival_df["tenure_months"],
            bins=[0, 6, 12, 24, 36, 48, 60, np.inf],
            labels=["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "60+"],
            include_lowest=True,
        )
        segment_risk = (
            survival_df.groupby("customer_segment", dropna=False)
            .agg(avg_churn_probability=("ml_churn_probability", "mean"), customers=("customer_id", "count"), median_tenure=("tenure_months", "median"))
            .reset_index()
            .sort_values("avg_churn_probability", ascending=False)
        )
        city_risk = (
            survival_df.groupby("city_tier", dropna=False)
            .agg(avg_churn_probability=("ml_churn_probability", "mean"), customers=("customer_id", "count"))
            .reset_index()
            .sort_values("avg_churn_probability", ascending=False)
        )
        early_churn = (
            survival_df.groupby("tenure_bucket", observed=False)
            .agg(churn_intensity=("ml_churn_probability", "mean"), customers=("customer_id", "count"))
            .reset_index()
            .sort_values("churn_intensity", ascending=False)
        )
        top_segment = str(segment_risk.iloc[0]["customer_segment"]) if len(segment_risk) else "N/A"
        top_city = str(city_risk.iloc[0]["city_tier"]) if len(city_risk) else "N/A"
        earliest_bucket = str(early_churn.iloc[0]["tenure_bucket"]) if len(early_churn) else "N/A"
        retention_12 = 1 - survival_df.loc[survival_df["tenure_months"] <= 12, "ml_churn_probability"].mean()
        survival_24 = 1 - survival_df.loc[survival_df["tenure_months"] <= 24, "ml_churn_probability"].mean()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpi(c1, "Median Survival Time", "56 mo", "Notebook Kaplan-Meier output", ACCENT)
        kpi(c2, "12-Month Retention Rate", format_probability(float(retention_12)) if not pd.isna(retention_12) else "N/A", "Filtered cohort proxy", SUCCESS)
        kpi(c3, "Highest Risk Segment", top_segment, "Filtered cohort", DANGER)
        kpi(c4, "Highest Risk City Tier", top_city, "Filtered cohort", WARNING)
        kpi(c5, "Earliest Churn Cohort", earliest_bucket, "Highest churn intensity", DANGER)
        kpi(c6, "Survival Probability at 24 Months", format_probability(float(survival_24)) if not pd.isna(survival_24) else "N/A", "Filtered cohort proxy", ACCENT_2)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            dashboard_chart("survival_analysis.png", "Notebook-generated Kaplan-Meier survival artifact.")
        with c2:
            cohort_line = (
                survival_df.groupby(["tenure_bucket", "customer_segment"], observed=False)
                .agg(churn_intensity=("ml_churn_probability", "mean"), customers=("customer_id", "count"))
                .reset_index()
            )
            fig = px.line(
                cohort_line,
                x="tenure_bucket",
                y="churn_intensity",
                color="customer_segment",
                markers=True,
                title="Interactive Cohort Churn Timing",
            )
            fig = style_light_chart(fig, height=380)
            fig.update_layout(xaxis_title="Tenure bucket (months)", yaxis_title="Churn intensity")
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Cohort Comparison Panel</div>', unsafe_allow_html=True)
            compare_options = segment_risk["customer_segment"].astype(str).tolist()
            if len(compare_options) >= 2:
                a = st.selectbox("Segment A", compare_options, index=0, key="survival_segment_a")
                b = st.selectbox("Segment B", compare_options, index=min(1, len(compare_options) - 1), key="survival_segment_b")
                a_row = segment_risk[segment_risk["customer_segment"].astype(str) == a].iloc[0]
                b_row = segment_risk[segment_risk["customer_segment"].astype(str) == b].iloc[0]
                diff = float(a_row["avg_churn_probability"] - b_row["avg_churn_probability"])
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <b>{a}</b> median tenure: {a_row["median_tenure"]:.0f} months<br>
                        <b>{b}</b> median tenure: {b_row["median_tenure"]:.0f} months<br>
                        <b>Risk difference:</b> {diff:+.1%}<br>
                        <b>Interpretation:</b> The higher-risk cohort should receive earlier retention contact and lifecycle monitoring.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("At least two customer segments are needed for comparison.")
        with c2:
            st.markdown('<div class="section-header">Churn Timing Insight Cards</div>', unsafe_allow_html=True)
            insight_1 = f"{earliest_bucket} month tenure bucket shows the highest filtered churn intensity."
            insight_2 = f"{top_city} customers currently carry the highest city-tier risk."
            insight_3 = f"{top_segment} is the segment most exposed in the selected lifecycle view."
            for insight in [insight_1, insight_2, insight_3]:
                st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns([0.9, 1.1])
        with c1:
            st.markdown('<div class="section-header">Survival Milestone Table</div>', unsafe_allow_html=True)
            milestones = []
            for month in [6, 12, 24, 36, 48]:
                cohort = survival_df[survival_df["tenure_months"] <= month]
                milestones.append(
                    {
                        "milestone_month": month,
                        "customers": len(cohort),
                        "survival_probability_proxy": 1 - cohort["ml_churn_probability"].mean() if len(cohort) else np.nan,
                        "avg_churn_intensity": cohort["ml_churn_probability"].mean() if len(cohort) else np.nan,
                    }
                )
            white_table(pd.DataFrame(milestones), height=330)
        with c2:
            heatmap = (
                survival_df.pivot_table(
                    index="customer_segment",
                    columns="tenure_bucket",
                    values="ml_churn_probability",
                    aggfunc="mean",
                    observed=False,
                )
                .fillna(0)
            )
            fig = px.imshow(
                heatmap,
                color_continuous_scale="Reds",
                aspect="auto",
                title="Segment Risk Heatmap",
            )
            fig = style_light_chart(fig, height=330)
            fig.update_layout(xaxis_title="Tenure bucket", yaxis_title="Customer segment")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No customers match the selected survival filters.")


with tab_exports:
    st.markdown('<div class="section-header">Notebook Scoring Outputs</div>', unsafe_allow_html=True)
    export_tabs = st.tabs(["Training Scoring", "Test Scoring", "Predictions", "ROI Output", "Customer Cloning"])
    with export_tabs[0]:
        white_table(pd.read_csv(BASE_DIR / "retention_flow_outputs" / "retentionflow_train_scoring.csv").head(500), height=520)
    with export_tabs[1]:
        white_table(pd.read_csv(BASE_DIR / "retention_flow_outputs" / "ChurnZero_RetentionFlow_FullTestScoring.csv").head(500), height=520)
    with export_tabs[2]:
        white_table(pd.read_csv(BASE_DIR / "retention_flow_outputs" / "ChurnZero_RetentionFlow_Predictions.csv").head(500), height=520)
    with export_tabs[3]:
        white_table(load_roi_calculator().head(500), height=520)
    with export_tabs[4]:
        white_table(load_customer_cloning(), height=520)


with tab_predict:
    enable_enter_to_next_field()
    predictor_source = MODEL_REFERENCE.copy()

    def labelize(column: str) -> str:
        replacements = {
            "nps": "NPS",
            "clv": "CLV",
            "upi": "UPI",
            "emi": "EMI",
            "q4": "Q4",
            "q1": "Q1",
        }
        label = column.replace("_", " ").title()
        for old, new in replacements.items():
            label = label.replace(old.title(), new)
        return label

    def field_options(column: str) -> list[str]:
        return sorted(predictor_source[column].dropna().astype(str).unique().tolist())

    def model_default_value(column: str):
        series = predictor_source[column]
        if pd.api.types.is_numeric_dtype(series):
            return float(series.median()) if series.notna().any() else 0
        mode = series.dropna().astype(str).mode()
        return mode.iloc[0] if len(mode) else ""

    def render_predictor_field(container, column: str, value):
        key = f"predictor_random_v4_{column}"
        label = f"{labelize(column)} :red[*]"
        if column in {"has_loan", "loan_default_history"}:
            return container.selectbox(label, ["No", "Yes"], index=None, placeholder="Select option", key=key)
        series = predictor_source[column]
        if column.endswith("_flag") or set(series.dropna().unique()).issubset({0, 1}):
            answer = container.selectbox(label, ["No", "Yes"], index=None, placeholder="Select option", key=key)
            if answer is None:
                return None
            return 1 if answer == "Yes" else 0
        if pd.api.types.is_numeric_dtype(series):
            return container.text_input(
                label,
                value="",
                placeholder=f"Enter {labelize(column).lower()}",
                key=key,
            )
        options = field_options(column)
        return container.selectbox(
            label,
            options,
            index=None,
            placeholder=f"Select {labelize(column).lower()}",
            format_func=lambda option: str(option).title(),
            key=key,
        )

    predictor_groups = {
        "Customer Information": [
            "age",
            "gender",
            "occupation_type",
            "annual_income",
            "city_tier",
            "region",
        ],
        "Account Relationship": [
            "tenure_months",
            "number_of_products",
            "customer_lifetime_value",
            "loyalty_program_member",
            "referral_count",
            "last_contacted_days",
            "relationship_manager_assigned",
        ],
        "Transactions and Payments": [
            "avg_monthly_balance",
            "current_balance",
            "monthly_transaction_count",
            "monthly_transaction_value",
            "cash_withdrawal_count",
            "upi_transaction_count",
            "total_trans_count",
        ],
        "Credit and Loan": [
            "has_loan",
            "loan_default_history",
        ],
        "Digital Banking": [
            "total_digital_logins",
        ],
    }

    excluded_predictor_columns = {
        "customer_id",
        "churn",
        "notebook_churn_probability",
        "notebook_churn_prediction",
        "ml_churn_probability",
        "ml_predicted_churn",
        "risk_tier",
        "age_band",
        "income_band_calc",
        "priority_score",
        "revenue_at_risk",
        "recoverable_revenue",
        "net_benefit",
        "recommended_action",
        "recommendation_reason",
        "expected_impact",
        "recommended_channel",
        "num_risk_factors",
    }
    virtual_predictor_columns = {"has_loan", "loan_default_history"}
    grouped_columns = [
        column
        for columns in predictor_groups.values()
        for column in columns
        if (column in predictor_source.columns or column in virtual_predictor_columns) and column not in excluded_predictor_columns
    ]
    input_row = {column: None for column in grouped_columns}
    input_row["customer_id"] = int(predictor_source["customer_id"].max()) + 1
    numeric_predictor_columns = [
        column
        for column in grouped_columns
        if column in predictor_source.columns
        if pd.api.types.is_numeric_dtype(predictor_source[column])
        and not column.endswith("_flag")
        and not set(predictor_source[column].dropna().unique()).issubset({0, 1})
    ]
    st.markdown('<div class="predictor-shell">', unsafe_allow_html=True)
    with st.container():
        for group_index, (group_name, columns) in enumerate(predictor_groups.items()):
            visible_columns = [column for column in columns if column in input_row]
            if not visible_columns:
                continue
            if group_index:
                st.markdown('<div class="predictor-divider"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="predictor-section">{group_name}</div>', unsafe_allow_html=True)
            for start in range(0, len(visible_columns), 2):
                cols = st.columns(2)
                for container, column in zip(cols, visible_columns[start:start + 2]):
                    input_row[column] = render_predictor_field(container, column, input_row[column])

        submitted = st.button("Calculate Churn Risk", use_container_width=True, key="manual_churn_predictor_submit_v4")
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        missing_columns = [
            column
            for column in grouped_columns
            if input_row.get(column) is None or (isinstance(input_row.get(column), str) and input_row.get(column).strip() == "")
        ]
        if missing_columns:
            st.markdown(
                '<div class="predictor-note">Complete all required fields before prediction.</div>',
                unsafe_allow_html=True,
            )
        else:
            invalid_numeric = []
            parsed_values = {}
            for column in numeric_predictor_columns:
                raw_value = input_row.get(column)
                try:
                    parsed = float(str(raw_value).replace(",", "").strip())
                except ValueError:
                    invalid_numeric.append(column)
                    continue
                if column == "age" and parsed >= 100:
                    invalid_numeric.append(column)
                    continue
                parsed_values[column] = parsed

            if invalid_numeric:
                st.markdown(
                    '<div class="predictor-note">Enter valid numbers in the highlighted numeric fields.</div>',
                    unsafe_allow_html=True,
                )
                st.stop()

            input_row.update(parsed_values)
            model_row = {
                column: model_default_value(column)
                for column in predictor_source.columns
                if column not in excluded_predictor_columns
            }
            model_row.update(input_row)
            model_row["customer_id"] = input_row["customer_id"]
            if input_row.get("has_loan") == "Yes":
                model_row["has_loan"] = 1
                model_row["personal_loan_flag"] = 1
                if float(model_row.get("loan_outstanding_amount", 0) or 0) <= 0:
                    model_row["loan_outstanding_amount"] = float(predictor_source["loan_outstanding_amount"].quantile(0.75))
                if float(model_row.get("emi_amount", 0) or 0) <= 0:
                    model_row["emi_amount"] = float(predictor_source["emi_amount"].quantile(0.75))
            else:
                for loan_flag in ["personal_loan_flag", "home_loan_flag", "auto_loan_flag"]:
                    if loan_flag in model_row:
                        model_row[loan_flag] = 0
                model_row["has_loan"] = 0
                model_row["loan_outstanding_amount"] = 0.0
                model_row["emi_amount"] = 0.0
                model_row["emi_payment_delay_count"] = 0
            if input_row.get("loan_default_history") == "Yes":
                model_row["loan_default_history"] = 1
                model_row["loan_default_risk_score"] = float(predictor_source["loan_default_risk_score"].quantile(0.85))
                model_row["emi_payment_delay_count"] = max(int(model_row.get("emi_payment_delay_count", 0) or 0), 1)
            else:
                model_row["loan_default_history"] = 0
                model_row["loan_default_risk_score"] = float(predictor_source["loan_default_risk_score"].quantile(0.25))

            st.markdown('<div class="predictor-processing">Please wait...</div>', unsafe_allow_html=True)
            progress = st.progress(0, text="Please wait...")
            model_row_json = json.dumps(model_row, sort_keys=True, default=str)
            progress.progress(35, text="Please wait...")
            with st.spinner("Please wait..."):
                score, drivers = cached_manual_prediction(model_row_json)
            progress.progress(100, text="Done")
            progress.empty()

            tier, tier_color = model_tier(score)
            predicted_label = "Churn" if score >= model_threshold else "Retain"
            cost_exposure = score * fn_cost + (1 - score) * fp_cost
            score_text = format_probability(score)

            r1, r2, r3 = st.columns(3)
            kpi(r1, "Churn Risk Score", score_text, tier, tier_color)
            kpi(r2, "Model Decision", predicted_label, f"Threshold {model_threshold:.0%}", SUCCESS if predicted_label == "Retain" else DANGER)
            kpi(r3, "Cost Exposure", f"INR {cost_exposure:,.0f}", f"FN {fn_cost:,.0f} / FP {fp_cost:,.0f}", ACCENT)

            if len(drivers):
                st.markdown('<div class="section-header">Primary Risk Drivers</div>', unsafe_allow_html=True)
                white_table(drivers, height=300)

            st.markdown(
                f"""
                <div class="risk-meter">
                    <div class="risk-meter-label">ML Churn Probability: {score_text}</div>
                    <div class="risk-meter-track">
                        <div class="risk-meter-fill" style="width:{score * 100:.2f}%;background:{tier_color};"></div>
                    </div>
                    <div class="risk-meter-scale">
                        <span>0%</span>
                        <span>Threshold {model_threshold:.0%}</span>
                        <span>100%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


st.divider()
st.markdown(
    f"""
    <div style="text-align:center;color:#64748B;font-size:0.72rem;padding-bottom:0.5rem;">
        RetentionFlow - ChurnZero Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True,
)
