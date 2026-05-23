from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "Churnzero_data" / "CZ_raw"
PROCESSED_DATA_DIR = BASE_DIR / "Churnzero_data" / "CZ_processed"
ARTIFACT_DIR = BASE_DIR / "retention_flow_outputs" / "artifacts"
CHART_DIR = BASE_DIR / "retention_flow_outputs" / "charts"

ACCENT = "#0EA5E9"
ACCENT_2 = "#6366F1"
DANGER = "#E11D48"
WARNING = "#D97706"
SUCCESS = "#059669"
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

    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: white !important;
        border: 0 !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
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
        background: rgba(14, 165, 233, 0.10);
        border: 1px solid rgba(14, 165, 233, 0.30);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        color: var(--text);
        font-weight: 800;
        padding: 0.75rem 0.9rem;
        margin: 1rem auto 0;
        max-width: 1120px;
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


def notebook_chart(filename: str, caption: str | None = None) -> None:
    path = CHART_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Missing notebook chart: {filename}")


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


def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(ARTIFACT_DIR / "shap_feature_importance.csv")


def load_model_comparison() -> pd.DataFrame:
    comparison = pd.read_csv(ARTIFACT_DIR / "model_comparison_results.csv", index_col=0).reset_index()
    comparison = comparison.rename(columns={"index": "model"})
    return comparison


def top_model_drivers(row: pd.Series, limit: int = 8) -> pd.DataFrame:
    importance = load_feature_importance()
    engineered = add_model_features(pd.DataFrame([row])).iloc[0]
    baseline_frame = globals().get("MODEL_REFERENCE", globals().get("df", pd.DataFrame([row])))
    baseline = add_model_features(baseline_frame).median(numeric_only=True)
    rows = []
    for feature in importance["feature"]:
        if feature not in engineered.index:
            continue
        value = engineered[feature]
        model_importance = round(float(importance.loc[importance["feature"] == feature, "mean_abs_shap"].iloc[0]), 4)
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
MODEL_PROBABILITY_CUTS = tuple(np.quantile(predict_churn(df_train, MODEL_REFERENCE), [0.50, 0.75, 0.90]))


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
    fdf["ml_churn_probability"] = predict_churn(fdf, MODEL_REFERENCE)
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
    """
    <div style="display:flex;align-items:baseline;gap:1rem;">
        <div style="font-size:2rem;font-weight:900;color:#0F172A;">RetentionFlow</div>
        <div style="font-size:1rem;font-weight:800;color:#0EA5E9;letter-spacing:0.04em;">
            CHURNZERO INTELLIGENCE PLATFORM
        </div>
    </div>
    <div style="color:#64748B;font-size:0.9rem;margin-bottom:0.9rem;">
        Bank churn prediction, risk scoring, revenue recovery, and retention planning.
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

tab_overview, tab_eda, tab_features, tab_risk, tab_recs, tab_revenue, tab_predict = st.tabs(
    [
        "Overview",
        "EDA Deep-Dive",
        "Feature Insights",
        "Risk Scoring",
        "Recommendations",
        "Revenue Recovery",
        "Churn Predictor",
    ]
)


with tab_overview:
    st.markdown('<div class="section-header">Notebook Overview</div>', unsafe_allow_html=True)
    notebook_chart("eda_overview.png")
    st.divider()
    notebook_chart("risk_distribution.png")


with tab_eda:
    st.markdown('<div class="section-header">Numeric Distributions</div>', unsafe_allow_html=True)
    notebook_chart("eda_numeric.png")
    st.divider()
    st.markdown('<div class="section-header">Categorical Distributions</div>', unsafe_allow_html=True)
    notebook_chart("eda_categorical.png")
    st.divider()
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    notebook_chart("eda_correlation.png")


with tab_features:
    st.markdown('<div class="section-header">SHAP Feature Importance</div>', unsafe_allow_html=True)
    c0, c00 = st.columns(2)
    with c0:
        notebook_chart("shap_bar.png")
    with c00:
        notebook_chart("shap_summary.png")
    st.divider()
    st.markdown('<div class="section-header">SHAP Customer Waterfall</div>', unsafe_allow_html=True)
    notebook_chart("shap_waterfall.png")
    numeric = fdf.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [col for col in numeric if col not in {"churn", "customer_id", "ml_predicted_churn", "ml_churn_probability"}]
    corr = fdf[numeric + ["ml_churn_probability"]].corr(numeric_only=True)["ml_churn_probability"].drop("ml_churn_probability").dropna()
    corr_df = corr.abs().sort_values(ascending=False).head(25).reset_index()
    corr_df.columns = ["feature", "absolute_correlation"]
    corr_df["direction"] = np.where(corr[corr_df["feature"]].values > 0, "Raises model probability", "Lowers model probability")

    st.divider()
    st.markdown('<div class="section-header">Feature Summary</div>', unsafe_allow_html=True)
    white_table(corr_df.head(15), height=430)

    st.divider()
    st.markdown('<div class="section-header">Trained Model Benchmark</div>', unsafe_allow_html=True)
    model_comparison = load_model_comparison()
    notebook_chart("model_comparison.png")
    white_table(model_comparison, height=280)


with tab_risk:
    risk_df = fdf.copy()
    if "priority_score" not in risk_df.columns:
        risk_df["priority_score"] = (risk_df["ml_churn_probability"] * 100).round(1)
    risk_df["model_cost_exposure"] = risk_df["revenue_at_risk"] if "revenue_at_risk" in risk_df.columns else 0

    st.markdown('<div class="section-header">Risk Distribution and Threshold</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        notebook_chart("risk_distribution.png")
    with c2:
        notebook_chart("threshold_optimization.png")
    st.divider()
    st.markdown('<div class="section-header">Model Evaluation</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        notebook_chart("confusion_matrix.png")
    with c4:
        notebook_chart("pr_curves.png")

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
        notebook_chart("recommendations.png")

        st.markdown('<div class="section-header">Customer Action Table</div>', unsafe_allow_html=True)
        white_table(
            rec_df[
                [
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
            ].head(250),
            height=420,
        )
    else:
        st.info("No customers crossed the trained model threshold in the current filters.")


with tab_revenue:
    churner_df = fdf[fdf["ml_predicted_churn"] == 1].copy()
    total_risk = churner_df["revenue_at_risk"].sum() if "revenue_at_risk" in churner_df.columns else 0
    recoverable = churner_df["recoverable_revenue"].sum() if "recoverable_revenue" in churner_df.columns else 0
    net = churner_df["net_benefit"].sum() if "net_benefit" in churner_df.columns else 0
    spend = recoverable - net
    roi = net / (spend + 1)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Revenue at Risk", f"INR {total_risk / 1e6:.1f}M", "Notebook scoring output", DANGER)
    kpi(c2, "Recoverable Revenue", f"INR {recoverable / 1e6:.1f}M", "Notebook scoring output", SUCCESS)
    kpi(c3, "Retention Spend", f"INR {spend / 1e6:.2f}M", "Notebook scoring output", WARNING)
    kpi(c4, "Net Benefit Ratio", f"{roi:.1f}x", "Notebook scoring output", ACCENT)

    st.markdown("<br>", unsafe_allow_html=True)
    notebook_chart("revenue_recovery.png")


with tab_predict:
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

        submitted = st.button("Calculate Churn Risk", width="stretch", key="manual_churn_predictor_submit_v4")
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
            input_frame = pd.DataFrame([model_row])
            score = float(predict_churn(input_frame, MODEL_REFERENCE)[0])
            tier, tier_color = model_tier(score)
            predicted_label = "Churn" if score >= model_threshold else "Retain"
            cost_exposure = score * fn_cost + (1 - score) * fp_cost
            drivers = top_model_drivers(pd.Series(model_row), 12)
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
    """
    <div style="text-align:center;color:#64748B;font-size:0.72rem;padding-bottom:0.5rem;">
        RetentionFlow - ChurnZero Intelligence Platform
    </div>
    """,
    unsafe_allow_html=True,
)
