"""
Streamlit dashboard — Churn Risk Predictor.

Tabs:
  1. Risk Scorer      — score a single account, explain why, get actions
  2. Portfolio View   — score a book of business, prioritize, download
  3. Model Performance — honest evaluation metrics, plots, limitations
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explain import (
    load_artifact,
    explain_prediction,
    make_recommendation,
    format_for_csm,
    clean_feature_name,
    _clean_name,
    _parse_feat,
)
from preprocess import RAW_DATA_PATH


def display_image(image_path: str, width_percent: int = 80) -> None:
    import base64
    from pathlib import Path as _Path

    p = _Path(image_path)
    if p.exists():
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        ext = p.suffix.replace(".", "")
        st.markdown(
            f'<img src="data:image/{ext};base64,{data}" '
            f'style="width:{width_percent}%;display:block;'
            f'margin:auto;border-radius:8px;">',
            unsafe_allow_html=True,
        )
    else:
        st.warning(f"Image not found: {image_path}")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Risk Predictor",
    page_icon="📊",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter+Tight'
    ':ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# Design system
# Primary   #000000  main actions, key UI elements
# Secondary #82A1E8  supporting elements, hover
# BG        #FFFFFF  main canvas
# Surface   #CFCFCF  cards, elevated elements
# Text pri  #000000  headings, body
# Text sec  #82A1E8  captions, placeholders
# Base unit 4px

st.markdown("""
<style>
/* ── Base & font ── */
html, body, [class*="css"] {
    font-family: 'Inter Tight', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-weight: 400;
}
.stApp { background: #FFFFFF !important; }

/* ── Streamlit chrome ── */
#MainMenu, footer, [data-testid="stHeader"] { display: none !important; }
.stDeployButton                             { display: none !important; }
.block-container                            { padding-top: 1.5rem !important; }

/* ── Headings ── */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    font-family: 'Inter Tight', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: #000000 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Inter Tight', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    border: 1.5px solid #000000 !important;
    background: #FFFFFF !important;
    color: #000000 !important;
    transition: background 0.15s, color 0.15s !important;
}
.stButton > button:hover {
    background: #82A1E8 !important;
    border-color: #82A1E8 !important;
    color: #FFFFFF !important;
}
button[data-testid="baseButton-primary"] {
    background: #000000 !important;
    color: #FFFFFF !important;
    border-color: #000000 !important;
}
button[data-testid="baseButton-primary"]:hover {
    background: #82A1E8 !important;
    border-color: #82A1E8 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid #CFCFCF !important;
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'Inter Tight', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #000000 !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 20px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    font-weight: 700 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: #000000 !important;
    height: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top: 24px !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #CFCFCF !important;
    border-radius: 4px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Inter Tight', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #82A1E8 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Inter Tight', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: #000000 !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #CFCFCF !important;
    border-radius: 4px !important;
    background: #FFFFFF !important;
    margin-bottom: 4px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Inter Tight', sans-serif !important;
    font-weight: 600 !important;
    color: #000000 !important;
}

/* ── Inputs ── */
.stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label,
.stToggle label, .stCheckbox label {
    font-family: 'Inter Tight', sans-serif !important;
    font-size: 0.85rem !important;
    color: #000000 !important;
}

/* ── Captions ── */
.stCaption, [data-testid="stCaptionContainer"] p {
    color: #82A1E8 !important;
    font-family: 'Inter Tight', sans-serif !important;
}

/* ── Code blocks ── */
[data-testid="stCode"] {
    background: #F5F5F5 !important;
    border-radius: 4px !important;
    border: 1px solid #CFCFCF !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    font-family: 'Inter Tight', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 4px !important;
    background: #FFFFFF !important;
    border: 1.5px solid #CFCFCF !important;
    color: #000000 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #82A1E8 !important;
    border-color: #82A1E8 !important;
    color: #FFFFFF !important;
}

/* ── Dividers ── */
hr { border-color: #CFCFCF !important; border-top-width: 1px !important; }

/* ══ Custom HTML components ══ */

.risk-card {
    background: #CFCFCF; border-radius: 4px;
    padding: 20px 24px; margin-bottom: 16px;
}
.factor-card {
    background: #CFCFCF; border-left: 3px solid #000000;
    border-radius: 4px; padding: 12px 16px; margin-bottom: 8px;
}
.factor-card.protective { border-left-color: #82A1E8; }
.action-card {
    background: #CFCFCF; border-radius: 4px;
    padding: 12px 16px; margin-bottom: 8px;
}

/* Risk badges */
.badge-low {
    background: #82A1E8; color: #FFFFFF; padding: 4px 12px;
    border-radius: 2px; font-weight: 700; font-size: .75rem;
    display: inline-block; font-family: 'Inter Tight', sans-serif;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.badge-medium {
    background: #CFCFCF; color: #000000; padding: 4px 12px;
    border-radius: 2px; font-weight: 700; font-size: .75rem;
    display: inline-block; font-family: 'Inter Tight', sans-serif;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.badge-high {
    background: #000000; color: #FFFFFF; padding: 4px 12px;
    border-radius: 2px; font-weight: 700; font-size: .75rem;
    display: inline-block; font-family: 'Inter Tight', sans-serif;
    letter-spacing: 0.06em; text-transform: uppercase;
}

/* Timeframe labels */
.tf-label {
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .08em; color: #82A1E8; margin-bottom: 4px;
    font-family: 'Inter Tight', sans-serif;
}
.tf-this-week    { color: #000000; }
.tf-this-month   { color: #82A1E8; }
.tf-this-quarter { color: #6B7280; }

/* Empty state */
.empty-state {
    display: flex; align-items: center; justify-content: center;
    height: 320px; background: #CFCFCF; border-radius: 4px;
}

.disclaimer { color: #82A1E8; font-size: .75rem; font-family: 'Inter Tight', sans-serif; }
.subhead    { color: #82A1E8; font-size: .875rem; margin-top: -4px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)

# ── Artifact & data loading ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model and explainer…")
def _get_artifacts():
    import explain as expl
    expl._load_artifacts()
    return expl._explainer, expl._preprocessor, list(expl._feature_names)


@st.cache_data
def _load_metrics() -> dict:
    with open(ROOT / "models" / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def _demo_bytes() -> bytes:
    """Return a reproducible 50-account sample from the raw CSV as bytes."""
    df = pd.read_csv(RAW_DATA_PATH)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=min(50, len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True).to_csv(index=False).encode()


EXPLAINER, PREPROCESSOR, FEATURE_NAMES = _get_artifacts()
METRICS  = _load_metrics()
ARTIFACT = load_artifact()

INDUSTRIES     = ["Education", "Finance", "Healthcare", "Manufacturing",
                  "Media", "Retail", "Tech", "Telecom"]
CONTRACT_TYPES = ["Annual", "Monthly", "Multi-year"]

# ── Preset scenarios ──────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "healthy": {
        "industry": "Tech", "company_size": 500, "contract_type": "Multi-year",
        "tenure_months": 36, "acv_usd": 60000,
        "seats_purchased": 100, "seats_active_last_30d": 88, "seat_util_pct": 88,
        "logins_last_30d": 820, "admin_logins_last_30d": 45, "features_adopted": 9,
        "mfa_enabled_pct": 85, "api_calls_last_30d": 4200,
        "support_tickets_last_90d": 1, "critical_tickets_last_90d": 0,
        "nps_score": 9, "qbr_attendance_pct": 90,
        "exec_sponsor_changed": False, "days_since_last_login": 2,
        "discount_pct": 5, "payment_delays_last_year": 0,
        "expansion_revenue_last_year_usd": 12000,
    },
    "at_risk": {
        "industry": "Retail", "company_size": 250, "contract_type": "Monthly",
        "tenure_months": 18, "acv_usd": 28000,
        "seats_purchased": 60, "seats_active_last_30d": 35, "seat_util_pct": 58,
        "logins_last_30d": 310, "admin_logins_last_30d": 12, "features_adopted": 5,
        "mfa_enabled_pct": 40, "api_calls_last_30d": 900,
        "support_tickets_last_90d": 3, "critical_tickets_last_90d": 0,
        "nps_score": 6, "qbr_attendance_pct": 55,
        "exec_sponsor_changed": False, "days_since_last_login": 18,
        "discount_pct": 10, "payment_delays_last_year": 1,
        "expansion_revenue_last_year_usd": 0,
    },
    "critical": {
        "industry": "Retail", "company_size": 180, "contract_type": "Monthly",
        "tenure_months": 4, "acv_usd": 18000,
        "seats_purchased": 40, "seats_active_last_30d": 14, "seat_util_pct": 35,
        "logins_last_30d": 95, "admin_logins_last_30d": 3, "features_adopted": 2,
        "mfa_enabled_pct": 10, "api_calls_last_30d": 120,
        "support_tickets_last_90d": 6, "critical_tickets_last_90d": 3,
        "nps_score": 3, "qbr_attendance_pct": 25,
        "exec_sponsor_changed": True, "days_since_last_login": 47,
        "discount_pct": 15, "payment_delays_last_year": 2,
        "expansion_revenue_last_year_usd": 0,
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_customer(inp: dict) -> dict:
    """Convert UI-scale form inputs to model feature dict."""
    return {
        "industry":                      inp["industry"],
        "company_size":                  inp["company_size"],
        "contract_type":                 inp["contract_type"],
        "tenure_months":                 inp["tenure_months"],
        "acv_usd":                       float(inp["acv_usd"]),
        "seats_purchased":               inp["seats_purchased"],
        "seats_active_last_30d":         inp["seats_active_last_30d"],
        "seat_utilization_rate":         inp["seat_util_pct"] / 100.0,
        "logins_last_30d":               inp["logins_last_30d"],
        "admin_logins_last_30d":         inp["admin_logins_last_30d"],
        "features_adopted":              inp["features_adopted"],
        "mfa_enabled_pct":               inp["mfa_enabled_pct"],
        "api_calls_last_30d":            inp["api_calls_last_30d"],
        "support_tickets_last_90d":      inp["support_tickets_last_90d"],
        "critical_tickets_last_90d":     inp["critical_tickets_last_90d"],
        "nps_score":                     inp["nps_score"],
        "qbr_attendance_rate":           inp["qbr_attendance_pct"] / 100.0,
        "exec_sponsor_changed_last_180d": int(inp["exec_sponsor_changed"]),
        "days_since_last_login":         inp["days_since_last_login"],
        "discount_pct":                  inp["discount_pct"],
        "payment_delays_last_year":      inp["payment_delays_last_year"],
        "expansion_revenue_last_year_usd": float(inp["expansion_revenue_last_year_usd"]),
    }


def _risk_color(tier: str) -> str:
    return {"Low": "#82A1E8", "Medium": "#6B7280", "High": "#000000"}.get(tier, "#6B7280")


def _gauge(prob: float, tier: str) -> go.Figure:
    color = _risk_color(tier)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 52, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9CA3AF"},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#EEF2FB"},
                {"range": [30, 70], "color": "#F5F5F5"},
                {"range": [70, 100], "color": "#E8E8E8"},
            ],
        },
    ))
    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=30, b=0),
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter Tight, sans-serif"),
    )
    return fig


def _waterfall(customer_dict: dict) -> plt.Figure:
    """Build a SHAP waterfall chart for one customer dict."""
    features = list(PREPROCESSOR.feature_names_in_)
    X_raw = pd.DataFrame([{c: customer_dict.get(c) for c in features}])
    X_enc = PREPROCESSOR.transform(X_raw)
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()

    raw_exp = EXPLAINER(X_enc)
    bv = (float(raw_exp.base_values[0])
          if hasattr(raw_exp.base_values, "__len__")
          else float(raw_exp.base_values))

    exp = shap.Explanation(
        values=raw_exp.values[0],
        base_values=bv,
        data=X_enc[0],
        feature_names=[_clean_name(f) for f in FEATURE_NAMES],
    )
    plt.close("all")
    shap.plots.waterfall(exp, max_display=12, show=False)
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner="Scoring portfolio…")
def _score_portfolio(csv_bytes: bytes) -> pd.DataFrame | None:
    """Batch-score a portfolio CSV; returns display-ready DataFrame."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    required = list(PREPROCESSOR.feature_names_in_)
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None  # caller shows the error

    X = PREPROCESSOR.transform(df[required])
    if hasattr(X, "toarray"):
        X = X.toarray()

    probs = ARTIFACT["model"].predict_proba(X)[:, 1]

    sv_all = EXPLAINER.shap_values(X)
    if isinstance(sv_all, list):
        sv_all = sv_all[1]

    _QUICK = {
        "Seat Utilization":      "Run enablement session + licence review",
        "Exec Sponsor Change":   "Book intro call with new sponsor (14d)",
        "Days Since Last Login": "Re-engagement outreach to admin team",
        "QBR Attendance Rate":   "Offer async format; shorten cadence",
        "Critical Support Tickets": "Executive escalation review",
        "NPS Score":             "Discovery call — 90-day improvement plan",
        "Contract Type":         "Annual contract conversation at next QBR",
        "Customer Tenure":       "Onboarding health check",
    }

    rows = []
    for i in range(len(df)):
        row  = df.iloc[i]
        prob = float(probs[i])
        tier = "High" if prob >= 0.70 else ("Medium" if prob >= 0.30 else "Low")

        # Top positive-SHAP feature for this account
        sv   = sv_all[i]
        top_idx  = int(np.argmax(sv))
        top_feat = FEATURE_NAMES[top_idx]
        top_col, _ = _parse_feat(top_feat)
        top_display = clean_feature_name(top_col)
        action = _QUICK.get(top_display, "Schedule discovery call")

        rows.append({
            "Account":       row.get("company_name", row.get("account_id", f"ACC-{i:04d}")),
            "Industry":      row.get("industry", "—"),
            "ACV ($)":       float(row.get("acv_usd", 0)),
            "Tenure (mo)":   int(row.get("tenure_months", 0)),
            "Risk Score":    round(prob * 100, 1),
            "Risk Tier":     tier,
            "Top Risk Factor": top_display,
            "Suggested Action": action,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Risk Score", ascending=False)
        .reset_index(drop=True)
    )


def _action_plan_txt(portfolio_df: pd.DataFrame) -> str:
    """Generate a text action plan for the top 10 at-risk accounts."""
    top10 = portfolio_df.head(10)
    lines = ["CHURN RISK — CSM ACTION PLAN", "=" * 50, ""]
    for _, row in top10.iterrows():
        lines += [
            f"Account : {row['Account']}",
            f"Risk    : {row['Risk Score']:.0f}% ({row['Risk Tier']})",
            f"Driver  : {row['Top Risk Factor']}",
            f"Action  : {row['Suggested Action']}",
            "",
        ]
    return "\n".join(lines)


def _color_rows(row: pd.Series):
    tier = row.get("Risk Tier", "")
    bg = {"High": "#F0F0F0", "Medium": "#EEF2FB", "Low": "#FFFFFF"}.get(tier, "#FFFFFF")
    return [f"background-color: {bg}"] * len(row)


# ── Navigation ───────────────────────────────────────────────────────────────────────

active = st.query_params.get("tab", "scorer")


def _nav(label: str, tid: str) -> str:
    weight    = "700" if active == tid else "500"
    underline = (
        "border-bottom:2px solid #000000;"
        if active == tid
        else "border-bottom:2px solid transparent;"
    )
    return (
        f'<a href="?tab={tid}" target="_self" '
        f'style="font-family:Inter Tight,sans-serif;font-size:.9rem;'
        f"font-weight:{weight};color:#000000;text-decoration:none;"
        f'padding:8px 16px;{underline}">{label}</a>'
    )


st.markdown(
    f"""
<nav style="display:flex;align-items:center;padding:16px 0;
            border-bottom:1px solid #CFCFCF;margin-bottom:8px;">
  <a href="?tab=scorer" target="_self"
     style="font-family:'Inter Tight',sans-serif;font-size:1.4rem;font-weight:700;
            color:#82A1E8;text-decoration:none;letter-spacing:-0.025em;
            margin-right:48px;white-space:nowrap;">ChurnBurn</a>
  <div style="display:flex;gap:4px;align-items:flex-end;">
    {_nav("Risk Scorer",       "scorer")}
    {_nav("Portfolio View",    "portfolio")}
    {_nav("Model Performance", "performance")}
  </div>
</nav>
</p>
""",
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════
# TAB 1 — Risk Scorer
# ═════════════════════════════════════════════════════════════════

if active == "scorer":
    if "preset" not in st.session_state:
        st.session_state.preset = "healthy"
    if "explanation" not in st.session_state:
        st.session_state.explanation = None
    if "customer_dict" not in st.session_state:
        st.session_state.customer_dict = None

    left, right = st.columns([1, 2], gap="large")

    # ── Left column: inputs ──────────────────────────────────────
    with left:
        pc1, pc2, pc3 = st.columns(3)
        if pc1.button("💚 Healthy",  use_container_width=True):
            st.session_state.preset = "healthy";  st.session_state.explanation = None
        if pc2.button("⚠️ At-Risk",  use_container_width=True):
            st.session_state.preset = "at_risk";  st.session_state.explanation = None
        if pc3.button("🚨 Critical", use_container_width=True):
            st.session_state.preset = "critical"; st.session_state.explanation = None

        p = PRESETS[st.session_state.preset]
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        with st.expander("🏢 Account Profile", expanded=True):
            industry      = st.selectbox("Industry", INDUSTRIES,
                                         index=INDUSTRIES.index(p["industry"]))
            company_size  = st.slider("Company size (employees)", 50, 5000,
                                      p["company_size"], step=50)
            contract_type = st.radio("Contract type", CONTRACT_TYPES,
                                     index=CONTRACT_TYPES.index(p["contract_type"]),
                                     horizontal=True)
            tenure_months = st.slider("Tenure (months)", 1, 120, p["tenure_months"])
            acv_usd       = st.number_input("Annual contract value ($)",
                                            5000, 500_000, p["acv_usd"], step=1000)

        with st.expander("📊 Product Usage"):
            seats_purchased     = st.slider("Seats purchased", 5, 500,
                                            p["seats_purchased"])
            seats_active        = st.slider("Active seats (last 30d)", 0,
                                            seats_purchased,
                                            min(p["seats_active_last_30d"],
                                                seats_purchased))
            seat_util_pct       = st.slider("Seat utilization (%)", 0, 100,
                                            p["seat_util_pct"])
            logins              = st.slider("Logins (last 30d)", 0, 5000,
                                            p["logins_last_30d"], step=10)
            admin_logins        = st.slider("Admin logins (last 30d)", 0, 200,
                                            p["admin_logins_last_30d"])
            features_adopted    = st.slider("Features adopted", 0, 12,
                                            p["features_adopted"])
            mfa_pct             = st.slider("MFA enabled (%)", 0, 100,
                                            p["mfa_enabled_pct"])
            api_calls           = st.number_input("API calls (last 30d)",
                                                  0, 50_000,
                                                  p["api_calls_last_30d"], step=100)

        with st.expander("💬 Engagement & Health"):
            support_tickets  = st.slider("Support tickets (last 90d)", 0, 15,
                                         p["support_tickets_last_90d"])
            critical_tickets = st.slider("Critical tickets (last 90d)", 0, 7,
                                         p["critical_tickets_last_90d"])
            nps_score        = st.slider("NPS score", 0, 10, p["nps_score"])
            qbr_pct          = st.slider("QBR attendance (%)", 0, 100,
                                         p["qbr_attendance_pct"])
            exec_changed     = st.toggle("Exec sponsor changed (last 180d)",
                                         value=p["exec_sponsor_changed"])
            days_login       = st.slider("Days since last login", 0, 180,
                                         p["days_since_last_login"])

        with st.expander("💰 Commercial"):
            discount_pct    = st.slider("Discount (%)", 0, 50, p["discount_pct"])
            payment_delays  = st.slider("Payment delays (last year)", 0, 5,
                                        p["payment_delays_last_year"])
            expansion_rev   = st.number_input("Expansion revenue last year ($)",
                                              0, 200_000,
                                              p["expansion_revenue_last_year_usd"],
                                              step=500)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        score_btn = st.button("🔍 Score this account", type="primary",
                              use_container_width=True)

        if score_btn:
            raw_inputs = {
                "industry": industry, "company_size": company_size,
                "contract_type": contract_type, "tenure_months": tenure_months,
                "acv_usd": int(acv_usd), "seats_purchased": seats_purchased,
                "seats_active_last_30d": seats_active, "seat_util_pct": seat_util_pct,
                "logins_last_30d": logins, "admin_logins_last_30d": admin_logins,
                "features_adopted": features_adopted, "mfa_enabled_pct": mfa_pct,
                "api_calls_last_30d": int(api_calls),
                "support_tickets_last_90d": support_tickets,
                "critical_tickets_last_90d": critical_tickets,
                "nps_score": nps_score, "qbr_attendance_pct": qbr_pct,
                "exec_sponsor_changed": exec_changed,
                "days_since_last_login": days_login, "discount_pct": discount_pct,
                "payment_delays_last_year": payment_delays,
                "expansion_revenue_last_year_usd": int(expansion_rev),
            }
            customer = _to_customer(raw_inputs)
            with st.spinner("Computing churn probability and SHAP values…"):
                st.session_state.explanation   = explain_prediction(customer)
                st.session_state.customer_dict = customer

    # ── Right column: results ────────────────────────────────────
    with right:
        if st.session_state.explanation is None:
            st.markdown("""
            <div class="empty-state">
              <div style="text-align:center;color:#000000;font-family:'Inter Tight',sans-serif;">
                <div style="font-size:2.5rem;margin-bottom:8px;">🔍</div>
                <div style="font-size:1rem;font-weight:500;">
                  Fill in the form and click<br><strong>Score this account</strong>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            exp      = st.session_state.explanation
            customer = st.session_state.customer_dict
            prob     = exp["churn_probability"]
            tier     = exp["risk_tier"]
            rf       = exp["top_risk_factors"]
            pf       = exp["top_protective_factors"]

            # 1 ── Score card
            gc, bc = st.columns([2, 1])
            with gc:
                st.plotly_chart(_gauge(prob, tier), use_container_width=True)
            with bc:
                badge = f"badge-{tier.lower()}"
                st.markdown(f"""
                <div style="padding-top:60px;text-align:center;">
                  <div class="{badge}"
                       style="font-size:1.1rem;padding:8px 22px;">{tier} Risk</div>
                </div>""", unsafe_allow_html=True)

            n_sig   = len([f for f in rf if abs(f["shap_value"]) > 0.05])
            areas   = list(dict.fromkeys(f["feature"] for f in rf[:3]))
            area_str = ", ".join(areas[:2]) if areas else "multiple signals"
            st.markdown(
                f"<p style='color:#82A1E8;font-size:.875rem;text-align:center;"
                f"font-family:Inter Tight,sans-serif;margin-top:-8px;margin-bottom:20px;'>"
                f"This account shows <strong>{n_sig}</strong> significant risk "
                f"signal(s) — top drivers: <em>{area_str}</em></p>",
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # 2 ── Why this score
            st.markdown("#### 🔬 Why this score")
            with st.spinner("Building SHAP waterfall…"):
                wf = _waterfall(customer)
            st.pyplot(wf, use_container_width=True)
            plt.close(wf)

            st.markdown("**Top risk factors**")
            for f in rf[:3]:
                st.markdown(f"""
                <div class="factor-card">
                  <div style="font-weight:700;font-size:.875rem;color:#000000;font-family:'Inter Tight',sans-serif;">{f['feature']}</div>
                  <div style="font-size:.8rem;color:#000000;margin:4px 0;font-family:'Inter Tight',sans-serif;">{f['impact']}</div>
                </div>""", unsafe_allow_html=True)

            if pf:
                st.markdown("**Protective factors**")
                for f in pf[:2]:
                    st.markdown(f"""
                    <div class="factor-card protective">
                      <div style="font-weight:700;font-size:.875rem;color:#82A1E8;font-family:'Inter Tight',sans-serif;">{f['feature']}</div>
                      <div style="font-size:.8rem;color:#000000;font-family:'Inter Tight',sans-serif;">{f['impact']}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # 3 ── Recommended actions
            st.markdown("#### 🎯 Recommended actions")
            recs = make_recommendation(rf)
            for rec in recs[:4]:
                tf = rec.get("timeframe", "")
                tf_cls = f"tf-{tf.replace(' ', '-').lower()}" if tf else ""
                st.markdown(f"""
                <div class="action-card">
                  <div class="tf-label {tf_cls}">{tf}</div>
                  <div style="font-size:.875rem;color:#000000;margin-bottom:4px;font-family:'Inter Tight',sans-serif;">{rec['action']}</div>
                  <div style="font-size:.75rem;color:#82A1E8;font-style:italic;font-family:'Inter Tight',sans-serif;">{rec['internal']}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # 4 ── Talking points
            st.markdown("#### 📋 Talking points for next call")
            st.caption("Copy this into your CRM notes")
            notes = format_for_csm(exp, account_name="This Account")
            st.code(notes, language=None)

# ═════════════════════════════════════════════════════════════════
# TAB 2 — Portfolio View
# ═════════════════════════════════════════════════════════════════

elif active == "portfolio":
    st.markdown("""
    **Upload your book of business CSV** to get a prioritized list of at-risk accounts.
    The model will score every account and surface the ones needing immediate attention.
    """)

    # Template download
    template_cols = (["company_name", "account_id"]
                     + list(PREPROCESSOR.feature_names_in_))
    template_csv = pd.DataFrame(columns=template_cols).to_csv(index=False).encode()
    st.download_button("📥 Download CSV template", template_csv,
                       file_name="churn_portfolio_template.csv",
                       mime="text/csv",
                       use_container_width=True)

    st.markdown("---")
    uploaded = st.file_uploader("Upload portfolio CSV", type="csv",
                                label_visibility="collapsed")

    use_demo = st.button("📊 Or use our demo portfolio (50 sample accounts)",
                         use_container_width=True)

    csv_bytes: bytes | None = None
    source_label = ""
    if uploaded is not None:
        csv_bytes   = uploaded.read()
        source_label = uploaded.name
    elif use_demo or st.session_state.get("portfolio_demo"):
        csv_bytes   = _demo_bytes()
        source_label = "Demo portfolio (50 accounts)"
        st.session_state["portfolio_demo"] = True

    if csv_bytes:
        portfolio_df = _score_portfolio(csv_bytes)

        if portfolio_df is None:
            source_df = pd.read_csv(io.BytesIO(csv_bytes))
            required  = list(PREPROCESSOR.feature_names_in_)
            missing   = [c for c in required if c not in source_df.columns]
            st.error(f"CSV is missing required columns: {', '.join(missing[:8])}…  "
                     f"Download the template above for the correct schema.")
        else:
            source_df = pd.read_csv(io.BytesIO(csv_bytes))
            st.caption(f"Scored: **{source_label}**")

            # ── Headline metrics ──────────────────────────────────
            n_total  = len(portfolio_df)
            n_high   = (portfolio_df["Risk Tier"] == "High").sum()
            n_medium = (portfolio_df["Risk Tier"] == "Medium").sum()
            acv_at_risk = portfolio_df.loc[
                portfolio_df["Risk Tier"] == "High", "ACV ($)"
            ].sum()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total accounts",  n_total)
            m2.metric("🔴 High Risk",    n_high,
                      delta=f"{n_high/n_total:.0%} of portfolio",
                      delta_color="off")
            m3.metric("🟡 Medium Risk",  n_medium,
                      delta=f"{n_medium/n_total:.0%} of portfolio",
                      delta_color="off")
            m4.metric("💰 ACV at Risk",  f"${acv_at_risk:,.0f}")

            st.markdown("---")

            # ── Visualisations ────────────────────────────────────
            vc1, vc2 = st.columns(2)

            with vc1:
                tier_counts = (portfolio_df["Risk Tier"]
                               .value_counts()
                               .reindex(["High", "Medium", "Low"], fill_value=0))
                fig_dist = go.Figure(go.Bar(
                    x=tier_counts.index.tolist(),
                    y=tier_counts.values.tolist(),
                    marker_color=["#000000", "#82A1E8", "#CFCFCF"],
                    text=tier_counts.values.tolist(),
                    textposition="outside",
                ))
                fig_dist.update_layout(
                    title="Risk Distribution",
                    xaxis_title="Risk Tier", yaxis_title="Accounts",
                    height=320, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                )
                st.plotly_chart(fig_dist, use_container_width=True)

            with vc2:
                color_map = {"High": "#000000", "Medium": "#82A1E8", "Low": "#CFCFCF"}
                fig_scatter = go.Figure()
                for tier_val, grp in portfolio_df.groupby("Risk Tier"):
                    fig_scatter.add_trace(go.Scatter(
                        x=grp["Risk Score"],
                        y=grp["ACV ($)"],
                        mode="markers",
                        name=tier_val,
                        marker=dict(
                            color=color_map.get(str(tier_val), "#6B7280"),
                            size=8, opacity=0.75,
                            line=dict(width=0.5, color="white"),
                        ),
                        text=grp["Account"],
                        hovertemplate="<b>%{text}</b><br>Risk: %{x:.0f}%<br>ACV: $%{y:,.0f}<extra></extra>",
                    ))
                fig_scatter.update_layout(
                    title="ACV vs Churn Risk Score",
                    xaxis_title="Risk Score (%)", yaxis_title="ACV ($)",
                    height=320, margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    legend=dict(orientation="h", y=-0.2),
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("---")

            # ── Prioritised table ─────────────────────────────────
            st.markdown("#### Prioritised account list")
            display = portfolio_df.copy()
            display["Risk Score"] = display["Risk Score"].map("{:.0f}%".format)
            display["ACV ($)"]    = display["ACV ($)"].map("${:,.0f}".format)

            styled = display.style.apply(_color_rows, axis=1)
            st.dataframe(styled, use_container_width=True, height=400,
                         hide_index=True)

            # ── Downloads ─────────────────────────────────────────
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "📥 Download prioritized list (CSV)",
                    portfolio_df.to_csv(index=False).encode(),
                    file_name="churn_risk_prioritized.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                plan_txt = _action_plan_txt(portfolio_df)
                st.download_button(
                    "📥 Download CSM action plan (TXT)",
                    plan_txt.encode(),
                    file_name="csm_action_plan.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

# ═════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═════════════════════════════════════════════════════════════════

elif active == "performance":
    st.markdown("### How well does this work?")
    st.markdown(
        "<p class='subhead'>Transparency builds trust. Here's exactly how the model "
        "performs and where it falls short.</p>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Headline metrics ───────────────────────────────
    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.metric("ROC-AUC",  f"{METRICS['roc_auc']:.3f}",
               help="How well the model ranks churners above non-churners. "
                    "1.0 = perfect, 0.5 = random.")
    mm2.metric("PR-AUC",   f"{METRICS['pr_auc']:.3f}",
               help="Precision-recall AUC — more reliable than ROC-AUC "
                    "for imbalanced data like churn.")
    mm3.metric("Recall",   f"{METRICS['recall']:.3f}",
               help="Of all customers who actually churned, what % did we catch?")
    mm4.metric("Precision", f"{METRICS['precision']:.3f}",
               help="Of all customers we flagged as at-risk, what % actually churned?")

    st.markdown("---")

    # ── Section 2: 2×2 plot grid ─────────────────────────────────
    st.markdown("#### Evaluation plots")
    PLOTS = ROOT / "docs" / "plots"
    pg1, pg2 = st.columns(2)
    pg3, pg4 = st.columns(2)
    for col, fname, caption in [
        (pg1, "confusion_matrix.png", "Confusion Matrix (threshold = 0.50)"),
        (pg2, "roc_curve.png",        "ROC Curve"),
        (pg3, "pr_curve.png",         "Precision-Recall Curve"),
        (pg4, "calibration_curve.png","Calibration Curve"),
    ]:
        p = PLOTS / fname
        with col:
            st.caption(caption)
            display_image(str(p), 70)

    st.markdown("---")

    # ── Section 3: SHAP feature importance ───────────────────────
    st.markdown("#### Top features driving predictions")
    shap_img = PLOTS / "shap_summary.png"
    if shap_img.exists():
        display_image(str(shap_img), 90)
    st.markdown("""
**Reading this chart:** Each dot is one test-set account. The x-axis shows
the SHAP value — how much that feature pushed the model toward (right) or
away from (left) predicting churn. Colour indicates feature value: red = high,
blue = low.

Top 5 drivers observed in this dataset:
1. **QBR Attendance Rate** — low attendance is the strongest predictor; when
   customers stop showing up to reviews, churn follows.
2. **Seat Utilization Rate** — unused licences signal low adoption and weak
   product value realization.
3. **NPS Score** — a low NPS is both a signal and a cause: unhappy customers
   leave.
4. **Contract Type** — monthly contracts correlate strongly with churn; the
   lower switching cost removes a key retention mechanism.
5. **Days Since Last Login** — extended admin inactivity suggests the product
   has been abandoned in practice, even if the contract is live.
""")

    st.markdown("---")

    # ── Section 4: Limitations ────────────────────────────────────
    st.markdown("#### Known limitations")
    st.warning("""
**This model has important limitations you should understand before using it:**

- **Synthetic training data.** Designed to mirror B2B SaaS patterns, but
  real-world distribution shift should be expected. Retrain on your actual data.
- **No seasonality or macro-event modelling.** A market downturn or a
  competitor launch will shift churn rates in ways this model cannot detect.
- **Product-change blindness.** If your product ships a major feature or
  removes something customers rely on, the model's learned associations
  become stale.
- **Snapshot-in-time.** This is a point-in-time predictor, not a time-series
  model. Trajectory matters — an account improving vs. declining looks the
  same at a single snapshot.
- **Calibration.** Validated on synthetic data. Real-world calibration may
  differ. Verify P(churn) against observed rates before trusting the numbers.
- **Augment, don't replace.** Model scores should inform CSM judgment, not
  automate account decisions.
""")

    st.markdown("---")

    # ── Section 5: Why these metrics ─────────────────────────────
    st.markdown("#### Why these metrics?")
    st.markdown("""
**Why PR-AUC instead of accuracy?**

With ~20% churn rate, a model that predicts *nobody churns* achieves 80%
accuracy — useless, but technically "good". PR-AUC ignores the majority class
entirely and focuses on how well the model identifies the churners. A PR-AUC
of 0.57 (vs. 0.20 for random) means the model is genuinely surfacing at-risk
accounts.

**Why prioritise recall over precision?**

The business cost of a missed churn (false negative) is the full ACV of the
lost account — typically $20k–$100k+ and irreversible. The cost of a false
alarm (false positive) is one unnecessary CSM call — maybe an hour of time.
The asymmetry is stark: err toward catching more churners, not fewer.

**What's the precision–recall trade-off?**

At the default threshold of 0.50, the model catches ~69% of actual churners
(recall) with ~42% precision — roughly 3 false alarms for every 2 real
churners surfaced. Raising the threshold to ~0.65 (the F1-optimal point)
brings precision and recall to ~54% each. Neither is right or wrong; it
depends on your team's capacity for follow-up calls.

**Why ROC-AUC too?**

ROC-AUC is the universal benchmark stakeholders expect. At 0.80, this model
is meaningfully better than chance and competitive with industry baselines for
synthetic data. We report it alongside PR-AUC so you can compare to
third-party benchmarks, while using PR-AUC for internal decision-making.
""")

# ── Footer (all tabs) ─────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#82A1E8;font-size:.75rem;"
    "font-family:Inter Tight,sans-serif;'>"
    "Built by <strong style='color:#000000;'>Noor Akhter</strong> · "
    "<a href='https://github.com/noor-akhter' style='color:#82A1E8;'>GitHub</a> · "
    "<a href='https://linkedin.com/in/noorakhter' style='color:#82A1E8;'>LinkedIn</a> · "
    "</p>",
    unsafe_allow_html=True,
)
