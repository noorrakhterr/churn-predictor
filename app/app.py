"""
Streamlit dashboard — Churn Predictor.

Product context: the primary user is a CSM, not a data scientist.  The UI
prioritises:
  1. "Which accounts do I call today?" (save list)
  2. "Why is this account at risk?" (SHAP explanation)
  3. "How bad is the overall portfolio?" (leadership heat map)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from explain import FEATURE_LABELS, explain_account, load_artifact, score_accounts
from preprocess import PROCESSED_PATH, FEATURE_COLS

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
)

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def get_artifact():
    return load_artifact()


@st.cache_data(show_spinner="Scoring accounts...")
def get_scored_df():
    artifact = get_artifact()
    df = pd.read_csv(PROCESSED_PATH)
    return score_accounts(df, artifact)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Churn Predictor")
    st.caption("B2B SaaS · CSM Dashboard")
    st.divider()

    try:
        df = get_scored_df()
        artifact = get_artifact()
        data_loaded = True
    except FileNotFoundError:
        st.error("No processed data or model found.\nRun `make train` first.")
        data_loaded = False

    if data_loaded:
        risk_threshold = st.slider(
            "Risk threshold",
            min_value=0.1,
            max_value=0.9,
            value=float(artifact["threshold"]),
            step=0.05,
            help="Accounts above this probability are flagged as at-risk.",
        )
        segment_filter = st.multiselect(
            "Filter by segment",
            options=df["segment"].unique().tolist() if "segment" in df.columns else [],
            default=[],
        )

# ── Main ──────────────────────────────────────────────────────────────────────

if not data_loaded:
    st.stop()

filtered = df.copy()
if segment_filter:
    filtered = filtered[filtered["segment"].isin(segment_filter)]
filtered["at_risk"] = (filtered["churn_probability"] >= risk_threshold).astype(int)

at_risk = filtered[filtered["at_risk"] == 1]

# ── KPI row ───────────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total accounts", len(filtered))
col2.metric("At-risk accounts", len(at_risk), delta=f"{len(at_risk)/len(filtered):.0%} of portfolio")
at_risk_arr = at_risk["arr_usd"].sum() if "arr_usd" in at_risk.columns else 0
col3.metric("ARR at risk", f"${at_risk_arr:,.0f}")
col4.metric("Avg risk score (at-risk)", f"{at_risk['churn_probability'].mean():.0%}" if len(at_risk) else "—")

st.divider()

# ── Save list ─────────────────────────────────────────────────────────────────

st.subheader("Save list — prioritised by churn probability")

display_cols = ["company_name", "segment", "arr_usd", "churn_probability", "days_to_renewal"] if all(
    c in filtered.columns for c in ["company_name", "segment", "arr_usd", "days_to_renewal"]
) else ["account_id", "churn_probability"]

st.dataframe(
    at_risk[display_cols]
    .sort_values("churn_probability", ascending=False)
    .style.format({"churn_probability": "{:.0%}", "arr_usd": "${:,.0f}"}),
    use_container_width=True,
    height=300,
)

# ── Account deep-dive ─────────────────────────────────────────────────────────

st.subheader("Account deep-dive")

if len(at_risk) > 0:
    account_options = at_risk["company_name"].tolist() if "company_name" in at_risk.columns else at_risk["account_id"].tolist()
    selected = st.selectbox("Select an at-risk account", options=account_options)

    id_col = "company_name" if "company_name" in at_risk.columns else "account_id"
    row = at_risk[at_risk[id_col] == selected].iloc[0]

    dcol1, dcol2 = st.columns([1, 2])
    with dcol1:
        st.metric("Churn probability", f"{row['churn_probability']:.0%}")
        if "arr_usd" in row:
            st.metric("ARR", f"${row['arr_usd']:,.0f}")
        if "days_to_renewal" in row:
            st.metric("Days to renewal", int(row["days_to_renewal"]))

    with dcol2:
        st.markdown("**Top risk drivers**")
        try:
            drivers = explain_account(row, artifact)
            for d in drivers:
                direction_icon = "🔴" if d["direction"] == "increases" else "🟢"
                st.markdown(f"{direction_icon} **{d['feature']}** {d['direction']} churn risk")
        except Exception as e:
            st.warning(f"Could not compute SHAP explanations: {e}")
else:
    st.info("No at-risk accounts at the current threshold.")

# ── Portfolio distribution ────────────────────────────────────────────────────

st.subheader("Portfolio risk distribution")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(filtered["churn_probability"], bins=30, color="#E74C3C", alpha=0.7, edgecolor="white")
ax.axvline(risk_threshold, color="#2C3E50", linestyle="--", linewidth=1.5, label=f"Threshold ({risk_threshold:.0%})")
ax.set_xlabel("Churn probability")
ax.set_ylabel("Accounts")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
st.pyplot(fig, use_container_width=True)
