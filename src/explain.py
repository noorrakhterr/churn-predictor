"""
Generates SHAP-based explanations for individual account churn risk.

Product context: a risk score without an explanation is useless for a CSM.
"Account X is 78% likely to churn because: low login recency, high support
tickets, and declining NPS" is an actionable talking point.  This module
is the bridge between ML output and CSM action.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import shap

MODEL_PATH = Path("models/churn_model.pkl")

# Human-readable labels for the dashboard
FEATURE_LABELS: dict[str, str] = {
    "days_since_last_login": "Days since last login",
    "avg_weekly_logins": "Avg weekly logins",
    "feature_adoption_rate": "Feature adoption rate",
    "seat_utilisation": "Seat utilisation",
    "support_tickets_open": "Open support tickets",
    "support_tickets_30d": "Support tickets (30d)",
    "nps_score": "NPS score",
    "nps_trend": "NPS trend",
    "csm_meetings_90d": "CSM meetings (90d)",
    "executive_sponsor_engaged": "Executive sponsor engaged",
    "arr_bucket": "ARR tier",
    "support_intensity": "Support intensity",
    "login_recency_flag": "No login in 30d",
}


def load_artifact(model_path: Path = MODEL_PATH) -> dict:
    return joblib.load(model_path)


def score_accounts(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Returns df with a churn_probability and churn_flag column appended."""
    model = artifact["model"]
    threshold = artifact["threshold"]
    features = artifact["feature_cols"]
    X = df[features]
    probs = model.predict_proba(X)[:, 1]
    result = df.copy()
    result["churn_probability"] = probs
    result["churn_flag"] = (probs >= threshold).astype(int)
    return result.sort_values("churn_probability", ascending=False)


def explain_account(
    row: pd.Series,
    artifact: dict,
    top_n: int = 3,
) -> list[dict]:
    """
    Returns top_n SHAP-driven risk drivers for a single account.
    Each item: {"feature": human label, "direction": "increases"/"decreases", "shap_value": float}
    """
    model = artifact["model"]
    features = artifact["feature_cols"]

    X = pd.DataFrame([row[features]])
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    vals = shap_values.values[0]

    drivers = sorted(
        zip(features, vals), key=lambda x: abs(x[1]), reverse=True
    )[:top_n]

    return [
        {
            "feature": FEATURE_LABELS.get(feat, feat),
            "direction": "increases" if val > 0 else "decreases",
            "shap_value": round(float(val), 4),
        }
        for feat, val in drivers
    ]


if __name__ == "__main__":
    artifact = load_artifact()
    processed = Path("data/processed/features.csv")
    df = pd.read_csv(processed)
    scored = score_accounts(df, artifact)
    print(scored[["account_id", "churn_probability", "churn_flag"]].head(10).to_string(index=False))

    top_account = scored.iloc[0]
    drivers = explain_account(top_account, artifact)
    print(f"\nTop risk drivers for highest-risk account:")
    for d in drivers:
        print(f"  {d['feature']} {d['direction']} churn risk (SHAP={d['shap_value']:+.4f})")
