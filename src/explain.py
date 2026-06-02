"""
Generates SHAP-based explanations for individual account churn risk.

Product context: a risk score without an explanation is useless for a CSM.
"Account X is 78% likely to churn because: low seat utilization, a recent exec
sponsor change, and rising critical tickets" is an actionable talking point.
This module is the bridge between ML output and CSM action.

Note: the estimator (models/churn_model.pkl) is a CalibratedClassifierCV that
consumes already-encoded arrays, and the fitted encoder is loaded separately
from models/preprocessor.pkl. Scoring (`score_accounts`) transforms raw columns
through that preprocessor before predicting; SHAP attribution (`explain_account`)
does the same and explains the underlying tree model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import shap

# Make intra-package imports work whether run as `python src/explain.py` or
# `python -m src.explain` (mirrors the bootstrap in app/app.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import CATEGORICAL_COLS, RAW_DATA_PATH

MODEL_PATH = Path("models/churn_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

# Human-readable labels for the dashboard, keyed by raw column name.
FEATURE_LABELS: dict[str, str] = {
    "company_size": "Company size (employees)",
    "acv_usd": "Annual contract value",
    "tenure_months": "Tenure (months)",
    "seats_purchased": "Seats purchased",
    "seats_active_last_30d": "Active seats (30d)",
    "seat_utilization_rate": "Seat utilization rate",
    "logins_last_30d": "Logins (30d)",
    "admin_logins_last_30d": "Admin logins (30d)",
    "features_adopted": "Features adopted",
    "mfa_enabled_pct": "MFA enabled (%)",
    "api_calls_last_30d": "API calls (30d)",
    "support_tickets_last_90d": "Support tickets (90d)",
    "critical_tickets_last_90d": "Critical tickets (90d)",
    "nps_score": "NPS score",
    "qbr_attendance_rate": "QBR attendance rate",
    "exec_sponsor_changed_last_180d": "Exec sponsor changed (180d)",
    "days_since_last_login": "Days since last login",
    "discount_pct": "Discount (%)",
    "payment_delays_last_year": "Payment delays (last year)",
    "expansion_revenue_last_year_usd": "Expansion revenue (last year)",
    "industry": "Industry",
    "contract_type": "Contract type",
}


def load_artifact(model_path: Path = MODEL_PATH) -> dict:
    # Load the trained estimator bundle and attach the separately-saved fitted
    # preprocessor (encoding is owned by preprocess.py, never rebuilt here).
    artifact = joblib.load(model_path)
    artifact["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
    return artifact


def _pretty_name(transformed_name: str) -> str:
    """Map a ColumnTransformer output name back to a human-readable label.

    Examples:
      "num__seat_utilization_rate" -> "Seat utilization rate"
      "cat__contract_type_Monthly" -> "Contract type = Monthly"
    """
    if transformed_name.startswith("num__"):
        col = transformed_name[len("num__") :]
        return FEATURE_LABELS.get(col, col)
    if transformed_name.startswith("cat__"):
        rest = transformed_name[len("cat__") :]
        for col in CATEGORICAL_COLS:
            if rest.startswith(col + "_"):
                value = rest[len(col) + 1 :]
                return f"{FEATURE_LABELS.get(col, col)} = {value}"
        return rest
    return transformed_name


def score_accounts(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Returns df with a churn_probability and churn_flag column appended."""
    model = artifact["model"]
    threshold = artifact["threshold"]
    preprocessor = artifact["preprocessor"]
    # Select the raw input columns the preprocessor was fit on, then encode.
    X = preprocessor.transform(df[list(preprocessor.feature_names_in_)])
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
    preprocessor = artifact["preprocessor"]
    features = list(preprocessor.feature_names_in_)

    X = pd.DataFrame([row[features]])

    # Transform through the fitted preprocessor, then explain the tree model.
    calibrated = model  # the estimator is the CalibratedClassifierCV itself
    X_trans = preprocessor.transform(X)
    if hasattr(X_trans, "toarray"):  # OneHotEncoder yields a sparse matrix
        X_trans = X_trans.toarray()
    feat_names = list(preprocessor.get_feature_names_out())

    # CalibratedClassifierCV wraps fitted copies of the base estimator.
    base_estimator = calibrated.calibrated_classifiers_[0].estimator
    explainer = shap.TreeExplainer(base_estimator)
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):  # older shap returns one array per class
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    vals = shap_values[0]

    drivers = sorted(
        zip(feat_names, vals), key=lambda x: abs(x[1]), reverse=True
    )[:top_n]

    return [
        {
            "feature": _pretty_name(feat),
            "direction": "increases" if val > 0 else "decreases",
            "shap_value": round(float(val), 4),
        }
        for feat, val in drivers
    ]


if __name__ == "__main__":
    artifact = load_artifact()
    # Read raw account rows (with IDs/names) for the demo scoring run.
    df = pd.read_csv(RAW_DATA_PATH)
    scored = score_accounts(df, artifact)
    print(scored[["account_id", "churn_probability", "churn_flag"]].head(10).to_string(index=False))

    top_account = scored.iloc[0]
    drivers = explain_account(top_account, artifact)
    print(f"\nTop risk drivers for highest-risk account:")
    for d in drivers:
        print(f"  {d['feature']} {d['direction']} churn risk (SHAP={d['shap_value']:+.4f})")
