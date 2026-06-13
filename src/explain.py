"""
SHAP-based explanation module for the B2B SaaS churn predictor.

This module bridges the gap between raw ML output and CSM action: a risk score
without a reason is just noise.  "Account X is 78% likely to churn because of
low seat utilization, a recent exec-sponsor change, and 47 days of admin
inactivity" is an actionable talking point.

Public API (consumed by both the Streamlit app and this script's __main__):
  load_artifact()          → backward-compat dict loader for the app
  score_accounts()         → score a full account DataFrame
  explain_account()        → SHAP drivers for a single account row (app compat)
  explain_prediction()     → richer per-customer explanation dict (new API)
  make_recommendation()    → map risk factors to CSM actions
  format_for_csm()         → copy-pasteable CSM notes block

Global plot generation (run once via __main__):
  generate_global_plots()  → beeswarm + bar importance → docs/plots/

Runnable as:  python -m src.explain
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT          = Path(__file__).resolve().parent.parent
MODELS_DIR    = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
PLOTS_DIR     = ROOT / "docs" / "plots"
RAW_DATA_PATH = ROOT / "data" / "raw" / "saas_churn.csv"

# Categorical columns that were one-hot encoded by the preprocessor.
# Used to parse "cat__<col>_<value>" feature names back to (col, value) pairs.
CATEGORICAL_COLS = ["industry", "contract_type"]

# Human-readable display labels for raw column names.
# Exported so app.py can render clean names without re-importing this map.
FEATURE_LABELS: dict[str, str] = {
    "company_size":                    "Company size (employees)",
    "acv_usd":                         "Annual contract value ($)",
    "tenure_months":                   "Tenure (months)",
    "seats_purchased":                 "Seats purchased",
    "seats_active_last_30d":           "Active seats (30d)",
    "seat_utilization_rate":           "Seat utilization rate",
    "logins_last_30d":                 "Logins (30d)",
    "admin_logins_last_30d":           "Admin logins (30d)",
    "features_adopted":                "Features adopted",
    "mfa_enabled_pct":                 "MFA enabled (%)",
    "api_calls_last_30d":              "API calls (30d)",
    "support_tickets_last_90d":        "Support tickets (90d)",
    "critical_tickets_last_90d":       "Critical tickets (90d)",
    "nps_score":                       "NPS score",
    "qbr_attendance_rate":             "QBR attendance rate",
    "exec_sponsor_changed_last_180d":  "Exec sponsor changed (180d)",
    "days_since_last_login":           "Days since last login",
    "discount_pct":                    "Discount (%)",
    "payment_delays_last_year":        "Payment delays (last year)",
    "expansion_revenue_last_year_usd": "Expansion revenue (last year, $)",
    "industry":                        "Industry",
    "contract_type":                   "Contract type",
}

# ---------------------------------------------------------------------------
# Recommendation rules
# Each rule is matched in order against each top risk factor.
# ---------------------------------------------------------------------------
_RECOMMENDATION_RULES: list[dict] = [
    {
        "feature": "seat_utilization_rate",
        "condition": lambda v: float(v) < 0.60,
        "action": (
            "Schedule a meeting to understand what users need to adopt "
            "and run a review with the admin team."
        ),
        "internal": "Internal: pull a usage report from Tableau.",
    },
    {
        "feature": "exec_sponsor_changed_last_180d",
        "condition": lambda v: bool(int(v)),
        "action": (
            "Identify the new executive sponsor and book an intro call: "
            "brief them on the value delivered thus far and re-confirm business goals."
        ),
        "internal": "Internal: flag for AE review; update sponsor contact in CRM.",
    },
    {
        "feature": "days_since_last_login",
        "condition": lambda v: float(v) > 21,
        "action": (
            "Reach out to the customer and schedule a meeting "
            "with the admin team to surface any friction blocking active use."
        ),
        "internal": "Internal: review if there had been engagement with admins.",
    },
    {
        "feature": "qbr_attendance_rate",
        "condition": lambda v: float(v) < 0.60,
        "action": (
            "Restructure the QBR cadence; offer async video updates and shorter "
            "executive check-ins to accommodate schedule constraints."
        ),
        "internal": "Internal: propose a simplified QBR format with more detailed business value for low-attendance accounts.",
    },
    {
        "feature": "critical_tickets_last_90d",
        "condition": lambda v: int(v) >= 2,
        "action": (
            "Coordinate with Support for an escalation review."
        ),
        "internal": "Internal: loop in Support lead; confirm all critical tickets are resolved.",
    },
    {
        "feature": "nps_score",
        "condition": lambda v: float(v) < 6,
        "action": (
            "Conduct a discovery call to surface specific friction points; "
            "align on a 90-day improvement plan with clear owner and milestones."
        ),
        "internal": "Internal: escalate to CS manager if NPS < 4.",
    },
    {
        "feature": "contract_type",
        "condition": lambda v: str(v).strip().lower() == "monthly",
        "action": (
            "Open a conversation about annual contract incentives at the next QBR; "
            "quantify the multi-year discount value vs monthly flexibility."
        ),
        "internal": "Internal: involve AE — annual conversion is a commercial conversation.",
    },
    {
        "feature": "tenure_months",
        "condition": lambda v: float(v) < 6,
        "action": (
            "Run an onboarding health check; verify all milestones from the "
            "implementation plan are met and the key use case is live in production."
        ),
        "internal": "Internal: review implementation checklist with Solutions team.",
    },
]

_FALLBACK_RECOMMENDATION = {
    "action": (
        "Schedule a discovery call to understand the account's current state "
        "and surface any specific friction points or unmet needs."
    ),
    "internal": "Internal: review recent support history and product usage trends before the call.",
}

# ---------------------------------------------------------------------------
# Lazy-loaded module-level singletons
# (avoids repeated disk I/O when this module is imported by the Streamlit app)
# ---------------------------------------------------------------------------
_model        = None
_preprocessor = None
_explainer    = None
_feature_names: list[str] = []


def _load_artifacts() -> None:
    """Populate module-level singletons on first call; subsequent calls are no-ops."""
    global _model, _preprocessor, _explainer, _feature_names

    if _model is None:
        _model = joblib.load(MODELS_DIR / "churn_model.pkl")

    if _preprocessor is None:
        _preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")

    if not _feature_names:
        fp = MODELS_DIR / "feature_names.json"
        if fp.exists():
            with open(fp) as f:
                _feature_names = json.load(f)

    if _explainer is None:
        cached = MODELS_DIR / "shap_explainer.pkl"
        if cached.exists():
            _explainer = joblib.load(cached)
        else:
            # TreeExplainer accesses the tree structure directly — no sampling,
            # no approximation.  It is exact for XGBoost / tree ensembles.
            _explainer = shap.TreeExplainer(_model)


# ---------------------------------------------------------------------------
# Backward-compatible artifact loader (used by app.py)
# ---------------------------------------------------------------------------

def load_artifact(
    model_path: Path = MODELS_DIR / "churn_model.pkl",
) -> dict:
    """
    Load and return the model bundle as a dict the Streamlit app expects.

    The new train.py saves the XGBClassifier directly (not wrapped in a dict),
    so this function reassembles the expected {model, threshold, feature_cols,
    preprocessor} structure from the separate persisted artifacts.

    Parameters
    ----------
    model_path : path to the saved XGBClassifier (or the legacy bundle dict).

    Returns
    -------
    artifact dict with keys: model, threshold, feature_cols, preprocessor.
    """
    raw = joblib.load(model_path)

    # Handle both new format (bare XGBClassifier) and legacy format (dict).
    if isinstance(raw, dict):
        artifact = raw
    else:
        # New format: load supporting files from standard locations.
        threshold_path = MODELS_DIR / "threshold.json"
        threshold = 0.5
        if threshold_path.exists():
            with open(threshold_path) as f:
                threshold = json.load(f).get("optimal_threshold", 0.5)

        fn_path = MODELS_DIR / "feature_names.json"
        feature_cols: list[str] = []
        if fn_path.exists():
            with open(fn_path) as f:
                feature_cols = json.load(f)

        artifact = {
            "model":        raw,
            "threshold":    threshold,
            "feature_cols": feature_cols,
        }

    artifact["preprocessor"] = joblib.load(MODELS_DIR / "preprocessor.pkl")
    return artifact


# ---------------------------------------------------------------------------
# Backward-compatible account scoring (used by app.py)
# ---------------------------------------------------------------------------

def score_accounts(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """
    Score every row in df and return it sorted by descending churn probability.

    Appends churn_probability and churn_flag columns; all other columns are
    preserved so the Streamlit app can display company_name, account_id, etc.
    """
    model        = artifact["model"]
    threshold    = artifact["threshold"]
    preprocessor = artifact["preprocessor"]

    X = preprocessor.transform(df[list(preprocessor.feature_names_in_)])
    probs = model.predict_proba(X)[:, 1]

    result = df.copy()
    result["churn_probability"] = probs
    result["churn_flag"]        = (probs >= threshold).astype(int)
    return result.sort_values("churn_probability", ascending=False)


# ---------------------------------------------------------------------------
# Feature name utilities (shared by global plots and local explanations)
# ---------------------------------------------------------------------------

def _parse_feat(feat: str) -> tuple[str, str | None]:
    """
    Return (base_column_name, category_value_or_None) for a ColumnTransformer
    output feature name.

    "num__seat_utilization_rate"  → ("seat_utilization_rate", None)
    "cat__contract_type_Monthly"  → ("contract_type",         "Monthly")
    "cat__industry_Finance"       → ("industry",               "Finance")
    """
    if feat.startswith("num__"):
        return feat[5:], None
    if feat.startswith("cat__"):
        rest = feat[5:]
        for col in CATEGORICAL_COLS:
            if rest.startswith(col + "_"):
                return col, rest[len(col) + 1:]
        return rest, None
    return feat, None


def _clean_name(feat: str) -> str:
    """
    Map a ColumnTransformer output name to a human-readable display label.

    "num__seat_utilization_rate"  → "Seat utilization rate"
    "cat__contract_type_Monthly"  → "Contract type = Monthly"
    "cat__industry_Finance"       → "Industry = Finance"
    """
    col, cat_val = _parse_feat(feat)
    label = FEATURE_LABELS.get(col, col.replace("_", " ").title())
    return f"{label} = {cat_val}" if cat_val is not None else label


def _clean_names(feats: list[str]) -> list[str]:
    """Apply _clean_name to a list of feature names."""
    return [_clean_name(f) for f in feats]


# Explicit display names for raw dataset column names.
# Covers abbreviated or domain-specific names that title-casing alone
# would render poorly (e.g. "Nps Score", "Qbr Attendance Rate").
_DISPLAY_NAMES: dict[str, str] = {
    "seat_utilization_rate":           "Seat Utilization",
    "exec_sponsor_changed_last_180d":  "Exec Sponsor Change",
    "days_since_last_login":           "Days Since Last Login",
    "critical_tickets_last_90d":       "Critical Support Tickets",
    "qbr_attendance_rate":             "QBR Attendance Rate",
    "nps_score":                       "NPS Score",
    "contract_type":                   "Contract Type",
    "tenure_months":                   "Customer Tenure",
    "payment_delays_last_year":        "Payment Delays",
    "expansion_revenue_last_year_usd": "Expansion Revenue",
}


def clean_feature_name(name: str) -> str:
    """
    Convert a raw dataset column name to a readable display name.

    Looks up name in the explicit mapping first; falls back to replacing
    underscores with spaces and title-casing for any column not listed.

    Parameters
    ----------
    name : str
        Raw column name, e.g. "seat_utilization_rate".

    Returns
    -------
    Human-readable display name, e.g. "Seat Utilization".

    Examples
    --------
    >>> clean_feature_name("seat_utilization_rate")
    'Seat Utilization'
    >>> clean_feature_name("logins_last_30d")
    'Logins Last 30D'
    """
    return _DISPLAY_NAMES.get(name, name.replace("_", " ").title())


# ---------------------------------------------------------------------------
# Global SHAP plots
# ---------------------------------------------------------------------------

def generate_global_plots(
    X_test: np.ndarray,
    feature_names: list[str],
) -> None:
    """
    Compute SHAP values on the full test set and save two summary plots.

    shap_summary.png (beeswarm)
        Every dot is one test-set prediction.  X-position = SHAP value (impact
        on the log-odds of churn); colour = feature value magnitude (red=high,
        blue=low).  Reading left-to-right: which features push the model toward
        or away from predicting churn, and how strongly.  Preferred over the
        bar chart for understanding *direction* of effects.

    shap_importance.png (bar — mean |SHAP|)
        Ranks features by their average absolute SHAP value across all test
        samples — a model-native importance metric that accounts for feature
        interactions and non-linearities.  Unlike XGBoost's gain/split
        importance, SHAP is consistent: a feature with high gain but low
        average impact still ranks low here.

    Parameters
    ----------
    X_test        : preprocessed test matrix (n_samples × n_features).
    feature_names : ColumnTransformer output names (with num__/cat__ prefixes).
    """
    _load_artifacts()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    clean = _clean_names(feature_names)

    sv = _explainer.shap_values(X_test)
    # XGBClassifier (binary) returns a 2D array (n × p) directly.
    # Older SHAP versions return a list [class0_array, class1_array].
    if isinstance(sv, list):
        sv = sv[1]

    # --- Beeswarm (summary) ---
    plt.figure()
    shap.summary_plot(sv, X_test, feature_names=clean, show=False, max_display=20)
    plt.title("SHAP Feature Impact — Beeswarm")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Bar chart of mean |SHAP| ---
    plt.figure()
    shap.summary_plot(
        sv, X_test, feature_names=clean, plot_type="bar", show=False, max_display=20
    )
    plt.title("Mean |SHAP| Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Global SHAP plots saved → {PLOTS_DIR}/")


# ---------------------------------------------------------------------------
# Local explanation — backward-compatible (used by app.py)
# ---------------------------------------------------------------------------

def explain_account(
    row: pd.Series,
    artifact: dict,
    top_n: int = 3,
) -> list[dict]:
    """
    Return the top_n SHAP-driven churn drivers for a single account row.

    Each item: {"feature": human label, "direction": "increases"/"decreases",
                "shap_value": float}

    Compatible with the existing Streamlit app call signature.
    """
    model        = artifact["model"]
    preprocessor = artifact["preprocessor"]
    features     = list(preprocessor.feature_names_in_)
    feat_names   = list(preprocessor.get_feature_names_out())

    X_raw   = pd.DataFrame([row[features]])
    X_enc   = preprocessor.transform(X_raw)
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()

    explainer  = shap.TreeExplainer(model)
    sv         = explainer.shap_values(X_enc)
    if isinstance(sv, list):
        sv = sv[1]
    vals = sv[0]

    drivers = sorted(zip(feat_names, vals), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return [
        {
            "feature":    _clean_name(feat),
            "direction":  "increases" if val > 0 else "decreases",
            "shap_value": round(float(val), 4),
        }
        for feat, val in drivers
    ]


# ---------------------------------------------------------------------------
# Local explanation — new rich API
# ---------------------------------------------------------------------------


def interpret_shap_value(shap_value: float) -> str:
    """
    Translate a raw SHAP value into a plain-English string for CSM display.

    SHAP values are log-odds contributions; the thresholds here (±0.1, ±0.2)
    are calibrated to this model's typical output range so "strongly" actually
    means a meaningful shift in predicted probability, not just a large number.

    Parameters
    ----------
    shap_value : float
        A single SHAP value from the model (positive = pushes toward churn,
        negative = pushes away from churn).

    Returns
    -------
    Plain-English impact string suitable for display in CSM notes or UI.
    """
    if shap_value > 0.2:
        return "Strongly increases churn risk"
    if shap_value > 0.1:
        return "Moderately increases churn risk"
    if shap_value > 0.0:
        return "Slightly increases churn risk"
    if shap_value < -0.2:
        return "Strongly reduces churn risk"
    if shap_value < -0.1:
        return "Moderately reduces churn risk"
    if shap_value < 0.0:
        return "Slightly reduces churn risk"
    return "Minimal impact on churn risk"


def explain_prediction(customer_dict: dict) -> dict:
    """
    Compute a rich, structured churn explanation for a single customer.

    Takes a raw customer dict (column names matching the dataset schema),
    applies the saved preprocessor, runs SHAP attribution on the tree model,
    and returns a structured explanation ready for CSM consumption.

    Parameters
    ----------
    customer_dict : dict
        Raw feature values keyed by column name, e.g.
        {"seat_utilization_rate": 0.35, "contract_type": "Monthly", ...}.
        ID columns (account_id, company_name) are ignored if present.

    Returns
    -------
    {
        "churn_probability": float,          # model score in [0, 1]
        "risk_tier": "Low"|"Medium"|"High",  # <30% / 30-70% / >70%
        "top_risk_factors": [                # features pushing toward churn
            {
                "feature":     str,   # raw column name, e.g. "seat_utilization_rate"
                "value":       any,   # original value from customer_dict
                "shap_value":  float, # positive → increases churn log-odds
                "impact":      str,   # plain-English description
            },
            ...  # up to 5
        ],
        "top_protective_factors": [          # features pushing away from churn
            { ...same shape... },            # up to 3
        ],
    }

    Notes
    -----
    Categorical features are deduplicated: if multiple OHE columns for the
    same original column (e.g. contract_type_Monthly, contract_type_Annual)
    appear, only the one with the largest |SHAP| is surfaced — showing
    both would confuse a CSM reading the output.
    """
    _load_artifacts()

    # Build a single-row DataFrame with the columns the preprocessor expects.
    features = list(_preprocessor.feature_names_in_)
    X_raw    = pd.DataFrame([{col: customer_dict.get(col) for col in features}])
    X_enc    = _preprocessor.transform(X_raw)
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()

    prob = float(_model.predict_proba(X_enc)[0, 1])

    if prob < 0.30:
        tier = "Low"
    elif prob < 0.70:
        tier = "Medium"
    else:
        tier = "High"

    # SHAP values: shape (1, n_features) for a single encoded row.
    sv = _explainer.shap_values(X_enc)
    if isinstance(sv, list):
        sv = sv[1]
    shap_vals = sv[0]   # shape (n_features,)

    # --- Build per-feature factor dicts ---
    # For categoricals we deduplicate by base column, keeping the entry with
    # the largest |SHAP|.  This prevents "Contract type = Monthly" and
    # "Contract type = Annual" both appearing for the same account.
    seen_base_cols: dict[str, tuple[int, float]] = {}  # col → (idx, |shap|)
    ordered_indices = np.argsort(shap_vals)[::-1]       # high→low

    def _make_factor(feat: str, shap_val: float) -> dict:
        col, cat_val = _parse_feat(feat)
        if cat_val is not None:
            raw_val = customer_dict.get(col)   # e.g. "Monthly" from the raw dict
        else:
            raw_val = customer_dict.get(col)
        return {
            "feature":    clean_feature_name(col),
            "value":      raw_val,
            "shap_value": round(float(shap_val), 4),
            "impact":     interpret_shap_value(shap_val),
        }

    all_factors = [
        _make_factor(_feature_names[i], shap_vals[i])
        for i in range(len(_feature_names))
    ]

    # Deduplicate categorical base columns: keep max-|SHAP| entry per col.
    seen: dict[str, float] = {}
    deduped: list[dict] = []
    for f in sorted(all_factors, key=lambda x: abs(x["shap_value"]), reverse=True):
        col = f["feature"]
        if col not in seen:
            seen[col] = abs(f["shap_value"])
            deduped.append(f)

    top_risk        = [f for f in deduped if f["shap_value"] > 0][:5]
    top_protective  = [f for f in reversed(deduped) if f["shap_value"] < 0][:3]
    # reversed(deduped) gives ascending shap order → most negative first
    top_protective  = sorted(
        [f for f in deduped if f["shap_value"] < 0],
        key=lambda x: x["shap_value"],
    )[:3]

    return {
        "churn_probability":    round(prob, 4),
        "risk_tier":            tier,
        "top_risk_factors":     top_risk,
        "top_protective_factors": top_protective,
    }


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def make_recommendation(top_risk_factors: list[dict]) -> list[dict]:
    """
    Map a list of risk factors to concrete CSM actions.

    Iterates through the ordered risk factors and matches each against the
    rule table.  Returns one recommendation per matched rule (deduped by
    rule to avoid repeats).  If no rule matches any factor, the fallback
    discovery-call recommendation is appended.

    Parameters
    ----------
    top_risk_factors : list of factor dicts from explain_prediction's
        "top_risk_factors" key.  Each dict must have "feature" and "value".

    Returns
    -------
    list of {"action": str, "internal": str} dicts, max one per matched rule.
    """
    recommendations: list[dict] = []
    matched_rules: set[str] = set()

    for factor in top_risk_factors:
        feat  = factor.get("feature", "")
        value = factor.get("value")
        if value is None:
            continue
        for rule in _RECOMMENDATION_RULES:
            rule_key = rule["feature"]
            if rule_key in matched_rules:
                continue
            if feat != rule_key:
                continue
            try:
                if rule["condition"](value):
                    recommendations.append(
                        {"action": rule["action"], "internal": rule["internal"]}
                    )
                    matched_rules.add(rule_key)
            except (TypeError, ValueError):
                continue

    # Fallback: if no rule fired, recommend a discovery call.
    if not recommendations:
        recommendations.append(_FALLBACK_RECOMMENDATION)

    return recommendations


# ---------------------------------------------------------------------------
# CSM notes formatter
# ---------------------------------------------------------------------------

def format_for_csm(
    explanation: dict,
    account_name: str = "Account",
) -> str:
    """
    Format a churn explanation as a copy-pasteable CSM notes block.

    Pulls the top risk-factor impact strings and calls make_recommendation
    internally so the output is self-contained.

    Section headers and content adapt to risk tier:
    - High / Medium: lead with risk drivers, then recommendations.
    - Low: lead with protective factors (what's keeping the account healthy),
      then note any minor risk signals, then recommendations (if applicable).

    Parameters
    ----------
    explanation  : dict returned by explain_prediction().
    account_name : customer / company name to include in the header.

    Returns
    -------
    Multi-line string, e.g.::

        Account: Acme Corp
        Risk: 78% (High)
        Top risk drivers:
          - Seat utilization at 35% (benchmark: 70%+)
          - Exec sponsor changed in the last 180 days
          - NPS score of 4 (healthy benchmark: 7+)
        Recommended next actions:
          1. Schedule an enablement session...
             Internal: flag for CS Ops...
          2. Identify the new exec sponsor...
    """
    prob        = explanation["churn_probability"]
    tier        = explanation["risk_tier"]
    risk_facs   = explanation.get("top_risk_factors", [])
    prot_facs   = explanation.get("top_protective_factors", [])

    lines = [
        f"Account: {account_name}",
        f"Risk: {prob * 100:.0f}% ({tier})",
    ]

    if tier == "Low":
        # For healthy accounts the story is what's working, not tiny risk signals.
        if prot_facs:
            lines.append("Key strengths (protective factors):")
            for f in prot_facs:
                lines.append(f"  - {f['impact']}")
        # Surface minor risk signals only if their SHAP values are non-trivial.
        notable_risks = [f for f in risk_facs if abs(f["shap_value"]) > 0.05]
        if notable_risks:
            lines.append("Minor risk signals:")
            for f in notable_risks:
                lines.append(f"  - {f['impact']}")
        else:
            lines.append("No significant risk signals detected.")
    else:
        lines.append("Top risk drivers:")
        for f in risk_facs:
            lines.append(f"  - {f['impact']}")
        if prot_facs:
            lines.append("Protective factors:")
            for f in prot_facs:
                lines.append(f"  - {f['impact']}")

    recs = make_recommendation(risk_facs)
    # Omit the fallback discovery-call for Low-risk accounts with no risk signals,
    # since scheduling a speculative call for a healthy account adds noise.
    if tier == "Low" and not [f for f in risk_facs if abs(f["shap_value"]) > 0.05]:
        lines.append("Recommended next actions:")
        lines.append("  - Maintain current engagement cadence; no urgent action required.")
    else:
        lines.append("Recommended next actions:")
        for i, rec in enumerate(recs, start=1):
            lines.append(f"  {i}. {rec['action']}")
            lines.append(f"     {rec['internal']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point — generate global plots and demo 3 sample customers
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Generate global SHAP plots and verify the explainer on three sample accounts.

    Samples are constructed to represent the range of risk levels:
    - healthy    : high utilization, good NPS, engaged admin, multi-year contract
    - medium     : moderate usage, some lag, monthly contract
    - at_risk    : low utilization, exec sponsor changed, critical tickets,
                   new customer, monthly contract
    """
    _load_artifacts()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Save the SHAP explainer for app reuse ----
    explainer_path = MODELS_DIR / "shap_explainer.pkl"
    joblib.dump(_explainer, explainer_path)
    print(f"SHAP explainer saved → {explainer_path}")

    # ---- Global plots on the test set ----
    print("\nGenerating global SHAP plots...")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    generate_global_plots(X_test, _feature_names)

    # ---- Local explanations for 3 sample customers ----
    samples = {
        "healthy": {
            "company_size": 500,
            "acv_usd": 60000,
            "tenure_months": 36,
            "seats_purchased": 100,
            "seats_active_last_30d": 88,
            "seat_utilization_rate": 0.88,
            "logins_last_30d": 820,
            "admin_logins_last_30d": 45,
            "features_adopted": 9,
            "mfa_enabled_pct": 85.0,
            "api_calls_last_30d": 4200,
            "support_tickets_last_90d": 1,
            "critical_tickets_last_90d": 0,
            "nps_score": 9,
            "qbr_attendance_rate": 0.90,
            "exec_sponsor_changed_last_180d": 0,
            "days_since_last_login": 2,
            "discount_pct": 5.0,
            "payment_delays_last_year": 0,
            "expansion_revenue_last_year_usd": 12000,
            "industry": "Tech",
            "contract_type": "Multi-year",
        },
        "medium": {
            "company_size": 250,
            "acv_usd": 28000,
            "tenure_months": 18,
            "seats_purchased": 60,
            "seats_active_last_30d": 35,
            "seat_utilization_rate": 0.58,
            "logins_last_30d": 310,
            "admin_logins_last_30d": 12,
            "features_adopted": 5,
            "mfa_enabled_pct": 40.0,
            "api_calls_last_30d": 900,
            "support_tickets_last_90d": 3,
            "critical_tickets_last_90d": 0,
            "nps_score": 6,
            "qbr_attendance_rate": 0.55,
            "exec_sponsor_changed_last_180d": 0,
            "days_since_last_login": 18,
            "discount_pct": 10.0,
            "payment_delays_last_year": 1,
            "expansion_revenue_last_year_usd": 0,
            "industry": "Retail",
            "contract_type": "Monthly",
        },
        "at_risk": {
            "company_size": 180,
            "acv_usd": 18000,
            "tenure_months": 4,
            "seats_purchased": 40,
            "seats_active_last_30d": 14,
            "seat_utilization_rate": 0.35,
            "logins_last_30d": 95,
            "admin_logins_last_30d": 3,
            "features_adopted": 2,
            "mfa_enabled_pct": 10.0,
            "api_calls_last_30d": 120,
            "support_tickets_last_90d": 6,
            "critical_tickets_last_90d": 3,
            "nps_score": 3,
            "qbr_attendance_rate": 0.25,
            "exec_sponsor_changed_last_180d": 1,
            "days_since_last_login": 47,
            "discount_pct": 15.0,
            "payment_delays_last_year": 2,
            "expansion_revenue_last_year_usd": 0,
            "industry": "Retail",
            "contract_type": "Monthly",
        },
    }

    print("\n" + "=" * 65)
    print("Sample Customer Explanations")
    print("=" * 65)

    for label, cust in samples.items():
        exp   = explain_prediction(cust)
        notes = format_for_csm(exp, account_name=f"Sample ({label})")
        print(f"\n{'─' * 65}")
        print(notes)

    print(f"\n{'─' * 65}")
    print("Done.")


if __name__ == "__main__":
    main()
