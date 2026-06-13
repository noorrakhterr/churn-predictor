"""Tests for the SHAP-based explanation and recommendation engine."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from explain import _load_artifacts, explain_prediction, make_recommendation

# Ensure singletons are loaded before any test touches them.
_load_artifacts()

_EXPECTED_KEYS = {"churn_probability", "risk_tier", "top_risk_factors", "top_protective_factors"}
_VALID_TIERS   = {"Low", "Medium", "High"}


# ---------------------------------------------------------------------------
# explain_prediction
# ---------------------------------------------------------------------------

def test_explain_prediction_returns_correct_shape(sample_customer_dict):
    result = explain_prediction(sample_customer_dict)
    assert _EXPECTED_KEYS == set(result.keys()), (
        f"Missing keys: {_EXPECTED_KEYS - set(result.keys())}"
    )
    assert result["risk_tier"] in _VALID_TIERS
    assert 0.0 <= result["churn_probability"] <= 1.0
    # Each factor must have the four documented fields
    for factor in result["top_risk_factors"] + result["top_protective_factors"]:
        assert {"feature", "value", "shap_value", "impact"} <= set(factor.keys())


def test_top_risk_factors_are_sorted_by_magnitude(high_risk_customer_dict):
    result = explain_prediction(high_risk_customer_dict)
    factors = result["top_risk_factors"]
    if len(factors) < 2:
        pytest.skip("Not enough risk factors to test ordering")
    shap_values = [f["shap_value"] for f in factors]
    # Risk factors are positive SHAP — larger values should come first
    assert shap_values == sorted(shap_values, reverse=True), (
        "top_risk_factors should be sorted by descending shap_value"
    )


# ---------------------------------------------------------------------------
# make_recommendation
# ---------------------------------------------------------------------------

def test_make_recommendation_returns_at_least_one_action(high_risk_customer_dict):
    explanation = explain_prediction(high_risk_customer_dict)
    recs = make_recommendation(explanation["top_risk_factors"])
    assert len(recs) >= 1
    for rec in recs:
        assert "action" in rec
        assert "internal" in rec
        assert len(rec["action"]) > 0


def test_make_recommendation_handles_unknown_factor():
    """A factor with no matching rule should trigger the fallback recommendation."""
    unknown_factors = [
        {
            "feature":    "completely_unknown_feature_xyz",
            "value":      42,
            "shap_value": 0.5,
            "impact":     "Strongly increases churn risk",
        }
    ]
    recs = make_recommendation(unknown_factors)
    assert len(recs) >= 1, "Fallback recommendation must always be returned"
    # The fallback action should reference a discovery call or similar
    combined = " ".join(r["action"].lower() for r in recs)
    assert any(
        kw in combined for kw in ("discovery", "call", "schedule", "understand")
    ), f"Expected fallback action text, got: {combined}"
