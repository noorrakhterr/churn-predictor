"""Tests for the trained churn model artifact."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))



def _encode(customer_dict: dict, preprocessor):
    """Helper: transform a single customer dict through the fitted preprocessor."""
    features = list(preprocessor.feature_names_in_)
    df = pd.DataFrame([{col: customer_dict.get(col) for col in features}])
    X = preprocessor.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return X


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_model_loads_successfully(loaded_model):
    # If the fixture resolves without error, the artifact exists and is valid.
    assert loaded_model is not None


def test_predict_proba_in_valid_range(loaded_model, loaded_preprocessor, sample_customer_dict):
    X = _encode(sample_customer_dict, loaded_preprocessor)
    proba = loaded_model.predict_proba(X)
    # predict_proba returns shape (n_samples, 2); both columns must sum to ~1
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 1] <= 1.0
    assert abs(proba[0].sum() - 1.0) < 1e-6


def test_high_risk_customer_scores_higher_than_low_risk(
    loaded_model, loaded_preprocessor, high_risk_customer_dict, low_risk_customer_dict
):
    """Critical sanity check: a clearly at-risk customer must outscore a healthy one."""
    X_high = _encode(high_risk_customer_dict, loaded_preprocessor)
    X_low  = _encode(low_risk_customer_dict, loaded_preprocessor)
    score_high = float(loaded_model.predict_proba(X_high)[0, 1])
    score_low  = float(loaded_model.predict_proba(X_low)[0, 1])
    assert score_high > score_low, (
        f"High-risk score ({score_high:.3f}) should exceed "
        f"low-risk score ({score_low:.3f})"
    )


def test_model_handles_single_customer_input(
    loaded_model, loaded_preprocessor, sample_customer_dict
):
    """The single-customer prediction path used by the app must not raise."""
    X = _encode(sample_customer_dict, loaded_preprocessor)
    assert X.shape[0] == 1
    result = loaded_model.predict_proba(X)
    assert result.shape[0] == 1
    assert 0.0 <= result[0, 1] <= 1.0
