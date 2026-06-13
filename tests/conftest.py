"""Shared pytest fixtures for the churn-predictor test suite."""

import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Customer feature dicts
# Values match the presets used in app.py so they're realistic and tested end-to-end.
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_customer_dict() -> dict:
    """Medium-risk baseline customer — sits in the ambiguous middle."""
    return {
        "industry": "Retail",
        "company_size": 250,
        "contract_type": "Monthly",
        "tenure_months": 18,
        "acv_usd": 28000.0,
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
        "expansion_revenue_last_year_usd": 0.0,
    }


@pytest.fixture
def high_risk_customer_dict() -> dict:
    """Clearly at-risk customer — should score well above 0.50."""
    return {
        "industry": "Retail",
        "company_size": 180,
        "contract_type": "Monthly",
        "tenure_months": 4,
        "acv_usd": 18000.0,
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
        "expansion_revenue_last_year_usd": 0.0,
    }


@pytest.fixture
def low_risk_customer_dict() -> dict:
    """Clearly healthy customer — should score well below 0.30."""
    return {
        "industry": "Tech",
        "company_size": 500,
        "contract_type": "Multi-year",
        "tenure_months": 36,
        "acv_usd": 60000.0,
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
        "expansion_revenue_last_year_usd": 12000.0,
    }


# ---------------------------------------------------------------------------
# Artifact fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def loaded_model():
    """The trained XGBClassifier loaded once for the whole session."""
    return joblib.load(ROOT / "models" / "churn_model.pkl")


@pytest.fixture(scope="session")
def loaded_preprocessor():
    """The fitted ColumnTransformer loaded once for the whole session."""
    return joblib.load(ROOT / "models" / "preprocessor.pkl")
