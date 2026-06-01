"""Tests for feature engineering logic."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import engineer_features, get_features_and_target, FEATURE_COLS


def _minimal_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "account_id": "abc-1",
                "company_name": "Acme",
                "segment": "SMB",
                "industry": "FinTech",
                "arr_usd": 20_000,
                "contract_months": 12,
                "days_to_renewal": 90,
                "days_since_last_login": 45,
                "avg_weekly_logins": 2.5,
                "feature_adoption_rate": 0.4,
                "seat_utilisation": 0.6,
                "support_tickets_open": 3,
                "support_tickets_30d": 5,
                "nps_score": 6,
                "nps_trend": -0.3,
                "csm_meetings_90d": 2,
                "executive_sponsor_engaged": True,
                "churned": 0,
            }
        ]
    )


def test_engineer_features_adds_columns():
    df = _minimal_df()
    out = engineer_features(df)
    assert "arr_bucket" in out.columns
    assert "support_intensity" in out.columns
    assert "login_recency_flag" in out.columns


def test_login_recency_flag_positive():
    df = _minimal_df()
    df["days_since_last_login"] = 45  # > 30
    out = engineer_features(df)
    assert out["login_recency_flag"].iloc[0] == 1


def test_login_recency_flag_negative():
    df = _minimal_df()
    df["days_since_last_login"] = 10  # <= 30
    out = engineer_features(df)
    assert out["login_recency_flag"].iloc[0] == 0


def test_support_intensity_no_divide_by_zero():
    df = _minimal_df()
    df["contract_months"] = 0
    out = engineer_features(df)
    assert out["support_intensity"].notna().all()


def test_get_features_and_target_shapes():
    df = engineer_features(_minimal_df())
    X, y = get_features_and_target(df)
    assert list(X.columns) == FEATURE_COLS
    assert len(y) == len(df)


def test_executive_sponsor_cast_to_int():
    df = _minimal_df()
    out = engineer_features(df)
    assert out["executive_sponsor_engaged"].dtype in (int, "int64", "int32")
