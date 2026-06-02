"""Tests for the preprocessing column lists and encoding pipeline."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    build_preprocessor,
    get_features_and_target,
)


def _minimal_df(n: int = 6) -> pd.DataFrame:
    """A small but schema-complete frame (several rows so imputation has stats)."""
    rng = np.random.default_rng(0)
    industries = ["Tech", "Finance", "Healthcare"]
    contracts = ["Monthly", "Annual", "Multi-year"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "account_id": f"acct-{i}",
                "company_name": f"Company {i}",
                "company_size": int(rng.integers(50, 50_000)),
                "industry": industries[i % len(industries)],
                "contract_type": contracts[i % len(contracts)],
                "acv_usd": int(rng.integers(5_000, 500_000)),
                "tenure_months": int(rng.integers(1, 120)),
                "seats_purchased": int(rng.integers(5, 500)),
                "seats_active_last_30d": int(rng.integers(0, 500)),
                "seat_utilization_rate": float(rng.uniform(0, 1)),
                "logins_last_30d": int(rng.integers(0, 1000)),
                "admin_logins_last_30d": int(rng.integers(0, 50)),
                "features_adopted": int(rng.integers(0, 12)),
                "mfa_enabled_pct": float(rng.uniform(0, 100)),
                "api_calls_last_30d": int(rng.integers(0, 100_000)),
                "support_tickets_last_90d": int(rng.integers(0, 20)),
                "critical_tickets_last_90d": int(rng.integers(0, 5)),
                "nps_score": float(rng.integers(0, 11)),
                "qbr_attendance_rate": float(rng.uniform(0, 1)),
                "exec_sponsor_changed_last_180d": bool(rng.integers(0, 2)),
                "days_since_last_login": int(rng.integers(0, 90)),
                "discount_pct": float(rng.uniform(0, 45)),
                "payment_delays_last_year": int(rng.integers(0, 12)),
                "expansion_revenue_last_year_usd": int(rng.integers(0, 100_000)),
                "churned": int(rng.integers(0, 2)),
            }
        )
    return pd.DataFrame(rows)


def test_get_features_and_target_columns():
    df = _minimal_df()
    X, y = get_features_and_target(df)
    assert list(X.columns) == NUMERIC_COLS + CATEGORICAL_COLS
    assert y.name == TARGET_COL
    assert len(y) == len(df)


def test_preprocessor_outputs_all_numeric():
    df = _minimal_df()
    X, _ = get_features_and_target(df)
    transformed = build_preprocessor().fit_transform(X)
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert arr.shape[0] == len(df)
    # No NaNs survive the imputers, and everything is numeric for the model.
    assert not np.isnan(arr).any()


def test_preprocessor_one_hot_expands_categoricals():
    df = _minimal_df()
    X, _ = get_features_and_target(df)
    pre = build_preprocessor().fit(X)
    # One-hot encoding produces more output columns than the raw feature count.
    assert len(pre.get_feature_names_out()) > len(NUMERIC_COLS) + len(CATEGORICAL_COLS)


def test_numeric_imputation_fills_nulls():
    df = _minimal_df()
    df.loc[0, "nps_score"] = np.nan  # nps_score is nullable in the real data
    X, _ = get_features_and_target(df)
    transformed = build_preprocessor().fit_transform(X)
    arr = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert not np.isnan(arr).any()
