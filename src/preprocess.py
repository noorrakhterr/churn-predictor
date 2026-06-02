"""
Loads raw account data and defines the preprocessing pipeline for the churn model.

Product context: feature encoding is where domain knowledge (CSM intuition) gets
turned into numbers a model can use.  All encoding — categorical → one-hot,
numeric → imputation — lives inside a single scikit-learn ColumnTransformer
(see `build_preprocessor`).  That guarantees the exact same transform is applied
at training time and at scoring time, with no manual encoding scattered across
the codebase.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RAW_PATH = Path("data/raw/saas_churn.csv")
PROCESSED_PATH = Path("data/processed/features.csv")

# Column lists derived from the actual dtypes in saas_churn.csv.
# NUMERIC_COLS  = int/float columns; CATEGORICAL_COLS = object/bool columns.
# The target ('churned') and ID columns (account_id, company_name) are excluded.
NUMERIC_COLS = [
    "company_size",
    "acv_usd",
    "tenure_months",
    "seats_purchased",
    "seats_active_last_30d",
    "seat_utilization_rate",
    "logins_last_30d",
    "admin_logins_last_30d",
    "features_adopted",
    "mfa_enabled_pct",
    "api_calls_last_30d",
    "support_tickets_last_90d",
    "critical_tickets_last_90d",
    "nps_score",
    "qbr_attendance_rate",
    "days_since_last_login",
    "discount_pct",
    "payment_delays_last_year",
    "expansion_revenue_last_year_usd",
]

CATEGORICAL_COLS = [
    "industry",
    "contract_type",
    "exec_sponsor_changed_last_180d",
]

TARGET_COL = "churned"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_preprocessor() -> ColumnTransformer:
    """
    The single source of truth for feature encoding.

    - Numeric columns: median imputation (covers the nullable nps_score).
    - Categorical columns: most-frequent imputation + one-hot encoding, with
      unknown categories ignored at scoring time so unseen values don't crash.
    """
    numeric_pipeline = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns the raw (unencoded) feature matrix X and target y.

    Encoding is intentionally NOT done here — it's the ColumnTransformer's job
    (see `build_preprocessor`), so the same transform applies at train and score.
    """
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    return X, y


def run(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """
    Materialize the model-ready table: select the feature + target + ID columns
    from the raw snapshot and write them to PROCESSED_PATH.  Encoding happens in
    the model pipeline at fit/predict time, so this file stays human-readable
    (IDs and raw values intact) for the dashboard.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_raw(raw_path)
    df.to_csv(out_path, index=False)
    print(f"Processed {len(df)} rows → {out_path}")
    return df


if __name__ == "__main__":
    run()
