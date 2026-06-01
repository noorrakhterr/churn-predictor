"""
Cleans raw account data and engineers features for the churn model.

Product context: feature engineering is where domain knowledge (CSM intuition)
gets encoded into numbers.  Each feature here maps to a real signal that an
experienced CSM would use to flag an at-risk account.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/accounts.csv")
PROCESSED_PATH = Path("data/processed/features.csv")

FEATURE_COLS = [
    "days_since_last_login",
    "avg_weekly_logins",
    "feature_adoption_rate",
    "seat_utilisation",
    "support_tickets_open",
    "support_tickets_30d",
    "nps_score",
    "nps_trend",
    "csm_meetings_90d",
    "executive_sponsor_engaged",
    # Engineered
    "arr_bucket",
    "support_intensity",
    "login_recency_flag",
]

TARGET_COL = "churned"


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # ARR bucket — product insight: SMB churn behaves differently from Enterprise
    out["arr_bucket"] = pd.cut(
        out["arr_usd"],
        bins=[0, 25_000, 75_000, 150_000, float("inf")],
        labels=["micro", "small", "mid", "large"],
    ).cat.codes  # ordinal encode

    # Support intensity: ticket rate per month of contract age
    contract_months_safe = out["contract_months"].clip(lower=1)
    out["support_intensity"] = out["support_tickets_30d"] / contract_months_safe

    # Binary flag: no login in last 30 days — strongest leading indicator
    out["login_recency_flag"] = (out["days_since_last_login"] > 30).astype(int)

    # Boolean → int for ML
    out["executive_sponsor_engaged"] = out["executive_sponsor_engaged"].astype(int)

    return out


def get_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def run(raw_path: Path = RAW_PATH, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_raw(raw_path)
    df = engineer_features(df)
    df.to_csv(out_path, index=False)
    print(f"Processed {len(df)} rows → {out_path}")
    return df


if __name__ == "__main__":
    run()
