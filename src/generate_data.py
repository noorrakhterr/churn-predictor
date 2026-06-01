"""
Generates a synthetic B2B SaaS account dataset.

Product context: real labelled churn data is rarely available at the start of
a new tool — this lets us iterate on the model and app before connecting a
live data source.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

RANDOM_SEED = 42
N_ACCOUNTS = 1_000
OUTPUT_PATH = Path("data/raw/accounts.csv")

fake = Faker()
rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


def _churn_probability(row: dict) -> float:
    """
    Heuristic churn score used to label synthetic accounts.
    Encodes domain knowledge from CSM experience:
    - Low logins + low feature adoption = strong signal
    - High support tickets = frustration signal
    - Low seat utilisation = low stickiness
    - Negative NPS trend = leading indicator
    """
    score = 0.0
    score += max(0.0, (row["days_since_last_login"] - 14) / 60) * 0.3
    score += (1 - row["feature_adoption_rate"]) * 0.25
    score += min(1.0, row["support_tickets_open"] / 5) * 0.2
    score += (1 - row["seat_utilisation"]) * 0.15
    score += (1 - (row["nps_trend"] + 1) / 2) * 0.1  # nps_trend in [-1, 1]
    return float(np.clip(score + rng.normal(0, 0.05), 0, 1))


def generate(n: int = N_ACCOUNTS) -> pd.DataFrame:
    segments = ["SMB", "Mid-Market", "Enterprise"]
    industries = ["FinTech", "HealthTech", "EdTech", "MarTech", "HRTech"]

    records = []
    for _ in range(n):
        row: dict = {
            "account_id": fake.uuid4(),
            "company_name": fake.company(),
            "segment": rng.choice(segments, p=[0.5, 0.35, 0.15]),
            "industry": rng.choice(industries),
            "arr_usd": int(rng.integers(5_000, 250_001)),
            "contract_months": int(rng.choice([12, 24, 36])),
            "days_to_renewal": int(rng.integers(1, 366)),
            # Usage signals
            "days_since_last_login": int(rng.integers(0, 91)),
            "avg_weekly_logins": round(float(rng.uniform(0, 20)), 1),
            "feature_adoption_rate": round(float(rng.beta(2, 2)), 3),
            "seat_utilisation": round(float(rng.beta(3, 2)), 3),
            # Health signals
            "support_tickets_open": int(rng.integers(0, 11)),
            "support_tickets_30d": int(rng.integers(0, 16)),
            "nps_score": int(rng.integers(0, 11)),
            "nps_trend": round(float(rng.uniform(-1, 1)), 2),
            # Relationship signals
            "csm_meetings_90d": int(rng.integers(0, 7)),
            "executive_sponsor_engaged": bool(rng.choice([True, False])),
        }
        p_churn = _churn_probability(row)
        row["churned"] = int(rng.random() < p_churn)
        records.append(row)

    return pd.DataFrame(records)


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    churn_rate = df["churned"].mean()
    print(f"Generated {len(df)} accounts → {OUTPUT_PATH}")
    print(f"Churn rate: {churn_rate:.1%}  ({df['churned'].sum()} churned accounts)")
