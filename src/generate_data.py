"""
Synthesize a realistic B2B SaaS churn dataset for the churn-predictor project.

This is SYNTHETIC data. No real customer information is used. The goal is to
mirror the *structure and correlations* of real mid-market B2B SaaS churn data
closely enough that a model trained on it learns meaningful, defensible patterns
(the kind a CSM would recognize) rather than memorizing noise.

How the data is generated
--------------------------
1. Firmographics (company_size, industry, contract_type, acv_usd, tenure) are
   drawn from skewed distributions that look like a real customer book — many
   small accounts, a long tail of large ones, ACV correlated with headcount.

2. A per-account latent "health" factor (unobserved) drives most usage and
   engagement signals coherently: healthy accounts show high seat utilization,
   frequent logins, broad feature adoption, strong QBR attendance, high NPS,
   and few critical tickets. This is what creates *realistic correlations*
   between the predictors themselves — not just between each predictor and the
   label.

3. The churn label is generated from a logistic model whose log-odds are a
   weighted sum of the SAME observed features, so every documented driver is
   genuinely present in the data and learnable:
       - low  seat_utilization_rate            -> higher churn
       - high days_since_last_login            -> higher churn
       - exec_sponsor_changed_last_180d=True   -> ~2x churn rate
       - high critical_tickets_last_90d        -> higher churn
       - low  qbr_attendance_rate              -> higher churn
       - Monthly contracts > Annual > Multi-year for churn risk
       - nps_score < 6                         -> higher churn
       - short tenure_months (<6)              -> higher churn (honeymoon risk)
   Secondary nudges come from payment delays, low feature adoption, and support
   load.

4. Irreducible Gaussian noise is added in log-odds space so the problem is NOT
   trivially separable (expect ROC-AUC in the low-0.80s, not 0.99). The
   intercept is calibrated by bisection to land the overall churn rate in the
   realistic ~18-22% band for mid-market B2B SaaS.

Output: data/raw/saas_churn.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

RANDOM_SEED = 42
N_RECORDS = 8_000
TARGET_CHURN_RATE = 0.20  # aim for the middle of the realistic 18-22% band
NOISE_SIGMA = 0.85  # irreducible noise in log-odds space (keeps it non-trivial)
OUTPUT_PATH = Path("data/raw/saas_churn.csv")

INDUSTRIES = [
    "Tech",
    "Finance",
    "Healthcare",
    "Retail",
    "Manufacturing",
    "Education",
    "Media",
    "Telecom",
]
INDUSTRY_WEIGHTS = [0.26, 0.16, 0.13, 0.12, 0.11, 0.09, 0.07, 0.06]

CONTRACT_TYPES = ["Monthly", "Annual", "Multi-year"]
CONTRACT_WEIGHTS = [0.25, 0.55, 0.20]
# Relative churn-risk contribution by contract type (Monthly riskiest).
CONTRACT_RISK = {"Monthly": 1.0, "Annual": 0.0, "Multi-year": -1.0}

fake = Faker()
rng = np.random.default_rng(RANDOM_SEED)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(logits: np.ndarray, target_rate: float) -> float:
    """Bisection-solve for the intercept that hits the target mean churn rate."""
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        rate = float(_sigmoid(logits + mid).mean())
        if rate < target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def generate(n: int = N_RECORDS) -> pd.DataFrame:
    # --- Latent, unobserved account health: the spine of all correlations. ---
    health = rng.normal(0.0, 1.0, n)

    # --- Firmographics ----------------------------------------------------
    company_size = np.clip(
        np.exp(rng.normal(6.5, 1.1, n)).round(), 50, 50_000
    ).astype(int)
    log_size = np.log10(company_size)

    industry = rng.choice(INDUSTRIES, size=n, p=INDUSTRY_WEIGHTS)
    contract_type = rng.choice(CONTRACT_TYPES, size=n, p=CONTRACT_WEIGHTS)

    # ACV correlated with headcount, log-normal spread, realistic bounds.
    acv_usd = np.clip(
        np.exp(7.0 + 0.6 * np.log(company_size) + rng.normal(0, 0.4, n)).round(),
        5_000,
        2_000_000,
    ).astype(int)

    # Tenure: gamma-shaped, plenty of young accounts (honeymoon-risk segment).
    tenure_months = np.clip(rng.gamma(2.0, 14.0, n).round(), 1, 120).astype(int)

    # --- Product usage ----------------------------------------------------
    # Not every employee gets a seat; bounded by headcount.
    seats_purchased = np.maximum(
        5, np.minimum(company_size, (company_size * rng.uniform(0.05, 0.5, n)).round())
    ).astype(int)

    # Utilization driven by health; active seats derived from it, then the
    # rate is recomputed exactly so the "derived" column is internally consistent.
    raw_util = np.clip(0.55 + 0.18 * health + rng.normal(0, 0.12, n), 0.02, 1.0)
    seats_active_last_30d = np.clip(
        (seats_purchased * raw_util).round(), 0, seats_purchased
    ).astype(int)
    seat_utilization_rate = (seats_active_last_30d / seats_purchased).round(3)

    # Logins scale with active seats and health (per-seat activity).
    per_seat_logins = np.clip(rng.normal(8 + 4 * health, 4, n), 0, None)
    logins_last_30d = (seats_active_last_30d * per_seat_logins).round().astype(int)
    admin_logins_last_30d = rng.binomial(logins_last_30d, 0.06)

    features_adopted = np.clip(
        (6 + 3 * health + rng.normal(0, 1.5, n)).round(), 0, 12
    ).astype(int)

    # MFA adoption (Okta-flavored): higher for larger, more security-mature,
    # healthier orgs.
    mfa_enabled_pct = np.clip(
        rng.normal(50 + 10 * health + 6 * (log_size - 3), 22, n), 0, 100
    ).round(1)

    # API usage: heavy-tailed, scales with active seats and health.
    api_calls_last_30d = (
        seats_active_last_30d * np.exp(rng.normal(2.0 + 0.5 * health, 1.0, n))
    ).round().astype(int)

    # --- Engagement / health ---------------------------------------------
    # Unhealthy + larger accounts file more tickets.
    ticket_lambda = np.clip(2.0 + 1.5 * (-health) + 0.5 * log_size, 0.1, None)
    support_tickets_last_90d = rng.poisson(ticket_lambda)
    crit_p = np.clip(0.05 + 0.12 * (-health), 0.01, 0.6)
    critical_tickets_last_90d = rng.binomial(support_tickets_last_90d, crit_p)

    # NPS driven by health; ~20% missing (surveys aren't always answered).
    nps_score = np.clip((7 + 1.8 * health + rng.normal(0, 1.5, n)).round(), 0, 10)
    nps_score = nps_score.astype(float)
    missing_nps = rng.random(n) < 0.20
    nps_score[missing_nps] = np.nan

    # QBR attendance: health + a bump for accounts on real (non-monthly) terms.
    contract_qbr_bonus = np.where(contract_type == "Monthly", -0.15, 0.08)
    qbr_attendance_rate = np.clip(
        0.6 + 0.2 * health + contract_qbr_bonus + rng.normal(0, 0.15, n), 0, 1
    ).round(2)

    exec_p = np.clip(0.12 + 0.05 * (-health), 0.02, 0.5)
    exec_sponsor_changed_last_180d = rng.random(n) < exec_p

    # Days since last login: low for healthy/active accounts; if a customer had
    # zero logins in 30 days, force a longer gap.
    days_scale = np.clip(10 - 5 * health, 1.0, None)
    days_since_last_login = np.clip(
        rng.exponential(days_scale).round(), 0, 90
    ).astype(int)
    no_logins = logins_last_30d == 0
    days_since_last_login[no_logins] = rng.integers(20, 91, size=no_logins.sum())

    # --- Commercial -------------------------------------------------------
    discount_pct = np.clip(
        rng.normal(8 + 5 * (np.log10(acv_usd) - 4), 6, n), 0, 45
    ).round(1)
    payment_delays_last_year = np.clip(
        rng.poisson(np.clip(0.4 + 0.6 * (-health), 0.05, None)), 0, 12
    )
    expand = rng.random(n) < np.clip(0.25 + 0.18 * health, 0.02, 0.8)
    expansion_revenue_last_year_usd = np.where(
        expand, (acv_usd * rng.uniform(0.05, 0.4, n)).round(), 0
    ).astype(int)

    # --- Churn label: logistic model over the OBSERVED features -----------
    util_term = np.clip(0.6 - seat_utilization_rate, -0.6, 0.6)
    days_term = days_since_last_login / 90.0
    exec_term = exec_sponsor_changed_last_180d.astype(float)
    crit_term = np.minimum(critical_tickets_last_90d, 5) / 5.0
    qbr_term = np.clip(0.7 - qbr_attendance_rate, -0.7, 0.7)
    contract_term = np.array([CONTRACT_RISK[c] for c in contract_type])
    nps_low_term = np.where(np.isnan(nps_score), 0.0, (nps_score < 6).astype(float))
    tenure_term = np.clip((6 - tenure_months) / 6.0, 0.0, 1.0)
    features_term = (12 - features_adopted) / 12.0
    support_term = np.minimum(support_tickets_last_90d, 10) / 10.0
    delays_term = np.minimum(payment_delays_last_year, 5) / 5.0
    # Interaction: brand-new accounts on month-to-month terms are the classic
    # honeymoon-churn risk — the combination is riskier than either signal alone.
    monthly_short_tenure_term = (
        (contract_type == "Monthly") & (tenure_months < 6)
    ).astype(float)

    logit = (
        1.6 * util_term
        + 1.4 * days_term
        + 0.75 * exec_term  # tuned so exec change ≈ 2x base churn rate
        + 1.1 * crit_term
        + 1.2 * qbr_term
        + 0.85 * contract_term
        + 0.9 * nps_low_term
        + 1.5 * tenure_term
        + 1.1 * monthly_short_tenure_term
        + 0.4 * features_term
        + 0.3 * support_term
        + 0.5 * delays_term
        + rng.normal(0, NOISE_SIGMA, n)  # irreducible noise
    )
    intercept = _calibrate_intercept(logit, TARGET_CHURN_RATE)
    churn_prob = _sigmoid(logit + intercept)
    churned = (rng.random(n) < churn_prob).astype(int)

    # Faker for human-readable identifiers (realistic account book).
    account_id = [fake.uuid4() for _ in range(n)]
    company_name = [fake.company() for _ in range(n)]

    df = pd.DataFrame(
        {
            "account_id": account_id,
            "company_name": company_name,
            # Firmographics
            "company_size": company_size,
            "industry": industry,
            "contract_type": contract_type,
            "acv_usd": acv_usd,
            "tenure_months": tenure_months,
            # Product usage
            "seats_purchased": seats_purchased,
            "seats_active_last_30d": seats_active_last_30d,
            "seat_utilization_rate": seat_utilization_rate,
            "logins_last_30d": logins_last_30d,
            "admin_logins_last_30d": admin_logins_last_30d,
            "features_adopted": features_adopted,
            "mfa_enabled_pct": mfa_enabled_pct,
            "api_calls_last_30d": api_calls_last_30d,
            # Engagement / health
            "support_tickets_last_90d": support_tickets_last_90d,
            "critical_tickets_last_90d": critical_tickets_last_90d,
            "nps_score": nps_score,
            "qbr_attendance_rate": qbr_attendance_rate,
            "exec_sponsor_changed_last_180d": exec_sponsor_changed_last_180d,
            "days_since_last_login": days_since_last_login,
            # Commercial
            "discount_pct": discount_pct,
            "payment_delays_last_year": payment_delays_last_year,
            "expansion_revenue_last_year_usd": expansion_revenue_last_year_usd,
            # Target
            "churned": churned,
        }
    )
    return df


def print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    churn_rate = df["churned"].mean()

    print(f"\nGenerated {n:,} synthetic accounts -> {OUTPUT_PATH}")
    print("=" * 60)
    print(f"Overall churn rate: {churn_rate:.1%}  ({df['churned'].sum():,} churned)")

    # ---- Missing values --------------------------------------------------
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    if missing.empty:
        print("  (none)")
    else:
        for col, cnt in missing.items():
            print(f"  {col:<32} {cnt:>6,} ({cnt / n:.1%})")

    # ---- Correlations of key drivers with churn --------------------------
    tmp = df.copy()
    tmp["exec_sponsor_changed_last_180d"] = tmp[
        "exec_sponsor_changed_last_180d"
    ].astype(int)
    tmp["monthly_contract"] = (tmp["contract_type"] == "Monthly").astype(int)

    key_features = [
        "seat_utilization_rate",
        "days_since_last_login",
        "exec_sponsor_changed_last_180d",
        "critical_tickets_last_90d",
        "qbr_attendance_rate",
        "monthly_contract",
        "nps_score",
        "tenure_months",
        "features_adopted",
        "logins_last_30d",
        "payment_delays_last_year",
    ]
    print("\nCorrelation with churned (Pearson; NaNs dropped pairwise):")
    corrs = (
        tmp[key_features + ["churned"]]
        .corr()["churned"]
        .drop("churned")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    for feat, c in corrs.items():
        arrow = "↑churn" if c > 0 else "↓churn"
        print(f"  {feat:<32} {c:+.3f}  {arrow}")

    # ---- Driver sanity checks --------------------------------------------
    print("\nDriver sanity checks (segment churn rates):")
    exec_yes = df.loc[df["exec_sponsor_changed_last_180d"], "churned"].mean()
    exec_no = df.loc[~df["exec_sponsor_changed_last_180d"], "churned"].mean()
    ratio = exec_yes / exec_no if exec_no else float("nan")
    print(
        f"  exec sponsor changed:  {exec_yes:.1%} vs {exec_no:.1%} "
        f"(no change)  -> {ratio:.1f}x"
    )

    for ctype in CONTRACT_TYPES:
        rate = df.loc[df["contract_type"] == ctype, "churned"].mean()
        print(f"  contract = {ctype:<11} {rate:.1%}")

    low_util = df.loc[df["seat_utilization_rate"] < 0.4, "churned"].mean()
    high_util = df.loc[df["seat_utilization_rate"] >= 0.7, "churned"].mean()
    print(f"  seat util <40%:        {low_util:.1%} vs {high_util:.1%} (>=70%)")

    honeymoon = df.loc[df["tenure_months"] < 6, "churned"].mean()
    tenured = df.loc[df["tenure_months"] >= 6, "churned"].mean()
    print(f"  tenure <6 months:      {honeymoon:.1%} vs {tenured:.1%} (>=6)")

    detractors = df.loc[df["nps_score"] < 6, "churned"].mean()
    promoters = df.loc[df["nps_score"] >= 6, "churned"].mean()
    print(f"  nps <6:                {detractors:.1%} vs {promoters:.1%} (>=6)")


if __name__ == "__main__":
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(OUTPUT_PATH, index=False)
    print_summary(df)
