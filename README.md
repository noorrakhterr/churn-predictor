# Churn Predictor — B2B SaaS

> Predict which accounts are likely to churn before renewal, so CSMs can
> intervene early rather than react late.

---

## Problem

B2B SaaS companies lose 5–7% of ARR to preventable churn every year.  
The signal is buried in product-usage logs, support tickets, and billing data —
and most CSMs only see it after it's too late to act.

**This tool surfaces that signal 30–60 days before renewal** by combining
ML-based risk scores with explainable feature attribution, giving CSMs a
prioritised save list with concrete talking points.

---

## Demo

```bash
make app   # launches Streamlit on http://localhost:8501
```

<!-- Replace with a real screenshot once the app is running -->
![App screenshot placeholder](docs/screenshot_placeholder.png)

---

## Results

| Metric | Value |
|---|---|
| Model | XGBoost |
| ROC-AUC | _TBD after training_ |
| Precision @ top decile | _TBD_ |
| Training data | Synthetic (1 000 accounts) |

---

## How It Works

```
Raw account data
      │
      ▼
 preprocess.py   ← cleans, engineers features (recency, frequency, support load)
      │
      ▼
   train.py      ← XGBoost with cross-validation; saves model + threshold
      │
      ▼
  explain.py     ← SHAP values → per-account "why is this account at risk?"
      │
      ▼
  app/app.py     ← Streamlit dashboard for CSMs / leadership
```

Key features engineered:

- **Login recency** — days since last active session  
- **Feature adoption rate** — % of licensed features used last 30 days  
- **Support ticket velocity** — open tickets / account age  
- **NPS trend** — direction of NPS over last two surveys  
- **Seat utilisation** — active seats / licensed seats  

---

## Local Setup

```bash
# 1. Clone and create a virtual environment
git clone <repo-url>
cd churn-predictor
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
make setup

# 3. Generate synthetic data and train
make train

# 4. Launch the app
make app
```

**Python 3.10+ required.**

---

## Roadmap

| Priority | Item | Status |
|---|---|---|
| P0 | Synthetic data generation | ✅ Done |
| P0 | Baseline XGBoost model | 🔲 In progress |
| P1 | SHAP explanation layer | 🔲 Planned |
| P1 | Streamlit CSM dashboard | 🔲 Planned |
| P2 | CRM alert integration (HubSpot / Salesforce) | 🔲 Backlog |
| P2 | Slack notification on high-risk accounts | 🔲 Backlog |
| P3 | Retraining pipeline (weekly cadence) | 🔲 Backlog |

---

## Project Structure

```
churn-predictor/
├── data/
│   ├── raw/          # source data — gitignored, see sourcing instructions
│   └── processed/    # cleaned feature matrices
├── notebooks/        # EDA and experimentation
├── src/
│   ├── generate_data.py   # synthetic account dataset
│   ├── preprocess.py      # cleaning + feature engineering
│   ├── train.py           # model training + evaluation
│   └── explain.py         # SHAP-based account explanations
├── app/
│   └── app.py             # Streamlit dashboard
├── models/               # saved .pkl artifacts — gitignored
├── tests/
└── docs/
    └── product-brief.md
```

---

## Contributing

1. Format: `make format`  
2. Test: `make test`  
3. Open a PR — describe the product impact, not just the code change.
