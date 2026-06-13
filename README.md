# Churn Risk Predictor: "ChurnBurn"

> AI-powered account prioritization for B2B SaaS Customer Success teams

**[Live Demo](https://churnburn.streamlit.app)** · [Project Brief](docs/product-brief.md) · [Model Card](docs/model-card.md)

---

## The Problem

The greatest constrain for B2B companies is retaining customers, a responsibility
that falls largely onto a Customer Success Manager. Churned customers are difficult
to spot early and present a large risk when they're already too far gone. After
nearly a year at Okta, I've watched customers slip through CSM's fingers due to a
lack of signaling on customer data such as seat utilization, login gaps, and support
tickets. 

---

## The Solution: ChurnBurn

ChurnBurn is a tool that scores an account's churn risk using machine learning and 
explains why it is at risk using SHAP feature attribution. The analytics allow for
targeted CSM actions grounded in how accounts actually churn. The tool aims to 
be a part of a CSM's daily or weekly workflow, helping them identify which accounts
are actually at risk and what necessary actions to take next.

---

## Key Features

- **Risk scoring** with calibrated probabilities (0–100%), color-coded by tier (Low / Medium / High)
- **Per-account explanations** via SHAP so every score includes the top drivers and protective signals in plain English
- **Action recommendations** grounded in B2B SaaS CSM domain knowledge
- **Portfolio Scorer** where a CSM can upload a CSV of their account list and get a ranked list with risk tiers, top drivers, and suggested plays
- **Model transparency** dashboard with full performance metrics, calibration curves, and explicit limitations

---

## Product Decisions Worth Highlighting

**Optimized for recall, not accuracy.** This choice was made because a false negative, or a churning account the model misses, means a lost customer and a missed save opportunity. A false positive means one unnecessary CSM call. That asymmetry is why I used PR-AUC as the primary optimization metric (not ROC-AUC, which is misleadingly optimistic on imbalanced data where ~80% of accounts are healthy).

**Synthesized realistic B2B data.** Standard churn datasets don't contain the signals B2B CSMs actually act on: seat utilization, exec sponsor changes, QBR attendance, or feature adoption. I designed an 8,000-row synthetic dataset based on my experience at Okta thus far with realistic cross-feature correlations.

**Surfaced SHAP explanations as plain English.** A raw SHAP value of +0.34 is useless to a CSM. Every driver gets translated into qualitative impact tiers ("Strongly increases churn risk") and then mapped to a specific action — so the output is "book an exec intro call within 14 days," not "exec_sponsor_changed_last_180d: 0.34."

**The model is not the product.** The explainability & action oriented nature of ChurnBurn is what the product is centered around. The churn score the model creates is a feature, but what comes after is what makes the tool usable.

---

## Results

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.799 | 0.556 | 0.707 | 0.405 | 0.514 |
| Random Forest | 0.793 | 0.531 | 0.241 | 0.699 | 0.357 |
| **XGBoost (tuned)** | **0.802** | **0.572** | **0.689** | **0.416** | **0.519** |

*LR and RF are 5-fold CV means on the training set. XGBoost figures are on the held-out test set (1,600 accounts) after hyperparameter tuning.*

In CSM terms: out of 100 customers who actually churn, the model flags 69. Of every 100 accounts it flags as at-risk, 42 go on to churn. For a CSM managing 50 accounts with a 20% churn rate, that translates to roughly 17 at-risk alerts per quarter — a manageable workload that catches 7 out of every 10 accounts that would otherwise be lost.

---

## Insights from EDA

Full analysis in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

**The first 90 days are a churn cliff — 0–3 month accounts churn at ~40% vs. ~20% baseline.** → Product implication: first-90-day activation is the highest-ROI retention investment. Build guided onboarding with explicit day-30 and day-60 health checks, not a one-time welcome email.

**Monthly contracts churn at ~33% vs. ~12% for multi-year — nearly 3×.** → Product implication: packaging incentives that move customers off month-to-month aren't just commercial wins; they're a structural retention lever. The Monthly base should be a standing watch-list.

**Seat utilization is the clearest behavioral separator (churned avg: 41% vs. retained: 58%).** → Product implication: utilization should be a first-class health metric with automated CSM alerts when an account drops below 60%, not a metric buried in a quarterly business review slide.

**Risk compounds — non-linearly.** One risk signal (low utilization, exec sponsor change, or short tenure) correlates with ~31% churn. Two signals push to ~51%. All three: ~72%. → Product implication: ship a composite health score, not a wall of individual metric alerts. Route scarce CSM time to accounts with 2+ simultaneous signals — that's where intervention ROI is highest.

**Industry is a flat predictor (~18–22% churn across all segments).** → Product implication: resist industry-specific churn playbooks. The signal is in behavior, not firmographics.

---

## How It Works

```
┌─────────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────┐    ┌────────────┐
│  Synthetic B2B  │ →  │  Preprocess  │ →  │ XGBoost  │ →  │ SHAP │ →  │ Streamlit  │
│  SaaS Data (8K) │    │  Pipeline    │    │  Model   │    │      │    │     UI     │
│  (generate_data)│    │  (sklearn)   │    │          │    │      │    │            │
└─────────────────┘    └──────────────┘    └──────────┘    └──────┘    └────────────┘
```

1. **Data generation** — 8,000 synthetic B2B SaaS accounts with realistic churn correlations (latent health factor drives usage signals coherently; logistic model sets the label)
2. **Preprocessing** — sklearn `ColumnTransformer`: median imputation + standard scaling for numerics, mode imputation + one-hot encoding for categoricals; fitted on train only
3. **Training** — 5-fold stratified CV baseline across Logistic Regression, Random Forest, and XGBoost; `RandomizedSearchCV` tunes the winner on PR-AUC
4. **Explanations** — SHAP `TreeExplainer` generates per-prediction attributions; a translation layer maps raw values to plain-English tiers and CSM action recommendations
5. **UI** — Streamlit serves three workflows: single-account scoring, portfolio prioritization, and model transparency

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data | pandas, NumPy, Faker (synthetic data generation) |
| Modeling | scikit-learn 1.4, XGBoost 2.0 |
| Explainability | SHAP 0.45 |
| UI | Streamlit 1.35, Plotly |
| Testing | pytest, ruff, black |
| Deployment | Streamlit Community Cloud |

---

## Local Setup

```bash
# Clone and set up
git clone https://github.com/noor-akhter/churn-predictor.git
cd churn-predictor
python -m venv venv && source venv/bin/activate

# Install dependencies
make setup

# Generate data and run the full pipeline
make all        # data → preprocess → train → explain

# Launch the app
make app        # opens http://localhost:8501
```

Run the test suite:

```bash
make test       # pytest tests/ -v
make lint       # ruff check src/ app/ tests/
```

**Python 3.10+ required.**

---

## Project Structure

```
churn-predictor/
├── app/
│   ├── app.py                 # Streamlit dashboard
│   └── data/
│       └── demo_portfolio.csv # committed demo data for deployed env
├── data/
│   ├── raw/                   # gitignored — generated locally
│   └── processed/             # gitignored — generated locally
├── docs/
│   ├── model-card.md
│   ├── product-brief.md
│   └── plots/
├── models/                    # gitignored — generated locally
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── generate_data.py       # synthetic B2B SaaS dataset
│   ├── preprocess.py          # cleaning + feature engineering pipeline
│   ├── train.py               # model training, CV, hyperparameter tuning
│   └── explain.py             # SHAP explanations + CSM recommendation engine
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_explain.py
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## What's Next

- **Recommendation feedback loop** — track which actions a CSM took and whether the account churned anyway, then weight recommendations by historical effectiveness
- **Time-series signals** — current model is a point-in-time snapshot; v2 would incorporate trend features (utilization trending down over 3 months is a stronger signal than a single low reading)
- **Segment-specific models** — enterprise and SMB accounts have different churn drivers; a single global model misses the nuance

---

## About

I'm a Customer Success Manager at Okta transitioning into Product Management. I built this project to demonstrate three things: the domain expertise I bring from working directly with B2B SaaS customers, the product judgment to frame a technical problem around user needs, and the technical fluency to ship a working ML product from scratch.

As a CSM, I lived this problem every week — the manual, inconsistent process of figuring out which accounts needed attention. I built the tool I wish I'd had.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Noor%20Akhter-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/noorakhter)
[![GitHub](https://img.shields.io/badge/GitHub-noor--akhter-181717?style=flat&logo=github)](https://github.com/noor-akhter)

Always open to feedback on the project and conversations about PM roles.
