# Churn Risk Predictor: "ChurnBurn"

> AI-powered account prioritization for B2B SaaS Customer Success teams

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://churnburn.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=flat)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-6236FF?style=flat)](https://shap.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)

**[Live Demo](https://churnburn.streamlit.app)** · [Project Brief](docs/product-brief.md) · [Model Card](docs/model-card.md)

---

## The Problem

Acquiring a B2B customer costs 5–25× more than retaining one (Bain & Company). Yet most CS teams still manage renewals reactively — a CSM only finds out an account is at risk when the customer goes quiet, stops attending QBRs, or submits a cancellation notice. I built this project after three years as a CSM at Okta, watching colleagues burn cycles on healthy accounts while genuinely at-risk ones slipped through. The signal was always there in the data — seat utilization trends, login gaps, support ticket spikes — but there was no systematic way to surface it.

---

## What This Is

A tool that scores every account's churn risk using machine learning, explains *why* it's at risk using SHAP feature attribution, and maps those drivers to specific CSM actions — not generic advice, but plays grounded in how B2B SaaS accounts actually churn. Built for CSMs who care about which three accounts to call this week, not data scientists who want to tune hyperparameters.

---

## Key Features

- 🎯 **Risk scoring** with calibrated probabilities (0–100%), color-coded by tier (Low / Medium / High)
- 🔍 **Per-account explanations** via SHAP — every score includes the top drivers and protective signals in plain English
- 💡 **Action recommendations** grounded in B2B SaaS CSM domain knowledge, timeframed by urgency (This Week / This Month / This Quarter)
- 📋 **Portfolio Scorer** — upload a CSV of your book of business, get a ranked list with risk tiers, top drivers, and suggested plays
- 📊 **Model transparency** dashboard with full performance metrics, calibration curves, and explicit limitations

---

## Product Decisions Worth Highlighting

**Optimized for recall, not accuracy.** A false negative — a churning account the model misses — means a lost customer and a missed save opportunity. A false positive means one unnecessary CSM call. That asymmetry is why I used PR-AUC as the primary optimization metric (not ROC-AUC, which is misleadingly optimistic on imbalanced data where ~80% of accounts are healthy).

**Synthesized realistic B2B data instead of using Telco.** Standard churn datasets are consumer-telecom — they don't contain the signals B2B CSMs actually act on: seat utilization, exec sponsor changes, QBR attendance, or feature adoption. I designed an 8,000-row synthetic dataset with realistic cross-feature correlations drawn directly from my CSM experience, then validated it against the patterns in the EDA.

**Surfaced SHAP explanations as plain English.** A raw SHAP value of +0.34 is useless to a CSM. Every driver gets translated into qualitative impact tiers ("Strongly increases churn risk") and then mapped to a specific action — so the output is "book an exec intro call within 14 days," not "exec_sponsor_changed_last_180d: 0.34."

**Made Portfolio Scorer a v1 feature, not v2.** Single-account scoring is a nice demo. Bulk prioritization — uploading a 50-account book and getting a ranked action list — is what actually changes a CSM's Monday morning. Scoping it to v1 reflects how I'd prioritize if this were a real product spec.

**Included a Model Performance tab.** Opaque "AI scores" get ignored or gamed. Showing precision, recall, the confusion matrix, calibration curves, and explicit limitations builds the trust with CS leadership that makes adoption possible. Transparency is a product feature.

**Treated the model as table stakes, not the product.** PR-AUC of 0.57 is solid but not magical. The differentiation is the explainability and action layer that sits on top. A churn score alone is a feature; the interpretation layer is what makes it a tool a CSM would actually open on Monday morning.

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
