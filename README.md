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

## Key Product Decisions

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

Out of 100 customers who actually churn, the model flags 69. Of every 100 accounts it flags as at-risk, 42 go on to churn. For a CSM managing 50 accounts with a 20% churn rate, that translates to roughly 17 at-risk alerts per quarter which is a relatively a manageable workload that catches 7 out of every 10 accounts that would otherwise be lost.

---

## Data insights from the EDA

Full analysis in [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb).

**Eatly adoption: 0–3 month accounts churn at ~40% vs. ~20% baseline.** First-90-day activation is the highest-ROI retention investment. Onboarding with explicit day-30 and day-60 health checks are key to keeping customers. 

**Contract packaging: Monthly contracts churn at ~33% vs. ~12% for multi-year** → Customers on the monthly base package should be consistently on a CSM's watch list. 

**Seat utilization: Churned avg: 41% vs. retained: 58%).** Utilization should be a highly prioritized health metric with automated CSM alerts when an account drops below ~60%.

**Risk compounds.** One risk signal (low utilization, exec sponsor change, or short tenure) correlates with ~31% churn, two signals are to ~51% and all three is ~72%. 

**Industry: Relatively not a useful predictor (~18–22% churn across all segments).** Industry-specific churn playbooks are not necessary as the more important signals are in customer behavior. 
---

## Structure 

```
┌─────────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────┐    ┌────────────┐
│  Synthetic B2B  │ →  │  Preprocess  │ →  │ XGBoost  │ →  │ SHAP │ →  │ Streamlit  │
│  SaaS Data (8K) │    │  Pipeline    │    │  Model   │    │      │    │     UI     │
│  (generate_data)│    │  (sklearn)   │    │          │    │      │    │            │
└─────────────────┘    └──────────────┘    └──────────┘    └──────┘    └────────────┘
```

1. **Data generation**: 8,000 synthetic B2B SaaS accounts with realistic churn correlations
2. **Preprocessing**: utilized sklearn `ColumnTransformer` fitted on train only
3. **Training**: 5-fold stratified CV baseline across Logistic Regression, Random Forest, and XGBoost; `RandomizedSearchCV` tunes the winner on PR-AUC
4. **Explanations** — SHAP `TreeExplainer` generates prediction based attributions; a translation layer maps raw values to plain-English tiers and CSM action recommendations 
5. **UI** — Streamlit allows for three separate workflows: single-account scoring, portfolio prioritization, and model transparency

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

## Next steps for v2

- **Recommendation feedback loop**: the app could track which actions a CSM took and whether the account churned anyway, then weight recommendations by historical effectiveness
- **Portfolio centric**: ideally, a CSM would seldom use the manual individual account tracker but rather upload their entire book of business. A v2 product would ultimately center around this.
- **Time-series signals**: current model is a point-in-time snapshot; v2 would incorporate trend features (utilization trending down over 3 months is a stronger signal than a single low reading)
- **Segment-specific models** : enterprise and SMB accounts have different churn drivers, so a single global model ultimately misses the nuance. I've mostly worked with SMB accounts so I based the model around this but it can be divided.

---

## About

I'm a current a Customer Success Specialist at Okta working on the Silver Scale side, looking to transitioning into Product Management. I built this project to demonstrate three things: the customer expertise I bring from working directly with B2B SaaS customers, the product skills to frame a technical problem around user needs, and the technical fluency to ship a working ML product from scratch and I built the tool I wish I'd had! 
