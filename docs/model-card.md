# Model Card: Churn Risk Predictor v1.0

## Overview
XGBoost classifier predicting 90-day customer churn for B2B SaaS
accounts. Trained on synthetic data designed to mirror real
mid-market SaaS churn patterns.

## Intended Use
- Primary: Help CSMs prioritize their book of business weekly
- Secondary: Surface patterns for product and CS leadership reviews

## Out-of-Scope Use
- Punitive pricing or contract decisions
- Hiring/firing decisions for CSMs based on book performance
- Automated customer-facing communications without human review

## Training Data
- 6,400 synthetic B2B SaaS accounts (80% of 8,000)
- ~20% churn rate (mid-market B2B SaaS realistic baseline)
- Features: usage, engagement, commercial, firmographic signals

## Performance (test set, n=1,600, threshold=0.50)

| Metric | Score |
|---|---|
| ROC-AUC | 0.8022 |
| PR-AUC | 0.5719 |
| Recall (churn class) | 0.6894 |
| Precision (churn class) | 0.4157 |
| F1 (churn class) | 0.5187 |

### Threshold sensitivity
The default 0.50 threshold is tuned for recall (missing a churn is worse
than a false alarm). Raising to the F1-optimal threshold of **0.65**
trades recall for precision:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 (default) | 0.42 | 0.69 | 0.52 |
| 0.65 (F1-optimal) | 0.54 | 0.54 | 0.54 |

## Known Limitations
- Synthetic data; real-world distribution shift expected
- No seasonality or macro-event modeling
- Does not account for product changes that may shift drivers
- Snapshot-in-time prediction; no time-series component

## Ethical Considerations
- Should augment, not replace, CSM judgment
- Risk scores should never be exposed directly to customers
- Re-evaluate quarterly to detect drift

## Maintenance
- Recommended retraining cadence: monthly (with real data)
- Monitor for concept drift via PSI on input distributions
- Re-evaluate threshold quarterly based on CSM feedback
