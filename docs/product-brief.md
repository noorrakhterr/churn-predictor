# Product Brief — Churn Predictor

## Problem Statement

### Who is the customer?

Customer Success Managers (CSMs) at mid-market B2B SaaS companies managing
20–80 accounts each, with renewals spread across the year.

### What job are they trying to do?

Identify at-risk accounts early enough to run a meaningful save play — not
react after a churn notice has already been submitted.

### What is the pain today?

- Account health is tracked manually in spreadsheets or CRM notes.
- CSMs rely on gut feel and relationship signals, or have to manually check in on data, missing product-usage patterns and overlapping data indicators
- Leadership has no portfolio-level view of churn risk heading into a quarter.

---

## Proposed Solution

A lightweight ML-powered dashboard that:

1. **Scores every account weekly** on a 0–100 churn-risk scale.
2. **Explains the top 3 risk drivers** per account in plain language (powered by SHAP).
3. **Prioritises the save list** so CSMs know which accounts to call first.
4. **Gives leadership a heat map** of ARR at risk by segment and renewal month.

---

## Success Metrics

| Metric | Baseline | Target (6 months) |
|---|---|---|
| Churn rate (% of ARR) | ~6% | < 4% |
| CSM response time to at-risk signal | > 14 days | < 5 days |
| Model precision @ top decile | — | > 70% |
| CSM adoption of dashboard | 0% | > 60% weekly active |

---

## Assumptions & Risks

| Assumption | Risk if wrong |
|---|---|
| Usage data is reliably logged | Model degrades; need data quality initiative |
| CSMs will trust ML scores | Low adoption; need explainability + quick wins |
| 30-day warning is enough to run a save play | Need to re-scope to 60-day window |
| Synthetic training data generalises | Model underperforms on real data; need labelled history |

---

## Out of Scope (v1)

- Automated outreach / email triggers
- Contract value prediction
- Upsell / expansion scoring
- Integration with billing systems

---

## Open Questions

1. What is the minimum labelled history needed before we can train on real data?
2. Which CRM do target CSMs use — HubSpot or Salesforce?
3. Should risk scores feed into QBR decks automatically?

---

## Competitive Landscape

| Tool | Strength | Gap |
|---|---|---|
| Gainsight | Deep CSM workflow integration | Expensive; black-box scores |
| Totango | Easy to set up health scores | Limited ML; rule-based |
| ChurnZero | Real-time alerts | No explainability layer |
| **This tool** | Explainable ML + open source | No native CRM integration (yet) |
