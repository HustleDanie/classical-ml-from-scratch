# Expert Scenario 28: Customer Segment Classification (Marketing Multiclass)

> **Compact walkthrough.** Builds on [26_customer_tier_classification.md](26_customer_tier_classification.md) (ordinal tiers). Distinct nuance: nominal segments (no ordering), often segments come from prior unsupervised clustering — model must respect that history without becoming a self-fulfilling prophecy.

---

## The Brief

A B2C marketing team gives you 4M customers with 6-9 segment labels assigned by an earlier K-Means clustering on engagement + spend + behavior. Segments: "Enthusiasts," "Bargain Hunters," "Lapsed," "New," "Premium Shoppers," "Window Browsers," etc. The goal: classify NEW customers into a segment based on first-30-day behavior, so marketing can route them to the right campaign.

---

## Problem Type

```
Type:           Multiclass classification, 6-9 nominal classes
Primary Metric: Macro-F1; per-segment recall
Secondary:      Confusion matrix interpretation (which segments confuse?)
Imbalance:      Mild (largest 30%, smallest 5%)
```

---

## Distinct Concerns

1. **Segments came from unsupervised clustering** — they reflect data patterns, not ground truth. Model error rates on rare segments may indicate the original clustering was unstable.
2. **Self-fulfilling prophecy risk** — if "Lapsed" customers are routed to a "win-back" campaign, their behavior changes; future training data is contaminated.
3. **Segment drift** — meaningful customer segments shift over months as product / market evolves.

---

## Recommended Approach

- **LightGBM Multiclass** — handles 6-9 classes naturally
- **Per-segment threshold tuning** for rare segments
- **Holdout validation** on segments labeled by domain experts (not just K-Means output) to validate alignment
- **Quarterly re-clustering** with stability check before retraining classifier

---

## Summary

| Technique | Beginner | Expert |
|-----------|----------|--------|
| Validation | Trust K-Means labels | Validate against expert-labeled holdout |
| Drift | Train once | Quarterly re-cluster + retrain |
| Self-fulfilling | Ignore | Track per-segment behavior change post-campaign |
| Output | Just label | Probability vector → marketing can hedge with multi-campaign |
