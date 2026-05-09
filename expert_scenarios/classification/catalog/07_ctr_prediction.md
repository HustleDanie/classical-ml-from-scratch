# Expert Scenario 7: Click-Through Rate (CTR) Prediction

> **Compact walkthrough.** For the deep-dive on FTRL + feature hashing + IPS bias correction, see the closely-related [48_rtb_click_prediction.md](48_rtb_click_prediction.md). This entry covers the **advertiser-side** CTR-prediction problem (different goal: rank user-ad pairs for an ad campaign budget, not bid auctions).

---

## The Brief

An advertiser with $20M monthly budget wants to choose which users (from a remarketing pool of 50M) to show ads to. The platform charges per impression; advertiser pays whether or not the user clicks. The brief: rank users by predicted P(click) so we target only the top-N each day given the budget.

This is different from the RTB scenario in two ways:
1. We're advertiser-side, not platform-side (we don't bid; we choose to target or not).
2. The decision is "who to target" (binary inclusion), not "what bid to set."

---

## Problem Type

```
Type:           Binary classification, calibrated probability + ranking
Primary Metric: Top-K precision; AUC; log loss for calibration
Imbalance:      ~1-3% click rate
Constraint:     Daily budget = N impressions; must rank well
```

---

## Key Differences from RTB Scenario

| Aspect | Advertiser CTR (this scenario) | Platform RTB (scenario 48) |
|--------|-------------------------------|----------------------------|
| Decision | Target / don't target | Bid amount |
| Latency | Batch daily (no auction) | < 10ms |
| Budget | Fixed daily impression count | Variable per-auction bid |
| Selection bias | Less severe (we control sampling) | Severe (only see won impressions) |
| Calibration | Less critical (top-K matters) | Critical (bid math) |

---

## Recommended Approach

- **LightGBM** (not FTRL — we have batch budget, no millisecond constraints)
- Features: standard CTR features (user, ad creative, context, history)
- Metric: top-K precision (does the targeted decile actually click?)
- Threshold: rank-based — pick top-N matching daily budget

---

## What Made This Different

- **No real-time constraint** lets us use richer models (gradient boosting, even ensembles) instead of FTRL.
- **Selection bias is solvable** — we can do exploratory targeting (random 5%) to maintain unbiased training data.
- **Calibration matters less** — we only need ranking quality, not absolute probability for bid math.
- **Diversity-aware targeting** — without it, we'd send the same ad to the same top users repeatedly. Add fatigue features: `times_targeted_30d`.

---

## Summary: Beginner vs Expert

| Technique | Beginner | Expert |
|-----------|----------|--------|
| Model | Treat like RTB; use FTRL | LightGBM (we have time) |
| Calibration | Mandatory | Skip (rank-only) |
| Bias correction | Skip | Maintain 5% random exploration arm |
| Fatigue features | Ignore | Track `times_targeted_30d`, `last_seen_ad_days_ago` |
| Threshold | Probability cutoff | Top-N rank-based to fit budget |
