# Expert Scenario 59: Customer Upsell Probability

> **Compact.** Closely related to [44_ad_lift_incrementality.md](../../regression/catalog/44_ad_lift_incrementality.md) (uplift modeling). For revenue-incremental upsell decisions, prefer uplift modeling over plain classification.

```
Type: Binary, mild imbalance; better as uplift problem
Approach: LightGBM baseline; uplift modeling (causal) for offer ROI
```

**Distinct:** "would have bought anyway" overlap dilutes apparent lift — measure incremental, not absolute conversion.
