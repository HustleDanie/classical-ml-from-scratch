# Expert Scenario 50: Package Volume / Weight Estimation

> **Compact.** Geospatial + product-driven. See [46_delivery_time_prediction.md](46_delivery_time_prediction.md).

```
Type: Regression on package volume
Approach: Gradient Boosting; product-category × dimension lookup as baseline
Metric: MAPE
```

**Distinct:** mis-declared weights (carrier surcharges) — labeled training data may have noise.
