# Expert Scenario 72: Generic Prediction Intervals

> **Compact.** Conformal prediction wrapper for any regression. See [70_p99_latency_prediction.md](70_p99_latency_prediction.md).

```
Type: P10/P50/P90 quantile regression
Approach: Gradient Boosting with quantile loss for each quantile; conformal wrapper
Metric: Coverage; interval width
```

**Distinct:** conformal prediction needs a proper holdout — don't bake into training.
