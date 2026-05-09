# Expert Scenario 14: Recovery Time Prediction

> **Compact.** Right-skewed clinical with censoring. See [02_hospital_los.md](02_hospital_los.md) and [73_customer_time_to_churn_survival.md](73_customer_time_to_churn_survival.md).

```
Type: Regression on days, right-skewed, censored
Approach: Gradient Boosting on log(days); survival models if censoring is heavy
Metric: MAE in days
```

**Distinct:** loss-to-followup is informative censoring — confounded with outcome.
