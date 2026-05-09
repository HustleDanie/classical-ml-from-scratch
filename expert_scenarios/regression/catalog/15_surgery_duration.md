# Expert Scenario 15: Surgery Duration Prediction

> **Compact.** Operational scheduling regression. See [02_hospital_los.md](02_hospital_los.md).

```
Type: Regression on minutes
Approach: Gradient Boosting; per-surgery-type models for high-volume procedures
Metric: MAE in minutes
```

**Distinct:** surgeon-specific effects — random effects model or include surgeon ID with smoothing.
