# Expert Scenario 12: Drug Dosage Recommendation

> **Compact.** Regulated regression. See [02_hospital_los.md](02_hospital_los.md) for clinical pattern.

```
Type: Regression, regulated, small-medium data
Approach: Linear regression with patient-specific features; XGBoost with monotonic constraints
Metric: MAE; clinically-relevant deviation (within ±10% of optimal)
```

**Distinct:** dose-response is non-linear and patient-specific — Bayesian methods preferred.
