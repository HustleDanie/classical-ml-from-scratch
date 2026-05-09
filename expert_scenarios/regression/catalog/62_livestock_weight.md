# Expert Scenario 62: Livestock Weight Gain

> **Compact.** Longitudinal animal regression. See [60_crop_yield_forecast.md](60_crop_yield_forecast.md).

```
Type: Regression on weight delta
Approach: Linear regression with random effects per animal; XGBoost challenger
Metric: MAE in kg
```

**Distinct:** breed and feed effects dominate — explicit features required.
