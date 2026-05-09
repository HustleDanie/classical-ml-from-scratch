# Expert Scenario 23: Manufacturing Defect Rate Prediction

> **Compact.** Bounded regression; analog to [21_manufacturing_yield.md](21_manufacturing_yield.md) but inverted (higher yield = lower defects).

```
Type: Regression on defect rate (DPM/PPM)
Approach: Lasso for feature selection in 200+ sensor space; LightGBM challenger
Metric: MAE in DPM/PPM
```

**Distinct:** zero-defect runs dominate; consider Tweedie or log(rate + epsilon).
