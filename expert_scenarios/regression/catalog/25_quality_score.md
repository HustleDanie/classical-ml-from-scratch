# Expert Scenario 25: Quality Score Prediction

> **Compact.** Continuous or ordinal manufacturing quality. See [21_manufacturing_yield.md](21_manufacturing_yield.md).

```
Type: Regression or ordinal multiclass
Approach: Gradient Boosting; ordinal regression if quality is graded
Metric: MAE; quadratic-weighted kappa for ordinal
```

**Distinct:** subjective quality scoring (humans grading) — inter-rater agreement bounds model accuracy.
