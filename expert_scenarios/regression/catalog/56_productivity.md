# Expert Scenario 56: Worker Productivity Prediction

> **Compact.** Mixed-measurement regression. See [54_salary_prediction.md](54_salary_prediction.md) (HR data + fairness).

```
Type: Regression on output per period
Approach: Linear regression with L2; per-team random effects
Metric: MAE; R²
```

**Distinct:** measurement bias — what counts as "productivity" varies by role; per-role model variants.
