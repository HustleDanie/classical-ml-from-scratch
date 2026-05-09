# Expert Scenario 19: Fraud Loss Estimation

> **Compact.** Heavy-tailed regression on fraud loss given confirmed fraud. See [05_insurance_claims.md](05_insurance_claims.md).

```
Type: Regression, heavy-tailed
Approach: Gradient Boosting on log loss; Tweedie if zero-inflated
Metric: RMSE on log loss; Gini on ranking
```

**Distinct:** investigation costs may not be in the loss number — confirm with stakeholders before modeling.
