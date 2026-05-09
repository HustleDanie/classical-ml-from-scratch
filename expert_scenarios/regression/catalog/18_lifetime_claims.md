# Expert Scenario 18: Lifetime Insurance Claims Forecast

> **Compact.** Combines [16_insurance_claims.md](../../../EXPERT_SCENARIO_5_INSURANCE_CLAIMS.md) (heavy-tail) and [73_customer_time_to_churn_survival.md](73_customer_time_to_churn_survival.md) (survival/censoring).

```
Type:           Regression with right-censoring (most policies still active)
Metric:         Concordance for ranking; MAE on cumulative losses
```

**Approach:** Two-part model — survival regression (time-to-claim using Cox PH) + claim severity model (Tweedie GLM). Multiply for expected lifetime claims.

| Beginner | Expert |
|----------|--------|
| Treat censored as zero claims | Survival-aware: censored = unknown, not zero |
| Single regression | Two-part: time-to-claim × claim-severity |
