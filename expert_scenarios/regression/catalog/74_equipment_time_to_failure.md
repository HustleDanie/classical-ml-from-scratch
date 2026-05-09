# Expert Scenario 74: Equipment Time-to-Failure (Survival)

> **Compact.** Survival regression; analog [73_customer_time_to_churn_survival.md](73_customer_time_to_churn_survival.md) but for industrial equipment.

```
Type:           Survival regression (Remaining Useful Life)
Metric:         Concordance index; MAE on uncensored holdout
```

**Approach:** Random Survival Forest; Cox PH baseline.

**Distinct:** preventive maintenance is right-censoring on purpose (replaced before failure). Must encode as censored, not as failure events.

| Beginner | Expert |
|----------|--------|
| Treat preventive replacement as failure | Encode as censored (different event type) |
| Single model | Per-equipment-class models |
