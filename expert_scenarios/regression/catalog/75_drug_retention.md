# Expert Scenario 75: Drug Retention / Persistence

> **Compact.** Survival regression. See [73_customer_time_to_churn_survival.md](73_customer_time_to_churn_survival.md) and [74_equipment_time_to_failure.md](74_equipment_time_to_failure.md).

```
Type: Survival regression on time on treatment
Approach: Cox PH; Kaplan-Meier baseline
Metric: Concordance; treatment-arm comparisons
```

**Distinct:** loss-to-followup is informative censoring — confounded with outcome (sick patients drop out).
