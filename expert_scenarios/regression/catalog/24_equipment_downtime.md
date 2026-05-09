# Expert Scenario 24: Equipment Downtime Prediction

> **Compact.** Right-skewed, censored. See [74_equipment_time_to_failure.md](74_equipment_time_to_failure.md) (survival pattern).

```
Type: Regression, right-skewed, censored
Approach: Survival regression for time-to-failure; Gradient Boosting for raw downtime
Metric: MAE in hours; concordance
```

**Distinct:** maintenance schedules confound — censor or model explicitly.
