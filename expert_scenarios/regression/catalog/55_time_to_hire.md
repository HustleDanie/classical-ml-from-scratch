# Expert Scenario 55: Time-to-Hire Prediction

> **Compact.** Closest analogs [54_salary_prediction.md](54_salary_prediction.md) (HR data) and [73_customer_time_to_churn_survival.md](73_customer_time_to_churn_survival.md) (right-censored time-to-event).

```
Type:           Regression on days-to-fill (right-skewed); censored
Metric:         MAE in days; concordance for censored holdout
```

**Approach:** Gradient Boosting on log(days) for non-censored; Cox PH if censoring is heavy.

**Distinct:** open requisitions that never fill are right-censored. Including them as "long time to hire" biases the model.

| Beginner | Expert |
|----------|--------|
| Treat unfilled as max value | Survival regression handles censoring |
| RMSE | MAE on log(days) |
