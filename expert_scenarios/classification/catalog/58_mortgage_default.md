# Expert Scenario 58: Mortgage Default Prediction

> **Compact.** Closest analog [73_loan_risk_scoring_calibrated.md](73_loan_risk_scoring_calibrated.md) (calibrated lending). Distinct: long observation horizons (30-year mortgages), economic regime sensitivity.

```
Type:           Binary classification, moderate imbalance
Metric:         AUC; KS statistic; calibration
Constraint:     Regulated; economic regime changes break models
```

**Approach:** Logistic Regression with L2 + monotonic XGBoost challenger (per scenario 73 pattern). Time-based split with both COVID-period and rate-hike-period in test set.

**Distinct:** mortgage outcomes play out over years, not months. Survivorship bias is severe (early defaulters censored differently than late). Consider survival regression as alternative.

| Beginner | Expert |
|----------|--------|
| Random split | Time-based with multiple economic regimes |
| Logistic only | Logistic baseline + monotonic XGBoost; pick by calibration |
