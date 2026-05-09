# Expert Scenario 13: Blood Pressure Prediction

> **Compact.** Clinical regression. See [02_hospital_los.md](02_hospital_los.md).

```
Type: Regression on systolic/diastolic mmHg
Approach: Random Forest baseline; longitudinal models if multi-visit data exists
Metric: MAE in mmHg
```

**Distinct:** white-coat hypertension — measurement context matters; flag clinic-only readings.
