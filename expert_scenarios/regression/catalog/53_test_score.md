# Expert Scenario 53: Standardized Test Score Prediction

> **Compact.** Educational regression. See [51_student_grade_prediction.md](51_student_grade_prediction.md).

```
Type: Regression on score (SAT/ACT scale)
Approach: Linear regression with L2; XGBoost challenger
Metric: RMSE on score scale
```

**Distinct:** test prep data confounds prediction — controls for prep are essential.
