# Expert Scenario 52: Cohort Graduation Rate Forecast

> **Compact.** Bounded regression on % graduating; closest analog [51_student_grade_prediction.md](51_student_grade_prediction.md) (per-student) but at cohort aggregate level.

```
Type:           Regression on bounded [0, 1] target
Metric:         MAE in percentage points
```

**Approach:** Beta regression for bounded target; Random Forest as alternative.

**Distinct:** Simpson's paradox risk — institution-level vs student-level relationships diverge. Track both.

| Beginner | Expert |
|----------|--------|
| Linear regression | Beta regression for bounded target |
| Aggregate only | Audit for Simpson's paradox |
