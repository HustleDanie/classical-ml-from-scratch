# Expert Scenario 45: Medical Diagnosis (Binary Disease Detection)

> **Compact.** Combines [71_cancer_screening.md](71_cancer_screening.md) (cost-sensitive sensitivity-at-fixed-specificity) and [29_icu_severity_triage.md](29_icu_severity_triage.md) (clinical regulated). Distinct: simpler binary output for non-screening diagnostic.

```
Type:           Binary classification, regulated, real-time
Metric:         Sensitivity at fixed specificity; per-disease AUC
Constraint:     FDA SaMD; calibrated probabilities
```

**Approach:** Logistic Regression with L2 + SHAP (interpretability mandatory) + ensemble challenger. Same pattern as scenario 71 but applied to single-disease diagnostic.

**Distinct vs scenario 71 (cancer screening):** simpler binary output; threshold determines positive prediction directly.

| Beginner | Expert |
|----------|--------|
| Threshold = 0.5 | Sensitivity at fixed specificity tuned |
| Black-box ensemble | Logistic Regression + SHAP for FDA audit |
