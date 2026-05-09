# Expert Scenario 56: Lab Experiment Outcome Classification

> **Compact.** Tiny-N classification. See [52_rare_disease_diagnosis.md](52_rare_disease_diagnosis.md) for small-data techniques.

```
Type: Binary, very small data (50-500 experiments)
Approach: Logistic Regression with L2; report effect sizes with confidence intervals
```

**Distinct:** with N=50, almost any model looks good in CV — Bayesian methods + bootstrap CIs are essential. Beware overfitting.
