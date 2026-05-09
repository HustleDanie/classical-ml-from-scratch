# Expert Scenario 53: Survey Response Classification

> **Compact.** Small-data ordinal multiclass. See [52_rare_disease_diagnosis.md](52_rare_disease_diagnosis.md) (small-data techniques) and [26_customer_tier_classification.md](26_customer_tier_classification.md) (ordinal).

```
Type: Ordinal multiclass (Likert), < 2K responses
Approach: Ordinal Logistic Regression; Random Forest with limited depth
Metric: Quadratic-weighted kappa
```

**Distinct:** response bias — opt-in surveys skew positive. Track non-response rate by demographic.
