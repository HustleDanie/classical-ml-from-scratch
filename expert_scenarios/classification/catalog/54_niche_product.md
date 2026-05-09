# Expert Scenario 54: Niche-Product Repeat-Purchase Prediction

> **Compact.** Small-data variant of [60_repeat_purchase.md](60_repeat_purchase.md). See [52_rare_disease_diagnosis.md](52_rare_disease_diagnosis.md) for small-data CV techniques.

```
Type: Binary, small data (200-5K customers)
Approach: Logistic Regression with L1; Decision Tree for interpretability
```

**Distinct:** look-alike features (similar to known repeat buyers) more useful than past purchase counts when N is tiny.
