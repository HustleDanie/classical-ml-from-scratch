# Expert Scenario 14: Customer Satisfaction (CSAT) Binary

> **Compact.** Standard binary classification; closest analog [11_titanic_survival.md](11_titanic_survival.md) (balanced binary, mixed-type tabular).

```
Type:           Binary, balanced (~45/55)
Metric:         Accuracy or F1
```

**Approach:** Logistic Regression or Random Forest with SHAP for top dissatisfaction drivers.

**Distinct:** Non-response bias — happy customers respond more than dissatisfied ones. The training data is biased toward positive responses.

| Beginner | Expert |
|----------|--------|
| Trust survey response | Account for response bias; weight unrepresented |
