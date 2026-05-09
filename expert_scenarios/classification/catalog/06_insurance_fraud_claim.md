# Expert Scenario 6: Insurance Fraud Claim Detection

> **Compact.** Adjacent to [01_fraud_detection.md](01_fraud_detection.md) (financial fraud) but for insurance claims. Distinct: investigator workflow (top-K daily inspections), label uncertainty (only flagged claims get confirmed).

```
Type:           Binary classification, 1-3% fraud rate
Metric:         Precision at top-K (investigator hit rate)
Constraint:     Daily investigation budget = K claims
```

**Approach:** XGBoost + SHAP. Top-K precision rather than F1.

**Distinct concerns:**
- Investigators only confirm flagged claims; "non-fraud" labels are noisy
- Adjuster notes (free text) often contain key signals — TF-IDF those notes
- Claim age matters (older claims harder to investigate)

| Beginner | Expert |
|----------|--------|
| F1 threshold | Top-K daily budget |
| Ignore notes | TF-IDF on adjuster notes |
| Trust labels | Acknowledge label uncertainty |
