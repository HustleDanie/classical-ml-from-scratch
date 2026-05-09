# Expert Scenario 70: Security Alert Triage (Cost-Sensitive)

> **Compact.** Cost-asymmetric variant of [01_fraud_detection.md](01_fraud_detection.md). FN cost (missed breach) ≫ FP cost (analyst time on false alarm).

```
Type:           Binary classification with explicit cost matrix
Metric:         Expected cost per scoring window
Cost ratio:     FN_cost / FP_cost typically 100-1000x
```

**Approach:** XGBoost; threshold = argmin(FN×cost_FN + FP×cost_FP) from PR curve.

**Distinct:** business-critical breach prevention; cost ratio shifts when business changes (re-tune threshold quarterly).

| Beginner | Expert |
|----------|--------|
| F1 threshold | Cost-function threshold |
| Static cost ratio | Quarterly cost ratio update |
