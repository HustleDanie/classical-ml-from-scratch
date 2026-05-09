# Expert Scenario 72: Fraud Cost-Aware Threshold

> **Compact.** Variant of [01_fraud_detection.md](01_fraud_detection.md) with explicit per-row cost matrix. See [70_security_alert_triage.md](70_security_alert_triage.md) for the cost-sensitive pattern.

```
Type: Binary with per-row cost weights
Approach: XGBoost; threshold = argmin(FN×cost_FN + FP×cost_FP) per row
```

**Distinct vs static cost:** cost depends on transaction amount. A $50 fraud miss is different from a $50K fraud miss. Per-row weighting in the loss function captures this.
