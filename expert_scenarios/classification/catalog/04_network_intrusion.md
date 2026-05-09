# Expert Scenario 4: Network Intrusion Detection

> **Compact.** Adjacent to [01_fraud_detection.md](01_fraud_detection.md) (severe imbalance, real-time scoring) and [09_bot_detection.md](09_bot_detection.md) (adversarial). Distinct: SOC analyst alert budget caps daily inspections; multiclass for attack-type routing.

```
Type:           Binary OR multiclass classification (5 attack types + normal)
Imbalance:      ~93/7 binary; per-attack-type 0.5-3%
Metric:         Recall at fixed alert volume (analyst budget)
Constraint:     Sub-100ms; weekly retraining; analyst labeling latency
```

**Approach:** Random Forest or LightGBM with `class_weight='balanced'`; threshold tuned to "top-100 alerts/day" budget. Per-attack-type recall audit. Adversarial model drift handling like scenario 9.

**Distinct vs scenario 1 (fraud):**
- Multiclass output (which attack type → different response playbook)
- Network features (packet size, port distribution, protocol entropy)
- Adversaries deliberately mimic normal traffic (less applicable to fraud)

**Distinct vs scenario 9 (bot):**
- Network packet/flow features instead of account behavioral features
- Real-time alerting vs at-signup classification
- Multi-attack-type routing matters

| Beginner | Expert |
|----------|--------|
| F1 optimum threshold | Top-K alerts/day budget |
| Single binary model | Multiclass for attack-type routing |
| Static threshold | Weekly retrain + drift detection |
