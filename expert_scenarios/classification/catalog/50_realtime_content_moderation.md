# Expert Scenario 50: Real-Time Content Moderation

> **Compact.** Combines [24_toxicity_moderation.md](24_toxicity_moderation.md) (toxicity model) with real-time constraint from [47_realtime_fraud_scoring.md](47_realtime_fraud_scoring.md).

```
Type: Binary, < 100ms latency
Approach: Two-tier — fast Logistic Regression filter + escalation to transformer for borderline cases
```

**Distinct:** the two-tier pattern is standard in production: fast cheap classifier handles 99% of cases; expensive transformer handles the 1% that's borderline.
