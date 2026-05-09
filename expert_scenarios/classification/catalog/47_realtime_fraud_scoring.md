# Expert Scenario 47: Real-Time Fraud Scoring

> **Compact.** The real-time variant of [01_fraud_detection.md](01_fraud_detection.md). Identical fraud-detection patterns but emphasizes inference latency and feature precomputation.

```
Type:           Binary classification, severe imbalance
Constraint:     < 50ms inference at transaction approval
Metric:         PR-AUC; recall at fixed FPR
```

**Approach:** Same model as scenario 1 (LightGBM + scale_pos_weight + threshold) BUT with bounded depth (8-10) and feature precomputation in a streaming feature store.

**Distinct (vs batch fraud detection):**
- Feature store precomputes velocity features (last-5-transaction patterns, account age) so inference is < 50ms.
- Model serving: gRPC service with model loaded once at startup; sparse-input avoidance.
- Feature freshness monitoring: alert if feature > 60s stale.

| Beginner | Expert |
|----------|--------|
| Compute features inline | Precompute in feature store; gRPC lookup |
| Re-load model per request | Model hot-loaded; reused across requests |
| Skip feature freshness | Alert on stale features |
