# Expert Scenario 49: Network Intrusion Real-Time Alert

> **Compact.** Real-time variant of [04_network_intrusion.md](04_network_intrusion.md). See [47_realtime_fraud_scoring.md](47_realtime_fraud_scoring.md) for real-time pattern.

```
Type: Binary or multiclass, < 100ms latency
Approach: Random Forest with bounded depth or single calibrated Decision Tree (fast inference)
```

**Distinct:** alert fatigue — analysts ignore high-FP feeds. Threshold pinned to alert budget per analyst per shift.
