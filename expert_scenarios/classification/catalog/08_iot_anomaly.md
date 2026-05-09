# Expert Scenario 8: IoT Sensor Anomaly Detection

> **Compact.** Extreme imbalance (< 0.1% labeled anomalies); often best framed as one-class learning. Adjacent to [01_fraud_detection.md](01_fraud_detection.md) (extreme imbalance) and [64_equipment_failure_window.md](64_equipment_failure_window.md) (sensor data).

```
Type:           One-class or binary, < 0.1% labeled anomalies
Metric:         PR-AUC; precision at top-k flagged readings
Constraint:     Massive data volume (1B+ readings)
```

**Approach:** Isolation Forest or one-class SVM as baseline (no labels needed); supervised LightGBM if labels are reliable.

**Distinct concerns:**
- Most "anomalies" in unlabeled data are benign sensor glitches
- True anomalies are extremely rare; few-shot or active learning helps
- Streaming data: model must update online

| Beginner | Expert |
|----------|--------|
| Train supervised on rare labels | Use Isolation Forest first; supervised only if labels reliable |
| Static threshold | Per-device rolling threshold |
| Ignore unlabeled data | One-class learning leverages it |
