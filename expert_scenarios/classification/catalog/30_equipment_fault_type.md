# Expert Scenario 30: Equipment Fault Type Classification

> **Compact.** Multiclass on sensor data; analog to [04_manufacturing_defect.md](04_manufacturing_defect.md). Distinct: 5-15 fault types each with different repair playbooks.

```
Type:           Multiclass classification, 5-15 fault types + OK
Metric:         Per-fault recall (each fault has different repair playbook)
```

**Approach:** Random Forest or LightGBM multiclass. Per-fault threshold tuning. Maintain "unknown fault" route for new types.

| Beginner | Expert |
|----------|--------|
| Force every prediction to a known type | "Unknown" fallback for new fault patterns |
