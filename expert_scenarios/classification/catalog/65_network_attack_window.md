# Expert Scenario 65: Network Attack Window Detection

> **Compact.** Sliding-window time-series-event multiclass. See [64_equipment_failure_window.md](64_equipment_failure_window.md) for the sliding-window pattern.

```
Type: Multiclass, sliding-window from packet streams
Approach: Random Forest or LightGBM on window features; HMM for sequential patterns
```

**Distinct:** attackers move slowly — windows must overlap multiple hours; multi-stage attack detection.
