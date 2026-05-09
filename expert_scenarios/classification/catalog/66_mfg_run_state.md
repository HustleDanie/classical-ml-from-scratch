# Expert Scenario 66: Manufacturing Run-State Classification

> **Compact.** Multiclass time-series state. See [64_equipment_failure_window.md](64_equipment_failure_window.md).

```
Type: Multiclass (startup / steady / wind-down / fault)
Approach: Random Forest or HMM
```

**Distinct:** state transitions are smooth, not crisp — boundary windows are inherently ambiguous.
