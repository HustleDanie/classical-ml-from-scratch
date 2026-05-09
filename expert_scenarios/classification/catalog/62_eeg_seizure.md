# Expert Scenario 62: EEG Seizure Event Detection

> **Compact.** Time-series event classification. See [64_equipment_failure_window.md](64_equipment_failure_window.md) (sliding-window pattern).

```
Type: Binary, severe imbalance, sliding-window
Approach: Random Forest or LightGBM on engineered window features (statistical, frequency-domain)
Metric: Per-event sensitivity; false alarms per hour
```

**Distinct:** subject-level CV mandatory — random split leaks across same patient. Patient-level holdout is the integrity test.
