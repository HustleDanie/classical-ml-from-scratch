# Expert Scenario 63: ECG Arrhythmia Classification

> **Compact.** Multiclass time-series-event. See [62_eeg_seizure.md](62_eeg_seizure.md) for the patient-level CV pattern.

```
Type: Multiclass, 5-10 arrhythmia types, imbalanced
Approach: Random Forest on engineered ECG features; deep learning for production
Metric: Per-class recall; macro-F1
```

**Distinct:** patient-level CV mandatory; arrhythmias correlated within a patient.
