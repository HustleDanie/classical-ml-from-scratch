# Expert Scenario 5: Manufacturing Defect Binary Classification

> **Compact.** Binary view of [04_manufacturing_defect.md](04_manufacturing_defect.md) (multiclass, 7 defect types). Use the binary form when "yes/no defect" is sufficient and corrective actions are similar.

```
Type:           Binary classification (defect=1, OK=0)
Imbalance:      ~1-5% defect rate
Metric:         Recall at fixed false-alarm rate (production line tolerance)
```

**Approach:** Same model family as scenario 4 but simpler output. LightGBM with `is_unbalance=True` + threshold tuning.

**When binary vs multiclass:** binary suffices if the corrective action is "stop the line and inspect" regardless of defect type. Multiclass matters if different defects route to different repair stations.

| Binary | Multiclass |
|--------|-----------|
| Single threshold | Per-class threshold |
| Stop-line action | Defect-type routing |
| Simpler deployment | Specialized teams |
