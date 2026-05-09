# Expert Scenario 76: Rare Exam Findings (Long-Tail Multilabel)

> **Compact.** Long-tail multilabel medical. See [41_xray_findings.md](41_xray_findings.md) (multilabel medical) and [77_long_tail_ecommerce_category.md](77_long_tail_ecommerce_category.md) (long-tail patterns).

```
Type: Multilabel binary, 100+ findings (head-tail split)
Approach: Per-finding Logistic Regression with L2 + per-class threshold; freeze rare-finding training when data is too sparse
```

**Distinct:** rare findings (< 5 examples) are unlearnable — keep an "alert radiologist for human review" path; track radiologist confirmations to grow training set.
