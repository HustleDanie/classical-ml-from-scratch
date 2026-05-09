# Expert Scenario 35: Gene Function Prediction

> **Compact.** Multilabel with hierarchy; analog to [33_document_tags_multilabel.md](33_document_tags_multilabel.md) and [69_icd10_codes.md](69_icd10_codes.md).

```
Type: Multilabel, 100-10K functional categories, sparse positives per gene
Approach: Per-label Logistic Regression with L2; structured-output methods if hierarchy is rich
Metric: Per-class AUC; micro-AUPRC
```

**Distinct:** label hierarchy (parent functions imply children) must be respected.
