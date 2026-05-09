# Expert Scenario 34: Image Tags Multilabel

> **Compact.** Multilabel image classification on engineered/embedding features. See [33_document_tags_multilabel.md](33_document_tags_multilabel.md) for multilabel architecture and [41_xray_findings.md](41_xray_findings.md) for medical image multilabel.

```
Type: Multilabel binary, 10-500 tags per image
Approach: Binary Relevance with Logistic Regression per tag on CNN embeddings
```

**Distinct:** rare tags (< 100 occurrences) are unlearnable from classical features — drop or fold into parent categories. Tag co-occurrence matters; consider Classifier Chains for top-N most-correlated tag pairs.
