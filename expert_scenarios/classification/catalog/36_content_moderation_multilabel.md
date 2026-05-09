# Expert Scenario 36: Content Moderation Multilabel

> **Compact.** Multilabel violation detection. See [24_toxicity_moderation.md](24_toxicity_moderation.md) (binary toxicity) and [33_document_tags_multilabel.md](33_document_tags_multilabel.md) (multilabel patterns).

```
Type: Multilabel binary, 5-30 violation types per item
Approach: Binary Relevance with Logistic Regression per violation; calibrated thresholds per class
```

**Distinct:** policies change quarterly — bake in retraining cadence; per-violation human review queues.
