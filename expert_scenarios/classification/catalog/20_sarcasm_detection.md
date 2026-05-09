# Expert Scenario 20: Sarcasm / Irony Detection

> **Compact.** Hard text task; classical ML caps at ~70-75%; transformer needed for production. See [12_sentiment_classification.md](12_sentiment_classification.md) for similar patterns.

```
Type: Binary text, balanced
Metric: F1; human ceiling is 70-80% on subtle cases
```

**Distinct:** dataset bias is severe (most "sarcasm" datasets are headline-style, not real conversation). Transfer learning rarely works between datasets.
