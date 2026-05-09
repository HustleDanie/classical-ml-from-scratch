# Expert Scenario 22: Document Type Classification

> **Compact.** Multiclass text + structural features. Closest analog [18_news_topic_classification.md](18_news_topic_classification.md).

```
Type: Multiclass, 5-15 classes
Approach: Logistic Regression Softmax on TF-IDF + structural features (page count, layout)
```

**Distinct:** OCR errors are noise — clean OCR pipeline before tuning model.
