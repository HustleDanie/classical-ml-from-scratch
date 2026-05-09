# Expert Scenario 21: Customer Support Ticket Routing

> **Compact.** Multiclass text; longer-form than [19_intent_detection.md](19_intent_detection.md), more like [18_news_topic_classification.md](18_news_topic_classification.md).

```
Type:           Multiclass text, 10-30 departments, uneven volume
Metric:         Macro-F1; routing accuracy weighted by ticket volume
```

**Approach:** Logistic Regression Softmax with `class_weight`. Avoid SVM with OvO (66+ binary models for 12 classes).

**Distinct:** Department definitions evolve; some departments split / merge. Maintain mapping table; retrain monthly.

| Beginner | Expert |
|----------|--------|
| Static dept definitions | Track dept changes; flag re-mapping |
| OvO SVM | Logistic Regression Softmax (much faster) |
