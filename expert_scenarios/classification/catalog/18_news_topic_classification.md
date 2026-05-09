# Expert Scenario 18: News Topic Classification

> **Compact.** Multiclass text classification; closest analog [17_email_spam_detection.md](17_email_spam_detection.md) (binary text). Distinct: 4-20 topics, evolving topic distributions over time.

```
Type:           Multiclass text classification, 4-20 topics
Imbalance:      Mostly balanced; some topics drift in volume
Metric:         Macro-F1; per-topic precision
```

**Approach:** Logistic Regression Softmax on TF-IDF (1-2 grams). For 10+ topics, prefer Softmax over OvR for efficiency.

**Distinct concerns:**
- **Topic drift**: election season expands "Politics" share; sports finals expand "Sports."
- **Hierarchical topics**: news taxonomies are often 2-3 levels (Sports → Football → NFL).
- **New topics**: emerging topics (e.g., "Cryptocurrency" pre-2020) need model retraining.

| Beginner | Expert |
|----------|--------|
| OvR with 20 binary models | Softmax (more efficient) |
| Single model | Per-language model variants |
| Retrain quarterly | Track topic distribution drift; alert on > 5% shift |
