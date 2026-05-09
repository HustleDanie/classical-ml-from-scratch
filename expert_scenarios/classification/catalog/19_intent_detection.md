# Expert Scenario 19: Customer Support Intent Detection

> **Compact.** Multiclass text; closest analog [21_ticket_routing](21_ticket_routing.md). Distinct: short utterances (chatbot single message) rather than full tickets.

```
Type:           Multiclass text, 10-30 intents, imbalanced
Metric:         Macro-F1; per-intent recall
```

**Approach:** Logistic Regression Softmax on char + word n-grams (short text needs char features).

**Distinct:** Rare intents (< 1%) need either oversampling OR fallback-to-human routing. Out-of-domain detection (utterance doesn't match any intent) is a separate binary head.

| Beginner | Expert |
|----------|--------|
| Word features only | Word + char n-grams (short text) |
| Force every utterance to an intent | Out-of-domain head + fallback to human |
