# Expert Scenario 23: Document Language Identification

> **Compact.** Multiclass text with many classes (25+ languages); character n-grams are the dominant feature.

```
Type:           Multiclass text, 25-100+ languages
Metric:         Per-language accuracy; macro-F1
```

**Approach:** Multinomial Naive Bayes on character n-grams (3-4 grams) is the gold standard — fast and surprisingly hard to beat.

**Distinct:** Related languages (Spanish/Portuguese, Norwegian/Danish) confuse models; need char-3-gram or higher. Short texts (< 20 chars) are inherently ambiguous.

| Beginner | Expert |
|----------|--------|
| Word features | Character n-grams |
| Single-language assumption | Detect mixed-language inputs |
