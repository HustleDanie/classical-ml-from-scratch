# Expert Scenario 12: Sentiment Classification (Movie Reviews)

> **Compact.** Adjacent to [17_email_spam_detection.md](17_email_spam_detection.md) (text classification with TF-IDF + Logistic). Distinct: balanced 50/50 target; sarcasm and negation are key challenges.

```
Type:           Binary text classification (positive/negative)
Imbalance:      Balanced (50/50)
Metric:         Accuracy or F1 (balanced); per-segment accuracy
Constraint:     Real-time at submission; explainability for moderators
```

**Approach:** Logistic Regression with L2 on TF-IDF (1-2 grams) + character n-grams. Per-domain models if reviews span domains (movies vs products vs restaurants).

**Distinct concerns:**
- **Negation handling**: "not good" should reverse polarity. Bigrams catch this.
- **Sarcasm**: classical methods perform poorly; transformer needed for production.
- **Domain shift**: model trained on movie reviews degrades on product reviews.

| Beginner | Expert |
|----------|--------|
| Word unigrams only | Word + character n-grams; bigrams catch negation |
| Single domain | Per-domain models; flag domain shift |
| Use TextBlob | Logistic + L2 + tuned C parameter |
