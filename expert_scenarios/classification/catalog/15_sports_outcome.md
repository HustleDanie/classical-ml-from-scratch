# Expert Scenario 15: Sports Match Outcome (Win/Loss)

> **Compact.** Binary form of [58_game_score_prediction.md](../../regression/catalog/58_game_score_prediction.md).

```
Type: Binary, balanced
Metric: Accuracy or log loss (if betting odds downstream)
```

**Approach:** Logistic Regression baseline, Gradient Boosting for accuracy. Temporal split mandatory.

| Beginner | Expert |
|----------|--------|
| Random split | Time-based |
| Single match | Use team form features (last-10) |
