# Expert Scenario 57: Athlete Performance Forecast

> **Compact.** Closest analog [58_game_score_prediction.md](58_game_score_prediction.md) (team scores). Distinct: per-athlete season stats; small sample per athlete.

```
Type:           Regression on stats (points, yards, etc.)
Metric:         MAE; rank correlation
```

**Approach:** Bayesian shrinkage toward position mean; XGBoost for raw fit.

**Distinct:** small samples (1-15 seasons per athlete) → high variance. Multi-season averaging helps. Position-specific models capture different stat dynamics.

| Beginner | Expert |
|----------|--------|
| Per-athlete model | Bayesian shrinkage toward position mean |
| Single season | Multi-season averaging |
