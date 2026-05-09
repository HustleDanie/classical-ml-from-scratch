# Expert Scenario 7: Bond Yield Prediction

> **Compact.** Adjacent to [06_stock_return_forecast.md](06_stock_return_forecast.md). Distinct: yields driven by macro factors (Fed rate, CPI, term structure); regime changes break models.

```
Type:           Regression on yield (bps)
Metric:         RMSE in bps; directional accuracy
```

**Approach:** Linear baseline + XGBoost; macro features (Fed funds rate, CPI, term spreads) dominate. Walk-forward CV.

**Distinct:** regime changes (QE → tightening) break models trained on prior regime.

| Beginner | Expert |
|----------|--------|
| Use price-based features | Macro factors (Fed rate, CPI, spreads) dominate |
| One global model | Per-regime model variants |
