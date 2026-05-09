# Expert Scenario 28: Oil / Gas Demand Forecast

> **Compact.** Time-series with macro drivers; closest analog [27_power_grid_load_forecast.md](27_power_grid_load_forecast.md). Distinct: macro / geopolitical events dominate.

```
Type:           Regression on barrels/MMBtu per day
Metric:         MAPE per region
Constraint:     Geopolitical event handling required
```

**Approach:** SARIMA baseline + Gradient Boosting on macro features (commodity prices, GDP, weather). Explicit event flags for OPEC announcements, sanctions, hurricanes.

**Distinct:** geopolitical events (OPEC cuts, sanctions, war) cause sudden distribution shifts. Event-flag features help; consider regime-based models.

| Beginner | Expert |
|----------|--------|
| Pure time-series | + macro indicators + event flags |
| One model | Regime-aware variants for crisis periods |
