# Expert Scenario 30: EV Charging Demand

> **Compact.** Time-series with behavioral + weather drivers. See [27_power_grid_load_forecast.md](27_power_grid_load_forecast.md) and [29_solar_generation.md](29_solar_generation.md).

```
Type: Regression on kWh demand
Approach: Gradient Boosting with calendar + weather features
Metric: MAPE; per-station accuracy
```

**Distinct:** rapidly growing market — historical data may not represent current state. Recent data weighted higher.
