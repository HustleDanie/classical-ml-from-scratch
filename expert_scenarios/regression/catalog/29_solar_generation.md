# Expert Scenario 29: Solar Power Generation Forecast

> **Compact.** Weather-driven time-series; closest analog [27_power_grid_load_forecast.md](27_power_grid_load_forecast.md).

```
Type:           Regression on kWh per hour
Metric:         MAPE; peak-hour accuracy
Constraint:     Physical limit (panel rating); clip predictions
```

**Approach:** LightGBM on weather features (cloud cover, irradiance, temperature). Cap predictions at panel max output.

**Distinct:** physics-bounded ceiling. Cloud cover is the dominant feature; irradiance forecasts from NWP are essential.

| Beginner | Expert |
|----------|--------|
| Predict freely | Clip at panel max output rating |
| Skip irradiance | Irradiance forecast is dominant feature |
