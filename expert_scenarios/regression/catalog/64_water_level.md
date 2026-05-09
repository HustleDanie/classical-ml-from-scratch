# Expert Scenario 64: Water Level Forecast

> **Compact.** Time-series, weather-driven. See [63_air_quality_index.md](63_air_quality_index.md) (environmental sensor pattern).

```
Type: Regression on water level (meters)
Approach: SARIMA + Gradient Boosting on weather
Metric: MAE in meters; flood-event recall
```

**Distinct:** rare flood events drive impact — weighted loss favoring extreme cases.
