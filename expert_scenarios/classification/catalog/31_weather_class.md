# Expert Scenario 31: Weather Class Classification

> **Compact.** Multiclass tabular + geospatial. See [27_power_grid_load_forecast.md](../../regression/catalog/27_power_grid_load_forecast.md) for weather-driven modeling.

```
Type: Multiclass, 4-8 weather classes
Approach: XGBoost/LightGBM with geospatial + sensor features
```

**Distinct:** rare classes (stormy) cluster in time and space. Per-region threshold tuning helps.
