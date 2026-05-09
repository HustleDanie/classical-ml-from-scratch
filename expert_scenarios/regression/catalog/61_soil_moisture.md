# Expert Scenario 61: Soil Moisture Prediction

> **Compact.** Geospatial + weather. See [60_crop_yield_forecast.md](60_crop_yield_forecast.md).

```
Type: Regression on % moisture
Approach: Linear regression on weather lags; Gradient Boosting challenger
Metric: MAE in percentage points
```

**Distinct:** sensor calibration drift — recent calibration data only.
