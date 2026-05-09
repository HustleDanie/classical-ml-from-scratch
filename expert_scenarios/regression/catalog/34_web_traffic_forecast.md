# Expert Scenario 34: Web Traffic Forecast

> **Compact.** Time-series, multi-seasonality, occasional viral spikes. See [31_daily_sales_forecast.md](31_daily_sales_forecast.md).

```
Type: Regression on page views per hour
Approach: SARIMA baseline; Gradient Boosting for non-linear effects
Metric: MAPE; P95 accuracy
```

**Distinct:** viral spikes are unpredictable — separate baseline model + spike-handling logic with anomaly tier.
