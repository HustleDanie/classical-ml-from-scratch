# Expert Scenario 35: Ride Volume Forecast

> **Compact.** Time-series + geospatial; combines [31_daily_sales_forecast.md](31_daily_sales_forecast.md) (multi-seasonality + LightGBM lag features) and [03_ride_pricing.md](03_ride_pricing.md) (rideshare-specific dynamics). Distinct: zone × hour granularity, weather + event effects compound.

```
Type:           Regression on rides per zone-hour
Granularity:    1-mile zones × hourly
Metric:         WAPE per-zone; peak-hour MAPE
```

**Distinct features:**
- Recent volume in adjacent zones (geospatial spillover)
- Event flags (concerts, sports, conferences within 5mi)
- Weather forecast 1h ahead
- Day-of-week × time-of-day interactions

**Approach:** LightGBM with lag features (1h, 24h, 7d) + zone-aggregate features + event indicators. Walk-forward CV.

| Beginner | Expert |
|----------|--------|
| Per-zone forecasts ignore spillover | Adjacent-zone aggregate as feature |
| Static event handling | Per-event-type explicit features (concert, sport, conference) |
| Single-model | Hierarchical: zone + region; reconcile aggregates |
