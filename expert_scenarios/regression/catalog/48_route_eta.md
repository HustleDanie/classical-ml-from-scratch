# Expert Scenario 48: Route ETA Prediction

> **Compact.** Logistics regression; closest analog [46_delivery_time_prediction.md](46_delivery_time_prediction.md) (food delivery ETA). Distinct: focused on route-segment ETAs at sub-minute precision; integrates with mapping/navigation.

```
Type:           Regression on travel time (seconds)
Constraint:     Real-time; per-corridor models for high-volume routes
Metric:         MAE in seconds; P95 over-arrival rate (don't promise too tight)
```

**Approach:** LightGBM with route + time-of-day + traffic features. Per-corridor models for top-100 high-volume routes. Asymmetric loss biased to under-promise.

**Distinct concerns:**
- Sub-minute precision matters (ride-share ETAs shown to passengers).
- Time-of-day non-linearities are extreme (rush hour vs 3am).
- Routes have different sensitivity to weather (highway vs urban).

| Beginner | Expert |
|----------|--------|
| Haversine / 30 mph | LightGBM with traffic + weather + route ID |
| Single global model | Per-corridor for top-100 routes |
| Symmetric loss | Asymmetric (penalize under-estimates more) |
