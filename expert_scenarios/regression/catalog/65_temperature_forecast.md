# Expert Scenario 65: Local Temperature Forecast

> **Compact.** Time-series weather forecasting; closest analog [27_power_grid_load_forecast.md](27_power_grid_load_forecast.md) (uses weather as input). Distinct: predicting weather itself, not consequences.

```
Type:           Regression on temperature (°C)
Constraint:     Classical ML cannot beat NWP (numerical weather prediction); use as bias correction
Metric:         MAE in °C
```

**Approach:** ALWAYS use NWP forecast as the dominant feature. Classical ML adds value only as bias correction:
- `predicted_temp = nwp_forecast + classical_residual_correction`
- Train on `(actual - nwp_forecast)` as the target

**Distinct concerns:**
- NWP itself has uncertainty; bias-correction model captures local effects (urban heat island, microclimate).
- Per-station bias correction (each weather station has its own systematic NWP bias).
- Don't expect to beat NWP by much; 0.3-0.8°C MAE improvement is realistic.

| Beginner | Expert |
|----------|--------|
| Train classical to compete with NWP | Use NWP as feature; correct residual bias |
| One global model | Per-station bias correction models |
| Mean prediction | Quantile bands for "feels like" range |
