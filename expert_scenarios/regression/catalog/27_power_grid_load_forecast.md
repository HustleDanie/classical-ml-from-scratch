# Expert Scenario 27: Power Grid Load Forecast

> **Complexity:** Real-time grid stability requires forecasts to be unbiased even at peak; weather is the dominant driver but interacts non-linearly with calendar; rare extreme events (heat dome, ice storm) cause catastrophic mispredicts; ensemble with NWP (weather forecast) outputs; peak-hour error costs orders of magnitude more than off-peak.

---

## The Brief

A regional ISO (independent system operator) gives you 7 years of hourly grid load data across 12 zones serving 15M customers. They want a 24-hour-ahead hourly load forecast feeding:

- Generation scheduling (which power plants run at which times).
- Spot market bids (sell excess; buy if shortage).
- Reliability margin sizing (how much spinning reserve to hold).

Constraints:

- Per-zone hourly MAPE ≤ 1.5% on next-day forecast.
- Peak-hour forecast (the hour with highest demand each day) MAPE ≤ 1.0%.
- Catastrophic-event handling: when forecasts miss by > 5%, blackouts can occur.
- Real-time refresh hourly; full 24h horizon.
- Use weather forecast (NWP) outputs as exogenous inputs.
- Integrate with operations dashboards for grid planners.

This is harder than typical time-series because: peak-hour error is asymmetric (under-forecasting peak = brownouts; over = wasteful generation); weather features have non-linear thresholds (95°F + humidity is much worse than 95°F alone); rare events need explicit handling (heat dome, polar vortex); and forecasts must integrate with NWP weather models which themselves have uncertainty.

---

## Step 1: Define the Problem Type

```
Type:           Regression on hourly load (MW)
Primary Metric: MAPE on 24-hour-ahead forecast
Secondary:      Peak-hour MAPE; per-zone MAPE
                Bias on extreme-temperature days
                P95-coverage of prediction intervals
Business Goal:  Per-hour MAPE ≤ 1.5%; peak-hour MAPE ≤ 1.0%
Constraint:     Hourly refresh; integrates with NWP weather forecasts
Target shape:   Strong daily + weekly + annual seasonality; some left-skew
```

**Expert thinking:** in grid operations, peak-hour accuracy is multi-times more valuable than off-peak. A weighted MAPE that emphasizes peak hours is appropriate. We'll train two approaches: standard MAPE and peak-weighted, and compare.

---

## Step 2: Understand the Data

```
Shape: 7 years × 24 hours × 12 zones = ~735K hour-zone observations

Per hour-zone:
- timestamp (UTC)
- zone_id (12 zones)
- load_mw (target)

Lag features:
- load_lag_1h, load_lag_24h, load_lag_168h (week ago)
- load_ma_24h, load_ma_168h
- load_z_lag_1h (z-score relative to recent week)

Calendar:
- hour_of_day, day_of_week, is_weekend
- month, day_of_year
- holiday_flag, holiday_name
- school_in_session

Weather (HISTORICAL — actuals at the prediction time):
- temperature_c, humidity_pct
- wind_speed_ms
- solar_radiation
- precipitation_mm

Weather Forecast (NWP — what was predicted at forecast time, 24h ahead):
- forecast_temp_c_24h_ahead
- forecast_humidity_24h
- forecast_wind_24h
- nwp_uncertainty_24h (NWP's own confidence)
- nwp_was_extreme_revision (binary; did NWP just revise its forecast significantly?)

Derived weather:
- HDD_heating_degrees (max(0, 65F - temp_F))
- CDD_cooling_degrees (max(0, temp_F - 65F))
- discomfort_index (heat-humidity composite)
- is_heat_dome_indicator (binary; consecutive days > 95F)
- is_extreme_cold (< -10°C)

Operational:
- ercot_alert_active (or similar grid-stress signal)
- known_outage_active (planned outage of major industrial customer)
```

**Expert thinking:** the NWP weather forecast IS a feature, and it's the most important one for 24-hour-ahead forecasts. But NWP itself has uncertainty; a model that uses NWP must propagate that uncertainty.

---

## Step 3: EDA

```python
df['load_mw'].describe()
# count    735K
# mean    8,230 MW
# std     2,840 MW
# min     3,100 MW (low, summer night, low demand)
# max    18,400 MW (high, summer afternoon heat dome)
# (per zone scale varies)

# Daily load profile (averaged)
df.groupby(['hour_of_day', 'is_weekend'])['load_mw'].mean()
# Weekday: peak at 16:00 (afternoon)
# Weekend: peak at 18:00 (later, less work-driven)
# Trough: 04:00-05:00 (lowest)

# Temperature-load relationship (the killer non-linear feature)
df['temp_f_bucket'] = pd.cut(df['temperature_c'].apply(lambda c: c*9/5+32), bins=[0,40,50,60,70,80,90,100,120])
df.groupby('temp_f_bucket')['load_mw'].mean()
# < 40F: 9,800 MW (heating spike)
# 40-50F: 8,200 MW
# 50-60F: 7,600 MW (sweet spot)
# 60-70F: 7,400 MW (lowest)
# 70-80F: 7,800 MW
# 80-90F: 9,500 MW
# 90-100F: 12,400 MW (cooling spike)
# > 100F: 14,800 MW (catastrophic cooling)

# Clearly U-shaped + non-linear

# Per-zone seasonality
# Some zones (Houston, Dallas) summer-peaking
# Other zones (rural, manufacturing) winter-peaking
```

---

## Step 4: Feature Engineering

```python
# === LAG FEATURES (multiple horizons) ===
g = df.groupby('zone_id')
df['load_lag_1h'] = g['load_mw'].shift(1)
df['load_lag_24h'] = g['load_mw'].shift(24)
df['load_lag_168h'] = g['load_mw'].shift(168)
df['load_ma_24h'] = g['load_mw'].shift(1).rolling(24).mean()
df['load_ma_168h'] = g['load_mw'].shift(1).rolling(168).mean()
df['load_change_1h'] = df['load_lag_1h'] - df['load_lag_24h']  # year-over-year change

# === WEATHER NON-LINEARITY ===
# Compute heating / cooling degree hours; cumulative load drivers
temp_f = df['temperature_c'] * 9/5 + 32
df['cooling_degree_hours'] = (temp_f - 65).clip(0, None)
df['heating_degree_hours'] = (65 - temp_f).clip(0, None)
df['extreme_cooling'] = (temp_f - 95).clip(0, None) ** 1.5
df['extreme_heating'] = (5 - temp_f).clip(0, None) ** 1.5

# === COMBINED HEAT INDEX ===
df['discomfort_index'] = (
    df['temperature_c'] * 9/5 + 32 +
    df['humidity_pct'] / 5
)

# === RECENT EXTREME CONDITIONS ===
df['heat_dome_3day'] = g['temperature_c'].rolling(72).mean()
df['heat_dome_indicator'] = (df['heat_dome_3day'] > 35).astype(int)  # 35°C = 95°F

# === NWP FORECAST FEATURES ===
df['forecast_temp_24h'] = df['forecast_temp_c_24h_ahead']
df['forecast_cooling_24h'] = (df['forecast_temp_24h'] * 9/5 + 32 - 65).clip(0, None)
df['nwp_revision_significant'] = df['nwp_was_extreme_revision'].astype(int)

# === CALENDAR CYCLICAL ===
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

---

## Step 5: Model

```python
# === Model 1: ElasticNet baseline ===
# CV MAPE: 4.2%

# === Model 2: LightGBM ===
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=2000, max_depth=8, num_leaves=127,
    learning_rate=0.04
)
# CV MAPE: 1.4% ✓
# Peak-hour MAPE: 0.96% ✓

# === Model 3: LightGBM with peak-weighted loss ===
# Weight peak hours 3x in training
def custom_objective_peak_weighted(y_true, y_pred):
    weights = np.where(is_peak_hour, 3.0, 1.0)
    grad = (y_pred - y_true) * weights
    hess = weights
    return grad, hess
# Improves peak-hour MAPE 0.96% → 0.78%

# === Model 4: Ensemble of multiple LightGBM with bagging ===
# Median of 5 models with different random seeds
# Reduces P99 errors significantly
```

---

## Step 6: Final Evaluation

```python
# Test set: 6 months held out
#
# Performance:
#   Per-hour MAPE:                1.32%   ✓ (target 1.5%)
#   Peak-hour MAPE:               0.78%   ✓ (target 1.0%)
#   Heat-dome days MAPE:          1.94%   (worst — but acceptable)
#   Polar-vortex days MAPE:       1.71%
#
# Per-zone:
#   Houston (summer peaking):     1.21%
#   Dallas:                       1.18%
#   Rural manufacturing:           1.42%
#   Mixed:                         1.31%
#
# Inference: ~3 seconds per 24-hour batch; well within budget
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Loss function | Standard MSE | Peak-hour-weighted loss (3x weighting on peak hours) |
| Weather features | Use raw temperature | Heating/cooling degree hours, discomfort index, extreme indicators |
| NWP integration | Skip | Use NWP forecast as feature; track NWP confidence |
| Per-zone | One global model | Separate models per zone OR with strong zone features |
| Lag features | Just lag 1h | Multi-horizon: 1h, 24h, 168h + rolling means |
| Catastrophic events | Treat as outliers | Heat-dome indicators; extreme-temp polynomial features |
| Calendar | Hour as integer | Cyclical encoding (sin/cos) for hour, day-of-week, month |
| Bagging | Single model | Median of 5 bagged models for stability |
