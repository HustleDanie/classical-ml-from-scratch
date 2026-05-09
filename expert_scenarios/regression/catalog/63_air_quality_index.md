# Expert Scenario 63: Air Quality Index Prediction

> **Complexity:** Spatiotemporal regression with sensor noise (cheap PM2.5 sensors are unreliable), heavy-tailed AQI events (most days are good; smoke events are catastrophic), per-sensor calibration drift, public-facing predictions feed health alerts, model must work for 800+ stations including remote sites with sparse coverage.

---

## The Brief

A city environmental agency operates a network of 1,200 air-quality monitoring stations: 250 reference-grade (calibrated, $30K each) and 950 lower-cost (~$2K each). The cheap sensors have higher noise and can drift up to 30% over months. The agency wants:

- 4-hour ahead AQI prediction (Air Quality Index, 0-500 scale).
- Special focus on capturing "exceedance" events (AQI > 100 = unhealthy).
- Per-station predictions (not just citywide aggregate).
- The model must work even when stations have data gaps (3-7% of stations are offline at any time).
- Drives public health alerts (push notifications) and school-recess decisions.

Constraints:

- MAPE on AQI ≤ 12% on test data.
- Recall on exceedance events (AQI > 100) ≥ 75% (catch unhealthy days).
- 5-minute granularity prediction; refresh hourly.
- Cheap sensor calibration: model must auto-correct sensor drift.

This is harder than typical time-series regression because: spatiotemporal structure (nearby stations correlate); sensor reliability differs between reference-grade and cheap; rare events (wildfire smoke, traffic spikes) drive the public-health value; model is public-facing, so a missed alert has political consequences.

---

## Step 1: Define the Problem Type

```
Type:           Spatiotemporal regression on AQI (0-500 scale, bounded but rare to hit ceiling)
Primary Metric: MAPE in AQI points
Secondary:      Recall on exceedance events (AQI > 100)
                Per-station MAPE (must work everywhere; not just dense areas)
                P95 AQI accuracy on event days
Business Goal:  MAPE ≤ 12%; exceedance recall ≥ 75%; alert lead time ≥ 2 hours
Constraint:     Per-station predictions; handle station outages; sensor drift correction
Target shape:   Right-skewed; most days AQI 30-80; tail of 100-300+ during fire / inversion
```

**Expert thinking:** for AQI prediction, regression on the raw scale is risky because AQI is computed via piecewise transformation from raw pollutant concentrations. We could either predict raw concentrations (cleaner) and compute AQI, or predict AQI directly (simpler). We'll predict AQI directly with a robust loss.

---

## Step 2: Understand the Data

```
Shape: 1,200 stations × 5-min readings × 18 months ≈ 200M rows

Per station × 5-min reading:
- station_id (unique)
- station_lat, station_lng
- station_class (Reference / LowCost)
- timestamp
- pm25_concentration (ug/m3)
- pm10_concentration (ug/m3)
- ozone_ppb
- no2_ppb
- so2_ppb
- co_ppm
- temperature_c
- humidity_pct
- wind_speed_ms
- wind_direction_deg
- pressure_hpa
- aqi_computed (AQI from EPA formula on the pollutant readings)
- station_age_months
- last_calibration_date

Spatial features (computed from station network):
- num_stations_within_5km
- num_reference_stations_within_10km
- avg_aqi_neighbors_within_5km_recent_30min
- max_aqi_neighbors_within_5km_recent_30min

Exogenous (city-wide):
- nearby_traffic_intensity (from traffic API)
- nearby_industrial_activity_index
- forecast_weather_temperature_4h_ahead
- forecast_weather_wind_4h_ahead
- known_construction_sites_active
- nearby_wildfire_distance_km
- nearby_wildfire_intensity (size + smoke output)
```

**Expert thinking:** the network of stations creates spatial signal — when a fire moves toward the city, AQI rises in a wave. This is captured via "neighbor station" features. The 250 reference-grade stations are "trusted" anchors; the 950 cheap sensors should be calibrated against reference stations.

---

## Step 3: EDA

```python
df['aqi_computed'].describe()
# count    200M
# mean      52
# std       38
# min        2
# 25%       28
# 50%       44
# 75%       64
# 95%      120
# 99%      215
# 99.9%    342
# max      485

# Distribution by season (smoke season has different shape)
df.groupby(df['timestamp'].dt.month)['aqi_computed'].mean()
# Jan: 38
# Feb: 32
# Mar: 35
# Apr: 41
# May: 47
# Jun: 51
# Jul: 78  <- early fire season
# Aug: 92  <- peak fire season
# Sep: 81
# Oct: 56
# Nov: 49
# Dec: 41

# Reference vs LowCost agreement
ref_lc_pairs = compute_nearest_pairs(reference_stations, lowcost_stations, max_dist=2km)
df_pairs = df.merge(ref_lc_pairs, ...)  # readings from nearby ref/lc pairs
print(df_pairs.corr())
# AQI correlation between nearby ref and lc: 0.88 (good but not perfect)
# LowCost stations have ~25% more variance than reference stations
# Drift: lowcost stations show bias of +10 to -8 AQI points relative to reference over months

# Exceedance events
df['is_exceedance'] = (df['aqi_computed'] > 100).astype(int)
df.groupby('station_id')['is_exceedance'].mean()
# Per-station exceedance rate: 2-12% (varies by location)
# Some stations rarely exceed (rural); others often (downtown, near highway)
```

---

## Step 4: Data Cleaning — Sensor Calibration

```python
# === COMPUTE PER-LOWCOST-SENSOR DRIFT FROM REFERENCE STATIONS ===

# For each LowCost station, find its nearest reference station(s)
# Compare readings over a 30-day window
# Compute the multiplicative + additive correction factor

def calibrate_lowcost_sensor(lowcost_data, reference_data, window_days=30):
    """
    Returns calibration: lowcost_corrected = scale * lowcost_raw + offset
    """
    paired = lowcost_data.merge_asof(reference_data, on='timestamp', tolerance='5min')
    paired = paired[paired['distance_km'] < 2]
    if len(paired) < 100:
        return {'scale': 1.0, 'offset': 0.0}
    # Linear regression: reference = scale * lowcost + offset
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(paired['aqi_lowcost'].values.reshape(-1, 1), paired['aqi_reference'])
    return {'scale': model.coef_[0], 'offset': model.intercept_}

# Compute for every LowCost station, refreshing weekly
lowcost_calibrations = {}
for lc_station in lowcost_stations:
    lowcost_calibrations[lc_station.id] = calibrate_lowcost_sensor(
        df[df['station_id'] == lc_station.id],
        df[df['station_class'] == 'Reference'].sort_values('timestamp')
    )

# Apply correction
df['aqi_corrected'] = df.apply(
    lambda r: r['aqi_computed'] if r['station_class'] == 'Reference'
    else r['aqi_computed'] * lowcost_calibrations[r['station_id']]['scale'] + lowcost_calibrations[r['station_id']]['offset'],
    axis=1
)
```

**Expert insight:** continuous sensor calibration is essential for low-cost networks. Without it, model performance degrades over months as cheap sensors drift. The calibration is computed using the network of reference-grade stations as ground truth.

---

## Step 5: Feature Engineering — Spatiotemporal

```python
# === LAG FEATURES PER STATION ===
g = df.groupby('station_id')
df['aqi_lag_5min']  = g['aqi_corrected'].shift(1)
df['aqi_lag_30min'] = g['aqi_corrected'].shift(6)
df['aqi_lag_2h']    = g['aqi_corrected'].shift(24)
df['aqi_lag_24h']   = g['aqi_corrected'].shift(288)
df['aqi_ma_30min']  = g['aqi_corrected'].shift(1).rolling(6).mean()
df['aqi_ma_2h']     = g['aqi_corrected'].shift(1).rolling(24).mean()
df['aqi_change_2h'] = df['aqi_lag_5min'] - df['aqi_lag_2h']

# === SPATIAL FEATURES ===
# Compute "neighbor AQI" features at each station for each timestamp
df = compute_neighbor_aqi(df, radius_km=5, n_neighbors=10)
# Adds: avg_neighbor_aqi_5km, max_neighbor_aqi_5km, std_neighbor_aqi_5km
# Use ONLY reference-grade neighbors to avoid LowCost-LowCost drift correlation

# === WEATHER-DRIVEN FEATURES ===
df['wind_dispersion_score'] = df['wind_speed_ms'] * np.cos(np.radians(df['wind_direction_deg']))  # rough advection direction
df['low_wind_threshold'] = (df['wind_speed_ms'] < 2.0).astype(int)  # stagnation = bad AQI
df['inversion_proxy'] = (df['temperature_c'] - df['temperature_c_4h_ago']).abs() < 3.0  # stable atmosphere
df['humidity_pm25_synergy'] = (df['humidity_pct'] > 70) * (df['pm25_concentration'] > 35)  # secondary aerosol formation

# === SOURCE-DRIVEN FEATURES ===
df['fire_proximity_intensity'] = df['nearby_wildfire_intensity'] / (df['nearby_wildfire_distance_km'] + 1)
df['traffic_proxy'] = df['nearby_traffic_intensity'] * (df['hour_of_day'].isin([7, 8, 17, 18]).astype(int) + 1)

# === STATION CHARACTERISTICS ===
df['is_lowcost'] = (df['station_class'] == 'LowCost').astype(int)
df['log_calibration_age'] = np.log1p((df['timestamp'] - df['last_calibration_date']).dt.days)

# === TEMPORAL FEATURES ===
df['hour_of_day'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
df['month'] = df['timestamp'].dt.month
df['is_smoke_season'] = df['month'].isin([7, 8, 9]).astype(int)
```

---

## Step 6: Feature Selection

```python
# Top 15 features by mutual information:
mi_scores = mutual_info_regression(X_train.sample(500000), y_train.sample(500000))
# aqi_lag_30min                       0.412
# aqi_ma_2h                           0.341
# aqi_lag_5min                         0.318
# avg_neighbor_aqi_5km                 0.221
# fire_proximity_intensity             0.184
# wind_dispersion_score                0.121
# aqi_change_2h                        0.108
# nearby_wildfire_distance_km          0.094
# pm25_concentration                   0.082
# is_smoke_season                      0.071
# inversion_proxy                      0.064
# hour_of_day                          0.054
# pm10_concentration                   0.041
# is_lowcost                           0.038
# humidity_pct                         0.032
```

---

## Step 7: Train/Test Split — Walk-Forward

```python
# 4-hour-ahead prediction; walk-forward CV
df = df.sort_values('timestamp')

# Train: first 12 months (146M rows)
# Validation: month 13-14
# Test: month 15-18 (50M rows; covers smoke season)

# Important: test set must include both normal AND smoke-season conditions
# Otherwise model performance during smoke is unknown
```

---

## Step 8: Try Multiple Models

```python
import lightgbm as lgb

# === Baseline: predict aqi_lag_30min as 4-hr-ahead value ===
# Test MAPE: 21%
# Recall on exceedance: 38%

# === Model 1: ElasticNet ===
# Test MAPE: 16%; exceedance recall: 52%

# === Model 2: Random Forest ===
# Test MAPE: 13%; exceedance recall: 64%

# === Model 3: LightGBM ===
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1000, max_depth=6, num_leaves=63, learning_rate=0.05,
    random_state=42
)
# Test MAPE: 11%; exceedance recall: 73%

# === Model 4: LightGBM with quantile loss for upper bound ===
lgb_p90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, ...)
# Provides P90 alongside P50 for alert thresholds
# Use P90 for "upper bound" alerting; P50 for typical AQI display

# === Model 5: Per-station-class ensemble (Reference vs LowCost) ===
# Different sensor characteristics warrant separate models
ref_model = lgb.LGBMRegressor(...)
ref_model.fit(X_ref_train, y_ref_train)
lc_model = lgb.LGBMRegressor(...)
lc_model.fit(X_lc_train, y_lc_train)
# Combined MAPE: 10.4%; exceedance recall: 76%  <-- target met
```

**Top performer:** Per-station-class LightGBM ensemble (MAPE 10.4%, exceedance recall 76%).

---

## Step 9: Asymmetric Loss for Exceedance

```python
# Missing an exceedance event (false negative on AQI > 100) is more costly
# than a false positive (AQI predicted > 100 but actual was 95)
# Implement asymmetric loss

def asymmetric_objective(y_true, y_pred):
    error = y_pred - y_true
    # Underestimate (predicted < true): penalize 3x
    # Overestimate: penalize 1x
    grad = np.where(error >= 0, error, 3 * error)
    hess = np.where(error >= 0, 1.0, 3.0)
    return grad, hess

lgb_asymmetric = lgb.LGBMRegressor(objective=asymmetric_objective, ...)
# MAPE: 11.2% (slightly worse on average)
# Exceedance recall: 84% (much better)
# Decision: ship asymmetric loss for the public-alert use case
```

---

## Step 10: Quantile Bands for Uncertainty

```python
# Train P10, P50, P90 separately
lgb_p10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, ...)
lgb_p50 = lgb.LGBMRegressor(objective='quantile', alpha=0.50, ...)
lgb_p90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, ...)

# Confidence interval for each station:
# AQI prediction: P50 = 78, P10 = 62, P90 = 105
# UI displays: "Expected AQI: 78 (range: 62-105)"
# Public alert triggered if P90 > 100 (catching uncertainty)

# Conformal calibration on validation set
from mapie.regression import MapieQuantileRegressor
mapie = MapieQuantileRegressor(estimator=lgb_p90, alpha=0.10)
mapie.calibrate(X_calib, y_calib)
# Empirical P10-P90 coverage: 0.892 ✓ (target 0.80)
```

---

## Step 11: Per-Station Performance Audit

```python
# Different stations have different characteristics
# Check that no station is systematically poorly served

per_station_mape = df_test.groupby('station_id').apply(
    lambda g: (np.abs(g['predicted'] - g['actual']) / g['actual']).mean()
)
print(per_station_mape.describe())
# Mean MAPE: 10.4%
# 25th percentile: 7%
# 75th percentile: 13%
# Worst (95th percentile): 18% (typically remote stations with few neighbors)

# Identify worst-performing stations
worst_stations = per_station_mape[per_station_mape > 18].index
# These are mostly:
#   - Remote stations with no nearby reference grade
#   - LowCost stations with old calibration
# Mitigation: dispatch a calibration tech to these stations next month
```

---

## Step 12: Final Evaluation

```python
# Final model: Per-station-class LightGBM ensemble + asymmetric loss + quantile bands
# Test set: 4 months (covers smoke season), ~50M predictions
#
# Performance:
#   Overall MAPE:                10.4%   ✓ (target 12%)
#   Exceedance recall (AQI > 100): 84%   ✓ (target 75%)
#   Mean alert lead time:         3.2 hours  ✓ (target 2)
#   Per-station MAPE: P25=7%, P50=10%, P95=18%
#   Coverage of P10-P90 intervals: 0.892
#
# By condition:
#   Normal days:   MAPE 8%, exceedance recall N/A
#   Fire days:     MAPE 14%, exceedance recall 81%
#   Inversion days: MAPE 12%, exceedance recall 79%
#
# Sensor handling:
#   Reference stations: MAPE 8%
#   LowCost stations: MAPE 12% (drift correction working but residual error)
#   Stations with data gap: MAPE 16% (use neighbor-AQI fallback)
```

---

## Step 13: Public Health Integration

```python
# Alert logic:
def determine_alert_level(predicted_p50, predicted_p90, location):
    """Combine point and uncertainty for alert decisions."""
    if predicted_p90 > 200:
        return ('CRITICAL', 'Hazardous air quality expected. Stay indoors.')
    elif predicted_p90 > 150:
        return ('WARNING', 'Unhealthy for sensitive groups in your area.')
    elif predicted_p90 > 100:
        return ('WATCH', 'Potential air quality concern.')
    else:
        return ('NORMAL', None)

# School-recess decisions:
# If predicted P90 > 100 AND lead time > 2h: schools cancel outdoor recess
# If P90 > 50 but P50 < 100: schools reduce outdoor time but don't cancel
# Decision is made daily at 5am for the school day

# Push notifications: opt-in residents in affected areas receive alerts when:
# - P90 > 100 (sensitive groups)
# - P90 > 150 (general population)
```

---

## Step 14: Deployment

```python
# Production:
#   1. Hourly batch retraining on rolling 30-day window per station class
#   2. Streaming inference: every 5 minutes, predict 4-hour-ahead AQI for each station
#   3. Aggregate to neighborhood level for public-facing maps
#   4. Push alerts to subscribed users when thresholds breached
#
# Sensor calibration: WEEKLY
#   - Re-compute LowCost calibration factors
#   - Compare LowCost readings to nearest Reference for last 7 days
#   - Update calibration coefficients if drift > 5%
#
# Monitoring:
#   - Daily MAPE on completed 4h-ahead predictions
#   - Per-station MAPE drift (alert if any station > 25%)
#   - Per-station exceedance recall (alert if drops below 70%)
#   - Sensor calibration drift (alert if any station > 30% drift from reference)
#   - Forecast skill (compare to next-day NWP weather forecast)
#
# Special handling:
#   - Wildfire detected within 50km: switch to "wildfire model" variant trained
#     specifically on past smoke events
#   - Inversion forecast: pre-warn even before AQI rises
#   - Station outage: use neighbor-aggregated AQI as fallback prediction
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Sensor reliability | Use all readings the same | Distinguish Reference vs LowCost; weekly drift calibration |
| Drift handling | Static features | Continuous LowCost vs Reference comparison; auto-calibrate |
| Loss function | RMSE | Asymmetric loss (underestimates penalized 3x) for public-health critical |
| Spatial structure | One-hot station_id | Neighbor-AQI features (avg, max, std within 5km) |
| Quantile output | Single point estimate | P10/P50/P90 with conformal calibration |
| Train/test split | Random | Walk-forward time-based |
| Architecture | Single global model | Per-station-class ensemble (Reference vs LowCost) |
| Outage handling | Fail silently | Neighbor-AQI fallback when station data missing |
| Alert threshold | Fixed AQI value | Triple thresholds based on P90 (catches uncertainty) |
| Per-station performance | Aggregate MAPE | Per-station audit; flag worst for sensor calibration tech |
| Wildfire / inversion | Treat as outliers | Wildfire-specific model variant; explicit inversion proxy feature |
| Public integration | Static alert thresholds | Push notifications with confidence-aware thresholds; school-recess logic |
