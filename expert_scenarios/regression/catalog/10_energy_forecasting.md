# Expert Scenario 10: Energy Consumption Forecasting for Smart Grid

> **Complexity:** Time-series forecasting using ML (not classical ARIMA), multiple seasonal patterns (hourly, daily, weekly, annual), exogenous weather data, 500+ smart meters with hierarchical structure (meter -> building -> district -> city), forecast at multiple horizons (1 hour, 24 hours, 7 days), concept drift from energy efficiency improvements, and deployment to real-time grid operations where under-prediction causes blackouts and over-prediction wastes $millions.

---

## The Brief

A regional electric utility serves 2.1M customers across 500 substations. They need to forecast electricity demand at the substation level at three horizons: 1 hour (real-time dispatch), 24 hours (day-ahead market bidding), and 7 days (maintenance planning). Currently they use simple historical averages + weather adjustment, resulting in 8.2% MAPE (Mean Absolute Percentage Error). Each 1% MAPE improvement saves ~$4.2M annually in reduced over-generation and avoided peak penalties. Target: reduce MAPE from 8.2% to under 4%.

This is complex because: electricity demand has multiple overlapping seasonal patterns, weather is the dominant exogenous driver but weather forecasts themselves have errors, 500 substations each have unique load profiles, special events (holidays, sports games, heat waves) cause anomalous demand, and the cost of under-prediction (blackouts) is 10x the cost of over-prediction (wasted generation).

---

## Step 1: Define the Problem Type

```
Type:           Multi-horizon Time Series Regression
                (predict continuous demand at 1h, 24h, 7d horizons)

Primary Metric: MAPE (Mean Absolute Percentage Error) -- industry standard
Secondary:      Pinball Loss (quantile forecasts for uncertainty)
Asymmetric:     Under-prediction penalty 10x over-prediction
                (blackouts vs wasted generation)

Granularity:    Per-substation, hourly
Scale:          500 substations x 8,760 hours/year = 4.38M predictions/year
Training data:  3 years of hourly data = 500 x 26,280 = 13.14M rows
```

---

## Step 2: Understand the Data

```
3 years of hourly data (2021-2024), 500 substations

Load data (target):
- substation_id, timestamp (hourly), load_mw (megawatts)
- 13.14M rows total

Weather data (primary exogenous driver):
- temperature, humidity, wind_speed, cloud_cover, precipitation
- feels_like_temperature (heat index in summer, wind chill in winter)
- solar_radiation (affects solar panel generation -- reduces net load)
- Actual (historical) and Forecast (for future predictions)
- Weather forecast accuracy degrades: 1h forecast MAE=1.2F, 24h=3.5F, 7d=6.8F

Calendar data:
- hour_of_day (0-23), day_of_week (0-6), month (1-12)
- is_holiday, holiday_name
- is_weekend, is_business_day
- school_in_session (affects residential patterns)
- daylight_savings_transition

Substation metadata:
- substation_id, district, capacity_mw
- customer_mix: pct_residential, pct_commercial, pct_industrial
- solar_penetration_pct (rooftop solar in service area)
- population_growth_rate (area growth)

Special events:
- major_sports_events (stadium in service area)
- concerts, festivals
- extreme_weather_alerts (heat wave, ice storm)
- planned_outages (maintenance -- shifts load to neighboring substations)

Historical anomalies:
- COVID-19 (2021): commercial down 30%, residential up 25%
- Ice storm Feb 2023: demand spike then outages
- Heat dome July 2023: all-time peak demand
```

---

## Step 3: EDA -- Understanding Load Patterns

```python
# Electricity demand has 4 overlapping seasonal patterns:

# 1. HOURLY pattern (within-day):
# Residential: low overnight (1-5am), morning ramp (6-8am), evening peak (5-8pm)
# Commercial: ramp up (7-9am), steady (9am-5pm), ramp down (5-7pm)
# Industrial: relatively flat (24/7 operations) with slight day shift

# 2. DAILY pattern (within-week):
# Weekdays: 15-20% higher than weekends (commercial activity)
# Monday morning ramp is steeper than other days
# Friday afternoon demand drops early

# 3. ANNUAL pattern (seasonal):
# Summer peak (AC cooling): July-August, +40% over spring
# Winter secondary peak (heating in some areas): December-January, +20%
# Spring/Fall: lowest demand ("shoulder season")
# The annual pattern is temperature-driven

# 4. LONG-TERM trend:
# 1.5% annual load growth (population + electrification)
# BUT: solar penetration growing 15%/year (reduces NET demand midday)
# "Duck curve": midday demand drops as solar generates, then steep evening ramp

# KEY FINDING: Temperature is the #1 predictor
# Below 65F: heating demand increases linearly
# 65-75F: minimal HVAC ("comfort zone")
# Above 75F: cooling demand increases EXPONENTIALLY (non-linear!)
# The temperature-demand curve is V-shaped with exponential right tail
```

---

## Step 4-5: Feature Engineering (Time-Series Specific)

```python
import pandas as pd
import numpy as np

# === Temporal Features ===
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)

# Cyclical encoding (so hour 23 is close to hour 0)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Binary calendar flags
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_holiday'] = df['timestamp'].dt.date.isin(holiday_dates).astype(int)
df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18) &
                           (df['is_weekend'] == 0)).astype(int)

# === Lag Features (CRITICAL for time series ML) ===
# Same hour yesterday, same hour last week, same hour last year
for lag in [1, 2, 3, 24, 48, 168, 8760]:
    df[f'load_lag_{lag}h'] = df.groupby('substation_id')['load_mw'].shift(lag)

# Rolling statistics (capture recent trend without leakage)
for window in [24, 168]:  # 1 day, 1 week
    df[f'load_rolling_mean_{window}h'] = (
        df.groupby('substation_id')['load_mw']
        .transform(lambda x: x.shift(1).rolling(window).mean())
    )
    df[f'load_rolling_std_{window}h'] = (
        df.groupby('substation_id')['load_mw']
        .transform(lambda x: x.shift(1).rolling(window).std())
    )

# IMPORTANT: ALL lags shift by at least 1 hour (no leakage!)
# For 24h forecast: lags must shift by 24+ hours
# For 7d forecast: lags must shift by 168+ hours

# === Weather Features (domain-engineered) ===
df['temp_deviation'] = df['temperature'] - df['temp_30yr_normal']  # how unusual
df['cooling_degree_hours'] = np.maximum(df['temperature'] - 65, 0)
df['heating_degree_hours'] = np.maximum(65 - df['temperature'], 0)

# Non-linear temperature effect (electricity demand is non-linear with temp!)
df['cooling_sq'] = df['cooling_degree_hours'] ** 2  # quadratic for AC load
df['temp_feels_like'] = df['feels_like_temperature']  # heat index matters more

# Temperature ramp rate (how fast is it getting hot/cold?)
df['temp_change_3h'] = df['temperature'] - df['temperature'].shift(3)
# Rapid warming = AC units kick on simultaneously = demand spike

# Sustained heat (multi-day heat wave builds thermal mass in buildings)
df['avg_temp_past_72h'] = df['temperature'].rolling(72).mean()
# Day 3 of a heat wave has higher demand than day 1 at the same temperature!

# Solar generation offset (reduces net demand)
df['estimated_solar_gen'] = (
    df['solar_radiation'] *
    df['solar_penetration_pct'] *
    substation_solar_capacity
)
# This creates the "duck curve" -- midday net demand drops

# === Interaction Features ===
df['hot_weekend'] = df['is_weekend'] * df['cooling_degree_hours']
# Saturdays in summer: residential AC peaks because everyone's home

df['cold_morning'] = (df['hour'].between(6, 9)).astype(int) * df['heating_degree_hours']
# Cold mornings: heating + hot water + cooking = morning demand spike

# === Substation-Specific Features ===
df['substation_base_load'] = df.groupby('substation_id')['load_mw'].transform('mean')
df['load_normalized'] = df['load_mw'] / df['substation_base_load']
# Normalize by substation size so model learns patterns, not scale

# Total features: ~55 per substation-hour observation
```

---

## Step 6: Train/Test Split (Time Series -- No Random Splitting!)

```python
# CRITICAL: Time series data CANNOT be randomly split
# Random split causes future data to leak into training

# Split strategy:
# Training:    Jan 2021 -- Jun 2023 (2.5 years)
# Validation:  Jul 2023 -- Dec 2023 (6 months, for tuning)
# Test:        Jan 2024 -- Jun 2024 (6 months, final evaluation)

# For the 24h and 7d horizons, we must create proper lag features:
# 24h forecast uses only data available 24+ hours ago
# 7d forecast uses only data available 168+ hours ago

# This means DIFFERENT feature sets for different horizons!
# 1h horizon: can use lag_1h, lag_2h, ..., lag_24h
# 24h horizon: can use lag_24h, lag_48h, ..., lag_8760h (NOT lag_1h!)
# 7d horizon: can use lag_168h, lag_336h, ..., lag_8760h

# Weather features also differ by horizon:
# 1h horizon: uses actual weather (just happened, very accurate)
# 24h horizon: uses 24h weather forecast (somewhat accurate)
# 7d horizon: uses 7d weather forecast (less accurate)
# Weather forecast error must be factored in!
```

---

## Step 7-10: Model Building (Multi-Horizon)

```python
# === Model Strategy ===
# Build SEPARATE models for each horizon (different feature sets, different challenges)

# ============ 1-HOUR HORIZON ============
# This is the easiest -- we have very recent data (lag_1h, lag_2h)
# Recent load IS the best predictor of next-hour load (autocorrelation > 0.95)

import lightgbm as lgb

# Model: LightGBM with lag-heavy features
lgb_1h = lgb.LGBMRegressor(
    n_estimators=1000, max_depth=8, learning_rate=0.05,
    num_leaves=63, min_child_samples=50,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1
)

# Features for 1h: lag_1h, lag_2h, lag_3h, lag_24h (same hour yesterday),
# lag_168h (same hour last week), rolling stats, weather (actual), calendar
# ~45 features

# Validation MAPE: 1.8% (excellent -- very short horizon)

# ============ 24-HOUR HORIZON ============
# Harder -- no lag_1h to lag_23h available
# Must rely on lag_24h+ and weather forecasts

# Model: LightGBM with weather-heavy features
lgb_24h = lgb.LGBMRegressor(
    n_estimators=800, max_depth=7, learning_rate=0.03,
    num_leaves=50, min_child_samples=100,
    subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.5, reg_lambda=0.5  # more regularization than 1h
)

# Features for 24h: lag_24h, lag_48h, lag_168h, lag_8760h,
# rolling stats (past 24h+), weather FORECAST, calendar, temperature features
# ~40 features

# IMPORTANT: Weather forecast error propagates into load forecast error
# Strategy: train on FORECAST weather (not actual) so model learns to handle noise
# Use actual weather for lag features (that's in the past, we know it)

# Validation MAPE: 3.6%

# ============ 7-DAY HORIZON ============
# Hardest -- no recent lags at all, weather forecast is unreliable
# Must rely on seasonal patterns, long-term lags, and broader weather trends

# Model: LightGBM with seasonal features dominant
lgb_7d = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    num_leaves=31, min_child_samples=200,
    reg_alpha=1.0, reg_lambda=1.0  # heavy regularization
)

# Features for 7d: lag_168h, lag_336h, lag_8760h (same week last year),
# rolling mean/std (past week+), weather forecast (less reliable),
# calendar, seasonal, temperature normals
# ~35 features

# 7d weather strategy: use ensemble of weather forecasts (GFS, ECMWF, NAM)
# Take the mean of 3 weather forecasts -> reduces individual forecast error

# Validation MAPE: 5.1%

# ============ BASELINES ============
# Persistence (same as yesterday/last week): 6.5% / 9.2% / 11.4%
# Historical average (same hour, same DOW, same month): 8.2% (current method)
# Linear Regression: 4.2% / 5.8% / 7.5%
# Random Forest: 2.3% / 4.1% / 5.8%

# COMPARISON TABLE:
# Horizon    Persistence  Historical  Linear  RF     LightGBM
# 1h         6.5%        8.2%        4.2%    2.3%   1.8%
# 24h        9.2%        8.2%        5.8%    4.1%   3.6%
# 7d         11.4%       8.2%        7.5%    5.8%   5.1%

# LightGBM wins at all horizons
# Biggest improvement at 1h (lag features dominate)
# 7d is hardest (weather uncertainty)
```

---

## Step 11: Hierarchical Reconciliation

```python
# 500 substations roll up to 45 districts roll up to 5 regions roll up to 1 total

# Problem: Sum of substation forecasts != regional forecast
# If each substation forecast is off by random 3%, the SUM is off by < 3%
# (errors partially cancel) BUT they don't add up consistently

# Solution: Hierarchical forecast reconciliation

# Step 1: Forecast at all levels independently
# - 500 substation models (bottom-up)
# - 45 district models (mid-level)
# - 5 regional models (top-down)
# - 1 total system model

# Step 2: MinTrace reconciliation (optimal combination)
# Project forecasts onto the space of coherent forecasts
# Minimizes total variance while ensuring:
# sum(substation forecasts in district D) = district D forecast
# sum(district forecasts in region R) = region R forecast
# sum(region forecasts) = total system forecast

# Result: Reconciled forecasts are 5-10% more accurate at every level
# Because the total system load is smoother (easier to forecast)
# and that information propagates down to improve substation forecasts

# Post-reconciliation MAPE:
# Before reconciliation: 1.8% / 3.6% / 5.1%
# After reconciliation:  1.6% / 3.3% / 4.7%
```

---

## Step 12: Asymmetric Loss & Probabilistic Forecasts

```python
# Under-prediction is 10x worse than over-prediction
# Under-predict -> buy expensive peaker plants or risk blackouts
# Over-predict -> slight waste in generation costs

# Solution 1: Asymmetric loss function in training
def asymmetric_mse(y_true, y_pred):
    residual = y_true - y_pred
    # Under-prediction (residual > 0): penalize 10x
    # Over-prediction (residual < 0): normal penalty
    loss = np.where(residual > 0, 10 * residual**2, residual**2)
    return loss.mean()

# Result: Model slightly over-predicts on average (bias = +1.2%)
# But MAPE decreases from 3.6% to 3.4% because under-predictions are rarer
# Under-prediction rate drops from 50% to 18% of hours

# Solution 2: Quantile forecasts (provide uncertainty bands)
# Train separate models for P10, P25, P50, P75, P90 quantiles
quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

for q in quantiles:
    lgb_q = lgb.LGBMRegressor(
        objective='quantile', alpha=q,
        n_estimators=500, max_depth=6
    )

# Output for each hour:
# "3pm tomorrow: 485 MW (P10: 462, P25: 474, P50: 485, P75: 498, P90: 518)"
# Grid operators use P90 for safety margin (only 10% chance demand exceeds this)

# Calibration check:
# Actual demand falls below P10 in 9.2% of hours (target: 10%) -- good
# Actual demand falls below P90 in 91.5% of hours (target: 90%) -- good
# Quantile forecasts are well-calibrated
```

---

## Step 13: Special Events & Anomaly Handling

```python
# Regular patterns work most of the time, but anomalies cause the biggest errors

# Type 1: Extreme Weather
# Heat waves: demand spikes 30-50% above normal for sustained periods
# Strategy: Detect heat wave from 72h temperature forecast
#           Use "heat wave regime" model trained only on heat wave days
#           avg_temp_past_72h feature captures thermal buildup
# Result: Heat wave MAPE drops from 12% to 6%

# Type 2: Holidays
# Christmas: commercial load drops 60%, residential up 20%, net down 35%
# Thanksgiving: similar but shorter
# July 4th: residential up (BBQs, AC), commercial down, net depends on weather
# Strategy: Holiday-specific lag features
#           load_same_holiday_last_year, load_day_before_holiday
# Result: Holiday MAPE drops from 15% to 7%

# Type 3: Planned Outages
# When a substation goes offline for maintenance, its load shifts to neighbors
# Strategy: If substation S is offline, add its typical load as a feature
#           for neighboring substations
# Result: Neighbor forecast error during outages drops from 18% to 8%

# Type 4: COVID-like Disruptions
# Commercial down, residential up, patterns completely changed
# Strategy: Include mobility data as exogenous feature
#           (Google Mobility Index as proxy for activity)
# This won't prevent the initial shock but adapts faster

# Type 5: Solar Eclipses / Rare Events
# Aug 2024 solar eclipse: net demand spiked when solar generation dropped suddenly
# These are manually flagged and handled with fallback forecasts
```

---

## Step 14: Final Results

```python
# Test set: Jan-Jun 2024 (6 months, out of sample)

# Final MAPE by horizon:
# Horizon    Old System   New LightGBM  Improvement
# 1h         8.2%*        1.6%          80% better
# 24h        8.2%         3.3%          60% better
# 7d         8.2%         4.7%          43% better

# * Old system used same method for all horizons

# Asymmetric evaluation:
# Under-prediction rate: old=50%, new=18% (at 24h horizon)
# Severe under-prediction (>10% below actual): old=12%, new=2.1%

# Financial impact:
# Each 1% MAPE improvement = $4.2M savings
# 1h horizon: 6.6% improvement = $27.7M
# 24h horizon: 4.9% improvement = $20.6M (most impactful for day-ahead market)
# 7d horizon: 3.5% improvement = $14.7M
# TOTAL ANNUAL SAVINGS: ~$63M
# (plus reduced blackout risk, which has uncapped value)

# Comparison with deep learning approaches:
# LSTM: 24h MAPE = 3.5% (slightly worse than LightGBM, much slower)
# Transformer: 24h MAPE = 3.2% (slightly better, but 100x slower to train)
# LightGBM chosen: 3.3% MAPE with 100x faster training,
# easier to explain, easier to maintain
```

---

## Step 15-16: Deployment to Grid Operations

```python
# Real-time deployment requirements:
# - 1h forecast: computed every 15 minutes (always fresh)
# - 24h forecast: computed every hour (24 hourly predictions)
# - 7d forecast: computed every 6 hours (168 hourly predictions)
# - Latency: <30 seconds for all 500 substations
# - Availability: 99.99% uptime (grid operations are critical infrastructure)

# Architecture:
# 1. Data pipeline (Apache Kafka):
#    - Smart meter data streams in real-time
#    - Weather API pulls every 15 minutes
#    - Calendar and event data from static tables

# 2. Feature store (Redis):
#    - Pre-computed lag features updated in near-real-time
#    - Weather forecast cache
#    - Rolling statistics maintained incrementally

# 3. Prediction service (FastAPI + LightGBM):
#    - Stateless: loads model from S3 at startup
#    - Computes 500 substations x 3 horizons in ~8 seconds
#    - Returns point forecast + quantile forecasts

# 4. Reconciliation service:
#    - Takes bottom-up forecasts from prediction service
#    - Applies MinTrace reconciliation
#    - Returns coherent hierarchical forecasts

# 5. Grid operations dashboard:
#    - Real-time demand forecast vs actual (updating every 15 min)
#    - Probabilistic fan chart (P10-P90 bands)
#    - Anomaly alerts (actual > P95 forecast = potential problem)
#    - Substation-level heat map
#    - Weather overlay

# Model retraining:
# Weekly: retrain with latest week's data (warm start from previous model)
# Monthly: full retrain with all historical data
# Annually: re-evaluate feature engineering, add new data sources

# Monitoring:
# - Live MAPE tracking per substation, per horizon
# - Drift detection: if weekly MAPE exceeds 2x baseline, alert
# - Feature importance tracking: if weather importance drops, weather pipeline broken?
# - Fallback model: if primary model fails, revert to persistence + historical average
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Time split | Random train/test split (leakage!) | Chronological split with proper lag constraints per horizon |
| Features | Raw timestamp columns | Cyclical encoding, 4 types of lags, rolling stats, interaction features |
| Temperature | Linear temp feature | Non-linear (quadratic cooling), heat index, 72h thermal buildup, ramp rate |
| Horizons | One model for all horizons | Separate models with horizon-appropriate features and lag availability |
| Weather | Use perfect weather (leakage for future) | Train on weather FORECASTS (which have errors), ensemble 3 weather sources |
| Hierarchy | Forecast each substation independently | Hierarchical reconciliation (MinTrace) improves all levels by 5-10% |
| Loss function | Symmetric MSE | Asymmetric loss (10x penalty for under-prediction), quantile regression for uncertainty |
| Solar | Ignore | Model net demand, duck curve features, solar generation offset |
| Special events | Treat all days the same | Heat wave regime model, holiday-specific lags, outage load transfer |
| Evaluation | MAPE only | MAPE + quantile calibration + under-prediction rate + financial impact |
| Deployment | Batch prediction in Jupyter notebook | Real-time streaming pipeline, 99.99% uptime, 30-second latency, fallback models |
| Business case | "MAPE improved" | $63M annual savings with per-horizon ROI breakdown |
