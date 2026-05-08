# Expert Scenario 3: Dynamic Ride-Share Pricing (Surge Prediction)

> **Complexity:** Spatio-temporal regression, real-time prediction, multi-granularity features (city zones x time slots), weather integration, extreme demand spikes, business constraints on maximum surge.

---

## The Brief

A ride-sharing company operates in a city divided into 120 zones. They want to predict the **surge multiplier** (1.0x to 5.0x) for each zone in the next 15-minute window. Currently they use a simple rule: if demand > supply by 20%, surge = 1.3x; by 50%, surge = 1.8x; etc. This reactive approach is too slow -- by the time surge kicks in, drivers have already missed the window. The company wants a **predictive** model that anticipates demand 15 minutes ahead. Bad predictions cost money: under-pricing means lost revenue, over-pricing means lost riders.

This is complex because: predictions must be zone-specific AND time-specific (120 zones x 96 fifteen-minute slots = 11,520 prediction points per day), features span multiple granularities (city-wide events, zone-level patterns, weather), the target has a floor at 1.0 (no negative surge), and the model must run in under 2 seconds for all 120 zones.

---

## Step 1: Define the Problem Type

```
Type:           Regression (predicting surge multiplier, continuous 1.0 to 5.0)
Primary Metric: MAE on surge multiplier (must beat current rule-based system's MAE of 0.31)
Secondary:      Revenue impact (predicted optimal pricing vs actual)
Special:        Target is bounded [1.0, 5.0] and heavily right-skewed
                90% of slots have surge = 1.0 (no surge)
                8% have surge 1.2-2.0 (mild surge)
                2% have surge 2.0-5.0 (extreme surge -- these matter most!)
Constraint:     Prediction latency < 2 seconds for all 120 zones
```

**Expert thinking:** This is similar to the hospital LOS problem with a skewed target, but adds spatial and temporal dimensions. The 90% of "no surge" slots are trivial to predict. The real challenge is catching the 2% of extreme surges. We might need a two-stage approach: (1) classify surge vs no-surge, (2) predict surge magnitude when it exists.

---

## Step 2: Understand the Data

```
Historical data: 18 months, 120 zones, 15-minute granularity
Total rows: 120 zones x 96 slots/day x 548 days = 6,312,960 rows

Features per row:
- zone_id (int, 1-120)
- timestamp (datetime, 15-min intervals)
- demand_count (int, ride requests in this zone/slot)
- supply_count (int, available drivers in this zone/slot)
- surge_multiplier (TARGET, float 1.0-5.0)
- completed_rides (int)
- avg_wait_time_minutes (float)
- avg_ride_distance_km (float)
- avg_ride_duration_minutes (float)
- cancellation_rate (float, 0-1)

Zone metadata:
- zone_type (categorical: downtown, residential, airport, entertainment, commercial, suburban)
- zone_area_km2 (float)
- num_bars_restaurants (int)
- num_offices (int)
- has_stadium (binary)
- has_hospital (binary)
- has_university (binary)
- nearest_highway_km (float)

Weather data (city-wide, hourly):
- temperature_c (float)
- precipitation_mm (float)
- wind_speed_kmh (float)
- weather_condition (categorical: clear, cloudy, rain, heavy_rain, snow, fog)
- visibility_km (float)

Events calendar:
- event_name (string, e.g., "Lakers Game", "Taylor Swift Concert")
- event_zone (int, which zone)
- event_start_time (datetime)
- event_end_time (datetime)
- event_capacity (int, number of attendees)
- event_type (categorical: sports, concert, conference, festival, holiday)
```

**Expert thinking:** Key challenges:
- `demand_count` and `supply_count` for the CURRENT slot are the target's direct cause -- but we're predicting 15 min AHEAD, so we need LAGGED versions
- Weather forecasts (not actuals) would be used in production
- Events create massive localized spikes (stadium zone goes from 1.0x to 4.5x when a game ends)
- Spatial spillover: high surge in zone A pushes demand to adjacent zones B and C

---

## Step 3: Exploratory Data Analysis (EDA)

**What EDA reveals:**

| Finding | Implication |
|---------|------------|
| 90.2% of slots have surge = 1.0 (no surge) | Extreme class imbalance if we classify surge/no-surge |
| Top 3 surge triggers: rain (+0.8 avg), events (+1.2 avg), Friday 11pm-2am (+0.6 avg) | Weather and events are key drivers |
| Airport zone has predictable surge at flight arrival times | Can engineer flight-schedule features |
| Downtown zones surge 2-3pm (lunch) and 5-7pm (commute) DAILY | Strong time-of-day patterns |
| Stadium zone goes 1.0x -> 4.5x in the 30 minutes after game end | Event end-time is more important than start-time |
| When it starts raining, surge takes 10-15 min to develop | Need lagged weather features (rain 15 min ago -> surge now) |
| Adjacent zones show correlated surge (spatial autocorrelation = 0.62) | Neighboring zone features matter |
| Surge patterns differ weekday vs weekend, summer vs winter | Multiple seasonality layers |
| Supply drops 40% during heavy rain (drivers go offline) | Supply-side shock is as important as demand-side |

---

## Step 4: Data Cleaning

```python
# Temporal leakage audit
# CANNOT use: demand_count, supply_count, completed_rides, avg_wait_time for the CURRENT slot
# These are the current reality -- we need to predict BEFORE they happen
# CAN use: all values from previous slots (lagged by 15+ minutes)

# Missing weather data (0.3% -- weather station outages)
# Forward-fill (weather doesn't change instantly)
weather['temperature_c'] = weather['temperature_c'].ffill()
weather['precipitation_mm'] = weather['precipitation_mm'].ffill()

# Missing demand/supply (0.1% -- system outages)
# Interpolate between adjacent time slots
df['demand_count'] = df.groupby('zone_id')['demand_count'].transform(
    lambda x: x.interpolate(method='linear')
)

# Anomaly detection in surge values
# Found 47 rows with surge > 5.0 (data error -- max should be 5.0)
df.loc[df['surge_multiplier'] > 5.0, 'surge_multiplier'] = 5.0
# Found 12 rows with surge < 1.0 (should never be below 1.0)
df.loc[df['surge_multiplier'] < 1.0, 'surge_multiplier'] = 1.0
```

---

## Step 5: Feature Engineering

This is the most critical phase -- we need to capture temporal patterns, spatial relationships, and external events.

```python
# ================================================================
# TEMPORAL LAG FEATURES (what happened recently in THIS zone?)
# ================================================================

# Lag features: values from previous time slots
for lag in [1, 2, 3, 4, 6, 8, 12]:  # 15min, 30min, 45min, 1h, 1.5h, 2h, 3h
    df[f'demand_lag_{lag}'] = df.groupby('zone_id')['demand_count'].shift(lag)
    df[f'supply_lag_{lag}'] = df.groupby('zone_id')['supply_count'].shift(lag)
    df[f'surge_lag_{lag}'] = df.groupby('zone_id')['surge_multiplier'].shift(lag)

# Demand/supply ratio (the fundamental driver)
df['demand_supply_ratio_lag1'] = df['demand_lag_1'] / (df['supply_lag_1'] + 1)
df['demand_supply_ratio_lag2'] = df['demand_lag_2'] / (df['supply_lag_2'] + 1)

# Rolling statistics (capturing trends)
for window in [4, 8, 16]:  # 1h, 2h, 4h rolling windows
    df[f'demand_rolling_mean_{window}'] = df.groupby('zone_id')['demand_count'].transform(
        lambda x: x.shift(1).rolling(window).mean()
    )
    df[f'demand_rolling_std_{window}'] = df.groupby('zone_id')['demand_count'].transform(
        lambda x: x.shift(1).rolling(window).std()
    )
    df[f'surge_rolling_mean_{window}'] = df.groupby('zone_id')['surge_multiplier'].transform(
        lambda x: x.shift(1).rolling(window).mean()
    )

# Demand acceleration (is demand increasing or decreasing?)
df['demand_acceleration'] = df['demand_lag_1'] - df['demand_lag_2']
# Positive = demand increasing, negative = demand decreasing

# Supply trend
df['supply_trend'] = df['supply_lag_1'] - df['supply_lag_4']  # supply change over last hour
# Negative = drivers leaving (bad sign for surge)

# ================================================================
# TEMPORAL CALENDAR FEATURES (when is it?)
# ================================================================

df['hour'] = df['timestamp'].dt.hour
df['minute_slot'] = df['timestamp'].dt.minute // 15  # 0, 1, 2, 3
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Mon, 6=Sun
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['month'] = df['timestamp'].dt.month

# Cyclical encoding for hour (23:45 and 00:00 are close)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Time-based surge patterns
df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
df['is_evening_rush'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
df['is_bar_close'] = ((df['hour'] >= 23) | (df['hour'] <= 2)).astype(int)
df['is_lunch'] = ((df['hour'] >= 11) & (df['hour'] <= 13)).astype(int)

# "Same time last week" -- very predictive for regular patterns
df['surge_same_time_last_week'] = df.groupby(['zone_id', 'day_of_week', 'hour', 'minute_slot'])\
    ['surge_multiplier'].shift(7 * 96)  # 7 days * 96 slots/day

# "Same time yesterday"
df['surge_same_time_yesterday'] = df.groupby(['zone_id', 'hour', 'minute_slot'])\
    ['surge_multiplier'].shift(96)  # 96 slots = 1 day

# Historical average for this zone/day/hour combination
zone_hour_avg = df.groupby(['zone_id', 'day_of_week', 'hour'])['surge_multiplier'].mean()
df['zone_hour_historical_avg'] = df.set_index(['zone_id', 'day_of_week', 'hour']).index.map(zone_hour_avg)

# ================================================================
# SPATIAL FEATURES (what's happening in neighboring zones?)
# ================================================================

# Define zone adjacency (each zone has 3-8 neighbors)
# zone_neighbors = {1: [2, 5, 6], 2: [1, 3, 5, 6, 7], ...}

# Average surge in neighboring zones (spatial spillover)
for lag in [1, 2]:
    df[f'neighbor_avg_surge_lag_{lag}'] = df.apply(
        lambda row: df[(df['zone_id'].isin(zone_neighbors[row['zone_id']])) &
                       (df['timestamp'] == row['timestamp'] - pd.Timedelta(minutes=15*lag))]
                      ['surge_multiplier'].mean(), axis=1
    )

# Max surge in any neighbor (hot spot detection)
df['neighbor_max_surge_lag1'] = ...  # highest surge among all neighbors

# Demand flow between zones
df['neighbor_demand_sum_lag1'] = ...  # total demand in adjacent zones

# ================================================================
# WEATHER FEATURES
# ================================================================

# Current weather (at prediction time)
df['is_raining'] = (df['precipitation_mm'] > 0).astype(int)
df['is_heavy_rain'] = (df['precipitation_mm'] > 5).astype(int)
df['is_snowing'] = (df['weather_condition'] == 'snow').astype(int)

# Weather CHANGE (rain just started = bigger surge than ongoing rain)
df['rain_started_recently'] = ((df['precipitation_mm'] > 0) &
                                (df['precipitation_mm_lag4'] == 0)).astype(int)
# Rain that started in the last hour causes bigger surge than rain for 6 hours

# Temperature extremes
df['extreme_cold'] = (df['temperature_c'] < -5).astype(int)
df['extreme_heat'] = (df['temperature_c'] > 35).astype(int)

# ================================================================
# EVENT FEATURES
# ================================================================

# Is there an active event in this zone right now?
df['event_active'] = ...  # binary: event happening now
df['event_ending_soon'] = ...  # binary: event ends within 30 minutes
# Event ENDING is more important than event happening (everyone leaves at once)

df['event_capacity'] = ...  # 0 if no event, else attendance number
df['minutes_until_event_end'] = ...  # countdown feature
df['minutes_since_event_end'] = ...  # aftermath feature (surge lingers 30-60 min)

# Is there an event in a NEIGHBORING zone?
df['neighbor_event_ending_soon'] = ...  # spillover from nearby events

# Event type matters: sports games end suddenly (spike), concerts end gradually (wave)
df['event_type_sports'] = ...
df['event_type_concert'] = ...

# ================================================================
# ZONE FEATURES (static, from metadata)
# ================================================================
# zone_type, has_stadium, has_university, etc. -- already available
# Create interactions:
df['entertainment_zone_weekend_night'] = (
    (df['zone_type'] == 'entertainment') & df['is_weekend'] & df['is_bar_close']
).astype(int)
df['airport_zone'] = (df['zone_type'] == 'airport').astype(int)
```

**Expert insight:** We created ~80 features from 4 data sources. The most powerful features are:
1. `surge_lag_1` and `demand_supply_ratio_lag1` (recent momentum)
2. `surge_same_time_last_week` (weekly pattern)
3. `event_ending_soon` (massive localized spikes)
4. `rain_started_recently` (weather shock)
5. `neighbor_avg_surge_lag1` (spatial spillover)

---

## Step 6: Feature Selection

```python
# MI scores (top 10):
#   surge_lag_1                    0.42
#   demand_supply_ratio_lag1       0.38
#   surge_same_time_last_week      0.31
#   surge_rolling_mean_4           0.28
#   zone_hour_historical_avg       0.25
#   neighbor_avg_surge_lag1        0.18
#   event_ending_soon              0.15
#   is_raining                     0.12
#   demand_acceleration            0.10
#   hour_sin                       0.08

# Removed 12 features with MI < 0.01:
#   zone_area_km2, nearest_highway_km, visibility_km (most hours),
#   extreme_cold (rare), month (weak after other time features included)

# Final: 68 features
```

---

## Step 7-8: Preprocessing & Split

```python
# Time-based split:
# Train: months 1-15 (first 15 months)
# Validation: month 16 (tune weights/thresholds)
# Test: months 17-18 (final evaluation, unseen future)

# Preprocessing:
# Tree models: no scaling needed
# Linear models: StandardScaler on numeric, OneHot on categoricals

# Drop first 24 hours of data (lag features are NaN)
df = df.dropna(subset=['demand_lag_12'])  # ensures all lags are available
```

---

## Step 9: Baseline

```python
# Baseline 1: Predict 1.0 (no surge) for everything
# MAE: 0.18 (seems low but misses ALL surges -- useless for business)

# Baseline 2: Current rule-based system
# MAE: 0.31 (reactive, always 15 min late)

# Baseline 3: "Same as last slot" (persistence)
# MAE: 0.14 (surprisingly strong -- surge is autocorrelated)

# Baseline 4: "Same time last week"
# MAE: 0.19 (captures weekly patterns but misses weather/events)
```

---

## Step 10: Model Comparison

```python
# Model 1: Ridge Regression
# CV MAE: 0.135

# Model 2: Random Forest
# CV MAE: 0.098

# Model 3: XGBoost
# CV MAE: 0.082

# Model 4: LightGBM
# CV MAE: 0.079  # Best

# Model 5: KNN (k=10)
# CV MAE: 0.142  (too many features, too slow)

# BUT: These MAEs average across all slots including the 90% with surge=1.0
# For the critical 2% of extreme surges (>2.0x):
# LightGBM MAE on extreme surges: 0.58  (off by 0.58x on average)
# XGBoost MAE on extreme surges: 0.62
# Rule-based MAE on extreme surges: 0.89  (we're already much better)
```

---

## Step 11: Two-Stage Model for Surge Events

```python
# Stage 1: Binary classifier -- will there be surge? (surge > 1.1)
# LightGBM Classifier, class_weight='balanced'
# Recall: 0.87 (catches 87% of surge events)
# Precision: 0.72 (28% false alarms)

# Stage 2: Regression -- how much surge? (only for predicted surge slots)
# LightGBM Regressor on surge-only subset
# MAE on surge slots: 0.34 (much better than 0.58 from single model)

# Combined: If Stage 1 says no surge -> predict 1.0
#           If Stage 1 says surge -> use Stage 2 prediction

# Two-stage results:
# Overall MAE: 0.074 (best yet)
# No-surge slots MAE: 0.02
# Mild surge (1.1-2.0) MAE: 0.21
# Extreme surge (>2.0) MAE: 0.42

# vs rule-based system:
# Overall improvement: 76% (0.074 vs 0.31)
# Extreme surge improvement: 53% (0.42 vs 0.89)
```

---

## Step 12-13: Tuning & Ensembling

```python
# Tuned LightGBM Stage 2 (surge regressor):
# n_estimators=2000, max_depth=8, learning_rate=0.03, num_leaves=127
# Added sample_weight: extreme surges get 3x weight

# Blend with single-model LightGBM (weight optimization):
# 60% two-stage + 40% single model
# Final MAE: 0.069

# Post-processing: clip predictions to [1.0, 5.0]
y_pred = np.clip(y_pred, 1.0, 5.0)
```

---

## Step 14: Final Evaluation

```python
# Test set: 2 months of unseen future data (1,105,920 zone-slots)
#
# Overall MAE: 0.069   (rule-based: 0.31 -- 78% improvement)
# No-surge MAE: 0.018
# Mild surge MAE: 0.198
# Extreme surge MAE: 0.402
#
# Revenue impact simulation:
# Rule-based system revenue: $48.2M over 2 months
# ML pricing revenue:        $52.7M over 2 months
# Revenue uplift:            $4.5M (+9.3%)
#
# Rider experience:
# Wait time with rule-based:  6.2 min average
# Wait time with ML pricing:  4.8 min average (predictive surge attracts drivers earlier)
```

---

## Step 15-16: Explainability & Deployment

```python
# Top SHAP features:
# 1. surge_lag_1 (momentum -- if surging now, likely continues)
# 2. demand_supply_ratio_lag1 (fundamental driver)
# 3. event_ending_soon (massive event-driven spikes)
# 4. surge_same_time_last_week (weekly rhythm)
# 5. rain_started_recently (weather shock)

# Deployment:
# - Runs every 15 minutes, scoring all 120 zones in 1.2 seconds
# - Outputs: predicted surge per zone + confidence
# - Ops team can override (e.g., cap surge at 3.0x during emergencies)
# - Retrain weekly with fresh data
# - Monitor: if MAE on extreme surges exceeds 0.6, alert + retrain
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Temporal data | Use current demand (leakage!) | Used only lagged features (15+ min back) |
| Time patterns | `hour` as integer | Cyclical encoding + same-time-last-week + calendar flags |
| Spatial data | Ignore neighboring zones | Computed neighbor surge averages and spillover |
| Events | Binary "event yes/no" | Minutes-until-end, capacity, event type, neighbor events |
| Weather | Temperature + rain | Rain-just-started (weather CHANGE), not just weather state |
| Skewed target | Single regression model | Two-stage: classify surge then regress magnitude |
| Evaluation | Overall MAE only | Stratified by surge severity (extreme surges matter most) |
| Business metric | Model accuracy only | Revenue impact simulation ($4.5M uplift) |
