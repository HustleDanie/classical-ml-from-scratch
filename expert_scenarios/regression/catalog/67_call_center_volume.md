# Expert Scenario 67: Call Center Volume Forecasting (Count Regression)

> **Complexity:** Integer count target with multi-seasonality, Poisson / Negative Binomial loss instead of Gaussian RMSE, intra-day patterns, holidays + special events as exogenous drivers, staffing decisions multiply forecast errors by labor cost.

---

## The Brief

A 2,000-agent call center across 6 service queues (billing, technical_support, sales, retention, claims, escalations) wants 15-minute granular volume forecasts 7 days ahead. The forecasts feed:

- Workforce-management (WFM) software that schedules agents.
- Lunch/break planning (don't schedule a break during forecast peak).
- Surge staffing (call out auxiliary agents 24h before predicted spikes).

Constraints:

- Forecast must be at 15-min × 6-queue granularity (~672 forecasts per day).
- Errors are NOT symmetric: under-forecasting by 10% leaves customers on hold (NPS hit, revenue loss); over-forecasting by 10% pays idle agents.
- Cost ratio: missing 1 call costs $35 (lost revenue + churn risk), idle agent-hour costs $28. Staffing for "expected" volume + a buffer is the operating model.
- Service Level (SL) target: 80% of calls answered within 30s.

The right framing isn't "predict the mean" — it's "predict the count with an uncertainty band" so WFM can staff to the upper bound of expected calls.

---

## Step 1: Define the Problem Type

```
Type:           Count regression (non-negative integer); time-series with multi-seasonality
Primary Metric: Poisson deviance (a.k.a. Poisson loss); MAPE secondary
Secondary:      WAPE; Coverage of P10-P90 prediction interval; Service Level forecast accuracy
Business Goal:  WAPE < 8% on hourly aggregates; 90% of intervals contain actual volume
Constraint:     15-min × 6-queue granularity; nightly retrain
Target shape:   Counts 0-180 per 15-min window; Poisson-like with overdispersion (use NB)
```

**Expert thinking:** the target is a non-negative integer count. Gaussian RMSE assumes errors are symmetric and normally distributed — neither is true for counts. Poisson and Negative Binomial likelihoods are designed for counts and naturally handle the variance-mean relationship.

---

## Step 2: Understand the Data

```
Shape: 15-min windows × 6 queues × 730 days = ~280K rows
Target: call_volume (int, 0 - 180 per 15-min window per queue)

Columns:

QUEUE / TIME:
- timestamp (15-min granularity)
- queue (Billing/TechSupport/Sales/Retention/Claims/Escalation)
- call_volume (target)

LAG / VOLUME HISTORY:
- volume_lag_15m
- volume_lag_1h
- volume_lag_24h (same time yesterday)
- volume_lag_7d (same time last week)
- volume_lag_14d
- volume_ma_1h
- volume_ma_24h_dow (rolling avg over recent same-day-of-week observations)
- volume_max_1h

CALENDAR:
- day_of_week (0-6)
- time_of_day_min (0-1439)
- is_business_hours (binary)
- is_weekend
- month
- quarter
- week_of_year

HOLIDAYS / EVENTS:
- holiday_flag (binary, federal holidays)
- holiday_name (string, NaN if not)
- day_after_holiday (binary; volume spikes day-after)
- post_long_weekend (binary)

EXOGENOUS:
- billing_cycle_day (1-31; spikes at 1st of month)
- product_launch_recent_7d (binary)
- known_outage_active (binary)
- promo_active (binary)
- weather_severe_event (binary; storms drive billing inquiries)

CALL CENTER CONTEXT:
- avg_handle_time_recent_24h (float, seconds)
- service_level_recent_24h (float, 0-1)
- agent_count_scheduled (int)
```

**Expert thinking:** key insights:
- The lag features (especially `volume_lag_7d`) are the most predictive (weekly seasonality dominant in call centers).
- Holidays/events break the lag pattern — explicit features needed.
- Service level recent_24h is a feedback feature: a poorly-staffed yesterday increases callbacks today.
- Different queues have very different patterns: billing peaks at month-end; tech support after product launches; retention right after billing-related complaints.

---

## Step 3: EDA

```python
df['call_volume'].describe()
# count   280K
# mean    34.5
# std     32.1
# min      0
# 25%      8
# 50%     22
# 75%     58
# max    178
# Skew: 1.1 (right-skewed, classic count distribution)

# Variance vs mean -- Poisson assumes equal; check
np.var(df['call_volume'])  # 1031
np.mean(df['call_volume']) # 34.5
# Variance/mean ratio = 30  <-- HEAVY OVERDISPERSION
# Use Negative Binomial, not Poisson
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Variance/mean ratio 30 | NB > Poisson; or use Tweedie with `variance_power > 1` |
| Mean 34.5 calls per 15-min, max 178 | Distribution wide |
| 90% of "high-volume" 15-min windows are predictable from `volume_lag_7d` | Weekly seasonality huge |
| Billing volume jumps 4x on the 1st of month | Billing cycle is structural |
| Tech support spikes day 1-3 after product launches | Product launch flag is critical |
| Severe weather in service area → 2.3x billing call volume | Weather as exogenous driver |
| Day-of-week pattern: Mon highest, Thu lowest | Standard pattern |
| Time-of-day: peak 10-11am and 2-3pm; trough 12pm (lunch) and after 7pm | Bimodal daily |

---

## Step 4: Data Cleaning

```python
# Outliers: don't cap. Big spikes are real (storm-induced volume, viral product issues).
# The model needs to predict spikes when context says spike is coming.

# But verify: any 15-min window with > 250 calls is system error
df = df[df['call_volume'] < 250]

# Missing handling: minimal
df['avg_handle_time_recent_24h'] = df['avg_handle_time_recent_24h'].fillna(
    df.groupby('queue')['avg_handle_time_recent_24h'].transform('mean')
)
df['service_level_recent_24h'] = df['service_level_recent_24h'].fillna(0.80)

df['holiday_name'] = df['holiday_name'].fillna('NoHoliday')
```

---

## Step 5: Feature Engineering

```python
# === LAG FEATURES PER QUEUE ===
df = df.sort_values(['queue', 'timestamp'])
g = df.groupby('queue')

df['volume_lag_15m']  = g['call_volume'].shift(1)
df['volume_lag_1h']   = g['call_volume'].shift(4)  # 4 * 15min
df['volume_lag_24h']  = g['call_volume'].shift(96)  # 96 * 15min
df['volume_lag_7d']   = g['call_volume'].shift(96 * 7)
df['volume_lag_14d']  = g['call_volume'].shift(96 * 14)

# Rolling features
df['volume_ma_1h']    = g['call_volume'].shift(1).rolling(4).mean()
df['volume_ma_4h']    = g['call_volume'].shift(1).rolling(16).mean()
df['volume_max_1h']   = g['call_volume'].shift(1).rolling(4).max()
df['volume_std_4h']   = g['call_volume'].shift(1).rolling(16).std()

# Same-day-of-week recent average (better than lag_7d alone)
def same_dow_avg(group, n_weeks=4):
    return group.shift(96 * 7).rolling(96 * 7 * n_weeks, step=96 * 7).mean()
df['volume_dow_ma_4w'] = g['call_volume'].apply(lambda x: x.shift(96 * 7).rolling(28, min_periods=2).mean())

# Trend
df['volume_trend_24h_vs_7d'] = df['volume_lag_24h'] / (df['volume_lag_7d'] + 1)

# === CALENDAR (CYCLICAL) ===
df['minute_of_day'] = df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute
df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# === BILLING CYCLE FEATURE ===
df['days_to_month_end'] = df['timestamp'].dt.daysinmonth - df['timestamp'].dt.day
df['is_first_of_month'] = (df['timestamp'].dt.day == 1).astype(int)
df['is_billing_peak_window'] = ((df['timestamp'].dt.day <= 3) | (df['days_to_month_end'] <= 1)).astype(int)

# === HOLIDAY PROXIMITY ===
df['days_to_next_holiday'] = compute_days_to_next_holiday(df['timestamp'])
df['days_since_last_holiday'] = compute_days_since_last_holiday(df['timestamp'])

# === FEEDBACK FROM YESTERDAY'S SERVICE LEVEL ===
df['callback_pressure'] = (df['service_level_recent_24h'] < 0.70).astype(int)
# When yesterday's SL was bad, today gets callback volume
```

---

## Step 6: Train/Test Split — Walk-Forward

```python
# Walk-forward time-series split
df = df.sort_values('timestamp')

# Train: first 22 months
# Validation: month 23
# Test: month 24 (held out)

# Walk-forward CV inside training: 5 folds, expanding window
```

---

## Step 7: Try Multiple Models

```python
import lightgbm as lgb

# Baseline: predict yesterday's same-time-same-queue volume
# Poisson deviance: 0.41
# WAPE: 12.4%

# Baseline: 7-day-ago value
# Poisson deviance: 0.32
# WAPE: 9.8%

# === Model 1: LightGBM with Poisson objective ===
poisson_model = lgb.LGBMRegressor(
    objective='poisson',
    n_estimators=1500,
    max_depth=6,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
# Walk-forward Poisson deviance: 0.18
# WAPE: 7.4%

# === Model 2: LightGBM with Tweedie objective (handles overdispersion) ===
tweedie_model = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5,  # tune
    n_estimators=1500,
    max_depth=6,
    num_leaves=63,
    learning_rate=0.05,
    random_state=42
)
# Walk-forward deviance: 0.16
# WAPE: 7.0%

# === Model 3: NegativeBinomial with statsmodels ===
import statsmodels.api as sm
# (slower, but proper NB likelihood)
nb_model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial())
# Slower; comparable performance to Tweedie LightGBM

# === Model 4: Prophet (per-queue) ===
# Captures multi-seasonality natively but doesn't use exogenous features well
# WAPE: 9.2% (worse than LightGBM with engineered features)
```

**Top performer:** LightGBM with Tweedie objective (WAPE 7.0%). Poisson is close behind.

---

## Step 8: Hyperparameter Tuning

```python
import optuna

def objective(trial):
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': trial.suggest_float('tv', 1.1, 1.9),
        'n_estimators': trial.suggest_int('n_estimators', 800, 2500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('mcs', 10, 100),
        'subsample': trial.suggest_float('sub', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('cbt', 0.6, 1.0),
    }
    model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42)
    return walk_forward_wape(model, X_train, y_train)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=80)
# Best WAPE: 6.4%
```

---

## Step 9: Quantile Bands for Staffing

```python
# WFM needs P10-P90 forecasts to plan staffing buffers
# Train separate quantile models

p10_model = lgb.LGBMRegressor(objective='quantile', alpha=0.10, ...)
p50_model = lgb.LGBMRegressor(objective='quantile', alpha=0.50, ...)
p90_model = lgb.LGBMRegressor(objective='quantile', alpha=0.90, ...)

p10_model.fit(X_train, y_train_log)
p50_model.fit(X_train, y_train_log)
p90_model.fit(X_train, y_train_log)

# Empirical coverage check
y_p10 = p10_model.predict(X_val)
y_p90 = p90_model.predict(X_val)
in_interval = ((y_val >= y_p10) & (y_val <= y_p90))
print(f"P10-P90 coverage: {in_interval.mean():.3f}")
# 0.842 (slightly under-cover; conformal prediction would tighten this)
```

---

## Step 10: Service Level Forecast (Downstream Calculation)

```python
# Erlang C formula maps (predicted_volume, predicted_AHT, agents_scheduled) -> SL
# We don't model SL directly; we model VOLUME, then WFM uses volume to compute required agents

def required_agents(volume_per_15min, aht_seconds, target_sl=0.80, target_asa_seconds=30):
    """Erlang C formula to compute agents needed."""
    # ... standard call center math
    return n_agents_needed

# At forecast time, for each (queue, 15-min window):
predicted_volume = best_model.predict(X_future)
predicted_volume_p90 = p90_model.predict(X_future)

# Staff to P90 forecast for safety margin
required_agents_for_sl = required_agents(predicted_volume_p90, recent_aht, target_sl=0.80)
```

---

## Step 11: Final Evaluation

```python
# Final model: LightGBM with Tweedie objective, tuned
# Test set: month 24 (most recent, held out)
#
# Performance:
#   Tweedie deviance:           0.14
#   WAPE on 15-min windows:     6.4%
#   WAPE on hourly aggregates:  4.8%   ✓ (target 8%)
#   Bias (avg pred - avg actual): +0.4% (slight over-forecast)
#
# Per-queue WAPE:
#   Billing:        7.1% (worst -- billing-cycle effects hardest to nail)
#   TechSupport:    6.0%
#   Sales:          5.4%
#   Retention:      6.8%
#   Claims:         5.9%
#   Escalations:    8.3% (worst -- low volume, rare events)
#
# Per-day-type WAPE:
#   Normal weekday:        5.1%
#   Weekend:               7.2%
#   Holiday:              12.6%  <- worst, but expected
#   Post-product-launch:   9.8%
```

---

## Step 12: Deployment

```python
# Nightly batch job:
#   1. Pull yesterday's actual volume for each (queue, 15-min)
#   2. Append to training table
#   3. Retrain LightGBM models (Tweedie + 3 quantiles) on rolling 18-month window
#   4. Generate next 7 days × 6 queues × 96 windows = 4032 forecasts
#   5. Apply Erlang C formula to get required agents
#   6. Write to WFM system

# Model size: ~50MB total (Tweedie + 3 quantile models)
# Nightly run: 22 min (well under target)

# Monitoring:
#   - Daily WAPE on hourly aggregates per queue
#   - P10-P90 coverage (should be ~0.80)
#   - Bias drift (cumulative over- or under-forecast over rolling 30 days)
#   - Holiday-effect catch (track holidays + post-holiday days as separate cohort)
#   - Major events: when known outage flagged, model forecasts higher; verify post-hoc
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Loss function | RMSE / MSE | Tweedie / Poisson (designed for counts; handles variance-mean relationship) |
| Distribution check | Skip | Variance/mean ratio = 30; chose NB / Tweedie over Poisson |
| Lag features | Just lag_1 | Lag 15m / 1h / 24h / 7d / 14d + rolling mean / max / std |
| Cyclical encoding | Use raw hour/day | sin/cos for minute_of_day, day_of_week, month |
| Domain features | Use raw date | Days_to_month_end, is_billing_peak_window, days_to/since_holiday |
| Service level | Predict directly | Predict volume; compute SL via Erlang C downstream |
| Staffing buffer | Use P50 prediction | Use P90 prediction; staff to upper bound |
| Train/test split | Random KFold | Walk-forward time-series CV |
| Outliers | Cap at 99th percentile | Keep — storm spikes are real signal |
| Per-queue handling | One global model | Queue as categorical + per-queue lag features |
| Holiday handling | Drop holiday days | Explicit holiday + day-after-holiday features |
| Feedback loop | Ignore | Use yesterday's SL as feature (callback pressure) |
