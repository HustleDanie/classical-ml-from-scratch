# Expert Scenario 31: Daily Sales Forecast

> **Complexity:** Time-series with multi-seasonality (weekly, yearly, holiday), hierarchy across 200 stores × 50 SKUs, walk-forward validation mandatory, promotional events dominate forecast errors, the wrong baseline lets weak models look good.

---

## The Brief

A retail chain gives you 3 years of daily sales for 200 stores × 50 SKUs (~11M rows). They want a 7-day-ahead forecast for each store-SKU combination, refreshed nightly. The forecast feeds:

- Replenishment orders to suppliers (overforecasting → excess inventory; underforecasting → stockouts).
- Labor scheduling (high-volume days need more staff).
- Marketing budget allocation (which stores to push promotions in).

Constraints:

- Forecast across the entire 200×50 matrix in under 30 minutes nightly.
- Promotion/holiday awareness — promotional days drive 35% of revenue.
- MAPE under 18% weighted by SKU revenue (top SKUs matter more).
- Per-store explainability — store managers ask "why is the model predicting low for next Tuesday?"

This is harder than a standard time-series benchmark because the hierarchy explodes search space, intermittent SKUs (sells 0 most days) break naive metrics, and a "good" point forecast is worthless if it ignores the once-a-month promo.

---

## Step 1: Define the Problem Type

```
Type:           Time-series regression on count-like daily sales
Primary Metric: Weighted MAPE (weighted by 90-day rolling SKU revenue)
Secondary:      RMSE; per-store MAPE
Business Goal:  WMAPE under 18%; never miss a promotion-day forecast by > 30%
Constraint:     30-minute nightly run for 10,000 store-SKU pairs
Target shape:   Right-skewed counts; 22% of rows are zero (intermittent SKUs)
```

**Expert thinking:** WMAPE (weighted MAPE) means a 5% error on a $10K-revenue SKU matters more than a 50% error on a $200-revenue SKU. This is the right metric for retail because dollars matter, not SKU counts. Standard MAPE would let "I never sell, predicted 0, actually sold 1" SKUs blow up the metric.

---

## Step 2: Understand the Data

```
Shape: 11,000,000 rows (3 years × 200 stores × 50 SKUs × ~365 days)
Target: 'units_sold' -- 0 to 1,200, median 4, P99 38

Columns:
- date          (datetime)
- store_id      (200 unique)
- sku_id        (50 unique)
- units_sold    (target -- non-negative integer)
- unit_price    (float)
- promo_flag    (binary: was this SKU on promo this day?)
- promo_discount (float, 0-50%, NaN when not on promo)
- holiday_flag  (binary: federal holidays)
- holiday_name  (string, NaN when not holiday)
- weather_temp_c (float, daily high)
- weather_precip_mm (float)
- store_format (Small/Medium/Large)
- store_region (5 regions)
- sku_category (10 categories: dairy, frozen, beverage, etc.)
- sku_subcategory (40 unique)
- competitor_promo (binary: known competitor promo this day)
```

**Expert thinking:** key observations:
- 200×50 = 10,000 series. Hierarchical: stores nest within regions, SKUs nest within categories.
- 22% of (store-SKU-day) rows are zero. Intermittent demand for niche SKUs.
- Promotional days are the high-leverage prediction targets. Missing a promo prediction = lost sales.
- Holidays compound with promos. Memorial Day + grill SKU + promo = extreme spike.
- Weather matters for seasonal SKUs (ice cream, soup, sunscreen).

---

## Step 3: Exploratory Data Analysis (EDA)

```python
df['units_sold'].describe()
# count    11,000,000
# mean      8.2
# std      19.4
# min       0
# 25%       0
# 50%       4
# 75%       9
# max     1,205
# Skew: 14.3 (heavy right tail driven by promos + top SKUs)

# Promo effect
df.groupby('promo_flag')['units_sold'].agg(['mean', 'median', 'count'])
# promo=0: mean 5.6, median 3
# promo=1: mean 38.4, median 21
# Promotional days lift sales 7x on average

# Day-of-week pattern
df.groupby(df['date'].dt.dayofweek)['units_sold'].mean()
# Mon-Thu: ~5-7
# Fri:      9.1
# Sat:     12.4
# Sun:      8.3

# Seasonal pattern
df.groupby(df['date'].dt.month)['units_sold'].mean()
# Dec: 14.2 (holiday spike)
# Nov: 9.8  (Thanksgiving / Black Friday)
# Aug: 7.1
# Feb: 5.4 (post-holiday lull)

# Zero-inflation pattern
zero_pct_by_sku = df.groupby('sku_id').apply(lambda g: (g['units_sold']==0).mean())
# 12 SKUs sell on 90%+ of days (top sellers)
# 18 SKUs sell on < 30% of days (intermittent)

# Holiday-promo compound effect
df[(df['holiday_flag']==1) & (df['promo_flag']==1)]['units_sold'].mean()  # 87.2
df[(df['holiday_flag']==0) & (df['promo_flag']==0)]['units_sold'].mean()  # 4.3
# Holiday + promo = 20x base
```

**Findings:**

| Finding | Implication |
|---------|------------|
| 22% zeros | Need Tweedie or zero-inflated framing OR aggregate to weekly |
| Promos lift 7x | The single most important feature — must be in training and known at forecast time |
| Holiday × promo compound | Need interaction features (or a model that learns interactions, e.g., GBM) |
| Day-of-week + month seasonality | Standard time-series features (sin/cos cyclical encoding or one-hot) |
| Top 12 SKUs continuous; bottom 18 intermittent | Two-tier model: continuous regression for high-volume; Croston's or zero-inflated for low |
| Weather correlates with seasonal SKUs (r=0.31 for ice cream) | Include weather features |

**Expert insight:** the 22% zero rate has a structural meaning. For a snack-cake SKU at a small rural store, "sells 0 most days" is the steady state — not noise. Don't try to predict that as a "regression" — predict zero confidently and only forecast non-zero on event days.

---

## Step 4: Data Cleaning

```python
# === HANDLE PROMOS ===
df['promo_flag'] = df['promo_flag'].fillna(0).astype(int)
df['promo_discount'] = df['promo_discount'].fillna(0)

# === HOLIDAYS ===
df['holiday_flag'] = df['holiday_flag'].fillna(0).astype(int)
# Encode holiday_name as the holiday "category"
df['holiday_name'] = df['holiday_name'].fillna('NoHoliday')

# === WEATHER ===
# Some weather data missing for some store-day combos (~3%)
df['weather_temp_c'] = df.groupby('store_region')['weather_temp_c'].transform(
    lambda x: x.fillna(x.rolling(7, min_periods=1).mean())
)
df['weather_precip_mm'] = df['weather_precip_mm'].fillna(0)

# === OUTLIERS ===
# Don't cap! A 1000-unit Black Friday sale is REAL.
# But mark severe outliers for the loss function:
df['is_outlier_sale'] = (df['units_sold'] > df.groupby(['store_id','sku_id'])['units_sold']
                                            .transform(lambda x: x.quantile(0.99) * 3)).astype(int)
# About 0.3% of rows -- usually one-off events
```

---

## Step 5: Feature Engineering — the Heart of Time-Series

```python
# === LAG FEATURES ===
# These are computed PER store-SKU and require careful handling at forecast time
df = df.sort_values(['store_id', 'sku_id', 'date'])
group = df.groupby(['store_id', 'sku_id'])

# Lags
df['units_lag_1']  = group['units_sold'].shift(1)
df['units_lag_7']  = group['units_sold'].shift(7)
df['units_lag_14'] = group['units_sold'].shift(14)
df['units_lag_28'] = group['units_sold'].shift(28)
df['units_lag_365'] = group['units_sold'].shift(365)  # year-ago

# Rolling means
df['units_ma_7']   = group['units_sold'].shift(1).rolling(7).mean()
df['units_ma_28']  = group['units_sold'].shift(1).rolling(28).mean()
df['units_ma_84']  = group['units_sold'].shift(1).rolling(84).mean()  # ~quarter

# Rolling std (volatility)
df['units_std_28'] = group['units_sold'].shift(1).rolling(28).std()

# Trend indicator
df['units_trend_7v28'] = df['units_ma_7'] / (df['units_ma_28'] + 0.1)

# Last promotional day -- how many days ago?
df['days_since_promo'] = group.apply(
    lambda g: g['promo_flag'].cumsum().diff().fillna(0).clip(lower=0).cumsum()
)  # simplified: actual implementation tracks the running gap

# === CALENDAR FEATURES ===
df['day_of_week'] = df['date'].dt.dayofweek
df['day_of_month'] = df['date'].dt.day
df['week_of_year'] = df['date'].dt.isocalendar().week
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

# Cyclical encoding (helps linear models, doesn't hurt trees)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# === HOLIDAY PROXIMITY ===
# Days until the next Thanksgiving / Christmas / July 4
holidays = pd.to_datetime(['2020-12-25', '2021-12-25', '2022-12-25', '2023-12-25',
                            '2020-11-26', '2021-11-25', '2022-11-24', '2023-11-23'])
df['days_to_christmas'] = df['date'].apply(
    lambda d: min((h - d).days for h in holidays if (h - d).days >= 0 and (h - d).days < 60)
    if any((h - d).days >= 0 and (h - d).days < 60 for h in holidays) else 60
)

# === PROMO FEATURES (CRITICAL) ===
# Promo info is available AT FORECAST TIME (planned in advance)
df['promo_lag_7']  = group['promo_flag'].shift(7)
df['promo_lead_7'] = group['promo_flag'].shift(-7)  # known a week ahead
df['promo_lead_3'] = group['promo_flag'].shift(-3)
df['promo_lead_1'] = group['promo_flag'].shift(-1)
df['planned_discount'] = group['promo_discount'].shift(-1)  # forecast-time available

# === STORE × SKU INTERACTIONS ===
# Target encoding for store_id × sku_id (computed on TRAIN only, in CV)
sku_store_mean = df_train.groupby(['store_id', 'sku_id'])['units_sold'].mean()
df['store_sku_avg'] = df.set_index(['store_id', 'sku_id']).index.map(sku_store_mean).fillna(0)

# === HIERARCHY FEATURES ===
df['region_avg_yesterday'] = group['units_sold'].shift(1).reset_index(level=[0,1], drop=True)
# Region-level signal: when ALL stores in a region had a high day, that's a regional signal
```

**Expert insight:** lag features (especially `units_lag_7`, `units_ma_28`, `units_lag_365`) are usually 60-70% of the predictive signal in retail forecasting. The `promo_lead_*` features are uniquely available because promos are planned in advance — never use lag features that wouldn't be available at prediction time, but always use leading features that ARE available.

---

## Step 6: Feature Selection

```python
# 60+ features after engineering
# Keep all -- LightGBM handles redundancy well, and removing any costs information
# Run a feature importance check after first model:

# Initial LightGBM, default params, all features:
# Top 10 importance:
#   units_lag_7:                28.3%
#   units_ma_28:                14.1%
#   promo_lead_1:               11.7%
#   units_lag_1:                 9.2%
#   day_of_week:                 6.1%
#   units_lag_365:               5.8%
#   month:                       4.9%
#   promo_flag (current row):    4.4%
#   store_sku_avg:               3.7%
#   days_to_christmas:           2.8%

# Bottom features:
#   weather_precip_mm:    0.4%
#   is_month_start:       0.2%
#   is_outlier_sale:      0.1%

# Drop bottom 5 features (noise more than signal)
```

---

## Step 7: Preprocessing

```python
# LightGBM doesn't need scaling. Just label-encode categoricals.

categoricals_for_lgbm = ['store_id', 'sku_id', 'sku_category', 'sku_subcategory',
                          'store_format', 'store_region', 'holiday_name']
# LightGBM can use raw category encoding via `categorical_feature` parameter

# For linear models (ElasticNet baseline), need StandardScaler + OneHot
```

---

## Step 8: Train/Test Split — Walk-Forward Validation

```python
# CARDINAL RULE OF TIME-SERIES: walk-forward validation, NEVER random split.
#
# Setup: 3 years of data (Jan 2021 - Dec 2023)
# Use last 90 days as final test
# Walk-forward CV on the remaining ~975 days

# Walk-forward fold scheme (5 folds):
#   Fold 1: Train [Jan 2021 - Jun 2022], Validate [Jul-Sep 2022]
#   Fold 2: Train [Jan 2021 - Sep 2022], Validate [Oct-Dec 2022]
#   Fold 3: Train [Jan 2021 - Dec 2022], Validate [Jan-Mar 2023]
#   Fold 4: Train [Jan 2021 - Mar 2023], Validate [Apr-Jun 2023]
#   Fold 5: Train [Jan 2021 - Jun 2023], Validate [Jul-Sep 2023]
# Final test: Oct-Dec 2023 (held out, never seen during model selection)

from sklearn.model_selection import TimeSeriesSplit
# (note: TimeSeriesSplit with custom splits since we want each fold to span >= 90 days)
```

**Expert insight:** the most common time-series modeling failure is using `KFold` (random split). It leaks future patterns into training, and the validation MAPE looks great. In production the model performs 30-50% worse. Walk-forward validation is non-negotiable.

---

## Step 9: Baselines

```python
# Baseline 1: predict yesterday's value (naive seasonal)
# WMAPE: 31%

# Baseline 2: predict last-week-same-day value
# WMAPE: 24% (captures weekly seasonality)

# Baseline 3: 28-day moving average (units_ma_28)
# WMAPE: 22%

# Baseline 4: SARIMA per series (top 50 highest-volume series only -- impractical for all 10K)
# WMAPE: 19% on the high-volume series, but does not handle promos

# Targets to beat: 22% WMAPE (the moving average baseline)
```

---

## Step 10: Try Multiple Models with Walk-Forward CV

```python
import lightgbm as lgb

# Model 1: ElasticNet (baseline linear)
# Walk-forward CV WMAPE: 21.4%

# Model 2: Random Forest (n=200, max_depth=12)
# Walk-forward CV WMAPE: 17.8%

# Model 3: Gradient Boosting (sklearn, n=300, max_depth=5)
# Walk-forward CV WMAPE: 16.9%
# But TRAINING TIME: 4.2 hours. Not viable for nightly retraining at this scale.

# Model 4: LightGBM (n=1000, max_depth=-1, num_leaves=64)
# Walk-forward CV WMAPE: 15.6%
# TRAINING TIME: 14 minutes. ✓

# Model 5: XGBoost
# Walk-forward CV WMAPE: 15.9%
# TRAINING TIME: 28 minutes.

# Model 6: Prophet (per-series, top 50 only)
# Walk-forward CV WMAPE on top 50: 18.2%
# Doesn't scale to all 10K series.
```

**Top performer:** LightGBM (15.6% WMAPE in CV, 14-min training).

---

## Step 11: Tweedie Loss for the Zero-Inflated Target

```python
# Target distribution: 22% zeros + right-skewed positives
# This is exactly the case Tweedie loss is built for

lgbm_tweedie = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.4,  # tune this
    n_estimators=1000,
    max_depth=-1,
    num_leaves=64,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
# Walk-forward CV WMAPE with Tweedie: 14.8%
# Better! Tweedie objective optimizes the zero-inflated count distribution natively.
```

**Expert insight:** Tweedie distribution is a generalization that smoothly interpolates between Poisson (counts) and Gamma (positive continuous) distributions. With `variance_power=1.4`, it handles "lots of zeros + some positive counts" — the exact retail demand pattern. RMSE loss treats zeros as negative-distance from positive predictions; Tweedie treats zeros as their own probability mass.

---

## Step 12: Hyperparameter Tuning

```python
# Use Optuna for efficient search across LightGBM space
import optuna

def objective(trial):
    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 500, 2000),
        'max_depth':       trial.suggest_int('max_depth', -1, 12),
        'num_leaves':      trial.suggest_int('num_leaves', 31, 255),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha':       trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda':      trial.suggest_float('reg_lambda', 0, 5),
        'tweedie_variance_power': trial.suggest_float('tv_power', 1.1, 1.9),
        'objective':       'tweedie'
    }
    # Run 3-fold walk-forward
    return walk_forward_wmape(params, X_train, y_train)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=80)
# Best WMAPE: 14.2%
# Best params: n=1500, max_depth=8, num_leaves=127, lr=0.04, tv_power=1.35
```

---

## Step 13: Hierarchical Reconciliation

```python
# Forecast 10,000 series independently. But the forecasts must "reconcile"
# (sum-up store forecasts == region forecast == total forecast)

# Without reconciliation, model often forecasts 200 store forecasts that
# sum to 5% more than the regional aggregate (because per-store models
# overshoot independently)

# Bottom-up: trust per-series forecasts, sum to higher levels
# Top-down: forecast totals, allocate by historical share
# MinT (Minimum Trace): optimal reconciliation that minimizes variance

# Use MinT (from python `hts` library or `nixtla/hierarchicalforecast`):
# Improves WMAPE on regional aggregates from 14.2% -> 12.8%
# Slightly degrades store-level forecasts (15.1% vs 14.2%)
# Trade-off acceptable: replenishment runs at SKU-day level (unchanged); inventory
# planning runs at regional level (improved)
```

**Expert insight:** for hierarchical forecasting, the choice of reconciliation strategy is a business decision, not a stats decision. Operations cares about higher-level aggregates being right; replenishment cares about per-SKU. MinT is a balanced compromise.

---

## Step 14: Promo-Specific Validation

```python
# Standard WMAPE averages over all days. But promo days are 12% of days
# generating 35% of revenue. We need to validate them specifically.

# WMAPE on promo days only:           17.3%
# WMAPE on non-promo days only:        12.1%
# WMAPE on holiday + promo days:       22.4% (worst!)

# The model is decent overall but worse on promo days.
# Investigation: top 5 underforecast events were "first-time-promoted SKUs" with
# no historical promo data for that SKU at that store.

# Mitigation: add a feature for "is_first_promo_for_this_sku_at_this_store"
# After re-training: WMAPE on promo days drops from 17.3% to 14.8%
```

---

## Step 15: Final Evaluation on Held-Out Test Set

```python
# Final model: LightGBM with Tweedie objective + MinT reconciliation
# Test set: Oct-Dec 2023 (90 days, 10,000 series, ~900K predictions)

# Performance:
#   Overall WMAPE:             13.9%        ✓ (target 18%)
#   WMAPE per region (best):   11.2%
#   WMAPE per region (worst):  16.1%
#   Promo-day WMAPE:           14.6%
#   Holiday-day WMAPE:         18.2% (worst day-type)
#   Bias (sum of pred - sum of actual): +1.2% (slight over-forecast)
#   Catastrophic errors (|MAPE| > 50%): 4.1% of forecasts (mostly intermittent SKUs)
```

---

## Step 16: Per-Store Explainability

```python
import shap

# For each store-SKU-day forecast, surface top 5 SHAP drivers
# Store managers ask "why is the model predicting low for next Tuesday?"

# Example output:
# Store 47, SKU "Frozen Pizza", date 2024-01-09 (Tuesday)
# Predicted: 8 units. Last Tuesday: 14 units.
# Top SHAP drivers:
#   1. units_lag_7: 14 units last Tuesday  (+5.2)
#   2. promo_lead_1: NO promo planned     (-3.8)  <-- this is why prediction is lower
#   3. day_of_week: Tuesday                (-1.4)
#   4. units_ma_28: 9.1                    (+1.2)
#   5. weather_temp_c: 28°F (cold)         (+0.7)
# Manager-readable explanation:
#   "Last Tuesday had a promo; this Tuesday doesn't. Without the promo, expect ~8 units."
```

---

## Step 17: Deployment Considerations

```python
# Model size: ~250MB (LightGBM with 1500 trees + categorical encodings)
# Nightly retrain: 14 minutes on 11M rows
# Nightly forecast generation (10K series × 7 days): 4 minutes
# Total nightly run: ~20 minutes ✓ (target 30)

# Production architecture:
#   1. ETL: pull yesterday's actuals; append to training set; refresh lag features
#   2. Retrain LightGBM on rolling 18-month window (newer = more relevant)
#   3. Generate 7-day forecast for all 10K series
#   4. Apply MinT reconciliation at region/total levels
#   5. Write to forecast warehouse; replenishment & labor systems consume

# Monitoring:
#   - Daily WMAPE on actuals (alerts if > 20%)
#   - Per-region WMAPE drift
#   - First-time-promo-SKU MAPE (separate cohort)
#   - Bias direction over rolling 30 days (alert if > 5% sustained)

# Drift mitigation:
#   - When new SKU launches: warm-start with category-mean for 7 days, then per-SKU
#   - When new store opens: warm-start with regional-mean
#   - When promo plan changes mid-week: recompute forecasts intra-day
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Train/test split | Random KFold | Walk-forward time-series CV |
| Loss function | RMSE | Tweedie (handles 22% zero + right-skew) |
| Lag features | None / shift(1) only | Lag 1/7/28/365 + rolling means + std |
| Promo feature | "promo today" only | Promo lead 1/3/7 (planned in advance, available at forecast time) |
| Categorical handling | One-hot 250 columns | LightGBM native categorical (or target encoding) |
| Outliers | Cap at 99th percentile | Keep — Black Friday IS the signal |
| Hierarchical structure | Forecast independently | MinT reconciliation: store + region + total all consistent |
| Promo-day validation | Overall MAPE only | Separate validation on promo-day subset |
| Metric | MAPE | WMAPE (revenue-weighted) — captures business value |
| First-time promos | Treated like established | Special feature flag; warm-start fallback |
| Explanation | "model predicts X" | Top-5 SHAP drivers per forecast for store managers |
| Reconciliation | None | MinT to keep regional aggregates consistent |
