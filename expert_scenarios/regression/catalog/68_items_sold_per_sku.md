# Expert Scenario 68: Items Sold per SKU per Day (Intermittent Demand)

> **Complexity:** Hierarchical SKU × store × day forecast at billions-of-row scale, intermittent demand (most cells are zero), Croston's method for low-volume SKUs, Tweedie loss handles zero-inflation, weighted-MAPE matches business value, hierarchical reconciliation maintains aggregate accuracy.

---

## The Brief

A grocery retailer with 5,000 stores × 80,000 SKUs × 365 days = ~146B store-SKU-day combinations. Most cells are zero (a typical SKU sells in only 30% of stores on any given day). They want a 30-day-ahead daily forecast feeding:

- Replenishment: how many units to ship to each store?
- Promotion planning: predict demand under planned promotions.
- Discontinuation analysis: identify SKUs to phase out.

Constraints:

- WAPE (revenue-weighted MAPE) ≤ 14%.
- Mean Absolute Scaled Error (MASE) — better than naive baseline by 25%+.
- 30-day forecast at SKU × store × day granularity.
- Inference + reconciliation completes nightly in < 4 hours on commodity hardware.
- Handles intermittent demand correctly (Croston-style for low-volume SKUs).

This is harder than typical demand forecasting because: 70%+ of cells are zero (regular forecasting metrics like MAPE explode); a single SKU's behavior varies widely by store; promotional events disrupt patterns; and the scale (billions of forecasts) requires careful engineering.

---

## Step 1: Define the Problem Type

```
Type:           Regression on intermittent count (non-negative integer; mostly zero)
Primary Metric: WAPE — weighted by revenue (weight = unit_price × historical units)
Secondary:      MASE (scale-free, vs naive seasonal baseline)
                Per-SKU forecast skill at different volume tiers
Business Goal:  WAPE ≤ 14%; MASE < 0.75
Constraint:     Nightly batch < 4 hours
Target shape:   Heavy zero-inflation; among non-zero, Poisson-like
```

**Expert thinking:** standard MAPE divides by actual demand, which explodes when actual is 0 or 1. WAPE divides by total revenue (or units), giving a stable, business-meaningful metric. MASE compares against a naive seasonal baseline, providing a scale-free comparison.

---

## Step 2: Understand the Data

```
Shape: 146B (theoretical) cells; ~9B non-zero historical observations
Each row = (store_id, sku_id, date)

Per cell:
- units_sold (target; 0-200+)
- unit_price
- promo_flag
- promo_discount_pct
- holiday_flag
- store_metadata (size, region, urban/rural)
- sku_metadata (category, subcategory, brand, perishable)
- weather_features (per store, per day)
- competitor_promo_active

Volume tiers per SKU × store:
  HIGH:    sells 1+ unit on > 50% of days (~5% of cells, 70% of revenue)
  MEDIUM:  sells 1+ on 10-50% of days
  LOW:     sells 1+ on < 10% of days (intermittent)
```

---

## Step 3: Architecture — Volume-Tier-Specific Models

```python
# Different models for different volume tiers:
# - HIGH-volume: continuous demand; LightGBM regression with rich features
# - MEDIUM-volume: occasional zeros; LightGBM with Tweedie loss
# - LOW-volume: intermittent; Croston's method or simple ZI

# Per-cell tier classification (computed quarterly)
def classify_tier(history):
    nonzero_pct = (history['units_sold'] > 0).mean()
    if nonzero_pct >= 0.50:
        return 'HIGH'
    elif nonzero_pct >= 0.10:
        return 'MEDIUM'
    else:
        return 'LOW'

cell_tier = df.groupby(['store_id', 'sku_id']).apply(classify_tier)
```

---

## Step 4: HIGH-Volume Model (LightGBM)

```python
# For HIGH-volume cells, full feature engineering pays off
# Lag features, calendar, promotional context, weather

features_high = [
    'units_lag_1d', 'units_lag_7d', 'units_lag_28d',
    'units_ma_7d', 'units_ma_28d',
    'promo_flag', 'promo_lead_3d', 'promo_lead_7d',
    'holiday_flag', 'day_of_week', 'is_weekend',
    'unit_price', 'price_change_pct',
    'weather_temp', 'weather_precip', 'is_severe_weather',
    'competitor_promo_active',
    'sku_category_avg_units_30d',
    'store_traffic_30d_avg'
]

lgb_high = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1500, max_depth=6, num_leaves=63,
    learning_rate=0.04
)
lgb_high.fit(X_train_high, y_train_high)
# WAPE on HIGH cells: 9.2%
```

---

## Step 5: MEDIUM-Volume Model (Tweedie LightGBM)

```python
# For MEDIUM-volume cells (10-50% non-zero)
# Tweedie distribution handles the zero-inflated count well

lgb_medium = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5,
    n_estimators=1000, max_depth=5, num_leaves=31,
    learning_rate=0.05
)
lgb_medium.fit(X_train_medium, y_train_medium)
# WAPE on MEDIUM cells: 18%
```

---

## Step 6: LOW-Volume / Intermittent Model (Croston's)

```python
# Classical method for intermittent demand
# Forecasts:
#   1. Time between demands (TBD): exponentially smoothed series of intervals
#   2. Demand size when occurs: exponentially smoothed series of demands
#   3. Forecast = demand_size / TBD

def crostons_forecast(history, alpha=0.1):
    """
    Croston's Method for intermittent demand.
    Returns daily expected units.
    """
    nonzero_idx = np.where(history > 0)[0]
    if len(nonzero_idx) < 2:
        return history.mean() if history.sum() > 0 else 0
    
    intervals = np.diff(nonzero_idx).tolist()
    demands = history[nonzero_idx].tolist()
    
    # Exponential smoothing
    smoothed_interval = intervals[0]
    smoothed_demand = demands[0]
    for i in range(1, len(intervals)):
        smoothed_interval = alpha * intervals[i] + (1 - alpha) * smoothed_interval
        smoothed_demand = alpha * demands[i] + (1 - alpha) * smoothed_demand
    
    return smoothed_demand / smoothed_interval

# Apply per-cell on the LOW-volume cells
low_forecasts = df_low.groupby(['store_id', 'sku_id']).apply(
    lambda g: crostons_forecast(g['units_sold'].values)
)
# Mean Absolute Scaled Error (MASE) on LOW cells: 0.62
# Better than naive baseline (0.95)
```

**Expert insight:** Croston's method (1972) is still the standard for intermittent demand in operations research. It avoids the trap of "predict 0 every day because most days are 0" by separately modeling the rate of demand events and the size when they occur.

---

## Step 7: Hierarchical Reconciliation

```python
# Per-cell forecasts × 5K stores × 80K SKUs = noise + drift
# Aggregate to higher levels (store-day, SKU-day, region-day) for sanity

# Top-down approach: forecast aggregate first, allocate to cells
# Bottom-up: forecast cells, sum to aggregates
# MinT: optimal mixed approach

# In practice, apply MinT post-hoc reconciliation:
# 1. Generate per-cell forecasts (independent)
# 2. Aggregate to (store-day, SKU-day, region-day) levels
# 3. Adjust per-cell forecasts so that aggregates match a reconciled total

from hierarchicalforecast.methods import MinTrace

reconciler = MinTrace(method='ols')
reconciled_forecasts = reconciler.reconcile(individual_forecasts, hierarchy_structure)

# Improvement: aggregate-level WAPE drops from 18% to 12%
# Per-cell WAPE: 14% → 14% (unchanged; reconciliation tightens aggregates)
```

---

## Step 8: Final Evaluation

```python
# Test period: 30 days held out
# Evaluation across all volume tiers
#
# Performance:
#   Overall WAPE:           13.4%   ✓ (target 14%)
#   Per-tier WAPE:
#     HIGH:    9.1%
#     MEDIUM: 18%
#     LOW:    35% (per-cell; aggregate is much better)
#   MASE:                   0.71    ✓ (better than 0.75 target)
#   
# Aggregate-level (after reconciliation):
#   Store × Day:   WAPE 8.2%
#   SKU × Day:     WAPE 9.5%
#   Region-level:  WAPE 5.4%
#
# Inference time: 3.2 hours nightly (within 4h budget)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Architecture | One global model | Per-volume-tier ensemble (HIGH/MEDIUM/LOW) |
| Loss for intermittent | RMSE / MSE | Tweedie for medium; Croston's for low-volume |
| Metric | MAPE | WAPE (revenue-weighted) + MASE (scale-free) |
| Hierarchical structure | Forecast each cell independently | MinT reconciliation across aggregation levels |
| Zero-inflation | Treat zeros as bad predictions | Croston's separates demand-rate from demand-size |
| Per-tier evaluation | Aggregate metric | Per-tier WAPE and MASE breakdown |
| Volume tier classification | Static | Quarterly recompute as patterns shift |
| Memory at scale | Eager evaluation | Stream per-store; batch per-tier |
