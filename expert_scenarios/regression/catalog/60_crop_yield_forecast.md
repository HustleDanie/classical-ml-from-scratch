# Expert Scenario 60: Crop Yield Forecast (Agricultural Regression)

> **Complexity:** Small annual data (one growing season per year), heavy weather dependence with non-linear thresholds, climate change drift makes old data less relevant, multi-spatial hierarchy (region → county → field), ground-truth labels arrive only at harvest, the predictions feed multi-million-dollar planting and storage decisions.

---

## The Brief

A regional grain cooperative gives you 12 years of corn yield data across 4,200 fields in 6 US Midwest states. Each field has:

- 12 seasonal yield observations (one per year)
- Soil characteristics (semi-static, measured every 3 years)
- Daily weather data (50+ features)
- Mid-season satellite-derived NDVI imagery features
- Planting and management records

The cooperative wants:

- August 1 yield forecasts for each field (~6 weeks before harvest).
- District-level (county) and regional aggregates with confidence intervals.
- A model that respects the climate-change drift — recent years matter more than 2013.
- Identification of which fields are at greatest risk of significant yield loss (so insurance products can be priced).

Constraints:
- Only 12 observations per field (one per year). Total ~50K field-year rows.
- Weather and satellite features are dense; soil features are sparse.
- Yield distribution is left-skewed (typical years are good; bad years are catastrophic).
- The August 1 prediction must beat USDA-NASS (national-statistics) August forecasts by at least 5% MAPE.

This is harder than typical regression because: extreme small-sample-per-field, weather-yield is non-linear (drought hurts, but flood also hurts), climate trends mean naive averaging of past years is biased, and stakeholder decisions (insurance pricing, futures hedging) are time-sensitive and money-laden.

---

## Step 1: Define the Problem Type

```
Type:           Regression on bushels per acre (target = 80-280, mean ~190)
Primary Metric: MAPE per field; Bias per region (we don't want systematic over- or
                under-prediction — that confounds policy decisions)
Secondary:      RMSE; coverage of P10-P90 prediction intervals;
                Districtwide aggregate accuracy
Business Goal:  MAPE < 8% per field; aggregate region MAPE < 3%
                Beat USDA NASS August forecast by >= 5pp MAPE
Constraint:     August 1 prediction; explainable (insurance underwriting requires it);
                respects climate-change drift
Target shape:   Left-skewed (top is bounded by physical limits; bottom can be much worse)
```

**Expert thinking:** the metric pair (per-field MAPE + region aggregate MAPE) reflects different stakeholders. Per-field accuracy serves insurance underwriting; aggregate accuracy serves storage and pricing decisions. The two diverge: a model can be well-calibrated regionally even if individual fields are noisy.

---

## Step 2: Understand the Data

```
Shape: ~50,000 field-years (4,200 fields × 12 years)
Target: yield_bushels_per_acre

Per field-year features:

FIELD STATIC (semi-static, refreshed every 3 years):
- field_id, county_id, state
- field_lat, field_lng
- field_size_acres
- soil_type (8 categories: silt loam, clay, sandy clay, etc.)
- soil_organic_matter_pct
- soil_pH
- soil_phosphorus, soil_potassium
- drainage_class (Excellent / Good / Moderate / Poor)
- irrigation_available (binary)
- avg_elevation_meters
- slope_pct

PLANTING / MANAGEMENT (annual):
- planting_date (day of year, 105-145 typical)
- seed_variety (40+ varieties)
- planting_density_seeds_per_acre
- seed_relative_maturity (days)
- nitrogen_total_lbs (60-300)
- phosphorus_lbs
- potassium_lbs
- previous_crop (Corn / Soybeans / Wheat / Cover Crop / Other)
- tillage_method (NoTill / Reduced / Conventional)

WEATHER (May 1 to July 31, daily — 92 days):
- temperature_max_daily (92 values)
- temperature_min_daily (92 values)
- precipitation_daily (92 values)
- solar_radiation_daily (92 values)
- vapor_pressure_deficit_daily (92 values)
- wind_speed_daily (92 values)

SATELLITE (4 imagery passes between June 15 and July 31):
- ndvi_jun_15  (vegetation index)
- ndvi_jul_01
- ndvi_jul_15
- ndvi_jul_31
- canopy_cover_jul_31
- biomass_index_jul_31

DROUGHT / STRESS:
- usda_drought_intensity_at_aug_1 (0-4 scale, 0=normal, 4=exceptional drought)
- consecutive_no_rain_days_max
- heat_stress_days_above_95F (count May-July)
- chill_stress_days_below_50F (count)

MARKET / EXOGENOUS:
- corn_futures_price_aug (proxy for market expectation)
- crop_insurance_coverage_level (used for policy adverse selection)
```

**Expert thinking:** the 92 daily weather values are the core signal. Most yield models use aggregated weather (e.g., July rainfall, June heat days) but lose information. Modern approaches keep the daily granularity and let the model find interactions.

---

## Step 3: EDA

```python
df['yield_bushels_per_acre'].describe()
# count    50,000
# mean    192.4
# std      34.6
# min      45  (catastrophic year/field)
# 25%     176
# 50%     195
# 75%     214
# max     280
# Skewness: -0.6 (LEFT-skewed -- top is bounded; bottom has long tail)

# Yield by year (climate trend visible)
df.groupby('year')['yield_bushels_per_acre'].mean()
# 2013: 168 (drought year)
# 2014: 184
# 2015: 191
# 2016: 197
# 2017: 195
# 2018: 191
# 2019: 175 (heavy spring rain)
# 2020: 199
# 2021: 203
# 2022: 198
# 2023: 207
# 2024: 211 (technology + climate trend)

# Yearly trend: ~3-4 bushels/acre/year improvement (genetics, technology)
# But high-variance: a drought year can drop average 25 bu/ac

# Soil correlation
df.groupby('soil_type')['yield_bushels_per_acre'].mean().sort_values()
# Sandy clay:        165
# Silt loam:          198 (best)
# Heavy clay:         172
# ...

# Weather threshold non-linearities
# Total July rainfall 2-4 inches: avg yield 198
# Total July rainfall < 1 inch:   avg yield 161 (drought)
# Total July rainfall > 6 inches: avg yield 178 (root rot)
# CLEAR U-shape: too dry OR too wet hurts
```

---

## Step 4: Data Cleaning

```python
# === HANDLE CLIMATE-CHANGE DRIFT ===
# The technology-yield trend is real and persistent. Naive averaging treats 2013 same as 2023.
# Detrend: subtract a yearly mean BEFORE training; add it back at inference

trend_per_year = df.groupby('year')['yield_bushels_per_acre'].mean()
# 2013 average yield 168, 2024 average yield 211 → 3.9/yr trend

# Fit a linear trend
import statsmodels.api as sm
years = sorted(trend_per_year.index)
trend_model = sm.OLS(trend_per_year.values, sm.add_constant(np.array(years).reshape(-1, 1))).fit()

# Detrended yield = actual - predicted_trend
df['yield_detrended'] = df['yield_bushels_per_acre'] - trend_model.predict(sm.add_constant(df['year'].values.reshape(-1, 1)))

# Train on yield_detrended; at inference add trend back

# === HANDLE MISSING SOIL DATA ===
# 8% of fields have outdated soil tests; impute with county median
soil_features = ['soil_organic_matter_pct', 'soil_pH', 'soil_phosphorus', 'soil_potassium']
for col in soil_features:
    df[col] = df.groupby('county_id')[col].transform(lambda x: x.fillna(x.median()))

# === REMOVE ANOMALOUS FIELDS ===
# Some fields have 200+ bushel decade-average; some have 80
# These are real (different field productivity)
# But fields with single-year drops > 50% from their own multi-year mean are usually
# data errors or completely unrelated events (hail damage, etc.)
df = df.groupby('field_id').filter(
    lambda g: ((g['yield_bushels_per_acre'] - g['yield_bushels_per_acre'].mean()).abs() < 60).all()
)
```

---

## Step 5: Feature Engineering

```python
# === WEATHER AGGREGATIONS ===
# For each daily weather variable, compute biologically meaningful aggregations
# Don't just sum / average — use thresholds that match crop physiology

# Growing degree days (GDD) -- corn-specific
# GDD = max(0, (max_temp - min_temp)/2 + min_temp - 50) ... bounded at 86
def calc_gdd(max_temps, min_temps):
    gdd = ((np.minimum(max_temps, 86) + np.maximum(min_temps, 50)) / 2 - 50).clip(0, None)
    return gdd.sum()

df['gdd_total_may_jul'] = df.apply(lambda r: calc_gdd(r['temp_max_daily'], r['temp_min_daily']), axis=1)

# === HEAT STRESS (pollination risk) ===
# Corn pollinates around July 5-15 in this region
# Heat above 95F during pollination dramatically reduces yield
df['heat_stress_pollination'] = df['temp_max_daily_jul_5_to_15'].apply(
    lambda x: (np.array(x) > 95).sum()
)

# === DROUGHT INDICES ===
df['drought_severity_jul'] = df['usda_drought_intensity_at_aug_1']  # already aggregated

# Vapor pressure deficit (VPD) -- combination of temperature and humidity
df['vpd_avg_critical'] = df['vapor_pressure_deficit_daily_jun_15_to_jul_31'].mean()

# === RAINFALL DISTRIBUTION ===
df['precip_total_may_jul'] = df['precipitation_daily'].apply(sum)
df['precip_no_rain_days_max'] = df['consecutive_no_rain_days_max']
df['precip_excess_days'] = df['precipitation_daily'].apply(lambda x: (np.array(x) > 1.5).sum())

# === SATELLITE FEATURES ===
df['ndvi_growth_rate'] = df['ndvi_jul_15'] - df['ndvi_jun_15']
df['ndvi_max'] = df[['ndvi_jul_01', 'ndvi_jul_15', 'ndvi_jul_31']].max(axis=1)
df['low_ndvi_at_pollination'] = (df['ndvi_jul_15'] < 0.5).astype(int)

# === COMPOSITE STRESS SCORE ===
df['stress_score'] = (
    (df['drought_severity_jul'] >= 2).astype(int) * 2 +
    df['heat_stress_pollination'] +
    df['low_ndvi_at_pollination'] * 2 +
    (df['precip_total_may_jul'] < 8).astype(int) +
    (df['precip_excess_days'] > 8).astype(int)
)

# === MANAGEMENT INTERACTIONS ===
df['nitrogen_per_acre_z_county'] = df.groupby('county_id')['nitrogen_total_lbs'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# === FIELD HISTORICAL YIELD ===
# Each field has its own typical yield (genetic potential of the soil/management)
# Use field's own past 3-year average as a feature
df['field_avg_yield_3y_back'] = df.groupby('field_id')['yield_bushels_per_acre'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

# === DETRENDED HISTORICAL ===
df['field_detrended_avg_3y_back'] = df.groupby('field_id')['yield_detrended'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
```

**Expert insight:** weather features must be engineered with crop biology in mind. Generic "average July temperature" is much weaker than "GDD during pollination" or "heat stress days during silking." The biological knowledge encoded in feature engineering is often worth more than model complexity.

---

## Step 6: Train/Test Split — Year-Based

```python
# CRITICAL: split by YEAR, not by field, to simulate real prediction
# Training: 2013-2022
# Validation: 2023
# Test: 2024 (most recent harvest year)

# This is leave-one-year-out CV applied to the temporal structure
# Random split would let the model see 2024 patterns in training -- impossible in practice

# Each fold: train on 9 years, predict 1 year
# Mean MAPE across folds = honest estimate of generalization
```

---

## Step 7: Try Multiple Models

```python
# === Model 1: Linear Regression ===
# Year-CV MAPE: 9.4%

# === Model 2: ElasticNet ===
# Year-CV MAPE: 8.6%

# === Model 3: Random Forest (n=300, max_depth=15) ===
# Year-CV MAPE: 7.1%

# === Model 4: LightGBM ===
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    max_depth=8,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
# Year-CV MAPE: 6.4%

# === Model 5: XGBoost ===
# Year-CV MAPE: 6.5%

# === Model 6: LightGBM with quantile loss for prediction intervals ===
# Train P10, P50, P90 separately
# All converge to similar MAPE for the median
```

**Top performer:** LightGBM (MAPE 6.4%).

---

## Step 8: Time-Decay Weighting (Climate-Aware)

```python
# Recent years are more representative of current climate / technology
# Down-weight 2013-2016 data; up-weight 2022-2024

# Exponential decay weights
def time_weight(year, current_year=2024, half_life=5):
    age = current_year - year
    return 0.5 ** (age / half_life)

df['sample_weight'] = df['year'].apply(time_weight)
# 2013 weight = 0.06
# 2018 weight = 0.18
# 2022 weight = 0.66
# 2023 weight = 0.87

lgb_weighted = lgb.LGBMRegressor(...)
lgb_weighted.fit(X_train, y_train, sample_weight=df_train['sample_weight'])
# Year-CV MAPE: 5.9% (improvement from 6.4%)
```

**Expert insight:** climate-change-aware time decay is a simple but powerful tool. Crop varieties improve, planting dates shift, weather patterns drift — old data actively misleads if treated equally with recent data.

---

## Step 9: Quantile Models for Insurance Pricing

```python
# Insurance products price losses against P50 baseline yield
# A "loss event" is yield below 75% of trend
# Insurance underwriters need P10 yield forecast -- the worst-case scenario

lgb_p10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10, ...)
lgb_p50 = lgb.LGBMRegressor(objective='quantile', alpha=0.50, ...)
lgb_p90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90, ...)

# For each field on Aug 1:
field_predictions = {
    'p10': lgb_p10.predict(X_field),  # downside scenario
    'p50': lgb_p50.predict(X_field),  # central forecast
    'p90': lgb_p90.predict(X_field),  # upside
}

# Insurance underwriting: if P10 < 0.75 * trend → flag for premium adjustment
# Conformal calibration on the prediction interval ensures empirical coverage matches
```

---

## Step 10: Field-Level vs Regional Aggregates

```python
# Per-field MAPE: 5.9%
# Region aggregate (district-level) MAPE: 2.4%
# Why aggregate is better: per-field errors partially cancel out at the regional level

# Aggregation strategy:
def regional_forecast(predictions, field_acres):
    return (predictions * field_acres).sum() / field_acres.sum()

# Per-region predictions feed:
#   - Storage capacity planning (do we need additional silos?)
#   - Futures hedging (long/short positions)
#   - Transportation logistics
```

---

## Step 11: Final Evaluation on 2024 Test Set

```python
# Final model: LightGBM with time-weighted training + climate-aware detrending
# Test set: 4,200 fields, 2024 harvest year
#
# Performance:
#   Per-field MAPE:                5.7%   ✓ (target 8%)
#   Per-county aggregate MAPE:     3.1%
#   Per-state aggregate MAPE:      2.2%
#   Bias (mean residual):          +0.4 bushels/acre (slight under-forecast on best year)
#
# vs USDA NASS August forecast:
#   USDA forecast bias:           -2.1% (over-prediction in 2024)
#   USDA MAPE:                    7.3%
#   Our advantage:               -1.6pp on bias, +1.6pp on MAPE  (beats by ~3pp)  ✓
#
# Coverage:
#   P10-P90 interval coverage:    0.852 (slightly under target 0.80; conformal helps)
#   After conformal calibration:   0.916 ✓
```

---

## Step 12: Per-Field Explainability

```python
# Insurance underwriters require explainability for premium adjustments
# SHAP for each high-risk field

# Example output:
# Field #4821 (county Henderson, IL)
# Predicted yield: 142 bu/ac (P50); P10: 102; P90: 178
# Compared to county average 198 bu/ac, P10 102 < 75% of trend → INSURABLE LOSS RISK
#
# Top SHAP drivers:
#   1. drought_severity_jul = 3 (extreme)         (-31 bu/ac)
#   2. ndvi_jul_15 = 0.42 (well below avg 0.62)   (-22 bu/ac)
#   3. heat_stress_pollination = 8 days           (-14 bu/ac)
#   4. precip_total_may_jul = 4.2 in              (-11 bu/ac)  [vs avg 9 in]
#   5. soil_drainage = poor                        (+4 bu/ac)  [poor drainage helped in dry year]
#
# Underwriter summary:
# "Drought stress is the primary risk factor. Mid-July satellite imagery confirms low canopy
#  health. Pollination occurred during 8 days of heat stress. Field is at material loss risk;
#  recommend premium adjustment."
```

---

## Step 13: Deployment

```python
# Annual cycle:
#   January: review performance from prior year, retrain with full new data
#   May 1 (planting time): preliminary "scenario" forecasts based on planting decisions
#   July 15: mid-season forecast (NDVI is in)
#   August 1: BUSINESS-CRITICAL forecast for stakeholders
#   August 15: refined forecast as additional weather data arrives
#   October: harvest data starts to arrive; track forecast accuracy in real time

# Annual retraining (in winter with new harvest data):
#   - Append 2024 harvest data
#   - Retrain LightGBM with updated time-weighting
#   - Validate against 2024 holdout (which becomes part of training next year)
#   - Update conformal intervals on most recent year's calibration data

# Monitoring:
#   - Track in-season forecast accuracy from June 15 onward
#   - Compare against USDA NASS forecasts when they're released
#   - Alert if region-level MAPE drifts > 5% (indicates structural change)

# Special handling:
#   - Catastrophic events (early frost, hail): flag affected fields, manual override
#   - New seed varieties: include as feature; cold-start for new variety = use category-typical
#   - Field replanting: flag as anomaly (not in normal training distribution)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Climate change | Train on all years equally | Time-decay weights; detrend yield by yearly mean |
| Train/test split | Random | Year-based, leave-one-year-out CV |
| Weather features | Aggregated monthly | Daily features + biologically-meaningful aggregations (GDD, heat stress at pollination) |
| Soil features | Use as is | Imputation by county median for missing |
| Field history | Ignore | Field's own 3-year detrended average is a top-5 feature |
| Output | Single yield estimate | P10/P50/P90 distribution for insurance pricing |
| Coverage | "I trained at q=0.10 so it must be 10%" | Conformal prediction for rigorous coverage |
| Aggregate forecast | Average per-field predictions | Acres-weighted regional aggregate (matches business decisions) |
| Insurance use case | Same as overall | P10 below trend triggers underwriting review |
| Stakeholder integration | Standalone forecast | Compare against USDA NASS; track delta |
| Climate awareness | None | Detrend, time-weight, monitor for distribution shifts |
| Explainability | "model said X" | SHAP per-field tied to underwriting decision; biological reasoning |
