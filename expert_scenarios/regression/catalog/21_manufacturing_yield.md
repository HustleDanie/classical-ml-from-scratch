# Expert Scenario 21: Manufacturing Yield Prediction

> **Complexity:** Bounded target [0, 1] (yield is a percentage), sensor multicollinearity (200+ correlated readings), drift over weeks (calibration drift), catastrophic regime changes (machine swap), the model directly impacts production scheduling and supply commitments.

---

## The Brief

A semiconductor wafer fab gives you 18 months of production-run data across 3 lithography lines and 12 process steps. Each "lot" of 25 wafers gets a yield score: the fraction of dies on the wafers that pass final electrical test. They want a model predicting yield BEFORE the lot completes the line, so:

- Production planning can adjust commitments to customers (don't promise volume that won't ship).
- Process engineers can intervene mid-line on lots predicted to fail.
- Procurement can adjust raw-material orders.

Constraints:

- Yield ranges 0.40 to 0.99 (40% to 99% pass rate).
- Predictions needed at the 8-hour mark of a 12-hour line (4 hours of lead time).
- MAPE within 3 percentage points (yield ± 3pp = good).
- Per-line drift handling: each line has its own personality and drifts week to week.
- Sensor data is rich but noisy (200+ channels, many highly correlated).

This is harder than typical regression because: the bounded target requires special treatment (logit-transform); sensor calibration drifts continuously; new "recipes" (process variants) appear monthly; and the cost of a wrong prediction is asymmetric (over-promising customers ≠ under-promising).

---

## Step 1: Define the Problem Type

```
Type:           Regression on bounded target [0, 1]
Primary Metric: MAPE in percentage points (mean absolute % error)
Secondary:      RMSE; Per-line MAPE; Calibration check (predicted vs actual yield distribution)
Business Goal:  MAPE ≤ 3pp; predicted yield within ±5pp of actual for 90% of lots
Constraint:     8-hour-mark prediction (lead time 4 hours);
                Monthly recipe additions; weekly per-line drift
Target shape:   Left-skewed (good runs cluster near 0.95-0.99, tail extends down to 0.40)
```

**Expert thinking:** raw regression on yield is brittle because the model can predict outside [0, 1]. The standard fix: transform target via logit `log(y / (1-y))`, train, and inverse-transform predictions.

---

## Step 2: Understand the Data

```
Shape: ~24,000 lots × 250+ features
Target: yield (float, 0.40 - 0.99)

Per-lot features (collected over 8-hour mark):

PROCESS METADATA:
- lot_id (unique)
- recipe_id (50+ unique recipes; some have only 30-100 historical lots)
- product_design (8 categories: A, B, C... different chips on different lines)
- start_timestamp
- production_line (3 lines: L1, L2, L3)
- shift (Day / Swing / Night)
- target_volume (wafers per lot, usually 25)

PROCESS STEP METRICS (per step, 12 steps):
- step_X_completion_time_min
- step_X_temperature_avg
- step_X_temperature_max
- step_X_temperature_std
- step_X_pressure_avg
- step_X_pressure_max
- step_X_chemical_flow_avg
- step_X_chemical_purity_avg
- step_X_dose_uniformity_score
- step_X_alignment_offset_microns
- ...
- (12 steps × ~15 metrics each = 180 step features)

WAFER MEASUREMENT (mid-line):
- pre_clean_uniformity (variance across wafer)
- thickness_avg
- thickness_std
- defect_count_8hr_mark (ML-detected from inspection)
- defect_density_per_cm2
- contamination_score
- ...
- ~30 wafer-level features

LINE / RECIPE HISTORICAL:
- line_avg_yield_30d
- recipe_avg_yield_30d
- line_avg_yield_90d
- shift_recent_yield
- machine_serial_for_critical_step (different physical equipment in same line)
- days_since_line_calibration

MAINTENANCE / DRIFT:
- days_since_last_PM (preventive maintenance)
- last_PM_was_corrective (binary; corrective PM after a failure)
- recipe_age_months (months since recipe was first used)
- recent_recipe_yield_trend (delta in last 30 days vs prior)
```

**Expert thinking:** 200+ features is a lot. Many sensors are highly correlated (e.g., temperature_avg and temperature_max usually move together). The model needs feature selection or strong regularization (Lasso, ElasticNet) to avoid multicollinearity issues and to surface which sensors actually drive yield.

---

## Step 3: EDA

```python
df['yield'].describe()
# count    24,000
# mean      0.881
# std       0.072
# min       0.412
# 25%       0.866
# 50%       0.901
# 75%       0.924
# max       0.987
# Skewness: -1.8 (LEFT-skewed)

# Yield by line over time
df.groupby([df['start_timestamp'].dt.year_month, 'production_line'])['yield'].mean().plot()
# L1: average 0.91, occasional dips
# L2: average 0.86, more variable
# L3: average 0.93 (newest line, best yields)
# All three show drift cycles tied to PM intervals

# Per-recipe yield variance
recipe_yield = df.groupby('recipe_id')['yield'].agg(['count', 'mean', 'std'])
# Some recipes are stable (mean 0.95, std 0.012); others are volatile (mean 0.78, std 0.09)

# Defect count vs yield
df.groupby(pd.cut(df['defect_count_8hr_mark'], 10))['yield'].mean().plot()
# Strong negative correlation: defect count > 8 -> yield drops 5-10pp
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Yield bounded [0,1], left-skewed | Use logit transform for training |
| Days_since_PM > 28: yield drops 1-2pp on average | PM cycle is a known driver |
| Recipe_avg_yield_30d is the strongest single feature (r=0.71 with target) | Historical performance is dominant; pure feature engineering matters less |
| Defect_count_8hr_mark correlates -0.63 with yield | Mid-line defect count is a leading indicator |
| Some sensor pairs correlate r > 0.95 | Multicollinearity; ElasticNet or PCA |

---

## Step 4: Data Cleaning

```python
# === LOGIT TRANSFORM TARGET ===
# Transform [0, 1] yield to (-inf, +inf) for unbounded training
df['yield_logit'] = np.log(df['yield'] / (1 - df['yield']))
# At inference: inverse-transform predictions
# yield_predicted = sigmoid(prediction_logit)

# Check transformed distribution
df['yield_logit'].describe()
# Mean 1.84, std 0.62, skew now ~-0.4 (much more Gaussian)

# === HANDLE OUTLIERS / BAD LOTS ===
# Some lots had partial defects unrelated to model factors (e.g., an emergency PM mid-lot)
# These are noise in training; flag and drop
known_anomalies = df['lot_id'].isin(emergency_pm_lots)
df = df[~known_anomalies]  # drop ~80 lots

# === HANDLE NEW RECIPES ===
# Some recipes have < 30 historical lots
# Two options:
#   1. Drop recipes with < 30 lots from training (loses signal but cleaner)
#   2. Keep all; rely on global features (line_avg_yield_30d, etc.)
# Choice: keep all but flag is_new_recipe (recipes with < 30 lots)

# === HANDLE MISSING SENSOR READINGS ===
# Sensors fail occasionally; ~3-5% of step metrics missing per lot
df = df.fillna(df.groupby(['production_line', 'recipe_id']).transform('median'))
df = df.fillna(df.median())  # fallback

# === FEATURE-SPECIFIC OUTLIER HANDLING ===
# Don't cap defect_count_8hr_mark — extreme defects are real signal
# But cap obvious sensor errors (e.g., pressure 1000kPa when normal range is 50-200)
for sensor in pressure_sensors:
    p99 = df[sensor].quantile(0.99)
    df[sensor] = df[sensor].clip(upper=p99 * 1.5)
```

---

## Step 5: Feature Engineering

```python
# === COMPRESS HIGHLY CORRELATED SENSOR PAIRS ===
# Use PCA on within-step sensor groups (e.g., temperature_avg, max, std all from step 3)
# Reduces 180 step features to ~60 PCA components retaining 95% variance

from sklearn.decomposition import PCA
sensor_groups = {
    'temperature': temperature_columns,
    'pressure':    pressure_columns,
    'chemical':    chemical_columns,
    'alignment':   alignment_columns
}

pca_features = {}
for group_name, cols in sensor_groups.items():
    pca = PCA(n_components=0.95)  # retain 95% variance
    pca_features[group_name] = pca.fit_transform(df[cols])
    print(f"{group_name}: {len(cols)} -> {pca.n_components_} components")
# temperature: 24 -> 4 components
# pressure:    18 -> 3 components
# chemical:    36 -> 8 components

# === DERIVED COMPOSITES ===
df['process_stress'] = (
    (df['step_3_temperature_max'] > df['step_3_temperature_avg'].quantile(0.95)).astype(int) +
    (df['step_5_alignment_offset_microns'] > 0.3).astype(int) +
    (df['step_7_chemical_purity_avg'] < 0.99).astype(int) +
    (df['step_9_dose_uniformity_score'] < 0.85).astype(int)
)

df['days_to_next_PM'] = df['scheduled_PM_date'] - df['start_timestamp']
df['days_to_next_PM_normalized'] = df['days_to_next_PM'] / df['typical_PM_interval']

# === HISTORICAL FEATURES ===
df['line_recipe_yield_30d'] = df.groupby(['production_line', 'recipe_id'])['yield'].rolling(30, min_periods=5).mean().shift(1).reset_index(level=[0,1], drop=True)
df['line_recipe_yield_volatility_30d'] = df.groupby(['production_line', 'recipe_id'])['yield'].rolling(30, min_periods=5).std().shift(1).reset_index(level=[0,1], drop=True)

# === SHIFT EFFECTS ===
df['shift_recent_avg_yield'] = df.groupby(['shift', 'production_line'])['yield'].transform(
    lambda x: x.shift(1).rolling(20, min_periods=5).mean()
)

# === NEW-RECIPE FLAG ===
df['is_new_recipe'] = (df.groupby('recipe_id').cumcount() < 30).astype(int)

# Final feature count: ~120 features
```

---

## Step 6: Feature Selection / Multicollinearity Check

```python
# Check for multicollinearity even after PCA
corr_matrix = df[final_features].corr().abs()
high_corr_pairs = np.where((corr_matrix > 0.85) & (corr_matrix < 1.0))
# A few pairs remain (e.g., line_recipe_yield_30d and shift_recent_avg_yield)
# OK because Lasso / ElasticNet will handle them

# Feature importance from initial model:
mi_scores = mutual_info_regression(X_train, y_train_logit, random_state=42)
# Top 15:
#   line_recipe_yield_30d           0.142
#   defect_count_8hr_mark           0.108
#   recipe_avg_yield_30d            0.094
#   process_stress                  0.067
#   days_since_last_PM              0.054
#   step_temperature_pca_2          0.041
#   line_avg_yield_30d              0.038
#   step_chemical_pca_3             0.031
#   step_alignment_offset_max        0.029
#   contamination_score             0.027
#   thickness_uniformity            0.024
#   shift_recent_avg_yield          0.021
#   is_new_recipe                   0.018
```

---

## Step 7: Train/Test Split

```python
# Time-based split + line-stratified
df = df.sort_values('start_timestamp')

# Train: first 14 months (2,000 lots / month × 14 = ~28K lots; we have 24K, so ~16 months)
# Validation: month 15-16
# Test: month 17-18 (most recent)

# Each line is represented in train and test (lines aren't dropped)
# But specific lot timestamps: train < val < test
```

---

## Step 8: Try Multiple Models

```python
# === Model 1: Linear Regression on logit-transformed yield ===
# Test MAPE: 4.8pp

# === Model 2: ElasticNet ===
# Test MAPE: 3.4pp (multicollinearity-aware)

# === Model 3: Random Forest ===
# Test MAPE: 2.8pp

# === Model 4: Gradient Boosting (sklearn) ===
# Test MAPE: 2.5pp

# === Model 5: LightGBM ===
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    max_depth=6,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
# Test MAPE: 2.4pp ✓ (target 3pp)

# === Model 6: LightGBM Tweedie (handles bounded distribution differently) ===
lgb_tweedie = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5,
    ...
)
# Trains on raw [0,1] target, not logit
# Test MAPE: 2.6pp (slightly worse; logit-transformed regression is cleaner here)

# === Model 7: Per-line ensemble ===
# 3 separate models, one per production line
# MAPE: L1 2.1pp, L2 2.7pp, L3 2.2pp; weighted avg 2.3pp
```

**Top performer:** Per-line LightGBM ensemble (MAPE 2.3pp).

---

## Step 9: Hyperparameter Tuning

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('mcs', 5, 50),
        'subsample': trial.suggest_float('sub', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('cbt', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    model = lgb.LGBMRegressor(**params, objective='regression', random_state=42)
    return walk_forward_mape(model, X_train, y_train_logit)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
# Best MAPE: 2.2pp
```

---

## Step 10: Drift Detection

```python
# Yield-prediction models drift over weeks
# Monitor by comparing recent predictions to recent actuals

def compute_drift_metrics(predictions_recent, actuals_recent):
    """Detect bias and variance shifts over rolling 30 days."""
    bias = (predictions_recent - actuals_recent).mean()
    variance = (predictions_recent - actuals_recent).var()
    mape = (np.abs(predictions_recent - actuals_recent) / actuals_recent).mean()
    return {'bias': bias, 'variance': variance, 'mape': mape}

# Alert thresholds:
# Bias >|0.005| (0.5pp systematic shift): retrain trigger
# MAPE > 3pp: retrain immediately
# Per-line MAPE > 4pp: line-specific retrain
```

---

## Step 11: Final Evaluation

```python
# Final model: Per-line LightGBM ensemble + logit-transformed target + tuned hyperparameters
# Test set: 2 months × 3 lines = 4,000 lots
#
# Performance:
#   Overall MAPE:                 2.3pp  ✓ (target 3pp)
#   90% of predictions within ±5pp: 91%   ✓
#   Per-line MAPE: L1: 2.1pp, L2: 2.7pp, L3: 2.2pp
#
# Calibration:
#   Predicted yield distribution very close to actual distribution
#   No systematic over- or under-prediction
#
# Lot-level outcomes:
#   Top 10% predicted-low lots: actual yield mean 0.83 (predicted 0.84)
#   Bottom 10% predicted-high: actual yield mean 0.94 (predicted 0.93)
#
# Inference: ~50ms per lot (well under any budget)
```

---

## Step 12: Per-Lot Explainability

```python
import shap

# For lots predicted with low yield, surface drivers:
# Lot #4821 -- predicted yield 0.78 (significantly below typical 0.91)
# Top SHAP drivers:
#   1. defect_count_8hr_mark = 22         (-0.08 yield contribution)
#   2. step_temperature_pca_2 = -1.4       (-0.04)  [unusual temp pattern in step 7]
#   3. days_since_last_PM = 32             (-0.03)  [PM overdue]
#   4. line_recipe_yield_30d = 0.83        (-0.03)  [line+recipe combo running low]
#   5. shift = Night                       (-0.02)
#
# Process engineer message:
# "Predicted yield 0.78. Major contributors: high defect count at 8h checkpoint,
#  unusual temperature signature in step 7, PM is 32 days overdue.
#  Recommend: pause line, run PM before continuing this lot."
```

---

## Step 13: Deployment

```python
# Inference is per-lot, triggered at 8-hour mark of 12-hour line
# Production:
#   1. Lot reaches step 8 (8-hour mark)
#   2. Pipeline collects all sensor data, mid-line measurements, historical features
#   3. Per-line LightGBM predicts yield
#   4. Production planning system receives prediction with confidence interval
#   5. If predicted < 0.85 (vs target 0.91): alert process engineer

# Retraining: WEEKLY
#   - Add last week's completed lots
#   - Retrain per-line models on rolling 6-month window
#   - A/B comparison: new model MAPE on this week's holdout vs incumbent

# Per-line drift handling:
#   - Compute rolling MAPE per line per week
#   - If line MAPE drifts > 1pp: retrain ONLY that line's model
#   - If multiple lines drift simultaneously: investigate process change

# Monitoring:
#   - Daily MAPE on completed lots
#   - Per-line MAPE trends
#   - Bias detection (systematic over/under prediction)
#   - Recipe coverage: track which recipes have least training data
#   - Alert if recipe yield drops > 2pp from 30-day baseline
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Bounded target | Treat as regression on raw [0,1] | Logit-transform to (-inf, inf) for training |
| Multicollinearity | Use all 180 sensor features | PCA per sensor group; ElasticNet regularization |
| Per-line drift | Single global model | Per-line ensemble; per-line drift monitoring |
| Train/test split | Random | Time-based; train on older, test on newer |
| New recipes | Drop or treat the same | Flag is_new_recipe; rely on historical features |
| Feature engineering | Use raw sensor values | Composites: process_stress, days-to-PM, line-recipe historical |
| Outlier handling | Cap everything | Cap clear sensor errors; keep real outliers (defect counts) |
| Loss function | RMSE on raw yield | RMSE on logit-transformed yield |
| Ensemble | One model | Per-line LightGBM ensemble |
| Calibration | Use raw model output | Verify per-line distribution matches actual |
| Explanation | "model said low yield" | Top SHAP drivers + actionable recommendation |
| Retraining | Quarterly | Weekly with per-line drift triggers |
| Recipe coverage | Ignore | Track training-data per recipe; alert on under-represented |
