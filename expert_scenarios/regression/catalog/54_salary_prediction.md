# Expert Scenario 54: Salary Prediction (Pay Equity-Aware)

> **Complexity:** Mixed-type tabular (numeric + categorical), heavy-tailed target ($30K to $1M+), pay-equity legal requirements, historical bias in training data, fairness audit across protected groups, cannot use protected attributes as features but proxies leak.

---

## The Brief

A multinational company gives you 8 years of HR records (~180,000 employees) with detailed compensation data. They want a model that predicts the "fair market salary" for an employee given their role, experience, location, and performance. The output is used for:

- **Pay-equity audits**: identify employees significantly under-paid relative to predicted fair value (pay-gap diagnostic).
- **New-hire compensation**: market-rate prediction for offer letters.
- **Promotion budget planning**: aggregate predicted salaries for headcount planning.

Constraints:

- MAPE in raw dollars within 8% on the held-out test set.
- Cannot use protected attributes (race, gender, age, national origin) as features under EEOC.
- Must audit for residual bias: are protected groups still under-paid AFTER controlling for legitimate factors?
- Must be explainable to HR business partners ("why is this offer this number?").
- Must handle compensation regime differences across regions (US, EU, APAC, LATAM).

This is harder than typical regression because: historical data ENCODES historical pay bias (training on biased data perpetuates the bias); proxy features (zip code, school, manager hierarchy) leak protected attribute information; and the deliverable is not just a number but a fairness diagnostic.

---

## Step 1: Define the Problem Type

```
Type:           Regression on annual salary, heavy-tailed target
Primary Metric: MAPE in raw dollars; symmetric MAPE on log(salary)
Secondary:      Per-region MAPE; per-role MAPE
                Residual bias by protected group (post-prediction analysis)
Business Goal:  MAPE ≤ 8%; residual pay gap by gender/race ≤ 2% after controlling for
                role/location/tenure
Constraint:     EEOC-compliant (protected attributes in audit but NOT features)
                Explainable per-employee
Target shape:   Heavy right tail; log-transform target
```

**Expert thinking:** the model serves two purposes that pull in opposite directions:
1. Prediction accuracy: best fit to historical salaries → reproduces historical bias.
2. Fairness diagnostic: identify under-paid employees → must NOT be biased by protected attributes.

Solution: train on historical data WITHOUT protected attributes (so model bases predictions on legitimate factors), then use protected attributes only in the audit phase to detect residual bias.

---

## Step 2: Understand the Data

```
Shape: 180,000 employees × ~50 features (after exclusions)
Target: annual_total_compensation (base + bonus + equity vesting; $35K-$1.2M)

Available features (categorized for legal review):

LEGITIMATE PREDICTORS (allowed in model):
- role_id (250+ unique roles)
- role_level (1-15 in level hierarchy)
- function (Engineering / Sales / Marketing / Operations / Finance / Other)
- years_in_role (0-25)
- years_with_company (0-30)
- total_years_experience
- highest_education (HS / Bachelor / Master / PhD / Professional)
- relevant_certifications_count
- city_id (200+ cities)
- country (45 countries)
- region (US / EU / APAC / LATAM)
- prior_company_seniority (Senior / Mid / Junior at last employer; 35% missing)
- num_direct_reports (0-50)
- has_management_role (binary)
- manager_level (numeric -- skip-level rank)
- recent_performance_rating (5-pt scale; 8% missing)
- recent_promotion_year (last 24 months)

PROXY-RISK FEATURES (use with caution; could leak protected attributes):
- zip3 (HIPAA-safe but correlates with race/income demographics)
- university_attended (correlates with race / class)
- referrer_type (Internal vs Referred vs External vs Recruiter)

PROTECTED ATTRIBUTES (NOT features; AUDIT ONLY):
- gender (M / F / Non-binary / Unknown)
- race / ethnicity (W / B / H / A / Multi / Decline)
- age (regulated under ADEA)
- national_origin
- disability_status (binary)
- veteran_status

SALARY HISTORY:
- starting_salary_at_hire
- salary_change_frequency_avg_24mo
- last_year_total_compensation
- ytd_compensation
```

**Expert thinking:** the proxy-risk features are the trickiest. zip3 correlates with race (residential segregation); university_attended correlates with race + class. They contain LEGITIMATE signal (cost of living, talent pool) but also encode bias. Standard practice: include them, then audit residual bias and remove if disparate impact emerges.

---

## Step 3: EDA

```python
df['annual_total_compensation'].describe()
# count    180,000
# mean    $147,000
# std     $112,000
# min      $35,000
# 25%      $78,000
# 50%     $115,000
# 75%     $172,000
# 95%     $342,000
# 99%     $612,000
# max   $1,180,000
# Skewness: 4.1 (heavy right-tail; tech execs, sales superstars)

# After log-transform
df['log_compensation'] = np.log1p(df['annual_total_compensation'])
df['log_compensation'].skew()  # 0.18 (much more normal)

# Compensation by region
df.groupby('region')['annual_total_compensation'].median()
# US:    $165,000
# EU:    $108,000
# APAC:  $89,000
# LATAM: $58,000
# Regional differences are LEGITIMATE (cost of living, market rates)

# Pay gap analysis (raw, controlling for nothing)
df.groupby(['gender', 'race']).agg(
    median_comp=('annual_total_compensation', 'median'),
    n=('employee_id', 'count')
)
# Notable: average disparity by gender ~12pp; by race ~8pp
# This is the BASELINE before any controls
# Goal: AFTER controlling for legitimate factors, residual gap should be < 2pp
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Heavy right tail in raw dollars | Use log(salary) as training target; report MAPE in dollars |
| Median raw pay gap ~12% by gender, ~8% by race | The "before adjustments" reality; we need to control for legitimate factors |
| Region differences large but legitimate | Include region in model; differential by region is fine |
| University correlates with starting comp | Proxy risk; audit needed |

---

## Step 4: Data Cleaning

```python
# === LOG-TRANSFORM TARGET ===
df['log_comp'] = np.log1p(df['annual_total_compensation'])
# Train on log_comp; inverse transform predictions

# === IDENTIFY AND HANDLE BIAS-LADEN HISTORICAL ROWS ===
# Some past compensation decisions were biased (historical reality)
# Option 1: include all data, audit residual bias post-hoc
# Option 2: weight RECENT data more (recent years had stronger pay-equity initiatives)
df['record_year'] = df['record_date'].dt.year
df['time_weight'] = 0.5 ** ((2024 - df['record_year']) / 5)  # half-life 5 years

# === HANDLE MISSING DATA ===
df['recent_performance_rating'] = df['recent_performance_rating'].fillna(3)  # neutral default
df['prior_company_seniority'] = df['prior_company_seniority'].fillna('Unknown')
df['has_recent_promotion'] = df['recent_promotion_year'].notna().astype(int)

# === REMOVE PROTECTED ATTRIBUTES BEFORE TRAINING ===
features_for_training = [...]  # legitimate predictors only
audit_attributes = ['gender', 'race', 'ethnicity', 'age', 'national_origin']
df_audit = df[audit_attributes + ['employee_id']].copy()  # save for audit phase
df_train_features = df[features_for_training + ['log_comp']]
```

**Expert insight:** explicitly separating "training features" from "audit attributes" prevents accidental leakage. A common mistake: a feature like "name" that subtly encodes gender. Audit your feature list against a checklist of protected attributes and known proxies.

---

## Step 5: Feature Engineering

```python
# === ROLE-LEVEL COMPENSATION ANCHORS (mean, median, percentile) ===
role_stats = df_train.groupby('role_id')['log_comp'].agg(['mean', 'median', 'std'])
df['role_level_mean_log_comp'] = df['role_id'].map(role_stats['mean'])
df['role_level_std_log_comp'] = df['role_id'].map(role_stats['std'])

# === REGIONAL COST-OF-LIVING ADJUSTMENT (CoL-adjusted comp) ===
city_col_index = {city_id: col_index for ...}  # external data: cost-of-living per city
df['city_col_index'] = df['city_id'].map(city_col_index)
df['col_adjusted_comp_log'] = df['log_comp'] - np.log(df['city_col_index'])

# === EXPERIENCE-SCALED FEATURES ===
df['experience_to_role_level_ratio'] = df['total_years_experience'] / df['role_level']
df['years_in_role_z_score'] = df.groupby('role_id')['years_in_role'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# === EDUCATION × ROLE INTERACTION ===
df['has_advanced_degree_for_high_role'] = (
    (df['highest_education'].isin(['Master', 'PhD', 'Professional'])) &
    (df['role_level'] >= 8)
).astype(int)

# === PROMOTION VELOCITY ===
df['promotion_velocity'] = df['has_recent_promotion'] / (df['years_in_role'] + 1)
df['fast_track'] = (df['promotion_velocity'] > 0.4).astype(int)

# === MANAGEMENT IMPACT ===
df['log_direct_reports'] = np.log1p(df['num_direct_reports'])

# Final feature count: ~55 features (excluding protected attributes)
```

---

## Step 6: Feature Selection

```python
# Mutual information ranking on log_comp target
mi_scores = mutual_info_regression(X_train, y_train_log, random_state=42)

# Top 15:
#   role_level_mean_log_comp        0.281
#   role_id (target-encoded)        0.241
#   role_level                      0.218
#   region                          0.094
#   total_years_experience          0.087
#   has_management_role             0.062
#   recent_performance_rating       0.054
#   highest_education               0.041
#   country                         0.038
#   experience_to_role_level_ratio  0.034
#   city_col_index                  0.029
#   prior_company_seniority         0.026
#   relevant_certifications_count   0.021
#   num_direct_reports              0.018
```

---

## Step 7: Try Multiple Models

```python
# === Model 1: Linear Regression on log target ===
# Test MAPE on dollars: 12.4%

# === Model 2: ElasticNet ===
# Test MAPE: 9.8%

# === Model 3: Random Forest ===
# Test MAPE: 8.2%

# === Model 4: Gradient Boosting ===
# Test MAPE: 7.6%

# === Model 5: LightGBM ===
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=500,
    max_depth=6,
    num_leaves=63,
    learning_rate=0.05,
    sample_weight=df['time_weight'],  # weight recent data more
    random_state=42
)
# Test MAPE: 6.9%  ✓ (target 8%)
```

**Top performer:** LightGBM with time-weighted training (MAPE 6.9%).

---

## Step 8: Hyperparameter Tuning

```python
import optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'min_child_samples': trial.suggest_int('mcs', 5, 100),
        'subsample': trial.suggest_float('sub', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('cbt', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    model = lgb.LGBMRegressor(**params, objective='regression', random_state=42)
    return cv_mape_dollars(model, X_train, y_train_log)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=80)
# Best MAPE: 6.4%
```

---

## Step 9: THE BIAS AUDIT

```python
# Apply the model to predict "fair market salary" for every employee
df['predicted_log_comp'] = best_model.predict(X)
df['predicted_comp'] = np.expm1(df['predicted_log_comp'])

# Compute residual: actual - predicted
df['comp_residual_pct'] = (df['annual_total_compensation'] - df['predicted_comp']) / df['predicted_comp']

# AUDIT: is residual systematically different by protected group?

audit = df_audit.merge(df[['employee_id', 'comp_residual_pct']], on='employee_id')

print("Residual pay gap by gender (controlling for role/location/tenure):")
audit.groupby('gender')['comp_residual_pct'].describe()
# M:        mean +0.4%, n=110K (men paid 0.4% above predicted fair value)
# F:        mean -1.8%, n=64K  (women paid 1.8% below predicted fair value)  <-- bias
# Non-bin:  mean +0.1%, n=2K
# Decline:  mean -0.6%, n=4K

# Gender gap: 2.2pp (target was < 2pp -- slight miss)

print("Residual pay gap by race:")
audit.groupby('race')['comp_residual_pct'].describe()
# W:  mean +0.6%
# B:  mean -1.4%   <-- 2.0pp gap from W
# H:  mean -0.9%   <-- 1.5pp gap from W
# A:  mean +0.2%
# Multi:  mean -0.3%
```

**Findings:**
- Women earn ~2.2% less than predicted fair value (bias signal)
- Black employees earn ~2.0% less; Hispanic ~1.5% less
- These gaps are RESIDUAL (after controlling for role/level/location/tenure)
- This is the bias the model surfaces; HR uses it to identify under-paid individuals

**Expert insight:** the model is a fairness diagnostic, not a corrective. It surfaces individual under-paid employees, who then get reviewed by HR (often resulting in pay raises). The aggregate residual gap is a corporate metric to track over time.

---

## Step 10: Identifying Under-Paid Employees (Use Case)

```python
# Flag employees in bottom 10% of residual (under-paid relative to predictions)
df['under_paid_flag'] = df['comp_residual_pct'] < df['comp_residual_pct'].quantile(0.10)

# Per-protected-group rate of under-pay
print("Under-paid rate by gender:")
audit.merge(df[['employee_id', 'under_paid_flag']]).groupby('gender')['under_paid_flag'].mean()
# M:  9.4%
# F:  12.8%   <-- women over-represented in under-paid bucket
# Non-bin: 10.1%

# This is the "Disparate Impact" rate
# DI ratio = under-paid rate F / under-paid rate M = 1.36
# > 1.0 indicates disproportionate impact
```

---

## Step 11: Final Evaluation

```python
# Final model: LightGBM with time-weighted training, no protected attributes in features
# Test set: 18,000 employees
#
# Performance:
#   MAPE on dollars:                 6.9%   ✓ (target 8%)
#   Per-region MAPE:
#     US:    6.4%
#     EU:    7.1%
#     APAC:  7.8%
#     LATAM: 8.4%  (less data; higher variance)
#
# Residual bias (post-prediction audit):
#   Gender gap: -2.2% (women under-predicted by 2.2pp on average)
#   Race gap:   -1.7% (Black/Hispanic under-predicted by ~1.7pp)
#   Note: this means the historical data showed bigger raw gaps; controlling for role/location
#   removes most but not all
#
# Recommendations to HR:
#   1. ~720 individual employees flagged as significantly under-paid (residual < -10%)
#   2. Investigate proxy features: are zip3 or university driving residual gaps?
#   3. Track residual gap over time; goal is < 1% within 3 years
#   4. Apply additional adjustment factor when computing "target salary" for offers
```

---

## Step 12: Per-Employee Explainability

```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Employee #4821 -- predicted $145K; actual $128K (residual -12%)
# Top SHAP drivers:
#   1. role_level = 6                      (+$22K)
#   2. region = US                          (+$18K)
#   3. years_in_role = 4                    (+$5K)
#   4. has_management_role = 1              (+$8K)
#   5. recent_performance_rating = 4        (+$6K)
#
# Total predicted: $145K (sum of contributions + base)
# Actual: $128K
# Gap of $17K; flagged for HR review
#
# HR can use this to:
#   - Investigate why this employee earns below predicted
#   - Compare to peers in same role/region (likely under-paid)
#   - Take action: raise, market adjustment, or retention discussion
```

---

## Step 13: Deployment

```python
# Production:
#   1. Quarterly batch run on full HR roster
#   2. Generate predicted "fair value" salary for every employee
#   3. Compute residual; flag bottom-decile under-paid
#   4. Bias audit by protected group (every quarter)
#   5. Surface flagged employees to HR business partners

# Annual recalibration:
#   - Retrain on most recent year's data
#   - Re-audit residual bias
#   - Track trend in aggregate residual gap

# Monitoring:
#   - Quarterly residual gap by protected group (alert if any group > 3pp residual)
#   - Per-region MAPE drift
#   - "Disparate impact ratio" of under-paid flagging (must stay > 0.80)
#   - Stakeholder feedback: HR partner overrides on flagged employees

# Compliance documentation:
#   - Model card describing features, exclusions, audit results
#   - Quarterly bias report
#   - Documentation of decisions to include/exclude proxy features
#   - External audit pathway (e.g., outside legal review)
```

---

## Step 14: Reporting

```
KEY DELIVERABLES:

1. PREDICTIVE MODEL:
   - LightGBM regression on log(salary)
   - 55 features (no protected attributes)
   - MAPE 6.9% on raw dollars

2. PAY-EQUITY DIAGNOSTIC:
   - Residual gap by gender: -2.2pp (down from raw -12pp; some bias remains)
   - Residual gap by race: -1.7pp (down from raw -8pp)
   - 720 individual employees identified as significantly under-paid
   - Disparate impact ratio: 0.79 for gender; 0.84 for race

3. RECOMMENDATIONS:
   - Pay-equity adjustments for 720 flagged employees (estimated cost $2.4M)
   - Quarterly retraining + audit
   - Investigate proxy features (zip3, university)
   - Track residual gap trend; target < 1pp within 3 years

4. COMPLIANCE:
   - EEOC-friendly: protected attributes not in features
   - Audit-ready: documented exclusion list, residual bias analysis
   - Explainable: top SHAP drivers per prediction
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Protected attributes | Use them as features | Exclude from features; use ONLY in audit phase |
| Proxy attributes (zip3, university) | Naively include | Include but audit for residual bias; investigate if needed |
| Bias-laden history | Train on all years equally | Time-weight recent data; recent has stronger pay-equity practices |
| Target | Predict raw salary | Log-transform target; report MAPE in dollars |
| Audit | Skip | Multi-protected-group residual analysis after prediction |
| Disparate impact | Ignore | Compute Disparate Impact ratio; flag if < 0.80 |
| Per-employee output | "Predicted $147K" | Top SHAP drivers; residual; flag if under-paid |
| Per-region accuracy | Aggregate | Per-region MAPE; regional model variants if needed |
| Cost-of-living | Ignore | Adjust comparisons by city COL index |
| Deliverable | Single number predictions | Predictions + bias diagnostic + flagged employees + DI ratio |
| Compliance | Train and ship | Documented model card; quarterly bias report; external review |
| Stakeholder integration | "Here's predicted comp" | "Here are 720 employees who need review; here's the residual gap trend" |
