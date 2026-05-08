# Expert Scenario 73: Customer Time-to-Churn (Survival Regression)

> **Complexity:** Right-censoring (most customers haven't churned yet at training time), Cox proportional hazards or Random Survival Forest, concordance index instead of MAE, time-varying covariates, downstream interventions need expected lifetime.

---

## The Brief

A SaaS subscription company gives you 800K customer histories spanning 5 years. They want to know, for every active customer, "how many more months until they churn?" — feeding:

- The retention team's intervention queue (high-risk, near-churn customers get outreach).
- The CFO's revenue forecasts (multiply expected remaining lifetime × MRR).
- Marketing budget allocation (don't pay $200 to acquire a customer whose remaining lifetime is 2 months).

The catch: **most customers haven't churned yet**. They're "right-censored" — we know they were active for at least N months, but their actual churn time is unknown. Treating censored customers as "they'll never churn" or "they'll churn at the censoring time" both bias the model. Survival regression handles this properly.

---

## Step 1: Define the Problem Type

```
Type:           Survival regression (right-censored time-to-event)
Primary Metric: Concordance index (C-index) -- analog of AUC for survival
Secondary:      Brier score for survival; integrated Brier score
                MAE on observed (non-censored) churn events
Business Goal:  C-index > 0.75; calibrated survival curves at 6/12/24 months
Constraint:     Quarterly retraining; explainable per-customer
                The expected remaining lifetime feeds revenue forecasts -- calibration matters
Censoring:      ~70% of training customers right-censored (still active at observation cut)
```

**Expert thinking:** standard regression on "time to churn" assumes you observe the event for everyone. With 70% censoring, that assumption is brutally wrong. Cox PH (proportional hazards) or Random Survival Forest natively handle censoring. The metric isn't MAE on a single number — it's how well the model ranks risk over time.

---

## Step 2: Understand the Data

```
Shape: 800,000 customer records
Outcome columns:
- months_active   (int, 1-60, capped at 60 for the 5-year observation window)
- churned         (binary, 1 if churned, 0 if right-censored)

Distribution:
- 30% churned during observation
- 70% right-censored (active at end of observation period)

Features (per customer, snapshot at acquisition or first 30 days):

ACQUISITION (5):
- customer_id (unique)
- acquisition_date
- acquisition_channel (Referral/Paid_Search/Social/Organic/Direct)
- acquisition_cohort (acquisition month bucket)
- acquired_via_promo (binary)

PLAN / SUBSCRIPTION (8):
- subscription_tier (Free/Basic/Premium/Enterprise)
- billing_frequency (Monthly/Annual)
- mrr_dollars (monthly recurring revenue)
- contract_length_months (1, 12, 24)
- num_seats (1-500)
- has_annual_discount (binary)
- promo_discount_pct (float, 0-30%)
- payment_method (CreditCard/ACH/Invoice)

USAGE FIRST 30 DAYS (15):
- num_logins_30d (int)
- avg_session_duration_min_30d (float)
- num_unique_features_used_30d (int, 0-50+)
- core_feature_adoption_pct_30d (float)
- num_team_members_invited_30d (int)
- has_used_integration_30d (binary)
- num_support_tickets_30d (int)
- avg_ticket_resolution_hours_30d (float)
- num_documentation_views_30d (int)
- mobile_pct_30d (float)
- weekend_active_30d (binary)
- has_imported_data_30d (binary)
- has_exported_data_30d (binary)
- num_workflows_created_30d (int)
- has_admin_role_assigned_30d (binary)

DEMOGRAPHICS / FIRMOGRAPHICS (8):
- company_size_employees (Small/Medium/Large/Enterprise)
- industry (12 categories)
- country (50+)
- timezone_offset
- account_role_at_signup (User/Manager/Admin)
- has_other_saas_tools_in_stack (binary, from third-party data)
- has_active_competitor (binary)

LIFECYCLE (3):
- onboarding_completed (binary)
- onboarding_completed_within_7d (binary)
- has_csm_assigned (binary, customer success manager)
```

**Expert thinking:** the 70% censoring rate means most of our training data is "still active at observation cut." We can't just train regression on `months_active` — we'd be training the model that *every customer churned at exactly their last observed month*. That's catastrophically wrong.

---

## Step 3: EDA

```python
df['churned'].value_counts(normalize=True)
# 0    0.70
# 1    0.30

# Months active distribution
df.loc[df['churned']==1, 'months_active'].describe()
# count  240,000
# mean    14.2 (months until churn for those who churned)
# 25%      3
# 50%      9
# 75%     22
# max     60

# For active customers, months_active is censoring time, not churn time
df.loc[df['churned']==0, 'months_active'].describe()
# count  560,000
# mean    19.4 (months observed, still active)
# 25%      6
# 50%     16
# 75%     34
# max     60
```

**Findings:**

| Finding | Implication |
|---------|------------|
| 30% churn rate | Significant churn but not extreme; survival modeling still appropriate |
| Churn happens fastest in months 1-6 (early churn dominant) | Model needs to capture rapid early-period decay |
| 75% of churners gone by month 22 | Long-term retention exists; "lifetime" is bounded |
| `core_feature_adoption_pct_30d > 0.5` → 0.18 churn rate vs 0.42 without | Strong signal: early adoption matters |
| `has_csm_assigned` → 0.13 churn rate vs 0.34 without | CSM assignment correlates strongly (modifiable) |
| `subscription_tier=Free` → 0.61 churn rate vs Premium 0.18 | Tier matters; free users churn fastest |
| Acquisition channel: Paid_Search → 0.41 churn vs Referral 0.18 | Channel quality predicts retention |

---

## Step 4: Data Cleaning

```python
# Already minimal cleanup needed for SaaS data (mostly clean)

# Format for survival models:
# duration: months_active (time until event OR censoring)
# event: churned (1 if event observed, 0 if censored)

# Drop customers with months_active=0 (immediate refund -- different process)
df = df[df['months_active'] > 0]

# Verify no impossible combinations
assert df.loc[df['churned']==1, 'months_active'].min() >= 1

# Missing handling: minimal in this dataset
df['onboarding_completed'] = df['onboarding_completed'].fillna(0).astype(int)
```

---

## Step 5: Feature Engineering

```python
# === USAGE INTENSITY COMPOSITES ===
df['login_intensity'] = df['num_logins_30d'] / 30  # logins per day
df['adoption_breadth'] = df['num_unique_features_used_30d'] / 50  # normalize

# === ENGAGEMENT QUALITY ===
df['power_user_score'] = (
    (df['num_team_members_invited_30d'] >= 1).astype(int) +
    df['has_used_integration_30d'] +
    df['has_imported_data_30d'] +
    df['has_admin_role_assigned_30d'] +
    (df['num_workflows_created_30d'] >= 3).astype(int)
)

# === SUPPORT FRICTION ===
df['high_support_load'] = (df['num_support_tickets_30d'] >= 3).astype(int)
df['slow_resolution'] = (df['avg_ticket_resolution_hours_30d'] > 24).astype(int)

# === PRICING / CONTRACT ===
df['log_mrr'] = np.log1p(df['mrr_dollars'])
df['has_annual_contract'] = (df['contract_length_months'] >= 12).astype(int)
df['mid_size_account'] = (df['num_seats'].between(10, 100)).astype(int)
```

---

## Step 6: Train Cox Proportional Hazards

```python
from lifelines import CoxPHFitter

# Format for lifelines:
# DataFrame with 'duration' and 'event' columns + features

survival_df = df.copy()
survival_df['duration'] = survival_df['months_active']
survival_df['event'] = survival_df['churned']

# Drop original temporal vars to avoid leakage
features_for_cox = [...]  # ~35 features after engineering
survival_df = survival_df[features_for_cox + ['duration', 'event']]

cph = CoxPHFitter(penalizer=0.01)  # mild L2 regularization
cph.fit(survival_df, duration_col='duration', event_col='event')

# Print summary
cph.print_summary()
# coef         exp(coef)  HR (95% CI)         p-value
# subscription_tier=Free       1.32  3.74    (3.51, 3.99)  < 0.001
# core_feature_adoption_30d   -0.89  0.41    (0.38, 0.44)  < 0.001
# has_csm_assigned            -0.74  0.48    (0.45, 0.51)  < 0.001
# acquisition_channel=Paid     0.54  1.72    (1.65, 1.79)  < 0.001
# acquired_via_promo           0.38  1.46    (1.40, 1.52)  < 0.001
# ...
```

**Expert insight:** the hazard ratio (`exp(coef)`) is interpretable as multiplicative effects on the churn hazard. `subscription_tier=Free` has HR=3.74 — Free users churn 3.74x faster than the baseline (Enterprise). `core_feature_adoption_30d` (continuous) HR=0.41 means a 1-unit increase in feature adoption multiplies churn hazard by 0.41 (much lower).

---

## Step 7: Cox PH C-Index Evaluation

```python
from lifelines.utils import concordance_index

# Cox C-index on training data
c_index_train = cph.concordance_index_
# 0.78

# C-index on validation cohort
c_index_val = concordance_index(
    val['duration'],
    -cph.predict_partial_hazard(val[features_for_cox]),  # higher hazard = shorter survival
    val['event']
)
# 0.762

# Equivalent of AUC for survival -- 0.5 = random, 1.0 = perfect ranking
```

---

## Step 8: Random Survival Forest as Challenger

```python
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

# sksurv expects target as structured array (event, duration)
y_train_surv = Surv.from_arrays(survival_df['event'], survival_df['duration'])

rsf = RandomSurvivalForest(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    n_jobs=-1,
    random_state=42
)
rsf.fit(X_train, y_train_surv)

# C-index
c_index_rsf = rsf.score(X_val, y_val_surv)
# 0.804  <- better than Cox PH

# RSF handles non-linear interactions automatically; Cox PH is linear in log-hazard
```

**Expert insight:** Cox PH assumes linear log-hazard relationships and proportional hazards over time. Random Survival Forest doesn't — it captures interactions for free. For datasets with 30+ features and significant interactions, RSF often beats Cox by 3-5 C-index points.

The trade-off: Cox PH gives interpretable hazard ratios; RSF gives feature importance + survival curves but no clean "this feature multiplies hazard by X" story.

---

## Step 9: Tuning Random Survival Forest

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 4, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 25),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 0.3, 0.5]),
    }
    rsf = RandomSurvivalForest(**params, n_jobs=-1, random_state=42)
    rsf.fit(X_train, y_train_surv)
    return -rsf.score(X_val, y_val_surv)

study = optuna.create_study()
study.optimize(objective, n_trials=40)
# Best C-index: 0.821
```

---

## Step 10: Per-Customer Survival Curves and Expected Lifetime

```python
# Predict survival function for each customer (probability still active at each month 0-60)
survival_curves = rsf.predict_survival_function(X_test)

# For customer #4821 (a sample):
times = np.arange(0, 61)
sf_at_times = survival_curves[4821](times)
# sf_at_times[6]  = 0.85  -- 85% chance still active at 6 months
# sf_at_times[12] = 0.71  -- 71% at 12 months
# sf_at_times[24] = 0.52  -- 52% at 24 months

# Expected remaining lifetime (integrate area under survival curve)
expected_lifetime_months = np.trapz(sf_at_times, times)
# ~31 months for this customer

# This is what feeds the CFO's revenue forecast:
expected_revenue = expected_lifetime_months * customer_mrr
# = 31 * $99 = $3,069 expected total future revenue
```

---

## Step 11: Calibration of Survival Curves

```python
# Survival predictions need calibration check too
# At time t=12 months, do customers predicted with 70% survival probability actually
# survive at 70% rate?

# Group customers into deciles by predicted survival at month 12
df_test['pred_surv_12mo'] = [sf(12) for sf in survival_curves]
df_test['decile'] = pd.qcut(df_test['pred_surv_12mo'], 10, labels=range(1, 11))

# For each decile, compute Kaplan-Meier estimate of actual 12-month survival
from lifelines import KaplanMeierFitter
calibration_table = []
for d in range(1, 11):
    mask = (df_test['decile'] == d)
    kmf = KaplanMeierFitter()
    kmf.fit(df_test.loc[mask, 'duration'], df_test.loc[mask, 'event'])
    actual_12mo = kmf.predict(12)
    predicted_12mo = df_test.loc[mask, 'pred_surv_12mo'].mean()
    calibration_table.append({
        'decile': d,
        'predicted': predicted_12mo,
        'actual_12mo': actual_12mo,
        'gap': predicted_12mo - actual_12mo
    })

# Decile  Predicted  Actual  Gap
# 1       0.32       0.30    +0.02
# 2       0.45       0.43    +0.02
# 3       0.55       0.54    +0.01
# 4       0.63       0.62    +0.01
# 5       0.71       0.69    +0.02
# 6       0.78       0.76    +0.02
# 7       0.85       0.84    +0.01
# 8       0.91       0.90    +0.01
# 9       0.95       0.94    +0.01
# 10      0.98       0.97    +0.01
# All deciles within 2pp -- well calibrated.
```

---

## Step 12: Final Evaluation

```python
# Final model: Random Survival Forest (tuned)
#
# Test set: 80,000 customers from 2023 (mature -- mostly observed at this point)
#
# Performance:
#   C-index:                  0.821
#   Brier score at 12 months: 0.084
#   Integrated Brier score:   0.092
#   Calibration ECE at 12mo:  0.018
#
# Compare:
#   Cox PH C-index:          0.762
#   Naive regression on months_active (treating censored as "lifetime=current_obs"):
#     RMSE:                   18.4 months (massively over-confident)
#     C-index when probabilistic predictions extracted: 0.71
#     Bias: predictions are systematically too low (ignores future churn for censored)
```

---

## Step 13: Per-Customer Explainability

```python
# Random Survival Forest gives feature importance globally
importance = rsf.feature_importances_
# Top 10 features by SHAP-like permutation importance:
#   subscription_tier=Free               0.142
#   core_feature_adoption_pct_30d        0.118
#   has_csm_assigned                     0.094
#   acquisition_channel                  0.083
#   power_user_score                     0.071
#   onboarding_completed_within_7d       0.064
#   ...

# Per-customer explanations require sksurv's SHAP-like methods
# Or use Cox PH coefficients as the linear approximation:
# "Customer #4821 has predicted 31-month lifetime
#  Top drivers vs population baseline:
#  + has_csm_assigned (HR 0.48) -> reduces churn hazard 52%
#  + core_feature_adoption=0.62 (HR 0.41 per unit) -> adopting features lowers hazard significantly
#  + acquisition_channel=Referral (HR 0.55 vs Paid_Search) -> referral customers stay longer
#  - subscription_tier=Basic (HR 1.4 vs Premium) -> upselling would help retention"
```

---

## Step 14: Two-Tier Workflow

```python
# Combine survival predictions with intervention triggers
#
# For each active customer:
#   1. Predict survival curve
#   2. Compute hazard rate at current month (slope of survival curve)
#   3. If hazard rate spikes (delta > 0.05 over 30 days), flag for retention outreach
#   4. If hazard rate stable but predicted < 6 months remaining, flag for upsell

# This is more sophisticated than thresholding "P(churn within 30 days) > 0.20"
# because it surfaces customers whose risk just JUMPED, even if absolute risk is moderate
```

---

## Step 15: Deployment Considerations

```python
# Model size: ~100MB (300-tree Random Survival Forest)
# Inference per customer: ~50ms

# Quarterly retraining:
#   - Roll forward: drop oldest 3 months of data, add 3 most recent
#   - Re-tune hyperparameters every other quarter
#   - Recompute calibration on most recent observed cohort

# Monitoring:
#   - C-index drift on observed-this-quarter cohort
#   - Calibration drift at 6/12/24 months
#   - Per-segment C-index (subscription tier, channel)
#   - When new product launches: model staleness expected; flag for retraining

# Quarterly external audit for finance:
#   - Compare predicted lifetime × MRR vs actual realized revenue
#   - Aggregate predictions should be unbiased estimator of total revenue
#   - Drift in this aggregation triggers immediate retrain
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Censoring | Treat censored customers as "lifetime = last observed month" | Survival regression (Cox PH or RSF) handles censoring natively |
| Loss / metric | RMSE on months_active | Concordance index + Brier score for survival |
| Output | Single number "expected churn month" | Survival curve over 0-60 months |
| Expected lifetime | Use predicted month directly | Integrate area under survival curve |
| Model choice | Linear regression on time | Cox PH (interpretable) + RSF (accurate) |
| Calibration | Skip | Per-time-point calibration check (predicted vs Kaplan-Meier actual) |
| Cox vs RSF | Use one | Use both: Cox for hazard ratios; RSF for accuracy |
| Feature engineering | Demographic only | Heavy emphasis on first-30-day usage features |
| Train/test split | Random | Cohort-based (acquisition quarter); never seen during training |
| Imbalance handling | (n/a for survival) | Naturally handled by survival models |
| Intervention triggers | "predict will churn" | Hazard rate change + lifetime threshold |
| Revenue forecast | Sum predicted churns | Integrate survival × MRR for proper expectation |
