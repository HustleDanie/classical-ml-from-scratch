# Expert Scenario 41: Customer Lifetime Value (CLV)

> **Complexity:** Heavy-tailed target (whales drive 40%+ of revenue), right-censoring (most customers are still active at training time), two-part model (will they stay? × how much will they spend?), long horizon (24 months) with quarterly retraining.

---

## The Brief

A subscription e-commerce platform gives you 4 years of customer transactions. They want to predict 24-month forward Customer Lifetime Value (CLV) for each customer at the moment of acquisition. The CLV prediction feeds:

- Acquisition spend caps per customer cohort (don't pay $400 to acquire someone whose 24-month CLV will be $80).
- VIP / whale identification (top 1% gets human concierge service).
- Churn-prevention triggers (CLV trajectory below baseline → intervention).

Constraints:

- The top 1% of customers ("whales") generate ~40% of revenue. Their predictions must NEVER be capped or clipped.
- Most customers in training data are right-censored (still active, true 24-month CLV unknown).
- Reporting metric: WAPE (weighted absolute percentage error) on customers whose 24-month window has fully elapsed.
- Quarterly retraining; calibrated probabilities feed downstream cost models.

This is harder than a standard regression because: censoring breaks naive supervised learning, the heavy tail makes RMSE dishonest, and customer cohorts behave differently (Q4 acquisitions ≠ Q1 acquisitions).

---

## Step 1: Define the Problem Type

```
Type:           Regression on cumulative spend (heavy-tailed) — best framed as a
                two-part model: (1) survival, (2) spend-per-active-period
Primary Metric: WAPE on rolled-off cohorts (those with 24mo elapsed)
Secondary:      Spearman rank correlation (ranking matters for VIP tier);
                Decile lift on revenue
Business Goal:  Top-decile cumulative revenue capture > 60%
Constraint:     Quarterly retrain; calibrated probabilities for cost-allocation
Target shape:   Heavy right-tail; ~30% zero (one-time purchasers); long-tail $0-$15K
```

**Expert thinking:** modeling cumulative-spend as one regression is the wrong frame. A customer who churns in month 3 has different dynamics from one who stays 24 months. The standard approach is two-part:
1. Survival regression: how long will they stay?
2. Spend regression: while active, how much will they spend per period?
Multiplied together = CLV. This decomposition gives better calibration and natural interpretability.

---

## Step 2: Understand the Data

```
Shape: ~3M customers (acquired between 2020-2024) x ~70 features

Target construction:
- For customers acquired before 2022-01-01: TRUE 24-month CLV is observable
- For customers acquired between 2022-01 and 2024-01: PARTIAL observation (right-censored)
- For customers acquired after 2024-01: < 12 months observation

Feature categories:

ACQUISITION (8):
- customer_id (unique)
- acquisition_date
- acquisition_channel (Paid_Search/Social/Organic/Affiliate/Direct/Referral)
- acquisition_cost ($1.50 - $400, by channel)
- first_visit_pages (int)
- first_session_duration_min (float)
- referrer_domain (high cardinality, 8000+ unique)
- utm_campaign (5000+ unique)

DEMOGRAPHICS / PROFILE (10):
- age (int, 18-85, 12% missing)
- gender (M/F/Non-binary/Prefer-not, 22% missing)
- country (40+, mostly US)
- state (US 50)
- zip3 (HIPAA-safe truncation)
- income_bracket (estimated, 5 brackets, 35% missing)
- preferred_language

FIRST-30-DAYS BEHAVIORAL (32 -- the heart of the model):
- first_purchase_date
- days_to_first_purchase
- num_orders_30d
- total_spend_30d
- avg_order_value_30d
- num_categories_purchased_30d
- num_brands_purchased_30d
- num_returns_30d
- email_opens_30d
- email_clicks_30d
- num_login_sessions_30d
- avg_session_duration_30d
- mobile_pct_30d
- weekend_purchase_pct_30d
- has_added_to_wishlist
- has_used_coupon
- has_used_subscription
- num_support_tickets_30d
- ... 14 more behavioral

PRODUCT MIX (15):
- top_category_pct (which top-level category did they buy most in?)
- num_unique_skus
- premium_sku_pct
- discount_avg_pct
- ... 11 more

LIFECYCLE STATE AT FEATURE-CUT (5):
- status_at_cut (Active/Churned/Dormant/Refunded)
- days_since_last_order
- months_since_acquisition
```

**Expert thinking:** the **first-30-days behavioral features** are the primary signal. Acquisition channel is the secondary lever. Demographics matter much less than people expect — what someone *does* in their first 30 days predicts their next 24 months far better than who they are.

**Right-censoring:** customers acquired in 2024 have only seen 6-12 months of activity. Their "24-month CLV" is unknown. We can't drop them (loses data); we can't treat their partial spend as final (biases low). Solution: train the survival model on censored data; train the spend model on per-period observations.

---

## Step 3: Exploratory Data Analysis (EDA)

```python
# Take customers with FULL 24-month observation (acquired before 2022-01)
mature = df[df['acquisition_date'] < '2022-01-01']
mature['actual_24m_clv'] = mature['actual_revenue_24m']

mature['actual_24m_clv'].describe()
# count    1,200,000
# mean        $89.40
# std        $312.10
# min          $0.00
# 25%         $19.00
# 50%         $48.00
# 75%         $104.00
# 95%        $312.00
# 99%        $890.00
# max     $14,820.00
# Skew: 18.2 (extreme)

# Top 1% capture
mature['actual_24m_clv'].quantile(0.99)  # $890
mature[mature['actual_24m_clv'] > 890]['actual_24m_clv'].sum() / mature['actual_24m_clv'].sum()
# 0.43 -- top 1% generate 43% of revenue
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Top 1% generate 43% of revenue | Whale identification matters more than overall RMSE |
| 30% of customers spend $0 over 24 months (one-time purchase or refund-only) | Two-part: P(stay) × E[spend\|stay] |
| First-30-day spend correlates 0.42 with 24m CLV | Strongest single predictor |
| Days_to_first_purchase < 1 day → CLV 2.4x higher than > 7 days | "Decisiveness" is a powerful signal |
| Acquisition_channel='Referral' → 3.1x CLV vs 'Paid_Search' | Channel quality matters massively |
| has_used_subscription → 4.8x CLV | Subscriptions are CLV gold |
| num_returns_30d > 2 → 0.4x CLV | High returners often refund-and-bail |
| Email clicks_30d > 3 → 1.8x CLV | Engagement signal is real |
| Q4-acquired customers have 1.3x CLV | Holiday cohorts are over-represented in subsequent purchases (gifting → repeat) |

**Expert insight:** the heavy tail (`skew=18`) means RMSE is dominated by mansion-equivalent customers — predicting $400 vs $400K matters very little for normal customers but enormously for whales. The reporting metric should be WAPE, and the modeling target should be log1p-transformed.

---

## Step 4: Data Cleaning

```python
# === HANDLE CENSORED CUSTOMERS ===
# For two-part model:
#   Part 1 (survival): use ALL customers, including censored. Time-to-churn or "still active at T".
#   Part 2 (spend): use CUSTOMER-PERIODS (not customers). Each customer contributes
#                   one observation per active period.

# Define cohorts:
df['observation_months'] = (
    pd.Timestamp('2024-12-31') - df['acquisition_date']
) / pd.Timedelta(days=30.44)
# Mature: observation >= 24
# Censored: observation 6-23 months
# Too-new: observation < 6

# === MISSING VALUES ===
# Age 12% missing -- impute by acquisition channel (channels skew demographic)
df['age'] = df.groupby('acquisition_channel')['age'].transform(lambda x: x.fillna(x.median()))

# Gender 22% missing -- treat as separate category
df['gender'] = df['gender'].fillna('Unknown')

# income_bracket 35% missing -- mostly self-reported; create a "_missing" flag
df['income_bracket_missing'] = df['income_bracket'].isnull().astype(int)
df['income_bracket'] = df['income_bracket'].fillna('Unknown')

# === HIGH-CARDINALITY CATEGORICALS ===
# referrer_domain: 8000+ unique values
# Bucket: top-50 by volume kept individual; rest → 'Other'
top_referrers = df['referrer_domain'].value_counts().head(50).index
df['referrer_domain'] = df['referrer_domain'].where(df['referrer_domain'].isin(top_referrers), 'Other')

# utm_campaign: 5000+ unique
# Same treatment: top 100 + 'Other'
top_campaigns = df['utm_campaign'].value_counts().head(100).index
df['utm_campaign'] = df['utm_campaign'].where(df['utm_campaign'].isin(top_campaigns), 'Other')

# === OUTLIERS ===
# DO NOT cap. Whales drive 43% of revenue. A $14K customer is the signal, not noise.
# But for log1p-target stability:
df['target_log1p'] = np.log1p(df['actual_24m_clv'])  # log1p handles zeros gracefully
```

---

## Step 5: Feature Engineering

```python
# === RFM-style FEATURES (computed from 30-day observation) ===
df['recency_30d'] = df['days_to_first_purchase']  # how recent was first purchase?
df['frequency_30d'] = df['num_orders_30d']
df['monetary_30d'] = df['total_spend_30d']

# Engagement-to-spend ratio
df['email_engagement_rate'] = df['email_clicks_30d'] / (df['email_opens_30d'] + 1)
df['return_ratio'] = df['num_returns_30d'] / (df['num_orders_30d'] + 1)

# Order velocity
df['orders_per_active_day'] = df['num_orders_30d'] / (df['num_login_sessions_30d'] + 1)

# === ACQUISITION CHANNEL HISTORICAL CLV ===
# Compute on TRAINING data only (with smoothing)
channel_clv = df_train.groupby('acquisition_channel')['target_log1p'].mean()
channel_volume = df_train.groupby('acquisition_channel').size()
overall_mean = df_train['target_log1p'].mean()
alpha = 1000
df['channel_clv_smoothed'] = (
    df['acquisition_channel'].map(
        (channel_volume * channel_clv + alpha * overall_mean) / (channel_volume + alpha)
    )
).fillna(overall_mean)

# === COHORT FEATURES (when did they join?) ===
df['acquisition_quarter'] = df['acquisition_date'].dt.quarter
df['acquisition_year'] = df['acquisition_date'].dt.year
df['is_holiday_season'] = df['acquisition_quarter'] == 4

# === FIRST-PURCHASE FEATURES ===
df['first_purchase_aov'] = df['avg_order_value_30d']
df['high_first_aov'] = (df['first_purchase_aov'] > df['first_purchase_aov'].quantile(0.75)).astype(int)

# === COMPOSITE INDEX (ENGAGEMENT BREADTH) ===
df['engagement_breadth'] = (
    (df['num_categories_purchased_30d'] >= 3).astype(int) +
    (df['num_brands_purchased_30d'] >= 3).astype(int) +
    df['has_added_to_wishlist'] +
    df['has_used_subscription'] +
    (df['email_engagement_rate'] > 0.2).astype(int)
)
# Range 0-5; higher = more invested in the platform

# === DEVICE / TIME PATTERNS ===
df['mobile_dominant'] = (df['mobile_pct_30d'] > 0.7).astype(int)
df['weekend_buyer'] = (df['weekend_purchase_pct_30d'] > 0.5).astype(int)
```

---

## Step 6: Feature Selection

```python
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(X_train, y_train_log, random_state=42)

# Top 15 features:
#   total_spend_30d              0.142
#   num_orders_30d               0.118
#   has_used_subscription        0.103
#   channel_clv_smoothed         0.094
#   engagement_breadth           0.083
#   first_purchase_aov           0.071
#   email_clicks_30d             0.062
#   num_categories_purchased_30d 0.058
#   days_to_first_purchase       0.054
#   acquisition_channel          0.049
#   num_returns_30d              0.041
#   email_engagement_rate        0.038
#   has_added_to_wishlist        0.034
#   ...
```

Drop features with MI < 0.005 (mostly demographic noise after channel encoding handles the lift).

---

## Step 7: Preprocessing

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = [...]  # ~30 numeric (after engineering)
categorical_features = ['acquisition_channel', 'gender', 'state', 'income_bracket',
                         'referrer_domain', 'utm_campaign', 'top_category']
binary_features = [...]   # ~10 binary

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('bin', 'passthrough', binary_features)
])
```

---

## Step 8: Train/Test Split — Cohort-Based

```python
# Cohort-based, NOT random:
#   - Train: customers acquired 2020-01 to 2022-12 (mature: full 24mo elapsed by 2024-12)
#   - Validation: customers acquired 2023-01 to 2023-06 (some censoring -- use only CLV-to-date as proxy)
#   - Test: customers acquired 2022-07 to 2022-12 (mature, NEVER seen during training)

# Why cohort-based:
#   - Random split leaks future cohort behavior into training
#   - Different acquisition cohorts have different macroeconomic context (post-COVID, recession, etc.)
#   - We need to know: how does the model perform on a held-out COHORT?

# Test set: ~600,000 customers with full 24-month CLV observed
```

---

## Step 9: Baselines

```python
# Baseline 1: predict the global mean CLV ($89) for everyone
# WAPE: 89%
# Useless.

# Baseline 2: predict by acquisition_channel mean
# WAPE: 76%

# Baseline 3: predict total_spend_30d * 6  (naive linear extrapolation)
# WAPE: 51%

# Baseline 4: simple Linear Regression on top-10 features
# WAPE: 39%
# Decile lift on top 10%: 4.2x base
```

---

## Step 10: Try Multiple Models with Cohort-Based CV

```python
# Split train into 4 sub-cohorts (one per quarter); 3-cohort train, 1-cohort validate

# Model 1: ElasticNet (interpretable baseline)
# Cohort CV WAPE: 36.4%

# Model 2: Random Forest (n=300, max_depth=12)
# Cohort CV WAPE: 28.1%

# Model 3: LightGBM with regression objective
# Cohort CV WAPE: 25.3%

# Model 4: LightGBM with Tweedie objective (handles zero-inflation)
# Cohort CV WAPE: 23.8%
# Tweedie better than Gaussian by 1.5pp

# Model 5: XGBoost
# Cohort CV WAPE: 24.2%

# Model 6: Two-part model (survival + spend, see Step 11)
# Cohort CV WAPE: 21.4%   <- BEST
```

**Top performer:** two-part model. Tweedie LightGBM is the best single-model option.

---

## Step 11: Two-Part Model — The Right Frame

```python
# Part 1: Survival -- "How long will they stay?"
# Use Cox Proportional Hazards on the 30-day features
# Target: time-to-churn (months active before going dormant)
# Censoring: customers still active at observation cut

from lifelines import CoxPHFitter

# Format for survival: (months_active, churned [event observed])
cph = CoxPHFitter(penalizer=0.01)
cph.fit(survival_df, duration_col='months_active', event_col='churned')
# C-index on validation: 0.74

# Predict expected lifetime in months
expected_lifetime = cph.predict_expectation(X_val)

# Part 2: Spend per active month
# Filter to customer-months where customer was active
# Train regression on those
spend_per_month = LGBMRegressor(objective='gamma', ...)
spend_per_month.fit(X_active_months, y_monthly_spend)
# CV WAPE on monthly spend: 18%

# Combine:
predicted_24m_clv = expected_lifetime.clip(upper=24) * predicted_monthly_spend
# clip at 24 because we're predicting 24-month CLV, not lifetime

# WAPE on test set: 21.4% (improvement from 23.8%)
```

**Expert insight:** the two-part model is more interpretable AND more accurate. You can answer "is this customer high CLV because they'll spend a lot per month, or because they'll stay a long time?" — that's a different intervention strategy in each case.

---

## Step 12: Hyperparameter Tuning

```python
# Tune the spend-per-month model with Optuna
import optuna

def objective(trial):
    params = {
        'objective': 'gamma',  # gamma = positive-only continuous, right-skewed
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    return cohort_cv_wape(params, X_train, y_train_monthly)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=80)
# Best WAPE on monthly spend: 16.2%
```

---

## Step 13: Calibration for Decile Lift

```python
# CLV predictions feed acquisition spend caps. The DECILE structure must be calibrated.
# (Customers ranked top-10% by predicted CLV should actually have top-10% realized CLV.)

# After two-part model:
# Predicted top decile mean CLV: $387
# Actual top decile mean CLV:    $356
# Calibration ratio: 1.09 (slight over-prediction at the top)

# Apply isotonic calibration on the predicted top deciles:
from sklearn.isotonic import IsotonicRegression

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(predicted_clv_train, y_train)
calibrated_clv = iso.predict(predicted_clv_test)

# Calibration ratio after isotonic: 1.01 (well-calibrated)
```

---

## Step 14: Whale Identification — Tail Performance Matters Most

```python
# Top 1% capture (whales)
# Predicted top 1% should overlap with actual top 1%

# Without whale-specific tuning:
# Predicted top 1% precision = 51% (51% of predicted whales were actual whales)
# Predicted top 1% recall    = 51%

# Whales are who the brief cares about most.
# Tactic: train a SECOND model specifically for binary whale prediction
#   target: actual_24m_clv > $890 (the top-1% threshold)
#   model: LightGBM classifier with class_weight='balanced'
# Test recall@precision=0.5: 0.74

# Combine: use main CLV model for everyone; use whale model to override top tier
final_predictions = main_clv_predictions.copy()
whale_probs = whale_classifier.predict_proba(X_test)[:, 1]
top_whales = np.argsort(whale_probs)[-int(0.01 * len(X_test)):]
final_predictions[top_whales] = main_clv_predictions[top_whales] * 1.2  # boost predicted CLV for confirmed whales

# After whale-specific layer:
# Top 1% precision = 0.69 (+18 pp)
# Top 1% recall    = 0.69
```

---

## Step 15: Final Evaluation on Held-Out Test Cohort

```python
# Final architecture:
#   Part 1: Cox PH for expected lifetime
#   Part 2: LightGBM (gamma objective) for spend per active month
#   Combine: expected_lifetime * monthly_spend, clipped at 24 months
#   Calibrate: isotonic on top decile
#   Whale layer: separate classifier boosts top 1% predictions

# Test set: 600K customers acquired 2022-07 to 2022-12 (full 24-month observation)
#
# Performance:
#   Overall WAPE:                21.4%
#   WAPE on bottom 50%:          18.2%
#   WAPE on top 10%:             24.6%
#   WAPE on top 1% (whales):     31.8%   (but precision/recall are what matter)
#   Spearman rank correlation:   0.71
#   Top-1% precision:            0.69
#   Top-1% recall:               0.69
#   Top-decile revenue capture:  62.4%   ✓ (target 60%)
```

---

## Step 16: Explainability

```python
import shap

# For the spend-per-month component, SHAP works directly
spend_explainer = shap.TreeExplainer(spend_per_month_model)

# For Cox PH, the coefficients ARE the explanation
cph.print_summary()
#  feature                hazard_ratio  p
#  total_spend_30d          0.92        <0.001  (more spend -> lower churn hazard)
#  has_used_subscription    0.71        <0.001
#  num_returns_30d          1.18        <0.001  (more returns -> higher churn hazard)
#  acquisition_channel=Referral  0.81  <0.001
#  ...

# Per-customer explanation:
# Customer #4821 -- Predicted 24mo CLV: $1,240 (top 5%)
# Lifetime component: 22 expected months active (vs 14 baseline)
#   - Drivers: high engagement_breadth (5/5), used subscription, 0 returns
# Spend component: $56/month (vs $19 baseline)
#   - Drivers: high first AOV ($89), 4 categories purchased, premium SKU mix

# Interpretable for marketing team: "this customer is high CLV because they'll stay long
# AND spend a lot per month -- target them with retention not new offers."
```

---

## Step 17: Deployment Considerations

```python
# Model size: ~150MB (Cox PH + LightGBM + whale classifier + calibrator)
# Per-customer inference: ~80ms (acceptable for batch acquisition decisions)

# Production:
#   Batch nightly: score new acquisitions and refresh CLV estimates for 30-day-old cohort
#   Real-time path: NOT needed -- CLV is decisioning input for budgets, not transactions

# Quarterly retraining:
#   - Roll forward training cohorts (drop oldest quarter, add newest mature quarter)
#   - Re-tune Tweedie variance power for new cohort
#   - Re-calibrate isotonic
#   - A/B test: compare new-model decile lift to old on hold-out cohort

# Monitoring:
#   - Per-cohort WAPE 30/60/90 days post-acquisition (early warning)
#   - Whale precision @ top 1% (weekly)
#   - Decile-revenue-capture trend (monthly)
#   - Acquisition spend ROI by predicted-CLV decile (monthly)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Censoring | Treat censored customers as "low CLV" | Survival regression (Cox PH); use censored data correctly |
| Heavy-tailed target | Cap whales at 99th percentile | Keep whales (they're 43% of revenue); use Tweedie / gamma loss |
| Train/test split | Random | Cohort-based (acquisition quarter); never seen during training |
| Single regression model | Predict total CLV directly | Two-part: P(stay) × E[spend\|active] |
| Categorical handling | One-hot 8000 referrers | Keep top-50, bucket rest as 'Other' |
| Loss | RMSE | WAPE (revenue-weighted) for reporting; Tweedie / gamma for training |
| Whale identification | Same model for everyone | Separate classifier; boost top-1% predictions |
| Calibration | Use raw model output | Isotonic on top deciles for accurate decile lift |
| Demographics weight | Heavy demo features | Demote demographics; emphasize first-30-day behavioral features |
| Evaluation | Mean WAPE | WAPE by decile; top-1% precision/recall; revenue capture |
| Per-customer explanation | "model said $X" | Decompose: lifetime component + spend-per-month component, each with drivers |
