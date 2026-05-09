# Expert Scenario 17: Insurance Premium Pricing (Actuarial GLM)

> **Complexity:** State-by-state regulated rate-filings, GLM with Tweedie distribution (matches insurance loss math), monotonicity constraints (more risk → higher premium, no exceptions), per-state coefficient scrutiny, can't use protected attributes (different prohibitions per state), competitive market response.

---

## The Brief

A national auto insurance company gives you 12M policies × 8 years of premium and claims history. They want a model to recommend the "technical premium" (pure risk-based price before margin/competition adjustment) for each new or renewing policy. The output:

- Drives initial quote pricing (marketing + underwriter system).
- Feeds into rate-filing submissions to state insurance commissioners.
- Provides a baseline for actuarial review.

Constraints:

- Loss-ratio MAPE ≤ 5% (target loss ratio is the metric, not premium MAE).
- Per-state regulatory compliance (different prohibitions: California prohibits credit-based features; Texas allows them).
- Monotonic constraints: e.g., a younger driver's premium MUST be at-or-above an older driver's premium with the same other characteristics. No "edge case where 25-year-old pays less than 50-year-old."
- Auditable for state rate-filing submissions: every coefficient must be defensible.
- Tweedie distribution to handle zero-claims (most policies have no claims) + heavy positive tail.

This is harder than typical regression because: regulatory regime varies state-by-state; monotonic constraints constrain the model class; the target distribution is zero-inflated heavy-tailed (most policies = 0, few are catastrophic); and the deliverable is a defendable rate-filing, not just predictions.

---

## Step 1: Define the Problem Type

```
Type:           Regression on annual premium (target = pure_premium = expected loss × frequency)
Primary Metric: Tweedie deviance; Loss-Ratio MAPE
Secondary:      Lift on highest-risk decile; Gini coefficient
                Per-state model accuracy
Business Goal:  Loss-ratio MAPE ≤ 5%; Lift on top decile ≥ 4x
Constraint:     State-specific regulatory prohibitions; monotonic constraints
                Auditable; rate-filing-defensible coefficients
Target shape:   Tweedie (zero-inflated; tail driven by catastrophic claims)
```

**Expert thinking:** Tweedie distribution is THE actuarial target distribution. With variance_power between 1 (Poisson) and 2 (Gamma), it captures the "lots of zeros + heavy tail" pattern of insurance losses. Standard GLM with Tweedie family is the actuarial gold standard.

---

## Step 2: Understand the Data

```
Shape: 100M policy-years (12M unique policies × ~8 renewals each)
Target: pure_premium = sum_of_losses_in_year (most are 0)

Per policy-year features:

DRIVER:
- driver_id (anonymized)
- driver_age (16-90)
- gender (some states prohibit)
- marital_status (some states prohibit)
- driver_education
- driving_experience_years
- driver_license_status
- prior_accidents_count (rolling 36mo)
- prior_violations_count (rolling 36mo)
- prior_at_fault_accidents
- prior_DUI_flag

VEHICLE:
- vehicle_make_model
- vehicle_year
- vehicle_age_years
- vehicle_class (Sedan/SUV/Truck/Sports/Luxury)
- vehicle_value
- annual_mileage (declared; verified for some)
- has_anti_theft (binary)
- has_safety_features_count

LOCATION (state-by-state regulated):
- garage_zip (varies by state on what's allowed)
- garage_lat, garage_lng
- urban_rural_classification
- territory_code (insurance industry's territory codes)

POLICY:
- coverage_level (Liability-only / Standard / Premium)
- liability_limits ($25K-$1M)
- deductible
- multi_policy_discount (binary; bundled with home, life, etc.)

STATE-SPECIFIC PROHIBITIONS:
- credit_score (USED in TX, NV, IL; PROHIBITED in CA, MI, MA)
- gender (USED in some; PROHIBITED in others, e.g. since 2019 CA)
- ZIP-level demographic factors (handled per-state)

LOSS HISTORY:
- losses_paid_year_t (target's source)
- num_claims_year_t
- claim_severity_distribution_year_t
```

---

## Step 3: EDA

```python
df['pure_premium'].describe()
# count    100M
# mean     $710 (per policy-year)
# std      $5,400 (extreme variance)
# min      $0
# 25%      $0
# 50%      $0    <-- 75%+ of policies have ZERO losses
# 75%      $382
# 95%      $2,100
# 99%      $11,800
# 99.9%    $48,000
# max      $720,000

# Variance / Mean = 30+ (heavily overdispersed for Poisson)
# Tweedie with variance_power ~1.5-1.7 fits well

# Per state policy patterns
df.groupby('state').agg(
    avg_pure_premium=('pure_premium', 'mean'),
    avg_claim_freq=('num_claims_year_t', 'mean'),
    avg_severity=('losses_paid_year_t', lambda x: x[x > 0].mean())
)
# Florida:    $1,420 avg premium, 8.4% freq, $16K severity
# California: $890,                 4.1%,    $21K
# Iowa:       $480,                 3.2%,    $14K
# Reflects different driving patterns, no-fault laws, hurricane risk, etc.
```

---

## Step 4: Per-State Modeling Strategy

```python
# REGULATORY REQUIREMENT: each state has its own rate-filing
# Cannot directly use a state's coefficients in another state
# Must train a separate model per state — OR a global model with state-specific
# per-state factor adjustments

# Hybrid approach:
# 1. Train a global "structural" model on shared features (driver age, vehicle)
# 2. Apply state-specific factor adjustments based on per-state historical loss ratios
# 3. Validate per-state filings

# Pure per-state would have data fragmentation issues (state with 50K policies has
# weak training set for tree methods)
```

---

## Step 5: GLM with Tweedie Distribution

```python
# Classical actuarial GLM with Tweedie distribution
import statsmodels.api as sm

# Pure premium GLM with Tweedie family
# variance_power=1.5 typical for auto insurance

X = pd.get_dummies(df[features], drop_first=True)  # One-hot for categoricals
formula = "pure_premium ~ ..."  # actuarial-style formula

model = sm.GLM(
    df['pure_premium'],
    X,
    family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log())
).fit()

# Coefficients are interpretable as multiplicative factors:
# coef for "driver_age_25_29" = 0.48 → multiplies premium by exp(0.48) = 1.62
# Each coefficient is a "rating factor" submitted to regulators

# Model summary:
# AIC, deviance, R-deviance for Tweedie
# Per-coefficient p-values + standard errors
```

---

## Step 6: LightGBM as Challenger

```python
import lightgbm as lgb

# Force monotonic constraints on age (younger = riskier)
# and prior_accidents (more accidents = riskier)
monotonic = {
    'driver_age': -1,           # negative (older drivers cheaper)
    'prior_accidents_count': 1,  # positive (more accidents = more cost)
    'prior_violations_count': 1,
    'prior_at_fault_accidents': 1,
    'prior_DUI_flag': 1,
    'driving_experience_years': -1,
}

lgb_model = lgb.LGBMRegressor(
    objective='tweedie',
    tweedie_variance_power=1.5,
    monotone_constraints={f: v for f, v in monotonic.items()},
    n_estimators=2000, max_depth=4, num_leaves=15,
    learning_rate=0.04, reg_alpha=1.0, reg_lambda=2.0,
    random_state=42
)

# CV Tweedie deviance: 1.42
# CV Gini coefficient: 0.42 (good for actuarial work)
# CV Loss-ratio MAPE: 4.7%  ✓
```

---

## Step 7: Per-State Adjustment

```python
# Compute global model + state-specific factor
# Factor accounts for state-specific risk environment beyond what the model captures

state_factors = {}
for state in df_train['state'].unique():
    mask = df_train['state'] == state
    actual_loss_ratio = df_train.loc[mask, 'pure_premium'].sum() / predicted_premium[mask].sum()
    state_factors[state] = actual_loss_ratio

# Apply to predictions
df['adjusted_predicted_premium'] = predicted_premium * df['state'].map(state_factors)
```

---

## Step 8: Final Evaluation

```python
# Final model: Tweedie LightGBM with monotonic constraints + per-state factors
# Test set: most recent 24 months
#
# Performance:
#   Loss-ratio MAPE:           4.4%     ✓ (target 5%)
#   Tweedie deviance:          1.39
#   Gini coefficient:          0.43
#   Top-decile lift:           4.6x     ✓ (target 4x)
#
# Per-state performance:
#   FL Loss-ratio MAPE:        5.2%
#   CA Loss-ratio MAPE:        3.8%
#   IA Loss-ratio MAPE:        4.1%
#
# Regulatory readiness:
#   All state-specific factor coefficients positive and within filing norms
#   Monotonic constraints respected
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Distribution | Gaussian regression | Tweedie GLM (matches insurance loss math) |
| Loss function | RMSE | Tweedie deviance |
| Per-state | One global model | Hybrid: global model + per-state factor adjustments |
| Monotonicity | Hope it's right | Hard constraints (younger = riskier; more accidents = pricier) |
| Protected attributes | Use them | Strict per-state prohibitions; documented |
| Metric | Premium MAE | Loss-ratio MAPE; Gini; top-decile lift |
| Audit | Skip | Per-state rate-filing submission package |
| Renewable | Annual retraining | Full audit + state filing each renewal cycle |
