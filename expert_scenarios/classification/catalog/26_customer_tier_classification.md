# Expert Scenario 26: Customer Tier Classification (Ordinal Multiclass)

> **Complexity:** Ordinal target (Bronze < Silver < Gold < Platinum) — accuracy is wrong metric, quadratic-weighted kappa correctly penalizes off-by-many predictions, business rules influence tier definitions, the tier is used for downstream pricing/perks (so calibration matters).

---

## The Brief

A subscription company gives you 8M customer records. They have 4 tiers (Bronze, Silver, Gold, Platinum) determined by an internal scoring algorithm using engagement + spend + tenure. They want a model that predicts a customer's "natural tier" given current features:

- Match the existing rule-based tier 75%+ of the time.
- For mismatches, the model's prediction should be off by at most 1 tier (Quadratic Weighted Kappa ≥ 0.80).
- Predictions feed into tier-specific perks pricing, A/B test eligibility, and cross-sell models.
- They want to evolve from rule-based to ML-based tiering eventually, but only if the model is well-aligned with current tiers.

This is harder than standard multiclass because: tier is ORDINAL — predicting Bronze when reality is Platinum is much worse than predicting Gold when reality is Platinum. Accuracy treats all errors equally, which misrepresents the cost.

---

## Step 1: Define the Problem Type

```
Type:           Ordinal multiclass classification (4 ordered tiers)
Primary Metric: Quadratic Weighted Kappa (QWK)
                Penalizes off-by-many predictions; rewards near-misses
Secondary:      Accuracy; Per-tier recall; Macro-F1
Business Goal:  QWK ≥ 0.80; Accuracy ≥ 75%
Constraint:     Aligned with current rule-based tier definitions
                Retrain quarterly as tier rules evolve
Imbalance:      Bronze 50% / Silver 30% / Gold 15% / Platinum 5%
```

**Expert thinking:** Quadratic Weighted Kappa is the right metric for ordinal predictions. It's effectively MSE on the ordinal scale, normalized to a -1 to 1 range. Predicting Silver when truth is Gold (1 tier off) is penalized 1; predicting Bronze when Platinum is the truth (3 tiers off) is penalized 9. This matches business reality.

---

## Step 2: Understand the Data

```
Shape: 8,000,000 customers × ~70 features
Target: customer_tier (Bronze / Silver / Gold / Platinum)

Features:

ENGAGEMENT (15):
- num_logins_30d
- num_active_days_30d
- avg_session_minutes_30d
- num_features_used_30d
- engagement_breadth (number of distinct features touched)
- last_login_days_ago
- weekly_active_days_avg_3mo
- ...

SPEND (10):
- monthly_revenue_avg_3mo
- annual_lifetime_value_to_date
- num_purchases_12mo
- avg_order_value_12mo
- premium_subscription_active
- num_addon_subscriptions
- ...

TENURE (5):
- account_age_months
- subscription_age_months
- num_renewal_cycles
- has_been_paid_subscriber (binary)
- months_in_current_plan

DEMOGRAPHICS:
- country
- age_band
- segment_at_signup
- referrer_type

PRODUCT MIX:
- num_categories_purchased_lifetime
- premium_product_purchases_pct
- has_used_advanced_features (binary)
- num_referrals_made

TARGET:
- customer_tier (assigned by current rule-based system)

```

---

## Step 3: EDA

```python
df['customer_tier'].value_counts(normalize=True)
# Bronze    0.50
# Silver    0.30
# Gold      0.15
# Platinum  0.05

# Tier transitions are usually monotonic in the underlying score
# Verify by computing average engagement / spend per tier
df.groupby('customer_tier').agg(
    avg_revenue=('monthly_revenue_avg_3mo', 'mean'),
    avg_logins=('num_logins_30d', 'mean'),
    avg_account_age=('account_age_months', 'mean')
)
# Bronze:    $24/mo, 8 logins, 4 months
# Silver:    $89/mo, 18 logins, 14 months
# Gold:      $245/mo, 32 logins, 28 months
# Platinum:  $890/mo, 48 logins, 38 months

# Clear ordinal progression — confirms the target is genuinely ordinal
```

---

## Step 4: Feature Engineering

```python
# === ORDINAL-AWARE COMPOSITE SCORES ===
df['engagement_score'] = (
    df['num_logins_30d'] / df['num_logins_30d'].max() +
    df['num_active_days_30d'] / 30 +
    df['avg_session_minutes_30d'] / df['avg_session_minutes_30d'].max() +
    df['engagement_breadth'] / df['engagement_breadth'].max()
)

df['spend_score'] = (
    np.log1p(df['monthly_revenue_avg_3mo']) +
    df['premium_subscription_active'].astype(int) +
    np.log1p(df['num_purchases_12mo']) +
    np.log1p(df['avg_order_value_12mo']) / 5
)

df['tenure_score'] = (
    np.log1p(df['account_age_months']) +
    df['has_been_paid_subscriber'].astype(int) +
    np.log1p(df['num_renewal_cycles'])
)

# === COMPOSITE TIER PREDICTOR ===
df['composite_score'] = (
    df['engagement_score'] * 0.4 +
    df['spend_score'] * 0.4 +
    df['tenure_score'] * 0.2
)

# === DELTAS ===
df['engagement_change_3mo'] = df['num_logins_30d'] / df['num_logins_30d_3mo_ago'] - 1
df['spend_change_3mo'] = df['monthly_revenue_avg_3mo'] / df['monthly_revenue_3mo_ago'] - 1
```

---

## Step 5: Model Choices for Ordinal

```python
# Three common approaches:
# 1. Treat as multiclass (e.g., Softmax) — ignores ordering
# 2. Train binary classifiers for each threshold (Cumulative Logit / Ordinal Regression)
# 3. Treat as regression on ordinal numeric value, then bin

# We'll compare all three

# === Model 1: LightGBM Multiclass (ignores ordering) ===
lgb_multi = lgb.LGBMClassifier(objective='multiclass', num_class=4, ...)
# QWK: 0.74; accuracy: 78%

# === Model 2: Ordinal Regression (Cumulative Logit) ===
# Train K-1 binary classifiers: P(tier > Bronze), P(tier > Silver), P(tier > Gold)
# Combine into ordinal probability
def fit_cumulative_ordinal(X, y, num_classes):
    binary_models = {}
    for k in range(num_classes - 1):
        y_binary = (y > k).astype(int)
        binary_models[k] = lgb.LGBMClassifier(...).fit(X, y_binary)
    return binary_models

ordinal_models = fit_cumulative_ordinal(X_train, y_train_ordinal, 4)

# Predict by:
# P(tier <= k) = P(>k) - P(>k+1)
# Sum cumulative; return argmax over P(tier == k)

# QWK: 0.81 (better!); accuracy: 76%
# Trade-off: slight accuracy drop for big QWK improvement

# === Model 3: Regression on tier_index then bin ===
# Treat tier as 0/1/2/3 numeric
# Train regression; round predictions to nearest integer
lgb_reg = lgb.LGBMRegressor(objective='regression', ...)
lgb_reg.fit(X_train, y_train.map({'Bronze':0,'Silver':1,'Gold':2,'Platinum':3}))
predictions_continuous = lgb_reg.predict(X_test)
predictions_binned = np.clip(np.round(predictions_continuous), 0, 3).astype(int)
# QWK: 0.82
# accuracy: 75%
```

**Top performer:** Either ordinal regression OR regression+bin (QWK ~0.81-0.82). Regression+bin is simpler.

---

## Step 6: Per-Tier Performance

```python
# Per-tier recall (using regression+bin model)
for tier in ['Bronze', 'Silver', 'Gold', 'Platinum']:
    tier_idx = {'Bronze':0, 'Silver':1, 'Gold':2, 'Platinum':3}[tier]
    mask = (y_test == tier)
    pred = predictions_binned[mask] == tier_idx
    print(f"{tier}: recall {pred.mean():.2f}")

# Bronze:   0.83
# Silver:   0.72
# Gold:     0.65
# Platinum: 0.78  -- decent because it's the most-distinct tier

# Confusion matrix:
#               Predicted
#              Br  Si  Go  Pl
# Bronze:    [83,  16,  1,  0]
# Silver:    [12,  72, 14,  2]
# Gold:      [ 1,  19, 65, 15]
# Platinum:  [ 0,   2, 20, 78]

# Most errors are off-by-one (adjacent tiers); few off-by-many
# This is exactly what QWK rewards
```

---

## Step 7: Calibration on Boundary Cases

```python
# Customers near the boundary between two tiers are the hardest
# E.g., a customer scoring 0.74 on the composite_score is between Silver (0.6-0.8) and Gold (0.8+)
# Their predicted tier is sensitive to noise

# Identify boundary customers
df['tier_confidence'] = predictions_continuous.std()  # for ensemble; or use proba spread for ordinal model
df['near_boundary'] = (df['tier_confidence'] < 0.4).astype(int)

# For boundary customers, business rule: round towards higher tier (give benefit of the doubt)
predictions_boundary_aware = np.where(
    df['near_boundary'],
    np.ceil(predictions_continuous),  # round up
    np.round(predictions_continuous)
)
```

---

## Step 8: Final Evaluation

```python
# Final model: LightGBM regression on tier-as-int + boundary-aware rounding
# Test set: 800K held-out customers
#
# Performance:
#   QWK:                    0.83  ✓ (target 0.80)
#   Accuracy:               74%   ✓ (target 75% — slightly under)
#   Macro-F1:               0.71
#   Off-by-1 errors:        20% (acceptable; QWK favorable)
#   Off-by-2+ errors:       3%   (rare)
#
# Per-tier:
#   Bronze recall:    83%
#   Silver recall:    72%
#   Gold recall:      65%
#   Platinum recall:  78%
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Metric | Accuracy | Quadratic Weighted Kappa (penalizes off-by-many) |
| Model framing | Softmax multiclass | Regression on ordinal index, then bin |
| Loss function | Cross-entropy | MSE on ordinal numeric |
| Boundary handling | Round to nearest | Boundary-aware rounding (give benefit of the doubt) |
| Per-tier audit | Skip | Confusion matrix; off-by-1 vs off-by-many breakdown |
| Domain composites | Use raw features | Engineered engagement / spend / tenure scores |
| Imbalance | class_weight | Inherent in regression framing |
| Tier evolution | Train once | Quarterly retrain as tier rules evolve |
| Output | Final tier only | Continuous score + binned tier + confidence |
| Business alignment | Ignore | Check that model agrees with rule-based system before deploying |
