# Expert Scenario 44: Ad Lift / Incrementality (Causal Inference / Uplift Modeling)

> **Complexity:** Causal regression — predict the *incremental* effect of treatment, not the post-treatment outcome. The fundamental problem of causal inference: you cannot observe the same person's outcome both with AND without the treatment. Uplift trees, Causal Forests, and meta-learners are the toolkit. Variance is high; large samples needed. RCT data preferred over observational when available.

---

## The Brief

A digital marketing team gives you 80M user records spanning 6 months. For each user:

- Treatment: was-shown-ad (yes/no), with random assignment for ~40% of impressions (legitimate AB test)
- Outcome: did-convert (yes/no, within 7 days of impression)
- Many user features (demographics, behavior, device)

The team wants:

- A model that predicts **uplift** for each user: P(convert | shown_ad) − P(convert | not_shown_ad)
- Top-decile by predicted uplift should capture 60%+ of the actual incremental conversions
- A way to identify users for whom ads are wasteful (uplift ≤ 0; they would have converted anyway, OR they are anti-targeted) vs. users where ads drive real lift

This is fundamentally different from a "will they convert" classifier. The user with highest conversion probability might be someone who would convert even WITHOUT the ad — showing them the ad is wasted spend. The model must identify the **persuadable** users (low base rate, but ad shifts them).

Constraints:

- AUUC (Area Under the Uplift Curve) ≥ 0.10
- Decile lift on incremental conversions ≥ 4x base rate
- Inference < 100ms per user (real-time bidding integration)
- Honest about uncertainty: high-variance estimates are useful in aggregate but unreliable per-user

This is harder than typical regression because: the target itself is unobservable per individual (you only see one outcome per person, not both); RCT data is scarce; the model must be evaluated on a held-out RCT to be credible.

---

## Step 1: Define the Problem Type

```
Type:           Causal regression on uplift = P(Y=1|T=1) − P(Y=1|T=0)
Primary Metric: AUUC (Area Under the Uplift Curve)
Secondary:      Qini coefficient
                Decile lift on incremental conversions (top decile)
                ATE (Average Treatment Effect) consistency
Business Goal:  AUUC ≥ 0.10; top decile lift ≥ 4x
Constraint:     Real-time scoring; high variance per-user expected
Treatment:      Binary; ~40% randomization (legitimate AB test)
```

**Expert thinking:** AUUC measures how well the model RANKS users by treatment effect. Even a noisy model can rank well if the signal-to-noise ratio is right. Per-user uplift estimates are extremely high-variance; aggregate-decile estimates are stable. We optimize ranking, not individual prediction.

---

## Step 2: Understand the Data

```
Shape: 80M records — each = (user, impression timestamp, treatment, outcome)

Per record:
- record_id (unique)
- timestamp
- user_id (anonymized)
- treatment (0 = not shown ad, 1 = shown ad)
- treatment_was_random (binary; 1 if from RCT arm)
- outcome (1 = converted within 7 days, 0 = no conversion)

User features (~100):
- demographic: age_band, gender, location_region
- past_engagement: clicks_30d, page_views_30d, time_on_site_30d
- past_conversions: prior_conversion_count, days_since_last_conversion
- product_affinity: top-3 product categories browsed
- device: mobile/desktop, OS
- session_recency: days_since_last_visit

Critical:
- ad_creative_id (which ad was shown)
- ad_placement (banner / interstitial / native)
```

**Expert thinking:** the most important feature in the data is `treatment_was_random`. Only the randomized portion of the data gives us unbiased estimates of uplift. Non-random treatment data is contaminated by selection bias (the model that decided which users to target last time encoded its biases).

---

## Step 3: EDA

```python
# Conversion rate by treatment (RCT data only)
df_rct = df[df['treatment_was_random'] == 1]
df_rct.groupby('treatment')['outcome'].mean()
# T=0 (control):   2.1%
# T=1 (treatment): 2.7%
# ATE = 0.6 percentage points (modest but real)

# Per-segment ATE (already heterogeneous)
for segment in ['young_male', 'young_female', 'old_male', 'old_female']:
    mask = (df_rct['user_segment'] == segment)
    ate = df_rct[mask & (df_rct['treatment'] == 1)]['outcome'].mean() - df_rct[mask & (df_rct['treatment'] == 0)]['outcome'].mean()
    print(f"{segment}: ATE {ate:.4f}")
# young_male: ATE +1.2pp (very persuadable)
# young_female: ATE +0.4pp
# old_male: ATE -0.1pp (slightly NEGATIVE — anti-targeted)
# old_female: ATE +0.3pp

# Heterogeneity is real and exploitable
```

---

## Step 4: Approach — Two-Model (T-Learner)

```python
# T-Learner: train two separate models
# Model 1: P(Y | X, T=0) — predict outcome for control
# Model 2: P(Y | X, T=1) — predict outcome for treatment
# Uplift estimate = M1.predict(X) - M2.predict(X)

# Use ONLY RCT data for unbiased causal estimates

from sklearn.ensemble import GradientBoostingClassifier

control_data = df_rct[df_rct['treatment'] == 0]
treatment_data = df_rct[df_rct['treatment'] == 1]

m_control = GradientBoostingClassifier(n_estimators=200, max_depth=4)
m_control.fit(X_control, y_control)

m_treatment = GradientBoostingClassifier(n_estimators=200, max_depth=4)
m_treatment.fit(X_treatment, y_treatment)

# Predict uplift
def predict_uplift(X):
    return m_treatment.predict_proba(X)[:, 1] - m_control.predict_proba(X)[:, 1]

predicted_uplift = predict_uplift(X_test)

# AUUC: 0.087
```

---

## Step 5: Approach — X-Learner (Better)

```python
# X-Learner: more sophisticated, handles imbalanced treatment groups better
# Particularly useful when treatment rate is unequal in observational data

from sklearn.ensemble import RandomForestRegressor

# Step 1: Estimate outcome models for each group
# Step 2: Compute pseudo-outcomes for each unit
# Step 3: Fit treatment-effect models on the pseudo-outcomes
# Step 4: Combine via propensity score weighting

# (skipping full code; using a library like causalml)
from causalml.inference.meta import BaseXRegressor

x_learner = BaseXRegressor(
    learner=RandomForestRegressor(n_estimators=200, max_depth=6),
    control_name=0
)
x_learner.fit(X_train_rct, treatment_train_rct, outcome_train_rct)
uplift_predictions = x_learner.predict(X_test)

# AUUC: 0.105 ✓ (target 0.10)
```

---

## Step 6: Approach — Causal Forest (Best for High-Variance)

```python
# Causal Forest: tree-based estimator designed for heterogeneous treatment effects
# Provides uncertainty estimates per prediction

from econml.dml import CausalForestDML

cf = CausalForestDML(
    n_estimators=300,
    max_depth=5,
    random_state=42
)
cf.fit(
    Y=outcome_train_rct,
    T=treatment_train_rct,
    X=X_train_rct
)

uplift_predictions, uplift_lower, uplift_upper = cf.effect(X_test, alpha=0.10)
# 90% CI per user

# AUUC: 0.108 ✓
```

---

## Step 7: Decile Lift Analysis

```python
# This is the actual deployment metric
# Sort users by predicted uplift; bucket into deciles
# Compute actual ATE in each decile (using held-out RCT data)

df_test['predicted_uplift'] = uplift_predictions
df_test['decile'] = pd.qcut(df_test['predicted_uplift'], 10, labels=range(10), duplicates='drop')

# For each decile, what's the actual ATE?
for d in range(10):
    decile_mask = df_test['decile'] == d
    treated = df_test[decile_mask & (df_test['treatment'] == 1)]['outcome'].mean()
    control = df_test[decile_mask & (df_test['treatment'] == 0)]['outcome'].mean()
    actual_ate = treated - control
    print(f"Decile {d}: predicted uplift {df_test[decile_mask]['predicted_uplift'].mean():.4f}, "
          f"actual ATE {actual_ate:.4f}")

# Decile 0 (lowest predicted uplift): predicted -0.005, actual -0.003 (negative ATE; anti-targeted)
# Decile 5 (middle):                    predicted 0.002,  actual 0.003
# Decile 9 (highest):                   predicted 0.018,  actual 0.024 (4x base rate)

# Decile 9 has 4.6x the lift of average — exploitable
```

---

## Step 8: Cumulative Lift Curve & AUUC

```python
# Sort users by predicted uplift descending
# At each cumulative percentage of users targeted, compute total incremental conversions

sorted_df = df_test.sort_values('predicted_uplift', ascending=False)

cumulative_lift = []
for pct in np.arange(0.01, 1.01, 0.01):
    n_targeted = int(pct * len(sorted_df))
    targeted_subset = sorted_df.head(n_targeted)
    treated_in_targeted = targeted_subset[targeted_subset['treatment'] == 1]['outcome'].sum()
    control_in_targeted = targeted_subset[targeted_subset['treatment'] == 0]['outcome'].sum() / 0.5  # adjust for treatment rate
    cumulative_incremental = treated_in_targeted - control_in_targeted
    cumulative_lift.append((pct, cumulative_incremental))

# AUUC = area under this curve
# Higher AUUC = better at ranking persuadable users to the top

# Decision rule: target users in top 30% of predicted uplift
# Captures ~70% of incremental conversions; saves 70% of ad spend
```

---

## Step 9: Final Evaluation

```python
# Final approach: Causal Forest with X-Learner backup
# Test set: held-out RCT users
#
# Performance:
#   AUUC:                       0.108  ✓ (target 0.10)
#   Qini coefficient:           0.062
#   Top decile lift:            4.6x   ✓ (target 4x)
#   Bottom decile lift:        -0.3x   (negative; correctly identifies anti-persuadable)
#   ATE consistency:            within 0.1pp of holdout RCT ATE
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Modeling target | Predict P(convert) | Predict P(convert|T=1) − P(convert|T=0) (uplift) |
| Data | Use all observational | Use ONLY RCT data for unbiased estimates |
| Approach | Single classifier | T-Learner / X-Learner / Causal Forest |
| Metric | Accuracy on conversion | AUUC, decile lift, Qini |
| Evaluation | Standard CV | Cumulative lift curve on held-out RCT |
| Heterogeneity | Single ATE | Identify per-segment uplift; deciles |
| Use case | "Target high-converters" | "Target high-uplift" (different users!) |
| Anti-targeting | Skip | Identify users with NEGATIVE uplift; SAVE money |
| Confidence | Point estimate | Causal Forest gives per-prediction CIs |
| Production | Score and rank | Decile-based decision rules; only top-X% targeted |
