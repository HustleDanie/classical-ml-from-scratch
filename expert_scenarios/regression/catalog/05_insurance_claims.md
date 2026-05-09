# Expert Scenario 5: Insurance Claim Amount Prediction (Zero-Inflated Target)

> **Complexity:** Two-stage problem (will they claim? + how much?), zero-inflated target (85% of policyholders never claim), extreme right-skew in claim amounts, regulatory requirement for non-discrimination, actuarial validation needed.

---

## The Brief

An auto insurance company has 2.3 million active policyholders. For each renewal cycle, they need to predict the **total claim amount** each policyholder will incur in the next 12 months. This drives premium pricing: if the model predicts a customer will claim $3,000, their premium must be at least $3,000 + overhead to be profitable. Currently, pricing is based on broad risk pools (age group x car type x region). They want individual-level predictions. The catch: 85% of policyholders will claim $0, and among those who claim, amounts range from $200 to $500,000+.

This is complex because: the target is "zero-inflated" (85% zeros), claim amounts are heavily right-skewed (median $2,800, mean $4,200, max $500,000+), the model must be explainable for regulatory reasons (why was this customer charged more?), and certain features (race, religion) are legally prohibited even if predictive.

---

## Step 1: Define the Problem Type

```
Type:           Regression (predicting dollar amount, 0 to 500,000+)
                BUT: Zero-inflated -- 85% of targets are exactly $0
Primary Metric: MAE on predicted claim cost (must be accurate for pricing)
Secondary:      Expected loss ratio (predicted claims / actual claims should be ~1.0)
Regulatory:     Model must not use prohibited features (race, gender, religion, disability)
                Model must be explainable to regulators
Special:        Target distribution:
                $0:          85%
                $1-$2,000:   7%
                $2K-$10K:    5%
                $10K-$50K:   2.5%
                $50K+:       0.5%
```

**Expert thinking:** You CANNOT train a single regression model on this data. 85% zeros will dominate, and the model will predict ~$600 (the mean) for everyone. Instead, this is a classic **two-stage "hurdle" model**:
- Stage 1: Will this person file a claim? (binary classification)
- Stage 2: If they claim, how much? (regression, only on claimants)
- Final prediction: P(claim) x E[amount | claim]

---

## Step 2: Understand the Data

```
2.3 million policyholders with:

Policyholder demographics:
- age (int, 16-95)
- gender (M/F/Other -- CANNOT USE for pricing in most states)
- marital_status (single, married, divorced, widowed)
- credit_score (int, 300-850 -- controversial but legal in most states)
- years_licensed (int, 0-70)
- education_level (high_school, bachelors, masters, phd, other)
- occupation (string, 180 categories)
- annual_income (float, $0 to $2M+)
- homeowner (binary)

Vehicle information:
- vehicle_year (int, 1985-2026)
- vehicle_make (string, 45 makes)
- vehicle_model (string, 680 models)
- vehicle_type (sedan, SUV, truck, sports, luxury, minivan)
- vehicle_value (float, current market value)
- safety_rating (1-5 stars)
- annual_mileage (float, estimated miles/year)
- primary_use (commute, pleasure, business)

Policy details:
- coverage_type (liability_only, collision, comprehensive, full)
- deductible_amount (float, $250 to $5,000)
- policy_start_date (datetime)
- continuous_years_insured (int, how long they've been a customer)

Driving history:
- num_accidents_3yr (int, number of at-fault accidents in last 3 years)
- num_violations_3yr (int, speeding tickets, DUI, etc.)
- num_claims_3yr (int, number of claims filed)
- total_claims_amount_3yr (float, total $ claimed in last 3 years)
- has_dui (binary)
- license_suspended_ever (binary)

Location:
- zip_code (string, 12,400 unique)
- state (string, 50 states)
- urban_rural (categorical: urban, suburban, rural)
- avg_commute_minutes (float, from census data)
- weather_risk_score (float, hail/flood/ice risk for the area)
- theft_rate_per_1000 (float, vehicle theft rate in the area)
- accident_rate_per_1000 (float, accident rate in the area)

Target:
- total_claim_amount (float, $0 to $487,000)
```

---

## Step 3: EDA Findings

| Finding | Implication |
|---------|------------|
| 85% of policyholders claim $0 | Need two-stage model |
| Among claimants, median $2,800, mean $4,200, max $487K | Right-skewed, need log transform for Stage 2 |
| `num_accidents_3yr` is the #1 predictor: 0 accidents = 11% claim rate, 3+ = 54% | Driving history dominates |
| `age` is U-shaped: 16-19 (22% claim rate), 30-55 (12%), 70+ (18%) | Non-linear age effect |
| `vehicle_type=sports` has 2.3x the claim rate of sedan | Car type matters |
| `credit_score < 600` has 2.1x the claim rate | Credit correlated with risk (controversial but legal) |
| Zip codes show 5x variation in claim rates (inner city vs rural) | Location is powerful |
| `deductible_amount` inversely correlated with claims (adverse selection: risky people choose low deductibles) | Self-selection bias -- useful signal |
| `vehicle_value` strongly predicts claim AMOUNT (not frequency) | Expensive cars = expensive claims |
| Claims vary by season: 15% more in winter (ice/snow) | Temporal pattern |

---

## Step 4-5: Cleaning & Feature Engineering

```python
# === Missing values ===
# credit_score (3% missing): impute with median by state + create flag
# annual_income (8% missing): impute with median by occupation + create flag
# annual_mileage (15% missing -- self-reported, unreliable): impute, create flag,
#   ALSO create 'mileage_reported' flag (people who report may be more careful)

# === Feature Engineering ===

# Driving risk score (composite)
df['driving_risk_score'] = (
    df['num_accidents_3yr'] * 3 +
    df['num_violations_3yr'] * 1.5 +
    df['has_dui'] * 5 +
    df['license_suspended_ever'] * 4
)

# Vehicle age and depreciation
df['vehicle_age'] = 2026 - df['vehicle_year']
df['vehicle_age_bucket'] = pd.cut(df['vehicle_age'], bins=[0, 2, 5, 10, 20, 50],
                                   labels=['new', 'recent', 'middle', 'old', 'very_old'])

# Value-to-income ratio (can they afford to fix it out of pocket?)
df['value_income_ratio'] = df['vehicle_value'] / (df['annual_income'] + 1000)

# Experience features
df['years_driving'] = df['years_licensed']
df['is_young_driver'] = (df['age'] < 25).astype(int)
df['is_new_driver'] = (df['years_licensed'] < 3).astype(int)
df['young_and_new'] = df['is_young_driver'] * df['is_new_driver']  # interaction

# Claims history features
df['avg_claim_size_3yr'] = df['total_claims_amount_3yr'] / (df['num_claims_3yr'] + 0.01)
df['has_any_claims_3yr'] = (df['num_claims_3yr'] > 0).astype(int)
df['claims_escalating'] = ...  # are claims getting more frequent/expensive?

# Location risk (from zip code -- high cardinality encoding)
zip_claim_rate = df.groupby('zip_code')['has_claim'].mean()
df['zip_claim_rate'] = df['zip_code'].map(zip_claim_rate)

zip_avg_amount = df[df['total_claim_amount'] > 0].groupby('zip_code')['total_claim_amount'].mean()
df['zip_avg_claim_amount'] = df['zip_code'].map(zip_avg_amount)

# Coverage interaction (low deductible + high risk = expects to claim)
df['low_deductible_high_risk'] = (
    (df['deductible_amount'] <= 500) & (df['driving_risk_score'] >= 5)
).astype(int)

# Loyalty features
df['is_long_customer'] = (df['continuous_years_insured'] >= 5).astype(int)
df['loyalty_discount_eligible'] = (df['continuous_years_insured'] >= 3) & (df['num_claims_3yr'] == 0)
```

---

## Step 6-8: Feature Selection, Preprocessing, Split

```python
# REGULATORY COMPLIANCE: Remove prohibited features
prohibited = ['gender', 'race', 'ethnicity', 'religion', 'disability']
# Note: some states also prohibit credit_score for insurance pricing
# Check state regulations before deploying

# Feature selection results (MI for Stage 1 -- claim probability):
# Top: driving_risk_score, num_accidents_3yr, age, credit_score, zip_claim_rate
# Top for Stage 2 (claim amount): vehicle_value, coverage_type, num_accidents_3yr

# Train/test: Stratified random split (80/20), stratified on claim/no-claim
# Use 5-fold CV with stratification during development
```

---

## Step 9-10: Two-Stage "Hurdle" Model

```python
# ================================================================
# STAGE 1: Will they file a claim? (Binary Classification)
# ================================================================
# Target: has_claim (0 or 1), 85% vs 15%

# Model comparison (5-fold Stratified CV):
# Logistic Regression:  AUC=0.78, F1=0.42
# Random Forest:        AUC=0.84, F1=0.51
# XGBoost:              AUC=0.87, F1=0.56
# LightGBM:             AUC=0.88, F1=0.57  <-- Best

# LightGBM Stage 1 details:
# At threshold 0.5:  Recall=0.52, Precision=0.48
# At threshold 0.15: Recall=0.82, Precision=0.23
# For pricing, we don't use hard threshold -- we use P(claim) directly

# Key: We need PROBABILITIES, not classifications
# P(claim) = 0.35 means "35% chance of filing a claim"
# This feeds into the pricing formula

# Calibration check (critical for insurance!):
# Among customers the model gives P=0.20, do exactly 20% actually claim?
# Use Platt scaling or isotonic regression for calibration
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(lgbm_stage1, method='isotonic', cv=5)

# After calibration:
# Predicted P=0.10 -> Actual rate: 0.098 (good!)
# Predicted P=0.30 -> Actual rate: 0.312 (good!)
# Predicted P=0.50 -> Actual rate: 0.483 (good!)
# Brier score: 0.091 (well-calibrated)

# ================================================================
# STAGE 2: How much will they claim? (Regression, only on claimants)
# ================================================================
# Target: total_claim_amount (only where > 0)
# Training set: 345,000 policyholders who filed claims
# Target: median $2,800, mean $4,200, std $12,500, max $487,000

# Log-transform the target (heavy right skew, skewness=8.4)
y_train_stage2_log = np.log1p(y_train_stage2)
# After log: approximately normal, skewness=0.6

# Model comparison on log-transformed claims:
# Ridge Regression:   MAE=$1,820
# Random Forest:      MAE=$1,450
# XGBoost:            MAE=$1,230
# LightGBM:           MAE=$1,190  <-- Best

# Special handling for catastrophic claims (>$50K):
# Only 0.5% of all policyholders, but 15% of total claim dollars
# These are the hardest to predict and the most impactful for pricing
# LightGBM MAE on $50K+ claims: $18,200 (off by $18K on average)
# Use sample_weight=5 for claims >$50K to improve

# With weighting:
# Overall MAE: $1,280 (slightly worse)
# Catastrophic claim MAE: $14,100 (much better on the expensive ones)
# Use weighted model -- catastrophic accuracy matters more for profitability

# ================================================================
# COMBINED PREDICTION
# ================================================================

# For each policyholder:
# Expected_claim = P(claim) * E[amount | claim]
#
# Example:
# Customer A: P(claim)=0.25, E[amount|claim]=$3,800
# Expected claim = 0.25 * $3,800 = $950
# Premium should be >= $950 + overhead
#
# Customer B: P(claim)=0.08, E[amount|claim]=$2,200
# Expected claim = 0.08 * $2,200 = $176
# Premium should be >= $176 + overhead
```

---

## Step 11: Actuarial Validation

```python
# Insurance-specific validation that regulators require:

# 1. Loss Ratio by Decile
# Sort customers by predicted expected claim, split into 10 groups
# For each group: actual claims / predicted claims should be ~1.0

# Decile | Predicted Avg | Actual Avg | Ratio
# 1      | $112          | $98        | 0.88  (we slightly overpriced -- safe)
# 2      | $234          | $251       | 1.07
# 3      | $398          | $382       | 0.96
# 4      | $612          | $635       | 1.04
# 5      | $890          | $867       | 0.97
# 6      | $1,245        | $1,310     | 1.05
# 7      | $1,780        | $1,695     | 0.95
# 8      | $2,620        | $2,750     | 1.05
# 9      | $4,100        | $3,890     | 0.95
# 10     | $8,950        | $9,240     | 1.03
# ALL ratios between 0.85 and 1.15 -- excellent calibration!

# 2. Lift Chart
# Top 10% riskiest customers account for what % of actual claims?
# Model: top 10% accounts for 42% of claims (strong discrimination)
# Current system: top 10% accounts for 28% of claims
# Our model is 50% better at identifying high-risk customers

# 3. Gini Coefficient
# Model: 0.61 (good -- typical for auto insurance is 0.3-0.5)
# Current system: 0.38
# Significant improvement in risk separation

# 4. Fairness Audit (CRITICAL for regulatory compliance)
# Even without using gender directly, the model might proxy it
# Test: Is the loss ratio the same for men vs women?
# Men:   Predicted avg $1,150, Actual avg $1,180, Ratio 1.03
# Women: Predicted avg $890,   Actual avg $870,   Ratio 0.98
# Both within acceptable range -- no disparate impact

# Same test for age groups, zip codes (proxy for race), etc.
# All groups within 0.85-1.15 loss ratio -- passes fairness audit
```

---

## Step 12-13: Tuning & Final Model

```python
# Tuned two-stage system:
# Stage 1: LightGBM classifier, calibrated, n_estimators=1500, max_depth=6
# Stage 2: LightGBM regressor on log(amount), weighted, n_estimators=2000

# Additional: Add Tweedie regression as an alternative
# Tweedie distribution naturally handles zero-inflated continuous targets
from lightgbm import LGBMRegressor
tweedie_model = LGBMRegressor(objective='tweedie', tweedie_variance_power=1.5)
# Tweedie MAE: slightly worse than two-stage but simpler (one model vs two)

# Final blend: 70% two-stage + 30% Tweedie
# MAE: $820 per policyholder (vs current system: $1,450)
# Improvement: 43%
```

---

## Step 14: Final Evaluation

```python
# Test set: 460,000 policyholders
#
# Expected Claim MAE:    $820 (current system: $1,450, improvement: 43%)
# Loss Ratio:            1.02 (near-perfect: collecting $1.02 for every $1 of claims)
# Gini Coefficient:      0.61 (current: 0.38)
#
# Business impact:
# Current: Loss ratio 1.08 in worst decile (losing money on riskiest 10%)
# Model:   Loss ratio 1.03 across all deciles (profitable everywhere)
# Annual profit improvement: ~$34M (better risk selection + pricing accuracy)
#
# High-risk detection:
# Customers model identifies as top 1% risk: actual claim rate 62%
# These customers were previously in average risk pools paying average premiums
# Now they're priced appropriately (or referred to high-risk carriers)
```

---

## Step 15-16: Explainability & Deployment

```python
# SHAP explanation for regulators:
# "Customer John Doe's premium is $2,400/year because:
#  + 2 at-fault accidents in past 3 years (+$680)
#  + Age 19, licensed 2 years (+$520)
#  + Sports car, vehicle value $35K (+$340)
#  + Zip code 90015, high accident area (+$280)
#  + Low deductible choice $250 (+$180)
#  - Clean violation record (-$120)
#  - 800 credit score (-$200)
#  Base rate: $720"

# Deployment:
# - Runs during annual policy renewal (batch, not real-time)
# - Output: predicted expected claim per policyholder
# - Pricing team adds overhead, profit margin, and caps
# - Regulatory filing: model documentation + fairness audit results
# - Monitoring: quarterly comparison of predicted vs actual loss ratios
# - Retrain annually with new claims data
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Zero-inflated target | Single regression (predicts ~$600 for everyone) | Two-stage hurdle: P(claim) x E[amount given claim] |
| Calibration | Ignore probability calibration | Isotonic regression + decile loss ratio validation |
| Skewed amounts | Predict raw dollars | Log-transform + catastrophic claim weighting |
| High cardinality (zip) | One-hot 12,400 zips (explosion!) | Target-encoded zip_claim_rate |
| Regulatory | Use all features, hope for the best | Remove prohibited features + fairness audit on protected groups |
| Evaluation | MAE only | Actuarial: loss ratio by decile, Gini, lift chart, fairness |
| Alternative model | Only two-stage | Also tried Tweedie regression (single-model alternative), blended both |
| Business metric | "MAE is $820" | "Annual profit improvement: $34M" |
