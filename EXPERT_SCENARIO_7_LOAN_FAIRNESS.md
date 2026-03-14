# Expert Scenario 7: Loan Default Prediction with Fairness Constraints

> **Complexity:** Regulatory requirement for fair lending (cannot discriminate by race, gender, age), proxy variable detection, disparate impact testing, model must be globally explainable for regulators, reject inference problem (you only see outcomes for APPROVED loans), competing objectives (risk accuracy vs fairness).

---

## The Brief

A consumer lending company processes 500,000 loan applications/year. They currently approve 62% and decline 38%. Their default rate is 8.3% (among approved loans). They want to replace their manual underwriting rules with an ML model that predicts default probability for each applicant. Federal regulations (ECOA, Fair Housing Act) REQUIRE that the model does not discriminate based on race, color, religion, national origin, sex, marital status, or age. The model must produce "adverse action reasons" (legally required: if you deny someone, you must explain why). Past loan decisions were made by human underwriters who may have been biased -- this bias is in the training data.

This is complex because: fairness is legally mandated (not optional), training data has selection bias (only approved loans have outcomes), the model must be interpretable enough to generate adverse action notices, proxy variables can cause illegal discrimination even if protected features are removed, and the stakes are high (wrong approval = financial loss, wrong denial = regulatory penalty + lawsuit).

---

## Step 1: Define the Problem Type

```
Type:           Binary Classification (default within 36 months: yes/no)
Primary Metric: AUC-ROC (discrimination power)
Fairness Metric: Demographic parity ratio, equalized odds
Regulatory:     Adverse action reasons for every denial
                Disparate impact ratio must be > 0.80 (four-fifths rule)
                Model documentation for OCC/CFPB examination
Constraint:     Only have outcomes for APPROVED loans (selection bias)
```

---

## Step 2: Understand the Data

```
Historical data: 3 years of loan decisions
Total applications: 1,500,000
Approved (with outcomes): 930,000 (62%)
  - Paid in full: 852,810 (91.7%)
  - Defaulted:     77,190 (8.3%)
Declined (NO outcome data): 570,000 (38%)

Applicant features:
- annual_income (float)
- employment_length_years (float, 15% missing)
- home_ownership (rent, own, mortgage)
- loan_amount_requested (float)
- loan_purpose (debt_consolidation, credit_card, home_improvement, medical, etc.)
- debt_to_income_ratio (float)

Credit bureau data:
- credit_score (int, 300-850)
- total_credit_lines (int)
- open_credit_lines (int)
- total_revolving_balance (float)
- revolving_utilization (float, 0-1+)
- delinquencies_last_2yr (int)
- months_since_last_delinquency (float, missing if no delinquency)
- inquiries_last_6mo (int)
- public_records (int, bankruptcies, judgments)
- total_credit_history_months (int)

PROTECTED FEATURES (cannot use directly, must monitor):
- race (White, Black, Hispanic, Asian, Other)
- gender (Male, Female, Non-binary)
- age (int)
- zip_code (proxy for race in many areas)
- marital_status
```

---

## Step 3: EDA -- Bias Detection in Historical Data

```python
# CRITICAL FIRST STEP: Check if historical decisions were biased

# Approval rates by race (among comparable credit scores):
# Credit Score 680-720:
#   White applicants: 74% approved
#   Black applicants: 61% approved
#   Hispanic applicants: 65% approved
#   Asian applicants: 78% approved
# Gap exists even at SAME credit scores -- historical bias confirmed

# Default rates by race (among approved loans):
#   White borrowers: 8.1% default
#   Black borrowers: 9.2% default
#   Hispanic borrowers: 8.8% default
# Default rates are CLOSER than approval rates suggest
# Historical underwriters were MORE cautious with minorities than warranted

# Income distribution by zip code:
# Zip codes with >60% Black population have avg income $42K
# Zip codes with >60% White population have avg income $68K
# If we use income without adjustment, it proxies for race
```

**Expert insight:** The historical data contains TWO types of bias:
1. **Selection bias:** We only see outcomes for approved loans. Declined applicants might have been good borrowers we'll never know about.
2. **Historical discrimination:** Underwriters approved minorities at lower rates even when credit profiles were similar. If we train on this data naively, our model inherits and perpetuates this bias.

---

## Step 4-5: Cleaning & Feature Engineering (WITH Fairness Awareness)

```python
# === Reject Inference: Handling Missing Outcomes for Declined Applicants ===
# Problem: 570,000 declined applicants have no outcome data
# If we only train on approved loans, the model learns "approved-customer patterns"
# and can't properly evaluate borderline applicants

# Strategy 1: Augmentation approach
# For declined applicants with credit scores just below the approval threshold (580-620):
# Assume their default rate would be 15-20% (extrapolated from trend)
# Create synthetic labels for a SUBSET of declines
# This broadens the model's training distribution

# Strategy 2: Heckman correction (econometric approach)
# Two-stage:
# Model 1: P(approved) -- selection model
# Model 2: P(default | approved) -- outcome model with selection correction term
# The correction term (inverse Mills ratio) adjusts for selection bias

# We use Strategy 2 (more rigorous)
from statsmodels.regression import HeckmanTwoStep
selection_model = ...  # logit: will the human approve this application?
inverse_mills = compute_inverse_mills(selection_model.predict(X_all))
# Add inverse_mills as a feature to the default prediction model

# === Feature Engineering (bias-aware) ===

# Financial stability features
df['loan_to_income'] = df['loan_amount_requested'] / (df['annual_income'] + 1)
df['total_debt_to_income'] = df['debt_to_income_ratio']
df['payment_to_income'] = estimated_monthly_payment / (df['annual_income'] / 12 + 1)
df['credit_utilization_squared'] = df['revolving_utilization'] ** 2  # non-linear risk

# Credit history features
df['credit_history_years'] = df['total_credit_history_months'] / 12
df['avg_balance_per_line'] = df['total_revolving_balance'] / (df['open_credit_lines'] + 1)
df['delinquency_recency'] = 1 / (df['months_since_last_delinquency'] + 1)
df['has_recent_delinquency'] = (df['months_since_last_delinquency'] < 12).astype(int)

# Behavioral signals
df['inquiries_velocity'] = df['inquiries_last_6mo']  # many inquiries = desperate for credit
df['new_credit_pct'] = (df['total_credit_lines'] - df['open_credit_lines']) / (df['total_credit_lines'] + 1)

# === PROXY VARIABLE DETECTION ===
# Even without using race directly, zip_code and income can proxy for race

# Test: Does zip_code predict race?
zip_race_corr = df.groupby('zip_code')['race'].apply(lambda x: x.mode()[0])
# Yes -- 72% of zip codes are >80% one race. Zip IS a race proxy.

# Solution: Instead of raw zip_code, use ECONOMIC characteristics of the zip
df['zip_median_income'] = ...    # from census data
df['zip_unemployment_rate'] = ... # from BLS data
df['zip_avg_credit_score'] = ... # from our own data
# These are ECONOMICALLY justified (ability to repay) rather than demographic proxies
# But still monitor their disparate impact

# WARNING: Do NOT use zip_code directly. Courts have ruled it's a race proxy.
# Use only economically-justified zip-level features.
```

---

## Step 6-10: Model Building with Fairness

```python
# ================================================================
# MODEL TRAINING -- Standard (without fairness constraints)
# ================================================================

# LightGBM (best performer):
# AUC: 0.84, KS statistic: 0.51

# DEFAULT the model:
# Approval threshold at P(default) < 0.15:
# Approval rate: 65%, Default rate among approved: 6.2%

# Now check FAIRNESS:
# Approval rates by race (with model predictions):
#   White:    68%
#   Black:    54%   <-- PROBLEM
#   Hispanic: 58%   <-- PROBLEM
#   Asian:    72%

# Disparate Impact Ratio (DIR): minority_approval / majority_approval
# Black vs White: 54/68 = 0.79  <-- BELOW 0.80 threshold! ILLEGAL!
# Hispanic vs White: 58/68 = 0.85  <-- Above 0.80, but barely

# The model, despite not using race, produces discriminatory outcomes
# because income, zip-area features, and credit history correlate with race

# ================================================================
# FAIRNESS-CONSTRAINED MODEL
# ================================================================

# Approach 1: Post-processing (threshold adjustment)
# Set different thresholds per group to equalize approval rates
# White threshold: P(default) < 0.15
# Black threshold: P(default) < 0.19
# Hispanic threshold: P(default) < 0.17
# Result: DIR > 0.80 for all groups
# PROBLEM: This is explicit race-based decision-making -- legally questionable!

# Approach 2: In-processing (fairness-aware training)
# Add a fairness penalty to the loss function
# Goal: minimize prediction error WHILE keeping disparate impact ratio > 0.80

# Using fairlearn library:
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_ratio

constraint = DemographicParity()
mitigated_model = ExponentiatedGradient(
    LGBMClassifier(n_estimators=1000, max_depth=6),
    constraints=constraint,
    eps=0.05  # tolerance
)
mitigated_model.fit(X_train, y_train, sensitive_features=race_train)

# Results:
# AUC: 0.82 (dropped from 0.84 -- small accuracy cost for fairness)
# Approval rates:
#   White: 66%, Black: 60%, Hispanic: 62%, Asian: 70%
# DIR: Black/White = 60/66 = 0.91  (PASSES!)
# DIR: Hispanic/White = 62/66 = 0.94  (PASSES!)
# Default rate among approved: 6.8% (slightly higher -- the cost of fairness)

# Approach 3: Pre-processing (remove proxy information)
# Train on residualized features: for each feature, regress out its
# correlation with race, then use the residuals
# This removes the "race-ness" from income, credit score, etc.
# AUC: 0.80 (drops more -- removes some legitimate risk signal too)

# DECISION: Use Approach 2 (in-processing)
# Best balance of accuracy (AUC 0.82) and fairness (DIR > 0.90)
# Small cost: 0.5% higher default rate among approved loans
```

---

## Step 11: Adverse Action Reasons (Legal Requirement)

```python
# When denying a loan, MUST provide top reasons (ECOA requirement)
# SHAP values provide exactly this

# Denial example:
# Applicant: John Smith, declined with P(default) = 0.28
# SHAP analysis:
#   Reason 1: High debt-to-income ratio (0.52 vs avg 0.35)     -> +0.08 risk
#   Reason 2: 3 delinquencies in past 2 years                   -> +0.06 risk
#   Reason 3: Short credit history (18 months vs avg 84 months)  -> +0.04 risk
#   Reason 4: High number of recent inquiries (5 in 6 months)    -> +0.03 risk

# These become the adverse action notice:
# "Your application was declined because:
#  1. Ratio of debt to income is too high
#  2. Number of delinquent accounts
#  3. Length of credit history is too short
#  4. Too many recent inquiries"

# Each reason corresponds to a specific SHAP contribution
# This is legally defensible because each reason is:
# (a) factually true about the applicant
# (b) statistically predictive of default
# (c) not based on a protected characteristic

# Automated reason code generation:
def generate_adverse_action_reasons(shap_values, feature_names, top_n=4):
    """Generate legally compliant adverse action reasons from SHAP values."""
    # Map features to regulation-approved reason codes
    reason_code_map = {
        'debt_to_income_ratio': 'Ratio of debt payments to income is too high',
        'delinquencies_last_2yr': 'Number of accounts with delinquency',
        'credit_history_years': 'Length of credit history',
        'revolving_utilization': 'Proportion of revolving balances to credit limits',
        'inquiries_last_6mo': 'Number of recent inquiries',
        'public_records': 'Presence of derogatory public records',
        ...
    }
    # Sort features by SHAP contribution (most harmful first)
    sorted_features = sorted(zip(feature_names, shap_values), key=lambda x: -x[1])
    # Return top N reasons
    reasons = []
    for feat, shap_val in sorted_features[:top_n]:
        if shap_val > 0 and feat in reason_code_map:
            reasons.append(reason_code_map[feat])
    return reasons
```

---

## Step 12-14: Final Results

```python
# Test set: 6 months of applications (250,000 applicants)
#
# Model Performance:
# AUC-ROC:  0.82
# KS Stat:  0.48
# Gini:     0.64
#
# Fairness (all pass four-fifths rule):
# DIR (Black/White):    0.91
# DIR (Hispanic/White): 0.94
# DIR (Female/Male):    0.97
# DIR (Age <30 / 30-50): 0.88
#
# Business Impact:
# Old system default rate:  8.3%
# New model default rate:   6.8%  (with slightly higher approval rate!)
# Annual loss reduction from bad loans: $12.4M
# Maintained approval rate: 64% (vs 62% before -- more inclusive!)
#
# Regulatory Readiness:
# - Model documentation: 45-page report with methodology, validation, fairness tests
# - Adverse action reasons generated for every denial (SHAP-based)
# - Fair lending analysis: disparate impact ratios all > 0.85
# - Ongoing monitoring: monthly fairness reports by demographic group
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Protected features | Just remove race/gender | Remove + detect proxy variables + test for disparate impact |
| Selection bias | Train only on approved loans | Heckman correction for reject inference |
| Historical bias | Train on biased data, inherit bias | Fairness-constrained training (ExponentiatedGradient) |
| Fairness testing | Skip it | Four-fifths rule, equalized odds, demographic parity across all groups |
| Explainability | Feature importance | SHAP-based adverse action reason codes (legally required) |
| Zip codes | One-hot encode (race proxy!) | Replace with economically-justified zip-level features |
| Threshold | Single threshold for everyone | Considered per-group thresholds but chose in-processing (more defensible) |
| Evaluation | AUC only | AUC + fairness metrics + regulatory documentation |
| Business case | "Model is accurate" | Higher approval rate + lower default rate + regulatory compliance |
