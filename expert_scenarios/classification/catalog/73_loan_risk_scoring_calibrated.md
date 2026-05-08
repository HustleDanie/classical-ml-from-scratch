# Expert Scenario 73: Loan Risk Scoring (Calibrated Probability)

> **Complexity:** Calibrated probability is the deliverable, not just the rank — downstream pricing engine multiplies the probability by exposure dollars to set rates. Regulated under ECOA / FCRA. Moderate imbalance (~9% default). Calibration drifts as economic regime changes.

---

## The Brief

A consumer-lender gives you 2.4M historical loans (30-month observation window) with default outcomes. Their pricing engine computes loan APR as:

```
APR = baseline_rate + (predicted_default_probability * loss_given_default * margin_factor) + ops_cost
```

The pricing engine multiplies the probability by dollars. A model that predicts 0.30 when truth is 0.50 doesn't just rank wrong — it under-prices loans by 67% in expected loss terms. They want:

- A score (0-1000 range, FICO-like) that's **well-calibrated** — when the model says "probability 0.10", 10% of those loans actually default.
- Calibration must hold at every decile (not just on average).
- Regulated: ECOA-compliant adverse action codes per declined loan; quarterly fairness audit.
- Stable across economic regimes (recession-resilient enough to not collapse rate-pricing in a downturn).

This is harder than "predict default" because the bar is calibration, not just AUC. Most models with great AUC have terrible calibration. The technique stack here is specifically designed around calibration.

---

## Step 1: Define the Problem Type

```
Type:           Binary classification, calibrated probability output
Primary Metric: Brier score (calibration); Expected Calibration Error (ECE)
Secondary:      AUC-ROC (rank quality); KS statistic
                Per-decile calibration (not just average)
Business Goal:  ECE < 0.015 globally; per-decile predicted vs actual gap < 1.5pp
Constraint:     ECOA-compliant; quarterly fairness audit; calibration must hold
                across protected groups
Imbalance:      91/9 -- moderate
```

**Expert thinking:** the headline metric is Brier score, NOT AUC. AUC measures ranking ("is this loan riskier than that one?") while Brier measures calibration ("is the absolute probability right?"). For pricing, calibration matters. A model can rank well (good AUC) and still systematically over- or under-predict.

The secondary metric — per-decile calibration — is even tighter. We need the predicted probability to match the actual rate at each risk band.

---

## Step 2: Understand the Data

```
Shape: 2,400,000 rows x 64 features
Target: defaulted_within_30mo -- 0 (91%), 1 (9%)

Features:

APPLICANT (8):
- loan_id (unique)
- application_date (datetime, 2018-2023)
- age (21-85)
- income_annual ($12K - $1.2M)
- employment_status (Employed/Self-employed/Retired/Unemployed/Other)
- employment_length_years (0-50, 6% missing)
- has_homeowner (binary)
- num_open_credit_lines (int, 0-50)

CREDIT BUREAU (15):
- fico_score (300-850, 1% missing)
- credit_history_years (1-50, 1% missing)
- num_late_30d_24mo (int)
- num_late_60d_24mo (int)
- num_late_90d_24mo (int)
- num_collections (int)
- num_bankruptcies (int)
- credit_utilization_pct (float, 0-100+)
- num_inquiries_6mo (int)
- has_recent_charge_off (binary)
- has_foreclosure (binary)
- avg_credit_age_months (int)
- num_credit_cards (int)
- num_installment_loans (int)
- has_prior_loan_with_us (binary)

LOAN TERMS (10):
- loan_amount ($1K-$100K)
- loan_term_months (12, 24, 36, 60)
- loan_purpose (Debt_Consolidation/Home/Auto/Medical/Other)
- requested_apr_pct (float)
- monthly_payment_to_income_ratio (float)
- debt_to_income_ratio (float, 0-100)
- num_other_active_loans (int)
- has_cosigner (binary)
- collateral_value (float, 90% missing -- only for secured)
- loan_to_value_ratio (float, 90% missing)

VERIFIED INFO (8):
- income_verified (binary)
- employment_verified (binary)
- residence_verified (binary)
- bank_account_age_years (int)
- avg_bank_balance_3mo (float, 22% missing)
- num_overdrafts_3mo (int)
- has_direct_deposit (binary)
- ssn_match_score (float, 0-1)

GEOGRAPHIC / DEMOGRAPHIC (8):
- state (50)
- zip3 (HIPAA-safe)
- median_local_income (float)
- local_unemployment_rate (float)
- urban_or_rural

PROTECTED ATTRIBUTES (4):
- race (W/B/H/A/Other; ECOA-protected, 13% Declined)
- ethnicity (ECOA-protected)
- gender (ECOA-protected)
- age (already counted; ECOA-protected)

ECONOMIC REGIME (3):
- year (2018-2023)
- s&p_index_at_application
- recession_indicator (NBER definition)

OUTCOME (target):
- defaulted_within_30mo (target)
```

**Expert thinking:** key signals are FICO + credit history + DTI + monthly_payment_to_income_ratio. Protected attributes (race, ethnicity, gender) cannot be used as features under ECOA. Age is also protected. Geographic features (state, zip3) can leak demographic information ("redlining proxy") and need fairness audit.

---

## Step 3: EDA

```python
df['defaulted_within_30mo'].value_counts(normalize=True)
# 0    0.913
# 1    0.087

# Default rate by FICO band
df['fico_band'] = pd.cut(df['fico_score'], bins=[300, 580, 620, 670, 740, 800, 850])
df.groupby('fico_band')['defaulted_within_30mo'].mean()
# (300, 580]:  0.301   <- "deep subprime"
# (580, 620]:  0.184   <- "subprime"
# (620, 670]:  0.108   <- "near-prime"
# (670, 740]:  0.052   <- "prime"
# (740, 800]:  0.018   <- "super-prime"
# (800, 850]:  0.005

# Default rate by economic regime
df.groupby('year')['defaulted_within_30mo'].mean()
# 2018: 0.082
# 2019: 0.075
# 2020: 0.124  <- COVID downturn
# 2021: 0.069
# 2022: 0.078
# 2023: 0.092
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Default rate spans 0.5% (super-prime) to 30% (deep subprime) | Calibration must hold across this 60x range |
| 2020 default rate 12.4% vs 2019's 7.5% | Economic regime is a confounder; recent data more relevant for current pricing |
| `monthly_payment_to_income_ratio > 0.30` → 16% default | Strong threshold effect |
| `num_inquiries_6mo >= 5` → 22% default | Hard pull frequency is a known signal |
| Self-employed → 11.4% default vs Employed → 7.8% | Employment type matters even controlling for income |
| Unverified income loans → 14% default vs verified → 6.8% | Verification status materially shifts rate |

**Expert insight:** the 2020 spike from COVID shows recession-time calibration must be robust. A model trained equally on 2018-2023 will be miscalibrated in any non-typical year. Economic regime features (recession_indicator, s&p_index) help the model adjust expected default rates.

---

## Step 4: Data Cleaning

```python
# Drop loan_id from features
# Keep ALL protected attributes for FAIRNESS AUDIT, but drop them before training:
protected = ['race', 'ethnicity', 'gender']
df_protected = df[protected]  # save aside

# FICO: 1% missing; impute with median by employment_status
df['fico_score'] = df.groupby('employment_status')['fico_score'].transform(lambda x: x.fillna(x.median()))

# employment_length_years: 6% missing
df['employment_length_missing'] = df['employment_length_years'].isnull().astype(int)
df['employment_length_years'] = df['employment_length_years'].fillna(df['employment_length_years'].median())

# Loan-secured features: 90% missing (only secured loans have collateral)
df['is_secured'] = df['collateral_value'].notna().astype(int)
df['collateral_value'] = df['collateral_value'].fillna(0)
df['loan_to_value_ratio'] = df['loan_to_value_ratio'].fillna(0)

# avg_bank_balance_3mo: 22% missing
df['has_bank_data'] = df['avg_bank_balance_3mo'].notna().astype(int)
df['avg_bank_balance_3mo'] = df['avg_bank_balance_3mo'].fillna(0)

# Outliers in income: extreme high earners present (true)
# But cap at 99.9th percentile for stability:
df['income_annual'] = df['income_annual'].clip(upper=df['income_annual'].quantile(0.999))
```

---

## Step 5: Feature Engineering

```python
# === FINANCIAL HEALTH RATIOS ===
df['payment_burden'] = df['loan_amount'] * (df['requested_apr_pct'] / 1200) / (df['income_annual'] / 12)
df['log_income'] = np.log1p(df['income_annual'])
df['log_loan_amount'] = np.log1p(df['loan_amount'])
df['loan_to_income_ratio'] = df['loan_amount'] / (df['income_annual'] + 1)

# === CREDIT BEHAVIOR COMPOSITE ===
df['credit_distress_score'] = (
    df['num_late_30d_24mo'] * 1 +
    df['num_late_60d_24mo'] * 2 +
    df['num_late_90d_24mo'] * 3 +
    df['num_collections'] * 4 +
    df['has_foreclosure'].astype(int) * 5 +
    df['num_bankruptcies'] * 8
)

df['recent_credit_seeking'] = (df['num_inquiries_6mo'] > 3).astype(int)
df['high_utilization'] = (df['credit_utilization_pct'] > 75).astype(int)

# === ECONOMIC REGIME FEATURES ===
df['rate_environment'] = df['year'] - 2018  # ordinal time index
df['recession_period'] = df['recession_indicator']

# === VERIFICATION SCORE ===
df['verification_score'] = (
    df['income_verified'].astype(int) +
    df['employment_verified'].astype(int) +
    df['residence_verified'].astype(int) +
    df['has_direct_deposit'].astype(int)
)

# === LOAN PURPOSE RISK (target encoding with smoothing on TRAIN ONLY) ===
purpose_default_rate = df_train.groupby('loan_purpose')['defaulted_within_30mo'].mean()
purpose_volume = df_train.groupby('loan_purpose').size()
overall = df_train['defaulted_within_30mo'].mean()
alpha = 1000
df['purpose_default_smoothed'] = df['loan_purpose'].map(
    (purpose_volume * purpose_default_rate + alpha * overall) / (purpose_volume + alpha)
).fillna(overall)
```

---

## Step 6: Feature Selection

```python
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

# Top 15:
#   fico_score                     0.094
#   credit_distress_score          0.082
#   debt_to_income_ratio           0.071
#   monthly_payment_to_income_ratio 0.068
#   credit_utilization_pct         0.054
#   loan_to_income_ratio           0.041
#   num_inquiries_6mo              0.038
#   verification_score             0.034
#   loan_amount                    0.029
#   employment_status              0.024
#   payment_burden                 0.022
#   credit_history_years           0.018
#   recent_credit_seeking          0.015
#   recession_period               0.012
#   purpose_default_smoothed       0.011
```

Drop features with MI < 0.001. Final: 53 features.

---

## Step 7: Preprocessing — standard pipeline (skipped for brevity)

---

## Step 8: Train/Test Split

```python
# TIME-BASED SPLIT plus economic regime stratification
df = df.sort_values('application_date')

# Train: 2018-Q1 to 2022-Q2 (4.5 years, includes COVID period)
# Validation: 2022-Q3 to 2022-Q4
# Test: 2023-Q1 to 2023-Q2 (most recent, never seen)

# Hold-out: 2023-Q3 to 2023-Q4 for final fairness audit and final calibration
```

---

## Step 9: Baseline

```python
# Baseline 1: predict global default rate (0.087) for everyone
# Brier score: 0.079
# AUC: 0.500
# Useless ranking, decent calibration

# Baseline 2: simple linear -- 1/(1+exp(0.05 * (700 - FICO)))
# Brier score: 0.063
# AUC: 0.74
# Calibration: poor (overshoots subprime, undershoots prime)
```

---

## Step 10: Try Multiple Models with TimeSeries CV

```python
# === Model 1: Logistic Regression with L2 ===
lr = LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', max_iter=500)
# AUC: 0.764, Brier: 0.054, ECE: 0.031

# === Model 2: Logistic Regression WITHOUT class_weight (more honest probabilities) ===
lr_unweighted = LogisticRegression(C=1.0, penalty='l2', max_iter=500)
# AUC: 0.762, Brier: 0.045, ECE: 0.011  <- much better calibration!

# === Model 3: XGBoost with monotonic constraints ===
xgb = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    monotone_constraints={'fico_score': -1, 'credit_distress_score': 1, ...}
)
# AUC: 0.794, Brier: 0.061, ECE: 0.058  <- great AUC, terrible calibration

# === Model 4: XGBoost + Isotonic calibration ===
from sklearn.calibration import CalibratedClassifierCV
cal_xgb = CalibratedClassifierCV(xgb, method='isotonic', cv=5)
# AUC: 0.794, Brier: 0.041, ECE: 0.013  <- best of both worlds

# === Model 5: Logistic Regression + Platt scaling (sanity check) ===
cal_lr = CalibratedClassifierCV(lr_unweighted, method='sigmoid', cv=5)
# AUC: 0.762, Brier: 0.044, ECE: 0.010
```

**Top performer:** Calibrated XGBoost (AUC 0.794, Brier 0.041, ECE 0.013).

**Critical insight:** the unweighted Logistic Regression has BETTER calibration than the class-weighted version. `class_weight='balanced'` distorts probabilities — it makes the model output what it would have if classes were balanced, not the true probability. For calibration, NEVER use class_weight unless you also recalibrate.

---

## Step 11: Imbalance Handling — Subtle for Calibrated Models

```python
# Standard imbalance tricks BREAK calibration:
#   - class_weight: produces over-confident probabilities (model thinks each class is 50%)
#   - SMOTE: synthesizes positives, distorts the actual rate
#   - threshold tuning: changes labels, not probabilities

# For CALIBRATED probability output:
#   1. Train without class weights (use raw class proportions)
#   2. Apply isotonic calibration on a held-out fold
#   3. Validate calibration on test set per decile
```

**Expert insight:** this is the most subtle part of calibrated modeling. People reach for `class_weight='balanced'` automatically — but when calibration matters, it's wrong. Train honest, then calibrate post-hoc.

---

## Step 12: Hyperparameter Tuning — Optimizing Brier, Not AUC

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 6),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
    }
    model = XGBClassifier(**params, monotone_constraints={...}, random_state=42)
    cal = CalibratedClassifierCV(model, method='isotonic', cv=5)
    cal.fit(X_train, y_train)
    proba = cal.predict_proba(X_val)[:, 1]
    return brier_score_loss(y_val, proba)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=80)
# Best Brier: 0.039
```

---

## Step 13: Calibration Validation Per Decile

```python
# CRITICAL CHECK: per-decile predicted vs actual default rate
y_proba = cal_xgb.predict_proba(X_test)[:, 1]
df_test['predicted_proba'] = y_proba
df_test['decile'] = pd.qcut(y_proba, 10, labels=range(1, 11))

calibration_per_decile = df_test.groupby('decile').agg(
    predicted_avg=('predicted_proba', 'mean'),
    actual_rate=('defaulted_within_30mo', 'mean'),
    count=('predicted_proba', 'count')
)

# Decile  Predicted   Actual    Gap
# 1       0.012       0.011    +0.001    ✓
# 2       0.025       0.022    +0.003    ✓
# 3       0.041       0.038    +0.003    ✓
# 4       0.058       0.054    +0.004    ✓
# 5       0.074       0.071    +0.003    ✓
# 6       0.094       0.092    +0.002    ✓
# 7       0.121       0.118    +0.003    ✓
# 8       0.158       0.151    +0.007    ✓
# 9       0.218       0.226    -0.008    ✓
# 10      0.348       0.354    -0.006    ✓

# All deciles within 1pp gap.
# Max gap at decile 8-10: model slightly under-predicts highest-risk loans
# Acceptable; conservative for pricing (we'd rather overprice than underprice high-risk)
```

---

## Step 14: Score Conversion (1-1000 like FICO)

```python
# Pricing engine wants a score, not a probability
# Convert: score = round(1000 * (1 - probability))
# So a 1% default probability → score 990, a 30% default probability → score 700

df_test['risk_score'] = (1000 * (1 - df_test['predicted_proba'])).round().astype(int)

# Set decision bands (negotiated with credit policy team):
# Score >= 800: AUTO-APPROVE at base rate
# Score 700-799: APPROVE with risk-based pricing
# Score 600-699: APPROVE only if compensating factors (cosigner, secured)
# Score < 600: DECLINE
```

---

## Step 15: Adverse Action Codes

```python
# ECOA requires that any decline include "specific principal reasons" -- adverse action codes
# Generate from SHAP values per declined application

import shap
explainer = shap.TreeExplainer(cal_xgb.calibrated_classifiers_[0].estimator)

for idx in declined_applications:
    contributions = shap_values[idx]
    top_3 = np.argsort(contributions)[-3:][::-1]  # most negative impact on score
    reasons = []
    for fi in top_3:
        feat = feature_names[fi]
        # Map feature to ECOA reason code from the bank's catalog
        reason = feature_to_reason_code(feat)
        reasons.append(reason)
    # Adverse action notice includes top 3 reasons:
    # "1. Insufficient credit history (Reason 02)
    #  2. Number of recent credit inquiries (Reason 04)
    #  3. Debt-to-income ratio too high (Reason 11)"
```

---

## Step 16: Fairness Audit

```python
# ECOA prohibits discrimination on protected attributes
# Audit BOTH model performance AND outcome rates by protected group

for group in ['race', 'ethnicity', 'gender']:
    for value in df_test[group].unique():
        mask = (df_test[group] == value)
        if mask.sum() < 200: continue
        # Performance audit
        auc = roc_auc_score(y_test[mask], y_proba[mask])
        brier = brier_score_loss(y_test[mask], y_proba[mask])
        # Outcome audit
        approval_rate = (df_test.loc[mask, 'risk_score'] >= 600).mean()
        decline_rate = 1 - approval_rate
        # Calibration audit per decile within group
        ece_within = expected_calibration_error(y_test[mask], y_proba[mask])
        print(f"{group}={value}: AUC={auc:.3f}, Brier={brier:.4f}, decline={decline_rate:.1%}, ECE={ece_within:.4f}")

# Findings:
#   race=W:  AUC 0.802, Brier 0.039, decline 18.4%
#   race=B:  AUC 0.781, Brier 0.044, decline 27.2%   <- gap to investigate
#   race=H:  AUC 0.778, Brier 0.043, decline 24.8%
#   race=A:  AUC 0.795, Brier 0.040, decline 19.1%

# Gap drivers:
#   1. Lower-FICO distribution among Black applicants (population-level effect)
#   2. Geographic features (zip3) acting as race proxy
#   3. Unequal underwriting history in our own data
#
# Mitigations (regulatory):
#   1. Drop zip3 (potential redlining proxy) and rerun -- verify decline rate gap narrows
#   2. Per-group calibration check -- ensure model doesn't OVER-predict default for any group
#   3. Document Disparate Impact ratio (Black approval / White approval); flag if < 0.80
```

**Expert insight:** disparities in OUTCOMES (decline rates) across protected groups don't automatically prove model bias — they may reflect real differences in financial situations driven by historical inequality. But they trigger heightened regulatory scrutiny: the bank must show the model's predictions are CALIBRATED for each group AND that excluding protected proxies (zip3) doesn't materially worsen disparity.

---

## Step 17: Final Evaluation on Held-Out Test Set

```python
# Final model: Calibrated XGBoost (monotonic constraints + isotonic calibration)
# Test set: ~150K loans from 2023 H2

# Performance:
#   AUC-ROC:                     0.798
#   Brier score:                 0.041     ✓ (target < 0.045)
#   Expected Calibration Error:  0.011     ✓ (target < 0.015)
#   KS statistic:                0.42

# Per-decile gap (predicted - actual):
#   Max gap: 0.008 (decile 9)
#   Mean gap: 0.003

# Fairness (Disparate Impact):
#   Black approval / White approval:  0.83  ✓ (target >= 0.80)
#   Hispanic approval / White:        0.91  ✓
#   Gender-female / Gender-male:      1.02  ✓
```

---

## Step 18: Deployment

```python
# Production: real-time loan approval API
# Latency: ~150ms per loan (XGBoost + isotonic + SHAP + reason codes)

# Quarterly retraining:
#   - Drop oldest quarter from training data; add new quarter
#   - Recompute isotonic calibration on most recent 4 quarters
#   - Re-audit fairness across protected groups
#   - Re-validate per-decile calibration
#   - Submit fairness report to compliance team

# Monitoring:
#   - Daily Brier score on closed loans (truth available 30+ months later)
#   - Weekly calibration drift per decile (alert if any decile shifts > 1.5pp)
#   - Monthly disparate impact ratio (alert if < 0.80)
#   - Recession indicator: if NBER recession declared, immediate retrain

# Documentation for audit:
#   - Model card (FAQ-style with performance + fairness)
#   - Risk scoring SOP (when to use, when not to)
#   - Adverse action mapping (feature -> ECOA reason code)
#   - Quarterly performance report submitted to compliance
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Metric | AUC | Brier score (calibration); per-decile gap |
| Imbalance | class_weight='balanced' | UNWEIGHTED training + isotonic calibration |
| Calibration | Use raw model output | Isotonic on held-out fold; validate per decile |
| Threshold | F1 optimum | Score bands negotiated with credit policy team |
| Train/test split | Random | Time-based with COVID-period preserved in training |
| Protected attributes | Drop them | Drop from training; KEEP for fairness audit; check proxy leakage from zip3 |
| Adverse action codes | "model said decline" | SHAP top-3 mapped to ECOA reason codes per declined app |
| Fairness | Ignore | Per-group AUC, Brier, decline rate, calibration; Disparate Impact ratio |
| Regulated path | Train and ship | Quarterly retrain + fairness audit + compliance submission |
| Monitoring | None | Daily Brier on closed loans; weekly calibration drift; monthly DI |
| Score format | Probability 0-1 | FICO-style 1000-down score (industry convention) |
| Recession robustness | Train on whole history | Include recession indicator + economic regime features |
