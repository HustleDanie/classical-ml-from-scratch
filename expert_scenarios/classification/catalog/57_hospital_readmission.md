# Expert Scenario 57: 30-Day Hospital Readmission Risk

> **Complexity:** Mixed-type tabular (numeric labs + categorical diagnoses + free-text notes), informative missingness, regulated under HIPAA, real-time at discharge, 11% positive rate, fairness scrutiny across DRG and demographic groups.

---

## The Brief

A 1,400-bed hospital system gives you 100,000 historical inpatient discharges from the past 4 years. Your task: predict at discharge whether each patient will be readmitted within 30 days. 11% of patients are readmitted. CMS penalizes the hospital ~$8 million annually for excess readmissions on selected DRGs (heart failure, pneumonia, COPD, hip/knee replacement, AMI). The hospital wants:

- A score available within seconds of discharge so the care team can adjust the discharge plan.
- Top 5 modifiable risk factors per patient (so they know what to act on).
- Calibrated probabilities (downstream cost models multiply by intervention costs).
- Compliance-ready: explainable, audited, and fair across racial / DRG groups.

This is harder than typical churn prediction because: outcomes interact with care processes, missing labs are sometimes informative ("not ordered" implies "not concerning"), DRG codes are 700+ high-cardinality categoricals, and federal regulations (HIPAA + CMS) constrain both the modeling and the deployment.

---

## Step 1: Define the Problem Type

```
Type:           Binary Classification (readmit_30d = 1, no_readmit = 0)
Primary Metric: AUC-ROC; per-DRG calibration error
Secondary:      Recall at fixed precision (intervention budget); per-group AUC
Business Metric: Excess readmission ratio (CMS Hospital Readmission Reduction Program)
Constraint:     Real-time at discharge (< 5 sec); interpretable per-patient; HIPAA-compliant
Imbalance:      89/11 (moderate) -- class_weight + threshold tuning
```

**Expert thinking:** AUC matters because the use case is *ranking* (which patients should the discharge planner spend extra time with?). Calibration matters because downstream the score gets multiplied by intervention cost ($800 per high-touch discharge plan) — wrong calibration means wrong intervention budget allocation.

---

## Step 2: Understand the Data

```
Shape: 100,000 rows x 87 columns (after some pre-flattening of EHR)
Target: 'readmit_30d' -- 0 (89%), 1 (11%)

Features (87 total) -- categorized:

DEMOGRAPHICS (9):
- patient_id (string, unique)
- age (int, 18-105)
- sex (M/F/Other)
- race (W, B, H, A, O, Unknown)
- ethnicity (Hispanic/Non-Hispanic/Unknown)
- preferred_language (60+ codes)
- marital_status
- zip3 (first 3 digits of zip; full zip is PHI)
- insurance_type (Medicare/Medicaid/Commercial/Self-pay/Other)

ADMISSION (8):
- admission_date (datetime)
- admission_type (Emergency/Urgent/Elective/Transfer)
- admission_source (ED/Outpatient/Transfer/SNF)
- los_days (length of stay this admission, 1-90)
- num_prev_admits_12m (int, 0-25+)
- num_ed_visits_12m (int)
- weekend_admit (binary)
- night_admit (binary)

CLINICAL (32 -- the heart of the model):
- primary_drg_code (700+ unique values)
- num_secondary_diagnoses (int, 1-25)
- charlson_comorbidity_index (int, 0-15)
- has_diabetes, has_chf, has_copd, has_ckd, has_cad, has_dementia (binary, derived from ICDs)
- num_medications_at_discharge (int)
- num_high_risk_meds (warfarin, opioids, insulin) (int)
- abnormal_labs_at_discharge (int, 0-15)
- albumin (float, 11% missing)
- creatinine (float, 4% missing)
- hemoglobin (float, 6% missing)
- sodium (float, 3% missing)
- potassium (float, 3% missing)
- bnp (float, 38% missing -- ordered for CHF concern)
- troponin (float, 41% missing -- cardiac concern)
- procalcitonin (float, 47% missing -- infection)
- ... 18 more lab values

OPERATIONAL (8):
- discharge_disposition (Home/Home_Health/SNF/Rehab/Hospice/AMA)
- num_consults (int)
- icu_los_days (float, 0 if no ICU)
- ventilator_days (int)
- surgery_during_admit (binary)
- transfusion (binary)
- catheter_days (int)
- icu_admit (binary)

SOCIAL DETERMINANTS (15):
- has_pcp_followup_scheduled (binary)
- days_to_pcp_followup (float, NaN if no followup)
- lives_alone (binary, from social work assessment)
- has_caregiver (binary)
- food_insecurity_screen_positive (binary, 60% missing -- only screened for high-risk)
- transportation_barrier (binary)
- prior_homelessness (binary)
- substance_use_disorder (binary)
- mental_health_dx (binary)
- ... 6 more social determinants
```

**Expert thinking:** the laboratory missingness pattern is critical:
- BNP missing 38%: this is ordered when CHF is suspected. Missing means "doctor wasn't worried about heart failure." The MISSING is a signal.
- Troponin missing 41%: same logic for cardiac concern.
- Albumin missing 11%: more random; missing because not ordered routinely.

Treating all missingness identically (e.g., median imputation) destroys real signal in the lab columns.

---

## Step 3: Exploratory Data Analysis (EDA)

```python
df['readmit_30d'].value_counts(normalize=True)
# 0    0.892
# 1    0.108

# Readmit rate by primary DRG (top 5 by volume + top 5 by readmit rate)
df.groupby('primary_drg_code')['readmit_30d'].agg(['count', 'mean']).sort_values('mean', ascending=False).head(10)

# Top readmit rates:
# DRG 871 (Septicemia w/o MV)        18.3%
# DRG 291 (Heart Failure w/o CC)     17.2%
# DRG 190 (COPD w/o CC)              15.8%
# DRG 689 (Kidney/UTI w/o CC)        14.9%
# DRG 313 (Chest pain)               13.4%
# Average:                           10.8%
```

**Findings:**

| Finding | Implication |
|---------|------------|
| 5 DRGs (heart failure, COPD, sepsis, pneumonia, kidney) account for 23% of admits but 41% of readmits | DRG-aware modeling matters; per-DRG calibration likely diverges |
| Charlson Index >= 4 → 21% readmit; <=1 → 5% readmit | Strong signal; non-linear above 4 |
| Albumin < 3.0 g/dL → 19% readmit | Malnutrition / chronic disease marker |
| BNP measured AND > 500 → 24% readmit | When BNP IS measured and high, very predictive |
| BNP NOT measured → 8% readmit | Missing means "no CHF concern" -- LOWER risk |
| `discharge_disposition='AMA'` (left against medical advice) → 31% readmit | Strong but rare (1.2%) |
| Patients with PCP followup scheduled → 8.4% readmit; without → 17.6% | The biggest *modifiable* lever |
| Lives alone + age > 75 → 19% readmit | Social determinants compound clinical risk |
| Race "Black" → 13.2% readmit; "White" → 10.1% | Disparity present; potentially confounded by SES → fairness audit needed |

**Expert insight:** the social determinants (PCP followup, lives_alone, food_insecurity) are not just predictive but ACTIONABLE. A high readmit risk that's driven by "no PCP followup scheduled" can be intervened on at discharge. A high risk driven by "Charlson 7" cannot. The model should surface modifiable factors separately.

---

## Step 4: Data Cleaning

```python
# Drop PHI / non-features
drop_cols = ['patient_id']  # unique ID
# Keep zip3 -- not full zip, OK under HIPAA Safe Harbor

# Lab missingness -- INFORMATIVE for some, RANDOM for others
# Strategy: for each lab, create _missing flag, then impute
informative_missing_labs = ['bnp', 'troponin', 'procalcitonin']
random_missing_labs = ['albumin', 'creatinine', 'hemoglobin', 'sodium', 'potassium']

for col in informative_missing_labs:
    df[f'{col}_measured'] = df[col].notna().astype(int)
    df[col] = df[col].fillna(df[col].median())
    # The _measured flag carries the "did we order this?" signal
    # The actual value, when imputed, is just the median (which has weak signal alone)
    # The model uses BOTH features

for col in random_missing_labs:
    df[col] = df[col].fillna(df.groupby('primary_drg_code')[col].transform('median'))
    # Impute by DRG group -- "what's the median albumin for this kind of admission?"
    # Falls back to overall median if DRG group has all-missing

# Social determinants missingness
# food_insecurity_screen_positive 60% missing -- only screened for high-risk
df['food_insecurity_screened'] = df['food_insecurity_screen_positive'].notna().astype(int)
df['food_insecurity_screen_positive'] = df['food_insecurity_screen_positive'].fillna(0).astype(int)

# days_to_pcp_followup NaN if no followup scheduled
df['has_pcp_followup_scheduled'] = (df['days_to_pcp_followup'].notna()).astype(int)
df['days_to_pcp_followup'] = df['days_to_pcp_followup'].fillna(60)  # 60 = "not scheduled"

# Outliers: don't cap los_days, num_medications, etc -- extremes are high-risk patients
# DO clip at 99.5 percentile for numeric stability of linear models:
for col in ['los_days', 'num_medications_at_discharge', 'num_secondary_diagnoses']:
    df[col] = df[col].clip(upper=df[col].quantile(0.995))
```

**Expert insight:** for clinical labs, the rule is "informative missingness for ordered-on-suspicion tests; random missingness for routine tests." Encode this distinction explicitly. The `_measured` flag often has higher mutual information than the lab value itself.

---

## Step 5: Feature Engineering

```python
# === DRG handling: 700+ codes -- target encode, not one-hot ===
# Compute on training fold only inside CV
drg_readmit_rate = df_train.groupby('primary_drg_code')['readmit_30d'].mean()
drg_volume = df_train.groupby('primary_drg_code').size()
df['drg_readmit_rate'] = df['primary_drg_code'].map(drg_readmit_rate).fillna(0.108)
df['drg_volume'] = df['primary_drg_code'].map(drg_volume).fillna(1)
# Smooth low-volume DRGs toward overall mean (Bayesian smoothing)
alpha = 100  # equivalent of "100 prior observations"
df['drg_readmit_smoothed'] = (
    (df['drg_volume'] * df['drg_readmit_rate'] + alpha * 0.108) /
    (df['drg_volume'] + alpha)
)

# === HOSPITAL UTILIZATION FEATURES ===
df['admits_per_year'] = df['num_prev_admits_12m']  # already aligned
df['ed_to_admit_ratio'] = df['num_ed_visits_12m'] / (df['num_prev_admits_12m'] + 1)
df['high_utilizer'] = (df['num_prev_admits_12m'] >= 3).astype(int)

# === COMORBIDITY INTERACTIONS ===
df['multi_chronic'] = (df['has_diabetes'] + df['has_chf'] + df['has_copd'] + df['has_ckd']) >= 2
df['multi_chronic'] = df['multi_chronic'].astype(int)
df['frailty_proxy'] = (df['age'] >= 75).astype(int) * df['multi_chronic']

# === MEDICATION COMPLEXITY ===
df['polypharmacy'] = (df['num_medications_at_discharge'] >= 10).astype(int)
df['high_risk_med_ratio'] = df['num_high_risk_meds'] / (df['num_medications_at_discharge'] + 1)

# === FOLLOWUP RISK ===
df['no_followup_high_risk'] = (
    (df['has_pcp_followup_scheduled'] == 0) &
    (df['drg_readmit_smoothed'] > 0.15)
).astype(int)
# Patients with high baseline DRG risk AND no scheduled followup

# === SOCIAL VULNERABILITY SCORE ===
df['social_vuln_score'] = (
    df['lives_alone'].astype(int) +
    (1 - df['has_caregiver'].astype(int)) +
    df['food_insecurity_screen_positive'].astype(int) +
    df['transportation_barrier'].astype(int) +
    df['prior_homelessness'].astype(int)
)
# Range 0-5; higher = more vulnerable

# === DISCHARGE DISPOSITION RISK ===
disposition_risk = {'Home': 0.10, 'Home_Health': 0.13, 'SNF': 0.16,
                     'Rehab': 0.12, 'Hospice': 0.04, 'AMA': 0.31}
df['disposition_risk'] = df['discharge_disposition'].map(disposition_risk).fillna(0.108)
```

**Expert insight:** the engineered "modifiable risk" features (`no_followup_high_risk`, `social_vuln_score`) directly enable the actionable explanation the brief asked for. When the model fires on a patient, we can show: "this patient is high risk AND has no followup scheduled — schedule one."

---

## Step 6: Feature Selection

```python
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

# Top features:
#   drg_readmit_smoothed        0.038
#   num_prev_admits_12m         0.031
#   charlson_comorbidity_index  0.029
#   has_pcp_followup_scheduled  0.024
#   bnp_measured                0.022   <- the "missing" flag!
#   albumin                     0.019
#   discharge_disposition       0.018
#   social_vuln_score           0.016
#   los_days                    0.015

# Bottom features (MI < 0.001):
#   marital_status              0.0008  -> drop
#   weekend_admit               0.0005  -> drop
#   night_admit                 0.0003  -> drop
#   preferred_language (60+ values, sparse)  0.0002  -> drop after one-hot
```

Drop 4 low-MI features and high-cardinality `preferred_language`. Final: 78 features after one-hot.

---

## Step 7: Preprocessing

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric = [...]  # ~35 numeric (lab values, counts, derived ratios)
categorical_low_card = ['sex', 'race', 'ethnicity', 'admission_type',
                         'admission_source', 'discharge_disposition', 'insurance_type']
categorical_high_card = []  # primary_drg_code is now target-encoded, not one-hot
binary = [...]  # ~30 binary flags

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_low_card),
    ('bin', 'passthrough', binary)
])
```

---

## Step 8: Train/Test Split

```python
# TIME-BASED SPLIT
df = df.sort_values('admission_date')
cutoff = '2024-01-01'  # all admits before; last 12 months as test
train = df[df['admission_date'] < cutoff]   # ~78,000 rows
test = df[df['admission_date'] >= cutoff]   # ~22,000 rows

# Why time-based: care patterns change. Pandemic. New CMS rules. Drug formularies.
# A model trained on randomly-mixed time data leaks future practices.
```

**Expert insight:** never use random split for healthcare data when models will be deployed prospectively. Care delivery changes, formularies change, billing codes change. Time-based split simulates production reality.

---

## Step 9: Baseline

```python
# Baseline 1: predict majority (no readmit)
# Accuracy: 89.2%, Recall: 0%, AUC: 0.50
# Useless.

# Baseline 2: simple rule -- flag if num_prev_admits_12m >= 3 OR Charlson >= 5
# Recall: 38%, Precision: 32%, AUC: 0.71
# Beats nothing-model. ML target.
```

---

## Step 10: Try Multiple Models with Stratified 5-Fold CV

```python
from sklearn.model_selection import StratifiedKFold

# Model 1: Logistic Regression with L2 (regulated baseline)
# CV AUC: 0.762, recall@30%fpr: 0.66

# Model 2: Logistic Regression with L1
# CV AUC: 0.758, recall@30%fpr: 0.65

# Model 3: Random Forest (n=300, max_depth=10)
# CV AUC: 0.787, recall@30%fpr: 0.71

# Model 4: Gradient Boosting (n=200, max_depth=4)
# CV AUC: 0.796, recall@30%fpr: 0.73

# Model 5: XGBoost (with monotonic constraints on actionable features)
# CV AUC: 0.798, recall@30%fpr: 0.74

# Model 6: LightGBM
# CV AUC: 0.797, recall@30%fpr: 0.73

# Model 7: SVM with RBF -- TIMED OUT after 2 hours on 78K rows. Skip.

# Model 8: KNN (k=15)
# CV AUC: 0.731 -- distances unreliable in 78-feature space
```

**Top performers:** XGBoost (0.798), LightGBM (0.797), Gradient Boosting (0.796).
**Regulated baseline:** Logistic Regression with L2 (0.762).
**Gap:** ensemble models are 3.5% AUC ahead of LR — material in healthcare.

---

## Step 11: Imbalance + Calibration

```python
# Imbalance: 89/11 -- moderate
# class_weight='balanced' for LR / RF
# scale_pos_weight = (1-0.108)/0.108 ~= 8.3 for XGBoost / LightGBM

# CALIBRATION (critical for downstream cost models)
# XGBoost / LightGBM raw probabilities are NOT calibrated
# Apply isotonic regression on a held-out calibration fold

from sklearn.calibration import CalibratedClassifierCV

calibrated_xgb = CalibratedClassifierCV(
    XGBClassifier(scale_pos_weight=8.3, ...),
    method='isotonic',
    cv=5
)
calibrated_xgb.fit(X_train, y_train)
# Brier score before calibration: 0.083
# Brier score after calibration:  0.078
# Expected calibration error before: 0.034
# Expected calibration error after:  0.012
```

**Expert insight:** if the downstream system multiplies the probability by intervention cost, miscalibration becomes wrong dollars. A model that predicts 0.30 but is actually 0.50 over-allocates intervention budget by 67%. Always calibrate when probabilities feed downstream cost logic.

---

## Step 12: Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

param_dist_xgb = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'min_child_weight': [5, 10, 20],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0.1, 1.0, 5.0]
}

search = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=8.3, random_state=42),
    param_dist_xgb, n_iter=80, cv=StratifiedKFold(5),
    scoring='roc_auc', n_jobs=-1
)
# Best: n_estimators=500, max_depth=4, lr=0.05, subsample=0.8, ...
# Tuned CV AUC: 0.811
```

---

## Step 13: Threshold Tuning to Intervention Budget

```python
# Hospital can afford ~3,500 high-touch discharge plans per year out of 25K discharges
# That's 14% of discharges -- the top 14% by predicted risk

y_proba = calibrated_xgb.predict_proba(X_test)[:, 1]
threshold = np.quantile(y_proba, 0.86)  # top 14%
# threshold ~ 0.28 (i.e., predict positive when P >= 0.28)

# At threshold 0.28:
#   Recall: 0.66 (catches 66% of actual readmits)
#   Precision: 0.50
#   Top-14% lift: 4.7x base rate
#   Annual readmits prevented at 50% intervention efficacy: ~770 readmits avoided
#   CMS penalty avoided: ~$3.5M
#   Intervention cost: 3,500 * $800 = $2.8M
#   Net benefit: ~$700K + better outcomes
```

**Expert insight:** threshold tuning to *budget*, not to *F1*, is the right framing for resource-constrained interventions. F1 optimizes a trade-off the operations team didn't ask for.

---

## Step 14: Fairness Audit

```python
# CMS scrutinizes readmission rates and intervention allocation across protected groups
# Audit AUC and threshold-positive rate by race / sex / insurance

for group in ['race', 'sex', 'insurance_type']:
    for value in df[group].unique():
        mask = (X_test[group] == value)
        if mask.sum() < 100: continue
        auc = roc_auc_score(y_test[mask], y_proba[mask])
        flag_rate = (y_proba[mask] >= 0.28).mean()
        true_rate = y_test[mask].mean()
        print(f"{group}={value}: AUC {auc:.3f}, flagged {flag_rate:.1%}, true {true_rate:.1%}")

# Findings:
# race=W:  AUC 0.812, flagged 14.1%, true rate 10.1%
# race=B:  AUC 0.794, flagged 17.8%, true rate 13.2%
# race=H:  AUC 0.788, flagged 13.0%, true rate 11.5%
# race=A:  AUC 0.772, flagged 11.2%, true rate  9.3%

# Black patients flagged at 17.8% (above their 13.2% true rate -- over-flagging)
# Asian AUC notably lower (0.772 vs overall 0.811) -- worse model performance for this group
```

**Expert insight:** flagging rates that exceed true rates by group are a fairness flag. Two responses:
1. Adjust thresholds per group (reduces disparity but is legally risky in some jurisdictions).
2. Investigate WHY the model over-flags Black patients (is it the social vulnerability score? insurance proxy?) — and remove proxies if they're driving disparity beyond clinical reality.

This audit is non-negotiable for a hospital deployment under CMS scrutiny.

---

## Step 15: Final Evaluation on Held-Out Test Set

```python
# Final model: XGBoost (calibrated isotonically), threshold 0.28
# Test set: 22,000 admissions, 11.4% readmit rate

# Performance:
#   AUC-ROC:           0.811
#   AUC-PR:            0.41
#   Recall:            0.66
#   Precision:         0.50
#   Brier score:       0.078
#   Calibration error: 0.012

# Per-DRG calibration (top 5 high-volume DRGs):
#   Heart failure: predicted 0.18, actual 0.17 (well-calibrated)
#   COPD:          predicted 0.15, actual 0.16 (well-calibrated)
#   Pneumonia:     predicted 0.12, actual 0.13 (well-calibrated)
#   Sepsis:        predicted 0.16, actual 0.18 (slightly under-predicting)
#   Hip/knee:      predicted 0.07, actual 0.06 (well-calibrated)
```

---

## Step 16: Per-Patient Explainability (SHAP)

```python
import shap

explainer = shap.TreeExplainer(calibrated_xgb.calibrated_classifiers_[0].estimator)
shap_values = explainer.shap_values(X_test_processed)

# For each patient flagged as high-risk, surface:
# 1. The top 5 drivers of THIS patient's risk
# 2. Whether each is modifiable

# Example:
# Patient #4821 -- Risk score 0.42 (top 5%)
# Top 5 drivers (SHAP):
#   1. has_pcp_followup_scheduled = NO   (+0.10)  [MODIFIABLE -- schedule one]
#   2. charlson_comorbidity_index = 6     (+0.08)  [not modifiable]
#   3. albumin = 2.4                      (+0.07)  [not modifiable directly]
#   4. social_vuln_score = 4              (+0.06)  [partially modifiable -- social work consult]
#   5. polypharmacy = 1 (14 meds)         (+0.04)  [MODIFIABLE -- pharmacy reconciliation]

# UI surfaces TWO actions:
#   - Schedule PCP follow-up within 7 days
#   - Pharmacy reconciliation before discharge
```

**Expert insight:** SHAP values aren't just for explaining — they're for actioning. By labeling each driver as modifiable / not modifiable, the discharge team gets a checklist of interventions, not a black-box "this patient is risky" alert.

---

## Step 17: Deployment Considerations

```python
# Production architecture:
#   1. Model artifact: ~80MB (XGBoost + isotonic calibrator + preprocessor)
#   2. Latency: < 500ms end-to-end (feature pull from EHR + score + SHAP top-5)
#   3. Integration: scoring service called by Epic at "Discharge Order Signed" event
#   4. Output: writes risk score + top 5 drivers to a discharge planning note

# Retraining cadence: QUARTERLY
#   - Care patterns drift slowly in inpatient medicine
#   - Quarterly is enough; monthly burns reviewer time
#   - Gate retraining on quarterly fairness audit re-passing

# Monitoring:
#   - Daily AUC on confirmed-readmit / not-readmit
#   - Weekly per-DRG calibration drift (alert if any DRG ECE > 0.05)
#   - Monthly per-race / sex flagging rate drift
#   - User feedback: % of flagged patients where care team disagrees ("this isn't actually high risk")

# Safety
#   - "Override / disagree" button on the discharge UI -- writes to feedback log
#   - Never AUTO-prevents discharge -- always advisory to the human
#   - Yearly external audit of fairness metrics (per CMS guidance)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Lab missingness | Median impute everything | Distinguish informative vs random missingness, encode `_measured` flag |
| 700-DRG handling | One-hot encode (700 columns) | Target encode + Bayesian smoothing for low-volume DRGs |
| Probability output | Use raw XGBoost output | Isotonic calibration (downstream cost model needs it) |
| Train/test split | Random | Time-based (care patterns drift) |
| Class imbalance | SMOTE | scale_pos_weight + threshold tuning |
| Threshold | F1 optimum | Top-14% by intervention budget |
| Fairness | Ignore | Per-group AUC + flagging rate audit; investigate over-flagging |
| Explanation | "model said high" | Top 5 SHAP drivers, labeled modifiable / not |
| Deployment | Direct integration, no override | Advisory to human + override button + feedback log |
| Monitoring | None | Daily AUC, weekly per-DRG calibration, monthly fairness drift |
| Compliance | Train and ship | HIPAA + CMS-aware design; external audit pathway built-in |
