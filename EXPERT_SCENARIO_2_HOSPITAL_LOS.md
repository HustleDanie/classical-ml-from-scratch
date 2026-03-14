# Expert Scenario 2: Predicting Hospital Length of Stay

> **Complexity:** Heavily right-skewed target, multiple data sources to join, temporal leakage risks, medical text fields, explainability for clinical staff.

---

## The Brief

A hospital network gives you data on 85,000 patient admissions over 3 years. They want to predict how many days each patient will stay at the time of admission (before treatment begins). Accurate predictions help them manage bed capacity, staff scheduling, and discharge planning. Currently, nurses guess manually and are off by an average of 2.8 days. The hospital wants to beat that. They also want to understand which factors drive longer stays so they can intervene early.

This is complex because: the target is heavily right-skewed (most stays 1-3 days, some 90+ days), you have multiple data sources to join (admissions, labs, diagnoses), there's meaningful missing data, text fields from doctor notes, and the model must be interpretable for clinical staff.

---

## Step 1: Define the Problem Type

```
Type:           Regression (predicting number of days, continuous)
Primary Metric: MAE (Mean Absolute Error) -- must beat nurses' 2.8-day average error
Secondary:      RMSE (penalizes big misses on long-stay patients)
Business Goal:  Predict at ADMISSION TIME (before treatment starts)
Constraint:     Model must be explainable to doctors (no black boxes)
Special:        Target is heavily right-skewed (median 3 days, mean 5.2 days, max 94 days)
```

**Expert thinking:** This is NOT a simple regression. The target distribution looks like:
```
Days  | 1-2  | 3-5  | 6-10 | 11-20 | 21-50 | 50+
%     | 35%  | 30%  | 20%  | 10%   | 4%    | 1%
```
Most patients stay 1-5 days. But the long-tail matters a LOT -- a patient staying 40 days who was predicted at 5 days is a massive planning failure. We might need to transform the target or treat this as a mixed problem.

---

## Step 2: Understand the Data

We receive 4 separate tables that need to be joined:

**Table 1: Admissions (85,000 rows)**
```
- patient_id (string, 52,000 unique -- some patients have multiple admissions)
- admission_id (string, unique per admission)
- admission_date (datetime)
- admission_type (categorical: emergency, urgent, elective, newborn)
- admission_source (categorical: ER, physician_referral, transfer, walk-in)
- insurance_type (categorical: private, medicare, medicaid, self-pay, military)
- age (int, 0-102)
- gender (categorical: M, F)
- weight_kg (float, 18% missing)
- height_cm (float, 22% missing)
- admission_diagnosis_text (free text, doctor's initial assessment -- e.g., "chest pain, r/o MI, hx HTN")
- attending_physician_id (string, 340 unique doctors)
- ward (categorical: general, ICU, cardiac, surgical, maternity, pediatric, psych)
- length_of_stay (TARGET -- integer, 1 to 94 days)
```

**Table 2: Diagnoses (312,000 rows -- multiple per admission)**
```
- admission_id
- diagnosis_code (ICD-10 code, e.g., "I21.0" = acute MI, 4,200 unique codes)
- diagnosis_type (primary, secondary, complication)
- diagnosis_rank (1st, 2nd, 3rd... up to 15 per admission)
```

**Table 3: Lab Results (1,400,000 rows -- multiple per admission)**
```
- admission_id
- lab_test_name (string: "WBC", "hemoglobin", "creatinine", etc. -- 89 unique tests)
- lab_value (float)
- lab_unit (string)
- result_flag (categorical: normal, abnormal_high, abnormal_low, critical)
- collected_time (datetime -- hours after admission)
```

**Table 4: Previous Admissions History (aggregated)**
```
- patient_id
- num_admissions_past_year (int)
- num_er_visits_past_year (int)
- num_surgeries_past_year (int)
- days_since_last_discharge (float, NaN if first admission)
- avg_previous_los (float, average length of stay in prior visits)
```

**Expert thinking:** This is a JOIN + AGGREGATE problem. I need to:
1. Join all tables on `admission_id` or `patient_id`
2. Aggregate diagnoses into features (primary diagnosis, number of comorbidities)
3. Aggregate lab results (but ONLY labs from the first 6 hours -- later labs leak information about the stay)
4. Handle the text field `admission_diagnosis_text`
5. Be very careful about temporal leakage -- anything that happens DURING the stay can't be used

---

## Step 3: Exploratory Data Analysis (EDA)

```python
# Target distribution
length_of_stay.describe()
# count    85,000
# mean     5.2 days
# std      6.1 days     # std > mean = heavily right-skewed
# min      1 day
# 25%      2 days
# 50%      3 days (median)
# 75%      6 days
# max      94 days

# Missing values
weight_kg             15,300 (18.0%)
height_cm             18,700 (22.0%)
days_since_last_discharge  33,200 (39.1%)  # first-time patients
avg_previous_los      33,200 (39.1%)       # same patients -- no history
```

**What EDA reveals:**

| Finding | Implication |
|---------|------------|
| Target is heavily right-skewed (skewness = 3.8) | Need log transform or specialized loss function |
| ICU patients average 11.2 days vs general ward 3.1 days | Ward is a strong predictor |
| Emergency admissions average 6.8 days vs elective 3.4 days | Admission type matters |
| Patients with 5+ diagnoses average 9.3 days vs 3.1 for 1-2 diagnoses | Number of comorbidities is crucial |
| Patients with abnormal creatinine average 8.7 days | Kidney function is a key lab |
| Readmitted patients (within 30 days) average 7.9 days | Prior admission history is predictive |
| Top 10 attending physicians have avg LOS ranging from 3.2 to 7.8 days | Some doctors keep patients longer (practice patterns vary) |
| `weight_kg` missing more for emergency (25%) vs elective (8%) | Missing NOT at random -- emergency patients often can't be weighed |
| `admission_diagnosis_text` contains abbreviations: "r/o" "hx" "s/p" "c/o" | Need medical abbreviation handling |
| Age has a U-shape with LOS: babies (7.2 days avg) and elderly 80+ (8.1 days) drop, working age (3.8 days) shortest | Non-linear age relationship |

**Critical temporal leakage check:**
- Lab results that were collected on day 3 of a 5-day stay CANNOT be used -- they wouldn't be available at admission time
- Filter to only labs collected within first 6 hours of admission
- Diagnosis codes added during the stay (complications) are leakers -- only use primary + secondary diagnoses at admission

---

## Step 4: Data Cleaning

```python
# === Handle temporal leakage in labs ===
# Only keep labs from first 6 hours after admission
labs['hours_after_admission'] = (labs['collected_time'] - labs['admission_time']).dt.total_seconds() / 3600
labs_early = labs[labs['hours_after_admission'] <= 6]
# This drops 62% of lab records -- but they're future data we can't use

# === Handle temporal leakage in diagnoses ===
# Only keep diagnoses ranked 1-3 (primary + first 2 secondaries)
# Later diagnoses are often complications discovered DURING the stay
diag_admission = diagnoses[diagnoses['diagnosis_rank'] <= 3]

# === Missing values ===
# weight_kg (18% missing): Create missing flag + impute with median BY ward
df['weight_missing'] = df['weight_kg'].isnull().astype(int)
df['weight_kg'] = df.groupby('ward')['weight_kg'].transform(
    lambda x: x.fillna(x.median())
)
# Why by ward? ICU patients weigh differently than maternity patients

# height_cm (22% missing): Same approach
df['height_missing'] = df['height_cm'].isnull().astype(int)
df['height_cm'] = df.groupby('ward')['height_cm'].transform(
    lambda x: x.fillna(x.median())
)

# days_since_last_discharge (39% missing = first-time patients)
df['is_first_admission'] = df['days_since_last_discharge'].isnull().astype(int)
df['days_since_last_discharge'] = df['days_since_last_discharge'].fillna(-1)
# -1 signals "no prior admission" -- tree models will learn to split on this

# avg_previous_los (39% missing = same first-time patients)
df['avg_previous_los'] = df['avg_previous_los'].fillna(0)  # no history = 0

# === Outliers ===
# Keep long stays! A 60-day ICU patient is real data, not an error.
# But check for data entry errors:
# - LOS = 0? Impossible (minimum is 1 day) -> found 23, set to 1
# - Age = 0 with ward = "cardiac"? Likely error -> found 4, investigate
# - Weight = 3kg for adult? Error -> found 7, set to NaN and re-impute
```

**Expert insight:** Missing data in hospitals is almost NEVER random. Emergency patients miss weight/height because they're unconscious. First-time patients miss history because they don't have any. The **pattern** of missingness is itself informative -- we encode it as features.

---

## Step 5: Feature Engineering

This is where 60% of our time goes. We need to transform 4 raw tables into one rich feature matrix.

```python
# ============================================================
# DIAGNOSIS FEATURES (from 312,000 rows -> 85,000 admission-level)
# ============================================================

# Primary diagnosis category (ICD-10 first 3 characters)
df['primary_diag_category'] = df['primary_diagnosis_code'].str[:3]
# "I21" = heart attack, "J18" = pneumonia, "S72" = hip fracture, etc.
# Reduces 4,200 unique codes to ~280 categories

# Number of diagnoses at admission (comorbidity burden)
diag_counts = diag_admission.groupby('admission_id').size().reset_index(name='num_diagnoses')
df = df.merge(diag_counts, on='admission_id', how='left')
df['num_diagnoses'] = df['num_diagnoses'].fillna(1)

# Presence of specific high-impact diagnoses (binary flags)
# These were identified by domain knowledge + statistical analysis
high_impact_codes = {
    'has_sepsis': ['A41', 'R65'],
    'has_heart_failure': ['I50'],
    'has_pneumonia': ['J18', 'J15', 'J13'],
    'has_hip_fracture': ['S72'],
    'has_copd': ['J44'],
    'has_diabetes_complicated': ['E11.6', 'E11.5'],
    'has_stroke': ['I63', 'I61'],
    'has_kidney_failure': ['N17', 'N18'],
    'has_cancer': ['C'],  # any code starting with C
}
# Sepsis patients average 14.2 days, hip fracture 8.7 days, pneumonia 6.1 days
# These binary flags give the model direct access to high-signal diagnoses

# Charlson Comorbidity Index (standard medical score)
# Maps ICD-10 codes to a 0-30 comorbidity severity score
df['charlson_index'] = compute_charlson(df['diagnosis_codes'])
# Score 0: healthy, Score 5+: seriously ill, Score 10+: very high mortality risk

# ============================================================
# LAB FEATURES (from first-6-hour labs -> admission-level)
# ============================================================

# Pivot key labs: one column per lab test, value = first result
key_labs = ['WBC', 'hemoglobin', 'creatinine', 'BUN', 'sodium', 'potassium',
            'glucose', 'albumin', 'troponin', 'lactate', 'platelet_count']

for lab in key_labs:
    lab_subset = labs_early[labs_early['lab_test_name'] == lab]
    # Take the FIRST result (closest to admission)
    first_result = lab_subset.sort_values('collected_time').groupby('admission_id').first()
    df = df.merge(first_result[['lab_value']].rename(columns={'lab_value': f'lab_{lab}'}),
                  on='admission_id', how='left')

# Number of labs ordered in first 6 hours (proxy for how sick the patient is)
df['num_labs_ordered'] = labs_early.groupby('admission_id').size()
# Sicker patients get more labs -- 15 labs in 6 hours vs 3 for routine admission

# Number of abnormal/critical lab results
df['num_abnormal_labs'] = labs_early[labs_early['result_flag'].isin(['abnormal_high', 'abnormal_low', 'critical'])]\
    .groupby('admission_id').size()
df['num_critical_labs'] = labs_early[labs_early['result_flag'] == 'critical']\
    .groupby('admission_id').size()

# Lab missingness features
for lab in key_labs:
    df[f'lab_{lab}_missing'] = df[f'lab_{lab}'].isnull().astype(int)
# Missing troponin means doctor didn't suspect heart attack -- informative!
# Missing albumin means not checking nutritional status -- less sick

# Impute missing labs with population median
for lab in key_labs:
    df[f'lab_{lab}'] = df[f'lab_{lab}'].fillna(df[f'lab_{lab}'].median())

# ============================================================
# TEXT FEATURES (from admission_diagnosis_text)
# ============================================================

# Clean medical abbreviations
abbreviation_map = {
    'r/o': 'rule out', 'hx': 'history', 's/p': 'status post',
    'c/o': 'complaining of', 'w/': 'with', 'w/o': 'without',
    'dx': 'diagnosis', 'tx': 'treatment', 'sx': 'symptoms',
    'sob': 'shortness of breath', 'cp': 'chest pain',
    'abd': 'abdominal', 'htn': 'hypertension', 'dm': 'diabetes',
    'cad': 'coronary artery disease', 'chf': 'congestive heart failure',
}
def clean_medical_text(text):
    text = text.lower()
    for abbr, full in abbreviation_map.items():
        text = text.replace(abbr, full)
    return text

df['clean_diagnosis_text'] = df['admission_diagnosis_text'].apply(clean_medical_text)

# TF-IDF on diagnosis text (top 100 terms)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
text_features = tfidf.fit_transform(df['clean_diagnosis_text'])
# Top terms: "chest pain", "shortness breath", "rule out", "hip fracture", "abdominal pain"
# These add signal not captured by ICD codes (free text is more nuanced)

# Text statistics
df['diagnosis_text_length'] = df['admission_diagnosis_text'].str.len()
df['diagnosis_text_word_count'] = df['admission_diagnosis_text'].str.split().str.len()
# Longer descriptions = more complex cases = longer stays

# ============================================================
# DEMOGRAPHIC & ADMISSION FEATURES
# ============================================================

# BMI (from weight and height)
df['bmi'] = df['weight_kg'] / (df['height_cm'] / 100) ** 2
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 40, 100],
                             labels=['underweight', 'normal', 'overweight', 'obese', 'morbidly_obese'])

# Age features (U-shaped relationship)
df['age_squared'] = df['age'] ** 2  # captures U-shape
df['is_elderly'] = (df['age'] >= 75).astype(int)
df['is_infant'] = (df['age'] <= 1).astype(int)
df['age_group'] = pd.cut(df['age'], bins=[0, 1, 18, 40, 65, 80, 120],
                          labels=['infant', 'pediatric', 'young_adult', 'middle', 'senior', 'elderly'])

# Admission timing
df['admission_month'] = df['admission_date'].dt.month
df['admission_day_of_week'] = df['admission_date'].dt.dayofweek
df['is_weekend_admission'] = (df['admission_day_of_week'] >= 5).astype(int)
# Weekend admissions average 0.5 days longer (fewer discharges on weekends)

df['is_winter'] = df['admission_month'].isin([12, 1, 2]).astype(int)
# Winter has more respiratory cases = longer average stays

# ============================================================
# PHYSICIAN PRACTICE PATTERNS
# ============================================================

# Each doctor has different tendencies -- some keep patients longer
physician_avg_los = df.groupby('attending_physician_id')['length_of_stay'].mean()
df['physician_avg_los'] = df['attending_physician_id'].map(physician_avg_los)
# CRITICAL: Must compute this only on training set to avoid leakage

# ============================================================
# PATIENT HISTORY FEATURES
# ============================================================

df['is_readmission_30d'] = (df['days_since_last_discharge'] <= 30) & (df['days_since_last_discharge'] >= 0)
df['is_readmission_30d'] = df['is_readmission_30d'].astype(int)
# 30-day readmissions average 7.9 days (sicker patients)

df['is_frequent_flyer'] = (df['num_admissions_past_year'] >= 3).astype(int)
# Patients with 3+ admissions/year average 8.2 days
```

**Expert insight:** We've gone from 4 raw tables to a single feature matrix with ~180 features. The key transformations:
- **Diagnoses** -> comorbidity count, Charlson index, high-impact flags
- **Labs** -> pivoted key values, abnormality counts, missingness flags
- **Text** -> TF-IDF features + statistics
- **Time** -> month, day of week, weekend flag
- **Physician** -> historical average LOS (practice pattern encoding)

---

## Step 6: Feature Selection

```python
# Stage 1 -- Drop the obvious
drop_cols = [
    'patient_id', 'admission_id',               # identifiers
    'admission_date',                             # raw date (extracted features already)
    'admission_diagnosis_text',                   # raw text (TF-IDF already extracted)
    'clean_diagnosis_text',                       # intermediate
    'attending_physician_id',                     # raw ID (physician_avg_los already extracted)
    'primary_diagnosis_code',                     # raw code (category already extracted)
]

# Stage 2 -- Mutual Information for regression
from sklearn.feature_selection import mutual_info_regression

mi = mutual_info_regression(X_train, y_train, random_state=42)

# Top 15 features by MI:
#   ward                    0.18   (ICU is the #1 predictor)
#   num_diagnoses           0.12
#   charlson_index          0.11
#   admission_type          0.09
#   has_sepsis              0.08
#   num_abnormal_labs       0.07
#   lab_creatinine          0.06
#   physician_avg_los       0.06
#   age                     0.05
#   num_labs_ordered        0.05
#   is_readmission_30d      0.04
#   lab_albumin             0.04
#   has_hip_fracture        0.04
#   diagnosis_text_length   0.03
#   bmi                     0.02

# Bottom features (MI ~ 0):
#   gender                  0.001  -> keep (slight effect on surgical stays)
#   is_winter               0.002  -> keep (marginal seasonal effect)
#   lab_sodium_missing      0.000  -> REMOVE (everyone gets sodium checked)
#   admission_day_of_week   0.001  -> merge into is_weekend only

# Stage 3 -- RFE with LightGBM
# Cross-validated RFE selects 65 of 180 features
# Reduced set gives RMSE within 0.5% of full set

# Final: 65 features (from 180 engineered, from 4 raw tables)
```

---

## Step 7: Preprocessing

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

numeric_features = ['age', 'bmi', 'lab_creatinine', 'lab_hemoglobin', ...]  # 40 numeric
categorical_features = ['ward', 'admission_type', 'insurance_type', ...]     # 8 categorical
binary_features = ['has_sepsis', 'is_elderly', 'weight_missing', ...]        # 17 binary
tfidf_features = [...]  # 100 TF-IDF columns (already numeric)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('bin', 'passthrough', binary_features),  # binary features don't need scaling
    ('text', 'passthrough', tfidf_features),  # TF-IDF is already scaled 0-1
])
```

**CRITICAL: Target transformation**

```python
# The target is heavily right-skewed (skewness = 3.8)
# Linear models will be pulled by the long tail -- predicting 5.2 (the mean) for everyone
# Solution: log-transform the target

import numpy as np
y_train_log = np.log1p(y_train)  # log(1 + LOS) to handle LOS=0 edge case
# Now the target is approximately normal: mean=1.4, std=0.7, skewness=0.4

# REMEMBER: Predictions must be inverse-transformed back
# y_pred = np.expm1(model.predict(X))  # exp(prediction) - 1

# For tree-based models: log transform is optional (trees handle skew natively)
# But we'll compare both approaches
```

---

## Step 8: Train/Test Split

```python
# Time-based split (patients are admitted over 3 years)
# Train: first 2.5 years (71,250 admissions)
# Test: last 6 months (13,750 admissions)

train = df[df['admission_date'] < '2025-07-01']
test = df[df['admission_date'] >= '2025-07-01']

# Why time-based?
# 1. Simulates production (predicting future stays from past patterns)
# 2. Physician practice patterns may change over time
# 3. Seasonal flu/COVID waves affect winter stays differently
# 4. New doctors join, old ones leave -- temporal validation catches this

# Also: some patients appear in BOTH train and test (readmissions)
# This is fine and realistic -- we SHOULD learn from their history
```

---

## Step 9: Baseline

```python
# Baseline 1: Predict the mean (5.2 days) for everyone
# MAE: 3.4 days, RMSE: 6.1 days

# Baseline 2: Predict the median (3 days) for everyone
# MAE: 2.9 days, RMSE: 5.8 days
# (Median is better than mean for skewed targets!)

# Baseline 3: Nurses' manual estimate
# MAE: 2.8 days (what we need to beat)

# Baseline 4: Simple rule -- ward-specific average
# ICU -> predict 11.2, general -> 3.1, surgical -> 5.5, etc.
# MAE: 2.3 days
# This simple rule already beats average AND nurses!
# Our ML model needs to beat 2.3 days MAE convincingly
```

**Expert insight:** The ward-specific average is a surprisingly strong baseline. It tells us that `ward` alone explains a big chunk of LOS variance. Our ML model's advantage will come from WITHIN-ward variation -- two ICU patients might stay 3 days or 30 days, and we need to predict which.

---

## Step 10: Try Multiple Models (5-Fold CV)

```python
# Two approaches: (A) predict raw LOS, (B) predict log(LOS) then back-transform

# === Approach A: Raw target ===
# Model 1: Ridge Regression
# CV MAE: 2.42, RMSE: 4.31

# Model 2: Lasso Regression
# CV MAE: 2.38, RMSE: 4.25 (sparse -- zeroed out 30 features)

# Model 3: Random Forest (n_estimators=500)
# CV MAE: 2.01, RMSE: 3.65

# Model 4: XGBoost (default params)
# CV MAE: 1.89, RMSE: 3.42

# Model 5: LightGBM (default params)
# CV MAE: 1.85, RMSE: 3.38

# Model 6: KNN (k=15)
# CV MAE: 2.55, RMSE: 4.80 (struggles with 65 features -- curse of dimensionality)

# Model 7: SVR (RBF kernel)
# CV MAE: 2.31, RMSE: 4.15 (okay, but slow to train on 71K rows)

# === Approach B: Log-transformed target ===
# LightGBM on log(LOS):
# CV MAE: 1.92, RMSE: 3.51 (slightly worse -- tree models handle skew natively)

# Ridge on log(LOS):
# CV MAE: 2.18, RMSE: 3.82 (better than Ridge on raw -- log helps linear models)

# VERDICT: Tree models work best on raw target. Linear models benefit from log transform.
# Top 3: LightGBM (1.85), XGBoost (1.89), Random Forest (2.01)
```

| Model | MAE | RMSE | Time | Notes |
|-------|-----|------|------|-------|
| LightGBM | 1.85 | 3.38 | 12s | Best overall |
| XGBoost | 1.89 | 3.42 | 28s | Close second |
| Random Forest | 2.01 | 3.65 | 45s | Solid third |
| Lasso (log target) | 2.18 | 3.82 | 2s | Best linear model |
| Ridge (log target) | 2.22 | 3.88 | 1s | Fast, interpretable |
| SVR | 2.31 | 4.15 | 8min | Too slow, not worth it |
| KNN | 2.55 | 4.80 | 30s | Can't handle high dimensions |
| **Ward-average baseline** | **2.30** | **4.20** | **0s** | **The bar to beat** |
| **Nurses' estimate** | **2.80** | **--** | **--** | **Current system** |

All tree models beat both baselines. LightGBM is 34% better than nurses (1.85 vs 2.8 MAE).

---

## Step 11: Handling the Skewed Target (Instead of Class Imbalance)

```python
# Problem: MAE of 1.85 averages across all patients.
# But for long-stay patients (>14 days), the errors are much bigger:
#
# Short stay (1-3 days):   MAE = 0.9 days  (good!)
# Medium stay (4-10 days): MAE = 2.1 days  (okay)
# Long stay (11+ days):    MAE = 5.8 days  (terrible!)
#
# The model is BAD at predicting long stays because:
# 1. Only 15% of training data is long-stay
# 2. Standard loss (MSE/MAE) optimizes for the majority
# 3. Long-stay patients are the ones who MOST need accurate predictions

# Solution 1: Sample weighting -- give long stays higher weight
sample_weights = np.where(y_train > 14, 3.0,    # 3x weight for long stays
                 np.where(y_train > 7, 1.5, 1.0))  # 1.5x for medium

lgbm = LGBMRegressor(n_estimators=1000)
lgbm.fit(X_train, y_train, sample_weight=sample_weights)
# Overall MAE: 1.92 (slightly worse) but long-stay MAE: 4.1 (much better!)

# Solution 2: Quantile regression -- predict the 50th percentile (robust to outliers)
lgbm_quantile = LGBMRegressor(objective='quantile', alpha=0.5)
# MAE: 1.88, Long-stay MAE: 4.9

# Solution 3: Two-stage model
# Stage 1: Classify into short/medium/long (classification)
# Stage 2: Predict exact days within each group (3 separate regression models)
# This lets each sub-model specialize

# Stage 1: LightGBM Classifier (short/medium/long)
# Accuracy: 82%, F1-macro: 0.71
# Stage 2:
#   Short model (1-3 days):  MAE = 0.7
#   Medium model (4-10 days): MAE = 1.6
#   Long model (11+ days):   MAE = 3.9
# Combined MAE: 1.78 -- BEST RESULT!

# DECISION: Use the two-stage approach
# It has the best overall MAE (1.78) AND the best long-stay MAE (3.9)
```

**Expert insight:** The two-stage approach is a powerful technique for skewed regression. Instead of one model struggling to predict 1-94 days, we have three specialists. The classifier decides "roughly how long" (short/medium/long), then the specialist regressor fine-tunes within that range. This is similar to how doctors think: "This is a quick-stay case" vs "This patient will be here a while."

---

## Step 12: Hyperparameter Tuning

```python
# Tune all 4 models in the two-stage system:
# 1. The classifier (short/medium/long)
# 2. Short-stay regressor
# 3. Medium-stay regressor
# 4. Long-stay regressor

# Classifier tuning:
param_dist_clf = {
    'n_estimators': [500, 1000],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05],
    'num_leaves': [31, 63],
    'min_child_samples': [20, 50],
    'class_weight': ['balanced', None]
}
# Best: n_estimators=1000, max_depth=6, learning_rate=0.05, class_weight='balanced'
# Accuracy: 84%, F1-macro: 0.74

# Long-stay regressor tuning (most important -- highest errors):
param_dist_long = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [4, 6, 8, 12],
    'learning_rate': [0.01, 0.03, 0.05],
    'num_leaves': [15, 31, 63],
    'min_child_samples': [10, 20, 50],
    'reg_alpha': [0, 0.5, 1.0],
    'reg_lambda': [0, 0.5, 1.0],
}
# Best: n_estimators=2000, max_depth=6, learning_rate=0.03
# Long-stay MAE: 3.5 (improved from 3.9)

# Final two-stage tuned results:
# Overall MAE: 1.68    (vs nurses' 2.8 -- a 40% improvement!)
# Short-stay MAE: 0.6
# Medium-stay MAE: 1.4
# Long-stay MAE: 3.5
```

---

## Step 13: Ensemble / Stacking

```python
# For this problem: stack the two-stage model with a single LightGBM model
# The single model is better at "borderline" cases (e.g., 3 days vs 4 days)
# The two-stage model is better at extreme cases

# Approach: Average the two predictions with optimized weights
# y_pred = 0.6 * two_stage_pred + 0.4 * single_lgbm_pred

# How to find optimal weights: grid search on validation set
best_mae = float('inf')
for w in np.arange(0.0, 1.05, 0.05):
    blended = w * two_stage_pred + (1 - w) * single_pred
    mae = mean_absolute_error(y_val, blended)
    if mae < best_mae:
        best_mae = mae
        best_weight = w
# Best: w=0.55 (slightly favoring two-stage)

# Blended MAE: 1.62 (improved from 1.68 -- small but consistent gain)
```

---

## Step 14: Final Evaluation on Test Set

```python
# Final model: Blended (55% two-stage + 45% single LightGBM)
# Test set: 13,750 admissions from the last 6 months

# Overall Results:
# MAE:   1.62 days    (nurses: 2.8 -- we're 42% better!)
# RMSE:  3.21 days
# R2:    0.73         (our features explain 73% of LOS variance)

# By stay category:
# Short (1-3 days, 65%):    MAE = 0.58  Median error: 0 days (exact!)
# Medium (4-10 days, 25%):  MAE = 1.42  Median error: 1 day
# Long (11+ days, 10%):     MAE = 3.51  Median error: 2 days

# Error distribution:
# Within 0 days: 38% of predictions (exactly right)
# Within 1 day:  67% of predictions
# Within 2 days: 82% of predictions
# Within 3 days: 90% of predictions
# Off by 5+ days: 5% of predictions (mostly long-stay patients)

# By ward:
# General:    MAE = 1.1
# Surgical:   MAE = 1.5
# Cardiac:    MAE = 1.8
# ICU:        MAE = 3.2 (hardest to predict -- high variance)
# Maternity:  MAE = 0.8 (most predictable -- standard delivery protocols)
# Psych:      MAE = 4.1 (very hard -- discharge depends on behavioral assessment)

# Business impact:
# Nurse estimate:  2.8 days MAE -> 38,500 bed-days of error per year
# ML model:        1.62 days MAE -> 22,300 bed-days of error per year
# Improvement:     16,200 fewer bed-days of planning error per year
# Each misplanned bed-day costs ~$500 in inefficiency
# Annual savings:  ~$8.1 million
```

---

## Step 15: Explainability

```python
import shap

# Global feature importance (SHAP on single LightGBM model):
# 1. ward (ICU vs general is the strongest signal)
# 2. num_diagnoses (more comorbidities = longer stay)
# 3. charlson_index (comorbidity severity score)
# 4. has_sepsis (adds average 8 days)
# 5. lab_creatinine (kidney function)
# 6. admission_type_emergency (emergency stays are longer)
# 7. age (older = longer, with U-shape for infants)
# 8. num_abnormal_labs (proxy for acuity)
# 9. has_hip_fracture (predictable long stay: surgery + rehab)
# 10. physician_avg_los (practice pattern variation)

# PER-PATIENT EXPLANATION (for doctors):
# Patient #4582: Predicted 12 days (actual: 14 days)
# "This patient's predicted stay is 12 days because:
#  + ICU admission (+4.2 days above average)
#  + Sepsis diagnosis (+3.8 days)
#  + 7 comorbidities, Charlson score 8 (+2.1 days)
#  + Creatinine 3.2 mg/dL -- kidney distress (+1.4 days)
#  + Age 78 (+0.8 days)
#  - Elective admission (-0.3 days)"

# PARTIAL DEPENDENCE PLOTS -- how each feature affects LOS:
# Age: U-shaped -- infants (7 days), adults (3.5 days), 80+ (7+ days)
# Creatinine: flat until 2.0, then sharply increases predicted LOS
# Num diagnoses: roughly linear -- each additional diagnosis adds ~0.7 days
# ICU: binary jump of ~6 days over general ward
```

**Expert insight:** Doctors don't trust black boxes. By providing per-patient SHAP explanations ("this patient will likely stay 12 days because of X, Y, Z"), doctors can validate the prediction against their clinical judgment. If the model flags something the doctor missed (e.g., an abnormal lab), it adds clinical value beyond just bed planning.

---

## Step 16: Deployment & Monitoring

```python
import joblib

# Save all components
joblib.dump(preprocessor, 'los_preprocessor.joblib')
joblib.dump(classifier, 'los_classifier.joblib')
joblib.dump(short_regressor, 'los_short_model.joblib')
joblib.dump(medium_regressor, 'los_medium_model.joblib')
joblib.dump(long_regressor, 'los_long_model.joblib')
joblib.dump(single_lgbm, 'los_single_model.joblib')
joblib.dump(tfidf_vectorizer, 'los_tfidf.joblib')

# Production integration:
# - Trigger: runs within 1 hour of admission
# - Input: admission record + first labs + diagnosis codes
# - Output: predicted LOS (integer days) + confidence interval + top 3 SHAP reasons
# - Display: on nurse's dashboard and discharge planning board

# Monitoring:
# - Track weekly MAE on confirmed discharges
# - Alert if MAE exceeds 2.2 days (25% degradation)
# - Retrain monthly with new data
# - Watch for concept drift: if ICU avg LOS shifts (new protocols, pandemics)
# - Track per-ward performance separately (psych ward may need its own model)

# Edge cases to handle:
# - Patient transferred between wards mid-stay: use admission ward
# - Patient dies during stay: include in training (model should predict total stay)
# - Patient leaves against medical advice: flag but include
# - Readmission within 24 hours: treat as continuation of original stay
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Data structure | Work with one flat table | Joined 4 tables with careful temporal awareness |
| Lab features | Use all labs | Only first-6-hour labs (later labs are temporal leakers) |
| Diagnosis codes | One-hot 4,200 codes (explosion!) | Charlson index + high-impact flags + category grouping |
| Text data | Ignore it or bag-of-words | Clean medical abbreviations + targeted TF-IDF |
| Missing values | Drop or `fillna(median)` | Encoded missingness as features (clinically meaningful) |
| Skewed target | Predict raw and hope | Two-stage model: classify first, then regress within groups |
| Physician effect | Ignore it | Target-encoded physician practice patterns |
| Evaluation | "R2 = 0.73, done" | Stratified analysis: short/medium/long stay + per-ward |
| Explainability | Feature importance bar chart | Per-patient SHAP explanations for clinical staff |
| Deployment | Save model, done | Edge case handling, monitoring, retraining plan, dashboards |
| Baseline | Skip or use mean | Four baselines including nurses' current performance |
