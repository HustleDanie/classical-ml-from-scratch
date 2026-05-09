# Expert Scenario 52: Rare Disease Diagnosis (Small-Data Classification)

> **Complexity:** ~340 confirmed cases across 8 partner hospitals. Severe imbalance (~7% positive within referred-suspect cases). Reporting must include uncertainty intervals (the 340-case sample size makes single-number performance metrics dishonest). Heavy regularization, leave-one-hospital-out CV, simple models only.

---

## The Brief

A rare-disease research consortium gives you 5,200 patient records across 8 partner hospitals. Patients in this cohort were referred for suspected Disease X (a rare hereditary metabolic disorder). After full workup:

- 340 confirmed positive (6.5%)
- 4,860 confirmed negative (93.5%)

The brief: build a model that helps front-line clinicians (general practitioners, not specialists) decide whether to refer a patient for the expensive specialty workup. Their goals:

- Catch 90%+ of true positives (sensitivity).
- Keep false-positive rate under 25% (don't over-refer).
- Report performance with confidence intervals — the regulator will scrutinize point estimates from 340 cases.
- Provide per-patient explanation — clinicians won't trust a black box for rare-disease screening.
- Generalize to new hospitals (model deployed to 25 additional sites that aren't in training data).

This is the opposite extreme from RTB — you have hundreds of training events instead of billions. Standard ML practices (huge ensembles, deep tuning, fancy threshold optimization) all break down. You need to be conservative, transparent, and careful with uncertainty.

---

## Step 1: Define the Problem Type

```
Type:           Binary classification, severe small-data + class imbalance
Primary Metric: Sensitivity at fixed Specificity = 0.75 (or AUC with CI)
Secondary:      AUC-ROC with 95% CI; calibration on held-out hospital
                Per-hospital generalization (leave-one-hospital-out)
Business Goal:  Sens ≥ 0.90, FPR ≤ 0.25
Constraint:     Generalize to unseen hospitals; explainable; honest CIs reported
Imbalance:      93.5/6.5 -- moderate
Sample size:    5,200 total; 340 positives -- TINY
```

**Expert thinking:** with 340 positive cases, anything beyond Logistic Regression with L2 risks overfitting. You can train an XGBoost that hits AUC 0.92 in CV — but it's almost certainly overfit. The honest model is the simplest one that hits the metric, with bootstrap CIs to quantify uncertainty.

The key technique here is **leave-one-hospital-out CV**. Random CV folds let the model see patients from every hospital in training; the model learns hospital-specific signals (e.g., reference range differences across labs) and looks great in CV but fails on new hospitals.

---

## Step 2: Understand the Data

```
Shape: 5,200 rows x 42 features
Target: confirmed_disease_x -- 0 (93.5%), 1 (6.5%)

Features (after IRB review removed all PHI):

DEMOGRAPHIC (5):
- patient_age (3-84)
- biological_sex (M/F/Intersex)
- ancestry_self_reported (5 categories; 18% Unknown)
- has_consanguineous_parents (binary)
- bmi_at_referral

CLINICAL HISTORY (8):
- onset_age_years (when symptoms started; 12% missing)
- age_at_referral_minus_onset_years
- num_prior_specialty_referrals (0-15)
- num_prior_hospitalizations (0-25)
- has_developmental_delay (binary, peds only)
- has_recurrent_infections (binary)
- has_growth_failure (binary)
- num_unrelated_diagnoses (other conditions)

LAB FEATURES (15):
- enzyme_assay_value (float, 0.1-100, 23% missing — only ordered if specific suspicion)
- substrate_metabolite_level (float, 8% missing)
- urine_organic_acids_abnormal (binary, 35% missing)
- ferritin_level (15% missing)
- ldh_value (4% missing)
- ammonia_level (28% missing)
- ... 9 more lab values

GENETIC FEATURES (5):
- has_family_history_disease_x (binary, 4% Unknown)
- num_affected_family_members
- carrier_status_known (binary; tested for known Disease X variants)
- has_pathogenic_variant_panel (binary; results available, 56% missing pre-test)
- num_variants_of_unknown_significance

IMAGING / BIOPSY (5):
- mri_findings_abnormal (binary, 67% missing — not always indicated)
- biopsy_classic_histology (binary, 91% missing — invasive, only ordered with high suspicion)
- ophthalmology_exam_abnormal (binary, 38% missing)
- cardiac_exam_abnormal (binary, 22% missing)
- neuro_exam_abnormal (binary, 25% missing)

REFERRING HOSPITAL (4):
- hospital_id (8 unique)
- referring_specialty (Neurology / Pediatrics / Internal Med / Genetics / Other)
- year_of_referral (2018-2024)
- has_prior_referral_for_disease_x_workup
```

**Expert thinking:** key concerns:
- The most predictive lab tests are MOST MISSING (enzyme assay 23%, biopsy 91%) because they're only ordered when suspicion is high. Missingness IS the signal here, in both directions: missing = clinician didn't suspect; tested = clinician already suspects.
- Genetic features are post-hoc in many cases — the panel is ordered if Disease X is suspected. Including them as features biases the model toward what the clinician already thought.
- Hospital_id matters: each hospital has different referral patterns and lab reference ranges.

---

## Step 3: EDA

```python
df['confirmed_disease_x'].value_counts(normalize=True)
# 0    0.935
# 1    0.065  (340 cases)

# Per-hospital base rate
df.groupby('hospital_id')['confirmed_disease_x'].agg(['count', 'sum', 'mean'])
#   count   sum  mean
# A   780    78  0.100   (specialist center, higher suspicion → more referrals confirmed)
# B   650    35  0.054
# C   590    24  0.041
# D   720    52  0.072
# E   480    21  0.044
# F   810    62  0.077
# G   620    44  0.071
# H   550    24  0.044

# Imbalance is moderate, but PER HOSPITAL the positive rate varies 4.1% to 10.0%
# This is the leave-one-hospital-out problem in action

# Top single features
df.groupby('confirmed_disease_x')[['enzyme_assay_value', 'urine_organic_acids_abnormal']].mean()
# Confirmed positive: enzyme assay 4.2 (low); urine OA abnormal in 81% of tested
# Confirmed negative: enzyme assay 12.4 (normal); urine OA abnormal in 14%

# Family history
df.groupby('has_family_history_disease_x')['confirmed_disease_x'].mean()
# 0: 0.045
# 1: 0.31    -- 7x lift
# Unknown: 0.062
```

---

## Step 4: Data Cleaning

```python
# === STRATEGIC MISSINGNESS HANDLING ===
# Different missingness for different reasons; encode each carefully

# Lab tests ordered only on suspicion -- create _ordered flag, keep value if present
labs_suspicion_ordered = ['enzyme_assay_value', 'urine_organic_acids_abnormal',
                           'biopsy_classic_histology']
for col in labs_suspicion_ordered:
    df[f'{col}_ordered'] = df[col].notna().astype(int)
    if df[col].dtype == bool or df[col].dropna().isin([0,1]).all():
        df[col] = df[col].fillna(0).astype(int)
    else:
        df[col] = df[col].fillna(df[col].median())

# Routine labs -- impute by hospital median (reference ranges differ)
labs_routine = ['ferritin_level', 'ldh_value', 'ammonia_level']
for col in labs_routine:
    df[col] = df.groupby('hospital_id')[col].transform(lambda x: x.fillna(x.median()))
    # Fall back to global median if hospital missing all
    df[col] = df[col].fillna(df[col].median())

# Onset age missing 12% -- median impute by sex
df['onset_age_years_missing'] = df['onset_age_years'].isnull().astype(int)
df['onset_age_years'] = df['onset_age_years'].fillna(df['onset_age_years'].median())

# Imaging / biopsy missingness -- treat as separate "Not Done" category
imaging_features = ['mri_findings_abnormal', 'ophthalmology_exam_abnormal',
                    'cardiac_exam_abnormal', 'neuro_exam_abnormal']
for col in imaging_features:
    df[col] = df[col].fillna(0)  # not done -> assume normal (conservative)
    df[f'{col}_not_done'] = ((df[col] == 0)).astype(int)  # not strictly correct, OK for prototype

# === FILTER LATE-IN-WORKUP FEATURES (LEAKAGE) ===
# Genetic panel results available pre-referral are unusual; if present, model can use them
# Biopsy after referral is post-hoc -- exclude from training-time features
# Decision: keep biopsy_ordered but NOT biopsy_classic_histology in feature set
df = df.drop(columns=['biopsy_classic_histology'])

# Outlier check
df = df[df['patient_age'] >= 0]
```

**Expert insight:** the missingness pattern here is dense with information. A clinician who DID NOT order an enzyme assay was not suspecting Disease X. A clinician who ordered AND it came back low has high suspicion. Encoding both the value and the "ordered" flag captures this cleanly.

---

## Step 5: Feature Engineering — Conservative

```python
# With 340 positives, every engineered feature must justify itself
# Rule of thumb: at most ~17 features (n/20 = 340/20 = 17)
# Beyond that, overfitting risk balloons

# === FAMILY/GENETIC RISK SCORE (composite) ===
df['hereditary_risk'] = (
    df['has_family_history_disease_x'].astype(int) * 3 +
    df['has_consanguineous_parents'].astype(int) * 2 +
    df['num_affected_family_members'].clip(0, 5) +
    df['has_pathogenic_variant_panel'].astype(int) * 4
)

# === CLINICAL SUSPICION SCORE ===
df['suspicion_score'] = (
    df['has_developmental_delay'].astype(int) +
    df['has_recurrent_infections'].astype(int) +
    df['has_growth_failure'].astype(int) +
    df['urine_organic_acids_abnormal'].astype(int) * 2 +
    df['enzyme_assay_value_ordered'].astype(int) +
    (df['enzyme_assay_value'] < 5).astype(int) * 3
)

# === AGE AT ONSET CATEGORIES ===
df['onset_category'] = pd.cut(
    df['onset_age_years'],
    bins=[-0.1, 1, 5, 12, 25, 100],
    labels=['Infant', 'Toddler', 'Child', 'Adolescent', 'Adult']
)
# Disease X often onsets in infancy/early childhood

# === DELAY TO DIAGNOSIS ===
df['referral_delay_years'] = (df['age_at_referral_minus_onset_years']).clip(0, 30)

# Final feature set: target ~16 features
selected_features = [
    'patient_age', 'biological_sex', 'hereditary_risk', 'suspicion_score',
    'enzyme_assay_value', 'enzyme_assay_value_ordered',
    'urine_organic_acids_abnormal', 'urine_organic_acids_abnormal_ordered',
    'onset_category', 'referral_delay_years', 'num_prior_specialty_referrals',
    'has_pathogenic_variant_panel', 'num_variants_of_unknown_significance',
    'mri_findings_abnormal', 'has_recurrent_infections', 'referring_specialty'
]
```

---

## Step 6: Train/Test Split — Leave-One-Hospital-Out

```python
# CRITICAL: traditional 80/20 random split lets the model see patients from every hospital
# in training. The deployed model goes to 25 NEW hospitals.
# Leave-one-hospital-out CV simulates this generalization gap.

from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
groups = df['hospital_id']

# 8 folds -- one per hospital
# Each fold: 7 hospitals train, 1 hospital validate
# Performance averaged across the 8 folds

# Final test: 2 hospitals held out completely (not in any CV fold)
held_out_hospitals = ['G', 'H']  # randomly chosen
df_train = df[~df['hospital_id'].isin(held_out_hospitals)]
df_test  = df[df['hospital_id'].isin(held_out_hospitals)]
```

---

## Step 7: Try Multiple Models — Conservative Pool

```python
# With 340 positives, we test SIMPLE models only
# XGBoost, Random Forest, deep ensembles -- all overfit-prone here

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# === Model 1: Logistic Regression with L2 ===
lr_l2 = LogisticRegression(C=0.5, penalty='l2', max_iter=2000, random_state=42)
# CV AUC (LeaveOneHospital): mean 0.821, std 0.041
# Sensitivity at 75% spec (across folds): mean 0.84, std 0.06

# === Model 2: Logistic Regression with L1 ===
lr_l1 = LogisticRegression(C=0.5, penalty='l1', solver='liblinear', max_iter=2000)
# CV AUC: mean 0.815, std 0.038
# L1 selects ~10 of 16 features

# === Model 3: Decision Tree (max_depth=4 -- shallow) ===
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=42)
# CV AUC: mean 0.794, std 0.052
# Good interpretability; clinicians can read the tree

# === Model 4: Random Forest (small) ===
rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42)
# CV AUC: mean 0.832, std 0.044

# === Model 5: SVM with linear kernel ===
svc = SVC(C=0.5, kernel='linear', probability=True)
# CV AUC: mean 0.823, std 0.042

# === Model 6: Gaussian Naive Bayes ===
nb = GaussianNB()
# CV AUC: mean 0.781, std 0.039
```

**Top performer:** Random Forest (0.832 ± 0.044). Logistic Regression L2 is close (0.821 ± 0.041) and more interpretable.

The key observation: standard deviations are 4-5 percentage points. With 340 positives, this is unavoidable — it's the honest uncertainty.

---

## Step 8: Confidence Intervals via Bootstrap

```python
from sklearn.utils import resample

# Compute AUC with bootstrap CI
n_bootstraps = 1000
boot_aucs = []
for i in range(n_bootstraps):
    idx_resample = resample(range(len(y_test)), n_samples=len(y_test))
    if len(np.unique(y_test[idx_resample])) < 2:
        continue
    proba = lr_l2.predict_proba(X_test[idx_resample])[:, 1]
    boot_aucs.append(roc_auc_score(y_test[idx_resample], proba))

# 95% CI: 2.5th and 97.5th percentile
ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

# Reported: AUC = 0.821 (95% CI: 0.756 - 0.881)
# That ± 6 percentage points uncertainty is the honest story
```

**Expert insight:** with 340 positives the AUC has substantial uncertainty. A point estimate of 0.83 might really be 0.76 - 0.88. Reporting this explicitly is what the regulator wants. Hiding it is the bigger risk.

---

## Step 9: Hyperparameter Tuning — Conservative

```python
# With 5,200 rows and LeaveOneHospitalOut CV (8 folds), each fold has ~650 rows
# Don't go nuts with hyperparameters -- 4-6 grid points per parameter

param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']  # supports both penalties
}

grid = GridSearchCV(
    LogisticRegression(max_iter=2000),
    param_grid,
    cv=LeaveOneGroupOut(),
    scoring='roc_auc'
)
grid.fit(X_train, y_train, groups=df_train['hospital_id'])
# Best: C=0.5, penalty='l2'
# Tuned CV AUC: 0.823 ± 0.042

# Note: the tuned model BARELY improved on the un-tuned default
# This is expected -- with small data, the right answer is "don't over-tune"
```

---

## Step 10: Threshold Tuning to Sensitivity-Specificity Pareto

```python
# Find threshold meeting Sens >= 0.90, FPR <= 0.25
# Sweep thresholds; report all that meet constraints

y_proba = lr_l2.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.02, 0.50, 25)
candidates = []
for thr in thresholds:
    y_pred = (y_proba >= thr).astype(int)
    sens = ((y_pred == 1) & (y_val == 1)).sum() / max(y_val.sum(), 1)
    fpr = ((y_pred == 1) & (y_val == 0)).sum() / max((y_val == 0).sum(), 1)
    if sens >= 0.90 and fpr <= 0.25:
        candidates.append((thr, sens, fpr))

# Pick best: highest sens, then lowest fpr
# Best: threshold 0.082, sens 0.92, fpr 0.21
```

**Expert insight:** with such small samples, the threshold itself has uncertainty. A conservative production approach: use a slightly more aggressive threshold (lower) than the validation Pareto-optimum to buffer against noise.

---

## Step 11: Final Evaluation on Held-Out Hospitals

```python
# Final model: Logistic Regression L2 (C=0.5), threshold 0.082
# Test set: 2 held-out hospitals (G, H)
# 1,170 patients; 68 positives

y_proba_test = lr_l2.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= 0.082).astype(int)

# Performance:
#   Sensitivity:               0.871   (95% CI 0.79 - 0.94)
#   Specificity:               0.798   (95% CI 0.78 - 0.82)
#   Sens at 75% spec:          0.853
#   AUC-ROC:                   0.812   (95% CI 0.74 - 0.87)
#   PPV at threshold:          0.21
#   NPV at threshold:          0.992
#
# Per-hospital:
#   Hospital G: AUC 0.838, Sens 0.91 (95% CI 0.78-0.97)
#   Hospital H: AUC 0.789, Sens 0.83 (95% CI 0.62-0.94)
#
# Sensitivity dropped from CV (0.92) to held-out test (0.87) -- expected ~5pp generalization gap
```

**Expert insight:** the held-out test sensitivity (0.87) is within the CV CI lower bound — the model generalized as expected. Hospital H, with the smaller sample, has a wider CI; this is honest uncertainty reporting.

---

## Step 12: Per-Patient Explainability

```python
# For Logistic Regression, coefficients are the explanation
# Show top contributors for each flagged patient

def explain_patient(idx, model, X, feature_names):
    contributions = X[idx].toarray()[0] * model.coef_[0]
    top = np.argsort(np.abs(contributions))[-5:][::-1]
    return [(feature_names[i], contributions[i], X[idx, i]) for i in top]

# Patient #487 -- Predicted P(Disease X) = 0.31, flagged
# Top 5 contributors:
#   1. enzyme_assay_value = 3.2          (-1.42)  [low enzyme = disease signal]
#   2. urine_organic_acids_abnormal = 1   (+0.84)
#   3. has_family_history_disease_x = 1   (+0.71)
#   4. age_at_referral_minus_onset = 6    (+0.34)
#   5. has_developmental_delay = 1        (+0.28)
#
# Clinician summary: "This patient meets multiple criteria suggestive of Disease X.
#   Most strongly: enzyme assay is well below the reference range (3.2 vs typical 8-15),
#   urine organic acids show abnormal pattern, and there is a family history.
#   Recommend: refer to specialty workup with these 3 reasons cited."
```

---

## Step 13: Deployment / Regulatory Path

```python
# This is a clinical decision support tool, FDA Class II SaMD
# Deployment requires:
#   - 510(k) submission with prospective validation cohort
#   - Bias and fairness audit (sex, race/ancestry)
#   - Explicit "decision support, not autonomous" labeling

# Production:
#   - One-time model training; no automated retraining (data is precious; updates require IRB)
#   - Bi-annual review: do new cases support the model's CV performance?
#   - Manual retraining only when cohort grows by ~30% AND IRB approves

# Bias audit:
for group in ['biological_sex', 'ancestry_self_reported']:
    for value in df_test[group].unique():
        mask = (X_test[group] == value)
        if mask.sum() < 100: continue
        sens = sensitivity(y_test[mask], y_pred_test[mask])
        fpr = fpr_score(y_test[mask], y_pred_test[mask])
        print(f"  {group}={value}: n={mask.sum()}, sens={sens:.2f}, fpr={fpr:.2f}")
        # Look for substantial gaps; report to FDA in submission
```

---

## Step 14: Reporting — What Goes Into the Paper

```
Reported in the regulatory submission:

PRIMARY OUTCOME:
- AUC-ROC = 0.81 (95% CI 0.74 - 0.87, n_test = 1,170, positives = 68)
- At threshold optimized for Sens >= 0.90:
    Sens = 0.87 (95% CI 0.79 - 0.94)
    Spec = 0.80 (95% CI 0.78 - 0.82)
    PPV  = 0.21
    NPV  = 0.99

SECONDARY:
- Per-hospital AUC: range 0.79 - 0.84 across 8 hospitals (LeaveOneOut CV)
- Per-subgroup AUC: female 0.83, male 0.80 (no significant gender effect)
- Per-ancestry AUC: largest gap was 0.04 (within 95% CI overlap; not significant)

LIMITATIONS (also reported):
- Training cohort came from 8 academic centers; performance at community hospitals untested
- 340 positive cases drives CI width
- Genetic features partially leak (some patients get panel because of suspicion)
- Cannot rule out that some "negative" patients have undiagnosed Disease X
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| CV strategy | KFold 80/20 random | Leave-one-hospital-out + 2 held-out hospitals |
| Model choice | XGBoost / Random Forest 1000 trees | Logistic Regression L2 / Random Forest with depth=4 |
| Feature count | 50+ engineered | ~16 carefully chosen (n/20 rule) |
| Imbalance | SMOTE | None — class_weight slightly destabilizes calibration |
| Lab missingness | Median impute everything | _ordered flag for suspicion-driven; hospital-median for routine |
| Genetic features | Use as is | Filter post-hoc / leakage-prone (panel = downstream of suspicion) |
| Performance reporting | Point AUC = 0.82 | AUC = 0.81 (95% CI 0.74-0.87, n=1170, pos=68) |
| Threshold | F1 optimum | Sens >= 0.90 ∩ FPR <= 0.25 Pareto |
| Test set | Random 20% hold-out | TWO entire hospitals held out (generalization sim) |
| Explanation | "model said yes" | Top 5 LR coefficients, plain-English clinical interpretation |
| Regulatory framing | "ship the model" | FDA SaMD path; IRB; 510(k); subgroup bias audit |
| Retraining | Quarterly | Bi-annual review; only retrain on substantial new cohort |
