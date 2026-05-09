# Expert Scenario 29: ICU Severity Triage (5-Level Ordinal Multiclass, Real-Time)

> **Complexity:** 5-level ordinal classes (Mild → Moderate → Severe → Critical → Code-Blue), real-time clinical decision support, asymmetric cost matrix (under-triaging is far worse than over-triaging), missing labs are sometimes informative ("not ordered" = "not concerning"), regulated and life-critical.

---

## The Brief

A 1,200-bed academic medical center gives you 280,000 ED admission records over 4 years. They want a triage support model that classifies incoming patients into 5 acuity levels using vitals + presenting complaint + recent labs (when available). The output:

- Drives ED bed assignment (Critical bays vs Mild fast-track).
- Sets nursing acuity factor (1:1, 1:2, 1:4 staffing ratios).
- Triggers rapid-response team for predicted Critical or Code-Blue.

Constraints:

- Per-class recall: Critical ≥ 92%, Severe ≥ 85% (don't under-triage life-threatening cases).
- Quadratic-weighted kappa ≥ 0.78 (off-by-1 is OK; off-by-many is not).
- Inference < 30 seconds (during ED triage workflow).
- Calibrated probabilities feed nursing-load math.
- HIPAA-compliant; explainable per patient.
- Fairness across age, race, language.

This is harder than typical multiclass because: ordinal structure means accuracy is wrong; asymmetric cost (FN on Critical = patient death; FP = wasted bed); some patients arrive without complete data (informative missingness); and the model must integrate with established ESI (Emergency Severity Index) clinical guidelines without contradicting them.

---

## Step 1: Define the Problem Type

```
Type:           Ordinal multiclass classification (5 ordered levels)
Primary Metric: Quadratic Weighted Kappa
Secondary:      Per-class recall (especially Critical and Severe)
                Per-group calibration (race, age, language)
                Agreement with ESI scoring (not 1:1, but should correlate)
Business Goal:  QWK ≥ 0.78; Critical recall ≥ 92%; Severe recall ≥ 85%
Constraint:     < 30s inference; FDA Class II SaMD pathway; calibrated probabilities
Imbalance:      Mild 35% / Moderate 38% / Severe 18% / Critical 7% / Code-Blue 2%
```

**Expert thinking:** the cost matrix isn't symmetric. Under-triaging Critical → Severe is catastrophic (patient may die waiting); over-triaging Mild → Moderate is wasteful but not deadly. Train using cost-aware loss; then per-class recall is the safety floor.

---

## Step 2: Understand the Data

```
Shape: 280,000 ED admissions × 110 features
Target: triage_level (Mild/Moderate/Severe/Critical/Code-Blue)

Available at triage time (~5 min after arrival):

VITALS:
- temperature_c, heart_rate, respiratory_rate
- systolic_bp, diastolic_bp, oxygen_saturation
- pain_score (0-10, patient-reported)
- glasgow_coma_scale (3-15)
- mental_status (Alert/Verbal/Pain/Unresponsive)

DEMOGRAPHIC:
- age, sex, race, ethnicity, primary_language
- has_disability, requires_translator

PRESENTING COMPLAINT (free text):
- chief_complaint (string, ICD-10 mapped)
- mode_of_arrival (walked/ambulance/transferred/police)
- ambulance_priority_level (1-5 if ambulance)

HISTORY (from EHR if available):
- prior_admissions_12mo
- prior_icu_admissions_12mo
- has_diabetes, has_chf, has_copd, has_ckd, has_dementia, has_immunocompromise
- num_medications_current
- last_outpatient_visit_days_ago
- has_advanced_directive

LABS (often NOT ordered at triage; use what's available from recent prior visits):
- recent_lactate_mmol_L (40% missing)
- recent_creatinine (60% missing)
- recent_troponin (70% missing)
- recent_white_blood_cell_count (55% missing)

CURRENT VISIT LABS (rare at triage; sometimes from EMS):
- arrival_glucose (15% present)
- arrival_lactate (8% present — only when EMS pre-checked)

OPERATIONAL:
- arrival_hour
- day_of_week
- ed_current_census (how full is ED right now)
- avg_wait_time_at_arrival
```

**Expert thinking:** the chief_complaint text is the richest single feature. ICD-10 mapping helps but loses nuance ("chest pain radiating to jaw" vs "chest pain after meal" both map to chest pain but mean very different things).

---

## Step 3: EDA

```python
df['triage_level'].value_counts(normalize=True)
# Mild:       0.35
# Moderate:   0.38
# Severe:     0.18
# Critical:   0.07
# Code-Blue:  0.02

# Vital sign shock indicators
df.groupby('triage_level')[['heart_rate', 'systolic_bp', 'oxygen_saturation', 'glasgow_coma_scale']].describe()

# Mild:      HR 78, SBP 125, SpO2 98%, GCS 15
# Moderate:  HR 88, SBP 122, SpO2 97%, GCS 15
# Severe:    HR 105, SBP 110, SpO2 94%, GCS 14
# Critical:  HR 130, SBP 92, SpO2 88%, GCS 11
# Code-Blue: HR 158 or 38, SBP 65, SpO2 78%, GCS 5

# Vital signs cleanly separate the levels — strong signal

# Mode of arrival
df.groupby('mode_of_arrival')['triage_level'].value_counts(normalize=True)
# Walked-in:        Mild 60%, Moderate 30%, Severe 8%, Critical 1%, CB 0.1%
# Ambulance:         Mild 8%,  Moderate 35%, Severe 35%, Critical 18%, CB 4%
# Transferred:       Mild 5%,  Moderate 25%, Severe 40%, Critical 25%, CB 5%
# Police-escorted:   Mild 30%, Moderate 50%, Severe 15%, Critical 4%, CB 1%
```

---

## Step 4: Data Cleaning — Informative Missingness

```python
# === INFORMATIVE MISSINGNESS ON LABS ===
# Lab not ordered = clinician didn't suspect it = LOWER acuity (often)
# But also: lab not ordered because patient too sick to wait = HIGHER acuity (sometimes)
# This contextual missingness is hard to capture with a flag alone

informative_labs = ['recent_lactate', 'recent_troponin', 'recent_white_blood_cell_count']
for col in informative_labs:
    df[f'{col}_ordered'] = df[col].notna().astype(int)
    df[col] = df[col].fillna(df[col].median())

# === MISSING VITALS ===
# Vitals must be there at triage; if any missing, flag and impute conservatively
for vital in ['heart_rate', 'respiratory_rate', 'oxygen_saturation', 'systolic_bp']:
    df[f'{vital}_missing'] = df[vital].isnull().astype(int)
    df[vital] = df[vital].fillna(df[vital].median())

# === GLASGOW COMA SCALE ===
# Sometimes not assessed; assume max (15) when missing AND patient can communicate
df['glasgow_coma_scale'] = df['glasgow_coma_scale'].fillna(15)
```

---

## Step 5: Feature Engineering

```python
# === SHOCK INDEX ===
df['shock_index'] = df['heart_rate'] / (df['systolic_bp'] + 1)
# > 1.0 = shock; > 1.3 = severe shock; clinically validated

# === MODIFIED EARLY WARNING SCORE (MEWS) ===
# Standard nursing scoring system
df['mews_respiratory'] = pd.cut(
    df['respiratory_rate'], bins=[-1, 8, 14, 20, 30, 100], labels=[2, 0, 1, 2, 3]
).astype(int)
df['mews_hr'] = pd.cut(
    df['heart_rate'], bins=[-1, 40, 50, 100, 110, 130, 250], labels=[2, 1, 0, 1, 2, 3]
).astype(int)
df['mews_sbp'] = pd.cut(
    df['systolic_bp'], bins=[-1, 70, 81, 100, 199, 250], labels=[3, 2, 0, 0, 2]
).astype(int)
df['mews_temp'] = pd.cut(
    df['temperature_c'], bins=[-1, 35, 36, 38, 38.5, 50], labels=[2, 0, 0, 1, 2]
).astype(int)
df['mews_total'] = df['mews_respiratory'] + df['mews_hr'] + df['mews_sbp'] + df['mews_temp']

# === COMORBIDITY BURDEN ===
df['comorbidity_count'] = (
    df['has_diabetes'] + df['has_chf'] + df['has_copd'] + df['has_ckd'] + df['has_dementia'] + df['has_immunocompromise']
)

# === CHIEF COMPLAINT TEXT FEATURES ===
from sklearn.feature_extraction.text import TfidfVectorizer
chief_complaint_tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=20)
X_complaint = chief_complaint_tfidf.fit_transform(df['chief_complaint'])

# Specific high-acuity complaint flags (rule-based)
high_acuity_keywords = ['chest pain', 'shortness of breath', 'unresponsive', 'syncope', 'stroke',
                        'overdose', 'gunshot', 'stab', 'cardiac arrest', 'sepsis', 'altered mental']
df['has_high_acuity_complaint'] = df['chief_complaint'].apply(
    lambda c: any(kw in c.lower() for kw in high_acuity_keywords)
).astype(int)

# === ARRIVAL CONTEXT ===
df['arrived_via_ambulance'] = (df['mode_of_arrival'] == 'Ambulance').astype(int)
df['ed_busy'] = (df['ed_current_census'] > df['ed_current_census'].quantile(0.75)).astype(int)
```

---

## Step 6: Train Multiple Models — Ordinal-Aware

```python
# Try several approaches; ordinal structure matters

# === Model 1: Multinomial Softmax (ignores ordering) ===
lr = LogisticRegression(multi_class='multinomial', max_iter=500, class_weight='balanced')
# QWK: 0.71

# === Model 2: Ordinal Regression (Cumulative Logit) ===
# Train K-1 = 4 binary models; combine
# QWK: 0.78  ✓ (target)

# === Model 3: LightGBM Multiclass ===
# QWK: 0.76

# === Model 4: LightGBM Regression on ordinal index, then bin ===
# QWK: 0.79  ✓
# Critical recall: 89%
# Severe recall: 84%
```

**Top performer:** LightGBM regression on ordinal index (QWK 0.79).

---

## Step 7: Cost-Sensitive Threshold Adjustment

```python
# Default rounding doesn't account for asymmetric cost
# Pushing borderline Severe → Critical predictions UP costs us fewer FPs
# Pushing borderline Critical → Severe predictions DOWN costs us severe FNs

# Boundary-aware rounding: round UP for higher acuity bands
def cost_aware_predict(continuous_pred, boundary_aware=True):
    if boundary_aware:
        # If we're between 2.5 (boundary Moderate-Severe) and 3.0, round UP to Severe
        # If between 3.5 (boundary Severe-Critical) and 4.0, round UP to Critical
        return np.where(
            continuous_pred >= 3.4,  # near Critical
            np.ceil(continuous_pred),
            np.round(continuous_pred)
        )
    return np.round(continuous_pred)

# Critical recall improvement: 89% → 93% ✓
# QWK slight drop: 0.79 → 0.78 (acceptable trade)
```

---

## Step 8: Per-Group Audit

```python
# Audit triage decisions across protected groups
for group in ['race', 'primary_language', 'age_band']:
    for value in df_test[group].unique():
        mask = (df_test[group] == value)
        if mask.sum() < 200: continue
        critical_recall = compute_recall(y_test[mask], predictions[mask], target=Critical)
        qwk = quadratic_kappa(y_test[mask], predictions[mask])
        print(f"  {group}={value}: n={mask.sum()}, Critical recall={critical_recall:.2f}, QWK={qwk:.2f}")

# Findings:
#   race=White:           Critical recall 0.94, QWK 0.79
#   race=Black:           Critical recall 0.91, QWK 0.78
#   race=Hispanic:        Critical recall 0.92, QWK 0.77
#   primary_lang=English: Critical recall 0.93, QWK 0.79
#   primary_lang=Spanish: Critical recall 0.89, QWK 0.74  <-- gap
#   primary_lang=Other:   Critical recall 0.85, QWK 0.71  <-- bigger gap

# Investigation: language-barrier patients have less complete chief_complaint text
# Mitigation: prioritize translator; flag low-text-completeness for nurse review
```

---

## Step 9: Final Evaluation

```python
# Final model: LightGBM ordinal regression + cost-aware rounding
# Test set: 56,000 admissions over 8 months
#
# Performance:
#   QWK:                   0.78  ✓
#   Critical recall:       93%   ✓ (target 92%)
#   Severe recall:         85%   ✓ (target 85%)
#   Per-language Critical recall:
#     English:    93%
#     Spanish:    89%
#     Other:      85%   <-- gap; mitigated via translator-first protocol
#   Inference: ~12s per patient (well within 30s)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Metric | Accuracy | Quadratic Weighted Kappa + per-class recall on critical |
| Class imbalance | class_weight | Cost-aware rounding biased toward higher acuity |
| Lab missingness | Median impute | _ordered flag (informative); median imputation as fallback |
| Chief complaint | Drop or one-hot | TF-IDF on text + rule-based high-acuity keyword flag |
| Clinical scores | Skip | Compute MEWS, shock index from vitals (validated nursing scoring) |
| Ordinal handling | Multinomial softmax | Regression on index + bin (better QWK) |
| Per-group fairness | Skip | Audit per-language, per-race; flag gaps; route to translator |
| FDA pathway | Train and ship | SaMD Class II; clinical study; fairness documentation |
| Boundary rounding | Round to nearest | Round UP near Critical/Severe (asymmetric cost) |
| Real-time integration | Standalone tool | Integrate into triage workflow; nursing UI shows top SHAP drivers |
