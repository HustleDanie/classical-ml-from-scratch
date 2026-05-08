# Expert Scenario 71: Cancer Screening (Cost-Sensitive Classification)

> **Complexity:** Severe FN/FP cost asymmetry (missed cancer ≫ false biopsy), regulated under FDA/CLIA, mild imbalance (~6%), small data per cancer type, calibrated probabilities feed clinical decision rules, fairness across demographic groups under audit.

---

## The Brief

A radiology imaging center hands you 80,000 mammograms (with engineered features extracted by a CAD system — you're NOT doing image classification, you're doing classical ML on engineered radiology features) labeled by biopsy outcome over the past 5 years:

- 4,800 (6.0%) confirmed malignant
- 75,200 (94.0%) benign / no cancer

The brief asks for a model that:

- Reaches sensitivity >= 95% (catch 95%+ of cancers).
- Operates at specificity >= 80% (false-positive rate ≤ 20% — radiologists can't biopsy 50% of patients).
- Outputs a calibrated probability that matches downstream "biopsy / 6-month follow-up / 1-year follow-up" decision logic.
- Is auditable for FDA submission as a Software as Medical Device (SaMD).
- Has documented fairness across age, race, breast density category.

This is harder than fraud detection because: a missed cancer can kill someone, biopsy harm has real costs (anxiety, infection, scarring), the model must hit BOTH sensitivity AND specificity simultaneously, and FDA clearance demands bias audits.

---

## Step 1: Define the Problem Type

```
Type:           Binary classification (malignant=1, benign=0)
Primary Metric: Sensitivity at fixed Specificity = 0.80
                (alternative: Specificity at fixed Sensitivity = 0.95)
Secondary:      AUC-ROC; Brier score (calibration)
Cost Structure: FN_cost = $35,000 (delayed cancer diagnosis costs)
                FP_cost = $1,800 (unnecessary biopsy)
                Cost ratio: ~20:1 -- aggressive threshold tuning
Constraint:     FDA audit-ready; calibrated probabilities; bias-tested
Imbalance:      94/6 (moderate)
```

**Expert thinking:** the cost ratio (20:1) tells us where the threshold should sit. We optimize sensitivity at fixed specificity, NOT F1. Default 0.5 threshold will severely under-catch cancers; we'll need to push it down to maybe 0.10–0.15.

The fixed-specificity framing matters. "Sensitivity = 95% at 80% specificity" is the DEMI-ACR Pareto target for screening systems. F1 doesn't capture this constraint.

---

## Step 2: Understand the Data

```
Shape: 80,000 rows x 56 features
Target: malignant -- 0 (94%), 1 (6%)

Features (extracted by upstream CAD; we do NOT process images):

PATIENT (8):
- patient_id
- age (35-89)
- race (W/B/H/A/Other; 4% missing/declined)
- ethnicity
- prior_cancer_history (binary; cancer in any organ in family)
- mother_or_sister_breast_cancer (binary)
- BRCA_test_result (Pos/Neg/Unknown -- 88% Unknown)
- breast_density (BIRADS 1-4; 7% missing for older studies)

LESION FEATURES (28 -- engineered from imaging):
- num_lesions_detected (int, 0-15)
- max_lesion_size_mm (float; 0 if no lesion)
- lesion_shape_irregularity_score (float, 0-1)
- lesion_margin_score (float, 0-1; spiculated edges = high)
- lesion_density_score (float, 0-1)
- microcalcification_count (int)
- microcalcification_clustering_score (float)
- microcalcification_morphology_score (float)
- asymmetry_score_left_right (float)
- architectural_distortion_score (float)
- ... 18 more lesion features

CONTRAST / TEXTURE (12):
- contrast_uniformity (float)
- gradient_magnitude (float)
- texture_haralick_contrast (float)
- ... 9 more texture features

QUALITY / CONTEXT (5):
- image_quality_score (0-1)
- positioning_score (0-1)
- compression_force_kPa (float)
- prior_mammogram_available (binary; comparison helps)
- years_since_last_mammogram (float)

OPERATIONAL (3):
- screening_or_diagnostic (binary; diagnostic = symptomatic patient)
- referring_physician_specialty
- imaging_facility_id (5 facilities)
```

**Expert thinking:** key concerns immediately:
- BRCA result 88% Unknown → not because of randomness; only ~12% of women get genetic testing. Encode "Unknown" as separate category.
- Breast density 7% missing in older studies → impute with BIRADS-2 (most common) but flag.
- The 28 lesion features carry most of the signal; texture features are secondary.
- Diagnostic vs screening matters — diagnostic mammograms are biased high cancer rate (~14%) vs screening (~3%).

---

## Step 3: Exploratory Data Analysis (EDA)

```python
df['malignant'].value_counts(normalize=True)
# 0    0.94
# 1    0.06

# Key cancer-rate-by-feature stratifications
df.groupby('breast_density')['malignant'].mean()
# 1 (almost entirely fatty):    0.038
# 2 (scattered fibroglandular): 0.052
# 3 (heterogeneously dense):    0.067   <- harder to detect on imaging
# 4 (extremely dense):          0.078

df.groupby('screening_or_diagnostic')['malignant'].mean()
# 0 (screening):  0.029
# 1 (diagnostic): 0.142   <- 5x rate

df.groupby('age_band')['malignant'].mean()
# 35-44: 0.018
# 45-54: 0.041
# 55-64: 0.071
# 65-74: 0.094
# 75-89: 0.103
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Cancer rate 0.029 (screening) vs 0.142 (diagnostic) | Stratify model by visit type or include as strong feature |
| Microcalcification clustering > 0.6 → 22% cancer rate | Dominant single-feature signal |
| Lesion margin score > 0.7 (spiculated) → 38% cancer rate | Spiculated margins are high-risk |
| Architectural distortion score > 0.5 → 18% cancer rate | Less common but specific |
| BRCA Pos → 24% cancer rate | Strong but only seen in ~1% of patients |
| Prior mammogram available → 0.045 cancer rate vs 0.082 without | Comparisons help radiologists; same for model |
| Race: cancer rate similar (5.8-6.2%) across W/B/H/A | But STAGE at detection differs (Black women diagnosed later) — fairness audit needed |
| 5 facilities have cancer rates 4.8% to 7.6% | Some sampling difference; possibly population mix |

**Expert insight:** screening vs diagnostic is so different (5x cancer rate) that some teams build two separate models. We'll keep one model with "is_diagnostic" as a feature, and audit calibration separately on each subgroup.

---

## Step 4: Data Cleaning

```python
# Missing handling
df['breast_density'] = df['breast_density'].fillna(2)  # most common bracket

df['BRCA_test_result'] = df['BRCA_test_result'].fillna('Unknown')
# 88% 'Unknown' is fine — that's the clinical reality

df['race'] = df['race'].fillna('Declined')  # respect patient choice
df['ethnicity'] = df['ethnicity'].fillna('Declined')

# Outliers: DO NOT cap on lesion features
# Maximum lesion size 95mm IS the signal. Don't winsorize cancer indicators.
# But check for impossible values:
df = df[df['max_lesion_size_mm'] < 200]  # > 20cm is impossible — data error
df = df[df['compression_force_kPa'] < 200]  # physical limit
```

---

## Step 5: Feature Engineering

```python
# === COMPOSITE LESION RISK SCORES ===
df['has_suspicious_lesion'] = (
    (df['lesion_margin_score'] > 0.7) |
    (df['lesion_shape_irregularity_score'] > 0.7) |
    (df['microcalcification_clustering_score'] > 0.6)
).astype(int)

df['lesion_size_category'] = pd.cut(
    df['max_lesion_size_mm'],
    bins=[-0.1, 0, 5, 10, 20, 50, 200],
    labels=['none', 'tiny', 'small', 'medium', 'large', 'very_large']
)

# === FAMILY / GENETIC RISK COMPOSITE ===
df['hereditary_risk_score'] = (
    df['mother_or_sister_breast_cancer'].astype(int) +
    (df['BRCA_test_result'] == 'Pos').astype(int) * 3 +  # BRCA pos heavily weighted
    df['prior_cancer_history'].astype(int)
)

# === AGE × DENSITY INTERACTION ===
df['high_risk_age_density'] = (
    (df['age'] >= 50) & (df['breast_density'] >= 3)
).astype(int)
# Older women with dense breasts: hardest detection group + higher cancer rate

# === COMPARISON QUALITY ===
df['comparison_recent'] = (
    (df['prior_mammogram_available'] == 1) &
    (df['years_since_last_mammogram'] <= 2)
).astype(int)

# === IMAGE QUALITY GATE ===
df['low_quality'] = (
    (df['image_quality_score'] < 0.5) | (df['positioning_score'] < 0.5)
).astype(int)
```

---

## Step 6: Feature Selection

```python
# Mutual Information ranking
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

# Top 15:
#   microcalcification_clustering_score   0.071
#   lesion_margin_score                   0.064
#   has_suspicious_lesion                 0.061
#   max_lesion_size_mm                    0.052
#   lesion_shape_irregularity_score       0.048
#   architectural_distortion_score        0.041
#   age                                   0.037
#   screening_or_diagnostic               0.034
#   breast_density                        0.029
#   asymmetry_score_left_right            0.024
#   hereditary_risk_score                 0.018
#   ...

# Drop features with MI < 0.001 (mostly texture features that didn't add info)
# Final: 47 features
```

---

## Step 7: Preprocessing

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric = [...]  # ~35 numeric
categorical = ['race', 'ethnicity', 'BRCA_test_result', 'lesion_size_category',
               'imaging_facility_id', 'referring_physician_specialty']
binary = [...]  # ~10 binary

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('bin', 'passthrough', binary)
])
```

---

## Step 8: Train/Test Split

```python
# TIME-BASED + STRATIFIED FACILITY SPLIT
df = df.sort_values('study_date')
# 5 facilities: ensure each split sees all 5 facilities
# Time order also preserved (newer in test)

cutoff = df['study_date'].quantile(0.80)
train = df[df['study_date'] < cutoff]   # ~64K studies
test = df[df['study_date'] >= cutoff]   # ~16K studies

# Verify cancer rate similar across splits
print(train['malignant'].mean(), test['malignant'].mean())
# 0.058, 0.064 (close enough; slight upward drift is fine)
```

---

## Step 9: Baselines

```python
# Baseline 1: predict benign for all
# Sensitivity: 0%
# Specificity: 100%
# Useless

# Baseline 2: simple rule -- flag if has_suspicious_lesion=1 OR microcalcification_clustering > 0.6
# Sensitivity: 71%, Specificity: 84%
# Beats nothing-model but doesn't hit 95% sensitivity target

# Baseline 3: just use max_lesion_size_mm > 5
# Sensitivity: 64%, Specificity: 91%
# Higher specificity but worse sensitivity
```

---

## Step 10: Try Multiple Models with Stratified 5-Fold CV

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# === Model 1: Logistic Regression with L2 (regulated baseline) ===
lr = LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', max_iter=500)
# CV AUC: 0.847
# Sensitivity at 80% specificity: 0.79

# === Model 2: Random Forest ===
rf = RandomForestClassifier(n_estimators=500, max_depth=8, class_weight='balanced')
# CV AUC: 0.872
# Sensitivity at 80% specificity: 0.83

# === Model 3: Gradient Boosting ===
gbm = GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.05)
# CV AUC: 0.881
# Sensitivity at 80% specificity: 0.85

# === Model 4: XGBoost with monotonic constraints (regulated-friendly) ===
# Force monotonic constraints on age, lesion_size, microcalc_clustering (more = more cancer)
xgb = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    scale_pos_weight=15,  # 1/0.06
    monotone_constraints={'age': 1, 'max_lesion_size_mm': 1, ...}
)
# CV AUC: 0.879
# Sensitivity at 80% specificity: 0.86  <- BEST

# === Model 5: SVM with RBF kernel ===
svm = SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True)
# CV AUC: 0.864
# But not interpretable, harder to audit -- skip for FDA path

# === Model 6: Calibrated XGBoost (isotonic) ===
from sklearn.calibration import CalibratedClassifierCV
calibrated_xgb = CalibratedClassifierCV(xgb, method='isotonic', cv=5)
# CV AUC: 0.879 (calibration doesn't change ranking)
# Brier score: 0.041 (vs 0.057 uncalibrated)  <- much better calibration
# Sensitivity at 80% specificity: 0.86 (unchanged)
```

**Top performer:** Calibrated XGBoost with monotonic constraints (sensitivity 0.86 at specificity 0.80).
**Regulated baseline:** Logistic Regression L2 (sensitivity 0.79).

The 7pp sensitivity gap is meaningful — it represents ~340 cancer cases per year detected vs missed.

---

## Step 11: Imbalance Handling

```python
# scale_pos_weight = (1 - 0.06) / 0.06 = 15.67
# Already used in XGBoost. SMOTE not used (engineered features can't be safely interpolated).

# Cost-sensitive learning via sample_weight per row:
# Patients with hereditary_risk_score >= 2 weighted higher (cancer is more catastrophic in family carriers)
# But this is sensitive ground -- discuss with FDA clinical team before using
```

---

## Step 12: Hyperparameter Tuning

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'scale_pos_weight': 15,
    }
    model = XGBClassifier(**params, monotone_constraints={...}, random_state=42)
    # Custom scorer: sensitivity at fixed specificity 0.80
    return -sensitivity_at_fixed_specificity(model, X_train, y_train, target_spec=0.80)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=80)
# Best sensitivity: 0.872 at 80% specificity
```

---

## Step 13: Threshold Tuning to Cost Function

```python
# Cost function: minimize total expected cost
# Cost = FN_cost * P(FN) + FP_cost * P(FP)
# FN_cost = $35,000 (delayed diagnosis)
# FP_cost = $1,800 (unnecessary biopsy)

y_proba = calibrated_xgb.predict_proba(X_val)[:, 1]

best_thr, best_cost = 0.5, float('inf')
for thr in np.arange(0.02, 0.90, 0.02):
    y_pred = (y_proba >= thr).astype(int)
    fn = ((y_pred == 0) & (y_val == 1)).sum()
    fp = ((y_pred == 1) & (y_val == 0)).sum()
    cost = fn * 35000 + fp * 1800
    sens = (y_pred[y_val == 1] == 1).mean()
    spec = (y_pred[y_val == 0] == 0).mean()
    if sens >= 0.95 and spec >= 0.80 and cost < best_cost:
        best_thr, best_cost = thr, cost

# Best threshold: 0.082
# Sensitivity: 0.951, Specificity: 0.812 (both targets met)
# Total cost on val set: $X (much less than baseline)
```

**Expert insight:** the threshold (0.082) is far below default 0.5. This is correct given the 20:1 cost ratio. A radiologist seeing the model's flag has about 1-in-12 actual-positive odds — the model surfaces "this needs a closer look", not "this is definitely cancer." That semantic clarity is important for clinician trust.

---

## Step 14: Calibration

```python
# After isotonic calibration (already in pipeline), verify on test set:
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=20)
# Plot: should be near diagonal
# Brier score: 0.039 (well-calibrated)
# Expected calibration error: 0.011

# Per-decile calibration:
# Decile 1 (lowest prob): predicted 0.012, actual 0.014 -- well calibrated
# Decile 10 (highest):    predicted 0.84,  actual 0.81  -- well calibrated
```

---

## Step 15: Fairness Audit

```python
# FDA submission requires fairness across age, race, breast density
# Compute sensitivity at fixed specificity for each subgroup

audit_groups = ['race', 'age_band', 'breast_density']
for grp in audit_groups:
    for value in df[grp].unique():
        mask = (X_test[grp] == value)
        if mask.sum() < 100: continue
        sens = sensitivity_at_specificity(y_test[mask], y_proba[mask], target_spec=0.80)
        n = mask.sum()
        n_cancer = y_test[mask].sum()
        print(f"  {grp}={value}: n={n}, cancers={n_cancer}, sensitivity={sens:.3f}")

# Findings:
#   race=White:        n=11400, cancers=684, sensitivity=0.952
#   race=Black:        n=2200,  cancers=148, sensitivity=0.918  <- 3.4 pp gap
#   race=Hispanic:     n=1500,  cancers=84,  sensitivity=0.940
#   race=Asian:        n=900,   cancers=44,  sensitivity=0.932
#
#   age_35_44:         n=2800,  cancers=58,  sensitivity=0.897  <- worst
#   age_45_54:         n=4900,  cancers=205, sensitivity=0.945
#   age_55_64:         n=4400,  cancers=313, sensitivity=0.961
#   age_65+:           n=3900,  cancers=384, sensitivity=0.964
#
#   breast_density=4 (extremely dense): sensitivity=0.911  <- worst category
```

**Expert insight:** young patients (35-44) and extremely dense breasts (BIRADS 4) are the hardest cases — fewer training cancer examples and harder imaging. The 3.4pp gap for Black patients warrants investigation: is it a data sampling issue (under-representation of Black women in training)? Or is the model missing a feature that correlates with race in a way that hurts performance?

This audit would be in the FDA submission, with mitigation steps:
- Augment training data with more cases from under-represented groups.
- Per-group threshold tuning if fairness regulators allow it.
- Conservative threshold (lower) for the hardest groups.

---

## Step 16: Final Evaluation on Held-Out Test Set

```python
# Final model: XGBoost (monotonic constrained) + isotonic calibration + threshold 0.082
#
# Test set: 16,000 mammograms, 6.4% cancer rate
#
# Performance:
#   Sensitivity:                 0.954     ✓ (target 0.95)
#   Specificity:                 0.823     ✓ (target 0.80)
#   PPV (precision):             0.273
#   NPV:                         0.997
#   AUC-ROC:                     0.886
#   Brier score:                 0.039
#   Expected calibration error:  0.011
#
# Cost analysis (annual, ~16K mammograms):
#   FN: 47 cancers missed (5K cancers * (1-0.954))
#   FP: 2,829 false biopsies
#   Total cost: 47 * $35K + 2829 * $1.8K = $6.7M
#
# vs. Current radiologist-only baseline:
#   Sensitivity 0.91, Specificity 0.88 (literature)
#   FN: 90 cancers, FP: 1,920 biopsies
#   Total cost: $6.6M (similar!)
#
# Model + radiologist (assistive use):
#   Combined sensitivity 0.978, specificity 0.86 (papers from CAD literature)
#   Catches 30+ more cancers per year
```

---

## Step 17: Per-Patient Explainability

```python
import shap

explainer = shap.TreeExplainer(calibrated_xgb.calibrated_classifiers_[0].estimator)

# For each flagged study:
# Patient #4821, Age 52, BIRADS-3 density
# Predicted cancer probability: 0.34 (above threshold, flag for radiologist review)
# Top SHAP drivers:
#   1. microcalcification_clustering_score = 0.78  (+0.21)
#   2. lesion_margin_score = 0.81                  (+0.18)
#   3. max_lesion_size_mm = 12                     (+0.10)
#   4. age = 52                                    (+0.04)
#   5. breast_density = 3                          (+0.03)
#
# Radiologist UI surfaces:
#   "Model flagged this study because: clustered microcalcifications (high suspicion score),
#    spiculated lesion margin, and 12mm lesion. Recommend close review at the 9 o'clock
#    position."
```

---

## Step 18: Deployment / FDA Path

```python
# This is a SaMD (Software as Medical Device) under FDA 21 CFR 820
# Deployment is NOT just "ship a model"

# FDA 510(k) submission requires:
#   1. Clinical study (multi-reader, multi-case) showing equivalence to predicate device
#   2. Risk analysis (FMEA): what happens when model fails?
#   3. Bias and fairness documentation
#   4. Post-market surveillance plan

# Operational deployment:
#   - Tier 1: Pre-screen (model flags 25% of studies as high priority for radiologist)
#   - Tier 2: Concurrent read (model + radiologist see study independently; agreement -> action,
#             disagreement -> second opinion)
#   - Tier 3: Aided read (model surfaces top 5 SHAP drivers as starting points for radiologist)
#
# We deploy as Tier 3 -- "decision support", not "autonomous diagnosis"
# This is critical: the model NEVER overrides the radiologist's final read

# Monitoring (HRMP -- Hospital Real-time Monitoring Plan):
#   - Daily: sensitivity on biopsied cases (when truth is known 1-2 weeks later)
#   - Monthly: per-subgroup sensitivity (alert if any subgroup drops > 0.05)
#   - Quarterly: external audit by independent radiologist
#   - Annual: re-submit fairness analysis to FDA
#   - Auto-lockout: if rolling sensitivity drops below 0.92, model is suspended pending review
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Metric | F1 | Sensitivity at fixed Specificity (matches DEMI-ACR target) |
| Threshold | 0.5 default | 0.082 (driven by cost ratio 20:1 and Pareto target) |
| Calibration | Use raw model output | Isotonic regression -- downstream clinical decisions need it |
| Constraints | None | Monotonic constraints on age, lesion_size, microcalc (regulated-friendly) |
| Imbalance | SMOTE | scale_pos_weight; SMOTE on engineered features unsafe |
| Train/test split | Random | Time-based + facility-stratified |
| Fairness | Ignore | Per-group sensitivity audit (race, age, density); investigate gaps |
| Explanation | Probability only | Top-5 SHAP drivers per study + spatial localization hint |
| Deployment | Direct integration | Decision support tier (radiologist always overrides), never autonomous |
| Monitoring | None | Auto-lockout if sensitivity drifts; FDA re-submission annually |
| Workflow framing | "model = classifier" | Multi-tier system with clear human-in-loop for safety |
| Documentation | Code + README | Risk analysis (FMEA) + clinical study + bias audit + post-market plan |
