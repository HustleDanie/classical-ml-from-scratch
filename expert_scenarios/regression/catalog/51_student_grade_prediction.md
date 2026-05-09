# Expert Scenario 51: Student Grade Prediction (Education)

> **Complexity:** Bounded ordinal target (letter grades or 0-100 scores), small-medium dataset per teacher, family-clustered data (siblings non-independent), ceiling effects, FERPA-regulated, the predictions inform interventions but must not stigmatize.

---

## The Brief

A K-12 school district gives you 4 years of student records covering 78,000 students across 32 schools. They want a model that predicts each student's end-of-semester grade ~6 weeks before the semester ends, so:

- Teachers can identify struggling students for intervention.
- Counselors can prioritize check-ins.
- District leaders can compare cohort performance.

Constraints:

- MAE within 5 grade points (on a 0-100 scale).
- Recall on at-risk students (predicted grade below 65 = D or F): ≥ 85%.
- Predictions cannot be used as inputs to grading itself (academic integrity).
- FERPA-compliant: explainable + auditable.
- The model must NOT amplify socioeconomic bias.
- Must work for new students entering the district mid-year.

This is harder than regression appears because: family-clustered data violates IID assumptions; ceiling effects (top scoring students max out at 100); intervention model must not become self-fulfilling (predicting low grade → student stigmatized → grade actually drops); and bias auditing is nuanced when "real" grades themselves carry historical bias.

---

## Step 1: Define the Problem Type

```
Type:           Bounded regression on grade points (0-100)
Primary Metric: MAE in grade points
Secondary:      Recall on at-risk students (grade < 65)
                Per-school MAPE
                Per-protected-group residual analysis
Business Goal:  MAE ≤ 5 points; recall@at-risk ≥ 85%
Constraint:     6-week-ahead prediction; FERPA-compliant; bias-aware
Target shape:   Mostly normal-ish in middle (75-90); long lower tail (struggling students)
                Ceiling effect at 100
```

**Expert thinking:** mean absolute error is the right metric for a 0-100 scale. RMSE would over-penalize the rare extreme misses; MAPE would explode for students near 0. MAE is interpretable in grade points and intuitive for stakeholders.

---

## Step 2: Understand the Data

```
Shape: 78,000 students × ~70 features
Target: end_of_semester_grade (0-100)

Features (collected at the 6-week mark):

DEMOGRAPHIC:
- student_id (anonymized)
- grade_level (K-12 → 1-13)
- enrolled_school
- enrollment_date
- is_new_to_district (binary)
- household_size
- num_siblings_in_district
- household_zip3 (HIPAA-safe)
- preferred_language
- english_learner_status (binary)
- has_iep (Individualized Education Plan)
- has_504_plan (accommodation plan)

ACADEMIC HISTORY:
- prior_year_avg_grade (NaN if new)
- prior_year_attendance_pct
- num_years_in_district
- previous_school_indicator
- standardized_test_score_recent (state assessment)
- math_skill_level (placement)
- reading_skill_level (placement)

CURRENT SEMESTER (6-week mark):
- midterm_grade (or running grade at week 6)
- attendance_pct_last_6_weeks
- num_assignments_completed
- num_assignments_late
- num_disciplinary_incidents
- num_extracurricular_activities

TEACHER:
- teacher_id
- teacher_avg_grade_in_class (rolling)
- teacher_years_experience
- class_size
- subject_area (Math/ELA/Science/Social Studies/...)

ENGAGEMENT:
- num_logins_to_LMS
- avg_assignment_submission_time_minutes_before_deadline
- num_parent_communications_30d (parent-teacher contact)

SOCIOECONOMIC PROXY (used CAREFULLY):
- school_free_lunch_eligibility_pct (school-level, NOT individual)
- household_income_quartile_estimate (from zip-level data)

PROTECTED ATTRIBUTES (NOT features; AUDIT ONLY):
- gender
- race
- ethnicity
- english_learner_status (already a feature, but also audited)
```

**Expert thinking:** household income (even at zip3 level) is a strong predictor of grades because of access to tutoring, materials, and stability. But predicting that a student in a low-income zip will get a low grade can become a self-fulfilling prophecy. We include it but audit residual gaps.

---

## Step 3: EDA

```python
df['end_of_semester_grade'].describe()
# count    78,000
# mean      82.4
# std       11.8
# min        0
# 25%       77
# 50%       85
# 75%       92
# max      100
# Skewness: -1.4 (left-skewed; ceiling effect)

# Per-school distribution
df.groupby('enrolled_school')['end_of_semester_grade'].mean()
# Range: 76 (Title I school) to 89 (high-resource school)

# At-risk rate by demographic
df['is_at_risk'] = (df['end_of_semester_grade'] < 65).astype(int)
df.groupby('english_learner_status')['is_at_risk'].mean()
# 0: 7%
# 1: 18%

df.groupby('has_iep')['is_at_risk'].mean()
# 0: 7%
# 1: 32% -- IEP students are at much higher risk

# Strongest single feature
df['midterm_grade'].corr(df['end_of_semester_grade'])  # 0.84 -- THE dominant feature
df['attendance_pct_last_6_weeks'].corr(df['end_of_semester_grade'])  # 0.51
df['prior_year_avg_grade'].corr(df['end_of_semester_grade'])  # 0.71
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Midterm grade r=0.84 with end-of-semester | Dominant predictor; everything else is incremental |
| EL students have 2.5x at-risk rate | Need EL-specific modeling or attention |
| IEP students have 4.6x at-risk rate | Different model dynamics; consider per-IEP-status models |
| Ceiling effect at 100 | Bounded target; consider beta regression or just clip predictions |
| Family clustering: siblings have correlated grades | Account for clustered data |

---

## Step 4: Data Cleaning

```python
# === HANDLE MID-YEAR ENROLLMENTS ===
df['has_prior_year_data'] = df['prior_year_avg_grade'].notna().astype(int)
df['prior_year_avg_grade'] = df['prior_year_avg_grade'].fillna(df['prior_year_avg_grade'].median())

# === DOMAIN-SPECIFIC OUTLIER HANDLING ===
# Some grades are 0 due to missing assignments; not the same as failing
# Distinguish: actual 0 vs incomplete assignments
df = df[df['num_assignments_completed'] >= 1]

# === BOUNDS HANDLING ===
# Don't transform target -- 0-100 scale is interpretable
# We'll just clip predictions at 0 and 100

# === HANDLE SMALL CLASSES ===
# If a teacher has < 10 students, exclude from training (not enough signal)
# Their students still get predictions from a global model
small_classes = df['teacher_id'].value_counts()
small_class_teachers = small_classes[small_classes < 10].index
df_train = df[~df['teacher_id'].isin(small_class_teachers)]
```

---

## Step 5: Feature Engineering

```python
# === ATTENDANCE-BASED RISK ===
df['attendance_concerning'] = (df['attendance_pct_last_6_weeks'] < 0.85).astype(int)
df['chronically_absent'] = (df['attendance_pct_last_6_weeks'] < 0.90).astype(int)

# === ENGAGEMENT TRAJECTORY ===
df['lms_engagement_score'] = (
    np.log1p(df['num_logins_to_LMS']) +
    (df['num_assignments_completed'] / (df['num_assignments_completed'] + df['num_assignments_late'] + 0.1)) +
    (df['avg_assignment_submission_time_minutes_before_deadline'] / 60).clip(-2, 24)
)

# === HISTORICAL PERFORMANCE ===
df['grade_trend'] = df['midterm_grade'] - df['prior_year_avg_grade']
df['outperforming_history'] = (df['grade_trend'] > 5).astype(int)

# === TEACHER × STUDENT INTERACTION ===
df['relative_to_class_avg'] = df['midterm_grade'] - df['teacher_avg_grade_in_class']

# === PARENT INVOLVEMENT ===
df['has_recent_parent_contact'] = (df['num_parent_communications_30d'] >= 1).astype(int)

# === GRADE-LEVEL FEATURES ===
df['is_transition_year'] = df['grade_level'].isin([6, 9]).astype(int)  # entering middle/high school

# === ACCOMMODATIONS AS POSITIVE FEATURE ===
# Students with proper accommodations may perform BETTER than predicted
# Don't treat IEP/504 as a negative; they're factors that partially explain variance
df['has_accommodations'] = (df['has_iep'] | df['has_504_plan']).astype(int)
```

---

## Step 6: Train/Test Split — Cohort-Based

```python
# Split by SCHOOL YEAR, not random
# Train: years 1-3
# Validation: year 3 last semester
# Test: year 4

# This way the model is tested on a fresh cohort
# (Unlike random split where same student appears in train and test across semesters)
```

**Expert insight:** an even more rigorous test: leave-one-school-out. The model must generalize to schools it hasn't seen.

---

## Step 7: Try Multiple Models

```python
# === Model 1: Linear Regression ===
# MAE: 7.4 points

# === Model 2: ElasticNet ===
# MAE: 6.8 points

# === Model 3: Random Forest ===
# MAE: 5.4 points

# === Model 4: Gradient Boosting ===
# MAE: 4.8 points

# === Model 5: LightGBM ===
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(
    objective='regression_l1',  # MAE-aligned objective
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
# MAE: 4.2 points  ✓ (target 5)
# Recall on at-risk (grade < 65 predicted): 87%  ✓
```

**Top performer:** LightGBM with MAE objective (4.2 grade points).

---

## Step 8: Bias Audit

```python
# Compute residuals (actual - predicted)
df['residual'] = df['actual_grade'] - df['predicted_grade']

# By protected group
print(df.groupby('race')['residual'].describe())
# White:    mean -0.2 (slight under-prediction; performing slightly better than predicted)
# Black:    mean -0.8 (predicted lower than actual by 0.8 pts)  <-- favorable but small
# Hispanic: mean -0.5
# Asian:    mean +0.4
# AI/AN:    mean -1.1  <-- larger gap but small N

# By EL status:
# Non-EL: mean -0.1
# EL:     mean -1.4 (predicted lower; actual better than predicted) -- be careful

# Disparate impact in at-risk flagging:
print(df.groupby('race')['flagged_at_risk'].mean())
# White:    7.8%
# Black:    11.6% (over-flagged)
# Hispanic: 10.2%
# Asian:    5.1%
# AI/AN:    14.8%

# DI ratio for race: 0.67 -- BELOW 0.80 threshold
# Action: investigate. Are SES proxies driving over-flagging?
```

**Expert insight:** in education, an over-flagged student gets unnecessary intervention attention; an under-flagged one misses help. Both have costs. The audit reveals systematic over-flagging of Black and Native American students at this district — likely driven by SES proxies. Fix: investigate which features are driving and consider whether to drop them.

---

## Step 9: Final Evaluation

```python
# Final model: LightGBM with MAE loss on year-4 test
#
# Performance:
#   Overall MAE:                4.2 grade points  ✓ (target 5)
#   At-risk recall:             87%  ✓ (target 85%)
#   Per-school MAE: range 3.8 - 5.6 (worst is Title I school)
#   Per-grade-level MAE: range 3.5 (10th grade) - 5.3 (1st grade)
#
# Bias diagnostics:
#   Race residual gap: White +0.0, Black -0.8 (model under-predicts Black students)
#   This means Black students score HIGHER than the model expects -- not a bias against
#   them in the bad direction
#
#   Disparate impact: at-risk flagging gap by race = 0.67 (CONCERN; below 0.80 threshold)
#   This is the OPPOSITE of grade prediction: the model OVER-FLAGS Black students despite
#   under-predicting them on average. Means flagging variance is high for that group.
#
# Mitigations:
#   - Increase intervention threshold for over-flagged groups (more conservative)
#   - Drop SES proxy features as candidate
#   - Per-group threshold tuning
```

---

## Step 10: Deployment

```python
# Production:
#   1. Per-semester batch prediction at week 6
#   2. Flag at-risk students; surface to counselors and teachers
#   3. NO automated interventions; always require teacher approval
#
# Annual retraining:
#   - Add prior year data
#   - Re-audit bias; track DI ratio quarterly
#   - Document feature importance changes
#
# Stakeholder use:
#   - Counselor dashboard: list of flagged students with top SHAP drivers
#   - Teacher view: see prediction + supportive context (NOT a stigma label)
#   - Parent view: NOT directly shown the prediction; teacher uses it as one input
#
# Monitoring:
#   - End-of-semester realized vs predicted (rolling MAE)
#   - Quarterly DI audit
#   - Feature drift detection
#   - Intervention outcome tracking: did flagged students improve?
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Loss function | RMSE | MAE (matches stakeholder reporting; less affected by outliers) |
| Bounded target | Beta regression | Just clip; 0-100 is interpretable |
| Train/test split | Random | Year-based (cohort split) |
| Family clustering | Ignore | Account for sibling correlation in CV |
| Protected attributes | Drop, train, ship | Drop from features; use ONLY in audit phase |
| Intervention | Auto-flag | Counselor approval required; teachers retain agency |
| Bias direction | Look at one direction | Audit BOTH grade prediction AND at-risk flagging (different gaps) |
| Disparate impact | Ignore | Compute DI ratio; flag if < 0.80 |
| New students | "Need history" | `has_prior_year_data` flag; works without history |
| Per-school audit | Aggregate | Per-school MAE; alert if gap > 1.5 grade points |
| Stigma awareness | Just predict | Feature design avoids self-fulfilling-prophecy via parent_involvement, etc. |
