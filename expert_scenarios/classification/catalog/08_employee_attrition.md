# Expert Scenario 8: Employee Attrition Prediction (Small Data, High Stakes)

> **Complexity:** Small dataset (only 4,800 employees), high dimensionality relative to samples, sensitive HR data with ethical concerns, need to predict attrition AND prescribe interventions, ordinal survey features, text from exit interviews, model must not be used punitively.

---

## The Brief

A mid-size tech company (4,800 employees) has 18% annual attrition. Each departure costs $45,000-$180,000 depending on role (recruiting, onboarding, lost productivity, knowledge loss). HR wants to: (1) predict which employees will leave in the next 6 months, (2) understand WHY so they can fix systemic issues, and (3) prescribe personalized retention actions. The CEO insists the model must NEVER be used to preemptively terminate employees or deny promotions. Legal requires explicit consent for data use.

This is complex because: only 4,800 data points (small for ML), many ordinal survey features that are noisy, text from exit interviews is sparse, the model must balance prediction with prescription, ethical guardrails are strict, and overfitting is a constant risk with small data.

---

## Step 1: Define the Problem Type

```
Type:           Binary Classification (leave within 6 months: yes/no)
Primary Metric: AUC-ROC (must identify at-risk employees)
Secondary:      Precision in top 200 (HR can only intervene with ~200 employees at a time)
Ethical Constraint: Model output goes to HR for POSITIVE intervention only
                    Cannot be used for termination/demotion decisions
                    Employee consent required for data inclusion
Data Challenge: Only 4,800 rows -- most ML models need 10,000+
```

---

## Step 2: Understand the Data

```
4,800 employees, 18% annual attrition = ~864 leavers/year
For 6-month prediction window: ~432 leavers (9%)

Data sources:

HRIS (HR Information System):
- employee_id, hire_date, department, team, manager_id
- job_level (1-8, from junior to C-suite)
- job_title (string, 120 unique titles)
- salary, bonus_pct, stock_grants
- last_promotion_date, promotions_in_5yr
- performance_rating (1-5, ordinal, from annual review)
- performance_rating_trend (improving, stable, declining over 3 years)
- work_location (office, hybrid, remote)
- overtime_hours_monthly (float)

Employee Survey (annual, 78% response rate):
- engagement_score (1-10 Likert scale)
- work_life_balance (1-5)
- relationship_with_manager (1-5)
- career_growth_satisfaction (1-5)
- compensation_satisfaction (1-5)
- company_culture_fit (1-5)
- would_recommend_company (1-5)
- open_text_feedback (free text, 45% filled in)

System logs:
- avg_daily_active_hours (from laptop/VPN logs)
- badge_swipes_per_week (physical office presence)
- internal_job_applications (int, applied to other roles internally)
- training_courses_completed_6mo (int)
- slack_message_count_trend (increasing/stable/decreasing)

Manager data:
- manager_tenure (how long they've been a manager)
- manager_team_attrition_rate (past attrition in this manager's team)
- manager_avg_review_score (do they rate generously or harshly)
- team_size

External:
- glassdoor_rating_trend (company's public rating)
- industry_unemployment_rate (is it easy to find jobs elsewhere?)
- competitor_hiring_rate (are competitors poaching?)
```

---

## Step 3: EDA -- What Drives Attrition?

| Finding | Implication |
|---------|------------|
| No promotion in 3+ years: 28% attrition vs 11% overall | Career stagnation is #1 driver |
| Manager with >25% team attrition: their reports leave at 26% | Bad managers cause attrition (not just bad employees) |
| Engagement score < 5: 31% attrition vs 8% for score 8+ | Engagement is strong but 22% didn't respond to survey |
| internal_job_applications > 0: 34% attrition within 6 months | Looking internally = about to look externally |
| Salary below market by >15%: 24% attrition | Compensation gap matters above a threshold |
| Remote workers: 14% attrition vs 21% for office-mandated | Flexibility matters |
| Performance rating = 4 or 5 (high performers): 15% attrition | Even top performers leave -- but for different reasons (poached) |
| Survey non-respondents: 22% attrition vs 16% for respondents | Not responding IS a signal of disengagement |

**Expert insight:** With only 4,800 rows and ~432 positive cases (leavers), we must be extremely careful about overfitting. Complex models with too many features will memorize the training data. We need aggressive feature selection and regularization.

---

## Step 4-5: Cleaning & Feature Engineering

```python
# === Small Data Strategies ===
# With 4,800 rows, every feature we add increases overfitting risk
# Rule of thumb: need ~10-20 observations per feature for stable models
# Target: 432 leavers -> 432/15 = ~28 features maximum
# We have 50+ raw features -> MUST reduce aggressively

# === Missing Data (big problem with small data) ===
# Survey data missing for 22% of employees (1,056 people)
# We CANNOT drop 22% of our data (only 4,800 rows!)
# Survey non-response IS a feature:
df['survey_responded'] = df['engagement_score'].notna().astype(int)
# Non-respondents have 22% attrition -- this flag alone is predictive

# For missing survey scores, DON'T impute with mean (misleading)
# Impute with a neutral value (3 for 1-5 scales) and add missing flags
for col in survey_columns:
    df[f'{col}_missing'] = df[col].isnull().astype(int)
    df[col] = df[col].fillna(3)  # neutral midpoint

# === Feature Engineering (targeted, domain-driven) ===

# Career velocity
df['years_since_promotion'] = (today - df['last_promotion_date']).dt.days / 365
df['expected_promotion_gap'] = avg_promotion_gap_by_level[df['job_level']]
df['promotion_overdue'] = (df['years_since_promotion'] > df['expected_promotion_gap']).astype(int)

# Compensation competitiveness
df['salary_vs_market'] = df['salary'] / market_salary_by_title[df['job_title']]
# 0.85 = 15% below market, 1.1 = 10% above market
df['total_comp'] = df['salary'] * (1 + df['bonus_pct']) + df['stock_grants']
df['underpaid'] = (df['salary_vs_market'] < 0.90).astype(int)

# Manager quality score
df['manager_quality'] = (
    df['relationship_with_manager'] * 0.4 +
    (1 - df['manager_team_attrition_rate'] / 0.3) * 0.3 +  # normalized
    df['manager_tenure'].clip(upper=5) / 5 * 0.3
)

# Engagement trajectory (more important than current score)
df['engagement_declining'] = (df['engagement_score'] < df['engagement_score_last_year']).astype(int)
df['engagement_drop'] = df['engagement_score'] - df['engagement_score_last_year']

# Flight risk signals
df['looking_internally'] = (df['internal_job_applications'] > 0).astype(int)
df['reduced_activity'] = (df['slack_message_count_trend'] == 'decreasing').astype(int)
df['presenteeism'] = (df['avg_daily_active_hours'] < 5).astype(int)  # "quiet quitting"

# Composite risk score (hand-crafted from domain knowledge)
df['flight_risk_heuristic'] = (
    df['promotion_overdue'] * 2 +
    df['underpaid'] * 1.5 +
    df['looking_internally'] * 3 +
    (df['engagement_score'] < 5).astype(int) * 2 +
    df['reduced_activity'] * 1.5
)

# Text features (from open feedback -- sparse but valuable)
# Simple: sentiment score + keyword flags
from textblob import TextBlob
df['feedback_sentiment'] = df['open_text_feedback'].apply(
    lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
)
df['feedback_mentions_leaving'] = df['open_text_feedback'].str.contains(
    'leaving|quit|resign|opportunity|elsewhere|better offer',
    case=False, na=False
).astype(int)
df['feedback_mentions_manager'] = df['open_text_feedback'].str.contains(
    'manager|boss|leadership|micromanag|toxic',
    case=False, na=False
).astype(int)

# Final feature count: 32 features (within our budget of ~28-35)
```

---

## Step 6: Feature Selection (Critical for Small Data)

```python
# With 4,800 rows, feature selection is SURVIVAL -- not optimization

# Step 1: Remove highly correlated pairs (keep the one with higher MI)
# engagement_score and would_recommend_company: r=0.82 -> drop would_recommend
# salary and total_comp: r=0.91 -> keep total_comp (more comprehensive)
# Removed: 5 redundant features

# Step 2: Mutual Information ranking
# Top 10 features by MI:
#   years_since_promotion     0.045
#   engagement_score          0.038
#   internal_job_applications 0.035
#   salary_vs_market          0.032
#   manager_team_attrition    0.028
#   performance_rating_trend  0.025
#   flight_risk_heuristic     0.024
#   overtime_hours_monthly    0.020
#   engagement_declining      0.018
#   survey_responded          0.015

# Step 3: Recursive Feature Elimination with CV (RFECV)
# Optimal number: 18 features
# Using 18 features, 5-fold CV AUC = 0.84
# Using all 32 features, 5-fold CV AUC = 0.81 (overfitting!)
# Feature selection IMPROVED performance by 3% (typical for small data)

# Final: 18 features
```

---

## Step 7-10: Model Building (Small Data Best Practices)

```python
# === Preprocessing ===
# StandardScaler for numeric (models need it)
# Ordinal keep as-is (1-5 scales are naturally ordinal, trees handle them fine)

# === Split ===
# With only 4,800 rows, we use nested cross-validation (no separate test set!)
# Outer loop: 5-fold CV (evaluation)
# Inner loop: 5-fold CV within each fold (hyperparameter tuning)
# This maximizes data usage while preventing leakage

# === Model Comparison (outer 5-fold CV) ===

# Model 1: Logistic Regression (L1 regularized)
# CV AUC: 0.82, F1: 0.48
# ADVANTAGE: Fully interpretable, coefficients tell the story

# Model 2: Random Forest (n_estimators=300, max_depth=5)
# CV AUC: 0.84, F1: 0.51
# Small max_depth CRITICAL for small data (prevents overfitting)

# Model 3: XGBoost (max_depth=3, reg_alpha=1.0, reg_lambda=2.0)
# CV AUC: 0.85, F1: 0.52
# Heavy regularization: alpha=1.0, lambda=2.0 (much more than default)

# Model 4: LightGBM (max_depth=4, num_leaves=15)
# CV AUC: 0.85, F1: 0.53

# Model 5: Logistic Regression + hand-crafted features only
# Using only: flight_risk_heuristic, engagement_score, years_since_promotion,
#             salary_vs_market, manager_quality
# CV AUC: 0.80, F1: 0.44
# SIMPLER but only slightly worse -- and fully transparent

# Model 6: Elastic Net (L1+L2, alpha=0.5)
# CV AUC: 0.83, F1: 0.49

# IMPORTANT: All tree models use max_depth 3-5 (shallow!)
# Deep trees on 4,800 rows = guaranteed overfitting
# Also: n_estimators limited to 300-500 (not 1000+)

# DECISION: Use LightGBM for prediction, Logistic Regression for explanation
# HR gets: (1) risk score from LightGBM, (2) reasons from Logistic Regression coefficients
```

---

## Step 11: Threshold & Capacity Optimization

```python
# HR team can actively intervene with ~200 employees per quarter
# (1-on-1 meetings, retention offers, role changes)

# At various thresholds:
# Top 50 flagged:   72% actually leave (36 caught) -- very precise but low coverage
# Top 100 flagged:  58% actually leave (58 caught)
# Top 200 flagged:  43% actually leave (86 caught) -- good balance
# Top 300 flagged:  34% actually leave (102 caught) -- diluted precision
# Top 500 flagged:  24% actually leave (120 caught) -- too many false alarms

# At 200 capacity:
# 86 actual leavers caught out of ~216 total leavers in 6 months = 40% recall
# 86 interventions worth doing * 65% retention success rate = 56 saved
# Value: 56 retained employees * $85K avg replacement cost = $4.76M saved
# Cost: Manager time + retention offers = ~$600K
# Net: $4.16M per 6 months = $8.3M annually

# CHOOSE: Flag top 200 risk scores per quarter
```

---

## Step 12: Prescriptive Analytics (What to DO About It)

```python
# For each flagged employee, identify the PRIMARY driver using SHAP:

# Driver clusters and prescribed actions:
# Cluster 1: Career Stagnation (38% of flagged)
#   Signal: years_since_promotion > 2.5, career_growth_satisfaction < 3
#   Action: Career pathing conversation, lateral move opportunity,
#           stretch assignment, promotion timeline discussion

# Cluster 2: Compensation Gap (22% of flagged)
#   Signal: salary_vs_market < 0.88, compensation_satisfaction < 3
#   Action: Market adjustment, equity refresh, retention bonus

# Cluster 3: Manager Problem (18% of flagged)
#   Signal: manager_team_attrition > 0.25, relationship_with_manager < 3
#   Action: Team transfer option, manager coaching/training,
#           skip-level meetings for the employee

# Cluster 4: Burnout (12% of flagged)
#   Signal: overtime_hours > 20, work_life_balance < 3, engagement declining
#   Action: Workload redistribution, sabbatical option, flexible schedule

# Cluster 5: Disengagement (10% of flagged)
#   Signal: reduced_activity, slack decreasing, presenteeism
#   Action: Re-engagement conversation, new project assignment,
#           cross-functional rotation

# Automated prescriptive output:
# "Employee #4582 (Senior Engineer, Level 5):
#  Risk Score: 0.72 (HIGH)
#  Primary Driver: Career Stagnation
#    - No promotion in 3.2 years (avg for level 5: 2.0 years)
#    - Career growth satisfaction: 2/5
#    - Applied to 2 internal roles in past quarter
#  Prescribed Actions:
#    1. Schedule career development meeting (manager + skip level)
#    2. Discuss promotion timeline or senior IC track
#    3. Offer stretch project in [area of interest from survey]"
```

---

## Step 13: Systemic Insights (Beyond Individual Prediction)

```python
# The model reveals company-wide patterns HR should fix:

# Insight 1: Engineering has 24% attrition vs 12% for Product
# Feature analysis: compensation_satisfaction is 1.5 points lower in Engineering
# Action: Engineering salary band review

# Insight 2: 3 managers have >30% team attrition (vs 18% company avg)
# Not just flagging individuals -- flagging systemic manager issues
# Action: Manager effectiveness training program

# Insight 3: Employees who go fully remote have 30% LESS attrition
# But company is pushing return-to-office
# Action: Data-driven input to remote work policy discussion

# Insight 4: The first 12 months are highest risk (22% attrition)
# New hire onboarding is the biggest opportunity
# Action: Enhanced onboarding program, 90-day check-ins, buddy system

# These systemic actions are MORE valuable than individual predictions
# Fixing a bad manager saves 10 employees, not just 1
```

---

## Step 14: Final Results

```python
# Nested 5-fold CV results:
#
# LightGBM:
# AUC-ROC:     0.85
# Precision@200: 43% (86 actual leavers in top 200 flagged)
# Recall:       40% (86 of ~216 leavers caught)
#
# Business impact (annualized):
# Employees retained through intervention: ~112/year
# Cost savings: $9.5M/year (at $85K avg replacement cost)
# Intervention costs: $1.2M/year
# NET SAVINGS: $8.3M/year
#
# Systemic insights value (harder to quantify):
# Manager training program: estimated -3% attrition company-wide
# Compensation adjustment: estimated -2% engineering attrition
# Onboarding improvement: estimated -4% first-year attrition
# Total systemic: potentially $15M+ in additional savings
```

---

## Step 15-16: Ethical Guardrails & Deployment

```python
# ETHICAL SAFEGUARDS (non-negotiable):

# 1. Model output NEVER used for termination or demotion
#    - Enforced by role-based access: only HR Business Partners see scores
#    - Not visible to hiring managers or in performance systems
#    - Legal review of all use cases

# 2. Employee awareness
#    - All employees informed their data is used for retention prediction
#    - Opt-out mechanism: employees can exclude their data
#    - Annual transparency report on how model is used

# 3. Bias monitoring
#    - Attrition risk scores should not systematically differ by
#      gender, race, age, disability status
#    - Quarterly fairness audit

# 4. Model governance
#    - Human-in-the-loop: HR reviews EVERY flagged employee before action
#    - No automated actions -- model is advisory only
#    - Regular review of intervention outcomes (did retention actions work?)

# Deployment:
# - Quarterly batch predictions (not real-time -- HR plans quarterly)
# - Dashboard showing: top 200 at-risk employees, driver clusters,
#   prescribed actions, systemic trend alerts
# - 6-month feedback loop: did flagged employees actually leave?
#   Retrain annually or when performance degrades
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Small data handling | Train LightGBM with 1000 trees | Shallow trees (max_depth=4), heavy regularization, nested CV |
| Feature count | Use all 50+ features | Aggressive reduction to 18 (4,800 rows demands it) |
| Missing surveys | Drop non-respondents (lose 22%!) | Non-response IS the feature + neutral imputation |
| Model complexity | Deep ensemble | Shallow LightGBM for scoring + Logistic Reg for explanation |
| Evaluation | Single train/test split | Nested cross-validation (outer=evaluation, inner=tuning) |
| Output | "This person will leave" | Risk score + primary driver + prescribed action |
| Scope | Individual prediction only | Systemic insights (bad managers, compensation policy, onboarding) |
| Ethics | Not considered | Consent, opt-out, anti-punishment guardrails, bias monitoring |
| Business case | "AUC is 0.85" | $8.3M/year net savings with detailed intervention ROI |
