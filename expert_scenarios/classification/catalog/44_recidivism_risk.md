# Expert Scenario 44: Recidivism Risk Prediction (Ethics, Bias, COMPAS Lessons)

> **Complexity:** Ethically fraught — the canonical case study in algorithmic bias (COMPAS / ProPublica). Historical data encodes systemic bias. The prediction directly affects pretrial detention. There is no "good model" — only models with documented trade-offs and rigorous fairness audits. This scenario is intentionally cautionary.

---

## The Brief

A state criminal justice agency requests a model to estimate the risk that an arrestee, if released pretrial, will be re-arrested within 24 months. The output is intended to help judges make pretrial release decisions.

This is the canonical landmine in fairness ML. Northpointe's COMPAS tool (2016) was extensively studied and criticized — including the ProPublica analysis showing it had different false-positive rates by race. There is no neutral technical answer here.

The brief asks you to:

1. Build a predictive model.
2. **Critically assess whether you should ship it.**
3. Document fairness across race, gender, age, neighborhood.
4. Surface the trade-offs to the agency so the policy decision is informed.

Constraints:

- AUC ≥ 0.70 (modest target — recidivism is genuinely hard to predict).
- Calibrated probabilities.
- Fairness audit non-negotiable: per-race calibration AND per-race FPR/FNR analysis.
- The model can NEVER use race as a feature. But every "race-neutral" feature (zip code, prior arrests) is correlated with race due to historical policing patterns.
- The author has the ethical obligation to recommend AGAINST deployment if fairness gaps cannot be reasonably mitigated.

This entry is included not because we recommend deploying such systems but because **understanding why they fail is essential** for any ML engineer who might be asked to build them.

---

## Step 0: A Note on Ethics

Before any technical work:

- **Recidivism prediction systems have caused documented harm.** The COMPAS-ProPublica controversy (2016) showed even sophisticated systems produce racially disparate FPR.
- **Historical training data encodes historical policing bias.** Arrests are not crimes; they reflect police patrol patterns, which themselves are biased.
- **The model's "ground truth" is itself biased.** A re-arrest is correlated with where you live, who you associate with, and police patrol density.
- **A "fair" predictive model can still produce unfair outcomes.** Even with perfectly calibrated per-race probabilities, if courts use the score differently for different defendants (which they do), the system is unfair.

**Recommendation:** before building, ask whether the agency has considered:
- Whether the prediction is necessary at all (most defendants successfully attend court without intervention).
- Whether resources should go to pretrial services (transportation, reminders) vs prediction.
- Whether the model could be replaced with a transparent point-based scale (Public Safety Assessment, used in NJ) that humans can audit directly.

This deep-dive proceeds for educational completeness but does NOT recommend deployment.

---

## Step 1: Define the Problem Type

```
Type:           Binary classification (re-arrest within 24 months = 1)
Primary Metric: AUC-ROC; per-race calibration; per-race FPR/FNR
Secondary:      Brier score; specific subgroup recall
Business Goal:  AUC ≥ 0.70; per-race FPR gap ≤ 5pp; per-race FNR gap ≤ 5pp
Constraint:     Race not in features but proxy correlates exist
                Logistic Regression only (FDA-style auditable)
                External fairness audit before any deployment recommendation
Imbalance:      ~32/68 (re-arrests are common but not the majority)
```

**Expert thinking:** AUC = 0.70 is a deliberately modest target. Recidivism is genuinely hard to predict — court attendance and re-arrest are determined by factors we mostly can't see (mental health crises, family stability, access to housing, etc.). A model claiming AUC 0.85 is probably overfit OR illegitimately incorporating proxy features for protected attributes.

---

## Step 2: Understand the Data

```
Shape: 240,000 arrestee records over 7 years
Target: re_arrested_within_24mo (32%)

Features collected at arraignment:

DEMOGRAPHIC (LEGAL TO USE):
- age_at_arrest
- gender
- has_dependents (binary)
- num_dependents
- highest_education_completed
- employment_status

CRIMINAL HISTORY:
- num_prior_arrests
- num_prior_convictions
- num_prior_violent_offenses
- months_since_last_release (if applicable)
- has_prior_failure_to_appear (FTA)
- num_prior_FTAs
- juvenile_record (binary)

CURRENT CHARGE:
- charge_severity (Misdemeanor / Felony / Class)
- charge_type (Property / Drug / Violent / Other)
- victim_status (None / Stranger / Acquaintance)
- weapon_involved (binary)

SOCIAL CONTEXT:
- has_stable_address (binary)
- months_at_current_address
- has_employer_letter (binary)
- has_family_in_area (binary)
- has_phone_number (binary)
- has_health_insurance (binary)

NOT USED AS FEATURES (audit only):
- race
- ethnicity
- zip_code (correlates strongly with race; deliberately excluded)
- neighborhood (similar reason)
```

**Expert thinking:** zip_code is excluded because residential segregation makes it a near-perfect race proxy. Arrest history features are themselves products of policing patterns — they correlate with race. There is no clean way to remove the proxy effect; the best we can do is audit and document.

---

## Step 3: EDA — Be Honest About Bias

```python
# Per-race re-arrest base rate
df.groupby('race')['re_arrested_within_24mo'].mean()
# White:           0.28
# Black:           0.41
# Hispanic:        0.36
# Asian:           0.18
# Other/Multi:     0.32

# Why is the rate higher for Black defendants? It is NOT because Black people commit more crimes.
# It is partially because:
#   - Police patrol predominantly Black neighborhoods more heavily (more arrests for the same activity)
#   - Re-arrest measures police contact, not criminal behavior
#   - Predictive policing creates feedback loops

# But our training labels reflect this biased data
# The model will learn to predict "more likely to be re-arrested = more likely to be Black"
# even though we don't include race as a feature, because proxies (prior arrests, neighborhood
# patrol density indirectly via charge type) carry the signal

# This is the ProPublica finding in microcosm
```

---

## Step 4: Data Cleaning — Acknowledge Limitations

```python
# Drop direct race / zip / neighborhood from features (NOT from audit data)
features = [...]  # exclude protected attributes
audit_data = df[['race', 'ethnicity', 'zip_code', 'neighborhood']].copy()

# === HANDLE PRIOR-ARREST BIAS ===
# Number of prior arrests is biased proxy
# Some ethicists argue you should normalize prior arrests by neighborhood arrest rate
# This is contested and not standard

# === VERIFY GROUND TRUTH ===
# "Re-arrest" includes arrests for non-criminal behavior (e.g., loitering, jaywalking)
# Some jurisdictions exclude minor offenses from the recidivism label
# Use reasonable subset:
df['re_arrest_serious'] = (df['re_arrest_charge_severity'] >= 'Misdemeanor').astype(int)
# This excludes failure-to-appear as recidivism (debatable but more legitimate)
```

---

## Step 5: Feature Engineering — Conservative

```python
# Keep features simple and transparent
df['log_prior_arrests'] = np.log1p(df['num_prior_arrests'])
df['log_prior_convictions'] = np.log1p(df['num_prior_convictions'])
df['has_violent_history'] = (df['num_prior_violent_offenses'] > 0).astype(int)
df['has_recent_arrest'] = (df['months_since_last_release'] < 12).astype(int)
df['has_stable_factors'] = (
    df['has_stable_address'].astype(int) +
    df['has_employer_letter'].astype(int) +
    df['has_family_in_area'].astype(int)
)
df['employment_stable'] = (df['employment_status'].isin(['Full-time', 'Part-time'])).astype(int)
df['charge_severity_score'] = df['charge_severity'].map({'Misdemeanor': 1, 'Felony': 2, 'Class A Felony': 3})

# Total features: ~25 (deliberately conservative — fewer features = less proxy signal)
```

---

## Step 6: Model — Logistic Regression Only

```python
# REJECT XGBoost / Random Forest / deep models for this use case
# Reasons:
# 1. Auditability: judges and oversight need to understand the score
# 2. Coefficient stability: simpler models are more stable to retrain
# 3. Bias scrutiny: easier to audit a logistic regression than a tree ensemble

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    C=0.5, penalty='l2', max_iter=500, random_state=42
)
# UNWEIGHTED (calibration target)

# Test AUC: 0.71 ✓ (just meets target)
# Per-race AUC:
#   White:    AUC 0.71
#   Black:    AUC 0.69 (model less accurate)
#   Hispanic: AUC 0.70
#   Asian:    AUC 0.75
```

---

## Step 7: THE FAIRNESS AUDIT (The Most Important Step)

```python
# Calibration per race
def calibration_by_decile(y_true, y_proba):
    deciles = pd.qcut(y_proba, 10, labels=False, duplicates='drop')
    return pd.DataFrame({
        'decile': range(10),
        'pred': [y_proba[deciles==d].mean() for d in range(10)],
        'actual': [y_true[deciles==d].mean() for d in range(10)]
    })

# Per-decile calibration White:
# Decile 1: predicted 0.08, actual 0.07 (well-calibrated)
# Decile 5: predicted 0.30, actual 0.31
# Decile 9: predicted 0.62, actual 0.59

# Per-decile calibration Black:
# Decile 1: predicted 0.13, actual 0.16 (model under-predicts at low end)
# Decile 5: predicted 0.41, actual 0.43
# Decile 9: predicted 0.71, actual 0.73 (well-calibrated at top)

# Calibration is roughly OK — model isn't claiming Black defendants are MORE likely to re-arrest than they actually are.

# But now check FPR / FNR at a fixed threshold:
threshold = 0.3  # arbitrary; let's say "predict re-arrest if P > 0.3"

# White: FPR 0.20, FNR 0.32
# Black: FPR 0.39, FNR 0.21  ← THE PROPUBLICA FINDING
# Hispanic: FPR 0.31, FNR 0.27
# Asian: FPR 0.10, FNR 0.45

# DEFINITION:
# FPR (false positive rate): of those who actually DID NOT re-arrest, what fraction did we flag?
# FNR (false negative rate): of those who actually DID re-arrest, what fraction did we miss?

# CONFLICT: The model is calibrated by race AND has different FPR/FNR by race.
# This is mathematically inevitable when base rates differ between groups.
# It's the impossibility theorem of fair classification (Chouldechova 2017).
```

**Expert insight:** the impossibility theorem says: when base rates differ across groups (which they do, because of historical bias in arrests), you cannot simultaneously achieve:
1. Calibration parity (same calibration curves)
2. Equal false positive rates
3. Equal false negative rates

The COMPAS controversy was largely about the courts choosing one (calibration) at the expense of the others (FPR parity). Both Northpointe and ProPublica were technically right; they were optimizing different fairness metrics.

This means: **there is no technical solution. The choice is policy, not engineering.**

---

## Step 8: Recommendations to the Agency

```
HONEST FINDINGS:

1. Predictive accuracy: AUC 0.71. Modestly predictive but limited.
   - Means the model ranks risk OK but doesn't predict individual outcomes well.
   - For comparison, a simple actuarial scale (4 questions) often achieves AUC 0.65.

2. Calibration parity: ACHIEVED across protected groups.
   - When the model says 30%, the actual re-arrest rate is ~30% across all races.

3. Error parity: NOT ACHIEVED.
   - Black defendants have ~2x the false positive rate of White defendants.
   - White defendants have ~1.5x the false negative rate.
   - This trade-off is mathematically inevitable given different base rates.

4. The model relies on proxy features (prior arrests, charge type) that carry
   historical bias from policing patterns.

RECOMMENDATIONS:

1. DO NOT DEPLOY this as the sole basis for pretrial detention decisions.
2. CONSIDER: replacing with a transparent point-based scale (PSA-Court) that
   judges can audit directly.
3. INVEST in pretrial services (court reminders, transportation, mental health
   support) which research shows reduce FTA more than risk prediction does.
4. IF the agency proceeds anyway:
   - Mandate human-in-the-loop review for every score above the release threshold
   - Document the FPR/FNR trade-off prominently
   - Include disclaimers about historical bias in training data
   - Audit deployed performance quarterly
   - Sunset clause: re-evaluate within 2 years

5. NEVER CLAIM the model is "race-neutral" or "fair" without specifying which
   fairness criterion is achieved (and which is not).
```

---

## Step 9: Final Evaluation

```python
# Final model: Logistic Regression with L2, calibrated probabilities
# Test set: 48,000 arrestees with 24-month follow-up
#
# Performance:
#   AUC:                       0.71
#   Brier score:               0.18
#   Calibration: well-calibrated by race
#
# Fairness gaps:
#   White-Black FPR gap:       19pp (significantly above 5pp target)
#   White-Black FNR gap:       11pp
#   Audit conclusion:          fairness target NOT met
#   Recommendation:            DO NOT DEPLOY without policy intervention
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Approach | "Build a model, ship it" | Refused to ship without policy framing; documented limitations |
| Model choice | XGBoost for accuracy | Logistic Regression for auditability |
| Feature engineering | Use all available | Deliberately exclude race + zip + neighborhood; understand proxies |
| Imbalance | class_weight | UNWEIGHTED (preserves calibration) |
| Fairness audit | Single metric (accuracy) | Calibration parity + FPR/FNR parity (acknowledge impossibility theorem) |
| Cost trade-off | Optimize F1 | Document the trade-off; let policy decide |
| Deployment recommendation | "Yes, ship" | Conditional / cautious; cite COMPAS as cautionary tale |
| Documentation | Model card | Limitations, harms, alternative options, sunset clause |
| Stakeholder communication | "AUC is 0.71" | Plain-language: model is modestly predictive; bias unavoidable; here are the trade-offs |
| Ethics | Skip | Whether-to-build is a real engineering decision |

**Takeaway: the most expert ML decision in this scenario is often "do not deploy."**
