# From Brief to Solution — Classification Methodology

How to take a fresh business brief plus a dataset and find the matching playbook in the [classification catalog](../catalog/CATALOG.md).

---

## The Core Idea

You won't be the first person to face this problem. Almost every classification brief fits one of ~80 archetypes already cataloged. Your job:

1. **Read the brief carefully** — what are they really asking for?
2. **Inventory the dataset** — what shape are you actually working with?
3. **Match to an archetype** — which catalog category (and which 1–3 scenarios within it)?
4. **Adapt** — what's different about your problem? What changes?

This guide gives you a repeatable 6-step protocol so you don't miss anything.

---

## Step 1: Extract the Prediction Target From the Brief

The brief is usually a paragraph. Reduce it to one sentence:

> *"Given X, predict whether Y."*

If you can't write that sentence, the brief is ambiguous and you need to clarify with the stakeholder before going further.

**Examples:**

| Brief excerpt | Reduced sentence |
|---------------|------------------|
| "We want to flag suspicious transactions before approval." | Given a transaction's features, predict whether it is fraud. |
| "Identify customers likely to cancel next month." | Given a customer's recent activity, predict whether they will churn next month. |
| "Tag support tickets with the right department." | Given a ticket's text, predict which of 12 departments should handle it. |
| "Decide which loan applications to approve." | Given an applicant's financials, predict whether they will default within 24 months. |

**Watch-out:** sometimes the stakeholder phrases it as a regression ("score the transaction risk"). Push back — is the downstream decision binary (approve/decline)? If yes, it's classification. If they truly need a calibrated probability for downstream cost optimization, it's still classification — you just need to evaluate calibration as well as accuracy.

---

## Step 2: Identify the Constraints (8 Signals to Read In)

Re-read the brief looking for these 8 signals. Each one points at a catalog category.

| # | Signal | Phrases that indicate it | Catalog category it points to |
|---|--------|-------------------------|-------------------------------|
| 1 | **Imbalance signal** | "rare", "0.x%", "anomaly", "few examples of X" | Category 1 (Imbalanced binary) or 15 (Long-tail) |
| 2 | **Regulation signal** | "explain", "audit", "ECOA", "fairness", "adverse action" | Category 7 (Regulated) |
| 3 | **Real-time signal** | "under N ms", "live", "instant decision", "real-time" | Category 8 (Real-time) |
| 4 | **Cost-asymmetry signal** | "missed X costs $A, false alarm costs $B" | Category 13 (Cost-sensitive) |
| 5 | **Time-series signal** | "next hour", "predict event window", sensor stream | Category 11 (Time-series classification) |
| 6 | **Multilabel signal** | "can have multiple", "tags", "categories" (plural) | Category 5 (Multilabel) |
| 7 | **Hierarchy signal** | "category > subcategory", taxonomy, "parent class" | Category 12 (Hierarchical) |
| 8 | **Calibration signal** | "probability", "score", downstream uses the number | Category 14 (Calibrated probability) |

**Rule:** A brief can match 2–3 signals at once. That's normal. The intersections are where most real problems live (e.g., "real-time imbalanced classification with cost asymmetry" = fraud detection).

---

## Step 3: Inventory the Dataset

Open the data and answer these 10 questions in 5 minutes flat:

| # | Question | Why it matters | Quick check |
|---|----------|----------------|-------------|
| 1 | How many rows? | Eliminates models | `len(df)` |
| 2 | How many features? | Curse of dimensionality | `df.shape[1]` |
| 3 | Feature types? (numeric / categorical / text / mixed) | Determines preprocessing | `df.dtypes.value_counts()` |
| 4 | Target distribution? (balance, # classes) | Imbalance / multiclass strategy | `df[target].value_counts(normalize=True)` |
| 5 | Missingness pattern? | Some imputers leak, missingness is sometimes informative | `df.isnull().mean()` |
| 6 | Are there ID-like columns? | Drop, don't model | High cardinality + few duplicates |
| 7 | Is there a temporal axis? | Affects train/test split | Date columns or implicit ordering |
| 8 | Are there protected attributes? | Fairness audit requirement | Race, gender, age, zip |
| 9 | Any obvious leakers? | Drop before modeling | "post-event" features that wouldn't be available at decision time |
| 10 | Cardinality of categoricals? | One-hot vs target encoding | `df[col].nunique()` per categorical |

**Watch-out:** if the dataset shape contradicts the brief (e.g., brief says "we have years of data" but dataset has 500 rows), stop and confirm. Don't model the wrong thing.

---

## Step 4: Match to an Archetype Category

You now have:

- A target sentence (Step 1)
- A constraints profile (Step 2 — which of 8 signals fired)
- A data shape (Step 3)

Open [CATALOG.md](../catalog/CATALOG.md). Pick the category whose "What unites them" line matches your constraint profile.

**Quick decision tree:**

```
Is the minority class < 10% of the data?
├── YES, < 1%        → Category 1 (Imbalanced) or Category 13 (Cost-sensitive)
├── YES, 1–10%       → Category 1 (Imbalanced)
└── NO (10%+)        → continue

Is there a calendar/temporal axis with prediction over a window?
└── YES              → Category 11 (Time-series classification)

Are the features mostly text?
└── YES              → Category 3 (Text/NLP)

Are the features images / image embeddings?
└── YES              → Category 6 (Image-feature)

Are there 3+ classes?
├── YES, with hierarchy   → Category 12 (Hierarchical)
├── YES, multiple labels per row → Category 5 (Multilabel)
├── YES, one of which is 50%+ + many tiny → Category 15 (Long-tail)
└── YES, otherwise         → Category 4 (Multiclass tabular)

Is the brief regulated (lending, healthcare, hiring, criminal-justice)?
└── YES              → Category 7 (Regulated)

Is the latency budget < 100ms?
└── YES              → Category 8 (Real-time)

Is the dataset < 1,000 rows?
└── YES              → Category 9 (Small data)

Otherwise:
└── Category 2 (Balanced binary) or Category 10 (Mixed-type tabular)
```

**Multiple matches are normal.** Fraud is in Category 1 (imbalanced) AND Category 8 (real-time) AND Category 13 (cost-sensitive). Pick the most defining constraint and start there; the others become adaptations.

---

## Step 5: Drill to the Closest 1–3 Scenarios

Inside the category, scan the entries. For each, ask:

1. **Is the data shape comparable?** (rows within an order of magnitude; same feature mix)
2. **Are the constraints similar?** (latency, regulation, imbalance)
3. **Is the metric the same?** (recall at fixed precision? PR-AUC? top-k?)

Pick the 1–3 closest. Read each in full. Even if your problem isn't an exact match, you now know:

- Which models the catalog recommends (the shortlist)
- Which metric to optimize
- The most common watch-out for this archetype

**Watch-out:** if NO scenario in the catalog feels close, see Section 6 below ("the no-match path").

---

## Step 6: Adapt — List the Deltas

Now do the most important step: write a one-page "deltas" doc. Compare your problem to the chosen scenario point by point:

| Aspect | Catalog scenario | My problem | Adaptation |
|--------|------------------|------------|------------|
| Rows | 1.2M | 50K | Worry about overfitting; smaller cross-validation folds |
| Imbalance | 0.17% | 8% | Less extreme — `scale_pos_weight=12` instead of 577 |
| Latency | < 100ms | Batch overnight | Drop latency constraint; consider stacking |
| Regulation | None | ECOA-regulated | Switch primary model to Logistic + L2; XGBoost only as challenger |
| Metric | PR-AUC | KS statistic | Different evaluation — compute KS, optimize for it |

Write this BEFORE you write any code. The deltas are where most of your engineering effort will go. Copying the catalog's playbook verbatim almost never works — adaptation is the skill.

---

## The "No Match" Path

What if no scenario in the catalog feels right?

**1. Check both categories first.** The brief might be a regression problem (if so, switch to the [regression methodology](../../regression/from_brief_to_solution/METHODOLOGY.md)) or a clustering problem (no library exists yet — see [MODEL_SELECTION_GUIDE_CLUSTERING.md](../../../MODEL_SELECTION_GUIDE_CLUSTERING.md)).

**2. Decompose the problem.** Often a "weird" problem is two scenarios stitched together (e.g., "predict outage type AND severity" = multiclass classification + regression). Solve each piece against its own catalog entry.

**3. The brief might be ill-defined.** If the target sentence (Step 1) is unclear or the metric is unspecified, go back to the stakeholder.

**4. Add a new scenario to the catalog.** If you genuinely solved a novel problem, contribute the new archetype back to the catalog so future-you (or someone else) finds it next time.

---

## Adaptation Checklist (12 Questions Before Copying Any Playbook)

Run through these BEFORE you start coding the matched scenario's playbook.

1. Is the row count within an order of magnitude of the catalog scenario?
2. Is the feature count within an order of magnitude?
3. Is the imbalance level within ±5 percentage points?
4. Are the feature types comparable (numeric / categorical / text mix)?
5. Is the target a binary, multiclass, or multilabel — and does that match the scenario?
6. Is the metric the same? If not, why is yours different?
7. Is the latency budget the same?
8. Is the regulation context the same?
9. Are there protected attributes I haven't accounted for?
10. Is there a temporal split required (and the catalog might have used random split)?
11. Are there leakers in my data that the catalog scenario didn't have?
12. Is my "deployment" reality the same? (real-time API vs batch CSV vs embedded device)

If any answer is "no" or "I don't know", that's a delta you need to handle.

---

## Three Worked Examples

### Worked Example A — Easy Match

**Brief:** "Our online retailer flags 0.5% of transactions as fraud. We want to catch more fraud without flooding the support team with false alarms. We have 6 months of transaction history (1.5M rows)."

| Step | Reasoning |
|------|-----------|
| 1. Target sentence | Given a transaction, predict whether it is fraud. |
| 2. Signals | Imbalance (0.5%) + cost-asymmetry implied (false alarm has support cost) + likely real-time |
| 3. Data shape | 1.5M rows, transaction features, 0.5% positive |
| 4. Category | Category 1 (Imbalanced binary) |
| 5. Closest scenario | Scenario 1 (Credit Card Fraud Detection) — exact match in shape and constraints |
| 6. Deltas | Imbalance is 0.5% vs 0.17% in scenario 1 (less extreme — `scale_pos_weight=200`); same playbook otherwise |

**Action:** copy the [01_fraud_detection.md](../catalog/01_fraud_detection.md) playbook with adjusted `scale_pos_weight`. Done.

---

### Worked Example B — Partial Match

**Brief:** "Insurance company wants to predict which auto claims are fraudulent so investigators can prioritize. About 2% of claims are confirmed fraud. We have 200K historical claims with claim, policy, claimant, and adjuster notes (free text)."

| Step | Reasoning |
|------|-----------|
| 1. Target sentence | Given a claim, predict whether it is fraudulent. |
| 2. Signals | Imbalance (2%) + cost-asymmetry (investigator time) + text features (adjuster notes) |
| 3. Data shape | 200K rows, mixed structured + text, 2% positive |
| 4. Category | Category 1 (Imbalanced binary) — primary; Category 3 (Text/NLP) — secondary |
| 5. Closest scenario | Scenario 6 (Insurance Fraud Claim Detection) — matches structurally |
| 6. Deltas | Notes field is text-heavy — need TF-IDF + structured features fused via `ColumnTransformer`; investigator capacity defines top-k; investigators only confirm flagged claims, so non-fraud labels are uncertain |

**Action:** start from scenario 6's planned-entry summary. Pull TF-IDF preprocessing pattern from scenario 17 (spam detection). Combine. Use top-k precision as the primary metric.

---

### Worked Example C — Multi-Bucket Match

**Brief:** "Hospital wants real-time prediction at admission of which patients will need ICU within 24 hours. About 8% of admissions go to ICU. Doctors will use the score to staff ICU beds. They need to explain the score for ethics review."

| Step | Reasoning |
|------|-----------|
| 1. Target sentence | Given a patient at admission, predict whether they will need ICU within 24 hours. |
| 2. Signals | Imbalance (8%) + real-time + regulation/explainability + cost-asymmetry (missing an ICU need is dangerous) |
| 3. Data shape | likely 50K–500K admissions, mixed types, lab values |
| 4. Categories | Category 1 (Imbalanced) + Category 7 (Regulated) + Category 8 (Real-time) + Category 13 (Cost-sensitive) — 4 buckets! |
| 5. Closest scenarios | Scenario 45 (Medical Diagnosis), Scenario 57 (Hospital Readmission), Scenario 70 (Security Alert Triage) — pull patterns from each |
| 6. Deltas | Combine Logistic + L2 (from regulated category) with cost-aware threshold tuning (cost-sensitive) and ms-fast inference (real-time); SHAP for ethics review; class weights for imbalance |

**Action:** Logistic Regression with L2 + SHAP as primary; XGBoost with monotonic constraints as challenger; threshold = `argmin(FN_cost × P(FN) + FP_cost × P(FP))`; per-shift latency monitoring.

---

## Common Mistakes When Reading Briefs

These are the traps most beginners (and many experienced ML engineers) fall into.

### 1. Assuming accuracy is the metric

If the brief says "model accuracy", that's a starting point — but **always check the imbalance**. A 99% imbalanced problem can hit 99% accuracy by always predicting "no". Force yourself to compute the trivial baseline (always predict majority) and compare.

### 2. Ignoring cost asymmetry

If the brief mentions different costs for different mistakes, that IS the metric, not whatever accuracy variant they nominally asked for. Build a cost function. Optimize the threshold to minimize it.

### 3. Random train/test splits when the data has a temporal axis

If transactions / interactions / events have timestamps, **split by time**. Random splits leak future information into the training set and silently inflate validation scores by 5–20%.

### 4. Treating censored data as "negative"

In churn / attrition / time-to-event problems, customers who haven't churned yet are NOT "non-churners" — they're censored. Survival models or careful framing needed.

### 5. One-hot encoding high-cardinality categoricals

14,000 unique merchant names → 14,000 one-hot columns → catastrophic. Use target encoding or frequency encoding. Holdout encoding to avoid leakage.

### 6. Forgetting protected attributes

In regulated contexts (lending, hiring, healthcare), removing race/gender doesn't remove bias — proxy features (zip code, name) carry it. Always fairness-audit final predictions across protected groups.

### 7. Optimizing for AUC when calibration matters

AUC ranks but doesn't calibrate. If the downstream system uses the probability directly (e.g., bidding, risk pricing), use Brier score and recalibrate (Platt or isotonic).

### 8. Stopping at the first shortlist

The catalog gives you a starting shortlist. Train all of them. Only after seeing CV results do you know which one wins for your specific data.

---

## Where to Go Next

- [CATALOG.md](../catalog/CATALOG.md) — the comprehensive scenario library
- [MODEL_SELECTION_GUIDE_CLASSIFICATION.md](../../../MODEL_SELECTION_GUIDE_CLASSIFICATION.md) — the 8 framing questions for narrowing models
- [ML_PIPELINE_GUIDE.md](../../../ML_PIPELINE_GUIDE.md) — the universal 12-phase pipeline every scenario instantiates
- [FEATURE_SELECTION_GUIDE.md](../../../FEATURE_SELECTION_GUIDE.md) — when feature count is a problem
