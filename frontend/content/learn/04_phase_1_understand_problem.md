# Phase 1: Understand the Problem

> Before any code, before any model — read the brief, inventory the data, pick the metric. The 5% of project time that determines the other 95%.

---

## What this phase is for

Most ML projects fail in Phase 1. The team builds an excellent model for the wrong problem, picks a metric that doesn't match the business question, or misses a constraint (regulation, latency, fairness) that should have been a primary design input. None of those mistakes get caught later — the model "works" technically while quietly failing the business.

Phase 1 is the cheap, fast, high-leverage phase. You're producing four artifacts:

1. **A one-sentence problem statement** — what you predict, given what.
2. **A constraint profile** — imbalance, regulation, latency, cost asymmetry, fairness, calibration.
3. **A data inventory** — rows, features, types, missingness, leakers, temporal axis.
4. **A metric pick** — primary + secondary. Both come from the brief, not from habit.

If any of these four are missing or wrong, **stop and fix them before Phase 2**.

---

## Where you are in the pipeline

```
►► PHASE 1 — Understand the Problem   ◄◄  (you are here)
   PHASE 2 — Data Exploration & Cleaning
   PHASE 3 — Feature Selection & Preprocessing
   PHASE 4 — Model Selection & Training
   PHASE 5 — Optimization
   PHASE 6 — Evaluation & Validation
   PHASE 7 — Deployment
```

## How much time to spend here

Roughly **5% of total project time**. Maybe 1-2 hours on a normal-sized problem. The cost of skipping is enormous; the cost of doing it carefully is tiny.

---

## Step 1.1 — Define the Problem Type

### What it is

Reduce the brief to a single sentence in the form:

> *"Given **X**, predict **Y**."*

If you can't write that sentence, the brief is ambiguous — go back to the stakeholder.

**Examples:**

| Brief excerpt | Reduced sentence |
|---|---|
| "Flag suspicious transactions before approval." | Given a transaction's features, predict whether it is fraud. |
| "Identify customers likely to cancel next month." | Given a customer's recent activity, predict whether they will churn within 30 days. |
| "Predict house sale prices for the agent tool." | Given a listing's features, predict the sale price in dollars. |
| "Tag support tickets with the right department." | Given a ticket's text, predict which of 12 departments should handle it. |
| "Cluster customers for marketing campaigns." | Given a customer's behaviour, assign them to one of K segments. |

Then match the verb to one of five problem types:

| You predict… | Problem type | Output shape |
|---|---|---|
| A number (price, count, time) | **Regression** | continuous scalar |
| A binary outcome (yes/no, fraud/not) | **Binary classification** | {0, 1} or probability |
| One of 3+ named classes | **Multiclass classification** | {0, 1, 2, ...} or probability vector |
| Multiple labels per row (a document is "finance" AND "europe") | **Multilabel classification** | binary vector |
| Group assignment with no labels | **Clustering** | cluster ID |
| Time-until-event with censoring (subscription end, equipment failure) | **Survival analysis** | hazard function / time |

### Watch out for…

- **"Score the risk" briefs**: the stakeholder phrases it as regression, but the downstream decision is binary (approve/decline). That's classification with calibrated probabilities, not regression.
- **"Predict which class"** when 3 classes are radically different sizes (one class is 50% + many tiny ones): that's *long-tail* classification — Category 15 in the catalog. Standard multiclass under-represents the rare classes.
- **"Cluster" briefs that are really classification in disguise**: if the stakeholder has labels in their head ("VIP customers", "at-risk", "new"), they want classification with those labels — clustering is unsupervised and may not produce the segments they want.

---

## Step 1.2 — Inventory the Data

### What it is

Open the data file. Answer these **10 questions in 5 minutes flat**:

| # | Question | Why it matters | Code |
|---|---|---|---|
| 1 | How many rows? | Eliminates models | `len(df)` |
| 2 | How many features? | Curse of dimensionality | `df.shape[1]` |
| 3 | Feature types? | Determines preprocessing | `df.dtypes.value_counts()` |
| 4 | Target distribution? | Imbalance + transformation strategy | `df[target].value_counts(normalize=True)` for classification; `df[target].describe()` + `.skew()` for regression |
| 5 | Missingness pattern? | Some patterns are informative | `df.isnull().mean()` |
| 6 | ID-like columns? | Drop, don't model | high cardinality + few duplicates |
| 7 | Temporal axis? | Affects train/test split | date columns or implicit ordering |
| 8 | Protected attributes? | Fairness audit | race, gender, age, zip |
| 9 | Obvious leakers? | Drop before modeling | "post-event" features not available at decision time |
| 10 | Categorical cardinality? | One-hot vs target encoding | `df[col].nunique()` per categorical |

```python
import pandas as pd

# Quick 5-minute inventory
df = pd.read_csv('data.csv')

print('Shape:', df.shape)
print('\nDtypes:'); print(df.dtypes.value_counts())
print('\nTarget distribution:'); print(df['target'].value_counts(normalize=True))
print('\nMissingness:'); print(df.isnull().mean().sort_values(ascending=False).head(10))
print('\nCardinality of categoricals:')
for c in df.select_dtypes(include='object'):
    print(f'  {c}: {df[c].nunique()} unique')
```

### Watch out for…

- **Dataset shape contradicts the brief** ("we have years of data" but the file has 500 rows): stop, confirm with the stakeholder.
- **All features are post-event** (e.g., predicting "did the patient die" with "cause of death" as a feature). That's leakage. Drop and re-check.
- **Target labels are inconsistent** (50% are `True`/`False`, 30% are `1`/`0`, the rest are nulls). Clean before modeling.
- **One column has 99% missingness**: it's either a sparse signal (drop) or a really important rare feature (flag + investigate). Decide deliberately.

---

## Step 1.3 — Identify the Constraints (8 Signals)

The brief always contains constraints that should drive the metric, the model family, and the deployment design. Re-read it looking for these 8 signals:

| # | Signal | Phrases that indicate it | What it implies |
|---|---|---|---|
| 1 | **Imbalance** | "rare", "0.x%", "anomaly", "few examples of X" | PR-AUC over ROC-AUC; class_weight or scale_pos_weight; threshold tuning |
| 2 | **Regulation** | "explain", "audit", "ECOA", "FCRA", "EEOC", "adverse action", "FOIA" | Prefer linear models; calibrated probabilities; per-row explanations; per-group fairness |
| 3 | **Real-time** | "under N ms", "live", "instant decision", "real-time" | Single model (no stacking); profile latency end-to-end including preprocessing |
| 4 | **Cost asymmetry** | "missed X costs $A, false alarm costs $B" | The metric IS a cost function. Cost-driven threshold tuning. |
| 5 | **Time series** | "next hour", "predict event window", sensor stream | Time-based train/test split; rolling features; walk-forward CV |
| 6 | **Multilabel** | "can have multiple", "tags", "categories" (plural) | One-vs-rest classifiers; per-label metrics; classifier chains |
| 7 | **Calibration** | "probability", "score", downstream uses the number | Brier score; isotonic / Platt calibration; calibration plot in the report |
| 8 | **Fairness** | "audit", "disparate impact", "equal opportunity", protected groups | Per-group metrics; group-aware threshold; in-processing constraints |

A real brief fires **2-3 signals at once**. That's normal. Fraud is imbalanced + cost-asymmetric + real-time + (often) calibrated. The combinations matter more than any single signal.

---

## Step 1.4 — Choosing a Metric

The metric is the first decision, and the one beginners get wrong most often. Get it right and the rest of the pipeline becomes obvious. Get it wrong and you'll ship a model that scores 0.95 on something nobody asked for.

### When to reach for each — the signals

Brief signals that should immediately raise the metric flag:

- **Class imbalance** (< 30% positive) → PR-AUC or recall@k, never accuracy alone.
- **Cost asymmetry** ("a missed fraud costs $5K, a false alarm $15") → cost-weighted function.
- **Capacity constraint** ("inspectors can visit 3,500/quarter") → recall@k where k is the budget.
- **Probabilistic decisions** (pricing, risk scoring) → calibration matters; Brier score.
- **Skewed regression target** (median $163K, P99 $720K) → MAE or MAPE, not RMSE.
- **Asymmetric regression cost** → quantile loss.

### The 12 metrics in detail

**Classification metrics**

| # | Metric | What it is | Use when | Code | Watch out for |
|---|---|---|---|---|---|
| 1 | **PR-AUC** (Average Precision) | Area under the precision-recall curve. Threshold-free. | Positive class < 30%; you care about catching positives more than avoiding false alarms. | `average_precision_score(y_val, probs)` | PR-AUC under 0.3 is normal at 1% prevalence — don't compare absolute values across datasets. |
| 2 | **Recall@k / Precision@k** | Among the top-k items ranked by score: how many are positive (precision@k) and what fraction of all positives did we catch (recall@k). | Fixed budget — fraud queues, marketing audiences, content moderation, lead scoring. Brief mentions a number ("50K calls/day", "3,500 inspectors"). | `np.sort(probs)[-k]` for threshold; `y[np.argsort(-probs)[:k]].sum() / y.sum()` for recall@k. | k comes from the brief, not from intuition. |
| 3 | **F1 / Fβ** | Harmonic mean of precision and recall. Fβ favours recall (β > 1) or precision (β < 1). | Roughly balanced classes, no strong cost asymmetry, single threshold-dependent number. | `fbeta_score(y_val, preds, beta=2)` | F1 changes with the threshold — always report which threshold you used. |
| 4 | **ROC-AUC** | Area under the TPR-vs-FPR curve. | Balanced classes (≥ 30% positive); you care about rank ordering. | `roc_auc_score(y_val, probs)` | ROC-AUC 0.85 under 1% prevalence is *mediocre*, not great. Always pair with PR-AUC when imbalanced. |
| 5 | **Cost function** | Custom money metric: `c_fp × FP + c_fn × FN`. | Brief gives explicit costs. | `15 * FP + 5000 * FN` — write the function yourself. | Tune the threshold on val, report cost on test. Same fold = cherry-picking. |
| 6 | **Brier score** | Mean squared error between predicted probability and outcome. | The score is consumed as a probability — insurance pricing, risk scoring, downstream cost-multiplier; compliance demands calibration. | `brier_score_loss(y_val, probs)` | Brier conflates calibration and sharpness — always pair with PR-AUC or ROC-AUC. |

**Regression metrics**

| # | Metric | What it is | Use when | Code | Watch out for |
|---|---|---|---|---|---|
| 7 | **MAE** (Mean Absolute Error) | Average absolute error `\|y − ŷ\|`. Same units as target. | Skewed or long-tailed target; want interpretable units; not dominated by outliers. | `mean_absolute_error(y_val, preds)` | MAE doesn't penalise large errors more than small ones. |
| 8 | **RMSE** (Root Mean Squared Error) | Root of mean squared error. Penalises large errors super-linearly. | Roughly symmetric target; cost scales super-linearly with error. | `mean_squared_error(y_val, preds, squared=False)` | On skewed targets the top 1% dominates. Always pair with MAE on raw target, or report RMSE on log-target. |
| 9 | **MAPE / MdAPE** | Mean (or median) absolute percentage error. | Target is strictly positive; stakeholders ask "how off, in percent?". | `np.mean(np.abs((y_val - preds) / y_val))` | Explodes near zero; under- and over-predictions are asymmetric. |
| 10 | **Quantile loss** | Loss for predicting a specific quantile of the conditional distribution. | Need uncertainty bands; asymmetric cost (under- vs over-predicting demand). | `lgb.LGBMRegressor(objective='quantile', alpha=0.9)` | Predictions at q=0.5 are the median, not the mean. |
| 11 | **R²** | Fraction of variance explained vs predicting the mean. | Stakeholder communication; comparing models on same dataset. | `r2_score(y_val, preds)` | Not a primary production metric — comparison number, not action number. |
| 12 | **Log loss / cross-entropy** | `-Σ y log(p) + (1-y) log(1-p)`. Training-time loss for most classifiers. | Sanity-check on validation; same scale as training loss. | `log_loss(y_val, probs)` | Hard for stakeholders; PR-AUC + Brier is the human-readable equivalent. |

### Decision flowchart

```
Classification?
├── Imbalance < 30%? 
│   ├── Yes → PRIMARY: PR-AUC or recall@k
│   │       SECONDARY: Brier (if probs consumed)
│   │       TERTIARY: cost function (if costs given)
│   └── No → PRIMARY: ROC-AUC + F1
│           SECONDARY: Brier

Regression?
├── Right-skewed target → PRIMARY: MAE (or MAE on log target)
│                       SECONDARY: MAPE if strictly positive
├── Symmetric           → PRIMARY: RMSE
│                       SECONDARY: MAE
└── Asymmetric cost     → PRIMARY: quantile loss at relevant q
```

### Common pairings from the catalog

| Brief shape | Primary | Secondary | Tertiary |
|---|---|---|---|
| Fraud (extreme imbalance + cost asymmetry) | PR-AUC | recall@k | cost ($) |
| Customer churn (moderate imbalance + ranking) | PR-AUC | recall@k | Brier |
| Hospital readmission (regulator) | PR-AUC | Brier | per-group recall |
| House price (skewed target) | MAE | MAPE | RMSE on log |
| Insurance claims (zero-inflated, long-tail) | MAE on positives | quantile@0.9 | Brier on stage 1 |
| Demand forecasting (asymmetric) | quantile@0.5 | quantile@0.9 | MAE |

If the brief doesn't fit any of these rows, **the brief is incomplete**.

### Habit metrics to avoid

| Habit metric | Why it's usually wrong |
|---|---|
| **Accuracy** | Useless under imbalance — predict "no fraud" → 99.9% accuracy. |
| **ROC-AUC alone** (heavy imbalance) | Stays optimistic. Useless ranker scores 0.5; mediocre ranker scores 0.85. |
| **F1 at threshold 0.5** | Assumes symmetric costs. Almost no business has them. |
| **RMSE on raw skewed target** | Dominated by the top 1% of rows. |

---

## Step 1.5 — Reading a Brief — a 6-step protocol

Combine everything above into a repeatable protocol you run on every new brief.

### 1. Extract the prediction target

Reduce the brief to one sentence: *"Given X, predict Y."* If you can't write that sentence, the brief is ambiguous — clarify before continuing.

### 2. Identify the 8 constraints

Re-read for: imbalance, regulation, real-time, cost asymmetry, time series, multilabel, calibration, fairness. A real brief fires 2-3 of these.

### 3. Inventory the dataset

Run the 10-question quick check in 5 minutes flat.

### 4. Match to a catalog archetype

Open [/scenarios](/scenarios) and find the archetype matching your constraint profile. The catalog has ~155 examples; chances are someone has solved this shape before.

```
Minority class < 10%?    → imbalanced binary
Temporal axis with windows? → time-series classification
Mostly text?              → text / NLP
3+ classes?               → multiclass (with sub-cases)
Regulated context?        → regulated (linear preference)
Latency < 100ms?          → real-time
Dataset < 1,000 rows?     → small-data
```

Multiple matches are normal. Fraud is imbalanced AND real-time AND cost-sensitive — pick the most defining constraint, the others become adaptations.

### 5. Drill to the closest 1-3 scenarios

Within the matched category, scan entries. Pick the 1-3 closest in data shape (rows within an order of magnitude), constraints (latency, regulation, imbalance), and metric.

### 6. Write a "deltas" doc

Compare your problem to the chosen scenario point by point:

| Aspect | Catalog scenario | My problem | Adaptation |
|---|---|---|---|
| Rows | 1.2M | 50K | Worry about overfitting; smaller CV folds |
| Imbalance | 0.17% | 8% | Less extreme — `scale_pos_weight=12` instead of 577 |
| Latency | < 100ms | Batch overnight | Drop latency; consider stacking |
| Regulation | None | ECOA | Switch primary to LogReg + L2; XGB only as challenger |
| Metric | PR-AUC | KS statistic | Different evaluation — compute KS, optimize for it |

The deltas are where most of your engineering work will go. Copying a scenario verbatim almost never works — adaptation is the skill.

---

## Decision tree for this phase

```
Start with the brief.

  Is the prediction target clear? Can I write "Given X, predict Y"?
    └── No → Clarify with stakeholder. Don't proceed.

  Which of the 5 problem types?
    └── Pick one. Match it to the catalog category.

  Inventory the data: 10 questions in 5 minutes.
    └── Dataset and brief consistent? → continue
    └── Inconsistent? → clarify

  Which of 8 constraint signals fired?
    └── List them. Each is a downstream decision.

  Pick the metric.
    ├── Imbalanced classification     → PR-AUC + recall@k
    ├── Cost asymmetry given          → cost function
    ├── Probability consumed          → add Brier
    ├── Skewed regression positive    → MAE + MAPE
    ├── Symmetric regression          → RMSE + MAE
    └── Asymmetric cost regression    → quantile loss

  Write deltas doc vs closest catalog scenario.
    └── Now you're ready for Phase 2.
```

---

## Common mistakes in this phase

| Mistake | Why bad | Fix |
|---|---|---|
| Assuming accuracy is the metric | Predicts majority class → wrong-but-impressive number | Force yourself to compute the trivial baseline. |
| Skipping the 10-question data inventory | Wastes weeks training on the wrong data | 5 minutes upfront saves 5 days later. |
| Ignoring cost asymmetry | The "F1" or "AUC" you optimised was the wrong target | If costs are in the brief, build the cost function. |
| Random train/test split when data has temporal axis | Leaks future into training; metrics inflate 5-20% | Use time-based split (see Phase 3). |
| Treating censored data as "negative" | Customers who haven't churned yet aren't "non-churners" | Survival models or careful framing. |
| Forgetting protected attributes | Removing race/gender doesn't remove bias — proxies carry it | Audit predictions across protected groups (Phase 6). |
| Optimising for AUC when calibration matters | AUC ranks but doesn't calibrate; downstream use breaks | Use Brier; calibrate (Platt or isotonic). |
| Stopping at the first scenario match | One match isn't enough; data shape may not fit | Drill to 1-3 closest, then write deltas. |

---

## Worked examples

### Easy match

**Brief:** "Online retailer flags 0.5% of transactions as fraud. We want more fraud caught without flooding support. 6 months of history, 1.5M rows."

| Step | Reasoning |
|---|---|
| Target sentence | Given a transaction, predict whether it is fraud. |
| Problem type | Binary classification. |
| Constraints | Imbalance (0.5%) + cost asymmetry (implied) + likely real-time. |
| Data shape | 1.5M rows, transaction features, 0.5% positive. |
| Catalog match | Category 1 (Imbalanced) — [01_fraud_detection](/scenarios/classification/01_fraud_detection). |
| Metric | PR-AUC + recall@analyst-capacity + cost function. |
| Deltas | Imbalance 0.5% vs 0.17% — `scale_pos_weight ≈ 200` instead of 577. |

### Partial match

**Brief:** "Insurance company wants to predict fraudulent auto claims so investigators can prioritise. 2% confirmed fraud. 200K claims with claim, policy, claimant, and adjuster notes (free text)."

| Step | Reasoning |
|---|---|
| Target sentence | Given a claim, predict whether it is fraudulent. |
| Problem type | Binary classification. |
| Constraints | Imbalance (2%) + cost asymmetry (investigator time) + text features. |
| Data shape | 200K rows, mixed structured + text, 2% positive. |
| Catalog match | Category 1 (Imbalanced) primary; Category 3 (Text/NLP) secondary. |
| Metric | precision@k (k = investigator capacity), since they only confirm flagged claims (label uncertainty). |
| Deltas | Notes field needs TF-IDF + structured features via ColumnTransformer; non-fraud labels are uncertain (investigators only confirm flagged claims). |

### Multi-bucket match

**Brief:** "Hospital wants real-time prediction at admission of which patients will need ICU within 24 hours. 8% of admissions go to ICU. Doctors will use the score to staff ICU beds. They need to explain the score for ethics review."

| Step | Reasoning |
|---|---|
| Target sentence | Given a patient at admission, predict whether they will need ICU within 24h. |
| Problem type | Binary classification. |
| Constraints | 4 buckets — imbalance (8%) + real-time + regulation + cost asymmetry (missing ICU need is dangerous). |
| Data shape | 50K-500K admissions, mixed types, lab values. |
| Catalog match | Category 7 (Regulated) + Category 8 (Real-time) + Category 1 (Imbalanced) + Category 13 (Cost-sensitive). |
| Metric | recall at fixed FPR + Brier + per-shift latency; per-group calibration audit. |
| Deltas | LogReg + L2 (regulated) + cost-aware threshold (cost-sensitive) + ms-fast inference (real-time) + SHAP for ethics. |

---

## Cheat sheet

```
5 things to produce in Phase 1:

  1. One-sentence problem statement
       "Given X, predict Y."

  2. Problem type
       regression / binary / multiclass / multilabel / clustering / survival

  3. Constraint profile (from 8 signals)
       imbalance · regulation · real-time · cost · time-series ·
       multilabel · calibration · fairness

  4. Data inventory (10 questions, 5 minutes)
       rows · features · types · target · missingness · IDs · time ·
       protected · leakers · cardinality

  5. Metric pick
       primary + secondary (at minimum)
       NEVER report alone: accuracy, ROC-AUC under imbalance,
       F1 @ 0.5, or RMSE on skewed target

If any of the 5 is missing → fix it before Phase 2.
```

---

## What comes next

You have a problem statement, a metric, and a data inventory. **Phase 2** is where the data work really begins: exploratory data analysis to understand what's in the data, cleaning to fix the obvious issues, and feature engineering to create the signals the model needs. This is the longest, highest-leverage phase — typically 40% of total project time.
