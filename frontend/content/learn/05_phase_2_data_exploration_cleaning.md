# Phase 2: Data Exploration & Cleaning

> The longest, highest-leverage phase. Look at the data carefully, fix what's broken, then create the features the model actually needs. Experts spend 60% of project time here. Beginners spend 5%.

---

## What this phase is for

Phase 2 is where the data becomes ready for modeling. Three steps in order:

1. **EDA** — look at what you actually have. Target distribution, missingness map, leakage hunt, correlations, group patterns.
2. **Cleaning** — fix the obvious problems. Missing values, duplicates, outliers, type mistakes, leakers.
3. **Feature engineering** — create the signals the model needs that aren't already columns.

By the end of Phase 2 you have a clean DataFrame with rich features, ready for feature selection and preprocessing in Phase 3.

The rule of thumb: **experts spend 60% of project time here**. Beginners under-invest because EDA / cleaning / FE feel less glamorous than model tuning. But the model can only learn from the signal the features carry. Time spent here returns 10× more than time spent on hyperparameters.

---

## Where you are in the pipeline

```
   PHASE 1 — Understand the Problem
►► PHASE 2 — Data Exploration & Cleaning   ◄◄  (you are here)
   PHASE 3 — Feature Selection & Preprocessing
   PHASE 4 — Model Selection & Training
   PHASE 5 — Optimization
   PHASE 6 — Evaluation & Validation
   PHASE 7 — Deployment
```

## How much time to spend here

Roughly **40% of total project time** — the single biggest phase. EDA is fast (a few hours); cleaning is medium (a day or two); feature engineering is open-ended (the more domain expertise you bring, the longer it goes — and the better the result).

---

## Step 2.1 — Exploratory Data Analysis (EDA)

### What it is

A structured look at the data. You're answering these questions before any modelling:

- What does the target look like? (skew, balance, mass at zero, censoring)
- Where are the missing values? Are they random or informative?
- Are there leakers? Features that wouldn't exist at prediction time?
- Are there obvious correlations with the target? Group differences?
- Are there duplicates? Time gaps? Suspicious values?

### The EDA checklist

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

# 1. Shape and types
print(df.shape)
print(df.dtypes.value_counts())
print(df.head())

# 2. Target distribution
print(df['target'].describe())
print(df['target'].value_counts(normalize=True))       # classification
print(df['target'].skew())                              # regression — > 1 means skewed
df['target'].hist(bins=50); plt.show()

# 3. Missingness
miss = df.isnull().mean().sort_values(ascending=False)
print(miss[miss > 0])                                   # which columns missing, how much

# 4. Duplicates
print(f'Duplicates: {df.duplicated().sum()}')

# 5. Correlations (numeric features vs target)
corr = df.corr(numeric_only=True)['target'].sort_values(key=abs, ascending=False)
print(corr.head(20))

# 6. Group differences (categorical vs target)
for c in df.select_dtypes(include='object').columns[:5]:
    print(f'\n{c}:')
    print(df.groupby(c)['target'].agg(['mean', 'count']).sort_values('mean', ascending=False).head())
```

### What to look for in the target

**For classification:**

```python
print(df['target'].value_counts(normalize=True))
# 0    0.92
# 1    0.08    ← 8% positive, moderate imbalance

# If you see 0.001 or 0.999 → you have extreme imbalance.
# If you see 3+ classes with one dominant + many tiny → long-tail.
```

**For regression:**

```python
y = df['target']
print(f'Mean: {y.mean():.2f}  Median: {y.median():.2f}  Std: {y.std():.2f}')
print(f'Skew: {y.skew():.2f}  Kurtosis: {y.kurtosis():.2f}')
print(f'P50: {y.quantile(0.5):.2f}  P99: {y.quantile(0.99):.2f}  ratio: {y.quantile(0.99) / y.quantile(0.5):.1f}x')
print(f'Zero fraction: {(y == 0).mean():.2%}')

# Read it:
#   skew > 1            → right-skewed; consider log1p
#   P99/P50 > 5         → long-tailed; raw RMSE will be dominated by tail
#   zero fraction > 30% → zero-inflated; consider two-stage or Tweedie
```

### Missingness patterns — three flavours

Not all missingness is the same:

- **MCAR (missing completely at random)** — random gaps. Imputation is safe. Indicator flag adds noise.
- **MAR (missing at random)** — missingness depends on other observed features. Impute by group; consider adding an indicator.
- **MNAR (missing not at random)** — missingness depends on the missing value itself. *Informative*. Add an indicator flag, then impute.

**Quick test for informative missingness:**

```python
# Is missingness related to the target?
df['col_missing'] = df['col'].isna().astype(int)
print(df.groupby('col_missing')['target'].mean())

# If the two means differ meaningfully (e.g., 0.04 vs 0.12 — 3x), missingness is informative.
# Keep the flag. Imputation alone would erase the signal.
```

Examples of informative missingness:

- "Renovation disclosed in last 12 months" (only 20% of owners fill this — non-fillers may be renovation-hiders)
- "Patient weight" missing in emergency admissions but present in elective ones
- "Credit score" missing for thin-file applicants
- "Pool quality" missing in homes without a pool (categorical 'None', not missing)

### Leakage hunt

A leaker is any feature that wouldn't be available at prediction time. They're catastrophic because they make validation metrics look amazing while the production model fails.

Hunt them aggressively in EDA:

```python
# 1. Features with suspiciously high correlation with target
print(df.corr(numeric_only=True)['target'].abs().sort_values(ascending=False).head(15))
# Anything > 0.95 with the target — suspect.

# 2. Per-target group statistics — do any features cluster perfectly by target?
for c in df.select_dtypes(include='number').columns:
    means = df.groupby('target')[c].mean()
    if (means.max() / (means.min() + 1e-9)) > 100:
        print(f'⚠ {c}: target=0 mean {means[0]:.2f}, target=1 mean {means[1]:.2f}')

# 3. Post-event features
#    e.g., for fraud: "chargeback_amount" is set ONLY for confirmed fraud — a leaker.
#    For hospital LOS: "discharge_diagnosis" is set at discharge, not admission — leaker if predicting at admission.
```

**Rule of thumb:** if a feature seems too good to be true, it is. Drop it and confirm the model still works.

### Group / segment analysis

For each high-cardinality categorical (cuisine, county, merchant), check whether the target rate varies meaningfully across groups:

```python
g = df.groupby('cuisine')['target'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print(g[g['count'] > 50].head(20))     # ignore tiny groups
```

Real signal? Target-encode the column in feature engineering. Noise? Drop the column or use frequency encoding.

### EDA outputs you should have at the end

1. **Target distribution plot** — histogram (regression) or bar chart (classification) with the imbalance number clearly visible.
2. **Missingness table** — per-column missing rate, with informative-missingness flags identified.
3. **Leakers list** — features confirmed to be drop-on-sight.
4. **Top-15 correlation table** — features that correlate with the target.
5. **One-pager of unusual findings** — duplicates, outliers, suspicious values, schema inconsistencies.

### Watch out for…

- **Looking only at means** when distributions are skewed. Always plot. The mean of a long-tailed feature is not representative.
- **Random correlations from small samples** — anything based on < 50 rows is noisy. Filter group stats by count.
- **Skipping EDA on the "boring" columns**. The leaker is usually in a column you didn't bother to look at.

---

## Step 2.2 — Data Cleaning

### What it is

Fixing the issues EDA surfaced. Cleaning decisions are not "what's mathematically right"; they're "what reflects the production reality". Always ask: at prediction time, would this row look like this? Would this missing value be filled in?

### The cleaning decision tree

| Issue | Action |
|---|---|
| Missing values < 5% in a column | Drop those rows OR fill with median/mode |
| Missing values 5-30%, MCAR | Impute (median for numeric, mode for categorical) |
| Missing values 5-30%, informative (MNAR) | Add `*_missing` flag THEN impute |
| Missing values > 30% in a column | Drop the column (unless domain-critical) |
| Categorical missing = absence ("no pool") | `fillna('None')` — not missing, it's a category |
| Exact duplicates | Drop |
| Near-duplicates (same row, different ID) | Investigate — likely a data-pipeline bug |
| Outliers in non-target features | Cap at 1st/99th percentile, or use robust models (trees) |
| Outliers in the target | Don't cap — they may be the signal you care about (fraud, anomaly) |
| Wrong data types (`"123"` string) | Coerce: `pd.to_numeric(df['col'], errors='coerce')` |
| Leaker columns | Remove. Re-check after every change. |
| Time-leakage in rolling features | Always `.shift()` before `.rolling()` |

### Code patterns

```python
# 1. Drop columns that are >30% missing AND not domain-critical
high_miss = df.isnull().mean()[lambda s: s > 0.3].index.tolist()
df = df.drop(columns=high_miss)
print(f'Dropped {len(high_miss)} high-missingness columns')

# 2. Add informative-missingness flags THEN impute
informative_cols = ['renovation_12m', 'credit_score']
for c in informative_cols:
    df[f'{c}_missing'] = df[c].isna().astype(int)
df[informative_cols] = df[informative_cols].fillna(df[informative_cols].median())

# 3. Categorical absence = category, not missing
df['pool_quality'] = df['pool_quality'].fillna('None')

# 4. Outlier capping in feature columns
for c in ['age', 'income', 'tenure']:
    lo, hi = df[c].quantile([0.01, 0.99])
    df[c] = df[c].clip(lower=lo, upper=hi)

# 5. Type coercion
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
df['date']   = pd.to_datetime(df['date'], errors='coerce')

# 6. Drop leakers (do this BEFORE feature engineering)
leakers = ['chargeback_amount', 'discharge_diagnosis', 'inspector_id']
df = df.drop(columns=[c for c in leakers if c in df.columns])

# 7. Group-aware imputation when the group matters
df['weight_kg'] = df.groupby('admission_ward')['weight_kg']\
    .transform(lambda s: s.fillna(s.median()))
```

### Watch out for…

- **Capping the wrong outliers**: in fraud detection, the $15,000 transaction *is* the signal. Don't cap target-correlated columns.
- **Imputing with the mean on skewed data**: median is almost always safer.
- **Imputing across groups**: if hospital weight is missing only in emergency admissions, imputing with the overall median fills emergencies with elective-patient values. Impute by group.
- **Order of operations**: drop leakers BEFORE feature engineering — otherwise you'll engineer features on top of a leaker and have to redo the work.
- **Cleaning train and test inconsistently**: any threshold (1st/99th percentile cap) must come from train only and be applied to test.

---

## Step 2.3 — Feature Engineering

### What it is

Creating features the model needs that don't exist as columns in the raw data. The single highest-leverage step in modern classical ML.

Tree-based models can fit any function in theory, but only with the right features. Velocity, target encoding, cyclical encoding, ratios — these are all manual transformations that tree models would need many splits to discover and linear models cannot discover at all.

Across the 155 expert scenarios, **six pattern families** do most of the work. Recognise the signal in the brief and the right pattern follows.

### Pattern A — Velocity / Momentum / Trend

**Signal.** Temporal sequences per entity — transactions per user, page views per visitor, hospital visits per patient.

**What it does.** Compares recent behaviour against a longer baseline. "How different is this transaction from this user's last 30 days?"

**Code.**
```python
# Customer transaction velocity
df['amount_vs_avg_30d'] = df['amount'] / df.groupby('user_id')['amount']\
    .transform(lambda s: s.shift().rolling(30, min_periods=5).mean())

df['txn_count_24h_vs_30d_avg'] = (
    df.groupby('user_id')['amount'].transform(lambda s: s.shift().rolling(1).count())
    / df.groupby('user_id')['amount'].transform(lambda s: s.shift().rolling(30).count() / 30)
)

# Engagement trend (positive = increasing, negative = declining)
df['minutes_change_1m'] = df.groupby('user_id')['monthly_minutes'].transform('diff')

# Recency-weighted prior violations (decays old events)
df['weighted_prior_critical'] = (
    df.groupby('restaurant_id')
      .apply(lambda g: (g['y'].shift() / (g['days_since_last_inspection'].shift() + 30)).cumsum())
      .reset_index(drop=True)
)
```

**Watch out.**
- `.shift()` is **mandatory** before `.rolling()` so the baseline doesn't include the current row (leakage).
- Cold-start: new users have no 30-day history. Add a `user_history_days` feature so the model can identify cold-start rows.
- Always group by the entity (`user_id`, `restaurant_id`).

### Pattern B — Target Encoding (high-cardinality categoricals)

**Signal.** A categorical feature with > 50 unique values (merchant ID, ZIP code, doctor ID, county).

**What it does.** Replace the category with its historical positive rate (or mean target). `merchant_id` becomes `merchant_fraud_rate`.

**Code (with Bayesian smoothing — critical for rare categories).**
```python
from category_encoders import TargetEncoder

te = TargetEncoder(cols=['merchant_id', 'zip_code', 'county'], smoothing=10)
X_train_enc = te.fit_transform(X_train, y_train)
X_test_enc  = te.transform(X_test)
```

**Watch out.**
- **Leakage**: computing target encoding on the same fold you train on leaks the target. Use out-of-fold encoding (`category_encoders.TargetEncoder` handles this with `cv` parameter).
- **Time leakage**: for temporal data, the encoding at time T must use only data before T. Trailing rolling windows, not full-history.
- **Unseen categories at test time**: always have a fallback to the global mean.

### Pattern C — Cyclical / Temporal Encoding

**Signal.** Anything with periodic patterns — fraud at night, demand in summer, traffic on weekends.

**What it does.** Encodes time so the model knows 23:00 and 00:00 are close, not far apart.

**Code.**
```python
import numpy as np

dt = pd.to_datetime(df['timestamp'])
df['hour_sin'] = np.sin(2 * np.pi * dt.dt.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * dt.dt.hour / 24)

df['dow_sin']   = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
df['dow_cos']   = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
df['month_sin'] = np.sin(2 * np.pi * dt.dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * dt.dt.month / 12)

# Useful binary flags too
df['is_weekend'] = dt.dt.dayofweek >= 5
df['is_night']   = dt.dt.hour.between(22, 6)
df['is_summer']  = dt.dt.month.isin([6, 7, 8])
```

**Watch out.** Don't add both raw `hour` AND `hour_sin/cos` — pick one. For linear models, sin/cos is mandatory. For trees, raw hour + binary flags is sufficient.

### Pattern D — Ratio / Composite Features

**Signal.** The brief mentions a quantity that is a ratio in the domain — debt-to-income, BMI, square footage per bedroom, loss ratio.

**What it does.** Tree models can learn ratios but need many splits. Giving them the ratio directly gets to the same answer with one split.

**Code.**
```python
# Finance
df['debt_to_income']   = df['monthly_debt'] / df['monthly_income']
df['credit_utilization'] = df['credit_used'] / df['credit_limit']

# Real estate
df['total_sf']         = df['basement_sf'] + df['first_floor_sf'] + df['second_floor_sf']
df['baths_per_bedroom'] = df['total_baths'] / df['bedrooms'].clip(lower=1)

# Healthcare
df['bmi']              = df['weight_kg'] / (df['height_m'] ** 2)
df['avg_los_per_visit'] = df['total_los_days'] / df['n_admissions'].clip(lower=1)

# Insurance
df['mileage_per_year'] = df['vehicle_mileage'] / df['vehicle_age_years'].clip(lower=1)
```

**Watch out.** Always clip the denominator (`clip(lower=1)`) to avoid div-by-zero. After creating `total_sf`, drop the component columns — keeping all four breaks linear models with multicollinearity.

### Pattern E — NLP Features from Free Text

**Signal.** A free-text column with signal (complaint logs, diagnosis notes, product descriptions).

**Code.**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF: top 100 unigrams + bigrams
vec = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=10,
    sublinear_tf=True,
)
X_text = vec.fit_transform(df['complaint_text'].fillna(''))

# Hand-crafted text signals
df['complaint_word_count'] = df['complaint_text'].str.split().str.len()
df['has_urgent_word'] = df['complaint_text'].str.contains(
    r'\b(urgent|emergency|asap|immediate)\b', case=False, regex=True
).astype(int)

# Sentiment (cheap and useful)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
df['complaint_neg_sentiment'] = df['complaint_text'].apply(
    lambda t: 0.0 if pd.isna(t) else sia.polarity_scores(t)['neg']
)
```

**Watch out.**
- Cap `max_features` aggressively (50-200). TF-IDF with 10K features dwarfs your structured features.
- Strip PII before vectorising (regulated contexts).
- Vocabulary drifts over time — refresh on retraining.

### Pattern F — Missingness Indicator Features

**Signal.** A sparse column where the missingness itself is informative.

**Code.**
```python
sparse_cols = ['renovation_12m', 'co_signer_present', 'previous_address_match']
for col in sparse_cols:
    df[f'{col}_missing'] = df[col].isna().astype(int)

# Categorical 'absence' (not really missing)
df['pool_quality'] = df['pool_quality'].fillna('None')

# Group-aware imputation alongside the flag
df['weight_imputed'] = df['weight_kg'].isna().astype(int)
df['weight_kg'] = df.groupby('admission_ward')['weight_kg']\
    .transform(lambda s: s.fillna(s.median()))
```

**Watch out.** Test informative-missingness before adding the flag. If `target_rate(col_missing=1) == target_rate(col_missing=0)`, the flag adds noise — drop it.

### Combining patterns

Most real briefs need 3-4 patterns simultaneously:

| Brief | Patterns |
|---|---|
| Fraud detection | A (velocity), B (merchant target encode), C (hour cyclical), F (missing IP score flag) |
| Customer churn | A (engagement trend), D (utilization ratio), E (complaint sentiment), F (no-recent-activity) |
| Hospital readmission | A (visits-per-month), D (BMI, Charlson index), E (admission text TF-IDF), F (weight-missing) |
| House price | B (neighborhood encode), C (sale month), D (total_sf), F (PoolQC = 'None') |
| Insurance claims | A (claims trajectory), B (zip rate), D (value/income), F (mileage-reported flag) |

The stack is similar across briefs — once you've internalised these six families, every new brief becomes recognisable.

---

## Decision tree for this phase

```
Step 2.1 — EDA
  - Compute target shape (skew/balance/zeros)
  - Map missingness; flag informative columns
  - Hunt leakers (high correlation; post-event features)
  - Top-15 correlations; group/segment differences

Step 2.2 — Cleaning
  For each column:
    Missing < 5%?            → drop rows or fill median/mode
    Missing 5-30%, MCAR?     → impute
    Missing 5-30%, MNAR?     → add flag + impute
    Missing > 30%?           → drop column
    Outliers in features?    → cap at 1st/99th
    Outliers in target?      → keep (likely signal)
    Type wrong?              → coerce
    Leaker?                  → DROP

Step 2.3 — Feature Engineering
  For each column:
    High-cardinality category (> 50 unique)?  → target-encode (B)
    Timestamp?                                → cyclical sin/cos + flags (C)
    Free text?                                → TF-IDF + sentiment (E)
    Numeric, part of a domain ratio?          → compute ratio (D)
    Temporal sequence per entity?             → velocity / momentum (A)
    Sparse, informative absence?              → missingness flag + impute (F)

  Audit:
    Any feature that uses the target?  → out-of-fold encode or drop.
    Any rolling feature?               → .shift() before .rolling().
```

---

## How much time to spend here

| Sub-step | Time | Goes long because… |
|---|---|---|
| EDA | 2-6 hours | Hidden leakers; weird target distributions |
| Cleaning | 0.5-2 days | Informative missingness investigations; merging messy multi-source data |
| Feature engineering | Open-ended (1 day to weeks) | Domain knowledge matters; more time = better features = better model |

**Rule of thumb:** if you've spent more than 2 days here, ask yourself: am I creating features that will plausibly help, or am I polishing dimes? More features beyond a certain point add noise.

---

## Common mistakes in this phase

| Mistake | Why bad | Fix |
|---|---|---|
| Skipping EDA, going straight to modeling | You'll never find the leaker | 5-minute checklist first, every time |
| Imputing with mean on skewed data | Mean is way off the median | Use median for numerics |
| Capping outliers in the target column | Caps the signal you care about | Cap only in non-target features |
| One-hot encoding 14,000 merchant names | Catastrophic dimensionality | Target encoding with smoothing |
| Random forward-looking features (no `.shift()`) | Leakage; metric inflated 10-20pp | Always `.shift()` before `.rolling()` |
| Skipping the missingness-vs-target check | Drops informative signal | Test `groupby('col_missing')['target'].mean()` |
| Engineering features before dropping leakers | Wastes work; pollutes feature space | Drop leakers first, then engineer |
| One-hot encoding ordinal features | Loses ordering ("Poor" < "Good" < "Excellent") | Use OrdinalEncoder with explicit order |

---

## Worked example from the catalog

[`/scenarios/regression/04_house_pricing`](/scenarios/regression/04_house_pricing) is the cleanest demonstration of pattern stacking:

- **EDA**: skew = 1.6, P99/P50 = 4.4 → right-skewed target → log1p transform later in Phase 4.
- **Cleaning**: `PoolQC` 99.5% missing — kept as categorical 'None' (it's *absence*, not missing); 4 documented partial sales dropped; outliers in features capped.
- **Feature engineering** stacking patterns B/C/D/F:
  - Pattern B: `NeighborhoodEnc` target-encoded with α=10 smoothing.
  - Pattern C: `month_of_sale` one-hot (discrete here).
  - Pattern D: `TotalSF`, `TotalBath`, `HouseAge` ratios + composites.
  - Pattern F: `PoolQC` filled to 'None'.

Result: ElasticNet on the engineered features hits RMSE 0.126 on log-target — within 1% of the stacked LightGBM. **The features did most of the work; the model did very little.**

For temporal-pattern stacking see [`/scenarios/classification/01_fraud_detection`](/scenarios/classification/01_fraud_detection) — patterns A, B, C, F all at once on transactions.

---

## Cheat sheet

```
EDA (do first, every time):
  - Target shape (skew/balance/zeros)
  - Missingness map (which cols, how much, MCAR vs MNAR)
  - Leakers (post-event features; suspiciously high corr)
  - Top-15 correlations
  - Group/segment statistics

Cleaning (deliberate decisions, not defaults):
  - Missing: < 5% drop rows; 5-30% impute; > 30% drop column
  - MNAR: add a flag THEN impute
  - Outliers in features: cap 1st/99th
  - Outliers in target: keep
  - Drop leakers BEFORE feature engineering

Feature engineering (6 patterns):
  A. Velocity / momentum / trend     (temporal entities)
  B. Target encoding + smoothing     (high-cardinality categoricals)
  C. Cyclical sin/cos + binary flags (timestamps)
  D. Ratio / composite               (domain quantities)
  E. TF-IDF + sentiment              (free text)
  F. Missingness indicator           (sparse, informative)

Audit checklist:
  - Any feature uses the target?    → out-of-fold
  - Any rolling feature?            → .shift() before .rolling()
  - Any leakers slipped through?    → re-grep "post-event" columns
```

---

## What comes next

You have a clean DataFrame with rich features. Most of them will be useful; some will be noise. **Phase 3** narrows the feature set, preprocesses each column appropriately, and splits the data into train and test sets — **the order matters**: preprocessing happens AFTER the split, never before, or you leak test statistics into training.
