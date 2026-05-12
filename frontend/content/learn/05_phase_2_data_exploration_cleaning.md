# Phase 2: Data Exploration & Cleaning

> EDA, cleaning, and feature engineering. The longest, highest-leverage phase (~40% of project time).

---

## Step 2.1 — EDA checklist

```python
print(df.shape, df.dtypes.value_counts())

# Target shape
print(df['target'].value_counts(normalize=True))     # classification
print(df['target'].skew(), df['target'].quantile([0.5, 0.99]).tolist())   # regression

# Missingness map
print(df.isnull().mean().sort_values(ascending=False).head(10))

# Correlations vs target (numeric only)
print(df.corr(numeric_only=True)['target'].sort_values(key=abs, ascending=False).head(15))
```

Hunt for **leakers** — features with suspiciously high correlation (`>0.95`), post-event fields (e.g. `chargeback_amount` for fraud), or any column you wouldn't have at prediction time.

Test **informative missingness** before deciding to impute:

```python
print(df.groupby(df['col'].isna())['target'].mean())
# Means differ meaningfully → keep a `*_missing` flag, then impute.
```

---

## Step 2.2 — Cleaning decision table

| Issue | Action |
|---|---|
| < 5% missing | Drop rows or fill with median/mode |
| 5–30% missing, MCAR | Impute |
| 5–30% missing, informative | `*_missing` flag THEN impute |
| > 30% missing | Drop the column (unless domain-critical) |
| Categorical "absence" (no pool) | `fillna('None')` — it's a category |
| Outliers in features | Cap at 1st/99th percentile |
| Outliers in target | **Keep** — often the signal |
| Wrong dtypes | `pd.to_numeric(..., errors='coerce')` |
| Confirmed leaker | Drop **before** feature engineering |

---

## Step 2.3 — Feature engineering: 6 patterns

| Signal in the data | Pattern | Example |
|---|---|---|
| Temporal sequence per entity | **Velocity / trend** | `amount / user_avg_30d.shift().rolling(30).mean()` |
| High-cardinality categorical (> 50 unique) | **Target encoding** with Bayesian smoothing | `category_encoders.TargetEncoder(smoothing=10)` |
| Timestamp | **Cyclical** sin/cos + binary flags | `np.sin(2π·hour/24)`, `is_night`, `is_summer` |
| Domain quantity that's a ratio | **Ratio / composite** | `debt_to_income`, `BMI`, `total_sf` |
| Free text with signal | **TF-IDF + sentiment** | `TfidfVectorizer(max_features=100, ngram_range=(1,2))` |
| Sparse + informative absence | **Missingness indicator** | `df['col_missing'] = df['col'].isna()` |

**Always**: `.shift()` before `.rolling()`. Target encoding must be out-of-fold to prevent leakage.

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Skipping EDA | 5-minute checklist first, every time |
| Capping outliers in the **target** | Cap only non-target features |
| Imputing without checking informative missingness | Test `groupby(col_missing)['target'].mean()` |
| Rolling feature without `.shift()` | Always shift first → no leakage |
| One-hot encoding 10K+ merchant IDs | Target-encode with smoothing |
| Engineering features before dropping leakers | Drop leakers first |
