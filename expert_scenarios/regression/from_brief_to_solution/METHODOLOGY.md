# From Brief to Solution — Regression Methodology

How to take a fresh business brief plus a dataset and find the matching playbook in the [regression catalog](../catalog/CATALOG.md).

---

## The Core Idea

Almost every regression brief fits one of ~80 archetypes already cataloged. Your job:

1. **Read the brief carefully** — what numeric quantity are they asking you to predict?
2. **Inventory the dataset** — what shape, what target distribution?
3. **Match to an archetype** — which catalog category (and which 1–3 scenarios within it)?
4. **Adapt** — what's different about your problem?

This guide gives you a repeatable 6-step protocol.

---

## Step 1: Extract the Prediction Target From the Brief

Reduce the brief to one sentence:

> *"Given X, predict Y, a numeric quantity in units U."*

**Examples:**

| Brief excerpt | Reduced sentence |
|---------------|------------------|
| "Estimate house sale prices from listing details." | Given a house's features, predict sale price in USD. |
| "Predict how long patients will stay in the hospital." | Given a patient's admission features, predict length of stay in days. |
| "Forecast next-day grid load." | Given today's load + weather, predict tomorrow's load in MW. |
| "Estimate insurance claim amounts." | Given a claim's features, predict claim amount in USD. |

**Watch-out 1:** classification dressed up as regression. "Score the risk from 0 to 100" — if the downstream decision is binary (approve/decline), it's classification with a calibrated probability. Push back.

**Watch-out 2:** the units matter. RMSE in dollars vs RMSE in log-dollars give very different optimization targets. Pick units that match what the stakeholder reads.

---

## Step 2: Identify the Constraints (8 Signals to Read In)

Re-read the brief looking for these 8 signals.

| # | Signal | Phrases that indicate it | Catalog category it points to |
|---|--------|-------------------------|-------------------------------|
| 1 | **Heavy-tailed signal** | "outliers matter", "claims", "whales", "long-tail" | Category 4 (Insurance), 9 (CLV), 16 (Count) |
| 2 | **Time-series signal** | "forecast", "next month", "history", lag features | Category 6 (Energy), 7 (Time-series), 28 (Count) |
| 3 | **Real-time / dynamic signal** | "live", "ms latency", "real-time pricing" | Category 8 (Dynamic pricing) |
| 4 | **Regulation signal** | "audit", "actuarial", "fair", regulated | Category 3 (Healthcare), 4 (Insurance) |
| 5 | **Censoring signal** | "we only see customers who stayed", drop-off | Category 18 (Survival) |
| 6 | **Uncertainty signal** | "P95", "worst-case", "intervals", risk capital | Category 17 (Quantile / interval) |
| 7 | **Bounded-target signal** | "% yield", "percentage", "rate" | Category 11 (Education), 14 (Agriculture) |
| 8 | **Causal signal** | "lift", "incrementality", "uplift", "treatment effect" | Category 9 (Marketing — uplift) |

**Rule:** Multiple signals are normal. The intersections matter — "real-time dynamic pricing in a heavy-tailed market" picks up signals 1+3 simultaneously.

---

## Step 3: Inventory the Dataset

Open the data and answer these 12 questions in 5 minutes.

| # | Question | Why it matters | Quick check |
|---|----------|----------------|-------------|
| 1 | How many rows? | Eliminates models | `len(df)` |
| 2 | How many features? | Curse of dimensionality | `df.shape[1]` |
| 3 | Feature types? | Determines preprocessing | `df.dtypes.value_counts()` |
| 4 | Target distribution shape? | Skewness / outliers | `df[target].describe()`, histogram |
| 5 | Target heavy-tailed? | Need log / Tweedie / quantile | Skew > 2, P99 / median ratio > 5 |
| 6 | Missingness pattern? | Imputation vs informative-missing | `df.isnull().mean()` |
| 7 | Are there ID-like columns? | Drop, don't model | High cardinality + few duplicates |
| 8 | Is there a temporal axis? | Time-based split required | Date columns or ordering |
| 9 | Geospatial? | Distance / clustering features | Lat/long or zip |
| 10 | Any obvious leakers? | Drop before modeling | Post-event features |
| 11 | Censoring? | Survival vs regression | "still active", "no churn yet" rows |
| 12 | Multicollinearity? | Affects linear models | `df.corr()` heatmap |

**Watch-out:** check for log-normality. If `df[target].apply(np.log).hist()` looks roughly Gaussian, ALL your modeling should happen on the log scale and you should report metrics on both scales.

---

## Step 4: Match to an Archetype Category

Quick decision tree:

```
Is the target a count (non-negative integer)?
└── YES              → Category 16 (Count regression) — Poisson / Tweedie

Is the target heavy-tailed (skew > 2)?
├── YES, financial / insurance → Category 4 (Insurance) or Category 2 (Financial)
└── YES, generic              → consider log-transform or Tweedie

Is there a temporal axis?
├── YES, forecasting volume   → Category 7 (Time-series)
├── YES, energy demand        → Category 6 (Energy)
└── YES, dynamic price action → Category 8 (Dynamic pricing)

Are records censored (some rows have unknown final outcome)?
└── YES              → Category 18 (Survival)

Does the brief require uncertainty / quantiles / intervals?
└── YES              → Category 17 (Quantile / interval)

Is the target bounded [0, 1] or [0, 100]?
└── YES              → consider Beta regression; Category 11 / 14 typically

Is the brief regulated (insurance, healthcare, finance)?
└── YES              → Category 3 (Healthcare) / 4 (Insurance) / 12 (HR)

Is the latency budget < 100ms?
└── YES              → Category 8 (Dynamic pricing — real-time pattern)

Otherwise:
└── Category 1 (Property pricing) for transactional values, or Category 5 (Manufacturing) for sensor-driven, or Category 9 (Marketing) for customer-level
```

---

## Step 5: Drill to the Closest 1–3 Scenarios

Inside the chosen category, scan entries and pick by:

1. Comparable data shape (rows within an order of magnitude)
2. Similar target distribution shape (skew, range, heavy-tailed-ness)
3. Same metric (MAPE? Tweedie deviance? Quantile pinball?)

If multiple scenarios feel close, read all of them — the watch-outs alone will save you hours of debugging.

---

## Step 6: Adapt — List the Deltas

Write a one-page deltas comparison BEFORE coding:

| Aspect | Catalog scenario | My problem | Adaptation |
|--------|------------------|------------|------------|
| Rows | 1.5M | 30K | Smaller-CV; consider simpler model |
| Target skew | 4.2 | 1.8 | Light log-transform sufficient (no Tweedie needed) |
| Latency | Batch | Real-time | Constrain model size; precompute features |
| Regulation | None | Insurance state-filed | Switch to GLM (Tweedie); auditable |
| Target unit | USD | USD per acre | Same model, normalize by area |
| Outliers | Heavy, kept | Light, can cap | Robust loss optional |

Adaptation is the engineering work. Don't copy verbatim.

---

## The "No Match" Path

**1. Check both categories.** If the target is binary (or could be reframed binary), see the [classification methodology](../../classification/from_brief_to_solution/METHODOLOGY.md).

**2. Decompose.** A two-part model (classification + regression) often fits problems that look "weird" alone. Insurance claims = (will there be a claim?) × (claim amount given there is one) — two scenarios, not one.

**3. Re-frame the target.** "Predict the rate of growth" → log-transform; "Predict the time until X" → survival; "Predict whether AND how much" → two-part model.

**4. Add a new scenario.** If genuinely novel, contribute it back to the catalog.

---

## Adaptation Checklist (12 Questions Before Copying Any Playbook)

1. Is the row count within an order of magnitude of the catalog scenario?
2. Is the feature count within an order of magnitude?
3. Is the target distribution shape similar (skew, modality, bounds)?
4. Are the feature types comparable (numeric / categorical / text mix)?
5. Is the metric the same? If not, why is yours different?
6. Is there a temporal axis I need to split on?
7. Is the latency budget the same?
8. Is the regulation context the same?
9. Are there censored records I need to handle?
10. Is the target bounded (and if so, am I respecting the bounds in predictions)?
11. Are there leakers in my data the catalog scenario didn't have?
12. Is the deployment reality the same? (real-time API vs batch CSV)

---

## Three Worked Examples

### Worked Example A — Easy Match

**Brief:** "Predict daily revenue for our retail locations next week. We have 3 years of daily sales for 200 stores."

| Step | Reasoning |
|------|-----------|
| 1. Target sentence | Given today's features, predict tomorrow's revenue per store in USD. |
| 2. Signals | Time-series + multi-store hierarchy |
| 3. Data shape | ~219K store-days, ~30 features, target moderately skewed |
| 4. Category | Category 7 (Time-series forecasting) |
| 5. Closest scenario | Scenario 31 (Daily Sales Forecast) — exact shape match |
| 6. Deltas | 200 stores → hierarchical model (per-store features OR per-store models with shrinkage); 3 years → temporal split with last 4 weeks as holdout; promotional dates need explicit features |

**Action:** Prophet baseline per store + LightGBM with lag features + store one-hot encoding. Walk-forward validation.

---

### Worked Example B — Partial Match

**Brief:** "We're an EV charging network. Predict next-hour kWh demand at each of our 5,000 stations to plan grid load. We have 18 months of 15-minute readings."

| Step | Reasoning |
|------|-----------|
| 1. Target sentence | Given a station and current features, predict next-hour kWh demand. |
| 2. Signals | Time-series + energy + geospatial |
| 3. Data shape | ~26M station-quarter-hours, weather + calendar features |
| 4. Category | Category 6 (Energy) primary; Category 7 (Time-series) secondary |
| 5. Closest scenario | Scenario 30 (EV Charging Demand) — exact match |
| 6. Deltas | 5,000 stations → hierarchical structure (per-region averages as feature); 18 months → temporal split with last month as holdout; rapid market growth → recent data weighted higher |

**Action:** LightGBM with calendar + weather + region-aggregate features; per-station effects via target encoding; weight recent data 2x.

---

### Worked Example C — Multi-Bucket Match

**Brief:** "Predict the lifetime value of new subscribers in their first 30 days. About 20% churn within 30 days; the rest stay 6+ months. We have 2 years of subscriber data with daily activity."

| Step | Reasoning |
|------|-----------|
| 1. Target sentence | Given a subscriber's first 30 days of activity, predict their lifetime spend. |
| 2. Signals | Heavy-tailed (whales) + censoring (still-active subscribers) + behavioral |
| 3. Data shape | likely 100K–10M subscribers, 30–80 features, heavy-tailed target with right-censoring |
| 4. Categories | Category 9 (CLV) + Category 18 (Survival) |
| 5. Closest scenarios | Scenario 41 (CLV) + Scenario 73 (Customer time-to-churn) — pull pieces from each |
| 6. Deltas | Censoring is heavy (active subscribers) → two-part: survival regression for time-to-churn × per-period spend regression; 30-day prediction window → use first-30-days behavioral signals only as features (no future info) |

**Action:** Two-part model: (1) Cox PH for time-to-churn from 30-day signals → expected lifetime; (2) Gradient Boosting for monthly spend → per-period revenue; multiply for total LTV.

---

## Common Mistakes When Reading Briefs

### 1. Optimizing RMSE on a heavy-tailed target

If your target is right-skewed (insurance claims, CLV, rent), RMSE rewards models that nail the median and ignore the tail. The tail is the dollars. Use **log-target RMSE**, **Tweedie deviance**, or **MAPE** to keep the tail honest.

### 2. Random train/test splits with time-ordered data

Same trap as classification — but worse, because regression CV scores on temporally-leaky splits look great while real-world performance tanks. **Always temporal-split** when there's a date column.

### 3. Ignoring censoring

Customers who haven't churned yet aren't "long-lived" — they're censored. Naive regression on lifetime treats them as final. Survival methods or careful re-framing required.

### 4. Reporting MAPE on near-zero values

MAPE = `|y - y_hat| / y`. When `y` is near zero, MAPE explodes. Use **MAE** for absolute error, **sMAPE** for symmetric, or **WAPE** for volume-weighted.

### 5. Predicting outside physical bounds

If yield is bounded [0, 1], your model can predict 1.2 if you don't constrain it. **Logit-transform** the target or **clip predictions** at inference.

### 6. Forgetting that `r²` can be negative

Negative R² means the model is worse than predicting the mean. If you see this, your features are anti-correlated, your validation set is mis-distributed, or your model has a bug.

### 7. Using RMSE when units differ across rows

Predicting "minutes" and rows range from 5 to 5,000? RMSE punishes large-magnitude errors disproportionately. Either MAPE (relative error) or weighted RMSE (per-row weight = 1/y) keeps it fair.

### 8. Stopping at the first model that fits

The catalog gives you a shortlist. Train all of them. The simplest model that meets the metric is the right answer — but you have to verify by training the alternatives.

---

## Where to Go Next

- [CATALOG.md](../catalog/CATALOG.md) — the comprehensive scenario library
- [MODEL_SELECTION_GUIDE_REGRESSION.md](../../../MODEL_SELECTION_GUIDE_REGRESSION.md) — the 6 framing questions for narrowing models
- [ML_PIPELINE_GUIDE.md](../../../ML_PIPELINE_GUIDE.md) — the universal 12-phase pipeline every scenario instantiates
- [FEATURE_SELECTION_GUIDE.md](../../../FEATURE_SELECTION_GUIDE.md) — when feature count is a problem
