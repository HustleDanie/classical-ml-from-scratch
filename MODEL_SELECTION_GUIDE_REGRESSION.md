# Model Selection Guide for Regression

How to narrow down from 12 models to 2–4 candidates using 6 questions about your data, then let metrics decide the winner.

---

## The 6 Questions (Guidelines)

### Question 1: How many rows do I have?

| Model | Minimum Rows to Work Well | Why |
|-------|--------------------------|-----|
| Linear Regression | 30+ | Simple model, learns fast |
| Ridge / Lasso / ElasticNet | 30+ | Same as linear but with protection against overfitting |
| Decision Tree | 100+ | Can memorize small datasets (overfits) |
| KNN | 50+ | But gets SLOW above 10,000 rows (must compare every row) |
| SVM (SVR) | 50–10,000 | Great on small data, but training time explodes on large data |
| Naive Bayes | 100+ | Simple model, works with less data |
| Random Forest | 500+ | Needs enough data to build diverse trees |
| Gradient Boosting | 500+ | Sequential learning needs sufficient examples |
| XGBoost | 500+ | Same as gradient boosting |
| LightGBM | 1,000+ | Designed for large datasets, overkill on tiny ones |

**Rule:** If you have < 500 rows, eliminate Random Forest, Gradient Boosting, XGBoost, LightGBM. If you have > 10,000 rows, eliminate KNN and SVM.

---

### Question 2: How many features (columns) do I have?

| Situation | What Happens | Best Models |
|-----------|-------------|-------------|
| Few features (< 10) | All models work fine | Any |
| Medium features (10–50) | Some models start to struggle | Tree-based models, Ridge, Lasso |
| Many features (50–500) | Risk of overfitting, slow training | Lasso (auto-selects features), LightGBM, Ridge |
| Very many features (500+) | Curse of dimensionality | Lasso, ElasticNet, PCA first then simpler model |

**Rule:** If you have more features than rows (e.g., 200 features, 100 rows), you MUST use Lasso, Ridge, or ElasticNet. Regular Linear Regression and tree models will overfit badly.

---

### Question 3: Is the relationship linear or non-linear?

**Linear** means: when feature goes up, target goes up (or down) proportionally. Like height vs weight.

**Non-linear** means: the relationship is curved, has interactions, or is complex. Like age vs income.

| Relationship | Best Models | Why |
|-------------|-------------|-----|
| Linear | Linear Regression, Ridge, Lasso, ElasticNet | Designed for straight-line patterns |
| Non-linear | Decision Tree, Random Forest, XGBoost, LightGBM, KNN | Trees split data into regions; KNN uses local neighborhoods |
| Don't know | Try both! Start linear, then try tree-based | Compare and see |

**How to check:** Plot each feature vs target. If the scatter plots look like straight lines, it's linear. If they curve or look messy, it's non-linear.

---

### Question 4: Do I have noisy data or outliers?

Outliers are extreme values (e.g., most houses cost $200K–$500K but one costs $10M).

| Situation | Avoid | Use Instead | Why |
|-----------|-------|-------------|-----|
| Outliers in target | Linear Regression (gets pulled by outliers) | Decision Tree, Random Forest, MAE-based models | Trees split by ranges, one extreme value doesn't ruin everything |
| Outliers in features | KNN (distance is distorted by outliers) | Tree-based models | Trees don't care about magnitude, only order |
| Very noisy data | Decision Tree (memorizes noise) | Random Forest, Ridge, Lasso | These average out or regularize away noise |

---

### Question 5: Do I need to explain the model to someone?

| Audience | Use | Avoid |
|----------|-----|-------|
| Manager / client / non-technical | Linear Regression, Decision Tree | XGBoost, Random Forest (black boxes without SHAP) |
| Regulated industry (banking, healthcare) | Linear/Logistic Regression, Decision Tree (+ SHAP for others) | Black-box models without explainability |
| Just need accuracy (Kaggle, internal tool) | XGBoost, LightGBM, Stacking | Nothing to avoid |

**Why:** Linear Regression gives you coefficients you can explain: "Each extra bedroom adds $15,000 to the price." XGBoost gives a number but can't easily explain *why*.

---

### Question 6: How fast does prediction need to be?

| Speed Need | Use | Avoid |
|-----------|-----|-------|
| Real-time (< 1ms per prediction) | Linear Regression, Logistic Regression, Naive Bayes | KNN (slow), large Random Forest |
| Batch (seconds OK) | Any model works | — |
| Training speed matters | Linear models, LightGBM | SVM (slow training), Gradient Boosting (sequential) |

---

## How to Apply All 6 Together

Go through each question and cross off models that don't fit. Whatever's left is your shortlist. Then train those and compare metrics.

---

## 15 Case Studies

### Case Study 1: Predicting House Prices
**Data:** 1,460 rows, 80 features, target = sale price (continuous)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 1,460 | Enough for all models |
| Q2 (features) | 80 | Many features — Lasso/Ridge good for selecting relevant ones |
| Q3 (linear?) | Mix — square footage is linear, but neighborhood effects are non-linear | Need both linear and tree models |
| Q4 (outliers) | Some mansions at $700K+ while median is $160K | Outliers exist |
| Q5 (explain?) | Not required | No constraint |
| Q6 (speed?) | Batch, not real-time | No constraint |

**Eliminated:** KNN (80 features = poor distance metrics), plain Linear Regression (outliers + 80 features)
**Shortlist:** Ridge, Lasso, ElasticNet, Random Forest, XGBoost, LightGBM
**Winner after training:** ElasticNet (R² = 0.93)

---

### Case Study 2: Predicting a Student's Exam Score
**Data:** 200 rows, 5 features (study hours, sleep, attendance, previous grade, income)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 200 | Small dataset — avoid heavy ensembles |
| Q2 (features) | 5 | Few features — all models fine |
| Q3 (linear?) | Likely linear (more study hours = higher score) | Linear models preferred |
| Q4 (outliers) | Few outliers | No concern |
| Q5 (explain?) | Teacher wants to explain to parents | Need interpretability |
| Q6 (speed?) | No constraint | No constraint |

**Eliminated:** XGBoost, LightGBM (overkill, 200 rows), Random Forest (may overfit)
**Shortlist:** Linear Regression, Ridge, Decision Tree
**Expected winner:** Linear Regression or Ridge

---

### Case Study 3: Predicting Insurance Claim Amounts
**Data:** 50,000 rows, 15 features, target = claim amount ($)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 50K | Plenty of data for any model |
| Q2 (features) | 15 | Moderate — all models fine |
| Q3 (linear?) | Non-linear (age, car type, region interact in complex ways) | Tree models preferred |
| Q4 (outliers) | Heavy outliers (most $500–$5K, some $200K+) | Avoid plain linear regression |
| Q5 (explain?) | Regulators want some explainability | Use SHAP with tree models |
| Q6 (speed?) | Batch (nightly) | No constraint |

**Eliminated:** Linear Regression (non-linear + outliers), KNN (50K rows = slow)
**Shortlist:** Ridge (baseline), Random Forest, XGBoost, LightGBM + SHAP
**Expected winner:** XGBoost or LightGBM

---

### Case Study 4: Predicting Daily Temperature
**Data:** 365 rows (1 year), 8 features (humidity, pressure, wind, month, etc.)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 365 | Small-medium |
| Q2 (features) | 8 | Few features |
| Q3 (linear?) | Mix — pressure is linear; month is cyclical/non-linear | Need flexible models |
| Q4 (outliers) | Few (weather is bounded) | No concern |
| Q5 (explain?) | Weather app dashboard — some explanation nice | Moderate |
| Q6 (speed?) | Real-time (app refreshes hourly) | Fast prediction needed |

**Eliminated:** SVM (not great for interpretability), complex ensembles unnecessary
**Shortlist:** Ridge, Random Forest, KNN (365 rows is fine, and similar days have similar temps)
**Expected winner:** Random Forest or Ridge

---

### Case Study 5: Predicting Employee Salary From Resume
**Data:** 5,000 rows, 300 features (TF-IDF from resume text + 10 structured features)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 5,000 | Good for all models |
| Q2 (features) | 300 | Many features — most are sparse TF-IDF |
| Q3 (linear?) | Non-linear (specific keywords matter in complex ways) | Trees or regularized linear |
| Q4 (outliers) | CEO salaries are extreme outliers | Avoid plain linear |
| Q5 (explain?) | HR wants to understand it | Moderate |
| Q6 (speed?) | Batch | No constraint |

**Eliminated:** KNN (300 dimensions = meaningless distances), plain Linear Regression (300 features)
**Shortlist:** Lasso/ElasticNet (kills irrelevant TF-IDF terms), Random Forest, LightGBM
**Expected winner:** LightGBM (accuracy) or Lasso (interpretability)

---

### Case Study 6: Predicting Crop Yield Per Acre
**Data:** 80 rows, 4 features (rainfall, soil pH, fertilizer type, temperature)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 80 | Very small dataset |
| Q2 (features) | 4 | Very few features |
| Q3 (linear?) | Mostly linear (more rain = more yield) | Linear models ideal |
| Q4 (outliers) | Drought years | Minor concern |
| Q5 (explain?) | Farmers need clear recommendations | Need interpretability |
| Q6 (speed?) | No constraint | No constraint |

**Eliminated:** Random Forest, XGBoost, LightGBM, Gradient Boosting (all need more data), KNN (80 rows = unstable)
**Shortlist:** Linear Regression, Ridge
**Expected winner:** Ridge

---

### Case Study 7: Predicting Monthly E-Commerce Revenue
**Data:** 10,000 rows, 25 features (ad spend, season, page views, discounts, etc.)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 10K | Good for everything |
| Q2 (features) | 25 | Moderate |
| Q3 (linear?) | Non-linear (discounts have diminishing returns; seasonal spikes) | Tree models |
| Q4 (outliers) | Black Friday / holiday spikes | Outliers present |
| Q5 (explain?) | Marketing team wants feature importance | Moderate |
| Q6 (speed?) | Batch (monthly forecast) | No constraint |

**Eliminated:** KNN (10K rows getting slow, 25 features), plain Linear Regression (non-linear)
**Shortlist:** Ridge (baseline), Random Forest, XGBoost, LightGBM
**Expected winner:** LightGBM

---

### Case Study 8: Predicting Patient Hospital Stay Duration
**Data:** 3,000 rows, 12 features (age, diagnosis code, surgery type, vitals)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 3,000 | Enough for all models |
| Q2 (features) | 12 | Few features |
| Q3 (linear?) | Non-linear (young recover fast, elderly slower, surgery type changes everything) | Trees preferred |
| Q4 (outliers) | Some patients stay 90+ days | Outliers present |
| Q5 (explain?) | Doctors need to explain to hospital admin | Interpretability important |
| Q6 (speed?) | Real-time (bed allocation) | Fast prediction |

**Eliminated:** SVM (slow for real-time), Stacking (too complex to explain)
**Shortlist:** Decision Tree (fully explainable), Ridge, Random Forest + SHAP
**Expected winner:** Decision Tree or Random Forest

---

### Case Study 9: Predicting Used Car Price
**Data:** 100,000 rows, 10 features (make, model, year, mileage, fuel, city, etc.)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 100K | Large — need efficient models |
| Q2 (features) | 10 | Few features |
| Q3 (linear?) | Non-linear (depreciation curve, brand premium) | Trees preferred |
| Q4 (outliers) | Rare cars priced very high | Present |
| Q5 (explain?) | Pricing app — no explanation needed | No constraint |
| Q6 (speed?) | Real-time (website search) | Fast prediction |

**Eliminated:** KNN (100K rows = too slow), SVM (won't scale)
**Shortlist:** LightGBM, XGBoost, Random Forest
**Expected winner:** LightGBM

---

### Case Study 10: Predicting Electricity Consumption for a Building
**Data:** 500 rows (daily readings, 1.5 years), 7 features (temperature, humidity, day of week, occupancy, etc.)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 500 | Small-medium |
| Q2 (features) | 7 | Few features |
| Q3 (linear?) | Non-linear (heating/cooling spikes at extreme temps — U-shaped) | Trees preferred |
| Q4 (outliers) | Holiday/maintenance days | Minor |
| Q5 (explain?) | Building manager wants to understand drivers | Moderate |
| Q6 (speed?) | Batch (daily forecast) | No constraint |

**Eliminated:** LightGBM/XGBoost (only 500 rows, risky)
**Shortlist:** Ridge, Random Forest, KNN, Decision Tree
**Expected winner:** Random Forest

---

### Case Study 11: Predicting Startup Funding Amount
**Data:** 150 rows, 30 features (industry, team size, revenue, location, patents, etc.)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 150 | Very small |
| Q2 (features) | 30 | More features than ideal for 150 rows (bad ratio!) |
| Q3 (linear?) | Non-linear (tech startups valued differently than restaurants) | But can't use complex models with 150 rows |
| Q4 (outliers) | Massive (most raise $1M, a few raise $500M) | Extreme outliers |
| Q5 (explain?) | Investors want to understand drivers | Interpretability needed |
| Q6 (speed?) | No constraint | No constraint |

**Eliminated:** Random Forest, XGBoost, LightGBM (150 rows + 30 features = guaranteed overfitting), Linear Regression (30 features on 150 rows = unstable)
**Shortlist:** Lasso, Ridge, ElasticNet
**Expected winner:** Lasso (auto-reduces 30 features to the 5–10 that matter)

---

### Case Study 12: Predicting Flight Delay (in minutes)
**Data:** 500,000 rows, 20 features (airline, origin, destination, time, weather, etc.)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 500K | Large dataset |
| Q2 (features) | 20 | Moderate |
| Q3 (linear?) | Non-linear (weather thresholds, airport congestion) | Trees preferred |
| Q4 (outliers) | Most flights 0 min delay, some 300+ min — heavy skew | Present |
| Q5 (explain?) | Internal operations tool | Not needed |
| Q6 (speed?) | Batch (pre-computed daily) | No constraint |

**Eliminated:** KNN (500K = prediction takes forever), SVM (won't train in reasonable time), Decision Tree alone (will overfit)
**Shortlist:** LightGBM, XGBoost, Random Forest, Ridge (baseline)
**Expected winner:** LightGBM

---

### Case Study 13: Predicting Soil Moisture Level
**Data:** 60 rows, 4 features (rainfall, temperature, soil type, elevation)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 60 | Tiny dataset |
| Q2 (features) | 4 | Very few features |
| Q3 (linear?) | Mostly linear (more rain = more moisture) | Linear is ideal |
| Q4 (outliers) | Minimal | No concern |
| Q5 (explain?) | Researchers need clear coefficients | Interpretability needed |
| Q6 (speed?) | No constraint | No constraint |

**Eliminated:** Everything except the simplest models — this is a textbook linear regression problem
**Shortlist:** Linear Regression, Ridge
**Expected winner:** Linear Regression

---

### Case Study 14: Predicting Customer Lifetime Value (CLV)
**Data:** 20,000 rows, 40 features (purchase history, demographics, engagement metrics, etc.)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 20K | Good for all models |
| Q2 (features) | 40 | Moderate-high, some correlated |
| Q3 (linear?) | Non-linear (engagement and purchases interact in complex ways) | Trees preferred |
| Q4 (outliers) | Whale customers spend 100x average | Extreme outliers |
| Q5 (explain?) | Marketing team wants top drivers | Moderate |
| Q6 (speed?) | Batch (monthly update) | No constraint |

**Eliminated:** KNN (40 features = poor distances), Linear Regression (non-linear + outliers), SVM (20K rows, slow)
**Shortlist:** Ridge (baseline), Random Forest, XGBoost, LightGBM + SHAP
**Expected winner:** XGBoost or LightGBM

---

### Case Study 15: Predicting Manufacturing Defect Rate
**Data:** 2,000 rows, 200 features (sensor readings from production line)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 2,000 | Moderate |
| Q2 (features) | 200 | Many features — many sensors measure similar things (correlated) |
| Q3 (linear?) | Non-linear (sensor thresholds trigger defects suddenly) | Trees preferred |
| Q4 (outliers) | Rare extreme defect spikes | Present |
| Q5 (explain?) | Engineers need to know which sensors cause defects | Must explain |
| Q6 (speed?) | Real-time (alert system) | Fast prediction |

**Eliminated:** KNN (200 features = useless distances), Linear Regression (200 correlated features = multicollinearity disaster), SVM (200 features on 2000 rows, slow)
**Shortlist:** Lasso/ElasticNet (identifies key sensors), Random Forest, LightGBM
**Expected winner:** ElasticNet or Random Forest. Use PCA first to reduce 200 features to ~20.

---

## Summary Pattern

Across all 15 cases:

1. **Small data (< 500 rows)** — Always lands on Linear / Ridge / Lasso
2. **Large data (> 10K rows)** — Always includes LightGBM or XGBoost
3. **Many features** — Always includes Lasso or ElasticNet
4. **Non-linear relationships** — Always includes a tree-based model
5. **Need explanations** — Always includes Linear Regression or Decision Tree
6. **Outliers present** — Always avoids plain Linear Regression, favors trees

**The shortlist is always 2–4 models. Then you train them all and let the metrics decide.**

---

## Complete Model-by-Model Reference Tables

Every model rated across all 6 guidelines. No model left out.

### Guideline 1: Dataset Size (Number of Rows)

| Model | < 100 Rows | 100–500 Rows | 500–10K Rows | 10K–100K Rows | 100K+ Rows |
|-------|-----------|-------------|-------------|--------------|-----------|
| Linear Regression | YES — works great | YES | YES | YES | YES |
| Ridge Regression | YES — best small-data pick | YES | YES | YES | YES |
| Lasso Regression | YES — best if many features | YES | YES | YES | YES |
| ElasticNet | YES | YES | YES | YES | YES |
| Decision Tree | RISKY — will memorize data | OK with pruning | YES | YES | YES |
| Random Forest | NO — not enough data for diverse trees | RISKY — may overfit | YES — sweet spot | YES | YES but slow training |
| Gradient Boosting | NO — overfits small data | RISKY | YES — sweet spot | YES | SLOW training |
| XGBoost | NO | RISKY | YES — sweet spot | YES | YES |
| LightGBM | NO — overkill | RISKY | YES | YES — sweet spot | YES — fastest |
| SVM (SVR) | YES — excels on small data | YES — sweet spot | YES | SLOW training | NO — won't scale |
| KNN | RISKY — too few neighbors | OK | YES | SLOW prediction | NO — O(n) per prediction |
| Naive Bayes | NO — regression variant rare | OK for simple cases | YES | YES | YES — very fast |

---

### Guideline 2: Number of Features (Columns)

| Model | < 10 Features | 10–50 Features | 50–200 Features | 200+ Features |
|-------|--------------|---------------|----------------|--------------|
| Linear Regression | YES | OK — watch multicollinearity | RISKY — overfits if rows < features | NO — unstable coefficients |
| Ridge Regression | YES | YES — handles correlated features | YES — regularization prevents overfitting | OK with enough rows |
| Lasso Regression | YES | YES — auto-selects important features | YES — zeros out irrelevant features | YES — best for feature selection |
| ElasticNet | YES | YES | YES — combines Ridge + Lasso benefits | YES — best for correlated + many features |
| Decision Tree | YES | YES | RISKY — may split on noise features | NO — overfits on irrelevant features |
| Random Forest | YES | YES — handles many features naturally | YES — random subsets help | OK — slower but works |
| Gradient Boosting | YES | YES | OK — can overfit without tuning | RISKY — slow + overfitting |
| XGBoost | YES | YES — built-in feature importance | YES — built-in regularization | OK with regularization |
| LightGBM | YES | YES | YES — fast even with many features | YES — handles natively |
| SVM (SVR) | YES | YES | RISKY — slow with many features | NO — training time explodes |
| KNN | YES | OK — distances still meaningful | RISKY — curse of dimensionality | NO — distances meaningless in high dimensions |
| Naive Bayes | YES | YES | YES — assumes feature independence | YES — scales well |

---

### Guideline 3: Linear vs Non-Linear Relationships

| Model | Linear Relationships | Mildly Non-Linear | Highly Non-Linear | Complex Interactions Between Features |
|-------|---------------------|-------------------|-------------------|--------------------------------------|
| Linear Regression | EXCELLENT — designed for this | POOR — can't capture curves | POOR | POOR — must manually create interaction features |
| Ridge Regression | EXCELLENT | POOR | POOR | POOR — same limitation as linear |
| Lasso Regression | EXCELLENT | POOR | POOR | POOR — same limitation |
| ElasticNet | EXCELLENT | POOR | POOR | POOR — same limitation |
| Decision Tree | OK — can approximate with many splits | GOOD — captures step-like patterns | GOOD | GOOD — naturally captures interactions |
| Random Forest | OK | GOOD | EXCELLENT — ensemble smooths the steps | EXCELLENT — each tree finds different interactions |
| Gradient Boosting | OK | GOOD | EXCELLENT | EXCELLENT — sequential correction finds complex patterns |
| XGBoost | OK | GOOD | EXCELLENT | EXCELLENT — best for tabular non-linear tasks |
| LightGBM | OK | GOOD | EXCELLENT | EXCELLENT — same as XGBoost |
| SVM (SVR) | EXCELLENT (linear kernel) | GOOD (RBF kernel) | GOOD (RBF kernel) | OK — kernel trick helps but limited |
| KNN | OK | GOOD — local patterns captured | GOOD — but needs lots of data | POOR — doesn't generalize interactions |
| Naive Bayes | OK for simple linear | POOR | POOR | POOR — assumes feature independence |

---

### Guideline 4: Robustness to Outliers and Noise

| Model | Outliers in Target | Outliers in Features | Noisy/Messy Data | Missing Values (natively) |
|-------|-------------------|---------------------|-----------------|--------------------------|
| Linear Regression | VERY POOR — single outlier shifts entire line | POOR — coefficients distorted | POOR — fits noise | NO — must impute first |
| Ridge Regression | POOR — better than linear but still affected | POOR | OK — regularization helps | NO — must impute first |
| Lasso Regression | POOR | POOR | OK — regularization helps | NO — must impute first |
| ElasticNet | POOR | POOR | OK — regularization helps | NO — must impute first |
| Decision Tree | GOOD — splits by thresholds, not magnitude | GOOD — only uses rank order | POOR — memorizes noise | SOME implementations handle it |
| Random Forest | GOOD — averaging many trees reduces outlier impact | GOOD | GOOD — averaging smooths noise | SOME implementations handle it |
| Gradient Boosting | OK — can be sensitive if not tuned | OK | OK — early stopping helps | SOME implementations handle it |
| XGBoost | OK — built-in regularization helps | OK | GOOD — has built-in regularization | YES — learns optimal direction for missing |
| LightGBM | OK — similar to XGBoost | OK | GOOD — has built-in regularization | YES — handles natively |
| SVM (SVR) | OK — epsilon tube ignores small errors | POOR — distances distorted | OK | NO — must impute first |
| KNN | POOR — outlier neighbors corrupt predictions | VERY POOR — distances dominated by outlier features | POOR — directly uses noisy values | NO — must impute first |
| Naive Bayes | OK — probability-based, less sensitive | OK | OK — smoothing helps | SOME implementations handle it |

---

### Guideline 5: Interpretability / Explainability

| Model | Interpretability Level | What You Can Explain | Suitable for Regulated Industries | Can Add SHAP? |
|-------|----------------------|---------------------|----------------------------------|--------------|
| Linear Regression | EXCELLENT — fully transparent | "Each unit of X increases Y by β" — exact coefficients | YES — gold standard | YES (but not needed) |
| Ridge Regression | EXCELLENT | Same as linear — coefficients (slightly shrunk) | YES | YES (but not needed) |
| Lasso Regression | EXCELLENT | Same as linear + "these features were dropped (coeff = 0)" | YES — shows which features matter | YES (but not needed) |
| ElasticNet | EXCELLENT | Same as Lasso | YES | YES (but not needed) |
| Decision Tree | EXCELLENT — visual rules | "If X > 5 AND Y < 3, predict Z" — full decision path | YES — easy to audit | YES (but not needed) |
| Random Forest | POOR — black box | Feature importance ranking only (no coefficients) | ONLY with SHAP | YES — commonly used |
| Gradient Boosting | POOR — black box | Feature importance ranking only | ONLY with SHAP | YES — commonly used |
| XGBoost | POOR — black box | Feature importance ranking only | ONLY with SHAP | YES — commonly used |
| LightGBM | POOR — black box | Feature importance ranking only | ONLY with SHAP | YES — commonly used |
| SVM (SVR) | POOR — black box (with RBF kernel) | Linear kernel: coefficients. RBF kernel: nothing intuitive | Linear kernel only | YES but less common |
| KNN | MODERATE — instance-based | "These 5 similar cases had these values" — show neighbors | OK for small datasets | YES but unusual |
| Naive Bayes | GOOD | "Feature X has P(Y) = 0.8 for this class" — conditional probs | OK | YES but unusual |

---

### Guideline 6: Prediction Speed and Training Speed

| Model | Training Speed (Time to Learn) | Prediction Speed (Time per New Row) | Scales to Millions of Rows | Memory Usage |
|-------|-------------------------------|------------------------------------|-----------------------------|-------------|
| Linear Regression | VERY FAST — milliseconds | VERY FAST — microseconds | YES | LOW |
| Ridge Regression | VERY FAST — milliseconds | VERY FAST — microseconds | YES | LOW |
| Lasso Regression | FAST — seconds (iterative solver) | VERY FAST — microseconds | YES | LOW |
| ElasticNet | FAST — seconds (iterative solver) | VERY FAST — microseconds | YES | LOW |
| Decision Tree | FAST — seconds | FAST — microseconds | YES | LOW |
| Random Forest | MODERATE — minutes (100 trees) | MODERATE — milliseconds (traverse 100 trees) | OK — parallelizable | MODERATE — stores all trees |
| Gradient Boosting | SLOW — minutes (sequential trees) | MODERATE — milliseconds | POOR — sequential training | MODERATE |
| XGBoost | MODERATE — faster than sklearn GB | MODERATE — milliseconds | OK — optimized C++ | MODERATE |
| LightGBM | FAST — fastest tree method | MODERATE — milliseconds | YES — designed for scale | MODERATE |
| SVM (SVR) | VERY SLOW — O(n² to n³) | MODERATE | NO — won't finish training | HIGH — stores support vectors |
| KNN | NONE — no training (stores data) | VERY SLOW — O(n) per prediction | NO — must scan all rows | HIGH — stores entire dataset |
| Naive Bayes | VERY FAST — single pass | VERY FAST — microseconds | YES | LOW |

---

### Master Summary Table: All Models × All Guidelines

| Model | Small Data (< 500) | Large Data (> 10K) | Many Features (> 50) | Non-Linear | Outlier Robust | Interpretable | Fast Predict |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Linear Regression | YES | YES | NO | NO | NO | YES | YES |
| Ridge Regression | YES | YES | YES | NO | NO | YES | YES |
| Lasso Regression | YES | YES | YES | NO | NO | YES | YES |
| ElasticNet | YES | YES | YES | NO | NO | YES | YES |
| Decision Tree | OK | YES | NO | YES | YES | YES | YES |
| Random Forest | NO | YES | YES | YES | YES | NO | OK |
| Gradient Boosting | NO | OK | OK | YES | OK | NO | OK |
| XGBoost | NO | YES | YES | YES | OK | NO | OK |
| LightGBM | NO | YES | YES | YES | OK | NO | OK |
| SVM (SVR) | YES | NO | NO | YES | OK | NO | OK |
| KNN | OK | NO | NO | YES | NO | OK | NO |
| Naive Bayes | OK | YES | YES | NO | OK | YES | YES |
