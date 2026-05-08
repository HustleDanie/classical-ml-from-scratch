# Model Selection Guide for Classification

How to narrow down from 12 models to 2–4 candidates using 8 questions about your data, then let metrics decide the winner.

---

## The 8 Questions (Guidelines)

### Question 1: How many rows do I have?

| Model | Minimum Rows to Work Well | Why |
|-------|--------------------------|-----|
| Logistic Regression | 30+ | Simple model, learns fast |
| Logistic + L2 / L1 / ElasticNet | 30+ | Same as Logistic but with protection against overfitting |
| Decision Tree | 100+ | Can memorize small datasets (overfits) |
| KNN | 50+ | But gets SLOW above 10,000 rows (must compare every row) |
| SVM | 50–10,000 | Great on small data, but training time explodes on large data |
| Naive Bayes | 30+ | Simple model, works with very little data — gold standard for text |
| Random Forest | 500+ | Needs enough data to build diverse trees |
| Gradient Boosting | 500+ | Sequential learning needs sufficient examples |
| XGBoost | 500+ | Same as Gradient Boosting |
| LightGBM | 1,000+ | Designed for large datasets, overkill on tiny ones |

**Rule:** If you have < 500 rows, eliminate Random Forest, Gradient Boosting, XGBoost, LightGBM. If you have > 10,000 rows, eliminate KNN and SVM.

---

### Question 2: How many features (columns) do I have?

| Situation | What Happens | Best Models |
|-----------|-------------|-------------|
| Few features (< 10) | All models work fine | Any |
| Medium features (10–50) | Some models start to struggle | Tree-based, regularized Logistic |
| Many features (50–500) | Risk of overfitting, slow training | Lasso Logistic (auto-selects features), LightGBM, Naive Bayes |
| Very many features (500+, e.g. TF-IDF) | Curse of dimensionality | Multinomial NB, Lasso Logistic, Linear SVM, ElasticNet |

**Rule:** If you have more features than rows (e.g., 5,000 TF-IDF terms, 1,000 rows), you MUST use a regularized linear model (L1/L2/ElasticNet Logistic) or Naive Bayes. Plain Logistic Regression and tree models will overfit badly. KNN will be useless because distances become meaningless in high dimensions.

---

### Question 3: Are the classes linearly separable?

**Linearly separable** means: a straight line (or hyperplane in higher dimensions) can split the classes well. Like separating low-spending vs high-spending customers based on income.

**Non-linear** means: the boundary between classes curves, has interactions, or is complex. Like spam detection where specific word combinations matter.

| Class Boundary | Best Models | Why |
|---------------|-------------|-----|
| Linearly separable | Logistic Regression, Linear SVM, Naive Bayes | Designed for linear decision boundaries |
| Non-linear | Decision Tree, Random Forest, XGBoost, LightGBM, KNN, SVM with RBF kernel | Trees split data into regions; KNN uses local neighborhoods; RBF kernel maps to higher dimensions |
| Don't know | Try both! Start with Logistic, then try a tree-based model | Compare and see |

**How to check:** Plot pairs of features as scatter plots colored by class. If you can draw a straight line between classes, it's linearly separable. If the classes form curved or interlocking patterns, it's non-linear.

---

### Question 4: Do I have noisy data, outliers, or mislabeled examples?

Classification has three forms of "noise":

- **Feature outliers** — extreme values in a column (e.g., one customer with 100x normal spend).
- **Label noise** — some rows have the wrong class label (mislabeled training examples).
- **Hard / overlapping examples** — borderline cases between classes.

| Situation | Avoid | Use Instead | Why |
|-----------|-------|-------------|-----|
| Outliers in features | KNN (distance is distorted) | Tree-based models, Naive Bayes | Trees don't care about magnitude, only order |
| Mislabeled examples | Decision Tree, KNN (memorize bad labels) | Random Forest, Logistic with L2, XGBoost with early stopping | Averaging and regularization smooth over wrong labels |
| Very noisy features | Decision Tree (memorizes noise) | Random Forest, regularized Logistic | These average out or regularize away noise |
| Hard / overlapping classes | Hard-margin SVM | Logistic Regression (probabilistic), GBM with calibration | Probabilistic models hedge instead of forcing a hard decision |

---

### Question 5: Do I need to explain the model to someone?

| Audience | Use | Avoid |
|----------|-----|-------|
| Manager / client / non-technical | Logistic Regression, Decision Tree | XGBoost, Random Forest (black boxes without SHAP) |
| Regulated industry (banking, healthcare, lending) | Logistic Regression, Decision Tree (+ SHAP for tree ensembles) | Black-box models without explainability |
| Just need accuracy (Kaggle, internal tool) | XGBoost, LightGBM, Stacking | Nothing to avoid |

**Why:** Logistic Regression gives you log-odds you can explain: "Each extra year of tenure decreases the odds of churn by 12%." Decision Tree gives explicit if/then rules. XGBoost gives a probability but can't easily explain *why* without SHAP.

---

### Question 6: How fast does prediction need to be?

| Speed Need | Use | Avoid |
|-----------|-----|-------|
| Real-time (< 1ms per prediction) | Logistic Regression, Naive Bayes, small Decision Tree | KNN (slow), large Random Forest |
| Batch (seconds OK) | Any model works | — |
| Training speed matters | Logistic Regression, Naive Bayes, LightGBM | SVM (slow training), Gradient Boosting (sequential) |

---

### Question 7: How balanced are the classes?

This question doesn't exist for regression but is critical for classification. Class imbalance breaks accuracy as a metric and biases most models toward predicting the majority class.

| Imbalance Level | Example | Strategy | Best Models |
|-----------------|---------|----------|-------------|
| Balanced (50/50 to 60/40) | Sentiment positive vs negative | None needed | Any |
| Mild (70/30 to 80/20) | Customer churn (~20%) | Use F1 / ROC-AUC, not accuracy | Any with stratified CV |
| Moderate (85/15 to 90/10) | Loan default | `class_weight='balanced'`, threshold tuning | Logistic, Random Forest, XGBoost |
| Severe (95/5) | Disease screening | SMOTE or undersampling, PR-AUC | XGBoost, LightGBM with `scale_pos_weight` |
| Extreme (99/1 or worse) | Fraud, intrusion | Probability output + threshold tuning, PR curves, anomaly-detection mindset | Gradient Boosting, XGBoost, calibrated Logistic |

**Rule:** If your minority class is < 10% of the data, NEVER use accuracy as your metric — a model that predicts "majority" for everything will look 90%+ accurate while being useless. Use Precision, Recall, F1, ROC-AUC, or PR-AUC depending on which mistake costs more.

**Models that handle imbalance natively (via parameter):** Logistic Regression (`class_weight`), Random Forest (`class_weight`), SVM (`class_weight`), XGBoost (`scale_pos_weight`), LightGBM (`is_unbalance` or `scale_pos_weight`).

**Models that don't:** KNN, Naive Bayes — these need data-level fixes (SMOTE, undersampling).

**How to check:** `df['target'].value_counts(normalize=True)`. Look at the smallest class's percentage.

---

### Question 8: Binary or multiclass?

| Model | Binary | Multiclass Strategy |
|-------|--------|---------------------|
| Logistic Regression | Native | Softmax (native) or One-vs-Rest |
| Decision Tree | Native | Native — handles many classes naturally |
| Random Forest | Native | Native |
| Gradient Boosting | Native | Native (one tree per class internally) |
| XGBoost | Native | Native (`multi:softmax` / `multi:softprob`) |
| LightGBM | Native | Native (`multiclass` objective) |
| SVM | Native | One-vs-Rest (default) or One-vs-One — slow with many classes |
| KNN | Native | Native — voting among neighbors works for any number of classes |
| Naive Bayes | Native | Native — Gaussian (continuous), Multinomial (counts/text), Bernoulli (binary features) |

**Rule:** If you have more than 10 classes, AVOID SVM with One-vs-One — it builds n*(n-1)/2 binary models and explodes. Prefer tree-based models, Softmax Logistic, or Multinomial Naive Bayes. For text classification with many classes, Multinomial NB is fast and surprisingly hard to beat.

**Naive Bayes variant choice:**
- **Gaussian NB** — continuous features (e.g., medical lab measurements)
- **Multinomial NB** — count-based features (e.g., word counts, TF-IDF)
- **Bernoulli NB** — binary features (e.g., word present/absent)

---

## How to Apply All 8 Together

Go through each question and cross off models that don't fit. Whatever's left is your shortlist. Then train those and compare metrics — choosing the right metric for your imbalance level (Q7) and class count (Q8).

---

## 20 Case Studies

### Case Study 1: Email Spam Detection
**Data:** 5,000 emails, 5,000+ TF-IDF features, target = spam vs ham (binary, ~30% spam)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 5,000 | Enough for all models |
| Q2 (features) | 5,000+ TF-IDF | Very high-dim sparse — kills KNN and tree models |
| Q3 (linear?) | Largely linear in TF-IDF space | Linear models excel |
| Q4 (noise) | Some mislabeled emails | Avoid Decision Tree alone |
| Q5 (explain?) | Users want "why was this flagged?" | Need interpretability |
| Q6 (speed?) | Real-time (inbox filtering) | Fast prediction needed |
| Q7 (balance?) | 70/30 — mild imbalance | F1 instead of accuracy |
| Q8 (binary?) | Binary | All models OK |

**Eliminated:** KNN (5,000 dimensions = meaningless distances), Decision Tree (overfits sparse text), Random Forest / GBM (slow on 5,000 sparse features), plain Logistic (will overfit 5,000 features without regularization)
**Shortlist:** Multinomial Naive Bayes, Lasso Logistic Regression, Linear SVM
**Expected winner:** Multinomial Naive Bayes (fast, accurate baseline) or Lasso Logistic (interpretable per-word coefficients)

---

### Case Study 2: Credit Card Fraud Detection
**Data:** 1.2M transactions, 28 features (PCA-anonymized + amount + time), 0.17% fraud (extreme imbalance)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 1.2M | Large — favor scalable models |
| Q2 (features) | 28 | Moderate — all models OK |
| Q3 (linear?) | Non-linear (fraud patterns are interaction-heavy) | Tree-based models |
| Q4 (noise) | Some labeled-late frauds | Random Forest / GBM smooth this |
| Q5 (explain?) | Regulators + fraud analysts want explanations | SHAP needed |
| Q6 (speed?) | Real-time (transaction approval) | Need fast scoring |
| Q7 (balance?) | 99.83/0.17 — extreme | NEVER use accuracy. Use PR-AUC + threshold tuning |
| Q8 (binary?) | Binary | All models OK |

**Eliminated:** KNN (1.2M rows = too slow), SVM (won't train at this scale), Naive Bayes (poor probability calibration matters when threshold-tuning), Decision Tree alone (overfits, no probability nuance)
**Shortlist:** XGBoost, LightGBM, Gradient Boosting + SHAP, Logistic Regression with `class_weight` (baseline)
**Expected winner:** XGBoost or LightGBM with `scale_pos_weight`, threshold tuned via PR curve

---

### Case Study 3: Customer Churn Prediction (Telecom)
**Data:** 4.2M subscribers, ~50 features (usage, billing, plan, tenure), 2.1% monthly churn

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 4.2M | Very large |
| Q2 (features) | ~50 | Moderate |
| Q3 (linear?) | Mix — tenure linear; plan changes non-linear | Tree models preferred |
| Q4 (noise) | Some incorrect churn timestamps | Ensembles handle this |
| Q5 (explain?) | Marketing wants top reasons | SHAP for tree models |
| Q6 (speed?) | Batch (monthly retention campaign) | No constraint |
| Q7 (balance?) | 97.9/2.1 — severe | F1 / PR-AUC, threshold tune |
| Q8 (binary?) | Binary | All OK |

**Eliminated:** KNN, SVM (won't scale to 4.2M), Decision Tree alone (overfits), Naive Bayes (independence assumption violated by correlated billing fields)
**Shortlist:** LightGBM, XGBoost, Logistic Regression (baseline + interpretable risk-score ranking)
**Expected winner:** LightGBM (handles 4.2M rows fastest), with Logistic Regression for the explainable ranking view

---

### Case Study 4: Loan Default Prediction (Regulated, Fairness-Sensitive)
**Data:** 930K approved loans, ~30 features, 8.3% default rate; regulated for fair lending (ECOA)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 930K | Large |
| Q2 (features) | ~30 | Moderate |
| Q3 (linear?) | Mostly linear (income, debt-to-income, FICO) | Linear models work well |
| Q4 (noise) | Clean — financial data | No concern |
| Q5 (explain?) | MUST explain (adverse-action notices) | Logistic mandatory |
| Q6 (speed?) | Batch (overnight underwriting) | No constraint |
| Q7 (balance?) | 91.7/8.3 — moderate | `class_weight='balanced'`, threshold tune |
| Q8 (binary?) | Binary | All OK |

**Eliminated:** Black-box models without SHAP (regulatory risk), KNN, SVM (no clean coefficients for adverse-action letters)
**Shortlist:** Logistic Regression with L2 (primary), XGBoost with monotonic constraints + SHAP (challenger), Decision Tree (interpretable benchmark)
**Expected winner:** Logistic Regression for production (regulatory-friendly), XGBoost as research benchmark

---

### Case Study 5: Employee Attrition Prediction
**Data:** 4,800 employees, ~80 features (demographics, performance, tenure, comp), 9% attrition over 6 months

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 4,800 | Moderate-small |
| Q2 (features) | ~80 | Many features for 4,800 rows — overfit risk |
| Q3 (linear?) | Mix (tenure linear; manager rating non-linear) | Try both |
| Q4 (noise) | Clean HR data | No concern |
| Q5 (explain?) | HR wants top retention drivers | Logistic + SHAP |
| Q6 (speed?) | Batch (quarterly review) | No constraint |
| Q7 (balance?) | 91/9 — moderate | `class_weight`, F1 |
| Q8 (binary?) | Binary | All OK |

**Eliminated:** XGBoost / LightGBM at full power (will overfit 80 features × 4,800 rows), KNN (high-dim distances unreliable)
**Shortlist:** Logistic Regression with L1 (auto-selects relevant ~15 features from 80), Random Forest with limited depth, Gradient Boosting with early stopping
**Expected winner:** Lasso Logistic (interpretable + handles feature explosion) or Random Forest with feature importance

---

### Case Study 6: Medical Diagnosis (Disease Detection)
**Data:** 600 patient records, 25 features (lab values, vitals, history), 35% positive

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 600 | Small |
| Q2 (features) | 25 | Moderate |
| Q3 (linear?) | Often non-linear (lab thresholds) | Trees or RBF SVM |
| Q4 (noise) | Possible measurement variation | Ridge Logistic helps |
| Q5 (explain?) | Doctor must justify diagnosis | Logistic + SHAP mandatory |
| Q6 (speed?) | Real-time (clinic visit) | Fast prediction |
| Q7 (balance?) | 65/35 — mild | No special handling |
| Q8 (binary?) | Binary | All OK |

**Eliminated:** XGBoost / LightGBM (only 600 rows — overfit risk), Random Forest (heavy for the data size)
**Shortlist:** Logistic Regression with L2, SVM with RBF kernel, Decision Tree (fully transparent)
**Expected winner:** Logistic Regression with L2 (interpretable + small-data-friendly) or Decision Tree

---

### Case Study 7: Titanic Survival Prediction
**Data:** 891 rows, 15 features (mixed types, missing values), 38% survived

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 891 | Small-medium |
| Q2 (features) | 15 | Few features after engineering |
| Q3 (linear?) | Non-linear (sex × class interactions) | Trees preferred |
| Q4 (noise) | Heavy missing (Cabin, Age) | Trees handle missing; Logistic needs imputation |
| Q5 (explain?) | Educational | Want both interpretable and accurate |
| Q6 (speed?) | No constraint | No constraint |
| Q7 (balance?) | 62/38 — balanced | No special handling |
| Q8 (binary?) | Binary | All OK |

**Eliminated:** LightGBM / heavy XGBoost (overkill for 891 rows), KNN (mixed types + missing values hurt distances)
**Shortlist:** Logistic Regression, Random Forest, Gradient Boosting (with early stopping), Decision Tree (educational baseline)
**Expected winner:** Random Forest or Gradient Boosting (handle missing values + interactions out of the box)

---

### Case Study 8: Sentiment Analysis (Movie Reviews)
**Data:** 25,000 reviews, 10,000+ TF-IDF features, 50/50 positive/negative

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 25K | Plenty |
| Q2 (features) | 10K+ TF-IDF | Sparse high-dim — only linear models + NB |
| Q3 (linear?) | Linear in TF-IDF space | Logistic / Linear SVM excel |
| Q4 (noise) | Some sarcasm-misclassified | Regularization helps |
| Q5 (explain?) | Useful: "what words drove this?" | Logistic coefficients are gold |
| Q6 (speed?) | Real-time | Fast |
| Q7 (balance?) | 50/50 | No issue |
| Q8 (binary?) | Binary | All OK |

**Eliminated:** KNN (10K dims), Random Forest / GBM (slow on sparse), Decision Tree (overfits)
**Shortlist:** Logistic Regression with L2, Linear SVM, Multinomial Naive Bayes
**Expected winner:** Logistic Regression with L2 (best accuracy + interpretable per-word coefficients)

---

### Case Study 9: News Topic Classification (20 Newsgroups subset)
**Data:** 4,475 posts, 5,000+ TF-IDF features, 4 topics (multiclass), roughly balanced

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 4,475 | Moderate |
| Q2 (features) | 5,000+ TF-IDF | Very high-dim sparse |
| Q3 (linear?) | Linear in TF-IDF | Linear models excel |
| Q4 (noise) | Some cross-topic posts | Regularization helps |
| Q5 (explain?) | Want top words per topic | Logistic coefficients per class |
| Q6 (speed?) | Batch | No constraint |
| Q7 (balance?) | Roughly balanced | No special handling |
| Q8 (multiclass?) | 4 classes | Native or OvR |

**Eliminated:** KNN, tree-based models (slow on 5K sparse features), SVM with OvO (would create 6 binary models)
**Shortlist:** Multinomial Naive Bayes, Logistic Regression with L2 (Softmax), Linear SVM with OvR
**Expected winner:** Logistic Regression Softmax or Multinomial NB

---

### Case Study 10: Handwritten Digit Recognition
**Data:** 60,000 images, 784 pixel features (28×28), 10 classes (0–9), balanced

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 60K | Large |
| Q2 (features) | 784 | Many but dense |
| Q3 (linear?) | Highly non-linear | Trees, KNN, RBF SVM |
| Q4 (noise) | Some ambiguous handwriting | RF / GBM smooth |
| Q5 (explain?) | Educational, no business constraint | No need |
| Q6 (speed?) | Real-time inference desirable | Avoid KNN at this size |
| Q7 (balance?) | Balanced | No issue |
| Q8 (multiclass?) | 10 classes | Native multiclass models |

**Eliminated:** Naive Bayes (independence violated by pixel correlations), Logistic Regression alone (linear, weak on images), KNN at full data (60K × 784 = too slow at predict time)
**Shortlist:** Random Forest, XGBoost, LightGBM, SVM with RBF kernel (on a subset), KNN (subset baseline)
**Expected winner:** Random Forest or XGBoost (deep learning would beat them, but among classical methods: XGBoost wins)

---

### Case Study 11: Iris Flower Species Classification
**Data:** 150 rows, 4 features (sepal/petal length/width), 3 species (50 each), perfectly balanced

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 150 | Tiny |
| Q2 (features) | 4 | Very few |
| Q3 (linear?) | Mostly linear (one species linearly separable from the others) | Linear works for 1 vs rest |
| Q4 (noise) | None | No concern |
| Q5 (explain?) | Educational | Want interpretable |
| Q6 (speed?) | Trivial | No constraint |
| Q7 (balance?) | Perfectly balanced | No issue |
| Q8 (multiclass?) | 3 classes | Any multiclass model |

**Eliminated:** Random Forest, XGBoost, LightGBM (overkill), Naive Bayes (correlations violated)
**Shortlist:** Logistic Regression (Softmax), Decision Tree, KNN, SVM
**Expected winner:** Logistic Regression Softmax or KNN (textbook accuracy ~97% on this dataset)

---

### Case Study 12: Credit Risk Tier (Low / Medium / High)
**Data:** 50,000 applications, 35 features, 3 ordinal classes (50% / 35% / 15%)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 50K | Plenty |
| Q2 (features) | 35 | Moderate |
| Q3 (linear?) | Mostly linear (income, DTI ladder up tiers) | Logistic OK |
| Q4 (noise) | Clean | No concern |
| Q5 (explain?) | Required (lending) | Logistic / Tree |
| Q6 (speed?) | Batch underwriting | No constraint |
| Q7 (balance?) | 50/35/15 — mild imbalance for "high risk" | `class_weight` |
| Q8 (multiclass?) | 3 ordinal classes | Native multiclass |

**Eliminated:** KNN, plain SVM with OvO (slow), Naive Bayes (correlations)
**Shortlist:** Logistic Regression Softmax, Random Forest, XGBoost (`objective='multi:softmax'`)
**Expected winner:** XGBoost (best accuracy) or Logistic Regression (regulatory simplicity)

---

### Case Study 13: Click-Through Rate Prediction
**Data:** 50M ad impressions, 40 features (user, ad, context), ~3% CTR

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 50M | Massive |
| Q2 (features) | 40 | Moderate |
| Q3 (linear?) | Non-linear (user × ad × context interactions) | Trees preferred |
| Q4 (noise) | Clicks themselves are noisy | Ensemble averaging helps |
| Q5 (explain?) | Some, for advertisers | SHAP |
| Q6 (speed?) | Real-time bidding (~ms) | Fast prediction critical |
| Q7 (balance?) | 97/3 — severe | `scale_pos_weight`, log-loss |
| Q8 (binary?) | Binary | All OK |

**Eliminated:** KNN, SVM, Naive Bayes (calibration), Decision Tree alone, plain Random Forest (too slow at predict time)
**Shortlist:** LightGBM (designed for this), XGBoost, Logistic Regression with L2 (calibrated, fast baseline)
**Expected winner:** LightGBM

---

### Case Study 14: Product Recommendation Category
**Data:** 200K customers, 60 features, 50 product categories (multiclass, long-tail)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 200K | Plenty |
| Q2 (features) | 60 | Moderate |
| Q3 (linear?) | Non-linear (purchase history × demographics) | Trees |
| Q4 (noise) | Some cold-start customers | Ensembles |
| Q5 (explain?) | Marketing wants segment view | SHAP / feature importance |
| Q6 (speed?) | Batch | No constraint |
| Q7 (balance?) | Imbalanced (long-tail categories) | `class_weight` |
| Q8 (multiclass?) | 50 classes | AVOID OvO SVM (1,225 binary models!) |

**Eliminated:** SVM with OvO, KNN (200K predictions), Naive Bayes (60 mixed features → assumption breaks)
**Shortlist:** XGBoost (`multi:softprob`), LightGBM (multiclass), Random Forest, Logistic Regression Softmax
**Expected winner:** XGBoost or LightGBM

---

### Case Study 15: Image Classification — Cat vs Dog vs Bird (Small Data)
**Data:** 1,200 images (400 each), 100 features (engineered: color histogram + HOG), balanced 3-class

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 1,200 | Small |
| Q2 (features) | 100 | Moderate-high for 1,200 rows |
| Q3 (linear?) | Highly non-linear | RBF SVM, trees |
| Q4 (noise) | Some mislabels possible | Regularization |
| Q5 (explain?) | Educational | No constraint |
| Q6 (speed?) | Batch | No constraint |
| Q7 (balance?) | Balanced | No issue |
| Q8 (multiclass?) | 3 classes | All OK |

**Eliminated:** LightGBM / large XGBoost (only 1,200 rows), KNN (curse of dimensionality with 100 features)
**Shortlist:** SVM with RBF kernel (sweet spot — small data + non-linear), Random Forest, Gradient Boosting with early stopping, Logistic Regression with L2 (baseline)
**Expected winner:** SVM with RBF kernel

---

### Case Study 16: Network Intrusion Detection
**Data:** 500K connection records, 41 features, 7% intrusion (multiclass: normal + 4 attack types)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 500K | Large |
| Q2 (features) | 41 | Moderate |
| Q3 (linear?) | Non-linear (attack signatures) | Trees |
| Q4 (noise) | Some misclassified attacks | Ensembles |
| Q5 (explain?) | SOC analysts want explanations | SHAP |
| Q6 (speed?) | Near real-time alerting | Fast inference |
| Q7 (balance?) | 93/7 binary view; per-class imbalanced | `class_weight` |
| Q8 (multiclass?) | 5 classes | Tree ensembles OK |

**Eliminated:** KNN (500K = too slow), SVM (won't scale)
**Shortlist:** Random Forest + threshold tuning, XGBoost with multiclass, LightGBM
**Expected winner:** Random Forest (interpretable + handles imbalance well) or LightGBM

---

### Case Study 17: Customer Support Ticket Routing
**Data:** 80,000 tickets, 8,000+ TF-IDF features, 12 departments (multiclass), uneven volume

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 80K | Plenty |
| Q2 (features) | 8K+ TF-IDF | High-dim sparse |
| Q3 (linear?) | Linear in TF-IDF | Linear models |
| Q4 (noise) | Some routing errors in training labels | Regularization helps |
| Q5 (explain?) | Want top words per department | Logistic coefficients |
| Q6 (speed?) | Real-time (route on submission) | Fast |
| Q7 (balance?) | Uneven (long tail) | `class_weight` |
| Q8 (multiclass?) | 12 classes | Native or OvR |

**Eliminated:** KNN, tree-based (slow on 8K sparse), SVM with OvO (66 binary models)
**Shortlist:** Multinomial Naive Bayes, Logistic Regression Softmax, Linear SVM with OvR
**Expected winner:** Logistic Regression Softmax (best accuracy + per-class explanations)

---

### Case Study 18: Manufacturing Defect Classification
**Data:** 30,000 parts, 200 sensor features, 3% defect rate (binary, severe imbalance)

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 30K | Plenty |
| Q2 (features) | 200 | Many — sensor correlations |
| Q3 (linear?) | Non-linear (sensor thresholds) | Trees |
| Q4 (noise) | Sensor noise present | Ensembles |
| Q5 (explain?) | Engineers want to know which sensors | SHAP |
| Q6 (speed?) | Real-time (production line) | Fast |
| Q7 (balance?) | 97/3 — severe | `class_weight`, PR curve |
| Q8 (binary?) | Binary | All OK |

**Eliminated:** KNN (200 features = bad distances), Naive Bayes (correlations), plain Logistic (200 correlated sensors → multicollinearity)
**Shortlist:** Lasso Logistic (sensor selection), Random Forest, LightGBM with `is_unbalance=True` + SHAP
**Expected winner:** LightGBM (best accuracy + handles imbalance natively)

---

### Case Study 19: Hospital Readmission Risk
**Data:** 100,000 patient discharges, 50 features (mixed types), 11% readmitted within 30 days

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 100K | Plenty |
| Q2 (features) | 50 | Moderate |
| Q3 (linear?) | Mix | Try both |
| Q4 (noise) | Missing values common | Trees handle natively, Logistic needs imputation |
| Q5 (explain?) | HIPAA + clinician trust | Logistic + SHAP |
| Q6 (speed?) | Real-time at discharge | Fast |
| Q7 (balance?) | 89/11 — moderate | `class_weight`, F1 |
| Q8 (binary?) | Binary | All OK |

**Eliminated:** KNN (100K = slow), SVM (slow + hard to explain)
**Shortlist:** Logistic Regression with L2, Random Forest, XGBoost + SHAP
**Expected winner:** Logistic Regression for production explainability, XGBoost for raw performance

---

### Case Study 20: Document Language Identification
**Data:** 200,000 documents, 3,000+ character n-gram features, 25 languages (multiclass), uneven distribution

| Question | Answer | Effect |
|----------|--------|--------|
| Q1 (rows) | 200K | Plenty |
| Q2 (features) | 3K+ char n-grams | Very high-dim sparse |
| Q3 (linear?) | Linear in n-gram space | Linear models excel |
| Q4 (noise) | Some misclassified scripts | Regularization helps |
| Q5 (explain?) | Educational | No constraint |
| Q6 (speed?) | Real-time per document | Fast |
| Q7 (balance?) | English dominant, others rare | `class_weight` |
| Q8 (multiclass?) | 25 classes | AVOID OvO entirely (300 binary models) |

**Eliminated:** SVM with OvO, KNN (200K + 3K dims), tree-based (slow on sparse), heavy ensembles
**Shortlist:** Multinomial Naive Bayes, Logistic Regression Softmax, Linear SVM with OvR
**Expected winner:** Multinomial Naive Bayes (gold standard for language ID — fast + accurate)

---

## Summary Pattern

Across all 20 cases:

1. **Small data (< 500 rows)** — Always lands on Logistic Regression / Naive Bayes / Linear SVM
2. **Large data (> 100K rows)** — Always includes LightGBM or XGBoost
3. **High-dim sparse text (TF-IDF)** — Always includes Multinomial NB or Logistic with L1/L2
4. **Severe imbalance (< 5% minority)** — Always uses GBM with `scale_pos_weight` and PR-AUC, never accuracy
5. **Many classes (> 10)** — Always avoids SVM with OvO; favors Softmax Logistic, trees, or Multinomial NB
6. **Need explanations** — Always includes Logistic Regression or Decision Tree (or SHAP atop ensembles)
7. **Outliers / mislabeled** — Always avoids plain Decision Tree and KNN, favors Random Forest or regularized linear models
8. **Real-time prediction** — Always avoids KNN; favors Logistic, Naive Bayes, or small trees

**The shortlist is always 2–4 models. Then you train them all and let the metrics decide.**

---

## Complete Model-by-Model Reference Tables

Every model rated across all 8 guidelines. No model left out.

### Guideline 1: Dataset Size (Number of Rows)

| Model | < 100 Rows | 100–500 Rows | 500–10K Rows | 10K–100K Rows | 100K+ Rows |
|-------|-----------|-------------|-------------|--------------|-----------|
| Logistic Regression | YES — works great | YES | YES | YES | YES |
| Logistic + L2 (Ridge) | YES — best small-data pick | YES | YES | YES | YES |
| Logistic + L1 (Lasso) | YES — best if many features | YES | YES | YES | YES |
| Logistic + ElasticNet | YES | YES | YES | YES | YES |
| Decision Tree | RISKY — will memorize data | OK with pruning | YES | YES | YES |
| Random Forest | NO — not enough data for diverse trees | RISKY — may overfit | YES — sweet spot | YES | YES but slow training |
| Gradient Boosting | NO — overfits small data | RISKY | YES — sweet spot | YES | SLOW training |
| XGBoost | NO | RISKY | YES — sweet spot | YES | YES |
| LightGBM | NO — overkill | RISKY | YES | YES — sweet spot | YES — fastest |
| SVM | YES — excels on small data | YES — sweet spot | YES | SLOW training | NO — won't scale |
| KNN | RISKY — too few neighbors | OK | YES | SLOW prediction | NO — O(n) per prediction |
| Naive Bayes | YES — works with very little | YES | YES | YES | YES — very fast |

---

### Guideline 2: Number of Features (Columns)

| Model | < 10 Features | 10–50 Features | 50–200 Features | 200+ Features |
|-------|--------------|---------------|----------------|--------------|
| Logistic Regression | YES | OK — watch multicollinearity | RISKY — overfits if rows < features | NO — unstable coefficients |
| Logistic + L2 | YES | YES — handles correlated features | YES — regularization prevents overfitting | OK with enough rows |
| Logistic + L1 | YES | YES — auto-selects important features | YES — zeros out irrelevant features | YES — best for feature selection |
| Logistic + ElasticNet | YES | YES | YES — combines L1 + L2 benefits | YES — best for correlated + many features |
| Decision Tree | YES | YES | RISKY — may split on noise features | NO — overfits on irrelevant features |
| Random Forest | YES | YES — handles many features naturally | YES — random subsets help | OK — slower but works |
| Gradient Boosting | YES | YES | OK — can overfit without tuning | RISKY — slow + overfitting |
| XGBoost | YES | YES — built-in feature importance | YES — built-in regularization | OK with regularization |
| LightGBM | YES | YES | YES — fast even with many features | YES — handles natively |
| SVM | YES (linear or RBF) | YES (RBF) | RISKY — slow with many features | NO — training time explodes; linear OK only |
| KNN | YES | OK — distances still meaningful | RISKY — curse of dimensionality | NO — distances meaningless in high dimensions |
| Naive Bayes | YES | YES | YES — assumes feature independence | YES — Multinomial NB excels (TF-IDF) |

---

### Guideline 3: Linearly Separable vs Non-Linear Class Boundaries

| Model | Linearly Separable | Mildly Non-Linear | Highly Non-Linear | Complex Interactions Between Features |
|-------|-------------------|-------------------|-------------------|--------------------------------------|
| Logistic Regression | EXCELLENT — designed for this | POOR — can't capture curves | POOR | POOR — must manually create interaction features |
| Logistic + L2 | EXCELLENT | POOR | POOR | POOR |
| Logistic + L1 | EXCELLENT | POOR | POOR | POOR |
| Logistic + ElasticNet | EXCELLENT | POOR | POOR | POOR |
| Decision Tree | OK — many splits to approximate | GOOD — captures step-like patterns | GOOD | GOOD — naturally captures interactions |
| Random Forest | OK | GOOD | EXCELLENT — ensemble smooths the steps | EXCELLENT — each tree finds different interactions |
| Gradient Boosting | OK | GOOD | EXCELLENT | EXCELLENT — sequential correction finds complex patterns |
| XGBoost | OK | GOOD | EXCELLENT | EXCELLENT — best for tabular non-linear tasks |
| LightGBM | OK | GOOD | EXCELLENT | EXCELLENT — same as XGBoost |
| SVM (linear kernel) | EXCELLENT | POOR | POOR | POOR |
| SVM (RBF kernel) | GOOD | EXCELLENT | EXCELLENT | OK — kernel trick helps but limited |
| KNN | OK | GOOD — local patterns captured | GOOD — but needs lots of data | POOR — doesn't generalize interactions |
| Naive Bayes | EXCELLENT for simple linear | POOR | POOR | POOR — assumes feature independence |

---

### Guideline 4: Robustness to Noise, Outliers, and Mislabeled Examples

| Model | Feature Outliers | Mislabeled Examples (Label Noise) | Noisy/Messy Data | Missing Values (natively) |
|-------|-----------------|----------------------------------|-----------------|--------------------------|
| Logistic Regression | POOR — coefficients distorted | POOR — fits noise | POOR | NO — must impute first |
| Logistic + L2 | POOR | OK — regularization smooths | OK | NO — must impute first |
| Logistic + L1 | POOR | OK | OK | NO — must impute first |
| Logistic + ElasticNet | POOR | OK | OK | NO — must impute first |
| Decision Tree | GOOD — only uses rank order | POOR — memorizes wrong labels | POOR — memorizes noise | SOME implementations handle it |
| Random Forest | GOOD | GOOD — averaging dilutes bad labels | GOOD — averaging smooths noise | SOME implementations handle it |
| Gradient Boosting | OK | OK — early stopping helps | OK | SOME implementations handle it |
| XGBoost | OK | GOOD — built-in regularization | GOOD | YES — learns optimal direction for missing |
| LightGBM | OK | GOOD | GOOD | YES — handles natively |
| SVM | POOR — distances distorted | OK — soft margin tolerates some | OK | NO — must impute first |
| KNN | VERY POOR — outlier neighbors corrupt predictions | VERY POOR — bad neighbor = bad prediction | POOR — directly uses noisy values | NO — must impute first |
| Naive Bayes | OK — probabilistic | OK — Laplace smoothing helps | OK — smoothing helps | SOME implementations handle it |

---

### Guideline 5: Interpretability / Explainability

| Model | Interpretability Level | What You Can Explain | Suitable for Regulated Industries | Can Add SHAP? |
|-------|----------------------|---------------------|----------------------------------|--------------|
| Logistic Regression | EXCELLENT — fully transparent | "Each unit of X multiplies odds by exp(β)" — exact log-odds | YES — gold standard | YES (but not needed) |
| Logistic + L2 | EXCELLENT | Same — coefficients (slightly shrunk) | YES | YES (but not needed) |
| Logistic + L1 | EXCELLENT | Same + "these features were dropped (coeff = 0)" | YES — shows which features matter | YES (but not needed) |
| Logistic + ElasticNet | EXCELLENT | Same as L1 | YES | YES (but not needed) |
| Decision Tree | EXCELLENT — visual rules | "If X > 5 AND Y < 3, predict class A" — full decision path | YES — easy to audit | YES (but not needed) |
| Random Forest | POOR — black box | Feature importance ranking only | ONLY with SHAP | YES — commonly used |
| Gradient Boosting | POOR — black box | Feature importance ranking only | ONLY with SHAP | YES — commonly used |
| XGBoost | POOR — black box | Feature importance ranking only | ONLY with SHAP | YES — commonly used |
| LightGBM | POOR — black box | Feature importance ranking only | ONLY with SHAP | YES — commonly used |
| SVM (linear) | GOOD | Coefficients per feature | YES | YES |
| SVM (RBF) | POOR — black box | Nothing intuitive | NO | YES but less common |
| KNN | MODERATE — instance-based | "These 5 similar cases voted for this class" — show neighbors | OK for small datasets | YES but unusual |
| Naive Bayes | GOOD | Conditional probability per feature per class — "P(class=A given feature=X) = 0.8" | OK | YES but unusual |

---

### Guideline 6: Prediction Speed and Training Speed

| Model | Training Speed (Time to Learn) | Prediction Speed (Time per New Row) | Scales to Millions of Rows | Memory Usage |
|-------|-------------------------------|------------------------------------|-----------------------------|-------------|
| Logistic Regression | VERY FAST — milliseconds | VERY FAST — microseconds | YES | LOW |
| Logistic + L2 | VERY FAST — milliseconds | VERY FAST — microseconds | YES | LOW |
| Logistic + L1 | FAST — seconds (iterative solver) | VERY FAST — microseconds | YES | LOW |
| Logistic + ElasticNet | FAST — seconds (iterative solver) | VERY FAST — microseconds | YES | LOW |
| Decision Tree | FAST — seconds | FAST — microseconds | YES | LOW |
| Random Forest | MODERATE — minutes (100 trees) | MODERATE — milliseconds (traverse 100 trees) | OK — parallelizable | MODERATE — stores all trees |
| Gradient Boosting | SLOW — minutes (sequential trees) | MODERATE — milliseconds | POOR — sequential training | MODERATE |
| XGBoost | MODERATE — faster than sklearn GB | MODERATE — milliseconds | OK — optimized C++ | MODERATE |
| LightGBM | FAST — fastest tree method | MODERATE — milliseconds | YES — designed for scale | MODERATE |
| SVM | VERY SLOW — O(n² to n³) | MODERATE | NO — won't finish training | HIGH — stores support vectors |
| KNN | NONE — no training (stores data) | VERY SLOW — O(n) per prediction | NO — must scan all rows | HIGH — stores entire dataset |
| Naive Bayes | VERY FAST — single pass | VERY FAST — microseconds | YES | LOW |

---

### Guideline 7: Class Imbalance Handling

| Model | Balanced (50/50) | Mild (80/20) | Severe (95/5) | Extreme (99/1) | Native Handling Parameter |
|-------|-----------------|--------------|--------------|---------------|--------------------------|
| Logistic Regression | YES | YES with `class_weight` | OK with `class_weight` + threshold | OK with calibration | `class_weight='balanced'` |
| Logistic + L2 | YES | YES with `class_weight` | OK | OK | `class_weight='balanced'` |
| Logistic + L1 | YES | YES with `class_weight` | OK | OK | `class_weight='balanced'` |
| Logistic + ElasticNet | YES | YES with `class_weight` | OK | OK | `class_weight='balanced'` |
| Decision Tree | YES | OK with `class_weight` | POOR — splits dominated by majority | POOR | `class_weight='balanced'` |
| Random Forest | YES | YES with `class_weight` | OK with `class_weight` + balanced bootstrap | OK | `class_weight='balanced'` or `balanced_subsample` |
| Gradient Boosting | YES | OK | OK with sample weights | RISKY without tuning | `sample_weight` |
| XGBoost | YES | YES with `scale_pos_weight` | YES — designed for it | YES — top choice | `scale_pos_weight` |
| LightGBM | YES | YES with `is_unbalance` | YES | YES — top choice | `is_unbalance` or `scale_pos_weight` |
| SVM | YES | YES with `class_weight` | OK | POOR — boundary collapses | `class_weight='balanced'` |
| KNN | YES | POOR — neighbors dominated by majority | VERY POOR | VERY POOR | NO — needs SMOTE or undersampling |
| Naive Bayes | YES | OK | POOR — priors dominate | POOR | NO — needs resampling |

**Note:** For severe / extreme imbalance, ALWAYS pair the model with PR-AUC (not ROC-AUC) and threshold tuning instead of the default 0.5 cutoff.

---

### Guideline 8: Binary vs Multiclass Support

| Model | Binary | Native Multiclass | OvR Wrapper Available | OvO Wrapper Available | Scales to 20+ Classes |
|-------|--------|-------------------|----------------------|----------------------|----------------------|
| Logistic Regression | YES — native | YES — Softmax | YES | NO | YES — Softmax efficient |
| Logistic + L2 | YES | YES — Softmax | YES | NO | YES |
| Logistic + L1 | YES | YES — Softmax | YES | NO | YES |
| Logistic + ElasticNet | YES | YES — Softmax | YES | NO | YES |
| Decision Tree | YES | YES — natively splits any number of classes | NO need | NO | YES |
| Random Forest | YES | YES — natively | NO need | NO | YES |
| Gradient Boosting | YES | YES — one-vs-rest internally | NO need | NO | OK — slower than Softmax |
| XGBoost | YES | YES — `multi:softmax` / `multi:softprob` | NO need | NO | YES |
| LightGBM | YES | YES — `multiclass` objective | NO need | NO | YES |
| SVM | YES | NO — needs wrapper | YES (sklearn default) | YES (n*(n-1)/2 models) | POOR — explodes with OvO |
| KNN | YES | YES — voting works for any number | NO need | NO | YES |
| Naive Bayes | YES | YES — natively (Gaussian / Multinomial / Bernoulli) | NO need | NO | YES — fast even with many classes |

**Note:** Naive Bayes variant choice depends on feature type — Gaussian (continuous), Multinomial (counts / TF-IDF), Bernoulli (binary).

---

### Master Summary Table: All Models × All Guidelines

| Model | Small Data (< 500) | Large Data (> 10K) | Many Features (> 50) | Non-Linear | Outlier / Noise Robust | Interpretable | Fast Predict | Native Imbalance |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Logistic Regression | YES | YES | NO | NO | NO | YES | YES | YES |
| Logistic + L2 | YES | YES | YES | NO | OK | YES | YES | YES |
| Logistic + L1 | YES | YES | YES | NO | OK | YES | YES | YES |
| Logistic + ElasticNet | YES | YES | YES | NO | OK | YES | YES | YES |
| Decision Tree | OK | YES | NO | YES | NO | YES | YES | OK |
| Random Forest | NO | YES | YES | YES | YES | NO | OK | YES |
| Gradient Boosting | NO | OK | OK | YES | OK | NO | OK | OK |
| XGBoost | NO | YES | YES | YES | OK | NO | OK | YES |
| LightGBM | NO | YES | YES | YES | OK | NO | OK | YES |
| SVM | YES | NO | NO | YES (RBF) | OK | OK (linear only) | OK | YES |
| KNN | OK | NO | NO | YES | NO | OK | NO | NO |
| Naive Bayes | YES | YES | YES | NO | OK | YES | YES | NO |
