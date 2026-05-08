# Start Here — A First-Timer's Guide to This Repo

A 4-week learning path through the 12 classical ML algorithms in this repo. Read it before you read anything else.

---

## The Mental Model First

When you sit down with a real ML problem, you go through **5 questions in order**, and this repo is organized to answer each one:

| # | Question | Where the repo answers it |
|---|----------|---------------------------|
| Q1 | What kind of problem is this? | [PROBLEM_TYPES_GUIDE.md](PROBLEM_TYPES_GUIDE.md) |
| Q2 | Which algorithm should I try? | [MODEL_SELECTION_GUIDE_REGRESSION.md](MODEL_SELECTION_GUIDE_REGRESSION.md) + [MODEL_SELECTION_GUIDE_CLASSIFICATION.md](MODEL_SELECTION_GUIDE_CLASSIFICATION.md) + [MODEL_SELECTION_GUIDE_CLUSTERING.md](MODEL_SELECTION_GUIDE_CLUSTERING.md) |
| Q3 | How does that algorithm work? | `HOW_IT_WORKS_XX_*.md` + `algorithms/XX_*/` |
| Q4 | How do I run the full pipeline? | [ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md) + [FEATURE_SELECTION_GUIDE.md](FEATURE_SELECTION_GUIDE.md) |
| Q5 | How does this play out in industry? | `EXPERT_SCENARIO_X_*.md` + [real_world_practice/](real_world_practice/) |

**So your learning path mirrors that flow — don't read the repo top to bottom, read it in the order you'd use it on a real problem.**

---

## Recommended 4-Week Path

### Week 1 — Build the Framing Brain (No Code Yet)

Read in this exact order. Each is short:

1. [PROBLEM_TYPES_GUIDE.md](PROBLEM_TYPES_GUIDE.md) — learn the 4 problem types (regression / classification / clustering / dimensionality reduction). After this you should be able to look at any business question and instantly say "this is a regression problem."
2. [ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md) — the universal 12-phase pipeline every ML task follows. This is the skeleton you'll fill in for every project.
3. [MODEL_SELECTION_GUIDE_REGRESSION.md](MODEL_SELECTION_GUIDE_REGRESSION.md), [MODEL_SELECTION_GUIDE_CLASSIFICATION.md](MODEL_SELECTION_GUIDE_CLASSIFICATION.md), and [MODEL_SELECTION_GUIDE_CLUSTERING.md](MODEL_SELECTION_GUIDE_CLUSTERING.md) — given problem type and data shape, which algorithm should you reach for first?
4. [FEATURE_SELECTION_GUIDE.md](FEATURE_SELECTION_GUIDE.md) — how to pick which columns matter.

**Goal at end of week 1:** if someone described a problem to you in plain English, you could tell them *what type of problem it is, what algorithms to try, and what pipeline steps you'd follow* — without writing a single line of code.

---

### Week 2 — Learn the Algorithms One at a Time (Theory + Code)

For each algorithm 01 → 12, do this loop:

1. Read `HOW_IT_WORKS_0X_*.md` — the deep "why does it work" intuition.
2. Read `algorithms/0X_*/README.md` — math + key concepts.
3. Run `algorithms/0X_*/<algorithm>.py` and look at the generated `plots/`.
4. Open the `.py` file and re-read it slowly — it's the from-scratch implementation.

**Order matters** — go 01 → 12, because each one builds on the last:

| # | Algorithm | What it teaches |
|---|-----------|-----------------|
| 01 | [Linear Regression](algorithms/01_linear_regression/) — [How It Works](HOW_IT_WORKS_01_LINEAR_REGRESSION.md) | Gradient descent, the foundation of everything |
| 02 | [Logistic Regression](algorithms/02_logistic_regression/) — [How It Works](HOW_IT_WORKS_02_LOGISTIC_REGRESSION.md) | Adapt linear models to classification (sigmoid, log-odds) |
| 03 | [Decision Trees](algorithms/03_decision_trees/) — [How It Works](HOW_IT_WORKS_03_DECISION_TREES.md) | A different paradigm: splits instead of equations |
| 04 | [Random Forest](algorithms/04_random_forest/) — [How It Works](HOW_IT_WORKS_04_RANDOM_FOREST.md) | Bag many trees to reduce variance |
| 05 | [Gradient Boosting](algorithms/05_gradient_boosting/) — [How It Works](HOW_IT_WORKS_05_GRADIENT_BOOSTING.md) | Boost trees sequentially to reduce bias |
| 06 | [SVM](algorithms/06_svm/) — [How It Works](HOW_IT_WORKS_06_SVM.md) | Maximum-margin classifiers, the kernel trick |
| 07 | [KNN](algorithms/07_knn/) — [How It Works](HOW_IT_WORKS_07_KNN.md) | The simplest "no training" algorithm |
| 08 | [Naive Bayes](algorithms/08_naive_bayes/) — [How It Works](HOW_IT_WORKS_08_NAIVE_BAYES.md) | Probabilistic classification, the gold standard for text |
| 09 | [k-Means](algorithms/09_kmeans/) — [How It Works](HOW_IT_WORKS_09_KMEANS.md) | First unsupervised algorithm — find groups |
| 10 | [DBSCAN](algorithms/10_dbscan/) — [How It Works](HOW_IT_WORKS_10_DBSCAN.md) | Density-based clustering + outlier detection |
| 11 | [PCA](algorithms/11_pca/) — [How It Works](HOW_IT_WORKS_11_PCA.md) | Compress features while keeping information |
| 12 | [Boosted Trees (XGBoost/LightGBM)](algorithms/12_boosted_trees/) — [How It Works](HOW_IT_WORKS_12_BOOSTED_TREES.md) | The industrial-grade version of #5 |

The chain: linear regression teaches gradient descent → logistic uses it → trees introduce splits → random forest bags trees → gradient boosting boosts trees → boosted trees (XGBoost / LightGBM) is the optimized industrial version.

**Goal at end of week 2:** for any of the 12 algorithms, you can sketch on paper how it makes a prediction and what its main hyperparameters do.

---

### Week 3 — Run Real, Messy Pipelines

This is where it clicks. The [real_world_practice/](real_world_practice/) folder is the bridge from "I know the algorithm" to "I can solve a problem." Do them in this order:

1. [real_world_practice/classification/titanic_pipeline.py](real_world_practice/classification/titanic_pipeline.py) — easiest. Missing values, mixed types, class imbalance.
2. [real_world_practice/regression/ames_housing_pipeline.py](real_world_practice/regression/ames_housing_pipeline.py) — 80+ features, skewed target. You'll need everything from week 1.
3. [real_world_practice/advanced_techniques/advanced_ml_pipeline.py](real_world_practice/advanced_techniques/advanced_ml_pipeline.py) — text data, deployment, drift detection. The "industry-grade" pipeline.

For each:
- **Run it once first** to see the output.
- **Read it slowly top to bottom**, mapping every block back to a step in [ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md).
- When you see something you don't understand, the relevant `HOW_IT_WORKS` doc is your reference.

**Goal at end of week 3:** you can take a messy CSV, run an end-to-end pipeline, compare 3+ models with cross-validation, and tune the winner — without copy-pasting from a tutorial.

---

### Week 4 — Stress-Test Yourself With Industry Scenarios

The 10 `EXPERT_SCENARIO_*.md` files are the final exam. Each one describes a realistic industry problem.

| # | Scenario |
|---|----------|
| 1 | [Fraud Detection](EXPERT_SCENARIO_1_FRAUD_DETECTION.md) |
| 2 | [Hospital Length of Stay](EXPERT_SCENARIO_2_HOSPITAL_LOS.md) |
| 3 | [Ride Pricing](EXPERT_SCENARIO_3_RIDE_PRICING.md) |
| 4 | [Manufacturing](EXPERT_SCENARIO_4_MANUFACTURING.md) |
| 5 | [Insurance Claims](EXPERT_SCENARIO_5_INSURANCE_CLAIMS.md) |
| 6 | [Telecom Churn](EXPERT_SCENARIO_6_TELECOM_CHURN.md) |
| 7 | [Loan Fairness](EXPERT_SCENARIO_7_LOAN_FAIRNESS.md) |
| 8 | [Employee Attrition](EXPERT_SCENARIO_8_EMPLOYEE_ATTRITION.md) |
| 9 | [E-commerce Segmentation](EXPERT_SCENARIO_9_ECOMMERCE_SEGMENTATION.md) |
| 10 | [Energy Forecasting](EXPERT_SCENARIO_10_ENERGY_FORECASTING.md) |

For each: read the **problem statement only**, close the file, and answer on paper:

1. What problem type? (regression / classification / clustering / dim-reduction)
2. Which algorithms would I try, and why?
3. What would my pipeline look like? (steps from [ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md))
4. What metric matters, and why?

Then read the rest of the file to check yourself.

**Goal at end of week 4:** you can read any new business problem (not in the repo) and within 10 minutes propose a credible end-to-end ML approach.

---

## How to Use This Repo Day-to-Day

Once you've finished week 4, the repo becomes a **reference manual**, not a textbook. Real workflow:

1. New problem comes in → consult [PROBLEM_TYPES_GUIDE.md](PROBLEM_TYPES_GUIDE.md) to type it.
2. Pick a shortlist → consult the [Regression](MODEL_SELECTION_GUIDE_REGRESSION.md), [Classification](MODEL_SELECTION_GUIDE_CLASSIFICATION.md), or [Clustering](MODEL_SELECTION_GUIDE_CLUSTERING.md) guide.
3. Forgot how an algorithm works → re-read its `HOW_IT_WORKS_XX_*.md`.
4. Building the pipeline → use [ML_PIPELINE_GUIDE.md](ML_PIPELINE_GUIDE.md) as a checklist; copy patterns from the closest `real_world_practice/` example.
5. Stuck on feature selection → [FEATURE_SELECTION_GUIDE.md](FEATURE_SELECTION_GUIDE.md).

---

## Three Ways to Start Right Now

Pick one:

- **(A) Start week 1 today** — open [PROBLEM_TYPES_GUIDE.md](PROBLEM_TYPES_GUIDE.md) and follow the week 1 reading list. Quiz yourself afterward by inventing a fake business problem and classifying it.
- **(B) Skip straight to week 2, algorithm 01** — open [HOW_IT_WORKS_01_LINEAR_REGRESSION.md](HOW_IT_WORKS_01_LINEAR_REGRESSION.md), then [algorithms/01_linear_regression/README.md](algorithms/01_linear_regression/README.md), then run [algorithms/01_linear_regression/linear_regression.py](algorithms/01_linear_regression/linear_regression.py).
- **(C) Pick an expert scenario as a north star** — read one `EXPERT_SCENARIO_*.md` problem statement, then work backward through the relevant chapters as you need them.

There's no wrong choice — what matters is starting and staying consistent.
