# Classical Machine Learning from Scratch

A comprehensive project implementing **12 classical ML algorithms** from scratch using Python and scikit-learn, with real datasets, full explanations, and visualizations — plus **real-world practice pipelines** on messy datasets.

Built as a learning portfolio to deeply understand every core concept behind classical machine learning.

---

## Algorithms Implemented

| #  | Algorithm           | Type                     | Best For                        | Status |
|----|---------------------|--------------------------|---------------------------------|--------|
| 01 | Linear Regression   | Regression               | Predicting numeric values       | ✅     |
| 02 | Logistic Regression | Classification           | Simple classification           | ✅     |
| 03 | Decision Trees      | Both                     | Interpretable models            | ✅     |
| 04 | Random Forest       | Ensemble                 | Strong general baseline         | ✅     |
| 05 | Gradient Boosting   | Ensemble                 | High accuracy tabular tasks     | ✅     |
| 06 | SVM                 | Classification           | Small high-dimensional datasets | ✅     |
| 07 | KNN                 | Both                     | Similarity-based prediction     | ✅     |
| 08 | Naive Bayes         | Classification           | Text tasks                      | ✅     |
| 09 | k-Means             | Clustering               | Grouping data                   | ✅     |
| 10 | DBSCAN              | Clustering               | Irregular clusters              | ✅     |
| 11 | PCA                 | Dimensionality Reduction | Feature compression             | ✅     |
| 12 | Boosted Trees       | Ensemble                 | Best tabular performance        | ✅     |

---

## Project Structure

```
classical-ml-from-scratch/
├── README.md
├── requirements.txt
│
├── algorithms/                    # Algorithm deep-dives (theory + code)
│   ├── 01_linear_regression/
│   │   ├── README.md
│   │   ├── linear_regression.py
│   │   └── plots/
│   ├── 02_logistic_regression/
│   ├── 03_decision_trees/
│   ├── 04_random_forest/
│   ├── 05_gradient_boosting/
│   ├── 06_svm/
│   ├── 07_knn/
│   ├── 08_naive_bayes/
│   ├── 09_kmeans/
│   ├── 10_dbscan/
│   ├── 11_pca/
│   └── 12_boosted_trees/
│
└── real_world_practice/           # End-to-end ML pipelines on messy data
    ├── classification/
    │   ├── titanic_pipeline.py    # Titanic survival (891 rows, messy)
    │   ├── best_model.joblib
    │   └── plots/
    ├── regression/
    │   ├── ames_housing_pipeline.py  # Ames Housing prices (1460 rows, 80+ features)
    │   ├── best_model.joblib
    │   └── plots/
    └── advanced_techniques/
        ├── advanced_ml_pipeline.py   # 20 Newsgroups text classification (4,475 posts)
        ├── deploy_api.py             # Flask REST API for model serving
        ├── model.joblib
        ├── tfidf_vectorizer.joblib
        └── plots/
```

### algorithms/
Each folder contains:
- **README.md** — Theory, math, key concepts, and how-to-use guide
- **`<algorithm>.py`** — Full implementation with training, evaluation, and visualization
- **plots/** — Saved visualizations

### real_world_practice/
Complete ML pipelines practicing every skill needed for real-world projects:

| Pipeline | Dataset | Task | Features | Rows | Key Challenges |
|----------|---------|------|----------|------|----------------|
| **Classification** | Titanic (seaborn) | Survival prediction | 15 | 891 | Missing values, class imbalance, mixed types |
| **Regression** | Ames Housing (OpenML) | Price prediction | 80+ | 1,460 | Skewed target, ordinal categoricals, multicollinearity, outliers |
| **Advanced Techniques** | 20 Newsgroups (sklearn) | Text classification | 5,000+ | 4,475 | NLP text data, high-cardinality categoricals, deployment |

Each pipeline covers 15 real-world skills:
1. Data exploration & profiling
2. Data cleaning (missing values, outliers, duplicates)
3. Feature engineering (domain-driven)
4. EDA with visualizations
5. Preprocessing pipelines (`ColumnTransformer` + `Pipeline`)
6. Training & comparing all classical ML models with 5-fold CV
7. Class imbalance handling / target transformation
8. Unsupervised features (PCA, K-Means, DBSCAN)
9. Feature selection (Mutual Information, RFE, SelectKBest)
10. Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
11. Stacking ensemble (meta-learner combining diverse models)
12. SHAP explainability (global + local model explanations)
13. Learning curves (overfitting diagnosis)
14. Model persistence (joblib save/load)
15. Residual analysis / inference demo

The **Advanced Techniques** pipeline adds 6 additional skills:
16. Text / NLP features (TF-IDF, CountVectorizer, text statistics)
17. High-cardinality categorical encoding (Frequency, Target, Hashing)
18. Threshold tuning (Precision-Recall curves, optimal cutoffs, F-beta)
19. Cross-validation variants (StratifiedKFold, GroupKFold, TimeSeriesSplit)
20. Data drift detection & monitoring (KS test, PSI, confidence tracking)
21. Model deployment (Flask REST API with health check, batch predict)

---

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

```bash
# Algorithm deep-dives
cd algorithms/01_linear_regression
python linear_regression.py

# Real-world practice
cd real_world_practice/classification
python titanic_pipeline.py

cd real_world_practice/regression
python ames_housing_pipeline.py

cd real_world_practice/advanced_techniques
python advanced_ml_pipeline.py
# To serve predictions via REST API:
# pip install flask
# python deploy_api.py
```

---

## Author

Built as a hands-on ML learning project.
