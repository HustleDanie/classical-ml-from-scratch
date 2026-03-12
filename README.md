# Classical Machine Learning from Scratch

A comprehensive project implementing **12 classical ML algorithms** from scratch using Python and scikit-learn, with real datasets, full explanations, and visualizations.

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
├── 01_linear_regression/
│   ├── README.md
│   ├── linear_regression.py
│   └── plots/
├── 02_logistic_regression/
├── 03_decision_trees/
├── 04_random_forest/
├── 05_gradient_boosting/
├── 06_svm/
├── 07_knn/
├── 08_naive_bayes/
├── 09_kmeans/
├── 10_dbscan/
├── 11_pca/
└── 12_boosted_trees/
```

Each folder contains:
- **README.md** — Theory, math, key concepts, and how-to-use guide
- **<algorithm>.py** — Full implementation with training, evaluation, and visualization
- **plots/** — Saved visualizations

---

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

```bash
cd 01_linear_regression
python linear_regression.py
```

---

## Author

Built as a hands-on ML learning project.
