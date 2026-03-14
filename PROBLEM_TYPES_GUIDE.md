# Problem Types in Classical Machine Learning

A guide to the **4 problem types** covered by the 12 algorithms in this project, with real-world use cases and a decision guide.

---

## 1. Regression — "Predict a number"

| Algorithm | Example Use Cases |
|-----------|-------------------|
| Linear Regression | House prices, salary prediction, sales forecasting |
| Decision Trees | Customer lifetime value, delivery time estimation |
| Random Forest | Insurance claims amount, stock price direction |
| Gradient Boosting | Revenue forecasting, demand prediction |
| KNN | Filling missing values, local price estimation |
| SVM (SVR) | Small-dataset regression with complex boundaries |
| Boosted Trees (XGB/LGBM) | Kaggle tabular competitions, credit scoring |

**Practice pipeline:** Ames Housing (predicting sale price from 80+ features)

---

## 2. Classification — "Predict a category"

| Algorithm | Example Use Cases |
|-----------|-------------------|
| Logistic Regression | Spam detection, churn prediction, medical diagnosis |
| Decision Trees | Loan approval, customer segmentation rules |
| Random Forest | Fraud detection, disease classification |
| Gradient Boosting | Click-through prediction, sentiment analysis |
| SVM | Image classification, text categorization (small data) |
| KNN | Recommendation (similar users), handwriting recognition |
| Naive Bayes | Spam filtering, document classification, NLP |
| Boosted Trees (XGB/LGBM) | Any tabular classification (often best performer) |

**Practice pipelines:** Titanic (survival prediction) + 20 Newsgroups (tech vs non-tech text classification)

---

## 3. Clustering — "Find natural groups" (unsupervised, no labels)

| Algorithm | Example Use Cases |
|-----------|-------------------|
| K-Means | Customer segmentation, image compression, market grouping |
| DBSCAN | Anomaly/outlier detection, geographic clustering, noise filtering |

**Key difference:** K-Means needs you to specify *k* groups upfront. DBSCAN finds groups automatically and handles noise/outliers.

---

## 4. Dimensionality Reduction — "Compress features"

| Algorithm | Example Use Cases |
|-----------|-------------------|
| PCA | Visualization of high-dimensional data, noise removal, speeding up other models, feature decorrelation |

**Used when:** You have too many features (e.g., Ames Housing's 80+ columns) and need to reduce them while keeping most information.

---

## Quick Decision Guide

```
What's your goal?
│
├── Predict a NUMBER?              --> Regression
│   ├── Small data, few features        --> Linear Regression
│   ├── Need interpretability           --> Decision Tree
│   ├── Best accuracy                   --> XGBoost / LightGBM
│   └── General strong baseline         --> Random Forest
│
├── Predict a CATEGORY?            --> Classification
│   ├── Text data                       --> Naive Bayes or Logistic Regression
│   ├── Need probabilities              --> Logistic Regression
│   ├── Small data, complex boundary    --> SVM
│   ├── Best accuracy                   --> XGBoost / LightGBM
│   └── Need explainability             --> Decision Tree
│
├── Find GROUPS in data?           --> Clustering
│   ├── Know how many groups            --> K-Means
│   └── Don't know / have outliers      --> DBSCAN
│
└── Too many FEATURES?             --> Dimensionality Reduction
    └── Compress / visualize            --> PCA
```

---

## What Classical ML Does NOT Cover

These 12 algorithms handle essentially every classical ML problem on **tabular and text data**. The major areas they don't cover are **deep learning tasks** — images, audio, video, and sequence generation — which require neural networks.
