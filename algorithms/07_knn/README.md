# 07 — K-Nearest Neighbors (KNN)

## Overview

K-Nearest Neighbors is one of the simplest and most intuitive machine learning algorithms. It is a **non-parametric**, **instance-based** (lazy) learning method that makes predictions based solely on the **similarity** between a new data point and its closest neighbors in the training set.

Unlike parametric models (linear regression, SVMs) that learn a fixed set of parameters during training, KNN **stores the entire training set** and defers all computation to prediction time. It is used for both **classification** and **regression**.

---

## Core Idea

> "Tell me who your neighbors are, and I'll tell you who you are."

Given a new point **x**:
1. Compute the **distance** from **x** to every point in the training set
2. Select the **k nearest** training points
3. Aggregate their labels:
   - **Classification** → majority vote among the k neighbors
   - **Regression** → mean (or weighted mean) of the k neighbors' values

---

## Distance Metrics

The choice of distance metric is critical. KNN relies entirely on measuring "closeness."

### Euclidean Distance (L2) — Most Common

$$d(\mathbf{x}, \mathbf{x'}) = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}$$

- Default in scikit-learn
- Sensitive to feature scale → always standardize first
- Works well when features are continuous and comparable

### Manhattan Distance (L1)

$$d(\mathbf{x}, \mathbf{x'}) = \sum_{i=1}^{n} |x_i - x'_i|$$

- More robust to outliers than Euclidean
- Better when features are on different scales or data is sparse
- Preferred in high-dimensional spaces

### Minkowski Distance (Generalized)

$$d(\mathbf{x}, \mathbf{x'}) = \left(\sum_{i=1}^{n} |x_i - x'_i|^p\right)^{1/p}$$

- `p=1` → Manhattan
- `p=2` → Euclidean
- `p→∞` → Chebyshev (max absolute difference)

### Cosine Distance

$$d(\mathbf{x}, \mathbf{x'}) = 1 - \frac{\mathbf{x} \cdot \mathbf{x'}}{||\mathbf{x}|| \cdot ||\mathbf{x'}||}$$

- Measures angle between vectors, not magnitude
- Common in text/NLP applications

---

## Choosing K

The hyperparameter **k** controls model complexity:

| K Value | Behavior | Risk |
|---------|----------|------|
| **k = 1** | Decision boundary wraps tightly around each point | High variance (overfitting) |
| **Small k** | Complex, jagged boundaries | Sensitive to noise |
| **Large k** | Smooth, simple boundaries | May miss local structure (underfitting) |
| **k = n** | Always predicts the majority class | Maximum bias |

### Rules of Thumb
- Start with `k = √n` (square root of training size)
- Use **odd k** for binary classification (avoids ties)
- Use **cross-validation** to find optimal k
- The "elbow" in a k-vs-error plot often gives a good choice

---

## Weighted KNN

Standard KNN gives equal weight to all k neighbors. **Weighted KNN** gives closer neighbors more influence:

$$\hat{y} = \frac{\sum_{i=1}^{k} w_i \cdot y_i}{\sum_{i=1}^{k} w_i}$$

Common weight schemes:
- **Uniform**: $w_i = 1$ (standard KNN)
- **Inverse distance**: $w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}$
- **Gaussian**: $w_i = \exp\left(-\frac{d(\mathbf{x}, \mathbf{x}_i)^2}{2\sigma^2}\right)$

Weighted KNN often outperforms uniform, especially when the nearest neighbors are at varying distances.

---

## KNN for Classification

For a new point **x**, its predicted class is the **majority vote** among the k nearest neighbors:

$$\hat{y} = \arg\max_c \sum_{i=1}^{k} \mathbb{1}(y_i = c)$$

With distance weighting:

$$\hat{y} = \arg\max_c \sum_{i=1}^{k} w_i \cdot \mathbb{1}(y_i = c)$$

---

## KNN for Regression

For regression, the prediction is the **average** of the k neighbors' target values:

$$\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i$$

Or with distance weighting:

$$\hat{y} = \frac{\sum_{i=1}^{k} w_i \cdot y_i}{\sum_{i=1}^{k} w_i}$$

---

## The Curse of Dimensionality

KNN performance **degrades rapidly** in high dimensions:

1. **Volume grows exponentially** — in high-d space, most of the volume is near the surface of a hypercube/hypersphere
2. **Points become equidistant** — all pairwise distances converge, making "nearest" meaningless
3. **Data becomes sparse** — the amount of training data needed grows exponentially with dimensions

**Practical implication**: Always apply **dimensionality reduction** (PCA, feature selection) before KNN on high-dimensional data.

### Illustration

In a unit hypercube in $d$ dimensions, to capture 10% of the data you need a sub-cube with side length:

$$\ell = 0.1^{1/d}$$

| Dimensions | Side Length |
|------------|-------------|
| 1          | 0.10        |
| 5          | 0.63        |
| 10         | 0.79        |
| 50         | 0.955       |
| 100        | 0.977       |

In 100 dimensions, capturing 10% of data requires 97.7% of the range in each dimension — the concept of "local neighborhood" breaks down.

---

## Feature Scaling

KNN is **distance-based**, so features with large ranges dominate the distance computation. **Always standardize or normalize features before using KNN**.

| Method | Formula | When to Use |
|--------|---------|-------------|
| **StandardScaler** | $(x - \mu) / \sigma$ | Gaussian-like distributions |
| **MinMaxScaler** | $(x - x_{min}) / (x_{max} - x_{min})$ | Bounded features |

---

## Computational Complexity

| Operation | Brute Force | KD-Tree | Ball Tree |
|-----------|------------|---------|-----------|
| **Build** | O(1) — no training | O(n·d·log n) | O(n·d·log n) |
| **Query** | O(n·d) per point | O(d·log n) avg | O(d·log n) avg |
| **Memory** | O(n·d) | O(n·d) | O(n·d) |

- **Brute force**: Best for small datasets or high dimensions
- **KD-Tree**: Efficient for low dimensions (d < 20), degrades in high-d
- **Ball Tree**: Works better than KD-Tree in higher dimensions

Scikit-learn's `algorithm='auto'` selects the best method based on data characteristics.

---

## Strengths & Weaknesses

### Strengths
- Zero training time (lazy learner)
- Naturally handles multi-class problems
- Non-parametric — no assumptions about data distribution
- Simple to understand and implement
- Good for small to medium datasets with clear cluster structure

### Weaknesses
- Slow prediction on large datasets (must scan all training points)
- Memory-intensive (stores entire training set)
- Very sensitive to irrelevant features and curse of dimensionality
- Requires careful feature scaling
- No built-in feature importance

---

## What This Implementation Covers

1. **From-scratch KNN Classifier** — Euclidean, Manhattan, weighted voting
2. **From-scratch KNN Regressor** — Mean and weighted-mean prediction
3. **Scikit-learn comparison** — KNeighborsClassifier / KNeighborsRegressor
4. **Optimal k selection** — Cross-validated k sweep with error curves
5. **Distance metric comparison** — Euclidean vs Manhattan vs Minkowski
6. **Weighted vs uniform** — Performance impact
7. **Decision boundary visualization** — How k changes the boundary
8. **Curse of dimensionality demo** — Accuracy vs number of features
9. **Feature scaling importance** — Scaled vs unscaled comparison
10. **KNN vs other classifiers** — Benchmarking

---

## How to Run

```bash
cd 07_knn
python knn.py
```

All plots are saved to `plots/`.
