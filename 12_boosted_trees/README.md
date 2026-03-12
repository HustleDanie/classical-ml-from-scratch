# 12 — Boosted Trees (XGBoost & LightGBM)

## Overview

Gradient-boosted decision trees (GBDT) are the **dominant algorithm for structured / tabular data** competitions and production ML systems. While we built a gradient boosting machine from scratch in [05_gradient_boosting](../05_gradient_boosting), this module focuses on the two industry-standard **optimised frameworks**: **XGBoost** and **LightGBM**, exploring their unique algorithmic innovations, tuning strategies, and advanced features.

---

## 1 — Quick Recap: How Gradient Boosting Works

Given a loss function $L(y, \hat y)$ and a current model $F_m$, we fit a new tree $h_{m+1}$ to the **negative gradient** (pseudo-residuals):

$$r_i^{(m)} = -\frac{\partial L(y_i, \hat y_i)}{\partial \hat y_i}\bigg|_{\hat y_i = F_m(x_i)}$$

Then update:

$$F_{m+1}(x) = F_m(x) + \eta \cdot h_{m+1}(x)$$

where $\eta$ is the **learning rate** (shrinkage). The model improves by greedily adding trees that correct the current residual errors.

---

## 2 — XGBoost: Key Innovations

### 2.1 Regularised Objective

XGBoost adds **L1 and L2 penalties** on leaf weights directly to the objective:

$$\text{Obj} = \sum_{i=1}^{n} L(y_i, \hat y_i) + \sum_{k=1}^{K}\left[\gamma T_k + \frac{1}{2}\lambda \sum_{j=1}^{T_k} w_{kj}^2 + \alpha \sum_{j=1}^{T_k} |w_{kj}|\right]$$

| Symbol | Meaning |
|--------|---------|
| $T_k$ | Number of leaves in tree $k$ |
| $w_{kj}$ | Weight (prediction value) of leaf $j$ in tree $k$ |
| $\gamma$ | Minimum loss reduction to make a split (tree complexity penalty) |
| $\lambda$ | L2 regularisation on leaf weights |
| $\alpha$ | L1 regularisation on leaf weights |

### 2.2 Second-Order (Newton) Approximation

Instead of just the gradient, XGBoost uses a **second-order Taylor expansion** of the loss:

$$\text{Obj}^{(m)} \approx \sum_{i=1}^{n}\left[g_i f_m(x_i) + \frac{1}{2}h_i f_m(x_i)^2\right] + \Omega(f_m)$$

where:
- $g_i = \partial L / \partial \hat y_i$ — **gradient** (first derivative)
- $h_i = \partial^2 L / \partial \hat y_i^2$ — **hessian** (second derivative)

This yields the **optimal weight for a leaf** $j$:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

And the **gain for a candidate split**:

$$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

where $G_L = \sum_{i \in \text{left}} g_i$, etc.

### 2.3 Other XGBoost Features

| Feature | Description |
|---------|-------------|
| **Column sub-sampling** | Random subset of features per tree or per level |
| **Weighted quantile sketch** | Approximate split-finding for large data |
| **Sparsity-aware** | Handles missing values natively (learns optimal default direction) |
| **Shrinkage** | Learning rate $\eta$ to slow down learning |
| **Cache-aware access** | Block structure for out-of-core computation |

---

## 3 — LightGBM: Key Innovations

### 3.1 Leaf-Wise (Best-First) Growth

| XGBoost (default) | LightGBM |
|---|---|
| **Level-wise** growth — split all leaves at same depth | **Leaf-wise** growth — split the leaf with highest gain |
| More balanced trees, slower convergence | Deeper trees, faster convergence |
| Controlled by `max_depth` | Controlled by `num_leaves` |

Leaf-wise growth can over-fit more easily but typically converges faster and gives lower loss.

### 3.2 Gradient-Based One-Side Sampling (GOSS)

Instead of using all data, GOSS:
1. **Keeps** all instances with large gradients (top $a \times 100\%$)
2. **Randomly samples** from instances with small gradients (keep $b \times 100\%$)
3. **Up-weights** the sampled small-gradient instances by $\frac{1-a}{b}$

This approximates the information gain using **fewer samples**, speeding up training while retaining accuracy.

### 3.3 Exclusive Feature Bundling (EFB)

For **sparse, high-dimensional** data, many features are mutually exclusive (never non-zero simultaneously). LightGBM:
1. Detects near-mutually-exclusive feature groups
2. **Bundles** them into a single feature using offset encoding

This reduces the effective number of features without information loss.

### 3.4 Histogram-Based Splitting

Instead of sorting feature values (expensive), LightGBM:
1. **Buckets** continuous values into discrete bins (e.g., 255 bins)
2. Builds histograms of gradients/hessians per bin
3. Finds the best split from histograms

| Metric | Exact (sorted) | Histogram |
|--------|----------------|-----------|
| Time complexity | $O(n \cdot d)$ | $O(\text{bins} \cdot d)$ |
| Memory | High (sorted indices) | Low (integers) |
| Accuracy | Exact | Slightly approximate |

> XGBoost also supports `tree_method='hist'` for histogram-based splitting.

---

## 4 — XGBoost vs LightGBM — Side by Side

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| Tree growth | Level-wise (default) | Leaf-wise |
| Split finding | Exact + approximate + hist | Histogram only |
| Missing values | Native (learned direction) | Native (learned direction) |
| Categorical features | One-hot encoding required | Native categorical support |
| Sampling | Column sub-sampling | GOSS + column sub-sampling |
| Feature bundling | None | EFB |
| Speed | Fast | Typically faster |
| Memory | Moderate | Lower |
| Over-fitting control | `max_depth`, regularisation | `num_leaves`, regularisation |

---

## 5 — Key Hyperparameters

### 5.1 Shared Parameters

| Parameter | XGBoost name | LightGBM name | Typical range |
|-----------|-------------|---------------|---------------|
| Learning rate | `eta` / `learning_rate` | `learning_rate` | 0.01 – 0.3 |
| Number of trees | `n_estimators` | `n_estimators` | 100 – 10000 |
| Max depth | `max_depth` | `max_depth` | 3 – 10 |
| Min child weight | `min_child_weight` | `min_child_samples` | 1 – 100 |
| Sub-sample ratio | `subsample` | `bagging_fraction` | 0.5 – 1.0 |
| Feature fraction | `colsample_bytree` | `feature_fraction` | 0.5 – 1.0 |
| L1 regularisation | `alpha` (`reg_alpha`) | `lambda_l1` | 0 – 10 |
| L2 regularisation | `lambda` (`reg_lambda`) | `lambda_l2` | 0 – 10 |
| Min split gain | `gamma` | `min_gain_to_split` | 0 – 5 |

### 5.2 LightGBM-Specific

| Parameter | Description | Typical range |
|-----------|-------------|---------------|
| `num_leaves` | Max number of leaves per tree (controls complexity instead of depth) | 20 – 300 |
| `max_bin` | Number of histogram bins | 63 – 512 |
| `top_rate` ($a$) | GOSS: fraction of large-gradient instances | 0.1 – 0.3 |
| `other_rate` ($b$) | GOSS: sampling fraction among small-gradient instances | 0.05 – 0.2 |

---

## 6 — Tuning Strategy

A practical order for tuning:

1. **Fix `learning_rate` = 0.1**, find a good `n_estimators` via early stopping
2. **Tune tree structure**: `max_depth` / `num_leaves`, `min_child_weight` / `min_child_samples`
3. **Tune sampling**: `subsample`, `colsample_bytree`
4. **Tune regularisation**: `reg_alpha`, `reg_lambda`, `gamma`
5. **Lower learning rate** (e.g., 0.01), increase `n_estimators`, re-run early stopping

---

## 7 — Feature Importance

Both frameworks offer multiple importance types:

| Type | Description |
|------|-------------|
| **weight / split** | Number of times a feature is used to split |
| **gain** | Average gain when the feature is used |
| **cover** | Average number of samples affected by splits using the feature |
| **SHAP values** | Game-theoretic feature attributions (additive, consistent) |

> **Gain-based importance can be biased** toward high-cardinality features. SHAP values are more reliable for interpretation.

---

## 8 — Early Stopping

Both frameworks support early stopping: training halts when the validation metric hasn't improved for `early_stopping_rounds` iterations:

```text
round 1:  val_loss = 0.65
round 2:  val_loss = 0.51  ← improved
...
round 150: val_loss = 0.22 ← best
round 151: val_loss = 0.22 ← no improvement (1/50)
...
round 200: val_loss = 0.23 ← no improvement (50/50) → STOP
```

This prevents overfitting and avoids manually choosing `n_estimators`.

---

## 9 — What We Implement

| # | Experiment | Purpose |
|---|-----------|---------|
| 1 | XGBoost Classifier | Breast Cancer binary classification |
| 2 | LightGBM Classifier | Same dataset for comparison |
| 3 | Early Stopping & Learning Curves | Loss vs boosting round |
| 4 | XGBoost Regressor | California Housing regression |
| 5 | LightGBM Regressor | Same dataset for comparison |
| 6 | Feature Importance (gain, split, SHAP) | Which features matter |
| 7 | Hyperparameter sweep — `max_depth` / `num_leaves` | Effect on performance |
| 8 | Learning Rate vs N Estimators | Trade-off analysis |
| 9 | XGBoost vs LightGBM vs Sklearn GBM | Speed and accuracy comparison |
| 10 | Regularisation sweep | Effect of L1/L2 on overfitting |

---

## 10 — Run

```bash
cd 12_boosted_trees
python boosted_trees.py
```

All plots will be saved to `plots/`.
