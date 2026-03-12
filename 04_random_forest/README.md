# 04 — Random Forest

## What Is a Random Forest?

A Random Forest is an **ensemble learning** method that builds many decision trees and combines their predictions. Instead of relying on a single tree (which is prone to overfitting), it trains a "forest" of diverse trees and aggregates them.

- **Classification** → majority vote across all trees
- **Regression** → average prediction across all trees

It answers: *"If I train many diverse trees, can their collective wisdom beat any single tree?"* — Yes, almost always.

---

## The Core Idea — Ensemble Learning

### Why Ensembles Work

A single decision tree is a **high-variance** model — small changes in data produce very different trees. Ensembles reduce variance by averaging many models:

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right) \approx \frac{\sigma^2}{B} \quad \text{(if trees are independent)}$$

The more **uncorrelated** the trees, the more the variance drops. Random Forest achieves this through two sources of randomness.

---

## Two Sources of Randomness

### 1. Bagging (Bootstrap Aggregating)

Each tree is trained on a **bootstrap sample** — a random sample of size $n$ drawn **with replacement** from the training set.

- Each bootstrap sample uses ~63.2% of unique training examples
- The remaining ~36.8% are **Out-of-Bag (OOB)** samples — free validation data!

$$P(\text{sample not picked in } n \text{ draws}) = \left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368$$

### 2. Feature Randomness (Random Subspace)

At each split, only a **random subset** of features is considered:

| Task | Default `max_features` |
|------|----------------------|
| Classification | $\sqrt{p}$ features |
| Regression | $p / 3$ features |

This forces trees to use different features → more diversity → less correlation.

---

## The Algorithm

```
function RandomForest(X, y, n_trees, max_features):
    forest = []
    for b = 1 to n_trees:
        X_boot, y_boot = bootstrap_sample(X, y)
        tree_b = build_decision_tree(X_boot, y_boot, max_features)
        forest.append(tree_b)
    return forest

function predict(X):
    predictions = [tree.predict(X) for tree in forest]
    if classification:
        return majority_vote(predictions)
    if regression:
        return average(predictions)
```

---

## Out-of-Bag (OOB) Score

Since each tree only sees ~63.2% of the data, the remaining 36.8% (OOB samples) are a **built-in validation set**.

For each training sample:
1. Find all trees that did **not** use it in training
2. Get their predictions for this sample
3. Aggregate → OOB prediction

This gives a **free estimate of generalization error** without needing a separate validation set!

---

## Key Hyperparameters

| Parameter | What It Controls | Typical Values |
|-----------|-----------------|----------------|
| `n_estimators` | Number of trees | 100–1000 |
| `max_depth` | Max depth per tree | None, 10–30 |
| `max_features` | Features per split | `'sqrt'`, `'log2'`, fraction |
| `min_samples_split` | Min samples to split | 2–10 |
| `min_samples_leaf` | Min samples per leaf | 1–5 |
| `bootstrap` | Use bootstrap sampling | True (default) |
| `oob_score` | Compute OOB error | True/False |

### The Most Important One: `n_estimators`

More trees = better (with diminishing returns). Unlike depth, more trees **never overfit** — they only reduce variance.

---

## Feature Importance

Random Forest provides two types of feature importance:

### 1. Impurity-Based (Default)
Sum of weighted impurity reductions across all trees for each feature. Fast but biased toward high-cardinality features.

### 2. Permutation Importance
Shuffle each feature column and measure the drop in score. More reliable but slower.

---

## Random Forest vs Single Decision Tree

| Property | Decision Tree | Random Forest |
|----------|--------------|---------------|
| Bias | Low | Low |
| Variance | **High** | **Low** (averaged out) |
| Overfitting | Very prone | Resistant |
| Interpretability | High (readable) | Lower (many trees) |
| Training speed | Fast | Slower (many trees) |
| Prediction speed | Fast | Slower (aggregate) |

---

## Advantages & Disadvantages

| Advantages | Disadvantages |
|-----------|---------------|
| Excellent out-of-the-box performance | Less interpretable than single tree |
| Resistant to overfitting | Slower than single tree |
| Handles missing values & outliers well | Memory-intensive (many trees) |
| Built-in OOB validation | Can be overkill for simple problems |
| Parallelizable (`n_jobs=-1`) | Feature importance can be biased |
| No feature scaling needed | |

---

## What This Implementation Covers

1. **From-scratch Random Forest Classifier** — Bootstrap + feature subsampling + majority vote
2. **From-scratch Random Forest Regressor** — Bootstrap + feature subsampling + averaging
3. **Scikit-learn comparison** — Classification & Regression
4. **OOB score demonstration** — Free validation without train/test split
5. **Number of trees vs performance** — Diminishing returns curve
6. **Feature importance** — Impurity-based + permutation importance
7. **Single tree vs Forest** — Direct comparison
8. **max_features impact** — How feature randomness affects performance
9. **Decision boundary comparison** — Single tree vs Random Forest
10. **Variance reduction visualization** — Showing how ensemble reduces variance

---

## How to Run

```bash
cd 04_random_forest
python random_forest.py
```

All plots are saved to `plots/`.

---

## Key Takeaways

- Random Forest = **Bagging** + **Feature Randomness** + Decision Trees
- Two randomness sources make trees **diverse and uncorrelated**
- More trees **never overfit** — only reduce variance (with diminishing returns)
- **OOB score** gives free validation; no separate validation set needed
- `max_features='sqrt'` for classification, `max_features=n/3` for regression
- Almost always beats a single decision tree with minimal tuning
- The **go-to strong baseline** for any tabular ML task
