# 03 — Decision Trees

## What Is a Decision Tree?

A Decision Tree is a **non-parametric supervised learning** algorithm that makes predictions by learning a hierarchy of **if-then-else rules** from the data. It recursively splits the feature space into regions, creating a tree-like structure where:

- **Internal nodes** = feature tests (e.g., "Is age > 30?")
- **Branches** = outcomes of the test
- **Leaf nodes** = final predictions (class label or numeric value)

It answers: *"What sequence of yes/no questions best separates the data?"*

Decision trees are one of the most **interpretable** models — you can literally read the logic.

---

## How It Works — Recursive Splitting

### The Algorithm (CART — Classification and Regression Trees)

```
function BuildTree(data, depth):
    if stopping_condition_met:
        return Leaf(prediction)
    
    best_feature, best_threshold = find_best_split(data)
    left_data  = data where feature <= threshold
    right_data = data where feature > threshold
    
    left_child  = BuildTree(left_data, depth + 1)
    right_child = BuildTree(right_data, depth + 1)
    
    return Node(best_feature, best_threshold, left_child, right_child)
```

At each node, we ask: **"Which feature and threshold split the data most purely?"**

---

## Splitting Criteria

### For Classification

#### Gini Impurity (default in scikit-learn)

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

- $p_k$ = proportion of class $k$ at node $t$
- Gini = 0 → perfectly pure node (all one class)
- Gini = 0.5 → maximum impurity for binary (50/50 split)

#### Entropy (Information Gain)

$$\text{Entropy}(t) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

**Information Gain** from a split:

$$\text{IG} = \text{Entropy}(\text{parent}) - \sum_{j} \frac{n_j}{n} \text{Entropy}(\text{child}_j)$$

We choose the split that **maximizes** Information Gain (= maximally reduces entropy).

#### Gini vs Entropy

| Property | Gini | Entropy |
|----------|------|---------|
| Range (binary) | [0, 0.5] | [0, 1.0] |
| Computation | Faster (no log) | Slightly slower |
| Behavior | Tends to isolate most frequent class | More balanced splits |
| In practice | Very similar results | Very similar results |

### For Regression

#### Mean Squared Error (MSE / Variance Reduction)

$$\text{MSE}(t) = \frac{1}{n} \sum_{i \in t} (y_i - \bar{y}_t)^2$$

Split to **minimize** the weighted MSE of child nodes. The prediction at a leaf is the **mean** of the target values.

---

## Stopping Criteria (When to Stop Splitting)

| Parameter | What It Controls |
|-----------|-----------------|
| `max_depth` | Maximum tree depth |
| `min_samples_split` | Minimum samples needed to split a node |
| `min_samples_leaf` | Minimum samples in a leaf node |
| `max_features` | Number of features considered per split |
| `max_leaf_nodes` | Maximum total leaf nodes |

Without stopping criteria, a tree will grow until every leaf is pure → **overfitting**.

---

## Pruning

### Pre-pruning (Early Stopping)
Stop growing the tree early using the stopping criteria above.

### Post-pruning (Cost-Complexity Pruning)
Grow a full tree, then remove branches that don't improve performance significantly.

The **cost-complexity** parameter $\alpha$ (called `ccp_alpha` in sklearn):

$$R_\alpha(T) = R(T) + \alpha \cdot |T|$$

- $R(T)$ = training error
- $|T|$ = number of leaf nodes
- Higher $\alpha$ → simpler tree (more pruning)

---

## Decision Trees for Regression

Instead of class labels, leaves predict a **numeric value** (mean of samples at that leaf).

Split criterion: minimize MSE (variance) in child nodes.

$$\text{Prediction at leaf } t = \bar{y}_t = \frac{1}{n_t} \sum_{i \in t} y_i$$

---

## Advantages & Disadvantages

| Advantages | Disadvantages |
|-----------|---------------|
| Highly interpretable | Prone to overfitting |
| No feature scaling needed | Unstable (small data changes → different tree) |
| Handles both numeric & categorical | Biased toward features with many levels |
| Captures nonlinear relationships | Can create overly complex trees |
| Fast training & inference | Greedy algorithm (not globally optimal) |

---

## What This Implementation Covers

1. **From-scratch Decision Tree Classifier** — Recursive splitting with Gini & Entropy
2. **From-scratch Decision Tree Regressor** — MSE-based splitting
3. **Scikit-learn comparison** — Classification & Regression
4. **Tree visualization** — Text-based (scratch) + graphical (sklearn)
5. **Gini vs Entropy comparison** — Side-by-side
6. **Overfitting demonstration** — Unrestricted vs pruned depth
7. **Cost-complexity pruning** — Finding optimal alpha
8. **Feature importance** — Which features matter most
9. **Decision boundary visualization** — 2D
10. **Regression tree** — Predicting continuous values

---

## How to Run

```bash
cd 03_decision_trees
python decision_trees.py
```

All plots are saved to `plots/`.

---

## Key Takeaways

- Decision trees split data recursively using **Gini impurity** or **entropy**
- They are **greedy** — each split is locally optimal, not globally
- Without pruning, they **overfit** heavily (memorize training data)
- **max_depth** is the single most important hyperparameter
- Trees need **no feature scaling** — they only care about ordering
- They are the **building block** for Random Forests and Gradient Boosting
- Use **cost-complexity pruning** (`ccp_alpha`) for principled tree simplification
