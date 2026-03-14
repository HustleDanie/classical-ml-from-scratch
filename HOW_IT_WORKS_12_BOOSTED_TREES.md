# How It Works: Boosted Trees (XGBoost & LightGBM)

> XGBoost and LightGBM are **production-grade gradient boosting frameworks** built on the same idea as Gradient Boosting (model 05) but with engineering optimizations that make them dramatically faster, more scalable, and more feature-rich. They are the dominant models in structured/tabular data competitions and real-world ML.

---

## The One-Sentence Idea

XGBoost and LightGBM build hundreds of shallow decision trees **one after another**, where each new tree fixes the mistakes of all the trees that came before it -- the same core idea as Gradient Boosting, but with clever tricks for speed, memory, and accuracy.

---

## Quick Recap: Gradient Boosting Foundation

If you've read HOW_IT_WORKS_05_GRADIENT_BOOSTING.md, you know the core loop:

```
1. Start with a simple prediction (e.g., the mean)
2. Compute residuals (errors)
3. Fit a small tree to predict those residuals
4. Add that tree's predictions (scaled by learning rate) to the running total
5. Repeat steps 2-4 for N rounds
```

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

XGBoost and LightGBM follow this same loop but optimize **how trees are built** and **how the algorithm scales**.

---

## What XGBoost Adds Over Vanilla Gradient Boosting

### 1. Regularized Objective Function

Vanilla gradient boosting minimizes loss only. XGBoost adds a **regularization term** to control tree complexity:

$$\text{Obj} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)$$

where:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $L$ = loss function (e.g., log loss, MSE)
- $T$ = number of leaves in the tree
- $w_j$ = weight (prediction value) of leaf $j$
- $\gamma$ = penalty per leaf (encourages fewer leaves)
- $\lambda$ = L2 regularization on leaf weights (encourages smaller predictions)

```
Vanilla Gradient Boosting:         XGBoost:
  Minimize: Loss only               Minimize: Loss + tree complexity penalty
  
  Loss = sum of errors               Loss = sum of errors
                                      + gamma * (number of leaves)
                                      + lambda * (sum of squared leaf weights)
  
  No penalty for complex trees.      Complex trees are penalized.
  Overfits easily.                    Built-in regularization.
```

### 2. Second-Order Approximation (Newton's Method)

Vanilla gradient boosting uses only the **first derivative** (gradient) of the loss. XGBoost uses both the **first and second derivatives**:

$$\text{Obj} \approx \sum_{i=1}^{n} \left[ g_i f_k(\mathbf{x}_i) + \frac{1}{2} h_i f_k^2(\mathbf{x}_i) \right] + \Omega(f_k)$$

where:
- $g_i = \frac{\partial L(y_i, \hat{y}_i)}{\partial \hat{y}_i}$ (gradient -- first derivative)
- $h_i = \frac{\partial^2 L(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}$ (hessian -- second derivative)

**Why is this better?** The second derivative tells you the **curvature** of the loss surface, allowing more precise steps. It's like the difference between walking downhill with your eyes closed (gradient only) vs. seeing the terrain ahead (gradient + curvature).

### 3. Optimal Leaf Weights

With both $g$ and $h$, XGBoost computes the **exact optimal weight** for each leaf:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

And the **optimal split quality** (gain) is:

$$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I_P} g_i)^2}{\sum_{i \in I_P} h_i + \lambda} \right] - \gamma$$

```
If Gain > 0 -> split is beneficial (reduces loss enough to justify new leaf)
If Gain <= 0 -> don't split (improvement doesn't justify added complexity)
                This is automatic pruning!
```

### 4. Column and Row Subsampling

Like Random Forest, XGBoost can randomly sample:
- **colsample_bytree**: fraction of features per tree
- **colsample_bylevel**: fraction of features per depth level
- **subsample**: fraction of training rows per tree

This reduces overfitting and speeds up training.

---

## How a Single Tree is Built in XGBoost

```
Building tree #47 (of 300):

Step 1: Compute g_i and h_i for each training point
  Using current predictions from trees 1-46:
  g_i = gradient of loss at current prediction
  h_i = hessian (second derivative) of loss

Step 2: Find the best split for the ROOT node
  For each feature (or a random subset):
    Sort data by that feature's values
    Scan through all possible split points
    For each split, compute Gain using g and h sums
  Choose the split with the highest Gain
  
Step 3: Split the data, repeat for child nodes
  Left child: points where feature <= threshold
  Right child: points where feature > threshold
  
Step 4: Continue until:
  - max_depth reached, OR
  - Gain <= 0 (regularization says stop), OR
  - min_child_weight not met (sum of h in a leaf is too small)

Step 5: Set leaf weights using the optimal formula:
  w* = -sum(g_i) / (sum(h_i) + lambda)

Result: A shallow tree (depth 3-8 typically) with optimized leaf values
```

---

## What LightGBM Does Differently

LightGBM was developed by Microsoft to handle **very large datasets faster** than XGBoost. The key innovations:

### 1. Histogram-Based Splitting (Also in XGBoost 1.0+)

Instead of trying every possible split point (exact greedy), bin continuous features into ~256 buckets:

```
Exact split (XGBoost original):       Histogram split (LightGBM):
  
  Feature: [1.2, 1.5, 2.3, 2.7,       Feature binned into 4 buckets:
            3.1, 4.5, 5.0, 5.8]         Bin 0: [1.2, 1.5]     -> try split after bin 0
                                          Bin 1: [2.3, 2.7]     -> try split after bin 1
  Try split at 1.2, 1.5, 2.3, 2.7,      Bin 2: [3.1]           -> try split after bin 2
  3.1, 4.5, 5.0, 5.8                     Bin 3: [4.5, 5.0, 5.8] -> try split after bin 3
  = 7 split candidates                  = 3 split candidates (much fewer!)

  Speed: O(N * features)                Speed: O(bins * features)
  For N=1M: slow                         For bins=256: fast regardless of N
```

### 2. Leaf-Wise Growth (vs. Level-Wise)

```
Level-wise (XGBoost default):         Leaf-wise (LightGBM default):

Depth 0:    [Root]                       [Root]
           /      \                     /      \
Depth 1: [L]      [R]                 [L]      [R]
        / \      / \                          / \
Depth 2:[LL][LR][RL][RR]                   [RL] [RR]
                                                / \
All leaves at each depth                     [RRL][RRR]
are split simultaneously.

Splits the best leaf FIRST,
even if it makes the tree unbalanced.

Level-wise: fair but slow.            Leaf-wise: faster convergence but
  All leaves grown equally.             can overfit (needs max_leaves limit).
  Wastes time on bad splits.            Only does the most beneficial splits.
```

### 3. GOSS (Gradient-Based One-Side Sampling)

Instead of using all training samples, LightGBM:
1. **Keeps all points with large gradients** (hard-to-predict, most informative)
2. **Randomly samples a fraction of small-gradient points** (easy, less useful)

```
All training points:
  | gradient |  -> Large gradient: model is wrong, informative point
  | gradient |  -> Small gradient: model is right, less useful

  Keep 100% of top-gradient points (say top 20%)
  Randomly sample 10% of remaining points
  
  Training set effectively reduced while maintaining accuracy!
```

### 4. Exclusive Feature Bundling (EFB)

For sparse features (many zeros), LightGBM bundles mutually exclusive features together:

```
Feature A:  [0, 0, 3, 0, 0, 5, 0]
Feature B:  [1, 2, 0, 4, 0, 0, 0]

These are mutually exclusive (when A != 0, B == 0 and vice versa).
Bundle into one feature: [1, 2, 3, 4, 0, 5, 0]

Result: Fewer features to scan -> faster splits
```

---

## XGBoost vs. LightGBM: Head-to-Head

| Feature | XGBoost | LightGBM |
|---------|---------|----------|
| Tree growth | Level-wise (default) | Leaf-wise (default) |
| Split algorithm | Histogram (v1.0+) or exact | Histogram always |
| Speed | Fast | Usually 2-5x faster |
| Memory | Higher | Lower (histogram-based) |
| Categorical features | Must encode (one-hot/label) | Native support |
| Missing values | Built-in handling (learns best direction) | Built-in handling |
| GPU support | Yes | Yes |
| Overfitting risk | Lower (level-wise is conservative) | Higher (leaf-wise needs tuning) |
| Sparse data | Excellent | Excellent (EFB helps) |
| Large datasets | Good | Better (GOSS + EFB) |

---

## What Happens Behind the Scenes: `.fit()` and `.predict()`

### XGBoost

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=300,       # number of boosting rounds
    max_depth=6,            # max tree depth
    learning_rate=0.1,      # shrinkage (eta)
    subsample=0.8,          # row sampling
    colsample_bytree=0.8,   # feature sampling per tree
    reg_alpha=0.0,          # L1 regularization (alpha)
    reg_lambda=1.0,         # L2 regularization (lambda)
    min_child_weight=1,     # minimum sum of hessians in a leaf
    gamma=0.0,              # minimum gain to split (gamma)
    eval_metric='logloss',
    early_stopping_rounds=50
)

# .fit() does this:
# 1. Convert data to DMatrix (optimized internal format)
# 2. Initialize predictions (base score, e.g., log-odds of mean)
# 3. For each boosting round (1 to 300):
#    a. Compute g_i and h_i for every training point
#    b. Build a tree using histogram splits
#       - For each node, find best split using Gain formula
#       - Stop when max_depth reached or Gain <= 0
#    c. Compute optimal leaf weights: w* = -sum(g)/sum(h+lambda)
#    d. Update predictions: pred += learning_rate * tree_prediction
#    e. If eval_set provided, check validation loss
#       - If no improvement for early_stopping_rounds, STOP
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# .predict() does this:
# 1. Pass each sample through ALL trees (say 200 if early stopped at 200)
# 2. Sum all leaf values: raw_score = sum of all tree predictions
# 3. Apply sigmoid (classification) or identity (regression):
#    probability = 1 / (1 + exp(-raw_score))
# 4. Apply threshold (default 0.5) for class label
preds = model.predict(X_test)
probs = model.predict_proba(X_test)
```

### LightGBM

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=300,         # number of boosting rounds
    num_leaves=31,            # max leaves per tree (KEY parameter)
    max_depth=-1,             # no depth limit (controlled by num_leaves)
    learning_rate=0.1,        # shrinkage
    subsample=0.8,            # row sampling
    colsample_bytree=0.8,     # feature sampling
    reg_alpha=0.0,            # L1 regularization
    reg_lambda=0.0,           # L2 regularization
    min_child_samples=20,     # minimum points in a leaf
)

# .fit() does this:
# 1. Bin continuous features into histograms (max_bin=255 by default)
# 2. Apply GOSS (if enabled): keep large-gradient points, sample small ones
# 3. Apply EFB: bundle sparse features
# 4. For each boosting round:
#    a. Compute g_i and h_i
#    b. Build tree LEAF-WISE:
#       - Find the leaf with highest potential gain
#       - Split it (not all leaves at a depth, just the best one)
#       - Repeat until num_leaves reached
#    c. Update predictions
#    d. Check early stopping if eval_set provided
model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

preds = model.predict(X_test)
probs = model.predict_proba(X_test)
```

---

## The Full Training Pipeline

```
XGBoost Training Pipeline:

Input: 100K samples, 50 features, binary classification

Step 1: Data preparation
  Convert to DMatrix (compressed sparse row matrix)
  Handle missing values (mark as "missing" direction to be learned)

Step 2: Initialize
  base_score = log(pos_count / neg_count)  (log-odds of positive class)
  All predictions start at base_score

Step 3: Round 1 of 300
  a. Compute gradients:
     g_i = predicted_prob_i - y_i              (first derivative of log loss)
     h_i = predicted_prob_i * (1 - predicted_prob_i)  (second derivative)
  
  b. Build tree (max_depth=6):
     Root: scan all 50 features, 256 bins each
           -> best split: Feature 12 <= bin 89, Gain = 245.3
     Depth 1 Left: scan features -> Feature 3 <= bin 42, Gain = 102.1
     Depth 1 Right: scan features -> Feature 27 <= bin 156, Gain = 88.7
     ... continue to depth 6
     
  c. Compute leaf weights:
     Leaf 0 (350 samples): w* = -sum(g)/sum(h + 1.0) = 0.23
     Leaf 1 (280 samples): w* = -0.15
     ... (up to 64 leaves for depth 6)
     
  d. Update predictions:
     pred_i += 0.1 * tree_1_prediction(x_i)   (learning_rate = 0.1)

Step 4: Round 2 of 300
  New gradients (residuals are smaller now)
  Build next tree on updated residuals
  ...

Step 5: Early stopping
  Validation logloss: [0.68, 0.62, 0.55, 0.51, ..., 0.32, 0.32, 0.32]
  No improvement for 50 rounds at round 210 -> STOP
  Best iteration: round 160

Output: 160 trees, each with up to 64 leaves
  Total parameters: ~160 * 64 = ~10K leaf weights
```

---

## Key Hyperparameters

### Learning Parameters

| Parameter | XGBoost Name | LightGBM Name | What It Does | Typical Range |
|-----------|-------------|---------------|--------------|---------------|
| Boosting rounds | n_estimators | n_estimators | Number of trees | 100-3000 |
| Learning rate | learning_rate (eta) | learning_rate | Shrinkage per tree | 0.01-0.3 |
| Tree depth | max_depth | max_depth | How deep each tree grows | 3-10 |
| Max leaves | max_leaves | num_leaves | Max leaves per tree | 15-255 |
| Min samples per leaf | min_child_weight | min_child_samples | Prevents small leaves | 1-100 |

### Regularization Parameters

| Parameter | XGBoost Name | LightGBM Name | What It Does |
|-----------|-------------|---------------|--------------|
| L1 on weights | reg_alpha | reg_alpha | Sparsity in leaf weights |
| L2 on weights | reg_lambda | reg_lambda | Prevents large leaf weights |
| Min split gain | gamma | min_split_gain | Minimum gain to make a split |
| Row sampling | subsample | subsample (bagging_fraction) | Fraction of rows per tree |
| Column sampling | colsample_bytree | colsample_bytree (feature_fraction) | Fraction of features per tree |

### The Two Most Important Parameters

```
XGBoost: max_depth controls complexity
  max_depth=3  -> simple trees (8 leaves max), underfits sometimes
  max_depth=6  -> moderate (64 leaves max), good default
  max_depth=10 -> complex trees (1024 leaves max), may overfit

LightGBM: num_leaves controls complexity
  num_leaves=15  -> simple, fast, less overfit
  num_leaves=31  -> moderate (default)
  num_leaves=127 -> complex, can overfit

  Rule of thumb: num_leaves should be <= 2^max_depth
  e.g., max_depth=6 -> num_leaves <= 64
```

---

## Feature Importance: Three Ways

### 1. Split-Based (weight)

How many times a feature was used for splitting across all trees.

### 2. Gain-Based (gain)

Total gain (improvement in objective) contributed by splits on each feature.

### 3. SHAP Values

Game-theoretic method that explains each prediction by attributing it to individual features.

```python
# XGBoost built-in
model.feature_importances_  # gain-based by default

# Plot importance
xgb.plot_importance(model, importance_type='gain', max_num_features=10)

# SHAP (most accurate)
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

```
Gain-based importance:              SHAP summary plot:
Feature 12:  |========== 35%        Feature 12: each dot is a sample
Feature 3:   |======= 22%                      colored by feature value
Feature 27:  |===== 15%                         position = SHAP value (impact)
Feature 1:   |==== 12%              
Feature 8:   |=== 8%                 Shows direction AND magnitude
Feature 19:  |== 5%                  of each feature's effect
Feature 42:  |= 3%                   on individual predictions
```

---

## Handling Missing Values

Both frameworks handle missing values **natively** -- no imputation needed:

```
XGBoost approach:
  When building a tree, for each split:
    Try sending missing values LEFT -> compute gain
    Try sending missing values RIGHT -> compute gain
    Choose the direction that gives higher gain
    
  This "default direction" is learned per split.
  
  At prediction time: if a value is missing,
  follow the learned default direction.

LightGBM approach:
  Similarly learns the best direction for missing values.
  Additionally handles zeros efficiently (sparse optimization).
```

```python
# Both handle NaN naturally
import numpy as np
X_train[5, 3] = np.nan  # missing value
model.fit(X_train, y_train)  # works fine, learns default direction
```

---

## Handling Categorical Features (LightGBM)

LightGBM can handle categorical features without one-hot encoding:

```python
model = lgb.LGBMClassifier()
model.fit(X_train, y_train, categorical_feature=[0, 3, 7])
# Features at indices 0, 3, 7 are treated as categorical

# Or with pandas:
df['city'] = df['city'].astype('category')
model.fit(df[features], df['target'])
# LightGBM auto-detects category dtype
```

LightGBM uses an **optimal split algorithm for categories** that partitions categorical values into two groups (not one-vs-rest like one-hot encoding would imply).

---

## Early Stopping: When to Stop Adding Trees

```python
model = xgb.XGBClassifier(
    n_estimators=3000,          # set high
    early_stopping_rounds=50    # stop if no improvement for 50 rounds
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # validation set to monitor
    verbose=False
)

# model.best_iteration tells you how many trees were actually used
print(f"Stopped at round {model.best_iteration}")
```

```
Validation Loss:
  |
  | *
  |   *
  |     *
  |       *
  |         *  *
  |              *  *  *
  |                       *  *  *  (best here, round 180)
  |                                 *  *  *  *  *  (no improvement)
  |                                                  STOP at round 230
  +--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--
     0  20  40  60  80 100 120 140 160 180 200 220 230
                       Boosting Round
```

---

## Learning Rate vs. Number of Trees Trade-Off

```
High learning rate (0.3) + few trees (100):
  Each tree makes BIG corrections
  Fast training, but may miss fine details
  Risk: underfitting

Low learning rate (0.01) + many trees (3000):
  Each tree makes SMALL corrections
  Slow training, but more precise
  Risk: overfitting if no early stopping

Sweet spot: learning_rate=0.05-0.1, n_estimators=500-1500 with early_stopping
```

$$ \text{trees needed} \approx \frac{1}{\text{learning\_rate}} \times c $$

Lower learning rate -> need more trees -> better generalization (with early stopping) -> slower training.

---

## Practical Tuning Strategy

```
Step 1: Set reasonable defaults
  learning_rate = 0.1
  n_estimators = 1000 (with early stopping)
  max_depth = 6 (XGBoost) or num_leaves = 31 (LightGBM)

Step 2: Find the right number of trees
  Use early_stopping_rounds = 50
  Train and let it stop naturally

Step 3: Tune tree complexity
  Grid search over:
    max_depth: [4, 6, 8]
    num_leaves: [15, 31, 63, 127]
    min_child_weight / min_child_samples: [1, 5, 20]

Step 4: Tune regularization
  subsample: [0.6, 0.8, 1.0]
  colsample_bytree: [0.6, 0.8, 1.0]
  reg_alpha: [0, 0.1, 1.0]
  reg_lambda: [0, 1.0, 5.0]
  gamma / min_split_gain: [0, 0.1, 0.5]

Step 5: Lower learning rate, increase trees
  learning_rate = 0.01 or 0.05
  n_estimators = 3000+ with early stopping
  
Step 6: Final model with best params
```

---

## Sklearn GBM vs. XGBoost vs. LightGBM

```
sklearn GradientBoosting:
  + Simple API, no extra install
  + Good for learning
  - Slow (no histogram binning before v1.0)
  - No built-in early stopping via eval_set
  - Limited regularization options

XGBoost:
  + Regularized objective
  + Second-order optimization
  + Handles missing values
  + Histogram binning (v1.0+)
  + GPU support
  + Very mature and stable
  - Slightly slower than LightGBM

LightGBM:
  + Fastest training (GOSS + EFB + leaf-wise)
  + Lowest memory usage
  + Native categorical feature support
  + GPU support
  + Handles very large datasets
  - Leaf-wise growth can overfit if not tuned
  - num_leaves tuning is less intuitive than max_depth
```

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| No early stopping | Trains all n_estimators, may overfit | Always use early_stopping with eval_set |
| Learning rate too high | Fast but inaccurate, jumpy convergence | Use 0.01-0.1 with more trees |
| max_depth too high (XGBoost) | Trees overfit to noise | Use 3-8, tune with CV |
| num_leaves too high (LightGBM) | Leaf-wise overfitting | Keep num_leaves <= 2^max_depth |
| Not tuning regularization | Model overfits training data | Tune subsample, colsample, reg_alpha, reg_lambda |
| One-hot encoding with LightGBM | Creates sparse, high-dimensional input | Use native categorical support |
| Ignoring feature importance | Model is a black box | Use SHAP for interpretability |
| Not scaling labels (regression) | Large targets cause numerical issues | Scale target, or use robust loss (e.g., Huber) |
| Too few features sampled | Trees are too random, slow convergence | colsample_bytree >= 0.5 |
| Not using eval_set | Can't monitor overfitting during training | Always pass validation set |

---

## Summary

| Aspect | XGBoost | LightGBM |
|--------|---------|----------|
| **Type** | Supervised (classification + regression) | Supervised (classification + regression) |
| **Core idea** | Gradient boosting with regularized objective + 2nd-order optimization | Gradient boosting with histogram binning + leaf-wise growth |
| **Key innovation** | Regularized objective, exact split, parallel tree building | GOSS, EFB, leaf-wise growth, native categoricals |
| **Speed** | Fast | Faster (typically 2-5x) |
| **Key parameter** | max_depth (tree complexity) | num_leaves (tree complexity) |
| **Missing values** | Handled natively | Handled natively |
| **Feature scaling?** | Not required (tree-based) | Not required (tree-based) |
| **Overfitting control** | gamma, lambda, alpha, subsample, colsample, early stopping | reg_alpha, reg_lambda, subsample, colsample, min_child_samples, early stopping |
| **When to use** | Tabular data, competitions, production ML, need interpretability (SHAP) | Very large datasets, need speed, categorical-heavy data |
| **When NOT to use** | Very small datasets (use simpler models), image/text data (use deep learning) | Same as XGBoost |
| **Best practice** | Low learning rate + many trees + early stopping | Low learning rate + many trees + early stopping + constrain num_leaves |
