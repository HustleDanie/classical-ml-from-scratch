# How It Works: Gradient Boosting

> Random Forest builds trees independently and averages them. Gradient Boosting builds trees **sequentially**, where each new tree tries to fix the mistakes of all the previous trees combined.

---

## The One-Sentence Idea

Gradient Boosting starts with a simple guess, then repeatedly builds small trees that predict the **errors** (residuals) of the current model, adding each new tree's corrections to gradually improve the overall prediction.

---

## Intuition: Learning from Mistakes

Imagine you're an archery student:
- **Shot 1:** You aim at the target and miss 30 cm to the right
- **Correction 1:** Aim 30 cm to the left of where you aimed before. Now you only miss by 5 cm up
- **Correction 2:** Aim 5 cm down. Now you only miss by 1 cm to the right
- **Correction 3:** Aim 1 cm to the left. Almost bullseye!

Each "correction" doesn't try to hit the target from scratch -- it just tries to fix the **remaining error** from all previous shots combined.

Gradient Boosting does exactly this with decision trees:
- Tree 1: Makes a rough prediction
- Tree 2: Predicts the errors of Tree 1
- Tree 3: Predicts the errors of Tree 1 + Tree 2
- ...
- Tree 100: Predicts the tiny remaining errors

---

## Random Forest vs. Gradient Boosting: The Core Difference

```
RANDOM FOREST (parallel, independent):

Tree 1 ---> prediction 1 \
Tree 2 ---> prediction 2  \
Tree 3 ---> prediction 3   >---> AVERAGE = final prediction
...                        /
Tree 100 -> prediction 100/

Each tree is independent, full-sized, and works on a bootstrap sample.


GRADIENT BOOSTING (sequential, dependent):

Tree 1 ---> rough prediction
       |
       v
    errors = actual - prediction
       |
       v
Tree 2 ---> predicts errors ---> adds correction to prediction
       |
       v
    remaining errors
       |
       v
Tree 3 ---> predicts remaining errors ---> adds correction
       ...
Tree 100 -> predicts tiny final errors -> final correction

Each tree is small, and depends on ALL previous trees.
```

---

## The Math: Step by Step

### Step 1: Start with a Simple Guess

For regression, start with the **average** of all target values:

$$F_0(x) = \bar{y} = \text{mean of all targets}$$

For example, if predicting house prices and the average is $300K, every house starts with a prediction of $300K.

### Step 2: Compute Residuals (Errors)

$$r_i = y_i - F_0(x_i)$$

| House | Actual Price | Current Prediction | Residual (Error) |
|-------|-------------|-------------------|-----------------|
| 1     | $450K       | $300K             | +$150K          |
| 2     | $200K       | $300K             | -$100K          |
| 3     | $350K       | $300K             | +$50K           |
| 4     | $280K       | $300K             | -$20K           |

### Step 3: Build a Small Tree to Predict the Residuals

We train a decision tree (typically `max_depth=3-6`, very small!) to predict these residuals:

```
Tree 1 (predicting residuals):

        sqft <= 1500?
       /             \
  beds <= 2?       beds <= 3?
  /       \        /        \
-$80K   -$10K   +$60K    +$130K     <-- predicted residuals
```

### Step 4: Update the Prediction (With a Learning Rate)

$$F_1(x) = F_0(x) + \eta \cdot h_1(x)$$

Where:
- $F_0(x)$ = previous prediction ($300K for everyone)
- $h_1(x)$ = what the new tree predicts for this house's residual
- $\eta$ = **learning rate** (e.g., 0.1) -- we only add a FRACTION of the correction

**Why not add the full correction?** If we add the full correction, we overfit to the training residuals. Adding only 10% of the correction is like taking small careful steps:

```
Learning rate = 1.0 (aggressive):     Learning rate = 0.1 (careful):
Jump to the correction immediately.    Take small steps toward the correction.
Faster but overshoots.                 Slower but more accurate.

Error                                  Error
|*                                     |*
|  *                                   | *
|    *    *                            |  *
|      * * * <- bouncing!              |   *
|        *                             |    *
|                                      |     *
                                       |      *  <- smooth convergence
```

For House 1:
- $F_0 = 300K$
- Tree 1 predicts residual = +$130K
- $F_1 = 300K + 0.1 \times 130K = 300K + 13K = 313K$
- Still not at $450K, but closer!

### Step 5: Repeat (Build More Trees)

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Each tree predicts the residuals of the **current ensemble**, not the original target.

```
After Tree 1:  Prediction for House 1 = $313K  (error: $137K)
After Tree 2:  Prediction for House 1 = $326K  (error: $124K)
After Tree 3:  Prediction for House 1 = $338K  (error: $112K)
...
After Tree 50: Prediction for House 1 = $442K  (error: $8K)
After Tree 100: Prediction for House 1 = $448K (error: $2K)
```

The final model is the **sum** of all trees:

$$F(x) = F_0 + \eta \cdot h_1(x) + \eta \cdot h_2(x) + ... + \eta \cdot h_T(x)$$

### Why "Gradient" Boosting?

The residuals $r_i = y_i - F(x_i)$ are actually the **negative gradient** of the MSE loss function:

$$\frac{\partial \text{MSE}}{\partial F(x_i)} = \frac{\partial}{\partial F(x_i)} (y_i - F(x_i))^2 = -2(y_i - F(x_i))$$

So fitting a tree to the residuals is equivalent to doing gradient descent in **function space** -- each tree is a step in the direction that decreases the loss the fastest.

This is powerful because we can use ANY differentiable loss function:
- **MSE** for regression: residuals = $y - \hat{y}$
- **Log Loss** for classification: residuals = $y - p$ (where $p$ is predicted probability)
- **Custom losses**: absolute error, quantile regression, etc.

---

## The Full Pipeline: What Happens Behind the Scenes

```
TRAINING (.fit):

Step 0: F_0 = mean(y) = $300K for all houses

Step 1: residuals = y - F_0
        +----+----------+--------+----------+
        | ID | Actual   | F_0    | Residual |
        +----+----------+--------+----------+
        | 1  | $450K    | $300K  | +$150K   |
        | 2  | $200K    | $300K  | -$100K   |
        | 3  | $350K    | $300K  | +$50K    |
        +----+----------+--------+----------+
        
        Train Tree_1 on (X, residuals)
        Tree_1 learns: if sqft>2000 and beds>3: residual = +$130K
        
        F_1 = F_0 + 0.1 * Tree_1
        House 1: F_1 = $300K + 0.1 * $130K = $313K

Step 2: residuals = y - F_1
        House 1: $450K - $313K = +$137K
        
        Train Tree_2 on (X, new residuals)
        F_2 = F_1 + 0.1 * Tree_2

... repeat 100-1000 times ...


PREDICTION (.predict):

New house: sqft=2200, beds=4

F_0 = $300K                        (base prediction)
+ 0.1 * Tree_1(new_house) = +$13K  (correction 1)
+ 0.1 * Tree_2(new_house) = +$11K  (correction 2)
+ 0.1 * Tree_3(new_house) = +$9K   (correction 3)
+ ...
+ 0.1 * Tree_100(new_house) = +$0.2K (tiny final correction)

Final prediction = $300K + $13K + $11K + ... = $448K
```

---

## For Classification: How It Works

For binary classification, Gradient Boosting predicts **log-odds** (just like logistic regression) and converts to probability via sigmoid:

```
Step 0: F_0 = log(p/(1-p)) where p = proportion of positive class
        If 30% of patients are sick: F_0 = log(0.3/0.7) = -0.847

Step 1: Convert to probability: p = sigmoid(F_0) = 0.30
        Residuals = y - p  (1 - 0.30 = 0.70 for sick patients, 0 - 0.30 = -0.30 for healthy)
        Train Tree_1 on (X, residuals)
        F_1 = F_0 + 0.1 * Tree_1

Step 2: Convert to probability: p = sigmoid(F_1)
        Residuals = y - p
        ...continue...

Final: P(sick) = sigmoid(F_0 + 0.1*Tree_1 + 0.1*Tree_2 + ... + 0.1*Tree_T)
```

---

## Key Hyperparameters and Their Interactions

### The Three Most Important Parameters

These three form a **triangle of trade-offs**:

```
             n_estimators (number of trees)
                 /\
                /  \
               /    \
              /      \
             /        \
learning_rate          max_depth
(step size)           (tree complexity)
```

**Rule of thumb:** Lower learning rate + more trees = better accuracy but slower training.

| Parameter | What It Does | Typical Range | Intuition |
|-----------|-------------|---------------|-----------|
| `n_estimators` | Number of sequential trees | 100-3000 | More corrections = better (but watch for overfitting) |
| `learning_rate` | How much of each correction to apply | 0.01-0.3 | Small steps are more careful |
| `max_depth` | Depth of each individual tree | 3-8 | Shallow trees = simple corrections |

**Why shallow trees?** Each tree should be a "weak learner" -- just slightly better than random. Deep trees overfit to the current residuals. The boosting process is what creates the complex model, not any individual tree.

```
Depth 3 tree (good for boosting):
        age <= 50?
       /         \
   BP<=130     Chol<=200
   /    \       /     \
  L1    L2    L3      L4     <-- 4 leaves, simple corrections

Depth 10 tree (too complex for boosting):
Has 1024 leaves, memorizes the current residuals
Next trees can't improve much -> wasteful
```

### Other Important Parameters

| Parameter | What It Does | Typical Range |
|-----------|-------------|---------------|
| `subsample` | Fraction of data for each tree | 0.5-1.0 |
| `colsample_bytree` | Fraction of features for each tree | 0.5-1.0 |
| `min_child_weight` / `min_samples_leaf` | Min samples in a leaf | 1-100 |
| `reg_alpha` (L1) | Lasso-like weight penalty | 0-10 |
| `reg_lambda` (L2) | Ridge-like weight penalty | 0-10 |

**`subsample` and `colsample_bytree`** add randomness similar to Random Forest, reducing overfitting. Setting both to 0.8 is a good starting point.

---

## When to Stop: Early Stopping

Unlike Random Forest (where more trees never hurts), in Gradient Boosting **too many trees causes overfitting**:

```
Error
  |
  |  *                                Training error
  |   *   * * * * * * * * * * * * *   keeps going down
  |     *
  |       *
  |
  |  *
  |    *
  |      *
  |        * * *                      Validation error
  |              * * *                goes UP after some point
  |                   * * * * * * *   (OVERFITTING!)
  +---|---|---|---|---|---|---|---|---
      50  100 150 200 250 300 350
              n_estimators
              
  STOP HERE ----^  (early stopping at ~200 trees)
```

**Early stopping:** Monitor validation loss. If it doesn't improve for `N` rounds, stop training.

```python
import xgboost as xgb

model = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.05)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50    # stop if no improvement for 50 rounds
)
# Might stop at 847 trees even though we set 3000
```

---

## XGBoost vs. LightGBM vs. sklearn GradientBoosting

All implement the same core idea, but with different optimizations:

| Feature | sklearn GB | XGBoost | LightGBM |
|---------|-----------|---------|----------|
| Speed | Slow | Fast | Fastest |
| Tree growth | Depth-first | Level-wise | Leaf-wise |
| Missing values | Must impute first | Handles natively | Handles natively |
| Categorical features | Must encode | Must encode | Handles natively |
| Regularization | Basic | L1 + L2 + pruning | L1 + L2 + pruning |
| GPU support | No | Yes | Yes |
| Typical use | Learning/small data | Competitions/production | Competitions/production |

**Leaf-wise vs. Level-wise growth:**

```
Level-wise (XGBoost):              Leaf-wise (LightGBM):
  
Grows ALL nodes at depth 1,        Grows the SINGLE leaf with
then depth 2, etc.                  the highest gain, wherever it is.

Depth 1:  [  1  ]                   Step 1: [  1  ]
Depth 2: [ 2 ][ 3 ]                Step 2: [  1  ][ 2 ]
Depth 3: [4][5][6][7]              Step 3: [  1  ][ 2 ]
                                                   [3][4]
Balanced tree.                      Unbalanced but more accurate.
```

LightGBM's leaf-wise approach is faster because it focuses computation where it helps most.

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Learning rate too high (0.3+) | Overfits quickly, needs fewer trees | Use 0.01-0.1 with more trees |
| No early stopping | Trains too many trees, overfits | Always use early stopping on validation set |
| Trees too deep (`max_depth=10+`) | Individual trees overfit to residuals | Use `max_depth=3-6` for boosting |
| Forgetting to tune `max_depth` + `learning_rate` together | Suboptimal combination | Lower LR needs deeper trees, higher LR needs shallower |
| Not using `subsample` | Can overfit on small data | Set `subsample=0.8, colsample_bytree=0.8` |
| Too few trees with low learning rate | Underfitting (not enough corrections) | Increase `n_estimators` or increase `learning_rate` |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it learns** | A sequence of small trees, each correcting previous errors |
| **How it learns** | Builds trees on residuals (gradient descent in function space) |
| **Prediction** | Sum of base prediction + all tree corrections (times learning rate) |
| **Key idea** | Sequential: each tree depends on all previous trees |
| **Most important params** | `n_estimators`, `learning_rate`, `max_depth` (triangle of trade-offs) |
| **Feature scaling?** | Not needed (tree-based) |
| **Handles missing data?** | XGBoost/LightGBM: yes. sklearn: no |
| **Interpretable?** | Less than single tree, but SHAP values available |
| **When to use** | Tabular data, competitions, highest accuracy needed |
| **When NOT to use** | Very small data (overfits), need real-time training, need simple model |
| **vs. Random Forest** | Usually more accurate, but harder to tune and easier to overfit |
