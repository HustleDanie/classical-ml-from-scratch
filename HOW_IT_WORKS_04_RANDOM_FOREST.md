# How It Works: Random Forest

> "Ask 100 people for their opinion, average the answers, and you'll be closer to the truth than any one person." That's the entire idea behind Random Forest.

---

## The One-Sentence Idea

Random Forest builds **hundreds of decision trees**, each trained on a slightly different random subset of the data and features, then combines their predictions by **voting** (classification) or **averaging** (regression).

---

## Intuition: The Wisdom of Crowds

Imagine you're trying to guess the number of jellybeans in a jar:
- **One person's guess:** Probably off by a lot (high variance)
- **Average of 100 people's guesses:** Surprisingly accurate

This works because individual errors are **random** -- some guess too high, some too low, and they cancel out when averaged. But this ONLY works if the guesses are **independent** (not everyone copies the same wrong answer).

Random Forest applies this principle to decision trees:
- One decision tree: high variance, overfits easily
- 100 independent trees averaged together: much more stable and accurate

---

## Why Single Trees Are Bad (The Problem Random Forest Solves)

Single decision trees have a fatal flaw -- **high variance**:

```
Dataset with 1000 samples:

Train on samples 1-1000:          Train on samples 1-900 + 100 replaced:
        Age <= 45?                         Income <= 50K?
       /         \                        /              \
   BP <= 130   Chol <= 200           Age <= 38         BP <= 145
   /    \       /     \              /    \             /     \
  ...   ...    ...    ...           ...   ...          ...    ...
  
Completely different tree!          Just from changing 10% of data!
```

Small changes in data produce completely different trees. This makes individual trees unreliable. Random Forest fixes this by building many trees and averaging out the instability.

---

## The Two Sources of Randomness

### Randomness #1: Bagging (Bootstrap Aggregating)

Each tree gets its own **bootstrap sample** -- a random sample of the training data, drawn **with replacement**.

```
Original data: [A, B, C, D, E, F, G, H, I, J]  (10 samples)

Tree 1 gets: [A, A, C, D, D, F, G, H, I, J]   (10 samples, some repeated)
Tree 2 gets: [B, B, C, D, E, F, G, I, I, J]   (different random sample)
Tree 3 gets: [A, C, C, E, F, F, G, H, J, J]   (different again)
```

Each bootstrap sample contains about **63%** of the original data (some samples appear multiple times, some not at all). The ~37% left out are called **Out-of-Bag (OOB)** samples -- these become free validation data!

**Why with replacement?** If we sampled without replacement, every tree would see the same data (just shuffled), and the trees would be nearly identical. Replacement makes each tree see a genuinely different subset.

### Randomness #2: Random Feature Subsets

At each split in each tree, only a **random subset of features** is considered:

```
All features: [age, income, BP, cholesterol, BMI, smoking, exercise, diet]  (8 features)

Tree 1, Node 1: considers [age, BP, exercise]        -> picks age
Tree 1, Node 2: considers [income, cholesterol, BMI]  -> picks cholesterol
Tree 2, Node 1: considers [BP, smoking, diet]          -> picks BP
Tree 2, Node 2: considers [age, BMI, exercise]         -> picks BMI
```

Typical number of features considered per split:
- **Classification:** $\sqrt{n\_features}$ (8 features -> consider 3 per split)
- **Regression:** $n\_features / 3$ (8 features -> consider 3 per split)

**Why?** Without this, every tree would split on the same "best" feature at the root. If `age` is the single best feature, all 100 trees would start with `age <= 50?` and be highly correlated. Random feature subsets force trees to explore different features, making them more diverse.

---

## The Math: How It All Works

### Training Phase

```
For t = 1, 2, ..., T (number of trees, e.g., T = 100):

  1. Draw a bootstrap sample of size N from training data (with replacement)
  
  2. Build a decision tree on this bootstrap sample:
     - At each node, randomly select m features (m = sqrt(n) for classification)
     - Among those m features, find the best split (lowest Gini/highest info gain)
     - Split the node
     - Repeat until stopping criteria (max_depth, min_samples_leaf, etc.)
  
  3. Do NOT prune the tree (let it grow deep -- overfitting is OK for individual trees!)
  
  Store tree_t
```

**Key insight:** Each tree is allowed to overfit! The averaging across many trees cancels out the overfitting.

### Prediction Phase

**Classification (voting):**

$$\hat{y} = \text{mode}\left(\text{tree}_1(x), \text{tree}_2(x), ..., \text{tree}_T(x)\right)$$

```
New patient: Age=55, BP=150, Cholesterol=230

Tree 1 predicts: Sick    (walked down tree 1 to a "Sick" leaf)
Tree 2 predicts: Sick    
Tree 3 predicts: Healthy
Tree 4 predicts: Sick
...
Tree 100 predicts: Sick

Vote count: 73 Sick, 27 Healthy
Final prediction: SICK (with 73% confidence)
```

**Regression (averaging):**

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} \text{tree}_t(x)$$

```
Tree 1 predicts: $320,000
Tree 2 predicts: $345,000
Tree 3 predicts: $298,000
...
Tree 100 average: $325,400     <-- smoother than any single tree
```

---

## Why Averaging Works: The Bias-Variance Trade-off

Every prediction error has two components:
- **Bias**: The model is systematically wrong (e.g., always predicts too low)
- **Variance**: The model's predictions jump around depending on training data

```
                  High Bias           High Variance
                  Low Variance        Low Bias

Target: X         . . .               .       .
                  . . .                  .
                  . . .   X          . X   .
                  . . .                 .
                  (all miss the same   (scattered around
                   direction)           the target)

                  Underfitting         Overfitting
                  (linear model on     (single deep tree)
                   non-linear data)
```

**Single decision tree:** Low bias (can fit complex patterns), HIGH variance (unstable)

**Random Forest:** Low bias (same as individual trees), LOW variance (averaging cancels randomness)

**Mathematically:** If you average $T$ independent predictions, each with variance $\sigma^2$:

$$\text{Variance of average} = \frac{\sigma^2}{T}$$

So 100 trees have 1/100th the variance of a single tree! In practice, trees are correlated (not fully independent), so the reduction is less dramatic but still substantial.

---

## What Happens Behind the Scenes: `.fit()` and `.predict()`

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,     # number of trees
    max_depth=None,       # let trees grow deep (no limit)
    max_features='sqrt',  # consider sqrt(n_features) per split
    min_samples_leaf=1,   # leaf can have just 1 sample (deep trees!)
    random_state=42
)

# .fit(X, y) does this:
# For each of the 100 trees:
#   1. Generate bootstrap sample (random sample with replacement)
#   2. Build a decision tree on that sample:
#      - At each split, randomly select sqrt(n) features
#      - Find the best split among those features
#      - Split the data
#      - Keep growing until leaves are pure (or min_samples_leaf reached)
#   3. Store the tree
model.fit(X_train, y_train)

# .predict(X) does this:
# For each test sample:
#   1. Pass it through all 100 trees
#   2. Each tree votes for a class
#   3. Return the majority vote
predictions = model.predict(X_test)

# .predict_proba(X) does this:
# Same as above, but returns the vote proportions as probabilities
# If 73 trees say "sick" and 27 say "healthy": [0.27, 0.73]
probabilities = model.predict_proba(X_test)
```

---

## Out-of-Bag (OOB) Score: Free Validation

Each tree only sees ~63% of the data. The other ~37% can be used as validation:

```
Tree 1 trained on: [A, A, C, D, D, F, G, H, I, J]
Tree 1 did NOT see: [B, E]  <-- use these to evaluate Tree 1

Tree 2 trained on: [B, B, C, D, E, F, G, I, I, J]
Tree 2 did NOT see: [A, H]  <-- use these to evaluate Tree 2

For sample A: Trees that didn't see A = {Tree 2, Tree 5, Tree 8, ...}
  OOB prediction for A = majority vote of those trees
  
OOB Score = accuracy of OOB predictions on all samples
```

This gives you a validation score **without needing a separate validation set!**

```python
model = RandomForestClassifier(n_estimators=100, oob_score=True)
model.fit(X_train, y_train)
print(model.oob_score_)    # e.g., 0.87 -- similar to cross-validation score
```

---

## Feature Importance in Random Forest

Random Forest averages feature importance across all trees, making it more reliable than a single tree:

$$\text{Importance}(f) = \frac{1}{T} \sum_{t=1}^{T} \text{Importance}_t(f)$$

Two methods:

### Method 1: Gini Importance (Default)
Sum up how much each feature reduced impurity across all trees (same as single tree, but averaged).

### Method 2: Permutation Importance (More Reliable)
1. Measure baseline accuracy on OOB data
2. Randomly shuffle one feature's values
3. Re-measure accuracy -- if it drops a lot, that feature was important
4. Repeat for each feature

```python
# Gini importance (fast, built-in):
model.feature_importances_

# Permutation importance (more reliable):
from sklearn.inspection import permutation_importance
result = permutation_importance(model, X_test, y_test, n_repeats=10)
result.importances_mean
```

**Why permutation is better:** Gini importance is biased toward high-cardinality features (features with many unique values). Permutation importance isn't.

---

## Key Hyperparameters: What They Control

| Parameter | What It Controls | Effect of Increasing |
|-----------|-----------------|---------------------|
| `n_estimators` | Number of trees | More stable predictions, slower training. Usually 100-500 is enough |
| `max_depth` | How deep each tree can grow | Deeper = more complex, more overfitting per tree (but averaging helps) |
| `max_features` | Features considered per split | Lower = more diversity between trees = less correlation = better ensemble |
| `min_samples_leaf` | Minimum samples in a leaf | Higher = simpler trees = less overfitting |
| `min_samples_split` | Minimum samples to split a node | Higher = simpler trees |
| `bootstrap` | Whether to use bootstrap sampling | True (default). False = each tree sees all data (less diversity) |

**The most important parameter is `n_estimators`:**

```
Accuracy
  |
  |   * * * * * * * * * * * * * * * * * * *  <-- plateaus after ~200 trees
  |  *
  | *
  |*
  +---|---|---|---|---|---|---|---|---|---|---
      50  100 150 200 250 300 350 400 450 500
                     n_estimators
```

More trees NEVER hurts accuracy (unlike max_depth which can cause overfitting). It only costs more time.

---

## Random Forest vs. Single Decision Tree

```
Single Tree:                    Random Forest:

Prediction surface:             Prediction surface:
Sharp, jagged steps             Smooth, averaged steps

Price                           Price
  |    ___                        |       ___
  |   |   |____                   |     /     \___
  |___|        |___               |   /           \___
  |                |___           | /                  \___
  +-----|-----|-----|---          +-----|-----|-----|---
        sqft                           sqft

Train accuracy: 100%            Train accuracy: 95%
Test accuracy:  72%             Test accuracy:  88%

One tree memorizes.             Many trees generalize.
```

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Too few trees (`n_estimators=10`) | Predictions are noisy | Use 100-500 trees (more is always safe) |
| Not setting `random_state` | Results change every run | Set `random_state=42` for reproducibility |
| Very high cardinality features | Trees always split on those features | Use permutation importance or limit `max_features` |
| Expecting extrapolation | RF can only predict within range of training data | All tree-based models have this limitation |
| Ignoring `max_features` | Trees are too correlated (all split on same features) | Use `'sqrt'` (classification) or `'log2'` |
| Using very deep trees on small data | Individual trees overfit severely | Set `max_depth` or `min_samples_leaf` |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it is** | Ensemble of many decision trees with randomness injected |
| **Two sources of randomness** | 1) Bootstrap sampling of rows, 2) Random subset of features per split |
| **Training** | Build T independent trees on bootstrap samples |
| **Prediction** | Majority vote (classification) or average (regression) |
| **Why it works** | Averaging reduces variance without increasing bias |
| **Key parameter** | `n_estimators` (more trees = better, but slower) |
| **Feature scaling?** | Not needed (tree-based) |
| **Interpretable?** | Moderate -- feature importance is available but you can't read 100 trees |
| **When to use** | Default "first try" model, tabular data, when you need good accuracy with minimal tuning |
| **When NOT to use** | When you need a single interpretable model, very high-dimensional sparse data, real-time prediction with strict latency |
