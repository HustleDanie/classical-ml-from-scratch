# How It Works: Decision Trees

> The most intuitive model in machine learning. A decision tree makes predictions exactly the way a human would -- by asking a series of yes/no questions.

---

## The One-Sentence Idea

A decision tree splits your data into smaller and smaller groups by asking "Is feature X above or below value Y?" at each step, until each group is mostly one class (classification) or has similar values (regression).

---

## Intuition: How You Already Think in Decision Trees

You use decision trees every day without realizing it:

```
"Should I bring an umbrella?"

                    Is it cloudy?
                   /            \
                 Yes              No
                /                   \
        Is humidity > 70%?        Don't bring it
        /               \
      Yes                No
      /                    \
  Bring it!            Check forecast
                       /         \
                    Rain?       No rain?
                    /               \
               Bring it!      Don't bring it
```

A machine learning decision tree does the same thing, but it **automatically discovers** the best questions to ask and the best order to ask them.

---

## The Math: How the Tree Decides What to Ask

### The Big Question: Which Feature to Split On?

At every node, the tree has to choose:
1. **Which feature** to split on (age? income? blood pressure?)
2. **What value** to split at (age > 50? income > $60K?)

It tries EVERY possible feature and EVERY possible split value, and picks the one that creates the **purest** groups.

### Measuring "Purity": Gini Impurity (Classification)

**Gini Impurity** measures how mixed a group is:

$$\text{Gini} = 1 - \sum_{k=1}^{K} p_k^2$$

Where $p_k$ is the proportion of class $k$ in the group.

**Examples:**
- **Pure group** (all one class): 50 cats, 0 dogs
  - Gini $= 1 - (1.0^2 + 0.0^2) = 1 - 1.0 = 0.0$ (perfect purity!)
- **Maximally mixed** (50/50): 25 cats, 25 dogs
  - Gini $= 1 - (0.5^2 + 0.5^2) = 1 - 0.5 = 0.5$ (worst)
- **Mostly one class**: 40 cats, 10 dogs
  - Gini $= 1 - (0.8^2 + 0.2^2) = 1 - 0.68 = 0.32$ (decent)

**The tree picks the split that reduces Gini the most** -- making the child groups purer than the parent.

### Alternative: Entropy / Information Gain

**Entropy** is another way to measure impurity:

$$\text{Entropy} = -\sum_{k=1}^{K} p_k \log_2(p_k)$$

- Pure group: Entropy = 0
- 50/50 mixed: Entropy = 1.0
- 80/20: Entropy = 0.72

**Information Gain** = parent entropy - weighted average of children's entropy. The tree picks the split with the highest information gain.

In practice, Gini and Entropy give nearly identical trees. Gini is default in sklearn because it's slightly faster to compute (no logarithm).

### For Regression: Variance Reduction

When predicting a number (not a class), the tree uses **variance** instead:

$$\text{Impurity} = \frac{1}{N} \sum (y_i - \bar{y})^2$$

The best split minimizes the total variance in the children groups. Each leaf predicts the **average** of the values in that leaf.

---

## Step-by-Step: Building a Tree

Let's walk through a concrete example: predicting whether someone will buy a product.

**Data:**

| Age | Income | Student? | Buys? |
|-----|--------|----------|-------|
| 25  | 30K    | Yes      | No    |
| 35  | 60K    | No       | Yes   |
| 45  | 80K    | No       | Yes   |
| 20  | 20K    | Yes      | No    |
| 50  | 90K    | No       | Yes   |
| 30  | 45K    | Yes      | No    |
| 40  | 70K    | No       | Yes   |
| 22  | 25K    | Yes      | No    |

### Iteration 1: Find the Best First Split

The tree tries every possible split:

```
Try: Age <= 27?
  Left (Age<=27):  [No, No, No]     -> Gini = 0.0 (pure!)
  Right (Age>27):  [Yes, Yes, No, Yes, Yes] -> Gini = 0.32
  Weighted Gini = (3/8)*0.0 + (5/8)*0.32 = 0.20

Try: Age <= 32?
  Left (Age<=32):  [No, No, No, No]  -> Gini = 0.0 (pure!)
  Right (Age>32):  [Yes, Yes, Yes, Yes] -> Gini = 0.0 (pure!)
  Weighted Gini = 0.0 + 0.0 = 0.0    <-- PERFECT SPLIT!

Try: Income <= 50K?
  Left:  [No, No, No, No]  -> Gini = 0.0
  Right: [Yes, Yes, Yes, Yes] -> Gini = 0.0
  Weighted Gini = 0.0             <-- Also perfect!

Try: Student = Yes?
  Left (Yes):  [No, No, No, No]  -> Gini = 0.0
  Right (No):  [Yes, Yes, Yes, Yes] -> Gini = 0.0
  Weighted Gini = 0.0            <-- Also perfect!
```

Multiple features give a perfect split. The tree picks one (say, Income <= 37.5K which is the midpoint between 30K and 45K for this data).

### Iteration 2: Split Each Child (If Needed)

```
                Income <= 37.5K?
               /                \
         Yes (<=37.5K)       No (>37.5K)
         [No, No, No]       [Yes, Yes, Yes, Yes, No]
         Gini = 0.0          Gini = 0.32
         LEAF: "No"          Need to split more...

For the right child, try more splits:
  Income <= 42.5K?
    Left:  [No]       -> Pure
    Right: [Yes, Yes, Yes, Yes] -> Pure

                Income <= 37.5K?
               /                 \
         LEAF: "No"           Income <= 42.5K?
                             /               \
                       LEAF: "No"        LEAF: "Yes"
```

### The Tree Grows Until a Stopping Condition

The tree keeps splitting until:
- **A leaf is pure** (Gini = 0) -- no more splits needed
- **Max depth reached** (e.g., `max_depth=5`)
- **Too few samples** in a node (e.g., `min_samples_split=10`)
- **No split improves purity** by more than a threshold

---

## What Happens Behind the Scenes: `.fit()` and `.predict()`

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)

# .fit(X, y) does this:
# 1. Start with ALL data in one node (the root)
# 2. For each feature, for each possible split value:
#      - Compute Gini impurity of the resulting left/right children
#      - Weighted average Gini = (N_left/N)*Gini_left + (N_right/N)*Gini_right
# 3. Pick the feature + value that gives the lowest weighted Gini
# 4. Split the data into two groups
# 5. Recurse: repeat steps 2-4 for each child node
# 6. Stop when stopping criteria are met
# 7. Store the tree structure (feature, threshold, left_child, right_child)
model.fit(X_train, y_train)

# .predict(X) does this:
# 1. Start at the root node
# 2. Check: is feature[split_feature] <= threshold?
#    - Yes: go to left child
#    - No: go to right child
# 3. Repeat until reaching a leaf
# 4. Return the majority class in that leaf (classification)
#    or the average value in that leaf (regression)
predictions = model.predict(X_test)
```

---

## The Full Pipeline: Visual Walkthrough

```
TRAINING (.fit):

All 1000 patients              Best split: Age <= 50?
+-----------------------+      +-----------+-----------+
| 600 healthy, 400 sick |  ->  | Left      | Right     |
| Gini = 0.48           |      | 500h, 100s| 100h, 300s|
+-----------------------+      | Gini=0.32 | Gini=0.38 |
                               +-----------+-----------+
                                    |              |
                               BP <= 130?     Chol <= 200?
                              /        \       /        \
                         LEAF:h    LEAF:s  LEAF:h    LEAF:s
                         480h,20s  20h,80s  60h,40s  40h,260s


PREDICTION (.predict):

New patient: Age=55, BP=150, Cholesterol=230

  Age <= 50?  --> No (55 > 50) --> Go RIGHT
  Chol <= 200? --> No (230 > 200) --> Go RIGHT
  LEAF: 40 healthy, 260 sick --> Predict: SICK (87% probability)
```

---

## How a Tree Handles Different Data Types

### Numerical Features (Age, Income)
The tree tries every possible threshold: Age <= 20? Age <= 25? Age <= 30? ...
It picks the threshold that gives the best split.

Actually, it's smarter: it **sorts** the values and only tries thresholds between consecutive unique values. If ages are [20, 25, 30, 35, 40], it tries: 22.5, 27.5, 32.5, 37.5.

### Categorical Features (Color = Red/Blue/Green)
For binary splits: try every possible grouping:
- {Red} vs {Blue, Green}
- {Blue} vs {Red, Green}
- {Green} vs {Red, Blue}

With many categories (like 50 states), this gets expensive. sklearn requires you to **one-hot encode** categorical features first.

---

## Overfitting: The Tree's Biggest Problem

An unrestricted tree will keep splitting until every leaf has exactly one sample. This **memorizes** the training data:

```
Unrestricted tree (overfitting):        Pruned tree (good):

         Age<=50?                           Age<=50?
        /       \                          /       \
    BP<=130?   Chol<=200?              LEAF:h    LEAF:s
    /    \      /      \               (85%)      (75%)
  ...    ...  ...     ...
  / \   / \   / \     / \
 h   s h   s h   s   h   s    <-- One sample per leaf = memorized!
```

### How to Prevent Overfitting

| Parameter | What It Does | Typical Value |
|-----------|-------------|---------------|
| `max_depth` | Limits how deep the tree can grow | 3-10 |
| `min_samples_split` | Minimum samples needed to split a node | 10-50 |
| `min_samples_leaf` | Minimum samples in a leaf node | 5-20 |
| `max_leaf_nodes` | Maximum number of leaf nodes | 20-100 |
| `max_features` | Number of features to consider per split | sqrt(n) or log2(n) |

**Pruning** is another approach: grow the full tree, then cut back branches that don't improve validation performance. sklearn uses `ccp_alpha` for cost-complexity pruning.

---

## Regression Trees: Predicting Numbers

Everything works the same, except:
- **Impurity measure**: Variance instead of Gini
- **Leaf prediction**: Average of values in the leaf (not majority vote)

```
Predicting house price:

            sqft <= 1500?
           /             \
      sqft <= 1000?    sqft <= 2500?
       /        \       /         \
  LEAF:$150K  LEAF:$220K  LEAF:$350K  LEAF:$500K
  (avg price) (avg price) (avg price)  (avg price)
```

**Important limitation:** A regression tree predicts a **constant value** within each leaf. The prediction function is a staircase, not a smooth curve:

```
Price
$500K |                    ___________
$350K |          _________|
$220K |   ______|
$150K |__|
      +-----|------|------|------|---
           1000   1500   2500   sqft
```

This is why single trees are rarely used alone for regression -- they can't predict values between the staircase steps.

---

## Feature Importance: Which Features Matter?

Decision trees naturally tell you which features are most important:

$$\text{Importance}(f) = \sum_{\text{nodes using } f} \frac{N_{\text{node}}}{N_{\text{total}}} \cdot \Delta\text{Impurity}$$

For each node that splits on feature $f$, we add up how much impurity it reduced, weighted by how many samples pass through that node.

```python
model.feature_importances_
# [0.45, 0.30, 0.15, 0.10]  -- Age is most important (45%)
```

**Caution:** Feature importance in single trees can be misleading:
- Correlated features split the importance between them
- A feature could be important just because it was checked first
- Importance doesn't tell you the DIRECTION of the effect

---

## Advantages and Limitations

### Advantages
1. **Instantly interpretable** -- you can draw the tree and explain it to anyone
2. **No feature scaling needed** -- trees don't care about magnitude
3. **Handles mixed data types** -- numbers and categories naturally
4. **Captures non-linear relationships** -- the staircase can approximate any shape
5. **Captures interactions** -- "Age > 50 AND BP > 140" happens naturally through nested splits
6. **Fast training and prediction**

### Limitations
1. **Overfits easily** -- without constraints, memorizes training data
2. **Unstable** -- small data changes can produce completely different trees
3. **Biased toward features with many levels** -- a feature with 100 unique values gets more split opportunities than one with 2
4. **Can't extrapolate** -- predicts within the range of training data only
5. **Axis-aligned splits** -- boundaries are always perpendicular to feature axes (can't draw diagonal boundaries)

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| No depth limit | Tree memorizes training data (100% train, 60% test accuracy) | Set `max_depth=5` or use pruning |
| High cardinality features | Tree favors features with many unique values | Limit with `max_features` |
| Imbalanced classes | Tree predicts majority class in most leaves | Use `class_weight='balanced'` |
| Expecting smooth predictions | Regression tree gives staircase predictions | Use ensemble (Random Forest, Gradient Boosting) |
| Interpreting a deep tree | 50-level tree is impossible to read | Keep depth <= 5 for interpretability |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it learns** | A tree of if/else rules (feature, threshold, left, right) |
| **How it learns** | Greedily picks the split that reduces impurity most (Gini or Entropy) |
| **Prediction** | Walk down the tree following yes/no questions to reach a leaf |
| **Output** | Class label + probability (classification) or average value (regression) |
| **Speed** | Fast to train, very fast to predict ($O(\log N)$ -- just tree depth) |
| **Interpretable?** | Yes -- the most interpretable model (can draw and read the tree) |
| **Feature scaling?** | Not needed |
| **When to use** | When you need interpretability, mixed data types, or as base for ensembles |
| **When NOT to use** | When you need high accuracy (single trees are weak learners) |
