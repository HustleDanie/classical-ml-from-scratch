# How It Works: K-Nearest Neighbors (KNN)

> KNN is the simplest model in all of machine learning. It has NO training step. When you ask it to predict, it just looks at the nearest points in the training data and copies their answer.

---

## The One-Sentence Idea

To predict for a new data point, KNN finds the **K closest data points** in the training set and returns their **majority vote** (classification) or **average** (regression).

---

## Intuition: Ask Your Neighbors

Imagine you move to a new neighborhood and want to know if you'll like the local restaurant. You don't build a mathematical model -- you just ask your 5 nearest neighbors. If 4 out of 5 say it's good, you predict: it's good.

That's literally all KNN does.

```
New point (?) wants to know: Am I class A or class B?

K=3:                            K=7:

  A   A                           A   A
    A   B    B                      A   B    B
  A   (?)  B                      A   (?)  B
    A   B B                         A   B B
  A     B                        A     B
  
3 nearest: A, A, B               7 nearest: A, A, A, A, B, B, B
Vote: A wins (2-1)               Vote: A wins (4-3)
Predict: A                       Predict: A
```

---

## The Math: How Distance Works

### Step 1: Define "Closeness" (Distance Metric)

The most common distance metric is **Euclidean distance** (straight-line distance):

$$d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{n} (x_j - x_j')^2}$$

**Example:** Point A = (age=30, income=50K), Point B = (age=40, income=70K)

$$d = \sqrt{(30-40)^2 + (50000-70000)^2} = \sqrt{100 + 400000000} = 20000.0$$

Wait -- the income difference completely dominates! This is why **feature scaling is absolutely critical** for KNN.

After StandardScaler:
- Age: 30 -> -0.5, 40 -> 0.5
- Income: 50K -> -0.7, 70K -> 0.3

$$d = \sqrt{(-0.5-0.5)^2 + (-0.7-0.3)^2} = \sqrt{1.0 + 1.0} = 1.41$$

Now both features contribute equally.

### Other Distance Metrics

| Metric | Formula | When to Use |
|--------|---------|------------|
| **Euclidean** (L2) | $\sqrt{\sum (x_j - x_j')^2}$ | Default, continuous features |
| **Manhattan** (L1) | $\sum |x_j - x_j'|$ | High dimensions, robust to outliers |
| **Minkowski** (Lp) | $(\sum |x_j - x_j'|^p)^{1/p}$ | General form (p=1 is Manhattan, p=2 is Euclidean) |
| **Cosine** | $1 - \frac{\mathbf{x} \cdot \mathbf{x}'}{||\mathbf{x}|| \cdot ||\mathbf{x}'||}$ | Text data, when direction matters more than magnitude |
| **Hamming** | Number of features that differ | Categorical features |

### Step 2: Find the K Nearest Neighbors

For a new point, compute its distance to EVERY training point, then sort:

```
Training points and their distances to the new point:

Point 1: distance = 2.3  (class A)
Point 2: distance = 5.7  (class B)
Point 3: distance = 1.1  (class A)    <-- 1st nearest
Point 4: distance = 3.8  (class B)
Point 5: distance = 1.5  (class A)    <-- 2nd nearest
Point 6: distance = 2.1  (class B)    <-- 3rd nearest
Point 7: distance = 8.2  (class A)
...

K=3: Nearest neighbors are Point 3 (A), Point 5 (A), Point 6 (B)
```

### Step 3: Vote or Average

**Classification:** Majority vote among the K neighbors

$$\hat{y} = \text{mode}(y_{n_1}, y_{n_2}, ..., y_{n_K})$$

K=3 neighbors: A, A, B -> Predict **A** (2 votes vs 1)

**Regression:** Average of the K neighbors' values

$$\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_{n_i}$$

K=3 neighbors: prices $300K, $320K, $280K -> Predict **$300K**

### Optional: Distance-Weighted Voting

Closer neighbors get more influence:

$$\text{weight}_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)^2}$$

```
K=3 neighbors:
Point 3: distance=1.1, class A, weight=1/1.21=0.83
Point 5: distance=1.5, class A, weight=1/2.25=0.44
Point 6: distance=2.1, class B, weight=1/4.41=0.23

Weighted vote: A gets 0.83+0.44=1.27, B gets 0.23
Predict: A (with much higher confidence than unweighted)
```

---

## What Happens Behind the Scenes: `.fit()` and `.predict()`

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')

# .fit(X, y) does this:
# 1. Store the entire training dataset in memory
# 2. Build a spatial index (KD-Tree or Ball Tree) for fast neighbor lookup
# 3. That's it! No "learning" happens!
model.fit(X_train, y_train)

# .predict(X) does this:
# For EACH test point:
#   1. Find the K=5 nearest training points (using the spatial index)
#   2. Look at their labels
#   3. Return the majority vote
predictions = model.predict(X_test)

# .predict_proba(X):
# Returns the proportion of each class among the K neighbors
# K=5 neighbors: A, A, A, B, B -> [0.6, 0.4]
probabilities = model.predict_proba(X_test)
```

**The "training" step is literally just storing the data.** This is why KNN is called a **lazy learner** -- it does all the work at prediction time.

---

## The Full Pipeline

```
TRAINING (.fit):  ← Almost nothing happens!

Training data:
[Point 1: (2.3, 4.5), class A]
[Point 2: (1.1, 3.2), class A]      Just store
[Point 3: (5.6, 7.8), class B]  ->  these in  ->  DONE!
[Point 4: (6.2, 8.1), class B]      memory
[Point 5: (3.0, 5.0), class A]      (+ build index)


PREDICTION (.predict):  ← All the work happens here!

New point: (4.0, 6.0), K=3

Step 1: Compute distances to ALL training points:
  d(new, P1) = sqrt((4-2.3)^2 + (6-4.5)^2) = 2.12
  d(new, P2) = sqrt((4-1.1)^2 + (6-3.2)^2) = 4.16
  d(new, P3) = sqrt((4-5.6)^2 + (6-7.8)^2) = 2.41
  d(new, P4) = sqrt((4-6.2)^2 + (6-8.1)^2) = 2.90
  d(new, P5) = sqrt((4-3.0)^2 + (6-5.0)^2) = 1.41

Step 2: Pick 3 nearest:
  P5 (d=1.41, A), P1 (d=2.12, A), P3 (d=2.41, B)

Step 3: Vote:
  A: 2 votes, B: 1 vote -> PREDICT A
```

---

## Choosing K: The Most Important Decision

K is the **only real hyperparameter**, and it controls the bias-variance trade-off:

```
K=1 (very complex):           K=5 (moderate):          K=N (trivial):
Every point is its own        Smooth boundary           Always predicts
neighborhood. Very            from averaging 5          the most common
jagged boundary.              neighbors.                class (useless).

  A|B A                         A    |    B               A A A A A
  B|A B B                       A  A |  B B               A A A A A
  A A|B A                       A    |  B                  A A A A A
     |B                              | B
  
  Overfits                      Good balance              Underfits
  (memorizes noise)             (generalizes well)        (too simple)
```

**Effect of K on decision boundary:**

| Small K (1-3) | Medium K (5-15) | Large K (50+) |
|---------------|----------------|---------------|
| Complex, jagged boundary | Smooth boundary | Very smooth, almost linear |
| Low bias, high variance | Balanced | High bias, low variance |
| Sensitive to noise | Robust | Too simple |

**How to choose K:**
- Odd K for binary classification (avoids ties)
- Try K = 1, 3, 5, 7, 11, 15 with cross-validation
- $K \approx \sqrt{N}$ is a common rule of thumb
- For 1000 training points: try K = 5-30

---

## Making KNN Fast: Spatial Data Structures

**Brute force:** Compute distance to all N training points. $O(N \cdot d)$ per prediction. For 1M training points with 100 features = 100M operations PER prediction. Way too slow!

**KD-Tree (K-Dimensional Tree):**

Partitions space into regions, so you only need to search nearby regions:

```
Full space:                    KD-Tree partition:
                               
  A  A   B                     |  A  A  |  B    |
  A    B  B                    |  A     | B  B  |
    A     B                    |---A----+----B--|
  A   A                        |  A   A |       |
                               |        |       |

Instead of checking all points,
only check the relevant partition.
```

**Ball Tree:** Similar idea but uses spherical regions instead of rectangular. Better for high dimensions.

**sklearn chooses automatically:**
- `algorithm='auto'` picks the best based on data
- < 30 features: KD-Tree
- 30+ features: Ball Tree or brute force
- < 20 training points: brute force is fine

---

## The Curse of Dimensionality: KNN's Kryptonite

KNN relies on the concept of "nearness." In high dimensions, **everything is far from everything else**:

```
Dimensions:   Points needed to "fill" the space:
1D            10 points cover [0,1] nicely
2D            100 points = 10x10 grid
3D            1,000 points = 10x10x10 grid
10D           10,000,000,000 points = 10^10
100D          10^100 points (more than atoms in the universe!)
```

In 100 dimensions, even the "nearest" neighbor might be very far away. The concept of "close" loses meaning.

**Practical impact:**
- KNN works great in 2-20 dimensions
- Starts degrading at 20-50 dimensions
- Essentially useless above 100 dimensions (without dimensionality reduction)

**Fix:** Apply PCA or feature selection to reduce dimensions before using KNN.

---

## KNN for Regression

Same algorithm, but average instead of vote:

```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

# Prediction: average of 5 nearest neighbors' target values
# K=5 neighbors have prices: $300K, $320K, $280K, $310K, $290K
# Prediction: ($300K + $320K + $280K + $310K + $290K) / 5 = $300K
```

**Limitation:** KNN regression predicts within the convex hull of training data. It **cannot extrapolate**:

```
Training data range: $200K - $500K
KNN will NEVER predict $600K, even if the pattern clearly suggests it should.
(Average of neighbors is always between the min and max of those neighbors)
```

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Not scaling features | Distance dominated by large-scale features | StandardScaler or MinMaxScaler |
| Too many features (>50) | Curse of dimensionality: all neighbors equally far | PCA or feature selection first |
| K=1 | Memorizes training data, overfits | Use K=5+ with cross-validation |
| Very large K | Predicts the majority class everywhere | Keep K small relative to class counts |
| Imbalanced classes | K neighbors dominated by majority class | Use distance weighting (`weights='distance'`) |
| Large training set | Prediction is very slow | Use KD-Tree, reduce training set, or use approximate NN |
| Expecting fast training | KNN is "lazy" -- fast to train but slow to predict | Consider other models for real-time applications |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it learns** | Nothing! Stores the entire training dataset |
| **How it "learns"** | Just memorizes the data (+ builds spatial index) |
| **Prediction** | Find K nearest neighbors, vote or average |
| **Key parameter** | K (number of neighbors). Small K = complex, large K = simple |
| **Feature scaling?** | REQUIRED (distance-based model) |
| **Training speed** | Instant (just stores data) |
| **Prediction speed** | Slow (must search through all/many training points) |
| **Interpretable?** | Somewhat -- "these are the 5 most similar cases" |
| **When to use** | Small-medium datasets, low dimensions, recommendation systems, anomaly detection |
| **When NOT to use** | Large datasets, high dimensions, need fast prediction, need extrapolation |
