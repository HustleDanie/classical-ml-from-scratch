# How It Works: K-Means Clustering

> K-Means is an **unsupervised** algorithm -- it finds groups in data WITHOUT labels. You give it data and a number K, and it discovers K natural groups (clusters).

---

## The One-Sentence Idea

K-Means places K "center points" (centroids) in your data, assigns each data point to the nearest centroid, then moves each centroid to the middle of its assigned points, repeating until nothing changes.

---

## Intuition: Sorting Balls by Color (Blindfolded)

Imagine you have 100 balls of 3 colors on a table, but you're blindfolded and can only feel their position. Here's what K-Means does:

1. **Drop 3 markers** randomly on the table (initial centroids)
2. **Assign each ball** to the nearest marker
3. **Move each marker** to the center of its balls
4. **Repeat** steps 2-3 until the markers stop moving

After a few iterations, each marker ends up in the middle of a color group!

---

## The Math: Step by Step

### Step 1: Choose K (Number of Clusters)

You must decide K beforehand. This is the biggest decision. (We'll discuss how to choose K later.)

### Step 2: Initialize K Centroids

Place K centroids $\mu_1, \mu_2, ..., \mu_K$ somewhere in the feature space.

**Random initialization:** Pick K random data points as initial centroids.

```
Data points (2D):                     After placing K=3 random centroids:

  *  *  *                              *  *  *
  *  * *                               *  * *      C2
    *                                    *       
                                                 
       * * *                                * * *
       * * *                                * * *
       *  *                                 *  *
                        C1
  * *  *                              * *  *
  * * *                               * * *
    *  *                                *  *     C3
```

### Step 3: Assign Each Point to Nearest Centroid

For each data point $\mathbf{x}_i$, find the closest centroid:

$$\text{assignment}_i = \arg\min_k ||\mathbf{x}_i - \mu_k||^2$$

(Euclidean distance, squared for convenience.)

```
After assignment (each point colored by its cluster):

  A  A  A
  A  A A
    A                     C2(B)
                                 
       B B B
       B B B
       B  B
             C1(A)
  C C  C
  C C C
    C  C             C3(C)
```

### Step 4: Move Centroids to Cluster Centers

$$\mu_k = \frac{1}{|S_k|} \sum_{\mathbf{x}_i \in S_k} \mathbf{x}_i$$

Each centroid moves to the **mean** (average position) of all points assigned to it.

```
Centroids move to cluster centers:

  A  A  A
  A  CA A           CA = centroid of cluster A (moved to center of A's)
    A
                                 
       B CB B       CB = centroid of cluster B (moved to center of B's)
       B B B
       B  B

  C C  C
  C CC C            CC = centroid of cluster C (moved to center of C's)
    C  C
```

### Step 5: Repeat Steps 3-4 Until Convergence

```
Iteration 1: Assign -> Move centroids
Iteration 2: Re-assign (some points change cluster) -> Move centroids
Iteration 3: Re-assign -> Move centroids (small movements now)
...
Iteration 7: No points change cluster -> CONVERGED! Stop.
```

Typically converges in 10-50 iterations.

---

## The Objective Function: What K-Means Minimizes

K-Means minimizes the **Within-Cluster Sum of Squares (WCSS)** -- also called **inertia**:

$$\text{WCSS} = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in S_k} ||\mathbf{x}_i - \mu_k||^2$$

In English: the total squared distance from every point to its assigned centroid. Smaller WCSS = tighter clusters.

Each iteration is **guaranteed** to decrease or maintain WCSS (never increase). This is why K-Means always converges.

**Proof sketch:**
- Step 3 (assign to nearest centroid): assigns each point to the centroid that minimizes its distance -- this can only decrease total distance
- Step 4 (move centroid to mean): the mean is the point that minimizes the sum of squared distances -- this can only decrease total distance

---

## The Full Pipeline: What Happens Behind the Scenes

```
TRAINING (.fit):

Input: 1000 data points, K=3

Iteration 0: Random centroids
   C1=(2,3), C2=(8,7), C3=(5,1)
   
Iteration 1:
   Assign 1000 points to nearest centroid:
     Cluster 1: 280 points (near C1)
     Cluster 2: 350 points (near C2)
     Cluster 3: 370 points (near C3)
   
   Move centroids:
     C1 = mean of 280 points = (1.8, 3.2)
     C2 = mean of 350 points = (7.5, 6.8)
     C3 = mean of 370 points = (5.2, 1.5)
   
   WCSS = 4250

Iteration 2:
   Re-assign: 23 points change cluster
   Move centroids: C1=(1.6, 3.5), C2=(7.8, 7.0), C3=(5.0, 1.3)
   WCSS = 3890 (decreased!)

Iteration 3:
   Re-assign: 8 points change
   WCSS = 3820

...

Iteration 8:
   Re-assign: 0 points change -> CONVERGED!
   Final WCSS = 3780
   Final centroids stored.


PREDICTION (.predict for new points):

New point: (6.0, 5.5)
   Distance to C1 = sqrt((6-1.6)^2 + (5.5-3.5)^2) = 4.82
   Distance to C2 = sqrt((6-7.8)^2 + (5.5-7.0)^2) = 2.28  <-- closest!
   Distance to C3 = sqrt((6-5.0)^2 + (5.5-1.3)^2) = 4.36
   
   Assign to Cluster 2
```

---

## The Initialization Problem

**K-Means can converge to different solutions depending on initial centroid placement:**

```
Good initialization:              Bad initialization:

  C1                                  C1 C2
  *  *  *                            *  *  *
  *  * *                             *  * *

       * * *    C2                        * * *
       * * *                              * * *

  * *  *                             * *  *
  * * *    C3                        * * *      C3
  
Finds 3 natural clusters.          Merges two clusters,
                                   splits one in half!
```

### Solution: K-Means++ Initialization

Instead of random centroids, K-Means++ spaces them out deliberately:

1. Pick the first centroid randomly from the data
2. For each remaining point, compute its distance to the nearest existing centroid
3. Pick the next centroid with probability proportional to distance squared
   (far-away points are more likely to be picked)
4. Repeat until K centroids are placed

This ensures centroids start spread out, not bunched together.

**sklearn uses K-Means++ by default** (`init='k-means++'`).

Additionally, sklearn runs K-Means **10 times** (`n_init=10`) with different initializations and keeps the best result (lowest WCSS).

---

## Choosing K: How Many Clusters?

### Method 1: The Elbow Method

Run K-Means for K=1,2,3,...,10 and plot WCSS:

```
WCSS
  |
  |*
  |  *
  |    *
  |      *  *  *  *  *  *  *
  |
  +--|--|--|--|--|--|--|--|--|
     1  2  3  4  5  6  7  8  9

"Elbow" at K=4 (WCSS stops decreasing dramatically)
```

The "elbow" is where adding more clusters doesn't help much.

**Problem:** Elbows are often ambiguous. Is it K=3 or K=4?

### Method 2: Silhouette Score

For each point, the silhouette score measures how well it fits in its cluster:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

- $a(i)$ = average distance to other points in the SAME cluster (cohesion)
- $b(i)$ = average distance to points in the NEAREST other cluster (separation)
- $s(i)$ ranges from -1 to +1:
  - +1: point is far from other clusters, well inside its cluster (great!)
  - 0: point is on the boundary between clusters
  - -1: point is probably in the wrong cluster

Average silhouette score across all points = overall clustering quality.

```python
from sklearn.metrics import silhouette_score

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"K={k}: Silhouette={score:.3f}")

# K=2: 0.45
# K=3: 0.52  <-- best!
# K=4: 0.48
# K=5: 0.39
```

### Method 3: Domain Knowledge

Often the best approach. If you're segmenting customers, marketing might say "we can handle 4-6 distinct campaigns." So try K=4,5,6 and pick the one that makes the most business sense.

---

## What Happens in `.fit()` and `.predict()` in Code

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=3,         # K=3 clusters
    init='k-means++',     # smart initialization (default)
    n_init=10,            # run 10 times, keep best
    max_iter=300,         # max iterations per run
    random_state=42
)

# .fit(X) does this:
# Repeat 10 times (n_init):
#   1. Initialize 3 centroids using K-Means++
#   2. Repeat up to 300 times:
#      a. Assign each point to nearest centroid
#      b. Move each centroid to mean of its points
#      c. If no assignments changed, break
#   3. Record final WCSS
# Keep the run with lowest WCSS
model.fit(X)

print(model.cluster_centers_)  # 3 centroid coordinates
print(model.labels_)           # cluster assignment for each training point
print(model.inertia_)          # WCSS (lower = tighter clusters)
print(model.n_iter_)           # number of iterations in best run

# .predict(X_new) does this:
# For each new point, compute distance to each centroid
# Assign to nearest centroid
new_labels = model.predict(X_new)

# .transform(X) does this:
# Returns distance from each point to each centroid
# Shape: (n_samples, n_clusters) -- useful as features for other models
distances = model.transform(X_new)
```

---

## Assumptions and Limitations

### What K-Means Assumes About Your Data

1. **Clusters are spherical** (roughly circular in 2D)
2. **Clusters are similar in size**
3. **Clusters are similar in density**

```
K-Means works well:               K-Means fails:

  * * *                            * * * * * * * * *
  * * * *      * * *               *               *
  * * *        * * * *             * * * * * * * * *
               * * *
                                   * *
  Spherical, similar size.         * *     Non-spherical shapes!
                                   
                                   
                * * *              
  * * * *       * * *              Large cluster:  Tiny cluster:
  * * * * *     * * *              * * * * * * *     *
  * * * *                          * * * * * * *     *
                                   * * * * * * *
                                   
  Similar density.                 Different sizes/densities!
                                   K-Means splits the big one
                                   and merges the small ones.
```

### Key Limitations

| Limitation | Why | Alternative |
|------------|-----|------------|
| Must specify K in advance | K-Means can't determine optimal K itself | Use Elbow/Silhouette, or try DBSCAN |
| Assumes spherical clusters | Uses Euclidean distance, measures from center | DBSCAN or Gaussian Mixture Models |
| Sensitive to outliers | Outliers pull centroids away | Remove outliers first, or use K-Medoids |
| Sensitive to scale | Feature with large range dominates distance | StandardScaler first |
| Can find local optima | Different starts give different results | Use n_init=10+ |
| Can't handle non-convex shapes | Centroid-based assignment forces convex boundaries | DBSCAN |

---

## Feature Scaling: Critical!

Just like KNN and SVM, K-Means uses distances. Unscaled features cause problems:

```python
# BAD: Income dominates
X = [[25, 30000], [30, 50000], [35, 80000]]
# Distance between point 1 and 2:
# sqrt((25-30)^2 + (30000-50000)^2) = 20000
# Age contributes 25 out of 400,000,025. Irrelevant!

# GOOD: After scaling
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)
# Now age and income contribute equally to distance
```

---

## Variants: Mini-Batch K-Means

For very large datasets (millions of points), standard K-Means is slow (must compute distance from every point to every centroid each iteration).

**Mini-Batch K-Means:** Uses random subsets (mini-batches) for each update step:

```python
from sklearn.cluster import MiniBatchKMeans

model = MiniBatchKMeans(n_clusters=5, batch_size=1000)
model.fit(X)  # Much faster for large datasets
```

- 10-100x faster than standard K-Means
- Slightly worse WCSS (1-3% higher)
- Good enough for most applications

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Not scaling features | Clustering based on feature with largest scale | StandardScaler |
| Wrong K | Too few: merges distinct groups. Too many: splits natural groups | Elbow method + Silhouette score + domain knowledge |
| n_init=1 | May get stuck in bad local optimum | Use n_init=10 (default) or higher |
| Non-spherical clusters | K-Means draws circular boundaries around non-circular groups | Use DBSCAN or Gaussian Mixture Models |
| Outliers | Centroid gets pulled toward outlier | Remove outliers first, or use K-Medoids |
| High dimensions | Distance loses meaning (curse of dimensionality) | PCA first, then K-Means |
| Interpreting as "truth" | K-Means always finds K clusters, even in random data | Validate with silhouette score and domain knowledge |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Type** | Unsupervised (no labels needed) |
| **What it learns** | K centroid positions |
| **How it learns** | Alternates between assigning points and moving centroids (EM-like) |
| **Prediction** | Assign new point to nearest centroid |
| **Objective** | Minimize Within-Cluster Sum of Squares (WCSS / inertia) |
| **Key parameter** | K (number of clusters) |
| **Feature scaling?** | REQUIRED (distance-based) |
| **Convergence** | Guaranteed (but may be local optimum) |
| **Speed** | Fast ($O(N \cdot K \cdot d \cdot \text{iterations})$) |
| **When to use** | Roughly spherical clusters, need fast clustering, customer segmentation, image compression |
| **When NOT to use** | Clusters are non-spherical, widely different sizes, unknown number of clusters |
