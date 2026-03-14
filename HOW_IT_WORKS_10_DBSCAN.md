# How It Works: DBSCAN (Density-Based Spatial Clustering)

> DBSCAN finds clusters by looking for **dense regions** of points separated by sparse regions. Unlike K-Means, it discovers the number of clusters automatically, can find clusters of any shape, and labels outliers as noise.

---

## The One-Sentence Idea

DBSCAN finds clusters by starting at a dense point, then **expanding outward** to include all nearby dense points, continuing until it reaches sparse regions where the cluster ends -- points in sparse regions are labeled as noise (outliers).

---

## Intuition: Finding Crowds at a Festival

Imagine looking at a festival from above. You see:
- A dense crowd near the main stage (cluster 1)
- A dense crowd near the food court (cluster 2)
- A few scattered people walking between areas (noise)

DBSCAN identifies crowds the way your eyes do: a crowd is a dense group of people where each person has many others nearby. Isolated individuals between crowds are noise.

```
  ** *                                      * 
 * ** *              * *                  
  * * *             * * *                 
   * *              * * *                  *
                     * *                       *
Cluster 1           Cluster 2            Noise (isolated points)
(dense region)      (dense region)       (sparse region)
```

---

## The Two Key Parameters

DBSCAN needs just 2 parameters:

### 1. `eps` (epsilon) -- The Neighborhood Radius

"How close must two points be to be considered neighbors?"

$$\text{Neighborhood of } \mathbf{x}: N_\epsilon(\mathbf{x}) = \{\mathbf{x}' : d(\mathbf{x}, \mathbf{x}') \leq \epsilon\}$$

Think of drawing a circle of radius $\epsilon$ around each point.

### 2. `min_samples` -- Minimum Points for Density

"How many neighbors must be within $\epsilon$ for a point to be considered part of a dense region?"

If a point has at least `min_samples` points within its $\epsilon$-neighborhood (including itself), it's a **core point** (definitely in a cluster).

```
eps = 1.0, min_samples = 4

Point A has 6 neighbors within eps:      Point B has 2 neighbors within eps:
     * *                                        *
    *[A]* *   -> CORE POINT (6 >= 4)           [B]*    -> NOT a core point (2 < 4)
     * *                                  
```

---

## The Three Types of Points

```
eps = radius of circles, min_samples = 4

     *  *                                     *
    * [C] *      [C] = Core point (5+ neighbors in circle)
     * * *

                 [B] = Border point (< 4 neighbors,
        [B]        but within eps of a core point)
       /
      /  
    * [C] *      Reachable from a core point.
     * *

                              *
                                  [N] = Noise point (< 4 neighbors,
                                    NOT reachable from any core point)
```

1. **Core point**: Has $\geq$ `min_samples` points within $\epsilon$. Forms the backbone of a cluster.
2. **Border point**: Has $<$ `min_samples` neighbors, but is within $\epsilon$ of a core point. On the edge of a cluster.
3. **Noise point**: Has $<$ `min_samples` neighbors and is NOT near any core point. An outlier.

---

## The Algorithm: Step by Step

```
Input: Dataset X, eps, min_samples
Output: Cluster labels for each point (or -1 for noise)

1. For each unvisited point P:
   a. Mark P as visited
   b. Find all points within eps distance of P (its neighborhood)
   c. If |neighborhood| < min_samples:
        Mark P as NOISE (temporarily -- may later become a border point)
   d. If |neighborhood| >= min_samples:
        P is a CORE POINT
        Create a new cluster C
        Add P to cluster C
        
        For each point Q in P's neighborhood:
          If Q is not yet visited:
            Mark Q as visited
            Find Q's neighborhood
            If |Q's neighborhood| >= min_samples:
              Q is also a core point
              Add Q's neighborhood to the expansion list
          If Q is not yet in any cluster:
            Add Q to cluster C

2. Any point still labeled as NOISE at the end stays as noise (label = -1)
```

### Walking Through an Example

```
Data:    A(1,1)  B(1.5,1)  C(1,1.5)  D(5,5)  E(5.5,5)  F(5,5.5)  G(3,3)
eps=1.0, min_samples=3

Step 1: Visit A(1,1)
  Neighbors within eps=1.0: {A, B, C}  (3 points)
  3 >= min_samples(3) -> A is CORE POINT
  Create Cluster 1, add A
  
  Expand to B(1.5,1):
    B's neighbors: {A, B, C}  (3 points)
    B is CORE POINT
    Add B to Cluster 1
    
  Expand to C(1,1.5):
    C's neighbors: {A, B, C}  (3 points)
    C is CORE POINT
    Add C to Cluster 1
    
  No more points to expand. Cluster 1 = {A, B, C}

Step 2: Visit D(5,5)
  Neighbors: {D, E, F}  (3 points)
  D is CORE POINT
  Create Cluster 2, add D
  Expand to E and F (both are core points)
  Cluster 2 = {D, E, F}

Step 3: Visit G(3,3)
  Neighbors: {G}  (1 point, only itself)
  1 < min_samples(3) -> G is NOISE (label = -1)

Final: Cluster 1 = {A, B, C}, Cluster 2 = {D, E, F}, Noise = {G}
```

---

## What Happens Behind the Scenes: `.fit()` and `.predict()`

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.5,           # neighborhood radius
    min_samples=5,     # minimum points for core point
    metric='euclidean' # distance metric
)

# .fit(X) does this:
# 1. For each point, find all neighbors within eps
#    (uses Ball Tree or KD-Tree for efficiency)
# 2. Identify core points (>= min_samples neighbors)
# 3. For each unvisited core point, start a new cluster
#    and expand by following connected core points
# 4. Assign border points to the cluster of their nearest core point
# 5. Label remaining points as noise (-1)
model.fit(X)

print(model.labels_)            # cluster label per point (-1 = noise)
print(model.core_sample_indices_)  # indices of core points
print(model.components_)        # coordinates of core points

# Number of clusters found:
n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)

# IMPORTANT: DBSCAN has no .predict() method in sklearn!
# It can't easily classify new points because it doesn't store centroids.
# Workaround: use nearest core point assignment or train a classifier on the labels.
```

**Why no `.predict()`?** DBSCAN clusters are defined by density connectivity, not by centroids. A new point's cluster depends on whether it connects to existing core points, which requires re-running the algorithm. In practice, you can classify new points by assigning them to the cluster of their nearest core point.

---

## The Full Pipeline

```
TRAINING (.fit):

Input: 500 points, eps=1.0, min_samples=5

Phase 1: Compute neighborhoods
  For each of 500 points, find all points within eps=1.0
  (Using KD-Tree, this is O(N log N) instead of O(N^2))

Phase 2: Identify core points
  Points with 5+ neighbors -> 320 core points
  Points with < 5 neighbors -> 180 non-core points

Phase 3: Cluster expansion
  Start at core point #1 -> expand -> Cluster 0 (152 points)
  Start at unvisited core point #153 -> expand -> Cluster 1 (98 points)
  Start at unvisited core point -> expand -> Cluster 2 (120 points)
  
Phase 4: Assign border points
  45 border points assigned to nearest cluster
  
Phase 5: Label noise
  85 points with no core point within eps -> Noise (label = -1)

Result: 3 clusters + 85 noise points
```

---

## Choosing Parameters: eps and min_samples

### Choosing eps: The K-Distance Plot

1. For each point, compute the distance to its K-th nearest neighbor (where K = `min_samples`)
2. Sort these distances from smallest to largest
3. Plot them -- look for an "elbow"

```
K-distance
  |
  |                                          *
  |                                      * *
  |                                   * *
  |                              * * *
  |                         * * *
  |                   * * * *
  |      * * * * * * *
  | * * *
  +--|--|--|--|--|--|--|--|--|--|--|--|
     0  50  100  150  200  250  300  350  400
                 Points (sorted)
                 
  "Elbow" at K-distance ~2.5 -> choose eps ~2.5
```

Points within clusters have small K-distances (many nearby neighbors). Points in sparse regions have large K-distances. The elbow is where cluster density transitions to sparse.

### Choosing min_samples: Rules of Thumb

- **min_samples >= dimensions + 1** (at minimum)
- **min_samples = 2 * dimensions** (common choice for higher-dimensional data)
- For 2D data: min_samples = 4-5 is a good start
- Larger min_samples = less sensitive to noise but may miss small clusters

### Effect of Parameters

```
Small eps, high min_samples:         Large eps, low min_samples:
(strict density requirement)          (relaxed density requirement)

  *  *  *                              [  *  *  *              ]
  *  *  -> Cluster 1                   [  *  *                 ]
                                       [         * * *         ]
       * * *                           [         * * *         ]
       * * * -> Cluster 2              [         * * *         ]
                                       [   * * *               ]
  * * *     *  <- noise!               [   * * *   *           ]
  * * * -> Cluster 3                   [   * * *               ]
                                       
Many small clusters + noise.           One giant cluster, no noise.
Too strict.                            Too relaxed.
```

---

## DBSCAN vs. K-Means: When to Use Which

```
K-MEANS WINS:                       DBSCAN WINS:

Spherical clusters:                 Non-spherical clusters:
  * * *      * * *                   * * * * * * * *
  * * * *    * * * *                 *               *
  * * *      * * *                   * * * * * * * * *
  K-Means: perfect!                 K-Means: fails!
  DBSCAN: also works.               DBSCAN: handles it!

Same-size clusters:                 Different-size clusters:
  * * *        * * *                * * * * * * * * *
  * * *        * * *                * * * * * * * * *
                                    * * * * * * * * *     *
  K-Means: perfect!                                      * *
  DBSCAN: also works.              K-Means: splits big one!
                                   DBSCAN: finds both!

Need to specify K in advance:      Don't know how many clusters:
  K-Means: yes (must specify K)    K-Means: must guess K
  DBSCAN: no (finds K itself)      DBSCAN: discovers K automatically!

No outliers:                       Data has outliers/noise:
  K-Means: fine                    K-Means: centroids pulled by outliers
  DBSCAN: labels some as noise    DBSCAN: outliers detected as noise!
```

| Feature | K-Means | DBSCAN |
|---------|---------|--------|
| Must specify K? | Yes | No (discovered automatically) |
| Cluster shape | Spherical only | Any shape |
| Handles outliers? | No (assigns all points to clusters) | Yes (labels noise as -1) |
| Cluster sizes | Assumes similar sizes | Can find different sizes |
| Speed | Fast ($O(NKd)$) | Can be slow ($O(N^2)$ worst case) |
| Reproducibility | Depends on initialization | Deterministic (same result every time for core points) |
| High dimensions | Works (but distance loses meaning) | Struggles more (density is harder to define) |
| Parameters | K (number of clusters) | eps (radius) + min_samples (density) |

---

## Handling Scale and High Dimensions

### Feature Scaling

Like all distance-based methods, DBSCAN **requires feature scaling**:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X_scaled)
```

### High Dimensions

DBSCAN suffers from the curse of dimensionality (same as KNN):
- In high dimensions, all points are roughly equally far apart
- The concept of "dense neighborhood" loses meaning
- eps becomes hard to set (distances converge to similar values)

**Fix:** Apply PCA or other dimensionality reduction before DBSCAN.

---

## Variants: HDBSCAN (Hierarchical DBSCAN)

HDBSCAN improves on DBSCAN by:
1. **Not requiring eps** -- it finds clusters across multiple density scales
2. Handling clusters of **varying density** (DBSCAN needs uniform density)
3. Providing a **cluster hierarchy** (dendrogram)

```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
labels = clusterer.fit_predict(X)

# Also provides confidence scores (-1 still means noise)
print(clusterer.probabilities_)  # how confident each assignment is
```

**When to use HDBSCAN over DBSCAN:** When clusters have different densities, or when you don't want to tune eps.

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Not scaling features | Distance dominated by large-scale feature | StandardScaler |
| eps too small | Everything is noise (no clusters found) | Use K-distance plot to choose eps |
| eps too large | Everything is one giant cluster | Use K-distance plot |
| min_samples too high | Only very dense regions form clusters, rest is noise | Start with 2*d, reduce if too much noise |
| High-dimensional data | All distances are similar, density meaningless | PCA first, then DBSCAN |
| Clusters of varying density | Dense cluster found, sparse cluster labeled as noise | Use HDBSCAN instead |
| Expecting .predict() | DBSCAN has no native predict for new points | Use nearest core point or train a classifier on labels |
| Very large dataset | O(N^2) is too slow | Use sample + assign, or HDBSCAN |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Type** | Unsupervised clustering (density-based) |
| **What it learns** | Core points, cluster assignments, noise labels |
| **How it learns** | Expands clusters from core points by following dense connectivity |
| **Key parameters** | eps (neighborhood radius), min_samples (density threshold) |
| **Output** | Cluster labels (0, 1, 2, ...) and noise label (-1) |
| **Feature scaling?** | REQUIRED (distance-based) |
| **Discovers K?** | Yes -- number of clusters is found automatically |
| **Handles outliers?** | Yes -- labels them as noise (-1) |
| **Cluster shapes** | Any shape (non-convex, elongated, ring-shaped, etc.) |
| **When to use** | Unknown number of clusters, non-spherical clusters, need outlier detection |
| **When NOT to use** | Very high dimensions, varying density clusters (use HDBSCAN), need to predict new points easily |
