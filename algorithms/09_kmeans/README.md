# 09 — k-Means Clustering

## Overview

k-Means is the most widely used **unsupervised learning** algorithm. It partitions $n$ data points into $k$ clusters, where each point belongs to the cluster with the nearest **centroid** (mean). It is an iterative algorithm that alternates between **assigning** points to clusters and **updating** centroids.

Unlike supervised algorithms (classification/regression), clustering has **no target labels** — the algorithm discovers structure in the data on its own.

---

## The Algorithm

### Lloyd's Algorithm (Standard k-Means)

**Input**: Data $X = \{x_1, ..., x_n\}$, number of clusters $k$

1. **Initialize** $k$ centroids $\mu_1, ..., \mu_k$ (randomly or via k-means++)
2. **Repeat** until convergence:
   - **Assignment step**: Assign each point to its nearest centroid
     $$c_i = \arg\min_j \| x_i - \mu_j \|^2$$
   - **Update step**: Recompute each centroid as the mean of its assigned points
     $$\mu_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i$$
3. **Convergence**: Stop when assignments don't change (or centroids move less than a threshold)

### Objective Function (Inertia)

k-Means minimizes the **Within-Cluster Sum of Squares (WCSS)**, also called **inertia**:

$$J = \sum_{j=1}^{k} \sum_{x_i \in S_j} \| x_i - \mu_j \|^2$$

This is a non-convex optimization — k-Means finds a **local minimum**, not necessarily the global one.

---

## Initialization Methods

Initialization is **critical** — bad starting centroids lead to poor local minima.

### Random Initialization

Pick $k$ random data points as initial centroids. Simple but unreliable — can produce vastly different results across runs.

### k-Means++ (Default in scikit-learn)

A smarter initialization that spreads initial centroids apart:

1. Choose the first centroid $\mu_1$ uniformly at random from the data
2. For each subsequent centroid $\mu_j$:
   - Compute $D(x_i)$ = distance from each point to its nearest existing centroid
   - Choose $\mu_j$ with probability proportional to $D(x_i)^2$
3. Repeat until all $k$ centroids are chosen

**k-Means++ guarantees** an expected approximation ratio of $O(\log k)$ to the optimal solution, and empirically converges much faster.

---

## Choosing K — The Elbow Method

Since k-Means requires $k$ as input, we need a method to select it.

### Elbow Method

Plot **inertia** (WCSS) vs $k$. The "elbow" — where the rate of decrease sharply changes — suggests a good $k$.

- For $k=1$: maximum inertia (all points in one cluster)
- For $k=n$: zero inertia (each point is its own cluster)
- The elbow balances compression vs. fidelity

### Silhouette Score

For each point $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = mean distance to other points in the **same cluster** (cohesion)
- $b(i)$ = mean distance to points in the **nearest other cluster** (separation)

Properties:
- $s(i) \in [-1, 1]$
- $s(i) \approx 1$: well-clustered
- $s(i) \approx 0$: on the boundary
- $s(i) < 0$: likely in the wrong cluster

The **average silhouette score** across all points measures overall clustering quality.

---

## Convergence Properties

- k-Means is **guaranteed to converge** (WCSS decreases monotonically)
- Convergence is typically **fast** — often within 10-20 iterations
- But convergence is to a **local minimum**, not global
- **Solution**: Run multiple times with different initializations (`n_init` parameter in sklearn, default=10)

---

## Variants of k-Means

### Mini-Batch k-Means

Uses random subsets (mini-batches) of the data for each update step:
- Much faster on large datasets
- Slightly worse cluster quality
- Trades accuracy for speed

### k-Medians

Uses **median** instead of mean for centroid updates. More robust to outliers.

### k-Medoids (PAM)

Constrains centroids to be **actual data points** (medoids). More robust but slower ($O(n^2 k)$).

### Bisecting k-Means

Hierarchical approach: start with 1 cluster, repeatedly split the worst cluster until reaching $k$.

---

## Assumptions & Limitations

### k-Means Assumes:
1. **Spherical clusters** — equal variance in all directions
2. **Similar-sized clusters** — roughly equal number of points
3. **Well-separated clusters** — clear gaps between clusters
4. **Euclidean geometry** — distance-based similarity

### Fails When:
- Clusters are **non-convex** (e.g., moons, rings) → use DBSCAN or spectral clustering
- Clusters have **very different sizes** → smaller clusters get absorbed
- Clusters have **very different densities** → dense clusters may split
- Data has **outliers** → centroids get pulled toward outliers

---

## Feature Scaling

Like all distance-based algorithms, k-Means is **sensitive to feature scale**. Features with larger ranges will dominate the distance computation.

**Always standardize features** before k-Means:
$$x' = \frac{x - \mu}{\sigma}$$

---

## Computational Complexity

| Operation | Complexity |
|-----------|-----------|
| One iteration | $O(n \cdot k \cdot d)$ |
| Full algorithm | $O(n \cdot k \cdot d \cdot i)$ |
| k-Means++ init | $O(n \cdot k \cdot d)$ |

Where $n$ = samples, $k$ = clusters, $d$ = features, $i$ = iterations.

k-Means is **very fast** — scales linearly with data size.

---

## What This Implementation Covers

1. **From-scratch k-Means** — random init, k-means++ init, iterative assignment/update
2. **Scikit-learn KMeans** comparison
3. **Elbow method** — inertia vs k plot
4. **Silhouette analysis** — per-cluster silhouette plots
5. **Initialization comparison** — random vs k-means++
6. **Iteration-by-iteration visualization** — watch clusters form
7. **Failure cases** — non-convex shapes, different sizes/densities
8. **Mini-Batch k-Means** — speed comparison on larger data
9. **Real data clustering** — Iris (with known labels for validation)
10. **Feature scaling impact**

---

## How to Run

```bash
cd 09_kmeans
python kmeans.py
```

All plots are saved to `plots/`.
