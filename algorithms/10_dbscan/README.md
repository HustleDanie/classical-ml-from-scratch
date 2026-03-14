# 10 — DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Overview

DBSCAN is a **density-based** clustering algorithm that groups together points that are closely packed and marks points in low-density regions as **outliers (noise)**. Unlike k-Means, DBSCAN:

- Does **not require specifying k** (number of clusters) in advance
- Can find **arbitrarily shaped clusters** (non-convex, elongated, nested)
- Has a **built-in notion of noise** — outlier detection for free
- Is robust to outliers

It was introduced by Martin Ester et al. in 1996 and remains one of the most cited clustering algorithms.

---

## Core Concepts

### Definitions

Given parameters **ε (eps)** and **MinPts (min_samples)**:

| Term | Definition |
|------|-----------|
| **ε-neighborhood** | The set of all points within distance ε from a point $p$: $N_\varepsilon(p) = \{q \in D \mid d(p, q) \leq \varepsilon\}$ |
| **Core Point** | A point $p$ with at least MinPts points in its ε-neighborhood: $|N_\varepsilon(p)| \geq \text{MinPts}$ |
| **Border Point** | Not a core point, but within ε of a core point |
| **Noise Point** | Neither a core point nor a border point (outlier) |

### Reachability

- **Directly density-reachable**: Point $q$ is directly density-reachable from $p$ if $p$ is a core point and $q \in N_\varepsilon(p)$
- **Density-reachable**: $q$ is density-reachable from $p$ if there exists a chain of core points connecting them
- **Density-connected**: $p$ and $q$ are density-connected if there exists a point $o$ such that both are density-reachable from $o$

A **cluster** is a maximal set of density-connected points.

---

## The Algorithm

```
DBSCAN(D, ε, MinPts):
    label = 0
    for each unvisited point p in D:
        mark p as visited
        N = ε-neighborhood of p
        if |N| < MinPts:
            mark p as NOISE
        else:
            label += 1
            expand_cluster(p, N, label, ε, MinPts)

expand_cluster(p, N, label, ε, MinPts):
    assign p to cluster label
    for each point q in N:
        if q is unvisited:
            mark q as visited
            N' = ε-neighborhood of q
            if |N'| >= MinPts:
                N = N ∪ N'  (add q's neighbors to explore)
        if q is not yet assigned to any cluster:
            assign q to cluster label
```

### Key Properties:
- Each cluster contains at least one core point
- A cluster = all core points that are density-reachable from each other + all border points within their ε-neighborhoods
- Noise points don't belong to any cluster (label = -1)

---

## Parameters

### ε (eps) — Neighborhood Radius

- **Too small**: everything becomes noise
- **Too large**: everything merges into one cluster
- Controls the **scale** of density

### MinPts (min_samples) — Density Threshold

- **Too small**: noise gets included in clusters
- **Too large**: small valid clusters become noise
- Rule of thumb: **MinPts ≥ d + 1** (where d = dimensionality), commonly **MinPts = 2 × d**
- For 2D data: MinPts = 4 is a common starting point

### How to Choose ε — The k-Distance Graph

1. For each point, compute the distance to its k-th nearest neighbor (k = MinPts)
2. Sort these distances in ascending order
3. Plot the sorted distances
4. The **elbow** in this plot suggests a good ε

The intuition: within a cluster, k-distances are small; for noise, k-distances jump up.

---

## DBSCAN vs k-Means

| Property | k-Means | DBSCAN |
|----------|---------|--------|
| **Clusters found** | Must specify k | Discovers k automatically |
| **Cluster shape** | Spherical only | Arbitrary shapes |
| **Noise handling** | None (assigns all points) | Built-in outlier detection |
| **Cluster sizes** | Tends toward equal sizes | Handles varying sizes |
| **Density** | Assumes uniform density | Finds dense regions |
| **Parameters** | k | ε, MinPts |
| **Deterministic** | No (depends on init) | Yes (core points always same) |
| **Scalability** | Very fast O(nkdi) | O(n²) worst case, O(n log n) with index |

---

## Strengths & Weaknesses

### Strengths
- No need to specify number of clusters
- Can find clusters of arbitrary shape
- Robust to outliers (built-in noise detection)
- Deterministic (for core points; border points may vary)
- Only two parameters (ε, MinPts)

### Weaknesses
- Struggles with **varying densities** — can't have one ε for clusters of different densities
- **Not suitable for high-dimensional data** — "curse of dimensionality" makes density meaningless
- Sensitive to ε parameter — small changes can produce very different results
- Doesn't assign cluster labels to new points (no `predict` for unseen data)
- O(n²) without spatial indexing

---

## Variants

### HDBSCAN (Hierarchical DBSCAN)
- Extends DBSCAN to handle **varying densities**
- Builds a hierarchy of clusterings for all ε values
- Extracts the most persistent clusters
- Only requires MinPts (no ε needed)
- Generally preferred over DBSCAN in practice

### OPTICS (Ordering Points To Identify Clustering Structure)
- Produces an **ordering** of points that captures density structure
- Can extract DBSCAN-like clusters for any ε from a single run
- Handles varying densities better than DBSCAN

---

## Computational Complexity

| Method | Time | Space |
|--------|------|-------|
| **Brute force** | O(n²) | O(n²) |
| **With KD-tree** | O(n log n) avg | O(n) |
| **With Ball-tree** | O(n log n) avg | O(n) |

Scikit-learn uses spatial indexing (KD-tree or Ball-tree) by default when possible.

---

## What This Implementation Covers

1. **From-scratch DBSCAN** — ε-neighborhood, core/border/noise classification, cluster expansion
2. **Scikit-learn DBSCAN** comparison
3. **k-Distance graph** for ε selection
4. **eps parameter impact** — same data, different ε
5. **min_samples parameter impact**
6. **Non-convex data** — moons, circles, complex shapes
7. **Noise detection visualization** — identifying outliers
8. **DBSCAN vs k-Means** — side-by-side on same datasets
9. **Varying density problem** — DBSCAN's limitation
10. **Real data clustering** — Iris with noise detection

---

## How to Run

```bash
cd 10_dbscan
python dbscan.py
```

All plots are saved to `plots/`.
