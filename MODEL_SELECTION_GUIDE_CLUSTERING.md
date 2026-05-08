# Model Selection Guide for Clustering

How to choose between K-Means and DBSCAN — really one core question with several follow-ups.

---

## Step 0: Should You Cluster At All?

Before picking an algorithm, check that clustering is actually the right tool.

| Situation | Cluster? | Why |
|-----------|---------|-----|
| You have labels and want to predict them | NO | Use classification, not clustering |
| You have a number to predict | NO | Use regression |
| You want to explore unlabeled data and find natural groups | YES | Clustering is for this |
| You want to detect outliers / anomalies | MAYBE | DBSCAN does this naturally; or use Isolation Forest instead |
| You want to compress / visualize high-dim data | NO | Use PCA first; cluster afterward if needed |

**Rule:** Clustering is unsupervised. If you have a `y` column, you probably don't need it.

---

## The Core Decision

This repo covers two clustering algorithms. The choice between them comes down to **one question**:

> **Do you know how many groups you want, and are the clusters roughly round in shape?**

- **YES to both** → K-Means
- **NO to either** → DBSCAN

Everything else is nuance on top of that.

---

## The 6 Questions

### Question 1: Do you know how many clusters (k) upfront?

| Situation | Best Model | Why |
|-----------|-----------|-----|
| Yes — business says "5 customer segments" | K-Means | k is an input |
| Yes — you've used elbow / silhouette to pick k | K-Means | Same |
| No — you want the algorithm to figure it out | DBSCAN | Discovers cluster count from density |

**How to estimate k for K-Means if you don't know:**
- **Elbow method:** plot k vs inertia (within-cluster sum of squares); pick the bend.
- **Silhouette score:** plot k vs silhouette; pick the peak.
- **Domain knowledge:** "We have 4 customer tiers" → k = 4.

---

### Question 2: What shape are your clusters?

| Cluster Shape | Best Model | Why |
|--------------|-----------|-----|
| Spherical (round blobs) | K-Means | Assumes Euclidean distance to centroid — works for round shapes |
| Elongated, curved, or irregular | DBSCAN | Density-based — follows arbitrary shapes |
| Concentric rings (one ring inside another) | DBSCAN | K-Means cannot separate concentric clusters |
| Don't know | Try K-Means first; if results look wrong, switch to DBSCAN | Visualize after PCA → 2D to check |

**How to check:** reduce to 2D with PCA, scatter-plot, and see if the groups look round or weirdly shaped.

---

### Question 3: Do you have noise or outliers?

| Situation | Best Model | Why |
|-----------|-----------|-----|
| Clean data, no expected outliers | K-Means | Outlier handling not needed |
| Noisy data with outliers | DBSCAN | Labels outliers as "noise" (cluster -1) automatically |
| Outlier detection IS the goal (fraud, anomalies, defective parts) | DBSCAN | Use the noise points as your anomalies |

**Rule:** K-Means forces every point into a cluster — outliers will distort centroids. DBSCAN leaves them out as noise.

---

### Question 4: How big is the dataset?

| Dataset Size | K-Means | DBSCAN |
|--------------|---------|--------|
| < 1K rows | YES | YES |
| 1K–100K rows | YES — fast | YES |
| 100K–1M rows | YES — still fast | OK — slower (O(n log n) with index, O(n²) without) |
| > 1M rows | YES — `MiniBatchKMeans` | NO — won't scale; use HDBSCAN or sample first |

**Rule:** K-Means scales much better. For very large data, K-Means is usually the only practical option.

---

### Question 5: Are clusters of similar density?

| Density | Best Model | Why |
|---------|-----------|-----|
| Similar density across clusters | Either | Both work |
| Wildly different density (one tight cluster, one spread out) | DBSCAN with caution | Single `eps` parameter struggles with mixed densities — consider HDBSCAN |
| Same density, varying size | K-Means | Size differences are fine for K-Means |

---

### Question 6: Do you need to assign new (future) points to clusters?

| Need | Best Model | Why |
|------|-----------|-----|
| Yes — production system labels new customers | K-Means | Centroids are reusable; predict new point → nearest centroid |
| No — one-time analysis | Either | DBSCAN is fine for one-shot work |
| Yes, AND need probability of belonging | Gaussian Mixture (not in this repo) or K-Means + distance | DBSCAN doesn't naturally extend to new points |

**Rule:** K-Means trains a reusable model (the centroids). DBSCAN doesn't — assigning new points requires re-running or custom logic.

---

## How to Apply All 6 Together

Walk down the 6 questions. If 4 or more lean toward one algorithm, that's your pick. If they're split, default to **K-Means first** (faster to try, easier to tune), then switch to DBSCAN if results look wrong.

---

## 6 Case Studies

### Case Study 1: Customer Segmentation
**Data:** 50,000 customers, 8 features (age, income, recency, frequency, monetary, tenure, etc.)

| Question | Answer |
|----------|--------|
| Q1 (know k?) | Yes — marketing wants 4 segments |
| Q2 (shape?) | Likely spherical (continuous metrics) |
| Q3 (outliers?) | Some whales, but want them in a segment |
| Q4 (size?) | 50K — fine for either |
| Q5 (density?) | Similar |
| Q6 (predict new?) | Yes — assign new customers monthly |

**Winner:** K-Means with k=4

---

### Case Study 2: Fraud / Anomaly Detection in Transactions
**Data:** 200,000 transactions, 15 features

| Question | Answer |
|----------|--------|
| Q1 (know k?) | No — don't know how many fraud patterns exist |
| Q2 (shape?) | Irregular |
| Q3 (outliers?) | Outliers ARE the goal |
| Q4 (size?) | 200K — DBSCAN borderline |
| Q5 (density?) | Mixed |
| Q6 (predict new?) | Yes, but anomaly score will do |

**Winner:** DBSCAN — noise points (cluster = -1) are the suspected fraud cases

---

### Case Study 3: Image Pixel Compression
**Data:** 1,000,000 pixels (RGB), 3 features

| Question | Answer |
|----------|--------|
| Q1 (know k?) | Yes — want 16 colors |
| Q2 (shape?) | Spherical in RGB space |
| Q3 (outliers?) | None — all valid pixels |
| Q4 (size?) | 1M — only K-Means scales |
| Q5 (density?) | Mixed but K-Means handles |
| Q6 (predict new?) | Yes — apply same palette to new images |

**Winner:** K-Means with k=16 (`MiniBatchKMeans` for speed)

---

### Case Study 4: Geographic Clustering of Delivery Drop-offs
**Data:** 80,000 GPS coordinates (lat, long)

| Question | Answer |
|----------|--------|
| Q1 (know k?) | No — clusters form around real-world dense areas |
| Q2 (shape?) | Highly irregular (city shapes, coastlines) |
| Q3 (outliers?) | Yes — rural deliveries, want to flag them |
| Q4 (size?) | 80K — DBSCAN OK |
| Q5 (density?) | Wildly different (urban vs suburban) |
| Q6 (predict new?) | One-time route planning |

**Winner:** DBSCAN (or HDBSCAN if density varies wildly)

---

### Case Study 5: Document Topic Discovery (after TF-IDF + PCA)
**Data:** 30,000 documents, 50 features (PCA-reduced from TF-IDF)

| Question | Answer |
|----------|--------|
| Q1 (know k?) | No — let topics emerge |
| Q2 (shape?) | Roughly spherical in PCA space |
| Q3 (outliers?) | Yes — off-topic documents |
| Q4 (size?) | 30K — both work |
| Q5 (density?) | Similar |
| Q6 (predict new?) | Yes |

**Winner:** Try K-Means with k chosen by silhouette first; if topics look forced, switch to DBSCAN

---

### Case Study 6: Manufacturing Defect Pattern Discovery
**Data:** 10,000 sensor-snapshot rows, 25 features, no labels (we don't know which patterns are defects)

| Question | Answer |
|----------|--------|
| Q1 (know k?) | No — unknown number of failure modes |
| Q2 (shape?) | Irregular sensor signatures |
| Q3 (outliers?) | Defects ARE outliers |
| Q4 (size?) | 10K — DBSCAN fine |
| Q5 (density?) | Variable |
| Q6 (predict new?) | One-time investigation |

**Winner:** DBSCAN

---

## K-Means vs DBSCAN — Quick Reference

| Property | K-Means | DBSCAN |
|----------|---------|--------|
| Need to specify number of clusters? | YES (k) | NO — finds them |
| Cluster shape | Spherical only | Any shape |
| Handles noise / outliers? | NO — forces every point into a cluster | YES — labels noise as -1 |
| Scales to large data? | YES | OK — slows down past 100K |
| Predicts cluster for new points? | YES — nearest centroid | NO — naturally |
| Main hyperparameter | `k` (number of clusters) | `eps` (neighborhood radius) and `min_samples` |
| Sensitive to feature scaling? | YES — must standardize | YES — must standardize |
| Good for clusters of varying density? | OK | POOR — single `eps` struggles |
| Interpretability | High — centroids are summaries | Moderate — clusters defined by density |
| Training speed | VERY FAST | MODERATE |
| Prediction speed | VERY FAST | N/A (no model to predict) |

---

## Summary Pattern

1. **Know k + spherical shape + clean data + need to score new points** → K-Means
2. **Don't know k + irregular shapes + outliers matter** → DBSCAN
3. **Massive data (> 1M rows)** → K-Means (or `MiniBatchKMeans`); DBSCAN won't scale
4. **Anomaly detection is the goal** → DBSCAN — use noise points as anomalies
5. **One-time exploratory analysis** → either; DBSCAN if you want to discover cluster count

**The choice is rarely close once you answer the 6 questions. When in doubt, run both, reduce to 2D with PCA, and visually check which one matches the actual structure of your data.**
