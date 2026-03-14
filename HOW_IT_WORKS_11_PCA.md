# How It Works: PCA (Principal Component Analysis)

> PCA is a technique that **reduces the number of features** in your data while keeping as much information (variance) as possible. It does this by finding new axes (principal components) along which the data varies the most.

---

## The One-Sentence Idea

PCA finds the directions in your data where the **variance is greatest**, then projects the data onto these directions, letting you drop dimensions that carry little information -- reducing features from, say, 100 down to 10 while retaining 95%+ of the information.

---

## Intuition: Viewing a 3D Object

Imagine you're photographing a sculpture. If you photograph it from the front, you see its full height and width -- lots of information. If you photograph it edge-on, you see almost nothing -- it looks like a thin line.

PCA finds the **best camera angle** -- the direction from which the data looks most spread out, capturing the most information. Then it takes the "photo" (projection) from that angle.

```
3D data:                         2D projection (PCA):
     * * *                        *   *
   * * * *  *                     * * * *
  * * * * * *    ---PCA--->       * * * *
   * * * *  *                     *   *
     * * *                        
                                  Lost the thin z-axis,
Spread in x and y,               kept x and y.
thin in z.                        95% of variance preserved.
```

---

## Why Reduce Dimensions?

| Problem | How PCA Helps |
|---------|--------------|
| Too many features (curse of dimensionality) | Reduce 100 features to 10 |
| Correlated features (multicollinearity) | PCA creates uncorrelated components |
| Visualization | Project 50D data to 2D for plotting |
| Speed | Fewer features = faster model training |
| Noise reduction | Dropping low-variance components removes noise |

---

## The Math: Step by Step

### Step 1: Center the Data (Subtract the Mean)

For each feature, subtract its mean so the data is centered at the origin.

$$\mathbf{X}_{\text{centered}} = \mathbf{X} - \bar{\mathbf{X}}$$

```
Original:                  Centered:
  Feature 1  Feature 2       Feature 1  Feature 2
    10         100              -5          -50
    15         150               0            0
    20         200               5           50

  Mean: [15, 150]            Mean: [0, 0]
```

**Why center?** PCA looks for directions of maximum variance. Centering ensures these directions pass through the origin.

### Step 2: Compute the Covariance Matrix

The covariance matrix captures how features vary together.

$$\mathbf{C} = \frac{1}{n-1} \mathbf{X}_{\text{centered}}^T \mathbf{X}_{\text{centered}}$$

For $d$ features, $\mathbf{C}$ is a $d \times d$ matrix.

$$\mathbf{C} = \begin{bmatrix} \text{var}(x_1) & \text{cov}(x_1, x_2) & \cdots \\ \text{cov}(x_2, x_1) & \text{var}(x_2) & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

- Diagonal: variance of each feature
- Off-diagonal: covariance between pairs of features
- The matrix is symmetric

### Step 3: Find Eigenvectors and Eigenvalues

Solve:

$$\mathbf{C} \mathbf{v} = \lambda \mathbf{v}$$

where:
- $\mathbf{v}$ = **eigenvector** (a direction / principal component axis)
- $\lambda$ = **eigenvalue** (how much variance lies along that direction)

```
Covariance Matrix C:
  [25   250]        Eigenvector 1: [0.0447, 0.999]   Eigenvalue 1: 2525.0
  [250 2500]        Eigenvector 2: [0.999, -0.0447]   Eigenvalue 2: 0.0

                    PC1 captures almost all the variance!
                    PC2 captures almost none.
```

The eigenvectors define new axes. The eigenvalues tell you how important each axis is.

### Step 4: Sort by Eigenvalue (Largest First)

$$\lambda_1 \geq \lambda_2 \geq \lambda_3 \geq \cdots \geq \lambda_d$$

The first principal component (PC1) is the direction of **maximum variance**. The second (PC2) has the **second-most variance** and is **perpendicular** to PC1. And so on.

### Step 5: Choose How Many Components to Keep

Calculate explained variance ratio:

$$\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

```
Component  Eigenvalue  Explained Var  Cumulative
PC1        4.2         52.5%          52.5%
PC2        2.1         26.3%          78.8%
PC3        0.9         11.3%          90.0%      <-- often a good cutoff
PC4        0.5          6.3%          96.3%
PC5        0.2          2.5%          98.8%
PC6        0.04         0.5%          99.3%
PC7        0.03         0.4%          99.6%
PC8        0.03         0.3%         100.0%

8 features -> keeping 3 PCs retains 90% of variance
           -> keeping 4 PCs retains 96.3% of variance
```

Rule of thumb: keep enough components to preserve **90-95% of variance**.

### Step 6: Project Data onto Selected Components

Create a matrix $\mathbf{W}$ from the top $k$ eigenvectors (as columns):

$$\mathbf{X}_{\text{reduced}} = \mathbf{X}_{\text{centered}} \cdot \mathbf{W}$$

This transforms $N \times d$ data into $N \times k$ data (fewer features).

```
Original: 100 points x 8 features             Reduced: 100 points x 3 features
                                    
  x1  x2  x3  x4  x5  x6  x7  x8     dot       PC1    PC2    PC3
  ----+---+---+---+---+---+---+---     with      ------+------+------
  ... ... ... ... ... ... ... ...      W         ...    ...    ...
  ... ... ... ... ... ... ... ...   -------->    ...    ...    ...
  ... ... ... ... ... ... ... ...                ...    ...    ...
  [100 x 8]                          [8 x 3]    [100 x 3]
```

---

## Concrete Example

```
Raw Data (4 students, 4 features):

Student  Math  Physics  History  Art
   A      90     88       40      35
   B      80     80       45      40
   C      40     35       90      88
   D      35     30       85      92

Step 1: Center
  Mean = [61.25, 58.25, 65, 63.75]

  Student  Math    Physics  History  Art
     A     28.75   29.75   -25      -28.75
     B     18.75   21.75   -20      -23.75
     C    -21.25  -23.25    25       24.25
     D    -26.25  -28.25    20       28.25

Step 2: Covariance Matrix (4x4)
  Math and Physics highly correlated (positive)
  Math and History highly correlated (negative)
  -> The variance structure is: STEM vs. Arts

Step 3: Eigendecomposition
  PC1: [0.5, 0.5, -0.5, -0.5]  eigenvalue = 5800  (captures STEM vs Arts)
  PC2: [0.1, -0.1, -0.2, 0.2]  eigenvalue = 12    (minor variation)
  PC3: ...                       eigenvalue = 5
  PC4: ...                       eigenvalue = 1

Step 4: Explained variance
  PC1: 5800/5818 = 99.7%!!
  PC2: 12/5818 = 0.2%
  
  -> 1 component captures nearly all variance!

Step 5: Project onto PC1
  Student A: 28.75*0.5 + 29.75*0.5 + (-25)*(-0.5) + (-28.75)*(-0.5) = 56.1
  Student B: ... = 42.1
  Student C: ... = -46.9
  Student D: ... = -51.4

  Result: 4 features compressed to 1 number!
  High PC1 = STEM-oriented, Low PC1 = Arts-oriented
  
  A: 56.1   -> STEM student
  B: 42.1   -> STEM student
  C: -46.9  -> Arts student
  D: -51.4  -> Arts student
```

---

## What Happens Behind the Scenes: `.fit()` and `.transform()`

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

# .fit(X) does this:
# 1. Compute the mean of each feature
#    self.mean_ = X.mean(axis=0)
# 2. Center the data: X_centered = X - self.mean_
# 3. Compute covariance matrix (or use SVD, which is faster and more stable)
# 4. Find eigenvectors and eigenvalues
# 5. Sort by eigenvalue (largest first)
# 6. Store top n_components eigenvectors as self.components_
# 7. Store eigenvalues as self.explained_variance_
pca.fit(X_train)

# .transform(X) does this:
# 1. Center the data: X_centered = X - self.mean_
# 2. Project: X_reduced = X_centered @ self.components_.T
X_reduced = pca.transform(X_train)

# Key attributes:
print(pca.components_)              # the eigenvectors (direction of each PC)
print(pca.explained_variance_)       # eigenvalue of each PC
print(pca.explained_variance_ratio_) # proportion of variance per PC
print(pca.n_components_)            # number of components kept
print(pca.mean_)                     # mean that was subtracted
```

### n_components Options

```python
# Fixed number of components
pca = PCA(n_components=5)         # keep exactly 5 PCs

# Keep enough for a target variance
pca = PCA(n_components=0.95)      # keep PCs until 95% variance explained

# Keep all (for analysis)
pca = PCA()                       # compute all, decide later
pca.fit(X)
cumvar = pca.explained_variance_ratio_.cumsum()
# Inspect cumvar to choose cutoff
```

---

## The Full Pipeline

```
TRAINING (.fit):

Input: 1000 samples x 50 features

Step 1: Compute mean of each feature
  mean = [3.2, 0.8, -1.1, ..., 2.4]   (50 values)

Step 2: Center data
  X_centered = X - mean              (1000 x 50)

Step 3: SVD decomposition
  X_centered = U * S * V^T
  (sklearn uses SVD instead of covariance + eigen for numerical stability)
  V^T rows = principal component directions
  S^2 / (n-1) = explained variances

Step 4: Sort by variance (already sorted by SVD)
  PC1 explains 35% of variance
  PC2 explains 20% of variance
  PC3 explains 12% of variance
  ...
  PC50 explains 0.01% of variance

Step 5: Keep top k components (say k=10 for 95% cumulative variance)
  Store: components_ (10 x 50 matrix), mean_, explained_variance_ratio_

TRANSFORMING:

Input: New data point [x1, x2, ..., x50]

Step 1: Center: x_centered = x - mean
Step 2: Project: x_reduced = x_centered @ components_.T
Step 3: Output: [pc1, pc2, ..., pc10]   (10 values instead of 50)

INVERSE TRANSFORM (reconstruct approximate original):

Step 1: x_reconstructed = x_reduced @ components_ + mean
Step 2: Output: [x1_approx, x2_approx, ..., x50_approx]
  (approximation because we dropped 40 components)
```

---

## The Scree Plot and Elbow Method

```
Explained
Variance
   |
35%| *
   |
20%|   *
   |
12%|     *
   |
 8%|       *
 5%|         *
 3%|           *
 1%|             * * * * * * ...
   +--+--+--+--+--+--+--+--+---
      1  2  3  4  5  6  7  8  ...
              Component Number

The "elbow" is around PC4-PC5.
Keep 4-5 components.
```

```python
import matplotlib.pyplot as plt

pca = PCA().fit(X)

plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_.cumsum(), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.legend()
plt.show()
```

---

## PCA Internally Uses SVD (Not Eigendecomposition)

Sklearn uses **Singular Value Decomposition (SVD)** instead of computing the covariance matrix and its eigenvectors. Why?

```
Method 1: Covariance + Eigen              Method 2: SVD (what sklearn does)
1. Center X                                1. Center X
2. Compute C = X^T X / (n-1)              2. Compute X = U * S * V^T
   (d x d matrix -- expensive if d>n)         (no need for covariance matrix)
3. Eigen decompose C                       3. Principal components = rows of V^T
                                            4. Variances = S^2 / (n-1)

Why SVD is better:
- Numerically more stable (no squaring condition numbers)
- Faster for n >> d or d >> n
- Randomized SVD available for very large datasets
```

---

## Reconstruction Error

When you reduce dimensions, some information is lost. The reconstruction error measures this:

$$\text{Reconstruction Error} = \| \mathbf{X} - \mathbf{X}_{reconstructed} \|^2$$

```python
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(X_reduced)

# Reconstruction error
error = ((X - X_reconstructed) ** 2).mean()
```

More components kept = less reconstruction error = more information preserved.

```
Components kept:  1    5    10   20   50 (all)
Explained var:   35%  80%  95%  99%  100%
Recon. error:    High Med  Low  Tiny  0
```

---

## PCA for Visualization

One of the most common uses -- projecting high-dimensional data to 2D for plotting:

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)  # X was 50-dimensional

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.show()
```

```
PC2 (20% var)
     |
     |  * * *         + + +
     |  * * * *       + + +
     |    * * *       + + +
     |  
     |        o o o
     |        o o o o
     |          o o o
     +------------------------------> PC1 (35% var)

Three classes clearly separated in 2D!
(Original data was 50-dimensional)
```

---

## Important Properties of PCA

1. **Components are orthogonal**: Each PC is perpendicular to all others
2. **Components are uncorrelated**: Correlation between any two PCs is zero
3. **Ordered by variance**: PC1 has most variance, PC2 next, etc.
4. **Linear transformation**: PCA can only find linear relationships
5. **Sensitive to scale**: Features must be standardized first

```
CRITICAL: StandardScaler before PCA

Without scaling:                    With scaling:
Feature 1: salary ($30K-$150K)     Feature 1: standardized (mean=0, std=1)
Feature 2: age (20-60)             Feature 2: standardized (mean=0, std=1)

PC1 dominated by salary            PC1 genuinely captures the most
(just because it has bigger         important direction of variation
numbers, not because it's           across all features equally.
more important!)
```

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),  # ALWAYS scale before PCA
    ('pca', PCA(n_components=0.95)),  # keep 95% variance
    ('model', SomeClassifier())
])
```

---

## Variants

| Variant | When to Use |
|---------|-------------|
| **PCA** | Standard, small-to-medium data, all in memory |
| **Incremental PCA** | Data doesn't fit in memory (process in batches) |
| **Sparse PCA** | Want components that use only a few original features (interpretability) |
| **Kernel PCA** | Non-linear relationships (uses kernel trick like SVM) |
| **Randomized PCA** | Very large datasets (faster approximation, default in sklearn when n_components << d) |

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Not scaling features | PCA dominated by high-magnitude features | StandardScaler before PCA |
| Using PCA for classification | PCA maximizes variance, not class separation | Use LDA if you want class-aware reduction |
| Interpreting PC values directly | PCs are linear combos, hard to interpret | Inspect loadings (components_) to understand meaning |
| Keeping too few components | Important information lost, model accuracy drops | Check cumulative variance (aim for 90-95%) |
| Keeping too many components | No dimensionality benefit, still slow | Choose the elbow or 95% threshold |
| Applying PCA to categorical data | PCA is for continuous features | Use MCA or one-hot encode + PCA with caution |
| Fitting PCA on test data | Data leakage | fit on train, transform on test |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Type** | Unsupervised dimensionality reduction |
| **What it learns** | Principal component directions (eigenvectors) and their importance (eigenvalues) |
| **How it learns** | Finds directions of maximum variance via SVD of centered data |
| **Key parameter** | n_components (number of dimensions to keep, or variance threshold) |
| **Output** | Transformed data with fewer features + explained variance ratios |
| **Feature scaling?** | REQUIRED (StandardScaler) |
| **Linear?** | Yes (use Kernel PCA for non-linear) |
| **Destroys interpretability?** | Yes -- features become linear combos of originals |
| **When to use** | Too many features, multicollinearity, visualization, noise reduction, preprocessing before other models |
| **When NOT to use** | Few features already, need feature interpretability, categorical-heavy data, want class-aware reduction (use LDA) |
