# 11 — PCA (Principal Component Analysis)

## Overview

Principal Component Analysis (PCA) is the most fundamental **dimensionality reduction** technique. It finds a new set of axes (principal components) that capture the **maximum variance** in the data, allowing you to represent high-dimensional data in fewer dimensions with minimal information loss.

PCA is an **unsupervised, linear** method — it uses no labels and finds linear combinations of the original features.

---

## Why Dimensionality Reduction?

| Problem | How PCA Helps |
|---------|--------------|
| **Curse of dimensionality** | Fewer features → algorithms work better |
| **Visualization** | Project to 2D/3D for human understanding |
| **Noise reduction** | Minor components often capture noise |
| **Multicollinearity** | PCA components are orthogonal (uncorrelated) |
| **Computational cost** | Fewer features → faster training |
| **Storage** | Compressed representation of data |

---

## The Math Behind PCA

### Step-by-Step Algorithm

Given data matrix $X$ with $n$ samples and $d$ features:

**Step 1: Center the data** (subtract the mean)
$$\bar{X} = X - \mu$$
where $\mu_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij}$ for each feature $j$

**Step 2: Compute the covariance matrix**
$$C = \frac{1}{n-1} \bar{X}^T \bar{X}$$
$C$ is a $d \times d$ symmetric matrix where $C_{ij}$ measures how features $i$ and $j$ co-vary.

**Step 3: Eigendecomposition**
$$C \mathbf{v} = \lambda \mathbf{v}$$
Find eigenvalues $\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_d$ and corresponding eigenvectors $\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_d$.

**Step 4: Select top-k components**
Keep the $k$ eigenvectors with the largest eigenvalues:
$$W = [\mathbf{v}_1 | \mathbf{v}_2 | ... | \mathbf{v}_k] \quad \text{(d × k matrix)}$$

**Step 5: Project the data**
$$Z = \bar{X} W$$
$Z$ is the $n \times k$ matrix of transformed data in the new coordinate system.

### Equivalence with SVD

PCA can also be computed via **Singular Value Decomposition** (more numerically stable):

$$\bar{X} = U \Sigma V^T$$

The columns of $V$ are the principal component directions, and the singular values $\sigma_i$ relate to eigenvalues by $\lambda_i = \sigma_i^2 / (n-1)$.

Scikit-learn uses SVD internally.

---

## Explained Variance

Each eigenvalue $\lambda_i$ represents the **variance captured** by the $i$-th principal component.

**Explained variance ratio** for component $i$:
$$\text{EVR}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}$$

**Cumulative explained variance**:
$$\text{CEV}_k = \sum_{i=1}^{k} \text{EVR}_i$$

### Choosing the Number of Components

| Method | Rule |
|--------|------|
| **Threshold** | Keep components until CEV ≥ 95% (or 99%) |
| **Elbow** | Look for a "knee" in the scree plot |
| **Kaiser's Rule** | Keep components with eigenvalue > 1 (for standardized data) |
| **Cross-validation** | Use downstream task performance |

---

## Geometric Interpretation

- **PC1** points in the direction of maximum variance in the data
- **PC2** is orthogonal to PC1 and captures the next most variance
- **PC3** is orthogonal to both PC1 and PC2, and so on
- The principal components form an **orthonormal basis** — they are unit vectors and mutually perpendicular

Projecting onto the first $k$ components finds the $k$-dimensional hyperplane that best fits the data (minimizes reconstruction error).

---

## Reconstruction

Given the projected data $Z$ and the component matrix $W$, we can **reconstruct** an approximation of the original data:

$$\hat{X} = Z W^T + \mu$$

The **reconstruction error** is:

$$\text{Error} = \|X - \hat{X}\|^2 = \sum_{i=k+1}^{d} \lambda_i$$

This equals the sum of the eigenvalues of the discarded components.

---

## Feature Scaling

PCA is **sensitive to feature scale** because it maximizes variance. Features with large ranges will dominate the principal components.

**Always standardize** (zero mean, unit variance) before PCA unless all features are already on the same scale:
$$x' = \frac{x - \mu}{\sigma}$$

Using `StandardScaler` before PCA is standard practice.

---

## PCA vs Other Dimensionality Reduction

| Method | Type | Linear? | Preserves |
|--------|------|---------|-----------|
| **PCA** | Unsupervised | Yes | Global variance |
| **LDA** | Supervised | Yes | Class separability |
| **t-SNE** | Unsupervised | No | Local structure |
| **UMAP** | Unsupervised | No | Local + global |
| **Autoencoders** | Unsupervised | No | Learned representation |

---

## Strengths & Weaknesses

### Strengths
- Fast and well-understood (closed-form solution)
- Optimal linear dimensionality reduction (minimizes reconstruction error)
- Components are uncorrelated — removes multicollinearity
- Great for visualization (2D/3D projections)
- No hyperparameters (except number of components)
- Useful as preprocessing for other algorithms

### Weaknesses
- **Linear only** — can't capture nonlinear structure
- **Variance ≠ importance** — high-variance directions may be noise
- **Interpretability** — components are linear combos of all features
- **Sensitive to outliers** — outliers inflate variance
- Assumes features are continuous (not ideal for categorical data)

---

## What This Implementation Covers

1. **From-scratch PCA** — centering, covariance matrix, eigendecomposition, projection, reconstruction
2. **Scikit-learn PCA** comparison
3. **Scree plot** — eigenvalues and explained variance ratio
4. **Cumulative explained variance** — choosing number of components
5. **2D & 3D visualization** — projecting high-dimensional data
6. **Reconstruction** — compress and decompress, visualize error
7. **Feature contribution** — loadings heatmap (which features drive each PC)
8. **Scaling impact** — PCA with vs without StandardScaler
9. **PCA as preprocessing** — classifier accuracy vs number of components
10. **Biplot** — samples and feature vectors in PC space

---

## How to Run

```bash
cd 11_pca
python pca.py
```

All plots are saved to `plots/`.
