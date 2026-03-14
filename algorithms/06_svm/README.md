# 06 — Support Vector Machines (SVM)

## What Is an SVM?

A Support Vector Machine is a **supervised learning** algorithm that finds the **optimal hyperplane** separating classes with the **maximum margin**. Instead of fitting probabilities or averaging trees, SVM focuses on geometry — finding the widest possible "street" between classes.

It answers: *"What is the decision boundary that maximizes the gap between the closest points of each class?"*

---

## The Core Idea — Maximum Margin

### Linear SVM (Linearly Separable Case)

Given data points $(\mathbf{x}_i, y_i)$ where $y_i \in \{-1, +1\}$, find hyperplane:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

The **margin** is the distance between the two decision boundaries:

$$\text{Margin} = \frac{2}{\|\mathbf{w}\|}$$

**Goal:** Maximize the margin → Minimize $\|\mathbf{w}\|$

### Support Vectors

The data points that **lie exactly on the margin boundaries** are called **support vectors**. They are the critical points — the entire model depends only on them. Remove any non-support-vector point, and the model doesn't change.

$$\mathbf{w}^T \mathbf{x}_i + b = +1 \quad \text{(positive support vectors)}$$
$$\mathbf{w}^T \mathbf{x}_i + b = -1 \quad \text{(negative support vectors)}$$

---

## The Optimization Problem

### Hard Margin (perfectly separable data)

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$$
$$\text{subject to: } y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 \quad \forall i$$

### Soft Margin (allowing some misclassifications)

Real data is rarely perfectly separable. Add **slack variables** $\xi_i \geq 0$:

$$\min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} \xi_i$$
$$\text{subject to: } y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i$$

- $C$ = regularization parameter
- Large $C$ → narrow margin, fewer violations (risk overfitting)
- Small $C$ → wide margin, more violations (risk underfitting)

---

## The Kernel Trick

### Problem: Nonlinear Data

Linear SVM fails when classes aren't linearly separable. Solution: **map data to a higher-dimensional space** where it becomes separable.

### Key Insight

We don't need to compute the transformation explicitly! The **kernel function** computes the dot product in the high-dimensional space directly:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$$

### Common Kernels

| Kernel | Formula | When to Use |
|--------|---------|-------------|
| **Linear** | $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$ | Linearly separable data, high-dim text |
| **RBF (Gaussian)** | $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$ | Default, most versatile |
| **Polynomial** | $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$ | When interactions matter |
| **Sigmoid** | $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)$ | Neural network connection |

### RBF Kernel — The Most Important One

$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2\right)$$

- $\gamma$ = controls the "reach" of each support vector
- High $\gamma$ → tight, complex boundaries (overfitting risk)
- Low $\gamma$ → smooth, simpler boundaries (underfitting risk)

RBF effectively projects data into **infinite-dimensional** space!

---

## The Hinge Loss (Alternative View)

SVM can be seen as minimizing the **hinge loss**:

$$L = \frac{1}{n} \sum_{i=1}^{n} \max(0, 1 - y_i f(\mathbf{x}_i)) + \frac{\lambda}{2} \|\mathbf{w}\|^2$$

- If correctly classified with margin ≥ 1: loss = 0
- If inside the margin or misclassified: loss > 0

This connects SVM to gradient-based optimization (used in our scratch implementation).

---

## Key Hyperparameters

| Parameter | What It Controls | Effect |
|-----------|-----------------|--------|
| `C` | Regularization | High C = narrow margin, low C = wide margin |
| `kernel` | Transformation | 'linear', 'rbf', 'poly', 'sigmoid' |
| `gamma` | RBF/poly reach | High = complex boundary, low = smooth |
| `degree` | Polynomial degree | Only for poly kernel |

### C and gamma Interaction (RBF)

| | Low gamma | High gamma |
|---|-----------|------------|
| **Low C** | Very smooth, underfitting | Moderate complexity |
| **High C** | Moderate complexity | Very complex, overfitting |

---

## SVM for Multiclass

SVM is inherently binary. For K classes:

| Strategy | How It Works | # of Models |
|----------|-------------|-------------|
| **One-vs-Rest (OvR)** | K binary classifiers | K |
| **One-vs-One (OvO)** | Pairwise classifiers | $K(K-1)/2$ |

Scikit-learn uses OvO by default for SVC.

---

## Advantages & Disadvantages

| Advantages | Disadvantages |
|-----------|---------------|
| Effective in high dimensions | Slow on large datasets ($O(n^2)$ to $O(n^3)$) |
| Memory efficient (only support vectors) | Sensitive to feature scaling |
| Versatile with kernels | Hard to interpret (black box with kernels) |
| Strong theoretical guarantees | Requires careful tuning of C and gamma |
| Works well with small datasets | No native probability estimates |

---

## What This Implementation Covers

1. **From-scratch Linear SVM** — Hinge loss + gradient descent
2. **Scikit-learn SVM** — Linear, RBF, Polynomial kernels
3. **Kernel comparison** — Same data, different kernels
4. **C parameter impact** — Margin width vs accuracy tradeoff
5. **Gamma parameter impact** — RBF boundary complexity
6. **Support vectors visualization** — Which points define the boundary
7. **Decision boundary comparison** — Linear vs RBF vs Poly
8. **Feature scaling importance** — Critical for SVM
9. **Multiclass SVM** — One-vs-One on Iris
10. **SVM vs other classifiers** — Comparison

---

## How to Run

```bash
cd 06_svm
python svm.py
```

All plots are saved to `plots/`.

---

## Key Takeaways

- SVM finds the **maximum margin** hyperplane between classes
- **Support vectors** are the only points that matter for the boundary
- The **kernel trick** enables nonlinear classification without explicit transformation
- **RBF kernel** projects to infinite dimensions — handles most nonlinear cases
- **C** controls margin width (regularization), **gamma** controls boundary complexity
- **Feature scaling is mandatory** — SVM relies on distances
- Best for **small to medium datasets** with high-dimensional features
- Think of SVM as "geometric classification" vs trees' "rule-based classification"
