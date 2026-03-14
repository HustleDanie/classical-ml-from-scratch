# 02 — Logistic Regression

## What Is Logistic Regression?

Despite the name, Logistic Regression is a **classification** algorithm, not regression. It predicts the **probability** that an input belongs to a particular class by passing a linear combination of features through the **sigmoid function**.

It answers: *"What is the probability that this input belongs to class 1?"*

---

## The Math Behind It

### The Sigmoid (Logistic) Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- Maps any real number $z$ to the range $(0, 1)$
- Output is interpreted as a **probability**: $P(y=1 | \mathbf{x})$
- Decision boundary: predict class 1 if $\sigma(z) \geq 0.5$, else class 0

### The Linear Model Inside

$$z = \mathbf{w}^T \mathbf{x} + b = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n$$

### Full Prediction

$$\hat{y} = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

---

## Why Not Use MSE? — The Cost Function

MSE with sigmoid creates a **non-convex** cost surface (many local minima). Instead, we use **Binary Cross-Entropy (Log Loss)**:

$$J(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Intuition:**
- If $y = 1$ and $\hat{y} \to 1$: cost → 0 (correct, no penalty)
- If $y = 1$ and $\hat{y} \to 0$: cost → $\infty$ (wrong, huge penalty)
- Same logic flipped for $y = 0$

This loss is **convex**, guaranteeing a single global minimum.

---

## How It Learns — Gradient Descent

The gradient of log loss is remarkably similar to linear regression:

$$\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \cdot x_{ij}$$

$$\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

Update rules:
$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

The key difference from linear regression: $\hat{y}_i = \sigma(\mathbf{w}^T \mathbf{x}_i + b)$, not a linear output.

---

## Decision Boundary

Logistic regression learns a **linear decision boundary**:

$$\mathbf{w}^T \mathbf{x} + b = 0$$

Points on one side → class 0, other side → class 1. The boundary is always a straight line (2D), plane (3D), or hyperplane (nD).

---

## Multiclass Extension

### One-vs-Rest (OvR)

For $K$ classes, train $K$ binary classifiers. Each one answers: "Is it class $k$ or not?" Predict the class with the highest probability.

### Softmax Regression (Multinomial)

Generalize sigmoid to $K$ classes:

$$P(y = k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Uses **Categorical Cross-Entropy** loss:

$$J = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

---

## Evaluation Metrics for Classification

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | $\frac{TP + TN}{Total}$ | Balanced classes |
| **Precision** | $\frac{TP}{TP + FP}$ | When false positives are costly |
| **Recall** | $\frac{TP}{TP + FN}$ | When false negatives are costly |
| **F1 Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Balance precision & recall |
| **ROC-AUC** | Area under ROC curve | Overall discriminative power |

### Confusion Matrix

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actually +** | True Positive (TP) | False Negative (FN)|
| **Actually −** | False Positive (FP)| True Negative (TN) |

---

## Regularization

| Variant | Penalty | scikit-learn param |
|---------|---------|-------------------|
| **L2 (Ridge)** | $+ \frac{\lambda}{2} \sum w_j^2$ | `penalty='l2'` (default) |
| **L1 (Lasso)** | $+ \lambda \sum |w_j|$ | `penalty='l1'`, `solver='saga'` |
| **ElasticNet** | L1 + L2 combined | `penalty='elasticnet'`, `solver='saga'` |
| **None** | No penalty | `penalty=None` |

In sklearn, regularization strength is controlled by `C = 1/λ`. Smaller C = stronger regularization.

---

## What This Implementation Covers

1. **From-scratch binary logistic regression** — Gradient descent with sigmoid
2. **From-scratch multiclass (softmax)** — Full multinomial implementation
3. **Scikit-learn comparison** — Binary and multiclass
4. **Decision boundary visualization** — 2D plots showing the learned boundary
5. **Threshold tuning** — Precision-recall tradeoff
6. **ROC curve & AUC** — Discrimination analysis
7. **Confusion matrix heatmap** — Visual error analysis
8. **Regularization comparison** — L1 vs L2 vs none
9. **Feature scaling impact** — Why it matters for logistic regression too

---

## How to Run

```bash
cd 02_logistic_regression
python logistic_regression.py
```

All plots are saved to `plots/`.

---

## Key Takeaways

- Logistic regression is **linear classification**, not regression
- The **sigmoid** squashes outputs to probabilities
- Use **log loss** (not MSE) — it's convex and penalizes confident wrong answers harshly
- The gradient update looks identical to linear regression, but $\hat{y}$ is sigmoid output
- **Softmax** generalizes to multiclass elegantly
- Always check **precision, recall, F1** — accuracy alone is misleading on imbalanced data
- **C parameter** controls regularization: lower C = more regularization
