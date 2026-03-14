# 01 — Linear Regression

## What Is Linear Regression?

Linear Regression is a **supervised learning** algorithm that models the relationship between one or more **independent variables** (features) and a **continuous dependent variable** (target) by fitting a straight line (or hyperplane) through the data.

It answers: *"Given these input features, what numeric value should I predict?"*

---

## The Math Behind It

### Simple Linear Regression (1 feature)

$$\hat{y} = w_0 + w_1 x$$

- $\hat{y}$: predicted value  
- $w_0$: bias (intercept)  
- $w_1$: weight (slope)  
- $x$: input feature  

### Multiple Linear Regression (n features)

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n = \mathbf{w}^T \mathbf{x} + b$$

In matrix form:

$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w}$$

---

## How It Learns — The Cost Function

We minimize the **Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

This measures the average squared difference between actual and predicted values. The goal is to find $\mathbf{w}$ that minimizes MSE.

---

## Two Ways to Find Optimal Weights

### 1. Normal Equation (Closed-Form Solution)

$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

**Pros:** Exact solution, no hyperparameters  
**Cons:** Slow for large datasets ($O(n^3)$ matrix inversion), fails if $\mathbf{X}^T\mathbf{X}$ is singular

### 2. Gradient Descent (Iterative Optimization)

Repeat until convergence:

$$w_j := w_j - \alpha \frac{\partial}{\partial w_j} \text{MSE}$$

The gradient of MSE with respect to weights:

$$\frac{\partial \text{MSE}}{\partial w_j} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) \cdot x_{ij}$$

**Pros:** Scales to large datasets, works with any differentiable loss  
**Cons:** Requires tuning learning rate $\alpha$ and number of iterations

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error (lower = better) |
| **RMSE** | $\sqrt{\text{MSE}}$ | Error in original units |
| **MAE** | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Average absolute error |
| **R² Score** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Proportion of variance explained (1.0 = perfect) |

---

## Assumptions of Linear Regression

1. **Linearity** — The relationship between X and y is linear  
2. **Independence** — Observations are independent of each other  
3. **Homoscedasticity** — Constant variance of residuals  
4. **Normality** — Residuals are normally distributed  
5. **No multicollinearity** — Features are not highly correlated with each other  

---

## Regularization Variants

| Variant | Penalty Term | Effect |
|---------|-------------|--------|
| **Ridge (L2)** | $+ \lambda \sum w_j^2$ | Shrinks weights, prevents overfitting |
| **Lasso (L1)** | $+ \lambda \sum |w_j|$ | Shrinks weights + feature selection (some go to 0) |
| **Elastic Net** | $+ \lambda_1 \sum |w_j| + \lambda_2 \sum w_j^2$ | Combines both L1 and L2 |

---

## What This Implementation Covers

1. **From-scratch implementation** — Gradient descent with cost history tracking  
2. **Normal equation** — Closed-form exact solution  
3. **Scikit-learn comparison** — Validates our scratch implementations  
4. **Regularization** — Ridge & Lasso via scikit-learn  
5. **Feature scaling** — Demonstrates why standardization matters  
6. **Polynomial regression** — Extending linear regression to nonlinear data  
7. **Visualizations** — Regression line, residuals, cost convergence, comparison plots  

---

## How to Run

```bash
cd 01_linear_regression
python linear_regression.py
```

All plots are saved to `plots/`.

---

## Key Takeaways

- Linear regression is the **foundation** of all regression methods
- Always **scale your features** before gradient descent
- Use **R² and RMSE** together to evaluate performance
- Use **Ridge/Lasso** when you have many features or risk overfitting
- The **Normal Equation** is elegant but doesn't scale; gradient descent does
