# 05 — Gradient Boosting

## What Is Gradient Boosting?

Gradient Boosting is a powerful **ensemble method** that builds models **sequentially**, where each new model corrects the errors of the previous ones. Unlike Random Forest (which builds trees in parallel), Gradient Boosting adds trees one at a time, each focusing on remaining mistakes.

It answers: *"Can I build a sequence of weak learners that, together, form a strong learner?"*

---

## Boosting vs Bagging

| Property | Bagging (Random Forest) | Boosting (Gradient Boosting) |
|----------|------------------------|------------------------------|
| Trees built | **In parallel** (independent) | **Sequentially** (dependent) |
| Each tree fits | A bootstrap sample of data | The **residual errors** of previous trees |
| Goal | Reduce **variance** | Reduce **bias** (and variance) |
| Overfitting risk | Low (more trees = better) | Higher (can overfit with too many trees) |
| Learning rate | N/A | Controls contribution of each tree |

---

## The Core Idea — Additive Modeling

Build the model as a **sum** of weak learners:

$$F_M(\mathbf{x}) = F_0 + \sum_{m=1}^{M} \eta \cdot h_m(\mathbf{x})$$

- $F_0$ = initial prediction (e.g., mean of $y$ for regression)
- $h_m$ = the $m$-th weak learner (shallow decision tree)
- $\eta$ = learning rate (shrinkage), typically 0.01–0.3
- $M$ = total number of boosting rounds (trees)

Each new tree $h_m$ fits the **negative gradient** of the loss function — hence "gradient" boosting.

---

## The Algorithm (for Regression with MSE Loss)

### Step-by-Step

1. **Initialize** with a constant prediction:
$$F_0(\mathbf{x}) = \bar{y}$$

2. **For** $m = 1$ to $M$:
   
   a. Compute **pseudo-residuals** (negative gradient of loss):
   $$r_i^{(m)} = -\frac{\partial L(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}$$
   
   For MSE loss: $r_i^{(m)} = y_i - F_{m-1}(\mathbf{x}_i)$ (simply the residuals!)
   
   b. **Fit** a weak learner (shallow tree) $h_m$ to predict the pseudo-residuals
   
   c. **Update** the model:
   $$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

3. **Final prediction**: $F_M(\mathbf{x})$

---

## Why "Gradient" Boosting?

The pseudo-residuals are the **negative gradient** of the loss function:

| Loss Function | Formula | Negative Gradient (pseudo-residuals) |
|---------------|---------|--------------------------------------|
| **MSE** (regression) | $\frac{1}{2}(y - F)^2$ | $y - F$ (residuals) |
| **Log Loss** (classification) | $-[y\log(p) + (1-y)\log(1-p)]$ | $y - p$ (where $p = \sigma(F)$) |
| **Huber** (robust regression) | Combined L1/L2 | Clipped residuals |

This framework lets gradient boosting work with **any differentiable loss function**!

---

## For Classification

Binary classification uses **log loss**. The model predicts log-odds, converted to probability via sigmoid:

$$p(\mathbf{x}) = \sigma(F_M(\mathbf{x})) = \frac{1}{1 + e^{-F_M(\mathbf{x})}}$$

Pseudo-residuals: $r_i = y_i - p_i$ (actual label minus predicted probability)

---

## Key Hyperparameters

| Parameter | What It Controls | Typical Values | Effect |
|-----------|-----------------|----------------|--------|
| `n_estimators` | Number of boosting rounds | 100–1000 | More rounds → more complex |
| `learning_rate` | Shrinkage per tree | 0.01–0.3 | Lower → needs more trees, better generalization |
| `max_depth` | Depth of each tree | 3–8 | Interaction complexity |
| `subsample` | Fraction of data per tree | 0.5–1.0 | <1.0 adds stochasticity (stochastic GB) |
| `min_samples_leaf` | Min samples per leaf | 1–20 | Regularization |

### The Learning Rate – n_estimators Tradeoff

$$\text{Lower learning rate} + \text{More trees} = \text{Better generalization}$$

But: more trees = slower training. The sweet spot depends on the problem.

---

## Regularization Techniques

1. **Learning rate (shrinkage)** — Scale each tree's contribution by $\eta < 1$
2. **Subsampling** — Use a fraction of data per tree (stochastic gradient boosting)
3. **Tree constraints** — Limit depth, min samples, max leaves
4. **Early stopping** — Stop adding trees when validation score stops improving

---

## Gradient Boosting vs Random Forest

| Property | Random Forest | Gradient Boosting |
|----------|--------------|-------------------|
| Approach | Parallel (bagging) | Sequential (boosting) |
| Trees | Deep, independent | Shallow, dependent |
| Bias | Moderate | **Low** |
| Variance | **Low** | Moderate (controlled by lr) |
| Tuning | Easy (few params) | More sensitive to tuning |
| Raw accuracy | Good | Often **best** |
| Training speed | Fast (parallelizable) | Slower (sequential) |

---

## What This Implementation Covers

1. **From-scratch Gradient Boosting Regressor** — MSE loss, residual fitting
2. **From-scratch Gradient Boosting Classifier** — Log loss, sigmoid
3. **Scikit-learn comparison** — GradientBoostingClassifier & Regressor
4. **Learning rate vs n_estimators tradeoff** — Interactive analysis
5. **Staged predictions** — Watch the model improve round by round
6. **Early stopping** — When to stop adding trees
7. **Subsample impact** — Stochastic gradient boosting
8. **Feature importance**
9. **Residual evolution** — How residuals shrink over boosting rounds
10. **Comparison: Single Tree vs RF vs Gradient Boosting**

---

## How to Run

```bash
cd 05_gradient_boosting
python gradient_boosting.py
```

All plots are saved to `plots/`.

---

## Key Takeaways

- Gradient Boosting builds trees **sequentially**, each correcting previous errors
- It works by fitting trees to the **negative gradient** of the loss function
- **Learning rate** is the most important hyperparameter — low lr + many trees = best results
- It can **overfit** if you add too many trees (unlike Random Forest)
- **Early stopping** on a validation set prevents overfitting efficiently
- Often achieves **highest accuracy** on tabular data among classical ML methods
- The foundation for XGBoost, LightGBM, and CatBoost (covered in algorithm 12)
