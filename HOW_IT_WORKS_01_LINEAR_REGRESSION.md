# How It Works: Linear Regression

> The simplest and most important model in all of machine learning. If you understand linear regression deeply, every other model becomes easier to learn.

---

## The One-Sentence Idea

Linear regression draws the **best-fitting straight line** (or flat plane, or hyperplane) through your data so you can predict a number from other numbers.

---

## Intuition: What Problem Does It Solve?

Imagine you have data about houses: square footage and sale price. You plot them on a graph and see that bigger houses tend to cost more. You want to draw a line through the dots so that when someone tells you a new house's square footage, you can trace up to the line and read off the predicted price.

```
Price ($)
  |                          *
  |                     *  *
  |                  * /
  |              * * /    <-- this line is the model
  |           * / *
  |        * /
  |     * /
  |   */
  |  /
  +--------------------------- Square Feet
```

That line is your linear regression model. The entire "training" process is just finding the line that fits best.

---

## The Math: Step by Step

### Step 1: The Equation of a Line

For one input feature (like square footage), the model is:

$$\hat{y} = w_1 x + b$$

- $x$ = input (square footage)
- $\hat{y}$ = predicted output (price)
- $w_1$ = **weight** (slope of the line -- how much price increases per square foot)
- $b$ = **bias** (y-intercept -- the baseline price when square footage is 0)

For multiple features (square footage, bedrooms, age):

$$\hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$

In compact notation with $n$ features:

$$\hat{y} = \sum_{j=1}^{n} w_j x_j + b = \mathbf{w}^T \mathbf{x} + b$$

**What the model "learns" are the weights $w_1, w_2, ..., w_n$ and the bias $b$.**

### Step 2: What "Best Fit" Means -- The Loss Function

How do we know if a line is good or bad? We measure the **errors** (called residuals):

$$\text{error}_i = y_i - \hat{y}_i = \text{actual price} - \text{predicted price}$$

Some errors are positive (predicted too low), some are negative (predicted too high). If we just add them up, they cancel out. So we **square** each error:

$$\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

This is called **Mean Squared Error (MSE)**. The "best" line is the one that makes this number as small as possible.

Why squared and not absolute value?
- Squaring **penalizes big errors more** (an error of 10 contributes 100, not 10)
- Squaring makes the math **smooth and differentiable** (we can use calculus)
- The absolute value has a sharp corner at zero that complicates optimization

### Step 3: Finding the Best Weights -- Two Methods

#### Method A: Normal Equation (Closed-Form Solution)

This is the "solve it in one shot" approach using linear algebra.

Arrange all your data into a matrix $\mathbf{X}$ (each row = one house, each column = one feature) and a vector $\mathbf{y}$ (all the prices):

$$\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

**What this does in plain English:**
1. $\mathbf{X}^T \mathbf{X}$ -- Computes how each feature relates to every other feature (a correlation-like matrix)
2. $(\mathbf{X}^T \mathbf{X})^{-1}$ -- Inverts that matrix (adjusts for features that overlap)
3. $\mathbf{X}^T \mathbf{y}$ -- Computes how each feature relates to the target
4. Multiply them -- gives you the optimal weight for each feature

**Pros:** Exact answer, no iteration needed
**Cons:** Inverting a matrix is slow for many features. If you have 10,000 features, $\mathbf{X}^T \mathbf{X}$ is a 10,000 x 10,000 matrix -- inverting that is expensive ($O(n^3)$ time)

#### Method B: Gradient Descent (Iterative)

This is the "get closer every step" approach. Think of it like being blindfolded on a hilly landscape and trying to find the lowest valley by always stepping downhill.

```
Loss
  |  \
  |   \
  |    \         Start here (random weights)
  |     \       /
  |      \     /
  |       \   /
  |        \ /   Step downhill repeatedly
  |         *    <-- minimum (best weights)
  +--------------------------- weight value
```

The algorithm:

```
1. Start with random weights: w = [0.01, 0.01, ...], b = 0
2. Repeat many times:
   a. Predict: y_hat = X @ w + b
   b. Compute error: error = y_hat - y
   c. Compute gradient (direction of steepest uphill):
      dw = (2/N) * X.T @ error    (how much each weight contributed to error)
      db = (2/N) * sum(error)      (how much bias contributed to error)
   d. Update weights (step downhill):
      w = w - learning_rate * dw
      b = b - learning_rate * db
3. Stop when changes become tiny
```

**The gradient** $\frac{\partial \text{Loss}}{\partial w_j}$ tells you: "If I increase weight $w_j$ by a tiny amount, how much does the loss increase?" If the gradient is positive, the weight is too high -- decrease it. If negative, increase it.

**Learning rate** ($\alpha$) controls step size:
- Too large: you overshoot the minimum and bounce around
- Too small: you take forever to reach the minimum
- Just right: smooth convergence in ~100-1000 steps

```
Too large:                  Too small:              Just right:
Loss                        Loss                    Loss
|  * *   *                  |  *                     |  *
|  * * *  * *               |   *                    |   *
|  *   * * *  *             |    *                   |    *
|       *   * *             |     *                  |      *
|            *              |      *                 |        *
|                           |       *                |          *
                            |        * * * * * ...
```

### Step 4: What Happens When You Call `.fit()` and `.predict()`

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

# .fit(X, y) does this behind the scenes:
# 1. Adds a column of 1s to X (for the bias term)
# 2. Computes w = (X^T X)^(-1) X^T y  (normal equation)
#    (sklearn actually uses a more numerically stable method called SVD decomposition)
# 3. Stores the weights in model.coef_ and bias in model.intercept_
model.fit(X_train, y_train)

print(model.coef_)        # [312.5, 15000, -2500]
# Meaning: +$312.50 per sqft, +$15,000 per bedroom, -$2,500 per year of age

print(model.intercept_)   # 45000
# Meaning: base price of $45,000

# .predict(X) does this:
# 1. Multiplies each feature by its weight
# 2. Adds them up plus the bias
# prediction = 312.5 * sqft + 15000 * bedrooms - 2500 * age + 45000
predictions = model.predict(X_test)
```

---

## The Full Pipeline: What Happens Behind the Scenes

```
TRAINING (.fit):
                                                    
Input Data          Feature Matrix       Loss Function          Optimization
[sqft, beds, age]   [1500, 3, 10]       MSE = (1/N) *          Normal Equation
[price]             [2000, 4, 5]        sum((y - Xw)^2)        OR Gradient Descent
                    [1200, 2, 20]                               
                    ...                                          Finds optimal w, b
                                                                that minimize MSE
                                         |
                                         v
                                   Learned Weights
                                   w = [312.5, 15000, -2500]
                                   b = 45000


PREDICTION (.predict):

New House            Multiply & Add       Prediction
[1800, 3, 8]    ->  312.5*1800 +     ->  $607,500
                    15000*3 +
                    (-2500)*8 +
                    45000
```

---

## Regularization: Preventing Overfitting

With many features, linear regression can overfit -- it might assign huge weights to noisy features. Regularization adds a **penalty** for large weights.

### Ridge Regression (L2 Regularization)

$$\text{Loss} = \frac{1}{N} \sum (y_i - \hat{y}_i)^2 + \lambda \sum w_j^2$$

The extra term $\lambda \sum w_j^2$ says: "Keep weights small." A weight of 100 adds $100^2 = 10,000$ penalty. So the model is forced to find a balance between fitting the data and keeping weights small.

**Effect:** Shrinks all weights toward zero, but doesn't eliminate any. Good when all features are somewhat useful.

### Lasso Regression (L1 Regularization)

$$\text{Loss} = \frac{1}{N} \sum (y_i - \hat{y}_i)^2 + \lambda \sum |w_j|$$

Uses absolute value instead of squared. This has a special property: **it can push weights exactly to zero**, effectively removing features.

**Effect:** Automatic feature selection. If you have 100 features, Lasso might keep only 15 with non-zero weights.

### Elastic Net (L1 + L2)

Combines both penalties. Best of both worlds.

**$\lambda$ (alpha) controls the trade-off:**
- $\lambda = 0$: No regularization (plain linear regression)
- $\lambda$ small: Slight penalty, weights are slightly shrunk
- $\lambda$ large: Heavy penalty, most weights are near zero, model is very simple

---

## Key Assumptions (When Linear Regression Works Well)

1. **Linearity**: The relationship between features and target is roughly linear
   - If price grows EXPONENTIALLY with size, a straight line won't fit
   - Fix: transform features (log, polynomial)

2. **Independence**: Each data point is independent of others
   - If your data is time-ordered (today's price depends on yesterday's), standard linear regression is wrong
   - Fix: use time-series models

3. **Homoscedasticity**: Errors are roughly the same size everywhere
   - If cheap houses have $5K errors but expensive houses have $50K errors, the model is less reliable for expensive houses
   - Fix: weighted regression or log-transform the target

4. **No multicollinearity**: Features shouldn't be almost identical
   - If "square footage" and "square meters" are both features, the model can't decide how to split the weight between them (weights become unstable)
   - Fix: drop one, or use Ridge regression

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Not scaling features | sqft (1000s) dominates age (10s) in gradient descent | StandardScaler, or use Normal Equation (not affected) |
| Too many features vs. samples | Model memorizes noise | Use Lasso/Ridge, get more data, or reduce features |
| Non-linear relationship | Line can't curve to fit the data | Add polynomial features ($x^2$, $x_1 \cdot x_2$), or use a non-linear model |
| Outliers | One mansion at $50M pulls the line up | Use robust regression (Huber loss) or remove outliers |
| Extrapolation | Predicting for a 100,000 sqft building (way outside training range) | Never trust predictions far outside your data range |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it learns** | A weight for each feature + a bias |
| **How it learns** | Minimizes Mean Squared Error (via normal equation or gradient descent) |
| **Prediction** | Multiply each feature by its weight, add them up + bias |
| **Output** | A continuous number (regression) |
| **Speed** | Very fast to train and predict |
| **Interpretable?** | Yes -- each weight tells you the effect of that feature |
| **When to use** | Linear relationships, need interpretability, baseline model |
| **When NOT to use** | Non-linear data, categorical heavy data, complex interactions |
