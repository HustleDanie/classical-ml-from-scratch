# How It Works: Logistic Regression

> Despite the name, logistic regression is a **classification** model, not regression. It predicts probabilities (0% to 100%) that something belongs to a category.

---

## The One-Sentence Idea

Logistic regression takes a linear equation (just like linear regression) and **squishes** its output through a special S-shaped curve so the result is always a probability between 0 and 1.

---

## Intuition: What Problem Does It Solve?

Imagine you're a doctor with patient data (age, blood pressure, cholesterol) and you want to predict: "Will this patient have a heart attack? Yes or No?"

Linear regression would give you numbers like -3.2 or 157.8 -- those aren't valid probabilities. You need something that always outputs a number between 0 and 1 that you can interpret as "72% chance of heart attack."

Logistic regression does exactly this.

---

## The Math: Step by Step

### Step 1: Start with a Linear Equation (Same as Linear Regression)

$$z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$

For our heart attack example:
$$z = 0.05 \cdot \text{age} + 0.03 \cdot \text{blood\_pressure} + 0.02 \cdot \text{cholesterol} - 8.5$$

This $z$ value (called the **log-odds** or **logit**) can be any number from $-\infty$ to $+\infty$.

### Step 2: The Sigmoid Function (The Magic Squisher)

We push $z$ through the **sigmoid function** (also called the logistic function):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This S-shaped curve transforms any number into a value between 0 and 1:

```
Output (probability)
1.0 |                          ____________
    |                        /
    |                      /
0.5 |                    /     <-- S-shaped curve
    |                  /
    |                /
0.0 |______________/
    +------|------|------|------
         -6      0      6      z (log-odds)
```

**How it works:**
- If $z$ is very negative (like -10): $e^{-(-10)} = e^{10} = 22026$, so $\frac{1}{1 + 22026} \approx 0.00005$ (nearly 0%)
- If $z = 0$: $\frac{1}{1 + e^0} = \frac{1}{1 + 1} = 0.5$ (50/50)
- If $z$ is very positive (like +10): $e^{-10} = 0.000045$, so $\frac{1}{1 + 0.000045} \approx 0.99995$ (nearly 100%)

So the full model is:

$$P(\text{heart attack}) = \frac{1}{1 + e^{-(w_1 x_1 + w_2 x_2 + ... + b)}}$$

### Step 3: Making a Decision (Threshold)

The model outputs a probability. To make a yes/no decision, we apply a **threshold** (default: 0.5):

$$\text{prediction} = \begin{cases} 1 \text{ (yes)} & \text{if } P \geq 0.5 \\ 0 \text{ (no)} & \text{if } P < 0.5 \end{cases}$$

The threshold of 0.5 is just a default -- you can change it:
- **Medical diagnosis**: Lower threshold (0.3) to catch more sick patients (fewer missed cases)
- **Spam filter**: Higher threshold (0.8) so you don't accidentally block real emails

### Step 4: The Loss Function -- Why NOT Use MSE?

In linear regression, we minimized Mean Squared Error. Why can't we do that here?

If we used MSE with the sigmoid curve, the loss landscape would be **bumpy** (non-convex) -- gradient descent could get stuck in bad spots:

```
MSE with sigmoid:           Log Loss (correct):
Loss                        Loss
|  *     *                  |  *
|   *   * *                 |   *
|    * *    *  <- bumps!    |    *
|     *      *              |     *
|             *             |       *
|              *            |          *    <- smooth bowl
```

Instead, logistic regression uses **Log Loss** (also called Binary Cross-Entropy):

$$\text{Loss} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

**Breaking this down for one patient:**

- If the patient DID have a heart attack ($y = 1$):
  - Loss $= -\log(\hat{p})$
  - If model predicted 0.95 (confident and correct): $-\log(0.95) = 0.05$ (tiny loss)
  - If model predicted 0.05 (confident and WRONG): $-\log(0.05) = 3.0$ (huge loss!)

- If the patient did NOT have a heart attack ($y = 0$):
  - Loss $= -\log(1 - \hat{p})$
  - If model predicted 0.1 (confident and correct): $-\log(0.9) = 0.1$ (tiny loss)
  - If model predicted 0.9 (confident and WRONG): $-\log(0.1) = 2.3$ (huge loss!)

**Key property:** The loss **explodes** when the model is confidently wrong. This forces the model to be careful with its confidence.

### Step 5: Finding the Best Weights -- Gradient Descent

Unlike linear regression, there's **no closed-form solution** for logistic regression. We must use gradient descent.

```
1. Start with random weights: w = [0, 0, ...], b = 0

2. Repeat until converged:

   a. For each training example:
      - Compute z = w^T * x + b
      - Compute p = sigmoid(z) = 1/(1 + e^(-z))
   
   b. Compute the gradient:
      dw = (1/N) * X^T * (p - y)     # same form as linear regression!
      db = (1/N) * sum(p - y)
   
   c. Update weights:
      w = w - learning_rate * dw
      b = b - learning_rate * db

3. Return final w and b
```

**Remarkable fact:** The gradient formula $(p - y)$ has the same form as linear regression! The sigmoid function makes the math work out elegantly.

---

## The Full Pipeline: What Happens Behind the Scenes

```
TRAINING (.fit):

Raw Input         Linear Combination     Sigmoid            Log Loss
[age=55,          z = 0.05*55 +          p = 1/(1+e^-z)    Loss = -[y*log(p)
 bp=145,              0.03*145 +         p = 0.78                 + (1-y)*log(1-p)]
 chol=220]            0.02*220 - 8.5                        
                  z = 1.25               78% chance         Loss = 0.25
                                                            
                  Gradient Descent adjusts w and b           
                  to minimize total Loss across             
                  all patients                              


PREDICTION (.predict_proba and .predict):

New Patient       Linear         Sigmoid         Threshold        Decision
[age=60,          z = 0.05*60    p = 1/(1+e^-z)  p >= 0.5?        "High Risk"      
 bp=160,          + 0.03*160     p = 0.85         0.85 >= 0.5      YES
 chol=240]        + 0.02*240                      = True
                  - 8.5
                  z = 1.70
```

---

## What the Weights Mean (Interpretability)

Logistic regression is one of the most interpretable models:

```python
# After training:
model.coef_ = [0.05, 0.03, 0.02]     # weights for [age, bp, cholesterol]
model.intercept_ = [-8.5]              # bias

# Each weight represents the change in LOG-ODDS per unit increase:
# age weight = 0.05: each additional year increases log-odds by 0.05
# In terms of odds ratio: e^0.05 = 1.051
# Meaning: each year of age multiplies the odds of heart attack by 1.051 (5.1% increase)

# bp weight = 0.03: each mmHg increase multiplies odds by e^0.03 = 1.03 (3% increase)

# A 10-year age increase: odds multiplied by e^(0.05*10) = e^0.5 = 1.65 (65% more likely)
```

**This is why doctors, banks, and regulators love logistic regression** -- every coefficient has a clear meaning.

---

## Multiclass: What If There Are More Than 2 Classes?

Heart attack is yes/no (binary). But what if you're classifying flowers into 3 species?

### One-vs-Rest (OvR)

Train 3 separate binary classifiers:
- Classifier 1: "Is it Setosa?" (yes/no)
- Classifier 2: "Is it Versicolor?" (yes/no)
- Classifier 3: "Is it Virginica?" (yes/no)

Each outputs a probability. Pick the class with the highest probability.

### Softmax (Multinomial)

Instead of sigmoid, use the **softmax** function that distributes probability across all classes:

$$P(\text{class } k) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Each class gets its own set of weights. The probabilities always sum to 1.

**Example:**
- $z_{\text{setosa}} = 3.2$ -> $e^{3.2} = 24.5$
- $z_{\text{versicolor}} = 1.1$ -> $e^{1.1} = 3.0$
- $z_{\text{virginica}} = 0.5$ -> $e^{0.5} = 1.6$
- Sum = 29.1
- $P(\text{setosa}) = 24.5 / 29.1 = 84\%$
- $P(\text{versicolor}) = 3.0 / 29.1 = 10\%$
- $P(\text{virginica}) = 1.6 / 29.1 = 6\%$

---

## Regularization in Logistic Regression

Just like linear regression, logistic regression can overfit with too many features.

$$\text{Loss} = -\frac{1}{N} \sum \left[ y \log(p) + (1-y) \log(1-p) \right] + \lambda \sum w_j^2$$

In sklearn, the parameter `C` controls regularization:
- `C` = $1/\lambda$
- **Large C** (like 1000): Weak regularization, model fits training data closely
- **Small C** (like 0.01): Strong regularization, model is simpler, weights are smaller
- **Default C = 1.0**: Moderate regularization

```python
from sklearn.linear_model import LogisticRegression

# Strong regularization (simpler model)
model = LogisticRegression(C=0.01)  # C = 1/lambda, so small C = big penalty

# Weak regularization (complex model)
model = LogisticRegression(C=100)
```

---

## Decision Boundary: What Logistic Regression Actually Draws

In 2D (two features), logistic regression draws a **straight line** separating the classes:

```
Feature 2
  |    o o o o
  |   o o o/o          o = class 0 (healthy)
  |  o o o/ *          * = class 1 (sick)
  |  o o / * *
  |  o /  * * *        / = decision boundary (where P = 0.5)
  |  /  * * * *
  | / * * * * *
  |/ * * * * *
  +--------------------------- Feature 1
```

The decision boundary is always a **straight line** (or hyperplane in higher dimensions). This is the fundamental limitation of logistic regression -- it can't learn curved boundaries.

**If your classes aren't separable by a straight line**, logistic regression will fail. Fix: add polynomial features ($x_1^2$, $x_1 \cdot x_2$) or use a non-linear model.

---

## Logistic vs. Linear Regression: Side-by-Side

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|-------------------|
| **Output** | Any number ($-\infty$ to $+\infty$) | Probability (0 to 1) |
| **Task** | Regression (predict a number) | Classification (predict a category) |
| **Equation** | $\hat{y} = \mathbf{w}^T\mathbf{x} + b$ | $p = \text{sigmoid}(\mathbf{w}^T\mathbf{x} + b)$ |
| **Loss** | MSE: $(y - \hat{y})^2$ | Log Loss: $-y\log(p) - (1-y)\log(1-p)$ |
| **Solution** | Closed-form (normal equation) | Gradient descent only (no closed form) |
| **Decision boundary** | N/A | Straight line / hyperplane |

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Features not scaled | Gradient descent converges slowly, weights misleading | StandardScaler before fitting |
| Perfect separation | Weights go to infinity (model is "too confident") | Use regularization (reduce C) |
| Non-linear relationship | Straight decision boundary fails | Add polynomial features or use different model |
| Imbalanced classes (95% vs 5%) | Model predicts majority class for everything | Use `class_weight='balanced'`, adjust threshold, or resample |
| Using accuracy on imbalanced data | 95% accuracy by predicting "no" every time | Use AUC-ROC, F1, or precision/recall instead |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it learns** | Weights for each feature + bias (same as linear regression) |
| **How it learns** | Minimizes Log Loss via gradient descent |
| **Key transformation** | Sigmoid function squishes linear output to [0, 1] probability |
| **Prediction** | Probability -> apply threshold -> class label |
| **Output** | Probability (0-1) and class label (0/1) |
| **Decision boundary** | Always a straight line (linear) |
| **Interpretable?** | Yes -- weights = log-odds ratios, very popular in medicine/finance |
| **When to use** | Binary/multiclass classification, need probabilities, need interpretability |
| **When NOT to use** | Non-linear decision boundaries, complex feature interactions |
