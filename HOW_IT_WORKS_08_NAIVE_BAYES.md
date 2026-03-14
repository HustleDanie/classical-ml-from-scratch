# How It Works: Naive Bayes

> Naive Bayes uses probability theory to classify data -- specifically, it applies Bayes' Theorem with the "naive" assumption that all features are independent. Despite this unrealistic assumption, it works surprisingly well, especially for text classification.

---

## The One-Sentence Idea

For each possible class, Naive Bayes computes "how likely is this data point given this class?" and picks the class with the highest probability.

---

## Intuition: Thinking Like a Doctor

A patient comes in with symptoms: fever, cough, body aches.

A doctor thinks:
- "How common is fever among flu patients? Very common (90%)."
- "How common is cough among flu patients? Pretty common (80%)."
- "How common is these symptoms among cold patients? Fever 30%, cough 70%."
- "Flu is more likely."

Naive Bayes does exactly this calculation, but with math.

---

## The Math: Bayes' Theorem

### The Foundation

$$P(\text{class} | \text{features}) = \frac{P(\text{features} | \text{class}) \cdot P(\text{class})}{P(\text{features})}$$

In English:

$$\text{Probability of disease given symptoms} = \frac{\text{Probability of symptoms given disease} \times \text{How common is the disease}}{\text{How common are these symptoms overall}}$$

Let's label the pieces:
- $P(\text{class} | \text{features})$ = **Posterior** (what we want to compute)
- $P(\text{features} | \text{class})$ = **Likelihood** (how typical are these features for this class?)
- $P(\text{class})$ = **Prior** (how common is this class before seeing any features?)
- $P(\text{features})$ = **Evidence** (how common is this feature combination overall?)

### The "Naive" Assumption

The likelihood $P(\text{features} | \text{class})$ involves ALL features together:

$$P(x_1, x_2, ..., x_n | \text{class})$$

For example: $P(\text{fever AND cough AND body aches} | \text{flu})$

Computing this joint probability requires a huge amount of data (every possible combination of symptoms). With 20 features, each with 10 values, that's $10^{20}$ combinations!

**The naive assumption:** Features are **conditionally independent** given the class:

$$P(x_1, x_2, ..., x_n | \text{class}) = P(x_1 | \text{class}) \cdot P(x_2 | \text{class}) \cdot ... \cdot P(x_n | \text{class})$$

Now we only need $20 \times 10 = 200$ probabilities instead of $10^{20}$!

**Is this assumption true?** Almost never. Fever and body aches are correlated (both occur together in flu). But despite this wrong assumption, Naive Bayes works surprisingly well because:
1. The classification decision only needs the **ranking** of probabilities to be correct, not the exact values
2. Errors from the independence assumption often cancel out across features
3. For text classification, the assumption is "less wrong" than for other data types

### The Full Naive Bayes Formula

$$P(\text{class}_k | \mathbf{x}) \propto P(\text{class}_k) \prod_{j=1}^{n} P(x_j | \text{class}_k)$$

We don't need to compute $P(\text{features})$ because it's the same for all classes. We just compare the **numerators** and pick the largest.

---

## A Concrete Example: Spam Detection

**Training data:**

| Email | Contains "free" | Contains "meeting" | Contains "winner" | Spam? |
|-------|----------------|-------------------|--------------------|-------|
| 1     | Yes            | No                | Yes                | Spam  |
| 2     | Yes            | No                | No                 | Spam  |
| 3     | No             | Yes               | No                 | Not   |
| 4     | No             | Yes               | No                 | Not   |
| 5     | Yes            | No                | Yes                | Spam  |
| 6     | No             | Yes               | No                 | Not   |

### Step 1: Compute Priors

$$P(\text{Spam}) = 3/6 = 0.5$$
$$P(\text{Not Spam}) = 3/6 = 0.5$$

### Step 2: Compute Likelihoods

| Feature | P(feature=Yes \| Spam) | P(feature=Yes \| Not Spam) |
|---------|----------------------|--------------------------|
| "free"  | 3/3 = 1.0           | 0/3 = 0.0               |
| "meeting" | 0/3 = 0.0         | 3/3 = 1.0               |
| "winner"  | 2/3 = 0.67        | 0/3 = 0.0               |

### Step 3: Classify a New Email

New email contains: "free" = Yes, "meeting" = No, "winner" = No

**For Spam:**
$$P(\text{Spam}) \times P(\text{free=Yes}|\text{Spam}) \times P(\text{meeting=No}|\text{Spam}) \times P(\text{winner=No}|\text{Spam})$$
$$= 0.5 \times 1.0 \times 1.0 \times 0.33 = 0.167$$

**For Not Spam:**
$$P(\text{Not}) \times P(\text{free=Yes}|\text{Not}) \times P(\text{meeting=No}|\text{Not}) \times P(\text{winner=No}|\text{Not})$$
$$= 0.5 \times 0.0 \times 0.0 \times 1.0 = 0.0$$

**Problem!** $P(\text{free=Yes}|\text{Not Spam}) = 0$, which makes the entire product zero. One unseen feature combination kills the probability.

### Step 4: Laplace Smoothing (Fixing the Zero Problem)

Add a small count (usually 1) to every feature count:

$$P(x_j | \text{class}) = \frac{\text{count}(x_j, \text{class}) + \alpha}{\text{count}(\text{class}) + \alpha \cdot |\text{vocabulary}|}$$

Where $\alpha = 1$ (Laplace smoothing) and $|\text{vocabulary}|$ = number of possible values.

Now $P(\text{free=Yes}|\text{Not Spam}) = \frac{0 + 1}{3 + 2} = 0.2$ instead of 0.

---

## Three Variants of Naive Bayes

### 1. Gaussian Naive Bayes (Continuous Features)

Assumes each feature follows a **normal (Gaussian) distribution** within each class:

$$P(x_j | \text{class}_k) = \frac{1}{\sqrt{2\pi\sigma_{jk}^2}} \exp\left(-\frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2}\right)$$

During training: compute the mean ($\mu$) and standard deviation ($\sigma$) of each feature for each class.

```
Feature: Temperature

     Flu patients:              Cold patients:
     mean = 102.5 F             mean = 99.8 F
     std = 1.2 F                std = 0.8 F

     |    *                          |
     |   * *                         |     *
     |  *   *                        |    * *
     | *     *                       |   *   *
     |*       *                      |  *     *
  ---+---+---+---+---            ---+---+---+---+---
        101  103                       99   101
```

New patient: temp = 103 F
- $P(103 | \text{Flu})$ = Gaussian(103, mean=102.5, std=1.2) = high
- $P(103 | \text{Cold})$ = Gaussian(103, mean=99.8, std=0.8) = very low
- -> Lean toward Flu

**Use when:** Features are continuous (age, height, temperature, etc.)

### 2. Multinomial Naive Bayes (Count Features)

Assumes features are **counts** (how many times something appears):

$$P(\mathbf{x} | \text{class}_k) \propto \prod_j P(x_j | \text{class}_k)^{x_j}$$

**The star of text classification!** Features are word counts:

```
Email: "free money free winner free"
Feature vector: {free: 3, money: 1, winner: 1}

P("free" | Spam) = 0.15   (15% of words in spam emails are "free")
P("money" | Spam) = 0.08
P("winner" | Spam) = 0.05

P("free" | Not Spam) = 0.01
P("money" | Not Spam) = 0.02  
P("winner" | Not Spam) = 0.005
```

**Use when:** Text data (TF-IDF or word counts), or any count-based features.

### 3. Bernoulli Naive Bayes (Binary Features)

Assumes features are **binary** (present or not):

$$P(x_j | \text{class}_k) = P(x_j = 1 | \text{class}_k)^{x_j} \cdot (1 - P(x_j = 1 | \text{class}_k))^{(1-x_j)}$$

Unlike Multinomial, Bernoulli **explicitly models absence**. If "free" is NOT in the email, it uses $P(\text{free=absent} | \text{class})$ as evidence.

**Use when:** Binary features (word present/absent, symptom yes/no).

**Key difference from Multinomial:** Bernoulli penalizes for missing words. "The word 'meeting' is absent" is evidence against Not Spam (because "meeting" is common in non-spam).

---

## What Happens Behind the Scenes: `.fit()` and `.predict()`

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

# .fit(X, y) does this:
# For each class k:
#   1. Count how many samples belong to class k -> P(class_k) = count_k / N
#   2. For each feature j:
#      - Compute mean of feature j for samples in class k -> mu_jk
#      - Compute std of feature j for samples in class k -> sigma_jk
#   3. Store these parameters
#
# That's it! Just counting and computing means/stds.
# No iteration, no optimization, no gradient descent!
model.fit(X_train, y_train)

print(model.class_prior_)    # [0.65, 0.35]  -- 65% class 0, 35% class 1
print(model.theta_)          # means for each feature for each class
print(model.var_)            # variances for each feature for each class

# .predict(X) does this:
# For each test point:
#   For each class k:
#     score_k = log(P(class_k)) + sum(log(P(x_j | class_k)))
#   Return class with highest score
#
# (Uses log probabilities to avoid tiny number underflow)
predictions = model.predict(X_test)
```

### Why Log Probabilities?

Multiplying many small probabilities gives TINY numbers:

$$0.01 \times 0.05 \times 0.002 \times 0.03 \times ... = 0.0000000000003$$

Computers can't represent numbers this small (underflow). Solution: work in log space:

$$\log P = \log P(\text{class}) + \sum_j \log P(x_j | \text{class})$$

Multiplication becomes addition. Numbers stay in a manageable range.

---

## The Full Pipeline

```
TRAINING (.fit):  ← Very fast! Just count and compute statistics.

Class 0 (healthy): 650 patients
  age:  mean=42, std=12
  bp:   mean=120, std=10
  chol: mean=190, std=25

Class 1 (sick): 350 patients
  age:  mean=58, std=10
  bp:   mean=150, std=15
  chol: mean=240, std=30

Prior: P(healthy)=0.65, P(sick)=0.35


PREDICTION (.predict):

New patient: age=55, bp=145, chol=230

For Healthy:
  log P(H) + log P(age=55|H) + log P(bp=145|H) + log P(chol=230|H)
  = log(0.65) + log(Gauss(55, 42, 12)) + log(Gauss(145, 120, 10)) + log(Gauss(230, 190, 25))
  = -0.43 + (-1.54) + (-4.63) + (-1.89)
  = -8.49

For Sick:
  log P(S) + log P(age=55|S) + log P(bp=145|S) + log P(chol=230|S)
  = log(0.35) + log(Gauss(55, 58, 10)) + log(Gauss(145, 150, 15)) + log(Gauss(230, 240, 30))
  = -1.05 + (-1.34) + (-1.29) + (-1.41)
  = -5.09    <-- HIGHER (less negative)

Predict: SICK (log probability -5.09 > -8.49)
```

---

## Why Naive Bayes Is Great for Text Classification

Text data has special properties that match Naive Bayes well:

1. **Very high dimensional** (10,000+ unique words) -- Naive Bayes handles this easily because it just needs P(word | class) for each word
2. **Sparse** (each document uses a tiny fraction of all words) -- works fine with the independence assumption
3. **The independence assumption is "less wrong"** -- word occurrences are somewhat independent given the topic
4. **Training is blazing fast** -- just count word frequencies per class

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Text classification pipeline
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(train_emails)  # 10000 features

model = MultinomialNB(alpha=1.0)  # alpha=1.0 is Laplace smoothing
model.fit(X_train_tfidf, y_train)  # Trains in milliseconds on 100K emails!

# Prediction: compute P(spam|words) vs P(not_spam|words)
```

---

## Naive Bayes vs. Logistic Regression

Both are linear classifiers. Here's how they compare:

| Aspect | Naive Bayes | Logistic Regression |
|--------|------------|-------------------|
| **Approach** | Generative (models P(features\|class)) | Discriminative (models P(class\|features) directly) |
| **Training** | Count/compute statistics (no iteration) | Gradient descent (iterative) |
| **Speed** | Extremely fast | Fast but slower than NB |
| **Small data** | Better (less prone to overfitting) | Worse (needs more data) |
| **Large data** | Logistic Regression catches up | Better accuracy |
| **Feature independence** | Assumes independence (naive) | No independence assumption |
| **Probability calibration** | Probabilities are often poorly calibrated | Better calibrated probabilities |
| **Accuracy on text** | Very good | Slightly better with enough data |

**Rule of thumb:** Naive Bayes works better with small training sets; Logistic Regression works better with large training sets.

---

## Common Pitfalls

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Zero probability (unseen feature value) | Entire class probability becomes 0 | Use Laplace smoothing (`alpha=1.0`) |
| Using Gaussian NB on count data | Gaussian doesn't fit counts well | Use MultinomialNB for counts, BernoulliNB for binary |
| Correlated features | Independence assumption severely violated | Consider Logistic Regression instead |
| Trusting the probability values | NB probabilities are often too extreme (0.001 or 0.999) | Calibrate with CalibratedClassifierCV |
| Using on tabular data with interactions | Misses feature interactions completely | Use tree-based models instead |
| Not applying Laplace smoothing | Crashes on unseen feature values in test data | Always set `alpha > 0` |

---

## Summary

| Aspect | Detail |
|--------|--------|
| **What it learns** | Prior probabilities P(class) + likelihood parameters P(feature\|class) for each feature |
| **How it learns** | Counts and statistics (no optimization!) |
| **Key assumption** | Features are conditionally independent given the class |
| **Prediction** | Pick class with highest P(class) * product of P(features\|class) |
| **Variants** | Gaussian (continuous), Multinomial (counts), Bernoulli (binary) |
| **Speed** | Extremely fast to train AND predict |
| **Feature scaling?** | Not needed for Multinomial/Bernoulli. Doesn't matter much for Gaussian. |
| **Interpretable?** | Yes -- P(word\|spam) directly tells you which words indicate spam |
| **When to use** | Text classification, small datasets, need speed, real-time classification |
| **When NOT to use** | Correlated features, need accurate probabilities, complex feature interactions |
