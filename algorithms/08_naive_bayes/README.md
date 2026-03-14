# 08 — Naive Bayes

## Overview

Naive Bayes is a family of **probabilistic classifiers** built on **Bayes' theorem** with a strong ("naive") assumption that all features are **conditionally independent** given the class label. Despite this unrealistic assumption, Naive Bayes performs remarkably well in practice — especially on text classification, spam detection, and other high-dimensional problems.

It is one of the fastest classifiers to train and predict, making it an excellent baseline model.

---

## Bayes' Theorem

The foundation of all Naive Bayes classifiers:

$$P(C \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid C) \cdot P(C)}{P(\mathbf{x})}$$

Where:
- $P(C \mid \mathbf{x})$ — **Posterior**: probability of class $C$ given features $\mathbf{x}$
- $P(\mathbf{x} \mid C)$ — **Likelihood**: probability of observing $\mathbf{x}$ in class $C$
- $P(C)$ — **Prior**: baseline probability of class $C$
- $P(\mathbf{x})$ — **Evidence**: probability of observing $\mathbf{x}$ (constant across classes)

Since $P(\mathbf{x})$ is the same for all classes, we only need to maximize the numerator:

$$\hat{C} = \arg\max_C \; P(\mathbf{x} \mid C) \cdot P(C)$$

---

## The "Naive" Assumption

The key simplification: **features are conditionally independent given the class**.

$$P(\mathbf{x} \mid C) = P(x_1 \mid C) \cdot P(x_2 \mid C) \cdots P(x_n \mid C) = \prod_{i=1}^{n} P(x_i \mid C)$$

This reduces the estimation problem from learning a joint distribution over all features (exponential in $n$) to learning $n$ independent distributions (linear in $n$).

### Why "Naive" Still Works

1. **Classification only needs the correct ranking** — even if probabilities are poorly calibrated, the most probable class is often correct
2. **Errors from independence assumption often cancel out** across features
3. **High-dimensional data** makes joint estimation impossible anyway — Naive Bayes provides a tractable approximation
4. **Regularizing effect** — the independence assumption prevents overfitting

---

## Naive Bayes Variants

The variants differ in how they model $P(x_i \mid C)$ — the likelihood of each feature given the class.

### 1. Gaussian Naive Bayes (GaussianNB)

Assumes each feature follows a **normal distribution** within each class:

$$P(x_i \mid C) = \frac{1}{\sqrt{2\pi\sigma_{C,i}^2}} \exp\left(-\frac{(x_i - \mu_{C,i})^2}{2\sigma_{C,i}^2}\right)$$

**Training**: Compute mean $\mu_{C,i}$ and variance $\sigma_{C,i}^2$ for each feature $i$ in each class $C$.

**Best for**: Continuous features that are roughly normally distributed (e.g., Iris, medical measurements).

### 2. Multinomial Naive Bayes (MultinomialNB)

Models features as **counts** or **frequencies** (multinomial distribution):

$$P(x_i \mid C) = \frac{N_{C,i} + \alpha}{N_C + \alpha \cdot n}$$

Where:
- $N_{C,i}$ = count of feature $i$ in class $C$
- $N_C$ = total count of all features in class $C$
- $\alpha$ = smoothing parameter (Laplace smoothing)
- $n$ = number of features

**Best for**: Text classification with word counts or TF-IDF features (bag-of-words).

### 3. Bernoulli Naive Bayes (BernoulliNB)

Models features as **binary** (present/absent):

$$P(x_i \mid C) = P(i \mid C)^{x_i} \cdot (1 - P(i \mid C))^{(1 - x_i)}$$

**Key difference from Multinomial**: Bernoulli explicitly penalizes the **absence** of a feature, while Multinomial ignores missing features.

**Best for**: Binary features, short text, document classification where feature presence/absence matters.

### 4. Complement Naive Bayes (ComplementNB)

A modification of MultinomialNB designed for **imbalanced datasets**. Instead of estimating $P(x_i \mid C)$, it estimates $P(x_i \mid \bar{C})$ — the probability given all classes *except* $C$:

$$\hat{C} = \arg\min_C \; \sum_{i=1}^{n} w_{C,i} \cdot x_i$$

**Best for**: Imbalanced text classification (e.g., the minority class in sentiment analysis).

---

## Laplace Smoothing

Without smoothing, if a feature value never appears in a class, $P(x_i \mid C) = 0$, making the entire posterior zero. **Laplace smoothing** (additive smoothing) prevents this:

$$P(x_i \mid C) = \frac{\text{count}(x_i, C) + \alpha}{\text{count}(C) + \alpha \cdot |V|}$$

- $\alpha = 1$: Laplace smoothing (uniform prior)
- $\alpha < 1$: Lidstone smoothing
- $\alpha = 0$: No smoothing (risky — zero probabilities possible)

---

## Log Probabilities

In practice, we work in **log space** to avoid numerical underflow when multiplying many small probabilities:

$$\log P(C \mid \mathbf{x}) \propto \log P(C) + \sum_{i=1}^{n} \log P(x_i \mid C)$$

This converts products into sums, which are numerically stable.

---

## Prior Probabilities

The **prior** $P(C)$ is estimated from the training data:

$$P(C) = \frac{\text{Number of samples in class } C}{\text{Total number of samples}}$$

For imbalanced datasets, priors naturally up-weight the majority class. You can override them with uniform priors or custom weights.

---

## Strengths & Weaknesses

### Strengths
- **Extremely fast** training and prediction (O(n·d))
- Works well with **high-dimensional** data (text, genomics)
- Needs **very little training data** to estimate parameters
- **Naturally handles multi-class** problems
- **Not sensitive to irrelevant features** (they contribute equally to all classes)
- Great **baseline** model

### Weaknesses
- **Poor probability calibration** — predicted probabilities are often extreme (near 0 or 1)
- **Independence assumption** is rarely true → can't capture feature interactions
- **Sensitive to feature distribution** — GaussianNB assumes normality
- **Zero-frequency problem** without smoothing
- Generally **lower accuracy** than ensemble methods on structured/tabular data

---

## When to Use Naive Bayes

| Scenario | Recommendation |
|----------|---------------|
| Text classification (spam, sentiment) | **MultinomialNB** — gold standard |
| Binary features (document has/lacks word) | **BernoulliNB** |
| Continuous features, small dataset | **GaussianNB** |
| Imbalanced text | **ComplementNB** |
| Need fast baseline | Any variant |
| Feature interactions matter | Don't use NB — try trees/ensembles |

---

## What This Implementation Covers

1. **From-scratch Gaussian Naive Bayes** — computing class priors, per-class means/variances, Gaussian PDF, log-probability prediction
2. **From-scratch Multinomial Naive Bayes** — with Laplace smoothing, log-space computation
3. **Scikit-learn comparison** — GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
4. **Text classification demo** — 20 Newsgroups with TF-IDF + MultinomialNB
5. **Smoothing parameter impact** — alpha sweep
6. **Prior probability visualization** — class distribution and learned priors
7. **Feature likelihood visualization** — Gaussian PDFs per class
8. **Calibration analysis** — predicted probability vs actual frequency
9. **Naive Bayes vs other classifiers** — benchmarking
10. **Multiclass classification** — Iris dataset with GaussianNB

---

## How to Run

```bash
cd 08_naive_bayes
python naive_bayes.py
```

All plots are saved to `plots/`.
