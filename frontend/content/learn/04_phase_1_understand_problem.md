# Phase 1: Understand the Problem

> Before any code: define the prediction, audit the data, pick the metric.

## What this phase produces

By the end of Phase 1 you have:

1. A **one-sentence problem statement** — *"Given X, predict Y."*
2. The **problem type** — regression, binary classification, multiclass, multilabel, clustering, or survival.
3. A **constraint profile** — which of the 8 signals apply (imbalance, regulation, real-time, cost asymmetry, time-series, multilabel, calibration, fairness).
4. A **data inventory** — rows, features, types, missingness, leakers, temporal axis.
5. A **metric** — primary + secondary.

If any one is missing, stop and fix it before Phase 2.

---

## Step 1.1 — Define the problem type

Reduce the brief to one sentence: *"Given X, predict Y."* If you can't write that sentence, the brief is ambiguous — clarify before continuing.

| You predict… | Problem type |
|---|---|
| A number | Regression |
| Binary outcome | Binary classification |
| One of 3+ classes | Multiclass |
| Multiple labels per row | Multilabel |
| Time-until-event with censoring | Survival |
| Group assignment with no labels | Clustering |

---

## Step 1.2 — Inventory the data (5 minutes)

```python
print(df.shape)
print(df.dtypes.value_counts())
print(df['target'].value_counts(normalize=True))   # imbalance
print(df['target'].skew(), df['target'].quantile([0.5, 0.99]).tolist())   # regression target shape
print(df.isnull().mean().sort_values(ascending=False).head())   # missingness
```

Check for: ID columns to drop, temporal axis, protected attributes, **leakers** (features that wouldn't exist at prediction time).

---

## Step 1.3 — Identify the 8 constraints

| Signal | Phrases | What it forces |
|---|---|---|
| Imbalance | "rare", "0.x%" | PR-AUC over ROC-AUC; threshold tuning |
| Regulation | "ECOA", "audit", "adverse action" | Linear models; calibration; per-group fairness |
| Real-time | "<100 ms" | Single model, no stacking |
| Cost asymmetry | "$A vs $B per error" | Cost-driven threshold |
| Time series | "next hour", sensor stream | Time-based split; rolling features |
| Multilabel | "tags", "categories" (plural) | One-vs-rest; per-label metrics |
| Calibration | "probability consumed downstream" | Brier score; isotonic/Platt |
| Fairness | "disparate impact", protected groups | Per-group metrics |

A real brief fires 2-3 of these.

---

## Step 1.4 — Choose a metric

| Brief shape | Primary | Secondary |
|---|---|---|
| Imbalanced classification | PR-AUC | recall@k (if budget) |
| Probabilities consumed | + Brier | calibration plot |
| Dollar costs given | Cost function | recall@k |
| Skewed regression target | MAE | MAPE (if strictly positive) |
| Symmetric regression | RMSE | MAE |
| Asymmetric regression cost | Quantile loss | — |

**Never report alone**: accuracy, ROC-AUC under imbalance, F1 @ 0.5, RMSE on a skewed target.

---

## Common mistakes

| Mistake | Fix |
|---|---|
| Using accuracy on imbalanced data | PR-AUC or recall@k |
| Treating "score the risk" as regression | It's calibrated classification |
| Missing the leaker hidden in the data | Audit suspicious-high-correlation features |
| Random split on temporal data | Time-based split (Phase 3) |
| One metric only | Always primary + secondary |
