# Expert Scenario 33: Document Tags (Multilabel Classification)

> **Complexity:** Multilabel (each document can have multiple tags), 30+ tag vocabulary, tag co-occurrence patterns matter, threshold tuning per-label, evaluation metrics that work for multilabel are non-obvious, head/tail tag imbalance.

---

## The Brief

A research repository (think arXiv-style) gives you 1.2M papers, each tagged with 1-5 of 32 possible subject categories (e.g., "Machine Learning", "Computer Vision", "Statistics — Methodology", "NLP — Generation"). Authors self-tag at submission, so the labels are noisy but mostly correct. You're asked to:

- Auto-suggest tags for new submissions (top-5 ranked).
- Auto-correct or auto-add when authors miss obvious tags.
- Surface tag co-occurrence patterns to help curators.
- Run inference at submission time (< 2 seconds per paper).

Constraints:

- Some tags are dominant (40% of papers tagged "Machine Learning"). Some are rare (< 0.5% tagged "Algebraic Topology").
- Average paper has 2.8 tags; max is 7.
- Per-tag F1 must average > 0.65 (micro-F1 is the headline; macro-F1 should also be tracked).
- Cannot use a deep transformer (compute budget) — must work with classical ML on TF-IDF features.

Multilabel is structurally different from multiclass. Each label is independent (in principle), but in practice tags co-occur — a paper tagged "NLP" almost always also gets "Machine Learning". The model must capture this without overfitting.

---

## Step 1: Define the Problem Type

```
Type:           Multilabel binary classification (32 binary labels per row)
Primary Metric: Micro-F1 (averages across labels weighted by support)
Secondary:      Macro-F1 (treats each label equally — penalizes ignoring rare tags)
                Subset accuracy (exact-match -- harsh; tracked but not optimized)
                Per-label F1 distribution (no label should fall below 0.4)
Business Goal:  Top-5 suggested tags include the right ones for 90% of papers
Constraint:     Inference < 2s; classical ML only
Tag distribution: head-heavy -- top 8 tags cover 70% of label-instances
```

**Expert thinking:** the structural choice is Binary Relevance (BR — one binary classifier per label) vs Classifier Chains (CC — model label dependencies) vs Label Powerset (LP — treat each combination as a class, explodes for 32 labels). For 32 labels:
- BR is fast, ignores correlations, 32 models.
- CC captures correlations, slower, more complex.
- LP has 2^32 possible classes — infeasible.

We'll start with BR as the baseline and see if CC adds enough lift to be worth it.

---

## Step 2: Understand the Data

```
Shape: 1,200,000 rows x text + metadata
Target: 32 binary tag indicators

Columns:
- paper_id (string, unique)
- title (string)
- abstract (string, ~150-1500 chars)
- author_count (int)
- num_references (int)
- num_figures (int)
- submission_date (datetime)
- tags (list of strings, len 1-7)

Tag distribution:
- "Machine Learning"        491,000 (41%)
- "Computer Vision"          210,000 (18%)
- "Statistics — Methodology" 156,000 (13%)
- "NLP — Foundations"        102,000 (8.5%)
- "Optimization"              89,000 (7.4%)
- ... 27 more tags ...
- "Algebraic Topology"         3,200 (0.27%)
- "Quantum Information"        2,100 (0.18%)

Tag count per paper:
- 1 tag:  102,000 (8.5%)
- 2 tags: 421,000 (35.1%)
- 3 tags: 401,000 (33.4%)  <- mode
- 4 tags: 188,000 (15.7%)
- 5 tags:  64,000 (5.3%)
- 6+ tags: 24,000 (2.0%)
```

**Expert thinking:** tags will hit very different sample sizes. "Machine Learning" sees 491K positive examples — easy. "Quantum Information" sees 2,100 — will need careful handling (lower min_df, possibly upsampling specific to this label).

---

## Step 3: Exploratory Data Analysis (EDA)

```python
# Tag co-occurrence matrix
from scipy.sparse import lil_matrix
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y_binary = mlb.fit_transform(df['tags'])  # 1.2M x 32

# Co-occurrence: Y^T Y / N
co_occ = (y_binary.T @ y_binary) / len(df)
# Diagonal = marginal probability of each tag
# Off-diagonal = joint probability

# Conditional probability P(tag_j | tag_i)
joint = y_binary.T @ y_binary
support = y_binary.sum(axis=0)
cond = joint / support[:, None]
# cond[i, j] = P(tag_j | tag_i)
```

**Findings:**

| Finding | Implication |
|---------|------------|
| P("ML" \| "NLP — Foundations") = 0.94 | NLP papers almost always also tagged ML — a clear correlation to exploit |
| P("ML" \| "Quantum Information") = 0.12 | Quantum papers rarely tagged ML — tag is structurally distinct |
| Per-paper tag count median is 3 | Top-3 ranked output likely captures most signal |
| Top 5 tags cover 70% of label-instances | Head/tail problem; rare tags need separate handling |
| Some tag pairs are mutually exclusive (CV ↔ Theoretical CS) | Could exploit, but rare |
| Author-self-tagged → ~5% noise | Some papers missing obvious tag (e.g., NLP paper not tagged ML) |
| Submission year 2018-2024 | "Machine Learning" tag share grew from 28% to 48% over 6 years -- temporal drift |

**Expert insight:** tag co-occurrence isn't symmetric. P(ML | NLP) >> P(NLP | ML). Classifier Chains can exploit this if we order the chain such that "parent" tags come first (ML before NLP, ML before CV, etc.).

---

## Step 4: Data Cleaning

```python
# === TEXT CLEANING ===
# Combine title + abstract; title gets 3x weight (it's denser)
df['text'] = (df['title'] + ' ') * 3 + df['abstract']

# Standard NLP cleaning:
import re
def clean_text(t):
    t = t.lower()
    t = re.sub(r'\d+', 'NUM', t)          # numbers cluster
    t = re.sub(r'\$[^$]+\$', 'EQUATION', t)  # LaTeX equations
    t = re.sub(r'http[s]?://\S+', 'URL', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

df['text_clean'] = df['text'].apply(clean_text)

# === LABEL CLEANUP ===
# Some tags appear in 2-3 spelling variants ("NLP - Foundations" vs "NLP — Foundations")
df['tags'] = df['tags'].apply(lambda lst: [normalize_tag(t) for t in lst])

# Drop papers with 0 tags (1.5% of corpus -- annotators forgot)
df = df[df['tags'].apply(len) > 0]

# === TEMPORAL SPLIT ===
df = df.sort_values('submission_date')
# We'll split by date in step 8, but flag the temporal drift now
```

---

## Step 5: Feature Engineering

```python
# === TEXT FEATURES ===
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100000,
    min_df=10,           # rare tags need words appearing in at least 10 papers
    max_df=0.85,
    sublinear_tf=True,
    stop_words='english'
)
X_text = tfidf.fit_transform(df['text_clean'])

# === STRUCTURAL FEATURES (small dense set) ===
df['title_word_count'] = df['title'].str.split().str.len()
df['abstract_word_count'] = df['abstract'].str.split().str.len()
df['has_equations'] = df['abstract'].str.contains(r'\$[^$]+\$').astype(int)
df['has_code'] = df['abstract'].str.contains(r'github|code', case=False).astype(int)
df['has_dataset'] = df['abstract'].str.contains(r'dataset|corpus|benchmark', case=False).astype(int)
df['has_theorem'] = df['abstract'].str.contains(r'theorem|proof|lemma', case=False).astype(int)
df['author_count_log'] = np.log1p(df['author_count'])
df['num_references_log'] = np.log1p(df['num_references'])
df['num_figures_log'] = np.log1p(df['num_figures'])

# === TEMPORAL ===
df['submission_year'] = df['submission_date'].dt.year
# One-hot or as drift-correction feature

# Stack: 100K TF-IDF + 10 dense
X_dense = StandardScaler().fit_transform(df[dense_cols])
from scipy.sparse import hstack
X = hstack([X_text, X_dense]).tocsr()
```

---

## Step 6: Feature Selection

```python
# Per-label feature relevance differs:
# - "Machine Learning" cares about words: gradient, training, neural, model, learn
# - "Algebraic Topology" cares about words: manifold, homology, sheaf, fiber bundle
#
# Per-label L1 regularization in the model handles this automatically.
# Skip global feature selection -- let each binary classifier select its own subset.
```

---

## Step 7: Preprocessing — Already done in Step 5

---

## Step 8: Train/Test Split

```python
# TIME-BASED SPLIT (drift is real -- ML tag share grew 28% -> 48% over 6 years)
df = df.sort_values('submission_date')
cutoff = df['submission_date'].quantile(0.85)
train = df[df['submission_date'] < cutoff]   # ~1.02M papers
test = df[df['submission_date'] >= cutoff]   # ~180K papers

# Verify tag distribution holds across split
# (it won't perfectly -- but no tag should drop > 50% in proportion)
print(train.explode('tags')['tags'].value_counts(normalize=True).head())
print(test.explode('tags')['tags'].value_counts(normalize=True).head())
```

---

## Step 9: Baselines

```python
# Baseline 1: Most-common tags ("Machine Learning")
# Predict every paper has top tag.
# Micro-F1: 0.41 / 1.0 = 0.58 (since "ML" is on 41% of papers)
# Useless.

# Baseline 2: Random tag assignment respecting per-tag support
# Micro-F1: ~0.05

# Baseline 3: Simple keyword rules
# (e.g., if "neural network" in abstract, tag ML)
# Micro-F1: 0.51
```

---

## Step 10: Try Multiple Models with Stratified-Multilabel CV

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold

# For multilabel, regular stratified CV doesn't work directly
# Use iterative stratification (preserves multilabel structure across folds)
from skmultilearn.model_selection import IterativeStratification
cv = IterativeStratification(n_splits=5, order=2)

# === MODEL 1: Binary Relevance with Logistic Regression L2 ===
model1 = OneVsRestClassifier(
    LogisticRegression(C=1.0, penalty='l2', max_iter=500, class_weight='balanced'),
    n_jobs=-1
)
# CV Micro-F1: 0.683
# CV Macro-F1: 0.412 (rare tags drag down)
# Per-label F1 spread: min 0.07 (Quantum Info), max 0.84 (ML)

# === MODEL 2: Binary Relevance with Logistic Regression L1 ===
model2 = OneVsRestClassifier(
    LogisticRegression(C=1.0, penalty='l1', solver='liblinear', class_weight='balanced'),
    n_jobs=-1
)
# CV Micro-F1: 0.679
# Slightly worse than L2 but more interpretable (sparse features per label)

# === MODEL 3: Linear SVM (one-vs-rest) ===
model3 = OneVsRestClassifier(LinearSVC(C=1.0, class_weight='balanced'), n_jobs=-1)
# CV Micro-F1: 0.674
# No probabilities (need calibration for thresholding)

# === MODEL 4: ComplementNB (designed for imbalanced text) ===
from sklearn.naive_bayes import ComplementNB
model4 = OneVsRestClassifier(ComplementNB(alpha=0.1), n_jobs=-1)
# CV Micro-F1: 0.712  <-- surprising winner!
# CV Macro-F1: 0.451

# === MODEL 5: Classifier Chains (CC) ===
from sklearn.multioutput import ClassifierChain
# Order chain by tag frequency: ML first (most common), niche tags last
order = np.argsort(-y_train.sum(axis=0))
model5 = ClassifierChain(LogisticRegression(C=1.0, class_weight='balanced'), order=order)
# CV Micro-F1: 0.701
# CC captures co-occurrence -- "if ML is predicted, it changes the prior on other tags"
```

**Top performer:** ComplementNB (0.712 micro-F1). CC is second (0.701).

**Why ComplementNB wins:** for high-imbalance text, CNB is built specifically for this case. It computes class-conditional probability using the *complement* of each class, which is more robust when one class is much smaller. Standard NLP wisdom.

---

## Step 11: Per-Label Threshold Tuning

```python
# OneVsRest with default 0.5 threshold is suboptimal -- per-label thresholds help massively
# Especially for rare tags

y_proba = model1.predict_proba(X_val)  # shape: (n_val, 32)

# For each label, find the threshold that maximizes F1 on validation
best_thresholds = np.zeros(32)
for i, label in enumerate(label_names):
    proba_i = y_proba[:, i]
    y_true_i = y_val[:, i]
    # Sweep thresholds and pick max F1
    best_thr, best_f1 = 0.5, 0
    for thr in np.arange(0.05, 0.95, 0.05):
        y_pred_i = (proba_i >= thr).astype(int)
        f1 = f1_score(y_true_i, y_pred_i, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    best_thresholds[i] = best_thr

# Per-label thresholds typically range 0.15 (rare tags) to 0.55 (head tags)
# Apply at inference:
y_pred = (y_proba >= best_thresholds).astype(int)

# Re-evaluate:
# Micro-F1: 0.711 (+0.028 from tuning)
# Macro-F1: 0.481 (+0.069 from tuning -- big lift on rare tags)
```

**Expert insight:** for multilabel with imbalanced labels, default 0.5 threshold is one of the biggest mistakes. Each label has its own optimum, often very different. The lift on macro-F1 (which weights all labels equally) is dramatic.

---

## Step 12: Hyperparameter Tuning

```python
import optuna

def objective(trial):
    alpha = trial.suggest_float('alpha', 0.001, 5.0, log=True)
    norm = trial.suggest_categorical('norm', [True, False])
    cnb = OneVsRestClassifier(ComplementNB(alpha=alpha, norm=norm), n_jobs=-1)
    micro_f1s = []
    for tr_idx, va_idx in cv.split(X_train, y_train):
        cnb.fit(X_train[tr_idx], y_train[tr_idx])
        proba = cnb.predict_proba(X_train[va_idx])
        # Apply pre-tuned thresholds
        pred = (proba >= best_thresholds).astype(int)
        micro_f1s.append(f1_score(y_train[va_idx], pred, average='micro'))
    return -np.mean(micro_f1s)

study = optuna.create_study()
study.optimize(objective, n_trials=40)
# Best: alpha=0.06, norm=False
# Tuned CV Micro-F1: 0.728
```

---

## Step 13: Tag-Cardinality Calibration

```python
# Multilabel models tend to predict either too many or too few tags per row
# Average actual tags per paper: 2.8
# Average predicted tags: 3.4 (over-tagging)

# Tactic: for each paper, only keep top-K tags by probability
# K = round(2.8) = 3 if we want to match average; or use probability cutoff

# Two-pass approach:
# Pass 1: predict all tags above per-label threshold
# Pass 2: if total tags > 5, keep top-5 by probability ; if 0, keep top-1

predictions = []
for proba_row in y_proba:
    flagged = np.where(proba_row >= best_thresholds)[0]
    if len(flagged) > 5:
        flagged = np.argsort(-proba_row)[:5]
    elif len(flagged) == 0:
        flagged = [np.argmax(proba_row)]
    predictions.append(flagged)
```

---

## Step 14: Ensembling

```python
# Top 3 models make different errors -- ensemble them
from sklearn.ensemble import VotingClassifier

# For multilabel, use prediction averaging
ensemble_proba = (
    0.5 * cnb.predict_proba(X_val) +
    0.3 * model5.predict_proba(X_val) +  # Classifier Chains
    0.2 * model1.predict_proba(X_val)    # OvR LR
)
# Apply per-label thresholds
y_ensemble = (ensemble_proba >= best_thresholds).astype(int)

# CV Micro-F1: 0.741
# CV Macro-F1: 0.502

# Marginal improvement, but accepted for production
```

---

## Step 15: Final Evaluation on Held-Out Test Set

```python
# Final model: Ensemble (CNB 50% + CC 30% + OvR LR 20%) with per-label thresholds + cardinality cap

# Test set: ~180,000 papers from 2024
# Performance:
#   Micro-F1:                   0.736
#   Macro-F1:                   0.498
#   Subset Accuracy (exact):    0.412
#   Top-1 hit rate (most prob tag in true tags): 0.91
#   Top-3 hit rate:                              0.94
#   Top-5 hit rate:                              0.96  ✓ (target 90%)
#
# Per-label F1 distribution:
#   Top 8 tags (head): mean F1 0.81
#   Mid 16 tags:       mean F1 0.62
#   Bottom 8 tags:     mean F1 0.34  <- below target
```

The bottom 8 tags (rarest) are still problematic. Possible mitigation:
- Active learning loop: surface uncertain rare-tag predictions to human curators for labeling.
- Drop rare tags from auto-suggest UI; surface them only as "consider also..." suggestions.

---

## Step 16: Explainability

```python
# For Logistic Regression component, top words per label are the explanation
# For each tag, surface the top-positive coefficient features

for i, label in enumerate(label_names):
    coefs = model1.estimators_[i].coef_[0]
    feature_names_text = tfidf.get_feature_names_out()
    top_words = np.argsort(coefs)[-10:][::-1]
    print(f"\nTag '{label}' top features:")
    for idx in top_words:
        print(f"  {feature_names_text[idx]:<25} {coefs[idx]:+.3f}")

# Tag 'Machine Learning' top features:
#   neural network             +2.84
#   gradient descent           +2.61
#   training                   +2.34
#   loss                       +2.07
#   model                      +1.92

# Tag 'Algebraic Topology' top features:
#   homology                   +4.12
#   manifold                   +3.78
#   sheaf                      +3.21
#   fiber bundle               +3.05

# Per-paper explanation for tag suggestion:
# "We suggest tag 'NLP — Foundations' because: paper contains 'attention mechanism' (+0.84),
#  'language model' (+0.72), 'tokenization' (+0.61)."
```

---

## Step 17: Deployment Considerations

```python
# Model size: ~120MB (TF-IDF vocabulary + 32 binary classifiers + thresholds)
# Inference: ~250ms per paper (computing 32 probabilities + threshold + cardinality logic)

# Production:
#   1. Pre-trained TF-IDF + ensemble pickled
#   2. Per-label thresholds in a JSON config (adjustable without retraining)
#   3. POST /suggest_tags endpoint takes title + abstract, returns top-5 ranked tags + per-tag confidence
#   4. Human curator UI: see suggestions + override if needed; their override feeds active learning

# Retraining: MONTHLY
#   - Tag share drifts (we saw ML tag grow 28%→48% over 6 years)
#   - Monthly retrain on rolling 4-year window
#   - Re-optimize per-label thresholds each retrain

# Monitoring:
#   - Daily Micro-F1 on confirmed tags (curator feedback loop)
#   - Per-label F1 drift: alert if any label drops > 0.05 from baseline
#   - Tag-cardinality calibration: alert if avg predicted tags drifts from avg actual
#   - New emerging tags: track unsupervised topic clusters in untagged predictions
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Multilabel framing | Use multiclass with stitched-string labels | Proper Binary Relevance / Classifier Chains |
| Threshold | 0.5 default for all 32 labels | Per-label tuned thresholds (range 0.15–0.55) |
| Imbalance per label | Same `class_weight` setting for all | ComplementNB which is designed for imbalance |
| Co-occurrence | Ignore (BR assumes label independence) | Classifier Chains ordered head-tag-first |
| Tag count per paper | Predict any number | Cardinality cap: top-5 max, top-1 min |
| CV strategy | Standard KFold | Iterative stratification (preserves multilabel structure) |
| Eval metric | Subset accuracy | Micro-F1 (headline) + Macro-F1 (rare tags) + per-label distribution |
| Train/test split | Random | Time-based (tag prevalence drifts over years) |
| Rare tags | Same model | Acknowledge limit; active learning loop for curator |
| Explanation | "model says these tags" | Per-tag top words; per-paper "we suggest this because..." |
| Cleaning | Lowercase only | Replace numbers/equations/URLs with tokens; weight title 3x |
