# Expert Scenario 24: Toxicity / Content Moderation Classification

> **Complexity:** Binary text classification with high precision requirement (false flags = censorship), dialect bias is the canonical fairness landmine, adversarial circumvention (users misspell to evade), context-dependent toxicity (sarcasm vs hate), human-disagreement on ground truth, multi-tier moderation (auto-remove vs review-queue vs allow).

---

## The Brief

A social platform with 2B daily comments wants a model to flag toxic content (hate, harassment, threats). The product team's research:

- 1.4% of comments are clearly toxic (confirmed by 3+ moderators).
- 3% are borderline (moderators disagree).
- Cost of false positive (over-removal): user complaints, perception of censorship, distrust.
- Cost of false negative: targeted users harmed; advertisers may pull spend.

Constraints:

- Recall on confirmed-toxic ≥ 88% at FPR ≤ 0.5% on non-toxic.
- Per-protected-group (race, gender, sexuality) FPR gap ≤ 1pp (canonical fairness target).
- Inference < 200ms per comment.
- Use only classical ML (deep transformers handled by upstream tier).
- Survive adversarial spelling perturbations (l33t-speak, character substitution).

This is harder than typical text classification because: training labels reflect annotator culture (may flag AAVE language as "toxic" when it's not); adversaries actively evolve to evade; precision requirement is brutal (anything > 0.5% FPR generates massive UX damage); and context is critical (a quote of toxic content is not toxic itself).

---

## Step 1: Define the Problem Type

```
Type:           Binary text classification (toxic = 1, not toxic = 0)
Primary Metric: Recall at fixed FPR (≤ 0.5% on non-toxic)
Secondary:      Per-protected-group FPR gap (must be < 1pp)
                Per-toxicity-subtype recall (hate / harassment / threats)
Business Goal:  Recall ≥ 88%; FPR ≤ 0.5%; FPR-by-group gap ≤ 1pp
Constraint:     < 200ms; adversarially robust; classical ML only
Imbalance:      98.6/1.4 -- moderate
```

**Expert thinking:** the fairness constraint is the real crux. A model that works great overall but flags Black users' content at 3x the rate of white users' content (the canonical AAVE bias) is a deployment blocker. The audit by protected-group FPR is non-negotiable.

---

## Step 2: Understand the Data

```
Shape: 8M comments × text + metadata + labels
Target: is_toxic (1.4%)

Per comment:
- comment_id (unique)
- comment_text (string, 5-1000 chars)
- author_id_hash
- timestamp
- in_reply_to_comment_id (None or hashed parent)
- thread_topic_category (Politics / Sports / Tech / Local / etc.)
- author_account_age_days
- author_recent_toxicity_rate (rolling)

LABELS (from 3-5 moderators):
- is_toxic_majority (binary; 3+ of 5 moderators agree)
- is_toxic_unanimous (binary; all moderators agree)
- toxicity_subtype (Hate / Harassment / Threat / Spam / NotToxic)
- annotator_disagreement_score (entropy across moderator labels)

INFERRED USER CONTEXT (from author_id_hash; via separate database, not in features but in audit):
- author_perceived_race (perceived from handle/username/profile, NOT used as feature)
- author_perceived_gender
- author_dialect_indicators (AAVE, formal, internet slang)
```

**Expert thinking:** the perceived demographic features are NEVER inputs to the model. They're ONLY used in the audit phase to detect bias. Even informally including them creates a bias-laundering pathway.

---

## Step 3: EDA

```python
df['is_toxic_majority'].mean()  # 0.014

# Per-thread-topic toxicity rate
df.groupby('thread_topic_category')['is_toxic_majority'].mean()
# Politics:    3.2% (highest; political topics correlate with hostile speech)
# Local news:  1.6%
# Sports:      1.2%
# Tech:        0.8%
# Entertainment: 1.0%

# Annotator disagreement
df['annotator_disagreement_score'].describe()
# Moderators ALL agree (entropy=0): 92% of comments
# Borderline (entropy 0.1-0.5): 5%
# Moderators split (entropy > 0.5): 3%

# CRITICAL: per-dialect FPR baseline
# Use perceived-AAVE-dialect indicator from author profile
print("Default model FPR by perceived dialect:")
# Standard English: 0.5% FPR
# AAVE indicators:  1.6% FPR  (3.2x higher!)
# This is the canonical fairness problem
```

**Findings:**

| Finding | Implication |
|---------|------------|
| 5% of comments are borderline | Annotator disagreement is itself a feature for the model |
| AAVE-pattern comments flagged 3x higher | Dialect bias must be addressed before deploy |
| Politics = 3x toxicity rate of other topics | Topic-aware modeling improves accuracy |
| Adversarial misspellings ("k!ll", "k1ll") evade naive token features | Need char-n-gram features in addition to word tokens |

---

## Step 4: Data Cleaning — Adversarial-Aware

```python
# === STANDARD TEXT NORMALIZATION (carefully) ===
# Don't normalize too aggressively — adversaries use l33t-speak, but legitimate users use slang too
# Keep content as is; let char-n-grams capture variations

import re
def normalize_text(text):
    # Replace URLs / mentions / hashtags as tokens
    text = re.sub(r'http[s]?://\S+', 'URL', text)
    text = re.sub(r'@\w+', 'MENTION', text)
    text = re.sub(r'#\w+', 'HASHTAG', text)
    # Collapse repeated characters: "yessss" -> "yes" (light)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text.strip()

df['comment_clean'] = df['comment_text'].apply(normalize_text)
```

---

## Step 5: Feature Engineering — Multi-Vocabulary

```python
# === WORD-LEVEL TF-IDF (catches normal toxic terms) ===
from sklearn.feature_extraction.text import TfidfVectorizer

word_tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=80000,
    min_df=10,
    max_df=0.85,
    lowercase=True,
    stop_words=None,  # don't remove stop words; "you" + "are" + slur is the pattern
    sublinear_tf=True
)
X_word = word_tfidf.fit_transform(df['comment_clean'])

# === CHARACTER-LEVEL TF-IDF (catches misspellings, l33t-speak) ===
char_tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    max_features=40000,
    min_df=20,
    max_df=0.85,
    sublinear_tf=True
)
X_char = char_tfidf.fit_transform(df['comment_clean'])

# === DENSE STRUCTURAL FEATURES ===
df['comment_length'] = df['comment_clean'].str.len()
df['caps_ratio'] = df['comment_text'].apply(
    lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
)
df['punct_density'] = df['comment_text'].apply(
    lambda t: (t.count('!') + t.count('?')) / max(len(t.split()), 1)
)
df['has_url'] = df['comment_clean'].str.contains('URL').astype(int)
df['has_mention'] = df['comment_clean'].str.contains('MENTION').astype(int)
df['author_recent_toxicity_rate'] = df['author_recent_toxicity_rate'].fillna(0)
df['author_account_age_log'] = np.log1p(df['author_account_age_days'])

# === STACK ===
from scipy.sparse import hstack, csr_matrix
X_dense_sparse = csr_matrix(StandardScaler().fit_transform(df[dense_features]))
X_full = hstack([X_word, X_char, X_dense_sparse])  # ~120K features
```

**Expert insight:** char n-grams catch adversarial misspellings that word n-grams miss. "k!ll" doesn't match "kill" in word vocab but its char trigrams (`k!l`, `!ll`, `kil`) match. Combining word + char TF-IDF is the standard pattern for adversarial text classification.

---

## Step 6: Train Multiple Models

```python
# === Model 1: Logistic Regression with L2 (interpretable baseline) ===
lr = LogisticRegression(
    C=1.0, penalty='l2', max_iter=500,
    class_weight='balanced'  # we'll undo via threshold
)
# Recall@FPR=0.5%: 0.71

# === Model 2: Linear SVM ===
# Recall@FPR=0.5%: 0.74

# === Model 3: Logistic Regression with L1 ===
# Recall@FPR=0.5%: 0.69

# === Model 4: Multinomial Naive Bayes ===
# Fast but less accurate
# Recall@FPR=0.5%: 0.61

# === Model 5: LightGBM on TF-IDF (slow on sparse 120K features) ===
# Trains in 2 hours; recall 0.72; not worth the latency
```

**Top performer:** Linear SVM (recall 0.74 at FPR 0.5%).

The 0.74 vs target 0.88 means we need significantly more lift — either ensemble with neural model upstream, or better feature engineering. We'll proceed with the architecture decisions and accept this as the classical-ML floor.

---

## Step 7: Per-Group FPR Audit

```python
# This is THE critical step

# For the SVM model at FPR target 0.5% overall:
threshold = compute_threshold_at_fpr(svm.decision_function(X_val), y_val, target_fpr=0.005)
predictions = (svm.decision_function(X_test) > threshold).astype(int)

# Per-perceived-dialect FPR
for dialect in ['Standard', 'AAVE-indicators', 'Internet-slang']:
    mask = (df_test['author_dialect_indicators'] == dialect)
    fpr = ((predictions[mask] == 1) & (y_test[mask] == 0)).sum() / max((y_test[mask] == 0).sum(), 1)
    print(f"  {dialect}: FPR {fpr:.3f}, n={mask.sum()}")

# Standard:           FPR 0.42% ✓
# AAVE-indicators:    FPR 1.34% (3.2x higher, BAD)
# Internet-slang:     FPR 0.61% (close)

# Per-perceived-race FPR
for race in ['White', 'Black', 'Hispanic', 'Asian']:
    ...
# White:    FPR 0.41%
# Black:    FPR 1.18%   <-- 2.9x gap, FAILS audit
# Hispanic: FPR 0.58%
# Asian:    FPR 0.39%
```

**Expert insight:** the AAVE / Black-dialect bias is a well-documented phenomenon in toxicity classifiers. The training data: annotators (often not from those communities) flag AAVE patterns as more toxic. Solutions:
1. Recruit more diverse annotators (long-term).
2. Audit features: drop char n-grams strongly correlated with dialect.
3. Per-group threshold adjustment (legally risky in some contexts).
4. Ensemble with a dialect-aware classifier.

We'll apply option 2: feature dropping, then re-train.

---

## Step 8: Bias-Mitigation — Drop Harmful Features

```python
# Identify char n-grams that disproportionately fire on AAVE comments without being toxic
# These are likely "borrow" features that catch dialect, not toxicity

dialect_correlation = compute_feature_dialect_correlation(X_full, df['author_dialect_indicators'])
toxicity_correlation = compute_feature_toxicity_correlation(X_full, y_train)

# Features with high dialect correlation but low toxicity correlation are bias-only signals
problematic_features = (dialect_correlation > 0.20) & (toxicity_correlation < 0.05)
# About 1,200 of 120K features qualify
# These are the dialect-leakage features

# Drop them and retrain
X_train_filtered = X_train[:, ~problematic_features.values]
svm_filtered = LinearSVC(C=1.0).fit(X_train_filtered, y_train)

# Re-audit:
# Standard:        FPR 0.43%
# AAVE-indicators: FPR 0.74% (down from 1.34%)
# Internet-slang:  FPR 0.62%
# Race gap reduced from 2.9x to 1.7x

# Overall recall: 0.71 (from 0.74)
# We sacrificed 3pp recall for substantial fairness improvement
```

---

## Step 9: Two-Tier Moderation Architecture

```python
# Single threshold can't satisfy: Recall 88% + FPR 0.5% + per-group FPR < 1pp
# Use multi-tier

# Tier 1 (very high confidence, P >= 0.92): auto-remove
# Tier 2 (medium 0.65 <= P < 0.92): send to human review queue
# Tier 3 (low P < 0.65): allow

# At Tier 1 threshold:
# Recall: 0.55, FPR: 0.18%
# Per-group FPR: all under 0.30%

# Tier 2 captures additional 28% of toxic content for human review
# Total addressed: 83% (still below 88% target without deeper model)
```

**Expert insight:** the multi-tier approach dramatically reduces fairness risk. Auto-removal only happens at very high confidence, where dialect bias is minimal. Borderline cases get human review (where context can be considered).

---

## Step 10: Final Evaluation

```python
# Final classical model: Linear SVM with bias-mitigation feature filtering + 2-tier
# Test set: 800K labeled comments from 30-day held-out period
#
# Tier 1 (auto-remove):
#   Recall:  55%  (caught toxicity removed automatically)
#   FPR:     0.18%  ✓
#   Per-group FPR: all < 0.30%
#
# Tier 2 (review queue):
#   Captures additional 28% of toxicity for human review
#
# Total addressed: 83% (vs target 88% — gap; would need transformer-based upstream tier for that)
#
# Inference: 90ms p99 ✓
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Vocab | Word TF-IDF only | Word + character n-grams (catches misspellings) |
| Bias check | Skip | Per-protected-group FPR audit; flag if > 1pp gap |
| Bias mitigation | Adjust thresholds per group | Drop bias-leakage features; retrain |
| Adversarial robustness | None | Char-n-grams + light text normalization |
| Architecture | Single threshold | Multi-tier: auto-remove + review queue + allow |
| Annotator disagreement | Treat majority as truth | Use disagreement score as feature; borderline cases go to review |
| Per-topic | One global model | Topic-aware features (politics vs sports) |
| Class weight | class_weight='balanced' | Train without; use threshold for FPR target |
| Annotator diversity | Single team | Recruit diverse annotators; track inter-annotator agreement by group |
| Deployment | Single classifier | Tier 1 auto + Tier 2 human review preserves fairness while catching toxicity |
