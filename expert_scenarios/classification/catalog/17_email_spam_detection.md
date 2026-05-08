# Expert Scenario 17: Email Spam Detection

> **Complexity:** High-dimensional sparse text features (TF-IDF), real-time inference at inbox scale, mild-to-moderate imbalance, adversarial drift, asymmetric cost (a misrouted legitimate email is worse than a missed spam).

---

## The Brief

A B2B email provider hands you 320,000 historical messages from the past 90 days. 28% are labeled spam. They want a model that runs at inbox-receipt time (must score in under 50ms per email) and:

- Flags spam with at least 95% recall.
- Keeps the false-positive rate (legitimate mail flagged as spam) under 0.5% — a misrouted legitimate email is the #1 customer complaint.
- Provides a "why" (top 3 words / signals) so users see the explanation in their spam folder.
- Tolerates concept drift — spammers change tactics weekly.

This is harder than it sounds because: TF-IDF features on 50,000+ vocabulary terms make most models impractical, the false-positive constraint is brutal (0.5% FPR while still recalling 95%), and adversaries reroute and mutate features within days.

---

## Step 1: Define the Problem Type

```
Type:           Binary Classification (spam = 1, ham = 0)
Primary Metric: Recall at fixed precision (>= 95% recall, FPR <= 0.5%)
Secondary:      F1; per-class precision
Business Metric: Customer complaints per million inboxes per day
Constraint:     < 50ms inference; explainable to end users
Imbalance:      28/72 (mild — not extreme, but FPR budget is the real challenge)
```

**Expert thinking:** Don't be fooled by 28% positive rate — the *constraint* is asymmetric. Recall is "how much spam do we catch", precision is "of what we flag, how much is actually spam." A 0.5% FPR ceiling means precision must be very high *because most of the inbox is legitimate*. This pushes us toward threshold-tuned linear models with calibrated probabilities.

---

## Step 2: Understand the Data

```
Shape: 320,000 rows x raw text + metadata
Target: 'is_spam' -- 0 (72%), 1 (28%)

Raw fields received:
- message_id (string, unique)
- timestamp (datetime)
- sender (string, hashed)
- sender_domain (string, ~12,000 unique)
- recipient (string, hashed)
- subject (string, free text)
- body (string, free text -- HTML stripped to plain)
- has_attachment (binary)
- num_recipients (int, 1-500+)
- num_links (int, 0-200+)
- num_images (int)
- spf_pass (binary, sender domain SPF check)
- dkim_pass (binary)
- dmarc_pass (binary)
- ip_country (string, 195 countries)
- ip_reputation_score (float, 0-1, third party, 4% missing)
- recipient_domain (string, ~50,000 unique)
- subject_length (int, derived)
- body_length (int, derived)
- is_reply (binary)
- thread_id (string, conversation grouping)
```

**Expert thinking:** I see two feature spaces:
1. **Sparse high-dim text** — subject + body need TF-IDF. Vocabulary will land at 30K-80K terms.
2. **Dense structured signals** — auth checks (SPF/DKIM/DMARC), counts, IP reputation. These are 15-25 features but extremely informative.

Many of the strongest spam signals are NOT in the text — they're in the auth checks and structural counts. A naive bag-of-words classifier will miss this.

---

## Step 3: Exploratory Data Analysis (EDA)

```python
df['is_spam'].value_counts(normalize=True)
# 0 (ham)   0.72
# 1 (spam)  0.28

df.groupby('is_spam')[['num_links', 'num_recipients', 'subject_length', 'body_length']].describe()
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Spam: avg 14 links per email; ham: avg 1.2 | Number of links is a strong signal alone |
| Spam: avg 87 recipients; ham: avg 1.4 | Mass-send is obvious |
| 78% of spam fails SPF check; 4% of ham fails | SPF pass is near-deterministic for ham |
| Spam: 31% from ip_country in {RU, CN, NG}; ham: 4% | Geo signal is useful but blunt — don't hard-code |
| Subject length: spam median 64 chars vs ham median 22 | Long, hyperbolic subjects are a tell |
| Body length: spam wider distribution (some very short "click here", some very long marketing) | Bimodal — capture with binning |
| `ip_reputation_score`: missing 4% — strongly correlated with new sender domains | Missingness is informative |

**Expert insight:** auth checks (SPF/DKIM/DMARC) are the cheapest, strongest signal. If you can only have 5 features, those would be 3 of them. But authoritative spammers (e.g., breached accounts) pass SPF — so we can't lean on it alone.

---

## Step 4: Data Cleaning

```python
# Drop useless columns
drop_cols = ['message_id', 'sender', 'recipient', 'thread_id']  # IDs, no predictive value
# sender_domain and recipient_domain we'll engineer features from

# IP reputation missingness
df['ip_rep_missing'] = df['ip_reputation_score'].isnull().astype(int)
# IMPORTANT: missing -> probably new/unknown sender -> impute with conservative LOW reputation
df['ip_reputation_score'] = df['ip_reputation_score'].fillna(0.3)

# Body / subject sanitization
# HTML already stripped, but normalize: lowercase, collapse whitespace, replace digits with NUM token
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', ' NUM ', text)  # numbers cluster (price, dosage, $$)
    text = re.sub(r'http[s]?://\S+', ' URL ', text)  # actual URLs become a token
    return text.strip()

df['body_clean'] = df['body'].apply(clean_text)
df['subject_clean'] = df['subject'].apply(clean_text)

# Outliers
# Don't cap. Long-body marketing spam IS the signal.
# But ensure body_length doesn't blow up training memory:
df['body_clean'] = df['body_clean'].apply(lambda t: t[:5000])  # truncate at 5K chars
```

**Expert insight:** Replacing all digits with `NUM` and all URLs with `URL` is a classic NLP trick. "$199.99" and "$49.99" are the same signal — the literal price is noise. Using the `NUM` token, the model learns "subject contains many numbers" which IS the spam signal.

---

## Step 5: Feature Engineering

```python
# === TEXT FEATURES (TF-IDF on cleaned subject + body) ===
from sklearn.feature_extraction.text import TfidfVectorizer

# Combine subject + body with a separator so the model can learn position effects
df['text_combined'] = 'SUBJ ' + df['subject_clean'] + ' BODY ' + df['body_clean']

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),          # unigrams + bigrams
    max_features=50000,          # top 50K terms
    min_df=5,                     # ignore words appearing in <5 emails
    max_df=0.95,                  # ignore words appearing in >95% (too common)
    sublinear_tf=True,            # log-scale TF (1 + log(tf))
    stop_words='english'
)
# Fit on TRAIN ONLY in production -- here showing the call
# X_text = tfidf.fit_transform(df['text_combined'])

# === STRUCTURAL FEATURES (15 dense numerics) ===
df['log_num_links']      = np.log1p(df['num_links'])      # heavy-skew -> log
df['log_num_recipients'] = np.log1p(df['num_recipients'])
df['log_body_length']    = np.log1p(df['body_length'])
df['subject_link_ratio'] = df['num_links'] / (df['subject_length'] + 1)
df['has_many_links']     = (df['num_links'] > 5).astype(int)
df['has_many_recipients'] = (df['num_recipients'] > 10).astype(int)

# === AUTHENTICATION SCORE ===
df['auth_score'] = df['spf_pass'] * 1 + df['dkim_pass'] * 1 + df['dmarc_pass'] * 1
# Range 0-3. 0 = nothing passes (very suspicious), 3 = all pass (very legitimate)

# === DOMAIN REPUTATION (target encoding -- DANGER ZONE) ===
# Compute historical spam rate per sender_domain on TRAIN ONLY:
# (compute inside CV; here showing the principle)
sender_spam_rate = df_train.groupby('sender_domain')['is_spam'].mean()
df['sender_spam_rate'] = df['sender_domain'].map(sender_spam_rate).fillna(0.28)  # default = base rate

# === GEO RISK ===
high_risk_countries = {'RU', 'CN', 'NG', 'IN', 'PK'}  # data-driven, not stereotype
df['is_high_risk_geo'] = df['ip_country'].isin(high_risk_countries).astype(int)

# === TEMPORAL ===
df['hour_of_day'] = df['timestamp'].dt.hour
df['is_business_hours'] = df['hour_of_day'].between(8, 18).astype(int)
# Spam volume spikes at 2am-5am sender-time

# Drop raw text after vectorization
df = df.drop(columns=['body', 'subject', 'sender_domain', 'recipient_domain', 'ip_country'])
```

**Expert insight:** the dense structural features (auth_score, sender_spam_rate, log_num_links, is_high_risk_geo) are typically 8-12 features that carry as much signal as the 50,000 TF-IDF terms combined. They feed into the same linear model and dramatically lift recall at low FPR.

**CRITICAL on target encoding:** `sender_spam_rate` MUST be computed on the training fold only inside CV. Otherwise you leak test labels into training and inflate validation scores by 2-3 percentage points.

---

## Step 6: Feature Selection

```python
# Stage 1 -- Already dropped: message_id, sender, recipient, thread_id, raw text columns

# Stage 2 -- Don't run mutual_info on 50K TF-IDF features (slow, noisy at low df)
# For TF-IDF, rely on min_df / max_df cutoffs in the vectorizer itself

# Stage 3 -- L1 regularization in the model itself (Lasso Logistic) acts as feature selection
# Expect ~5,000-8,000 of the 50K TF-IDF features to get non-zero coefficients

# For the dense structural features, all 12 are kept (each adds value, all cheap)

# Total: 50,000 TF-IDF + 12 structural + 1 sender_spam_rate = 50,013 features
# After L1 regularization, effective: ~5,000-8,000
```

**Expert insight:** at 50K features and 320K rows, mutual_info_classif takes 30+ minutes. Use the regularization in your final model (L1 on linear, built-in for trees) instead of explicit feature selection.

---

## Step 7: Preprocessing

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack

# TF-IDF features are already L2-normalized within rows (sublinear_tf + IDF)
# Dense structural features need StandardScaler

dense_features = ['auth_score', 'log_num_links', 'log_num_recipients',
                  'log_body_length', 'subject_link_ratio', 'has_many_links',
                  'has_many_recipients', 'sender_spam_rate', 'is_high_risk_geo',
                  'is_business_hours', 'ip_reputation_score', 'ip_rep_missing']

scaler = StandardScaler()
X_dense = scaler.fit_transform(df[dense_features])

# Stack TF-IDF (sparse) + dense structural (sparse-converted) horizontally
X = hstack([X_text, X_dense]).tocsr()

# X has 50,012 columns, sparse, ~98% sparsity
```

**Expert insight:** keep the matrix sparse — converting to dense would require 320,000 × 50,012 × 8 bytes = **128 GB RAM**. Logistic Regression and Linear SVM both accept sparse input natively.

---

## Step 8: Train/Test Split

```python
# TIME-BASED SPLIT (not random) -- spam patterns drift weekly
df = df.sort_values('timestamp')
cutoff = df['timestamp'].quantile(0.85)  # last 15% as test
train = df[df['timestamp'] < cutoff]  # ~272K rows
test  = df[df['timestamp'] >= cutoff] # ~48K rows

# Stratified within time? No -- preserve temporal order, accept slight imbalance differences
print(train['is_spam'].mean(), test['is_spam'].mean())
# 0.27, 0.30 -- close enough; spam rate trended up in last 15% of period
```

**Expert insight:** Random split is wrong here. Spammers actively adapt — a model trained on randomly-mixed past+future data sees future tactics in training, inflates validation, and underperforms in production. Time-based split is brutal but honest.

---

## Step 9: Baseline

```python
# Baseline 1: predict "ham" for everything
# Accuracy: 70%, Recall (spam): 0%, FPR: 0%
# Useless -- but reminds us accuracy is a poor metric here.

# Baseline 2: simple rule -- flag if num_links > 5 AND auth_score < 2
# Recall: 41%, Precision: 89%, FPR: 1.4%
# A simple rule already gets 41% recall at 1.4% FPR. The ML model needs to crush this.
```

---

## Step 10: Try Multiple Models with Stratified 5-Fold CV

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Model 1: Multinomial Naive Bayes (gold-standard text baseline)
# Note: TF-IDF + MultinomialNB requires non-negative input
# Actually, for TF-IDF use ComplementNB which handles imbalance better
from sklearn.naive_bayes import ComplementNB

mnb = ComplementNB(alpha=0.1)
# CV (recall, precision, FPR): 0.91, 0.78, 0.094
# Fast (15 sec to train), but FPR way too high

# Model 2: Logistic Regression with L1
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0,
                            class_weight='balanced', max_iter=500)
# CV: recall 0.96, precision 0.93, FPR 0.026
# Good! FPR still 5x our 0.5% target -- threshold tuning needed.

# Model 3: Logistic Regression with L2
lr_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0,
                            class_weight='balanced', max_iter=500)
# CV: recall 0.97, precision 0.92, FPR 0.030
# Marginally better recall, slightly worse FPR

# Model 4: Linear SVM
linsvm = LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, dual=False)
# CV: recall 0.95, precision 0.94, FPR 0.022
# Best precision, but no probability output (need calibration for threshold tuning)

# Model 5: SGDClassifier with log loss (for online learning later)
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(loss='log_loss', alpha=1e-5, class_weight='balanced',
                    max_iter=20, random_state=42)
# CV: recall 0.95, precision 0.91, FPR 0.034
# Same ballpark, easier to update incrementally
```

**Top performers:** Logistic Regression with L1, Linear SVM with calibration. ComplementNB ruled out (FPR 9.4%).

---

## Step 11: Handle Imbalance

```python
# Imbalance is mild (28/72) -- class_weight='balanced' is enough
# SMOTE on 50K-dim sparse text data is a trap:
#   - Synthetic samples in TF-IDF space are not real text
#   - Memory blows up
# Skip SMOTE. Stick with class_weight + threshold tuning.

# An alternative: undersample ham to 50/50
# CV result with undersampling: recall 0.97 (+0.01), precision 0.85 (-0.08), FPR 0.058 (much worse)
# Undersampling hurt precision -- too many ham examples discarded
# DECISION: keep class_weight='balanced', tune threshold instead
```

---

## Step 12: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Tuning Logistic Regression with L1
param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0],
}

grid = GridSearchCV(
    LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=500),
    param_grid,
    cv=StratifiedKFold(5),
    scoring='roc_auc',  # rank-based, doesn't depend on threshold
    n_jobs=-1
)
grid.fit(X_train, y_train)

# Best: C=1.0
# Tuned CV: recall 0.96, precision 0.93, FPR 0.024, AUC 0.992

# C below 1.0 over-regularized (recall dropped to 0.93)
# C above 1.0 overfit (test FPR rose from 0.024 to 0.033)
```

---

## Step 13: Threshold Tuning (THE Critical Step)

```python
# Default threshold 0.5 gives FPR ~2.4% -- 5x our 0.5% budget
# We need to RAISE the threshold (be more conservative about calling spam)
# This trades recall for FPR

y_proba = best_lr.predict_proba(X_val)[:, 1]

# Sweep thresholds:
# Threshold 0.50: Recall=0.96, Precision=0.93, FPR=0.024
# Threshold 0.65: Recall=0.95, Precision=0.96, FPR=0.012
# Threshold 0.75: Recall=0.93, Precision=0.97, FPR=0.007
# Threshold 0.82: Recall=0.91, Precision=0.98, FPR=0.0048  <-- meets FPR budget
# Threshold 0.85: Recall=0.89, Precision=0.985, FPR=0.0036
# Threshold 0.90: Recall=0.84, Precision=0.99, FPR=0.0021

# At threshold 0.82:
#   Recall: 0.91 (below 95% target)
#   FPR:    0.0048 (meets 0.5% budget)
#
# We CANNOT meet both targets simultaneously with this single model.
# The PR-curve simply doesn't allow recall=0.95 AND FPR<=0.005.
#
# Decision options:
#   1. Loosen one constraint (negotiate with stakeholder)
#   2. Add a "low confidence -> manual review" tier
#   3. Stack with a second-tier model
```

**Expert insight:** the brief asked for two things that together describe a Pareto frontier the model can't reach. Sometimes the right answer is to push back on requirements; other times you build a 2-tier system. We'll go with a 2-tier (Step 14).

---

## Step 14: Two-Tier System (Spam Folder + Manual Review Tier)

```python
# Tier 1 (high-confidence spam, P >= 0.92): auto-route to spam folder
# Tier 2 (uncertain, 0.55 <= P < 0.92): hold + show user "review prompt" in inbox
# Tier 3 (P < 0.55): deliver normally

# At thresholds 0.92 / 0.55:
# Tier 1: catches 81% of spam at FPR 0.0011 (well under budget)
# Tier 2: catches additional 13% of spam (held for user review) -- this counts toward recall
# Tier 3: 6% of spam slips through entirely

# Total "addressed" recall: 0.81 + 0.13 = 0.94 (still below 0.95)
# But TIER 1 FPR (the only auto-routing) is 0.0011 -- 5x BETTER than target

# Run again with Tier 1 threshold = 0.88 (slightly lower):
# Tier 1: catches 87% spam at FPR 0.0035 (within budget)
# Tier 2: catches additional 9% spam (P 0.55-0.88, held for review)
# Total addressed: 0.96 ✓
```

**Expert insight:** real production spam systems are almost always multi-tier. A single threshold on a single model can't beat the unavoidable Pareto frontier. The "manual review" tier acts as a soft action that's neither "spam folder" nor "inbox" — and that flexibility is what unlocks the dual constraint.

---

## Step 15: Final Evaluation on Held-Out Test Set (last 15% by time)

```python
# Final model: Logistic Regression with L1, C=1.0, class_weight='balanced'
# Tier 1 threshold: 0.88
# Tier 2 threshold: 0.55

# Test set: 48,000 emails, 30% spam (drift!)
#
# Tier 1 (auto-route to spam):
#   Routed: 13,400 emails
#   True spam in routed: 13,300
#   Tier 1 precision: 99.25%  (FPR 0.30%)
#
# Tier 2 (held for user review):
#   Held: 1,890 emails
#   True spam in held: 1,360
#   Tier 2 precision: 71.9% (acts as soft prompt only)
#
# Tier 3 (delivered):
#   Delivered: 32,710 emails
#   Spam slipping through: 540
#   Recall lost: 540 / 14,200 = 3.8%
#
# OVERALL:
#   Recall (Tier 1 + Tier 2 caught spam): 96.2%  ✓ (target 95%)
#   FPR (Tier 1 only): 0.30%               ✓ (target 0.5%)
#   p99 inference latency: 18ms             ✓ (target 50ms)
```

---

## Step 16: Explainability (User-Facing "Why")

```python
# Logistic Regression coefficients ARE the explanation
# For a flagged email, show the top 3 features driving the score

def explain_prediction(email_features, model, feature_names):
    contributions = email_features.toarray()[0] * model.coef_[0]
    top_indices = np.argsort(np.abs(contributions))[-3:][::-1]
    return [(feature_names[i], contributions[i]) for i in top_indices]

# Example:
# Email flagged with P=0.94. Top 3 reasons:
#   1. "free" appears 8 times (+0.42 spam contribution)
#   2. SPF check failed (+0.31)
#   3. 17 links in body (+0.28)
# UI surface: "We flagged this as spam because: contains 'free' frequently, sender failed authentication, has many links"
```

**Expert insight:** Logistic Regression coefficients are the killer feature for spam UIs. Tree-based models would need SHAP and the explanations would be vague ("decision path through 100 trees"). Linear models give literal word-level reasons that users can verify themselves.

---

## Step 17: Deployment Considerations

```python
# Model size: ~80MB on disk (sparse coefficients + TF-IDF vocabulary)
# Inference: 18ms p99 with 8 cores

# Production architecture:
#   1. TF-IDF vectorizer (frozen) -- 50MB
#   2. Logistic Regression model -- 30MB
#   3. Feature engineering (auth_score, log transforms) -- inline in scoring service
#   4. Tier router (if/elif on probability) -- 10 lines of code

# Retraining cadence: WEEKLY (mandatory)
#   - Spammers adapt within days
#   - Schedule: Monday 03:00 UTC, train on past 6 weeks, deploy by 09:00 UTC
#   - A/B test: 5% of inboxes get new model for 24h before full rollout

# Monitoring:
#   - Daily Tier 1 precision (must stay > 98%)
#   - Daily recall on next-day-confirmed spam (must stay > 92%)
#   - Vocabulary drift: % of words in incoming mail not in TF-IDF vocab (alert if > 8%)
#   - User-reported "this is not spam" rate (must stay < 0.1% of Tier 1)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Text preprocessing | Lowercase only | Replace digits/URLs with token, normalize whitespace, truncate |
| Vectorization | Basic TF-IDF default params | TF-IDF with min_df / max_df / sublinear_tf / 1-2-grams / 50K cap |
| Feature space | TF-IDF only | TF-IDF + 12 dense structural + target-encoded sender |
| Imbalance | SMOTE | class_weight + threshold tuning (SMOTE breaks sparse text) |
| Sparse handling | Convert to dense | Keep sparse end-to-end (saves 128GB) |
| Train/test split | random | TIME-based (spammers adapt) |
| Threshold | default 0.5 | Two-tier: 0.88 spam folder + 0.55 review prompt |
| Pareto-impossible constraints | Ship anyway, miss SLA | Negotiate via 2-tier system |
| Explainability | "model says spam" | Top-3 word/signal contributions surfaced in UI |
| Deployment | one-time train | Weekly retrain + A/B test + drift monitoring |
| Metric | accuracy | Recall at fixed FPR + per-tier precision |
