# Expert Scenario 48: Real-Time Bidding (RTB) Click Prediction

> **Complexity:** Sub-10ms inference budget, 1B+ daily events, severe imbalance (1-3% click rate), feature hashing for high-cardinality, FTRL online learning, calibrated probabilities feed bid math, training data is biased by past bidding policy.

---

## The Brief

A demand-side platform (DSP) bids on programmatic ad impressions. For each impression, you have ~10 milliseconds to decide a bid. The system flow:

```
Bid request arrives → predict P(click | impression)
                  → bid = pCTR × value_per_click × pacing_multiplier
                  → submit bid (or pass)
                  → if win, ad serves; if click, log
```

Volume: 800M bid requests/day, 150M wins, 2.4M clicks (1.6% on wins).

Constraints:
- Per-bid compute budget: < 8ms (we have 10ms but parsing + bid serialization eats 2ms).
- Calibrated probabilities (the bidding math multiplies by pCTR — calibration matters in dollars).
- Online learning (model must update from yesterday's clicks within hours).
- High-cardinality categoricals everywhere: user_id (250M+), site_id (180K), creative_id (50K).
- Memory budget for the model: < 2GB.
- Bid log is biased: we only see clicks on impressions we won, and we only won impressions we bid high on.

This is harder than a standard click prediction because: classical models can't fit 250M one-hot user IDs, sub-ms latency rules out tree ensembles for serving, the feedback loop creates bias (training data isn't IID with serving traffic), and the metric (log loss / Brier) requires calibrated output.

---

## Step 1: Define the Problem Type

```
Type:           Binary classification with calibrated probability output
Primary Metric: Log loss (calibration-sensitive); Brier score
Secondary:      AUC (rank quality); per-segment calibration
Business Metric: Net ad spend ROI; effective cost-per-click (eCPC)
Constraint:     < 8ms p99 inference; < 2GB model; online updates within hours
Imbalance:      ~98.4 / 1.6 -- severe; class_weight not used (would break calibration)
Bias:           Training data only contains wins (impressions we bid high enough on)
```

**Expert thinking:** the metric is log loss, NOT AUC. AUC ranks impressions by predicted CTR but doesn't tell us if 0.05 means 5% empirically. Bidding multiplies by the probability — if we predict 0.05 but truth is 0.02, we bid 2.5x too high. Calibration matters in literal dollars.

`class_weight='balanced'` is a trap here. It produces probabilities calibrated to a 50/50 world. We need probabilities calibrated to the real 1.6% world.

---

## Step 2: Understand the Data

```
Shape: ~150M won-impressions per day with click labels (target = 1.6%)
Stream-friendly format: each event is an independent row

Features (per bid request):

USER (15):
- user_id (hashed cookie / device_id, 250M+ unique)
- user_age_band (8 bins, demographic-third-party)
- user_gender (M/F/Unknown)
- user_income_band (5 bins, modeled, 60% missing)
- user_interest_segments (sparse boolean vector, ~500 segments per user)
- num_recent_clicks_7d (int, 0-50+)
- num_recent_impressions_7d (int)
- recent_ctr_30d (float, 0-0.10)
- bot_score (float, 0-1)
- mobile_or_desktop
- browser
- os
- screen_resolution_class
- timezone_offset
- country

CONTEXT (10):
- site_id (180K unique publishers)
- domain (top-level)
- page_category (50 IAB categories)
- page_url_path
- session_id
- session_depth (page index in user's current session)
- referrer_class
- time_of_day_hour
- day_of_week
- weekday_or_weekend

AD (12):
- creative_id (50K unique)
- advertiser_id (8K)
- campaign_id (40K)
- creative_format (banner_300x250 / banner_728x90 / video_15s / etc.)
- creative_dimension_pixels
- creative_size_kb
- creative_age_days
- product_category
- creative_text_features (TF-IDF on ad copy, ~10K terms)
- has_animation
- has_audio
- creative_color_dominant_hue

AUCTION (8):
- auction_type (first_price / second_price)
- floor_price
- exchange (Google / TheTradeDesk / etc.)
- ad_position (above_fold / below_fold)
- viewability_estimate (float, 0-1)
- bid_request_size_bytes
- competing_bidders_count_estimate
- recent_win_rate_in_segment

TARGET:
- clicked (binary)
```

**Expert thinking:** the cardinality wall is brutal. user_id has 250M values. site_id has 180K. creative_id has 50K. Standard approaches fail:
- One-hot: 250M-column matrix per row → impossible.
- Target encoding: better, but lookup table at inference is huge and doesn't help cold-start.
- Feature hashing: the standard solution for ad-tech.

---

## Step 3: EDA

```python
df['clicked'].mean()  # 0.016 (1.6%)

# Click rate by ad position
df.groupby('ad_position')['clicked'].mean()
# above_fold: 0.024
# below_fold: 0.011
# in_article: 0.018

# Click rate by creative format
df.groupby('creative_format')['clicked'].mean()
# video_15s:        0.038
# banner_300x250:   0.014
# banner_728x90:    0.011
# native:           0.022

# Click rate by user recency
df.groupby('num_recent_clicks_7d')['clicked'].mean()
# 0: 0.012
# 1: 0.024
# 5+: 0.061  -- recent clickers click again

# Hour-of-day pattern
df.groupby('time_of_day_hour')['clicked'].mean().plot()
# Peaks at 7-9am and 7-9pm; trough at 2-4am

# CRITICAL: bid log selection bias
# We only see won impressions. We won by bidding high. We bid high based on
# prior pCTR predictions. Past predictions influence training distribution.
# This is "selection bias" or "feedback loop bias" -- a major modeling concern.
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Video creatives 3x click rate of banner | Format is structural |
| User recent clicks (7d) strongly predicts | Use recent behavioral features |
| Above-fold ads 2x click rate | Position is a structural feature |
| Site_id, advertiser_id, creative_id all have cardinality > 10K | Need feature hashing or target encoding |
| Selection bias: won impressions are biased high-CTR | Need importance weighting or off-policy correction |
| Bot traffic ~ 4% of impressions; 0.0% click rate | bot_score threshold filter must be in pipeline |

---

## Step 4: Data Cleaning

```python
# Filter bot traffic out of training (they distort the click rate downward)
df = df[df['bot_score'] < 0.7]

# Remove duplicate impressions (rare bug at exchanges)
df = df.drop_duplicates(subset=['user_id', 'creative_id', 'timestamp_ms'])

# Missing handling
# user_income_band 60% missing -- model the missingness
df['user_income_missing'] = df['user_income_band'].isnull().astype(int)
df['user_income_band'] = df['user_income_band'].fillna('Unknown')

# user_interest_segments sparse vector -- use as is (already binary)
```

---

## Step 5: Feature Engineering — Hashing-First

```python
# === FEATURE HASHING (the key technique) ===
# Hash each high-cardinality categorical to a fixed N-dim space (typically 2^20 = 1M dims)
# Collisions are accepted as the price of constant-time / constant-memory inference

from sklearn.feature_extraction import FeatureHasher

# Hash user_id, site_id, creative_id together with cross-features:
hash_features = []
for row in stream:
    feature_strings = [
        f'user_id={row.user_id}',
        f'site_id={row.site_id}',
        f'creative_id={row.creative_id}',
        f'user_id={row.user_id} X creative_id={row.creative_id}',  # cross
        f'site_id={row.site_id} X creative_id={row.creative_id}',  # cross
        f'user_id={row.user_id} X site_id={row.site_id}',          # cross
    ]
    hash_features.append(feature_strings)

hasher = FeatureHasher(n_features=2**20, input_type='string')
X_hashed = hasher.transform(hash_features)
# Output: sparse matrix, 1M columns, ~10 non-zero per row

# === DENSE NUMERIC FEATURES ===
# These are NOT hashed; they're a small dense block alongside the hashed sparse
df['log_recent_impressions_7d'] = np.log1p(df['num_recent_impressions_7d'])
df['log_session_depth'] = np.log1p(df['session_depth'])
df['recent_ctr_30d'] = df['recent_ctr_30d'].clip(0, 0.20)  # tail noise
df['hour_sin'] = np.sin(2 * np.pi * df['time_of_day_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['time_of_day_hour'] / 24)

dense_features = ['log_recent_impressions_7d', 'log_session_depth', 'recent_ctr_30d',
                   'hour_sin', 'hour_cos', 'viewability_estimate', 'floor_price',
                   'creative_age_days', 'session_depth', 'mobile_or_desktop_int', ...]

# === STACK SPARSE HASHED + DENSE ===
from scipy.sparse import hstack
X_dense_sparse = csr_matrix(StandardScaler().fit_transform(df[dense_features]))
X_full = hstack([X_hashed, X_dense_sparse])
```

**Expert insight:** feature hashing is the workhorse of ad tech. Yes, two unrelated user IDs can hash to the same column (collision) — but with 1M columns and ~10 non-zeros per row, collision rate is low and a linear model degrades gracefully. The compute and memory savings are enormous: O(constant) per row regardless of vocabulary.

---

## Step 6: Cross-Features (Why FTRL Wins)

```python
# Interactions matter HUGELY in ad-tech. user × creative is the magic feature.
# A linear model on one-hot user_id and one-hot creative_id can't learn that
# user 4821 + creative "running_shoes" → high CTR.
# Adding user × creative cross-features creates that combined column.

# Above we added 3 cross-features. We could add more:
#   user × site, user × hour, user × creative_format, etc.
# Each cross adds one feature per row to the hashed input.

# This is why ad-tech historically used Logistic Regression with FTRL —
# linear model + heavy cross-feature engineering + online learning beats trees
# at this scale.
```

---

## Step 7: Model — FTRL Online Logistic Regression

```python
# FTRL (Follow-The-Regularized-Leader) is online optimization tuned for sparse logistic
# regression. It updates weights one example at a time with optimal sparsity properties.

from sklearn.linear_model import SGDClassifier

# sklearn's SGDClassifier supports log_loss + L1 + L2 regularization
# It's not exactly FTRL but close (FTRL is a specific variant of online SGD)

ftrl = SGDClassifier(
    loss='log_loss',
    learning_rate='constant',
    eta0=0.01,
    alpha=1e-7,        # very mild regularization
    l1_ratio=0.0001,   # L1 + L2 mix
    penalty='elasticnet',
    random_state=42,
    warm_start=True    # for incremental updates
)

# === ONLINE TRAINING (one mini-batch at a time) ===
batch_size = 50000
for batch_idx in range(num_batches):
    X_batch = X_full[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    y_batch = y_full[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    ftrl.partial_fit(X_batch, y_batch, classes=[0, 1])

# Inference time: ~3ms per impression (linear model, sparse dot product)
# Model size: ~16MB (1M weights × 16 bytes)
```

**Expert insight:** for true production FTRL, use Vowpal Wabbit (`vw`) or a specialized library. sklearn is good for prototyping; production at 1B-events/day uses `vw` or custom C++ pipelines.

---

## Step 8: Tree Models — Worth the Latency Hit?

```python
# Try LightGBM as a challenger (faster than XGBoost at predict time)
import lightgbm as lgb

# We can't use the full hashed feature matrix in trees -- too sparse, too high-dim.
# Instead: target-encode high-cardinality fields (computed daily, refreshed)

df['user_recent_ctr_30d'] = df['recent_ctr_30d']  # already user-level
df['site_ctr_28d'] = compute_site_ctr_target_encoding(df_train, smoothing=1000)
df['creative_ctr_28d'] = compute_creative_ctr_target_encoding(df_train)
df['advertiser_ctr_28d'] = compute_advertiser_ctr_target_encoding(df_train)

# Use these as numeric features alongside the dense block
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    num_leaves=127,
    learning_rate=0.05,
    objective='binary',
    random_state=42
)

# Inference: ~12ms per impression -- TOO SLOW for our 8ms budget
# (And we have no Tweedie / log_loss equivalent issue)

# Compromise: shallower trees
lgb_shallow = lgb.LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=31)
# Inference: ~3ms p99 -- within budget
# But test log loss: 0.0612 (vs FTRL 0.0594)
# Trees underperform here because the user × creative cross-features are
# critical and trees can't learn them as easily as cross-feature linear

# DECISION: stick with FTRL for serving; use LightGBM for offline analysis only
```

---

## Step 9: Calibration — Critical for Bidding

```python
# FTRL output is in logit space; pass through sigmoid to get probability
# For bidding correctness, calibration must be near-perfect

# Validation: bin predicted probabilities into deciles, check empirical CTR per decile
df_val['proba'] = ftrl.predict_proba(X_val)[:, 1]
df_val['decile'] = pd.qcut(df_val['proba'], 10, labels=range(1, 11))
calibration = df_val.groupby('decile').agg(
    pred_avg=('proba', 'mean'),
    actual_ctr=('clicked', 'mean'),
    impressions=('proba', 'count')
)

# Decile  Pred CTR   Actual CTR   Gap
# 1       0.0021     0.0019      +0.0002
# 2       0.0048     0.0041      +0.0007
# 3       0.0079     0.0070      +0.0009  ✓
# ...
# 8       0.0274     0.0241      +0.0033  <- slight over-prediction
# 9       0.0512     0.0476      +0.0036
# 10      0.0814     0.0712      +0.0102  <- 14% relative over-prediction at top

# Top decile over-predicts -- common after FTRL on sparse data
# Apply Platt scaling on a daily holdout

from sklearn.linear_model import LogisticRegression
calibrator = LogisticRegression()
calibrator.fit(df_holdout['proba'].values.reshape(-1, 1), df_holdout['clicked'])

# At inference:
calibrated_proba = calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
```

---

## Step 10: Selection Bias Correction

```python
# Training data is biased: only impressions we won are labeled
# Won impressions skew toward high-pCTR predictions (we bid more there)

# Mitigation: importance weighting using inverse propensity score (IPS)
# weight_i = 1 / win_probability_i

# Compute win probability from auction history:
#   For each (segment, hour), what fraction of bids did we win?
df_train['win_propensity'] = df_train.groupby(['site_id', 'hour'])['won'].transform('mean').clip(0.01, 1.0)
df_train['ips_weight'] = 1 / df_train['win_propensity']

# Pass weights to FTRL via sample_weight
ftrl.partial_fit(X_batch, y_batch, sample_weight=ips_weights, classes=[0, 1])

# Result: log loss on heldout test set drops 0.003 (from 0.0594 to 0.0564)
```

**Expert insight:** without IPS correction, the model overfits to "easy wins" and underestimates clicks in segments where we usually lose the auction. IPS correction is non-negotiable for a correctly-deployed bidder.

---

## Step 11: Pacing & Bidding Math

```python
# pCTR alone doesn't determine bid -- pacing does.
# Bidding strategy: bid = pCTR × value_per_click × pacing_multiplier

# pacing_multiplier shrinks bid when daily budget consumption is ahead of schedule
# (e.g., spent 60% of daily budget by noon → throttle bids)

# Real bid:
def compute_bid(pCTR, value_per_click, pacing, floor):
    raw_bid = pCTR * value_per_click * pacing
    return max(raw_bid, floor + 0.01)  # ensure clearing the floor

# Calibration error compounds: 10% over-pCTR → 10% over-bid → 10% wasted spend
# This is why log loss + Platt scaling matters more than AUC
```

---

## Step 12: Final Evaluation

```python
# Final model: FTRL Logistic Regression with feature hashing + IPS correction + Platt
#
# Daily eval on yesterday's bid data:
#   Log loss:                 0.0564     ✓ (below 0.06 target)
#   AUC:                      0.738
#   Calibration ECE:          0.0021
#   Brier score:              0.0152
#
# Per-segment performance:
#   Top 10% predicted: actual CTR 7.8% (predicted 8.2%) -- 5% rel error
#   Bottom 10%:        actual CTR 0.18% (predicted 0.22%) -- 22% rel error (small absolute)
#
# Inference (production):
#   p50 latency:  2.1ms
#   p99 latency:  6.4ms      ✓ (under 8ms budget)
#   Memory:       1.4GB      ✓ (under 2GB budget)
```

---

## Step 13: Deployment Considerations

```python
# Streaming pipeline:
#   1. Bid request arrives
#   2. Lookup user_recent_features from Redis (< 1ms)
#   3. Feature engineering inline (hash + dense)
#   4. FTRL predict_proba (sparse dot product)
#   5. Apply Platt calibration
#   6. Multiply by value_per_click and pacing
#   7. Submit bid

# Online learning:
#   - Hourly: append last hour's logged events to streaming queue
#   - FTRL.partial_fit on batch of 50K events
#   - Atomic swap: new model serves traffic
#   - Drift detection: if log loss on incoming events > 1.2x baseline, alert

# Cold-start handling:
#   - New user_id: hash to default cluster; use site/creative features only
#   - New creative_id: similarly handled
#   - New advertiser: 7-day "shadow" period with conservative bidding

# Monitoring (everything is calibration-related):
#   - Per-segment Brier score (alert if any segment drifts > 0.005)
#   - Per-decile predicted vs actual CTR gap
#   - Cumulative bid spend vs target (budget delivery)
#   - Win rate per segment (sanity check on bidding strategy)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| High-cardinality | One-hot or target encoding lookup | Feature hashing (constant memory, constant time) |
| Cross-features | Hope the model learns interactions | Explicitly cross user × creative × site |
| Model | XGBoost (too slow) or Logistic Regression (no online) | FTRL (online + sparse + L1 sparsity) |
| Imbalance | class_weight='balanced' | UNWEIGHTED training; calibrate post-hoc |
| Calibration | Use raw model output | Platt scaling on daily holdout |
| Bias correction | Train on all wins as IID | IPS importance weighting (inverse propensity) |
| Latency | Train, ship, hope for the best | Profile p99 inline; reject any path > 8ms |
| Cold-start | "no features for new user" | Hash to default cluster; use site/creative features |
| Online updates | Retrain nightly | partial_fit hourly; atomic model swap |
| Drift | Retrain monthly | Real-time log-loss monitoring; alert on drift |
| Metric | AUC | Log loss; Brier score; per-decile calibration |
| Bid math | Bid = pCTR × value | Bid = pCTR × value × pacing × floor-aware |
