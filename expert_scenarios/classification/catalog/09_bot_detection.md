# Expert Scenario 9: Bot / Fake-Account Detection (Adversarial Classification)

> **Complexity:** Adversarial environment where the targets actively evolve to evade the model, weekly retraining required, severe imbalance with cost asymmetry, the labeling itself is uncertain (analyst-confirmed bots vs. suspected), feature drift accelerates over time, no ground truth for "future bot tactics."

---

## The Brief

A social platform with 60M monthly active users gives you historical labeled data on bots / fake accounts:

- 4.2M accounts labeled as "bot" (confirmed by analyst review).
- 55.8M accounts labeled "human" (passed authentication + behavior gates).
- ~7% of all accounts are bot-like (varies week-to-week).

The brief: build a model that classifies accounts as bot/human at signup time. Constraints:

- Recall on bots ≥ 80% (catch most of them).
- False-positive rate ≤ 0.5% (don't ban real users — political nightmare).
- Inference < 100ms at signup.
- Weekly retraining mandatory (adversaries adapt within days).
- Account for label noise: ~5% of "human" labels are actually bots that got past previous models.
- Survive feature drift: a feature that worked last week may rot this week.

This is harder than typical classification because: the adversary actively evolves to evade your features; the cost asymmetry is enormous (banning a real user is far worse than missing a bot); training labels are noisy; and the feedback loop (your bans → adversary adapts → next training data shifts) creates instability.

---

## Step 1: Define the Problem Type

```
Type:           Binary classification, adversarial
Primary Metric: Recall on confirmed bots at fixed FPR ≤ 0.5%
Secondary:      Precision on flagged accounts; per-segment recall (different bot types)
Business Goal:  Recall ≥ 80% at FPR ≤ 0.5%; weekly recall maintained at 80%+ over time
Constraint:     < 100ms; weekly retrain; adversarial robustness
Imbalance:      ~93/7 -- moderate but cost ratio is 100:1 (ban-real-user is catastrophic)
```

**Expert thinking:** the false-positive constraint is extremely tight (0.5%) because banning real users is far worse than missing bots. A model with 99% recall but 5% FPR would ban 50 real users for every 100 bots caught — politically and operationally untenable.

---

## Step 2: Understand the Data

```
Shape: 60M accounts × ~80 features at signup time

PROFILE (signup-time):
- account_id (unique)
- created_at (signup timestamp)
- email_provider (Gmail / Yahoo / Outlook / 200+ rare)
- email_age_at_signup_days (estimated; can sometimes detect)
- has_avatar (binary)
- avatar_appears_default (binary)
- profile_filled_out_pct
- bio_length
- bio_contains_url (binary)
- claimed_age (NaN if not provided)
- claimed_location (string, 60% missing)

DEVICE / NETWORK:
- ip_address_hash
- ip_age_days (how long has this IP been seen)
- ip_country
- ip_geolocation_consistency (with claimed location)
- vpn_or_proxy_score (third-party, 0-1)
- ua_signature_hash (browser/device fingerprint)
- ua_seen_count_in_recent_30d (how many other accounts used this signature)
- mobile_or_desktop
- os_family
- timezone_offset_match_with_ip (binary)

BEHAVIORAL (first 24 hours):
- num_actions_24h
- avg_time_between_actions_seconds
- action_diversity_score (entropy of action types)
- num_followers_24h
- num_following_24h
- following_to_follower_ratio
- num_posts_24h
- num_likes_given_24h
- engagement_with_bot_signals_24h (interactions with known-bot graph)

VELOCITY / BURST:
- accounts_from_same_ip_24h
- accounts_with_same_ua_signature_24h
- accounts_with_similar_email_pattern_24h

LABEL:
- account_status (Active / Banned / Pending review)
- ban_reason (Bot / Spam / Other)
- ban_confidence (Confirmed / Suspected / Auto)
```

**Expert thinking:** the velocity features (multiple accounts from same IP, similar email patterns) catch coordinated bot attacks. They're high-signal but adversaries respond by spreading attacks across IPs. Always-evolving cat and mouse.

---

## Step 3: EDA

```python
df['is_bot'].mean()  # 0.07 (7% bots)

# Feature stability over time (CRITICAL for adversarial setting)
# Plot bot-detection performance per week using a fixed model
# Performance decays without retraining

# Per-bot-type recall (different bot strategies):
#   Crude spam bots:     85% recall (easy to catch)
#   Engagement farms:    70% recall (mimic humans more)
#   Scraping bots:       62% recall (very subtle)
#   Coordinated attacks: 78% recall (caught via velocity features)

# Velocity features are very predictive:
df.groupby(pd.cut(df['accounts_from_same_ip_24h'], 5))['is_bot'].mean()
# 1 account: 6%
# 2-3 accounts: 18%
# 4-9: 47%
# 10+: 82% bot rate
```

---

## Step 4: Data Cleaning

```python
# === LABEL NOISE HANDLING ===
# 5% of "human" labels are actually bots that escaped previous models
# Two options:
#   1. Trust the labels (some noise is acceptable)
#   2. Bootstrap: train a preliminary model, identify high-confidence "humans" that look bot-like

# We'll use option 2 conservatively
preliminary_model = LightGBMClassifier(...).fit(X_train, y_train)
proba = preliminary_model.predict_proba(X_train)[:, 1]
# Humans with proba > 0.95 are likely mislabeled
# Flag for re-review; don't auto-relabel
suspect_mislabels = X_train[(y_train == 0) & (proba > 0.95)]
# Send to analyst review queue (not used for training)

# === HANDLE NEW IPs / UAs ===
# New IPs / UAs are common; don't drop, but mark as new
df['is_new_ip'] = (df['ip_age_days'] < 7).astype(int)
df['is_new_ua'] = (df['ua_seen_count_in_recent_30d'] < 5).astype(int)
```

---

## Step 5: Feature Engineering — Adversarially-Aware

```python
# === COMPOSITE BOT SIGNATURES ===
df['default_avatar_score'] = (
    df['avatar_appears_default'].astype(int) +
    df['has_avatar'].astype(int) * (-0.5) +
    (df['profile_filled_out_pct'] < 0.3).astype(int)
)

# === BEHAVIORAL TIMING FEATURES ===
df['inhuman_speed'] = (df['num_actions_24h'] > 1000).astype(int)
df['robotic_consistency'] = (df['avg_time_between_actions_seconds'].std() < 5).astype(int)
df['bursty_actions'] = (df['num_actions_first_hour'] > 100).astype(int)

# === FOLLOWER NETWORK FEATURES ===
df['unusual_follow_ratio'] = (df['following_to_follower_ratio'] > 50).astype(int)
df['mass_follow_pattern'] = (df['num_following_24h'] > 200).astype(int)
df['followed_by_known_bots_pct'] = compute_followers_in_known_bot_graph(df)

# === COORDINATION FEATURES ===
df['ip_burst_score'] = np.log1p(df['accounts_from_same_ip_24h'])
df['email_pattern_burst_score'] = compute_email_pattern_similarity(df)
df['device_overlap_score'] = np.log1p(df['accounts_with_same_ua_signature_24h'])

# === PROFILE ANOMALY ===
df['claimed_age_under_18'] = (df['claimed_age'] < 18).astype(int)
df['email_provider_rare'] = (df['email_provider_seen_count'] < 100).astype(int)
df['bio_url_density'] = (df['bio_contains_url'] & (df['bio_length'] < 50)).astype(int)
```

---

## Step 6: Train/Test Split — Time-Based

```python
# CRITICAL for adversarial: train on past, test on future
# Random split lets the model see future bot tactics in training
# Adversaries adapt; what worked yesterday may not work tomorrow

df = df.sort_values('created_at')
train = df[df['created_at'] < cutoff_train]
val = df[(df['created_at'] >= cutoff_train) & (df['created_at'] < cutoff_test)]
test = df[df['created_at'] >= cutoff_test]
```

---

## Step 7: Try Multiple Models

```python
# === Model 1: Logistic Regression ===
# Recall@FPR=0.5%: 62%

# === Model 2: Random Forest ===
# Recall@FPR=0.5%: 71%

# === Model 3: LightGBM ===
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=1000, max_depth=6, num_leaves=63,
    learning_rate=0.05, reg_alpha=2.0, reg_lambda=2.0,  # robust to feature drift
    random_state=42
)
# Recall@FPR=0.5%: 81%  ✓ (target 80%)
# Inference: ~6ms

# === Model 4: XGBoost (with monotonic constraints) ===
# Recall@FPR=0.5%: 80%
```

**Top performer:** LightGBM with regularization (recall 81% at 0.5% FPR).

---

## Step 8: Threshold Tuning

```python
# At 0.5% FPR target, find the threshold that maximizes recall
y_proba = lgb_model.predict_proba(X_val)[:, 1]
fpr_target = 0.005
threshold = np.quantile(y_proba[y_val == 0], 1 - fpr_target)
# threshold = 0.94 (very high; only bans accounts model is very confident about)

predictions = (y_proba >= threshold).astype(int)
# Recall: 81%
# FPR: 0.49%
```

---

## Step 9: Adversarial Retraining Cadence

```python
# CRITICAL: weekly retraining
# Without it, recall decays by ~5-10pp per month as adversaries evolve

# Weekly process:
# 1. Append last week's confirmed bot labels (analyst-reviewed)
# 2. Retrain on rolling 12-week window
# 3. Validate on past week (not part of training)
# 4. Compare to previous week's recall — alert on drop > 3%
# 5. Champion-challenger: deploy if new model recall ≥ incumbent

# Monitoring per-feature stability:
# Compute per-feature mutual information with target weekly
# Features with sudden MI drop = adversaries learned to evade
# Examples in real data:
#   "Default avatar" worked great in 2018 (90% bots), down to 60% by 2022
#   "Email provider rare" became less effective as bots used Gmail
```

---

## Step 10: Per-Bot-Type Performance

```python
# Audit recall by bot type
bot_types = df_test[df_test['is_bot']]['ban_reason']
for bt in bot_types.unique():
    mask = (df_test['ban_reason'] == bt)
    recall_bt = predictions[mask].mean()
    print(f"  {bt}: recall {recall_bt:.2f}")

# Crude spam: 0.92
# Engagement farms: 0.78
# Scraping bots: 0.65
# Coordinated attacks: 0.85

# The "scraping bots" subtype is hardest because they mimic legitimate users very closely
# Often best caught by complementary models (graph analysis on what they read/scrape)
```

---

## Step 11: Final Evaluation

```python
# Final model: LightGBM, regularized, with weekly retraining
# Test set: 1.2M accounts from past 7 days (post-train cutoff)
#
# Performance:
#   Overall recall:                81%   ✓
#   FPR:                          0.49%  ✓
#   Per-bot-type:
#     Crude spam:                 92%
#     Engagement farms:           78%
#     Scraping bots:              65%   <- weakest
#     Coordinated attacks:        85%
#
# Per-week stability:
#   Week 1 recall: 81%
#   Week 2 recall: 80% (without retraining: would be 78%)
#   Week 4 recall: 79% (with retraining)
#
# Inference: ~6ms per account
```

---

## Step 12: Deployment

```python
# Production:
#   1. Real-time classification at signup (within 100ms)
#   2. If predicted bot with high confidence: auto-ban + send for review
#   3. If predicted bot with medium confidence: friction (CAPTCHA, phone verification)
#   4. If predicted human: standard signup flow
#
# Weekly retraining:
#   - Pull last week's analyst-confirmed bot/human labels
#   - Append to training window (drop oldest week)
#   - Retrain LightGBM
#   - A/B test new model on 10% of signups for 1 day before full deploy
#
# Monitoring (real-time):
#   - Recall against analyst-confirmed labels (lagged 24h)
#   - FPR against rapidly-flagged-as-mistake accounts
#   - Per-feature mutual information drift
#   - Distribution shift in input features (KL divergence weekly)
#
# Adversarial response triggers:
#   - Recall drop > 3%: investigate; emergency retrain
#   - New bot pattern surge: rapid retraining with new examples
#   - Feature-drift signal: deprioritize compromised features
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Adversarial awareness | Train one model, ship | Weekly retraining; expect adversary adaptation |
| Feature drift | Ignore | Track per-feature MI weekly; decay-aware features |
| Label noise | Trust all labels | 5% noise handling; flag suspect human labels for re-review |
| FPR constraint | F1 optimum | Hard FPR cap at 0.5%; threshold pinned to that |
| Per-bot-type | Aggregate metric | Audit per-subtype; flag scraping bots as hardest |
| Train/test split | Random | Strictly temporal (adversary evolution) |
| Velocity features | Ignore | Critical: same-IP burst, email pattern burst, device overlap |
| Multi-tier response | Auto-ban or allow | Tier: high confidence → auto-ban, medium → friction, low → allow |
| Retraining | Quarterly | Weekly mandatory; A/B test each new model |
| Champion-challenger | Skip | Required: don't deploy unless beats incumbent on past week |
| Drift response | Manual investigation | Real-time MI tracking + automatic retraining triggers |
