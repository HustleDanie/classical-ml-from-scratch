# Expert Scenario 1: Credit Card Fraud Detection

> **Complexity:** Extreme class imbalance (99.83% vs 0.17%), asymmetric costs, real-time scoring requirement, explainability needed.

---

## The Brief

A bank hands you 1.2 million transactions from the past 6 months. 0.17% are fraudulent (2,040 out of 1,200,000). They want a model that catches at least 90% of fraud while keeping false alarms under 5%. A missed fraud costs $5,000 on average. A false alarm costs $15 (customer gets a verification SMS). They also want to understand *why* the model flags a transaction so agents can explain it to customers.

This is brutal because: extreme class imbalance (99.83% vs 0.17%), the cost is asymmetric, you need high recall AND reasonable precision, and you need explainability.

---

## Step 1: Define the Problem Type

```
Type:           Binary Classification (fraud = 1, legitimate = 0)
Primary Metric: Recall (must catch 90%+ of fraud)
Secondary:      Precision (keep false positives under 5%)
Business Metric: Cost = (missed_fraud * $5000) + (false_alarms * $15)
Constraint:     Model must be explainable (agents need to tell customers WHY)
```

**Expert thinking:** Accuracy is USELESS here. A model that predicts "not fraud" for everything gets 99.83% accuracy but catches zero fraud. We need Recall >= 0.90 and Precision >= 0.50 (ideally higher). The real metric is the **cost function**: minimize total cost = FN * $5000 + FP * $15.

---

## Step 2: Understand the Data

```
Shape: 1,200,000 rows x 28 features
Target: 'is_fraud' -- 0 (99.83%), 1 (0.17%)

Features received:
- transaction_id (string)
- timestamp (datetime)
- amount (float, $0.01 to $25,000)
- merchant_name (string, 14,000 unique values)
- merchant_category (string, 42 categories: grocery, gas, online, etc.)
- card_type (categorical: visa, mastercard, amex, discover)
- card_present (binary: 0 = online/phone, 1 = physical card used)
- country (string, 85 unique countries)
- city (string, 3,200 unique cities)
- cardholder_age (int, 18-92)
- account_age_days (int, how long they've had the card)
- credit_limit (float)
- current_balance (float)
- num_transactions_24h (int, how many transactions in last 24 hours)
- num_transactions_7d (int)
- num_transactions_30d (int)
- avg_amount_30d (float)
- max_amount_30d (float)
- num_declined_30d (int)
- num_international_30d (int)
- distance_from_home (float, miles from cardholder's home address)
- time_since_last_transaction (float, minutes)
- is_weekend (binary)
- hour_of_day (int, 0-23)
- ip_risk_score (float, 0-1, from third-party service, 12% missing)
- device_fingerprint_match (binary: does the device match known devices, 8% missing)
- billing_shipping_match (binary: do billing and shipping addresses match, 40% missing -- only exists for purchases with shipping)
- previous_fraud_flag (binary: has this card had fraud before)
```

**Expert thinking:** I immediately notice:
- `transaction_id` is useless (unique ID)
- `merchant_name` has 14,000 values -- high cardinality, can't one-hot encode
- `city` has 3,200 values -- same problem
- `billing_shipping_match` is 40% missing -- but this might be INFORMATIVE (missing = no shipping = certain transaction types)
- `ip_risk_score` 12% missing -- need to handle carefully, this is probably important
- Class imbalance is extreme (577:1 ratio)

---

## Step 3: Exploratory Data Analysis (EDA)

```python
# Target distribution
is_fraud
0    1197960  (99.83%)
1       2040  (0.17%)

# Missing values
ip_risk_score              144,000  (12.0%)
device_fingerprint_match    96,000  (8.0%)
billing_shipping_match     480,000  (40.0%)
```

**What EDA reveals:**

| Finding | Implication |
|---------|------------|
| Fraud transactions have avg amount $847 vs $67 for legitimate | Amount is a strong signal, but many legitimate high-value transactions exist too |
| 78% of fraud happens with `card_present=0` (online/phone) | Online transactions are riskier, but most online transactions are still legitimate |
| Fraud rate jumps to 2.1% when `distance_from_home > 500 miles` | Distance is very predictive, but only 3% of transactions are far from home |
| `billing_shipping_match=0` has 8.3% fraud rate vs 0.1% overall | Mismatch is a huge red flag |
| MISSING `billing_shipping_match` has 0.05% fraud rate | Missing means no shipping -- these are in-store/digital purchases, actually SAFER |
| Fraud clusters at 2am-5am (fraud rate 0.8% vs 0.1% overall) | Time of day matters, but non-linearly |
| `num_transactions_24h > 10` has 5.2% fraud rate | Velocity spike is a strong fraud signal |
| `ip_risk_score` missing values have 0.3% fraud rate (higher than average) | Missingness itself is a feature! Missing may mean new/unknown device |

**Expert insight:** The fact that `billing_shipping_match` being MISSING is informative (not random) means I should NOT impute it with median. I should create a 3-way feature: `match=1`, `mismatch=0`, `not_applicable=-1`. Same for `ip_risk_score` -- create a binary `ip_score_missing` flag before imputing.

---

## Step 4: Data Cleaning

```python
# Drop useless columns
drop_cols = ['transaction_id']  # unique ID, no predictive value
# Keep merchant_name for now -- will engineer features from it

# Handle missing values -- CAREFULLY
# ip_risk_score (12% missing): Create flag, then impute with median
df['ip_score_missing'] = df['ip_risk_score'].isnull().astype(int)
df['ip_risk_score'] = df['ip_risk_score'].fillna(df['ip_risk_score'].median())

# device_fingerprint_match (8% missing): Missing = unknown device = suspicious
df['device_unknown'] = df['device_fingerprint_match'].isnull().astype(int)
df['device_fingerprint_match'] = df['device_fingerprint_match'].fillna(0)  # treat unknown as no match

# billing_shipping_match (40% missing): Missing = no shipping address needed
df['billing_shipping_match'] = df['billing_shipping_match'].fillna(-1)  # 3-way: 1, 0, -1
# This is NOT imputation -- it's encoding the missingness as a meaningful category

# Outliers in amount
# Don't cap! A $15,000 transaction IS suspicious -- outliers are the signal here
# For fraud detection, NEVER remove outliers. They are often the fraud.
```

**Expert insight:** In most ML tasks, you clean outliers. In fraud detection, **outliers are the target**. Capping at the 99th percentile would literally remove the fraud signal. Instead, we'll use tree-based models that handle outliers naturally.

---

## Step 5: Feature Engineering

This is where experts spend most of their time. Raw features aren't enough -- we need to capture **behavioral patterns**.

```python
# === Velocity Features (how fast is the card being used?) ===
df['txn_velocity_24h_vs_30d'] = df['num_transactions_24h'] / (df['num_transactions_30d'] / 30 + 0.01)
# Ratio > 3 means today is 3x busier than average -- suspicious

df['amount_vs_avg_30d'] = df['amount'] / (df['avg_amount_30d'] + 0.01)
# Ratio > 5 means this transaction is 5x larger than their average -- suspicious

df['amount_vs_max_30d'] = df['amount'] / (df['max_amount_30d'] + 0.01)
# Ratio > 1 means this is the largest transaction in 30 days

df['amount_vs_credit_limit'] = df['amount'] / (df['credit_limit'] + 0.01)
# Using 80%+ of credit limit in one transaction is suspicious

# === Time Features ===
df['is_night'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] <= 5)).astype(int)
# Night transactions (12am-5am) have 8x higher fraud rate

df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
# Cyclical encoding: hour 23 and hour 0 are close (sine/cosine captures this)

df['minutes_since_last_txn_bucket'] = pd.cut(
    df['time_since_last_transaction'],
    bins=[0, 1, 5, 30, 120, 1440, float('inf')],
    labels=[0, 1, 2, 3, 4, 5]  # 0 = within 1 min (very suspicious)
).astype(int)

# === Behavioral Deviation Features ===
df['balance_utilization'] = df['current_balance'] / (df['credit_limit'] + 0.01)
# High utilization + large transaction = risky

df['txn_per_day_lifetime'] = df['num_transactions_30d'] / 30
df['is_new_account'] = (df['account_age_days'] < 90).astype(int)
# New accounts have 4x higher fraud rate

# === Geographic Risk Features ===
df['is_international'] = (df['country'] != 'US').astype(int)
df['distance_bucket'] = pd.cut(
    df['distance_from_home'], bins=[0, 10, 50, 200, 1000, float('inf')],
    labels=[0, 1, 2, 3, 4]
).astype(int)

# === Merchant Risk (from the 14,000 merchant names) ===
# Can't one-hot encode 14,000 merchants. Instead: compute historical fraud rate per merchant
merchant_fraud_rate = df.groupby('merchant_name')['is_fraud'].mean()
df['merchant_fraud_rate'] = df['merchant_name'].map(merchant_fraud_rate)
# Merchants with >1% historical fraud rate are risky

# Same for city
city_fraud_rate = df.groupby('city')['is_fraud'].mean()
df['city_fraud_rate'] = df['city'].map(city_fraud_rate)

# === Interaction Features (combinations that are suspicious together) ===
df['night_and_online'] = df['is_night'] * (1 - df['card_present'])
# Online transaction at 3am = very suspicious
df['high_amount_and_new_account'] = (df['amount'] > df['avg_amount_30d'] * 3).astype(int) * df['is_new_account']
# New account + unusually large transaction
df['far_and_online'] = (df['distance_from_home'] > 200).astype(int) * (1 - df['card_present'])
# Far from home + no physical card

# Drop raw high-cardinality columns (replaced by engineered features)
df = df.drop(columns=['merchant_name', 'city', 'timestamp', 'transaction_id'])
```

**Expert insight:** We went from 28 raw features to 42 engineered features. The key features aren't the raw values -- they're the **ratios and deviations from normal behavior**. `amount=$500` means nothing alone. `amount/avg_amount_30d = 8.3` means "this is 8x their normal spending" -- that's the signal.

**CRITICAL WARNING about target encoding:** `merchant_fraud_rate` and `city_fraud_rate` use the target variable to create features. This MUST be computed on the training set only during cross-validation, or you get data leakage. In production, use historical rates from before the prediction window.

---

## Step 6: Feature Selection

```python
# Stage 1 -- Already dropped: transaction_id, merchant_name (replaced), city (replaced), timestamp (replaced)

# Stage 2 -- Mutual Information (not Pearson -- relationships are non-linear)
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
# Results:
# Top features by MI:
#   amount_vs_avg_30d       0.042
#   merchant_fraud_rate     0.038
#   distance_from_home      0.035
#   txn_velocity_24h_vs_30d 0.031
#   ip_risk_score           0.029
#   billing_shipping_match  0.028
#   night_and_online        0.025
#   ...
# Bottom features (MI ~ 0):
#   cardholder_age          0.001   -> weak but keep (might help in interactions)
#   card_type               0.000   -> Visa vs Mastercard doesn't predict fraud -> REMOVE

# Stage 3 -- Use RF importance to confirm
# Random Forest agrees: amount_vs_avg_30d, merchant_fraud_rate, distance are top 3
# cardholder_age gets 0.1% importance -> border case, keep it (cheap to include)
# card_type gets 0.0% importance -> confirm removal

# Final: 40 features kept (removed card_type and one redundant distance feature)
```

---

## Step 7: Preprocessing

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = ['amount', 'amount_vs_avg_30d', 'distance_from_home', ...]  # 35 numeric
categorical_features = ['merchant_category', 'country']  # 2 categorical (42 + 85 categories)

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])
# OneHotEncoder handle_unknown='ignore' because test set might have new countries
```

**Expert insight:** We only scale for models that need it (Logistic Reg, SVM, KNN). Tree-based models (RF, XGBoost) don't need scaling. But wrapping it in a Pipeline means we can test both.

---

## Step 8: Train/Test Split

```python
# Time-based split -- NOT random!
# Fraud patterns change over time. Random split would leak future patterns into training.
# Use first 5 months for training, last 1 month for testing.

df = df.sort_values('transaction_date')
train = df[df['transaction_date'] < '2025-11-01']  # months 1-5
test = df[df['transaction_date'] >= '2025-11-01']   # month 6

X_train, y_train = train.drop('is_fraud', axis=1), train['is_fraud']
X_test, y_test = test.drop('is_fraud', axis=1), test['is_fraud']

# Train: ~1,000,000 rows (1,700 fraud)
# Test: ~200,000 rows (340 fraud)
```

**Expert insight:** Random `train_test_split` would be WRONG here. In production, you always predict future transactions using past data. A time-based split simulates this reality. If you randomly split, the model might learn from December fraud patterns to predict October fraud -- that's cheating.

---

## Step 9: Baseline

```python
# Baseline 1: Predict all as "not fraud"
# Accuracy: 99.83% (useless)
# Recall: 0.00% (catches zero fraud)
# Cost: 340 missed frauds * $5000 = $1,700,000

# Baseline 2: Simple rule -- flag if amount > 3x avg_amount_30d AND card_present=0
# Recall: ~45% (catches 153/340 fraud)
# Precision: ~8% (flags 1,912 transactions, only 153 are actually fraud)
# Cost: 187 missed * $5000 + 1759 false alarms * $15 = $935,000 + $26,385 = $961,385
```

The simple rule cuts cost by 43%. Our ML model needs to beat this significantly.

---

## Step 10: Try Multiple Models (with Stratified 5-Fold CV)

```python
from sklearn.model_selection import StratifiedKFold
# MUST use StratifiedKFold -- regular KFold might put 0 fraud cases in a fold

results = {}
# Model 1: Logistic Regression
# CV Recall: 0.72, Precision: 0.12, F1: 0.21, AUC: 0.94

# Model 2: Random Forest (n_estimators=500, class_weight='balanced_subsample')
# CV Recall: 0.81, Precision: 0.31, F1: 0.45, AUC: 0.97

# Model 3: XGBoost (scale_pos_weight=577)  # ratio of negatives to positives
# CV Recall: 0.85, Precision: 0.38, F1: 0.53, AUC: 0.98

# Model 4: LightGBM (is_unbalance=True)
# CV Recall: 0.86, Precision: 0.41, F1: 0.55, AUC: 0.98

# Model 5: SVM (class_weight='balanced')
# CV Recall: 0.76, Precision: 0.15, F1: 0.25, AUC: 0.95
# Too slow on 1M rows -- took 45 minutes. Not practical.

# Model 6: KNN (k=5)
# CV Recall: 0.52, Precision: 0.08, F1: 0.14, AUC: 0.82
# Terrible. KNN struggles with extreme imbalance and high dimensions.
```

**Top performers:** LightGBM and XGBoost. Random Forest is a solid third.

---

## Step 11: Handle Class Imbalance

```python
# Strategy 1: class_weight / scale_pos_weight (already used above -- built into the model)

# Strategy 2: SMOTE on training set only
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.1, random_state=42)
# Don't make it 50/50 -- oversample fraud to 10% (not 50%). Going to 50/50 creates too many
# synthetic samples that don't resemble real fraud.
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
# Before: 1,000,000 legit + 1,700 fraud
# After:  1,000,000 legit + 100,000 synthetic fraud

# Retrain LightGBM on SMOTE data:
# CV Recall: 0.89, Precision: 0.35, F1: 0.50, AUC: 0.98
# Recall improved from 0.86 -> 0.89, but Precision dropped from 0.41 -> 0.35

# Strategy 3: Combine SMOTE with Tomek Links (clean boundary)
from imblearn.combine import SMOTETomek
# CV Recall: 0.88, Precision: 0.39, F1: 0.54, AUC: 0.98
# Better precision than SMOTE alone

# DECISION: Use LightGBM with is_unbalance=True (no SMOTE)
# Reason: SMOTE improved recall by 3% but introduces synthetic data artifacts.
# We'll use threshold tuning instead to push recall higher.
```

**Expert insight:** SMOTE isn't always the answer. For very extreme imbalances (500:1), the synthetic samples can be unrealistic. Built-in class weighting + threshold tuning often works better and is simpler.

---

## Step 12: Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

# Tuning LightGBM
param_dist = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [4, 6, 8, 12],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0],
    'is_unbalance': [True]
}

search = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1),
    param_dist,
    n_iter=100,
    cv=StratifiedKFold(5),
    scoring='recall',  # Optimize for recall (must catch fraud)
    random_state=42,
    n_jobs=-1
)
search.fit(X_train_processed, y_train)

# Best params:
# n_estimators=1000, max_depth=8, learning_rate=0.05, num_leaves=63,
# min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
# reg_alpha=0.1, reg_lambda=1.0

# Tuned CV Results: Recall: 0.88, Precision: 0.44, F1: 0.59, AUC: 0.985
```

---

## Step 13: Threshold Tuning (THE Secret Weapon)

```python
# Default threshold is 0.5: if model says P(fraud) >= 0.5, flag it.
# But we want to catch MORE fraud. Let's lower the threshold.

y_proba = model.predict_proba(X_test_processed)[:, 1]

# Test different thresholds:
# Threshold 0.50: Recall=0.88, Precision=0.44, FP=378,   Cost=$338,670
# Threshold 0.30: Recall=0.92, Precision=0.32, FP=670,   Cost=$145,050
# Threshold 0.20: Recall=0.94, Precision=0.24, FP=1,040, Cost=$115,600
# Threshold 0.15: Recall=0.95, Precision=0.18, FP=1,690, Cost=$105,350  <-- SWEET SPOT
# Threshold 0.10: Recall=0.97, Precision=0.11, FP=2,960, Cost=$89,400
# Threshold 0.05: Recall=0.99, Precision=0.05, FP=6,540, Cost=$103,110

# Cost = missed_fraud * $5000 + false_alarms * $15
# At threshold 0.15:
#   - Catches 323/340 frauds (95% recall)  -> exceeds 90% target
#   - 1,690 false alarms (0.85% FPR)       -> well under 5% target
#   - Total cost: 17 * $5000 + 1690 * $15 = $85,000 + $25,350 = $110,350
#   - vs baseline cost of $1,700,000 -- that's a 93.5% cost reduction!

# CHOOSE threshold = 0.15
```

**Expert insight:** Threshold tuning is the single most impactful trick for imbalanced classification. Most beginners never touch the 0.5 default. By lowering to 0.15, we get +7% recall with acceptable false alarms. The optimal threshold depends on the cost ratio: $5000/$15 = 333:1, meaning missing one fraud costs as much as 333 false alarms. The math heavily favors lower thresholds.

---

## Step 14: Ensemble / Stacking

```python
from sklearn.ensemble import StackingClassifier

# Stack the top 3 models -- they make different types of errors
stack = StackingClassifier(
    estimators=[
        ('lgbm', best_lgbm),
        ('xgb', best_xgb),
        ('rf', best_rf)
    ],
    final_estimator=LogisticRegression(class_weight='balanced'),
    cv=StratifiedKFold(5),
    passthrough=False
)
stack.fit(X_train_processed, y_train)

# Stacked model + threshold 0.15:
# Recall: 0.96, Precision: 0.21, AUC: 0.989
# Catches 326/340 frauds (1 more than LightGBM alone)
# But complexity increased significantly for marginal gain

# DECISION: Stick with LightGBM alone. Simpler, faster, nearly as good.
# In fraud detection, speed matters -- transactions must be scored in <100ms.
```

**Expert insight:** Stacking gave +1% recall but added complexity and latency. In fraud detection, latency matters (real-time scoring), so we go with the simpler single model. Stacking is more valuable when you have time (batch predictions, offline scoring).

---

## Step 15: Final Evaluation on Test Set

```python
# Final Model: LightGBM with threshold=0.15

# Results on month 6 test set (200,000 transactions, 340 fraud):
#
# Recall:    0.953 (324/340 fraud caught)          -> exceeds 90% target
# Precision: 0.189 (324 true fraud out of 1,714 flagged)
# F1 Score:  0.315
# AUC-ROC:   0.986
# AUC-PR:    0.62 (this is the RIGHT metric for imbalanced data)
#
# False Positive Rate: 0.70% (1,390 / 199,660)    -> well under 5% target
#
# Confusion Matrix:
#                  Predicted 0    Predicted 1
# Actual 0         198,270         1,390       (false alarms)
# Actual 1             16           324        (caught!)
#
# COST ANALYSIS:
# Missed fraud:   16 * $5,000  = $80,000
# False alarms:   1,390 * $15  = $20,850
# Total cost:     $100,850
#
# vs Baseline (no model): $1,700,000
# vs Simple rule:         $961,385
#
# ML MODEL SAVES: $1,599,150 per month ($19.2M per year)
```

---

## Step 16: Explainability

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_processed[:1000])

# Global top features (what drives fraud predictions overall):
# 1. amount_vs_avg_30d (8.3x normal spending? Fraud signal)
# 2. merchant_fraud_rate (buying from sketchy merchants)
# 3. distance_from_home (far from home address)
# 4. txn_velocity_24h_vs_30d (sudden spike in activity)
# 5. night_and_online (online transaction at 3am)
# 6. ip_risk_score (known risky IP)
# 7. billing_shipping_match (mismatch = red flag)
# 8. time_since_last_transaction (rapid-fire transactions)

# Per-transaction explanation (what the agent tells the customer):
# "Transaction #XYZ was flagged because:
#  - The amount ($2,340) is 12x your typical spending (+0.35 risk)
#  - The transaction was online at 3:17 AM (+0.18 risk)
#  - The merchant has a high historical fraud rate (+0.15 risk)
#  - The IP address has a risk score of 0.89 (+0.12 risk)"
```

---

## Step 17: Deployment Considerations

```python
# Save model
joblib.dump(model, 'fraud_model.joblib')
joblib.dump(preprocessor, 'fraud_preprocessor.joblib')

# Production requirements:
# - Latency: <100ms per transaction (LightGBM does ~0.5ms -- fine)
# - Threshold: 0.15 (stored as config, adjustable without retraining)
# - Retraining: Weekly (fraud patterns evolve quickly)
# - Monitoring: Track daily recall on confirmed frauds
#               If recall drops below 85%, alert + retrain
# - A/B test: Route 10% of traffic to new model for 1 week before full rollout
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Missing values | `fillna(0)` everywhere | Created `ip_score_missing` flag -- missingness IS a feature |
| High-cardinality merchants | One-hot encode (14,000 columns!) or drop | Target encoding: `merchant_fraud_rate` |
| Train/test split | `train_test_split(random)` | Time-based split (prevents temporal leakage) |
| Feature engineering | Use raw features only | Created 14 behavioral features (velocity, ratios, interactions) |
| Imbalance handling | SMOTE to 50/50 | `is_unbalance=True` + threshold tuning |
| Threshold | Default 0.5 | Optimized to 0.15 based on cost function |
| Metric | Accuracy (99.83%!) | Recall + cost function |
| Outlier handling | Clip to 99th percentile | Keep outliers (THEY ARE THE FRAUD) |
| Evaluation | "accuracy is 99.8%" | Cost analysis: model saves $19.2M/year |
| Explainability | Black box | SHAP per-transaction explanations for agents |
