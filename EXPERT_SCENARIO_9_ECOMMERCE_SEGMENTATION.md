# Expert Scenario 9: E-Commerce Customer Segmentation & Lifetime Value

> **Complexity:** Unsupervised clustering on behavioral data (no labels), high-cardinality categorical features (10,000+ products), temporal behavioral sequences, combining clustering with supervised CLV prediction, actionable marketing automation, cold-start problem for new customers, segment stability over time.

---

## The Brief

A mid-size e-commerce retailer (2.8M customers, $420M annual revenue) wants to: (1) segment customers into actionable groups for personalized marketing, (2) predict Customer Lifetime Value (CLV) for each segment, and (3) automate marketing campaigns per segment. Currently they use a basic "high/medium/low spenders" split and blast the same email to everyone. They lose $18M/year on ineffective promotions (discounting customers who would buy anyway) and churn customers who get irrelevant offers.

This is complex because: there are no labels (unsupervised), behavioral features span time (sequences not just snapshots), clustering must be ACTIONABLE (not just statistically interesting), new customers have no history, and combining unsupervised segmentation with supervised CLV prediction requires careful pipeline design.

---

## Step 1: Define the Problem Type

```
Type 1:         Unsupervised Clustering (discover customer segments)
Type 2:         Regression (predict CLV for each customer)
Type 3:         Classification (predict segment for new customers)

Pipeline: Cluster existing customers -> Profile segments -> Predict CLV
          -> Train classifier to assign new customers to segments

Primary Metric (Clustering): Silhouette Score + Business Interpretability
Primary Metric (CLV):        MAE in predicted 12-month spend
Business Metric:             Marketing ROI improvement, churn reduction
```

---

## Step 2: Understand the Data

```
2.8M customers, 18 months of transaction history

Transactional:
- order_id, customer_id, order_date, order_total
- items: product_id, category (200 categories), subcategory (1,200),
  quantity, unit_price, discount_applied
- 42M transaction rows over 18 months

Behavioral (web/app):
- page_views, session_duration, bounce_rate
- search_queries (text), product_views, add_to_cart_events
- wishlist_additions, review_submissions, review_ratings
- email_open_rate, email_click_rate, push_notification_response

Customer profile:
- registration_date, acquisition_channel (organic, paid, referral, social)
- device_type (mobile 62%, desktop 30%, tablet 8%)
- location (city, state -- 50 states)
- account_type (guest 35%, registered 65%)

Returns:
- return_rate, return_reasons (text), avg_days_to_return
- exchange_vs_refund ratio

Customer service:
- support_tickets_count, avg_resolution_time
- complaint_categories, satisfaction_score
```

---

## Step 3: RFM Analysis (Classic Foundation)

```python
# RFM = Recency, Frequency, Monetary -- the gold standard starting point

import pandas as pd
import numpy as np

# Calculate RFM metrics (using 18 months of data)
snapshot_date = pd.Timestamp('2024-07-01')

rfm = orders.groupby('customer_id').agg({
    'order_date': lambda x: (snapshot_date - x.max()).days,    # Recency
    'order_id': 'nunique',                                      # Frequency
    'order_total': 'sum'                                        # Monetary
}).rename(columns={
    'order_date': 'recency_days',
    'order_id': 'frequency',
    'order_total': 'monetary'
})

# RFM distributions are heavily skewed:
# Recency:  median=45 days, mean=89 days, max=540 days
# Frequency: median=3, mean=5.2, max=127
# Monetary: median=$187, mean=$340, max=$42,000

# Log-transform monetary and frequency (skew > 2)
rfm['log_monetary'] = np.log1p(rfm['monetary'])
rfm['log_frequency'] = np.log1p(rfm['frequency'])

# Expert insight: DON'T cluster on raw RFM alone
# It only captures volume, not behavior patterns
# Two customers spending $500 could be very different:
# - One buys 1 expensive item, the other buys 50 cheap items
# - One buys monthly, the other bought once 6 months ago
```

---

## Step 4-5: Advanced Feature Engineering

```python
# === Beyond RFM: Behavioral DNA ===
# Each customer gets a behavioral profile vector

# 1. Purchase Pattern Features
df['avg_order_value'] = df['monetary'] / df['frequency']
df['order_frequency_days'] = df['customer_tenure_days'] / df['frequency']
df['purchase_regularity'] = df.groupby('customer_id')['days_between_orders'].std()
# Low std = very regular buyer (subscription-like), high std = sporadic

# 2. Category Diversity (what do they buy?)
df['num_categories'] = orders.groupby('customer_id')['category'].nunique()
df['category_concentration'] = orders.groupby('customer_id').apply(
    lambda x: (x['category'].value_counts(normalize=True) ** 2).sum()
)
# Herfindahl index: 1.0 = single category, ~0.01 = even across 100 categories

# 3. Price Sensitivity
df['avg_discount_pct'] = orders.groupby('customer_id')['discount_applied'].mean()
df['pct_orders_with_discount'] = orders.groupby('customer_id').apply(
    lambda x: (x['discount_applied'] > 0).mean()
)
df['buys_only_on_sale'] = (df['pct_orders_with_discount'] > 0.8).astype(int)

# 4. Engagement Depth
df['review_rate'] = df['reviews_submitted'] / df['frequency']
df['return_rate'] = df['returns'] / df['items_purchased']
df['wishlist_to_purchase_ratio'] = df['wishlist_items'] / (df['items_purchased'] + 1)
df['email_engagement'] = df['email_open_rate'] * 0.5 + df['email_click_rate'] * 0.5
df['active_days_ratio'] = df['days_with_activity'] / df['customer_tenure_days']

# 5. Trend Features (behavioral trajectory)
df['spend_trend'] = df['spend_last_3mo'] / (df['spend_prev_3mo'] + 1)
# > 1.0 = increasing spend, < 1.0 = declining
df['frequency_trend'] = df['orders_last_3mo'] / (df['orders_prev_3mo'] + 0.1)
df['engagement_trend'] = df['sessions_last_3mo'] / (df['sessions_prev_3mo'] + 1)

# 6. Product Affinity (compressed from 200 categories to 8 dimensions)
# Create customer-category matrix, then apply NMF
from sklearn.decomposition import NMF
customer_category_matrix = orders.pivot_table(
    index='customer_id', columns='category',
    values='order_total', aggfunc='sum', fill_value=0
)
nmf = NMF(n_components=8, random_state=42)
category_embeddings = nmf.fit_transform(customer_category_matrix)
# 8 latent product affinity dimensions
# Component 0: Electronics + Tech accessories
# Component 1: Fashion + Beauty
# Component 2: Home + Kitchen
# ...etc (interpret from top-weighted categories per component)

# Total feature vector per customer: ~35 features
# [RFM(3) + patterns(5) + price_sensitivity(3) + engagement(5) +
#  trends(3) + product_affinity(8) + demographics(8)]
```

---

## Step 6: Clustering Algorithm Selection

```python
# With 2.8M customers and 35 features, we need scalable algorithms

# === Algorithm Comparison ===

# Attempt 1: K-Means (fast, scalable)
# Pros: Runs in minutes on 2.8M rows, produces clean spherical clusters
# Cons: Assumes spherical clusters, sensitive to outliers
# MUST normalize features first (StandardScaler)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Elbow method + Silhouette for choosing K
# K=4: Silhouette=0.38
# K=5: Silhouette=0.42 <-- best
# K=6: Silhouette=0.40
# K=7: Silhouette=0.36
# K=8: Silhouette=0.33

# Use MiniBatchKMeans for 2.8M rows (100x faster than standard KMeans)
kmeans = MiniBatchKMeans(n_clusters=5, batch_size=10000, random_state=42)
kmeans.fit(X_scaled)

# Attempt 2: Gaussian Mixture Model
# Can find elliptical clusters (more flexible than K-Means)
# BIC criterion says K=5 as well
# Silhouette=0.39 (slightly worse than K-Means)
# But gives soft assignments (probability per cluster)

# Attempt 3: HDBSCAN (density-based)
# Finds clusters of arbitrary shape, identifies outliers
# Problem: 2.8M rows is too large for standard HDBSCAN
# Solution: Sample 100K customers, cluster, then assign rest via nearest-centroid
# Found: 6 clusters + noise (12% unassigned)
# Silhouette on assigned: 0.45 (good, but 12% left out)

# Attempt 4: Two-stage clustering
# Stage 1: K-Means with K=20 (micro-segments)
# Stage 2: Hierarchical clustering to merge 20 micro-segments into 5 macro-segments
# This captures non-spherical shapes while remaining scalable
# Silhouette: 0.44

# DECISION: K-Means K=5 (simplest, most interpretable, best trade-off)
# Soft assignments from GMM kept for customers near cluster boundaries
```

---

## Step 7-8: Cluster Profiling

```python
# === The 5 Customer Segments ===

# SEGMENT 1: "Loyal Champions" (12% of customers, 38% of revenue)
# Recency: 14 days avg | Frequency: 18 orders/year | Monetary: $1,850/year
# category_concentration: LOW (buy across many categories)
# return_rate: 8% (low)
# discount_reliance: 25% orders on sale (buy regardless of promotions)
# engagement: HIGH (open 65% emails, leave reviews)
# trend: STABLE (consistent high spend)
# PERSONA: Long-tenure, multi-category shopper who loves the brand

# SEGMENT 2: "Rising Stars" (18% of customers, 22% of revenue)
# Recency: 21 days | Frequency: 8/year | Monetary: $620/year
# category_concentration: MEDIUM (exploring)
# discount_reliance: 40%
# engagement: MEDIUM-HIGH
# trend: INCREASING (spend_trend > 1.3)
# PERSONA: Newer customer with growing spend -- highest conversion potential

# SEGMENT 3: "Bargain Hunters" (25% of customers, 18% of revenue)
# Recency: 35 days | Frequency: 10/year | Monetary: $380/year
# discount_reliance: 78% orders on sale (almost never pays full price)
# return_rate: 22% (high -- buy multiple sizes, return the rest)
# category_concentration: HIGH (mostly fashion)
# engagement: MEDIUM (open emails with "SALE" in subject only)
# PERSONA: Discount-driven, high maintenance, but decent volume

# SEGMENT 4: "Drifting Away" (20% of customers, 14% of revenue)
# Recency: 95 days | Frequency: 4/year | Monetary: $290/year
# trend: DECLINING (spend_trend < 0.6, frequency_trend < 0.5)
# engagement: LOW and FALLING
# PERSONA: Former regular shopper who's disengaging -- at risk of churn

# SEGMENT 5: "One-and-Done" (25% of customers, 8% of revenue)
# Recency: 180+ days | Frequency: 1.2 orders total | Monetary: $85 total
# 60% made exactly 1 purchase
# engagement: VERY LOW
# PERSONA: Acquired but never converted to repeat buyer

# Revenue distribution tells the story:
# 12% of customers (Champions) = 38% of revenue
# 25% of customers (One-and-Done) = 8% of revenue
# Focusing on Champions + Rising Stars = 60% of revenue from 30% of customers
```

---

## Step 9-10: CLV Prediction Per Segment

```python
# Now add SUPERVISED prediction: what will each customer spend in next 12 months?

# Training data: use months 1-12 features to predict months 13-18 spend
# Then retrain on months 1-18 to predict months 19-30

# === Per-Segment CLV Models ===
# (Different models per segment because spending distributions differ)

# Champions: Log-normal distribution, use Ridge Regression on log(CLV)
# Rising Stars: Gamma distribution (right-skewed), use Gamma GLM
# Bargain Hunters: Mixed -- many small orders, use Random Forest
# Drifting Away: Zero-inflated (many may spend $0), use hurdle model
# One-and-Done: Almost all $0, use logistic regression P(returns) * E[spend|returns]

# Champions CLV model:
from sklearn.linear_model import Ridge
X_champ = features[segment == 1]
y_champ = np.log1p(future_spend[segment == 1])
ridge = Ridge(alpha=1.0)
# CV MAE: $142 (on $1,850 avg = 7.7% error)

# Rising Stars CLV model:
from sklearn.ensemble import GradientBoostingRegressor
X_rising = features[segment == 2]
y_rising = future_spend[segment == 2]
gbm = GradientBoostingRegressor(max_depth=4, n_estimators=300)
# CV MAE: $98 (on $620 avg = 15.8% error -- more variance)

# Drifting Away CLV model (two-stage hurdle):
# Stage 1: Will they purchase at all? (Logistic Regression)
# P(any_purchase) = 0.45 for this segment on average
# Stage 2: If yes, how much? (Ridge on log-spend)
# Combined MAE: $67 (but high relative error on low base)

# One-and-Done CLV model:
# P(returns) = 0.18 (only 18% make a second purchase)
# E[spend | returns] = $120
# Simple model: flag the 18% likely to return, target them
# Random Forest for P(returns): AUC=0.72
# Key features: acquisition_channel (referral=28% return vs paid=12%),
#               first_order_value, category_of_first_purchase, email_opened_first_week

# Overall CLV predictions:
# Weighted MAE across all segments: $104
# Median absolute % error: 14%
```

---

## Step 11: Segment-Specific Marketing Strategy

```python
# === Marketing Automation Rules Per Segment ===

# CHAMPIONS (12%, 38% revenue):
# Strategy: Reward and deepen relationship (DO NOT discount!)
# Actions:
#   - Early access to new products (exclusivity)
#   - Loyalty program tier upgrade
#   - Personalized recommendations (cross-category)
#   - Birthday/anniversary surprise gifts
#   - Invite to brand ambassador program
# Budget: $15/customer/quarter (high touch, low discount)
# Expected ROI: 8:1

# RISING STARS (18%, 22% revenue):
# Strategy: Accelerate -- turn them into Champions
# Actions:
#   - Targeted cross-sell (categories they haven't tried)
#   - Progressive loyalty rewards (spend $X more, get Y)
#   - Content marketing (how-to guides for their purchases)
#   - Limited-time bundle offers (introduce new categories)
# Budget: $10/customer/quarter
# Expected ROI: 12:1 (highest ROI segment!)

# BARGAIN HUNTERS (25%, 18% revenue):
# Strategy: Controlled discounting -- they'll buy on sale anyway
# Actions:
#   - Personalized sale alerts (not broad discounts)
#   - Flash sale exclusives (creates urgency)
#   - Gradually reduce discount depth (test 15% vs 25%)
#   - Recommend high-margin categories
#   - DO NOT increase discount -- test reducing it
# Budget: $5/customer/quarter (mostly automated emails)
# Expected ROI: 4:1 (acceptable but focus budget elsewhere)

# DRIFTING AWAY (20%, 14% revenue):
# Strategy: Win-back -- re-engage before they fully churn
# Actions:
#   - "We miss you" campaign with personalized offer
#   - Survey: "What would bring you back?"
#   - Time-limited win-back discount (15-20%)
#   - Showcase new products in their preferred categories
#   - If no response after 2 attempts: reduce marketing spend
# Budget: $8/customer/quarter (worthwhile if converted)
# Expected ROI: 6:1 (for the 35% who re-engage)

# ONE-AND-DONE (25%, 8% revenue):
# Strategy: Selective conversion -- only target likely returners
# Actions:
#   - For the 18% predicted to return: onboarding email sequence,
#     second-purchase incentive ($10 off next order)
#   - For the 82% unlikely to return: MINIMAL spend
#     (quarterly email only, not worth more)
#   - Focus acquisition budget on channels that produce
#     non-one-and-done customers (referral >> paid social)
# Budget: $2/customer/quarter (minimal for most)
# Expected ROI: 3:1 (low, but saves money by NOT marketing to the 82%)
```

---

## Step 12: Cold-Start Problem (New Customers)

```python
# New customers have no purchase history -- can't compute RFM or behavioral features
# How do we assign them to a segment?

# Solution 1: Early Behavior Classification
# After first purchase + 7 days of behavior:
# Features available:
#   - first_order_value, first_order_categories, first_order_items
#   - acquisition_channel, device_type, location
#   - email_opened_in_first_week (yes/no)
#   - browsing_sessions_first_week, pages_viewed_first_week
#   - wishlist_additions_first_week

# Train a classifier: predict which segment they'll be in at 6 months
# Training data: historical customers (we know their features at day 7
#                 AND their segment at month 6)

from sklearn.ensemble import RandomForestClassifier
rf_coldstart = RandomForestClassifier(n_estimators=200, max_depth=6)
# 5-fold CV accuracy: 62%
# But confusion matrix shows:
#   - Champions vs Rising Stars often confused (acceptable -- treat similarly at start)
#   - One-and-Done correctly identified 71% of the time
#   - Bargain Hunters identified 58% of the time

# Solution 2: Progressive Segmentation
# Day 1: Assign to "New Customer" meta-segment (standard onboarding)
# Day 7: Classify into preliminary segment using Solution 1
# Day 30: Re-classify with 30 days of behavioral data (accuracy: 74%)
# Day 90: Full RFM available, assign to final segment (accuracy: 88%)

# This progressive approach means marketing improves as we learn more
```

---

## Step 13: Segment Stability & Drift

```python
# Segments must be stable enough to be actionable
# If customers jump between segments weekly, the segments are useless

# Stability analysis:
# Month-over-month segment transition matrix:
#
#               -> Champ  Rising  Bargain  Drift  One&Done
# Champion         92%     3%      2%      3%     0%
# Rising Star      8%     82%      4%      5%     1%
# Bargain Hunter   1%      3%     88%      6%     2%
# Drifting          2%      2%      3%     78%    15%
# One-and-Done      0%      1%      1%      3%    95%

# Segments are reasonably stable (78-95% stay in same segment month-to-month)
# Champions are stickiest (92%), Drifting is least stable (which makes sense)

# IMPORTANT: Re-cluster quarterly, not monthly
# Monthly re-clustering creates too much churn in marketing campaigns
# Quarterly gives enough time for campaigns to work

# Drift detection: monitor if segment distributions shift over time
# If Champion% drops from 12% to 8%, that's a company-wide red flag
# Monthly tracking dashboard for segment proportions
```

---

## Step 14: Final Results

```python
# === Segmentation Quality ===
# Silhouette Score: 0.42 (good for behavioral data)
# 5 segments, all interpretable and actionable
# Segment stability: 78-95% month-over-month retention
# Cold-start assignment: 62% accuracy at day 7, 88% at day 90

# === CLV Prediction ===
# Overall MAE: $104 (14% median error)
# Champions MAE: $142 (7.7% of avg CLV)
# Rising Stars MAE: $98 (15.8%)

# === Business Impact (Year 1 projection) ===
#
# Before: Uniform marketing, 8% email conversion, $2.3M promo waste
# After:  Segment-specific campaigns
#
# Revenue gains:
#   Rising Stars acceleration:     +$8.2M (convert 15% to Champions faster)
#   Drifting re-engagement:        +$4.1M (35% win-back rate)
#   One-and-Done conversion:       +$2.8M (targeted second-purchase incentive)
#   Champion deepening:            +$3.5M (cross-category expansion)
#   Total revenue gain:            +$18.6M (4.4% revenue lift)
#
# Cost savings:
#   Reduced promo waste:           -$1.8M (stop discounting Champions)
#   Reduced One-and-Done marketing: -$0.9M (don't market to 82% who won't return)
#   Total cost savings:            -$2.7M
#
# NET IMPACT: +$21.3M annually
```

---

## Step 15-16: Deployment Architecture

```python
# Pipeline architecture:

# Quarterly batch job:
# 1. Extract 18 months of transaction/behavioral data from data warehouse
# 2. Compute all 35 features per customer
# 3. Run K-Means clustering (refit with latest data)
# 4. Stability check: if >20% of customers change segment, alert data team
# 5. Run per-segment CLV models
# 6. Generate marketing lists per segment with prescribed actions
# 7. Push to marketing automation platform (Braze/Iterable/etc.)

# Real-time cold-start:
# New customer makes first purchase -> trigger cold-start classifier after 7 days
# Assign preliminary segment -> route to appropriate onboarding campaign
# Re-assess at 30 and 90 days

# Monitoring dashboard:
# - Segment distribution over time (are proportions stable?)
# - Campaign performance per segment (A/B test results)
# - CLV prediction accuracy (predicted vs actual spend at 12 months)
# - Transition matrix (are Rising Stars becoming Champions?)
# - Revenue by segment (is Champion share growing?)

# Model retraining:
# Full retrain quarterly with new data
# Cold-start classifier retrained monthly (new acquisition patterns change fast)
# CLV models validated monthly, retrained quarterly
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Features | Raw RFM only (3 features) | 35 behavioral features including trends, NMF product affinity, price sensitivity |
| Clustering | K-Means on raw data | Scaled features, tested 4 algorithms, chose K-Means for interpretability |
| K selection | Elbow method only | Silhouette + business interpretability + stability analysis |
| CLV | One model for everyone | Per-segment models (different distributions need different approaches) |
| Zero-inflate | Ignore customers with $0 spend | Hurdle model for Drifting, probability model for One-and-Done |
| Cold start | Wait for enough data (lose early engagement window) | Progressive segmentation: day 7, 30, 90 with increasing accuracy |
| Marketing | Same campaign for all segments | Segment-specific strategies with different budgets and ROI targets |
| Stability | Cluster once, never check again | Monthly transition matrix, quarterly re-clustering, drift monitoring |
| Business case | "We found 5 clusters" | $21.3M annual impact with segment-specific revenue projections |
