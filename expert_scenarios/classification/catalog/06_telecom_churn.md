# Expert Scenario 6: Telecom Customer Churn with Survival Analysis Twist

> **Complexity:** Not just "will they churn?" but "WHEN will they churn?" (survival analysis), multiple data sources (usage logs, billing, support tickets, network quality), competing retention offers with different costs, customer lifetime value optimization.

---

## The Brief

A telecom company has 4.2 million subscribers. Monthly churn rate is 2.1% (88,200 customers/month). Each lost customer costs $450 to replace (acquisition cost for a new customer). The retention team has capacity to call 50,000 customers/month and offer retention deals. They need to know: (1) WHO will churn in the next 30 days, (2) WHEN they'll churn so the call happens at the right time (too early = wasted effort, too late = already left), and (3) WHICH retention offer will work best (discount, upgrade, loyalty points). Currently they call whoever opened a support ticket recently -- catching only 15% of actual churners.

This is complex because: you need both classification (who) and timing (when), there are multiple intervention options with different costs and success rates, the act of retention changes the outcome (survivorship bias in historical data), and data comes from 5 different systems.

---

## Step 1: Define the Problem Type

```
Type:           Binary Classification (churn within 30 days: yes/no)
                + Survival Regression (days until churn for risk-scored customers)
Primary Metric: Precision@50K (among top 50K flagged, what % actually churn?)
                Must beat current 15% precision
Secondary:      Recall (catch as many churners as possible within the 50K call budget)
Business Goal:  Maximize net retention value = (retained_customers * $450) - (call_costs + offer_costs)
Constraint:     Can only call 50,000 customers/month (team capacity)
```

**Expert thinking:** This isn't just classification. We need to RANK customers by churn risk, take the top 50K, and have those be as precise as possible. Also, the TIMING matters: if we predict someone will churn on day 25, we should call them on day 20, not day 5.

---

## Step 2: Understand the Data

```
5 data sources to join:

Table 1: Customer Profile (4.2M rows)
- customer_id, signup_date, plan_type (prepaid/postpaid)
- contract_type (month-to-month, 1-year, 2-year)
- monthly_charge, total_charges
- payment_method (auto-pay, manual, credit_card, bank_transfer)
- age, gender, partner, dependents
- tenure_months (how long they've been a customer)

Table 2: Usage Logs (420M rows/month -- aggregated to monthly)
- customer_id, month
- minutes_used, data_gb_used, sms_count
- international_minutes, roaming_days
- peak_hour_usage_pct (what % of usage is during peak hours)
- unique_contacts_called (social network size)

Table 3: Billing History (50M rows -- one per customer per month)
- customer_id, month
- bill_amount, payment_amount, payment_date
- days_late (payment delay), had_overcharge_dispute
- plan_changes (upgrades/downgrades this month)

Table 4: Support Tickets (2.8M rows -- one per ticket)
- customer_id, ticket_date
- issue_category (network, billing, technical, complaint, inquiry)
- resolution_time_hours
- satisfaction_score (1-5, from post-ticket survey, 40% missing)
- escalated (binary)

Table 5: Network Quality (per cell tower, daily)
- tower_id, date
- avg_signal_strength, dropped_call_rate, data_speed_mbps
- outage_minutes
(Joined to customers via their primary tower / home address)

Target: churned_within_30d (binary), days_until_churn (for those who churned)
```

---

## Step 3: EDA Findings

| Finding | Implication |
|---------|------------|
| Month-to-month contracts churn at 4.8% vs 0.5% for 2-year contracts | Contract type is huge |
| Customers who called support 3+ times in past month churn at 8.2% | Support frequency = distress signal |
| Usage dropped >30% month-over-month in 45% of churners | Usage decline precedes churn |
| Auto-pay customers churn at 1.2% vs 3.8% for manual payment | Payment method signals engagement |
| Customers in areas with >5% dropped calls churn at 3.5% vs 1.8% | Network quality drives churn |
| 70% of churners show a "churn trajectory": support ticket -> usage drop -> missed payment -> churn | There's a predictable sequence |
| Customers who received retention offers have LOWER churn | BUT: survivorship bias -- they were going to stay anyway OR the offer worked |
| Tenure < 6 months: 4.5% churn; 6-24 months: 2.1%; 24+ months: 1.1% | New customers churn more |
| Satisfaction score < 3 after support: 12% churn vs 1.5% for score 5 | Satisfaction is powerful but 40% missing |

---

## Step 4-5: Cleaning & Feature Engineering

```python
# === Survivorship Bias Handling ===
# Customers who received retention offers in the past are biased:
# - Some stayed BECAUSE of the offer (offer worked)
# - Some stayed anyway (wasted offer)
# - Some left despite the offer (offer failed)
# Solution: Create 'received_retention_offer' as a FEATURE, not ignore it
# Also: train on UNTREATED customers where possible for unbiased churn prediction
# For treated customers: model P(churn | no intervention) as counterfactual

# === Feature Engineering ===

# Usage trend features (most important category)
for window in [1, 2, 3]:
    df[f'minutes_change_{window}m'] = (
        df['minutes_used'] - df[f'minutes_used_lag{window}']
    ) / (df[f'minutes_used_lag{window}'] + 1)
    # -0.4 means 40% drop in usage -- strong churn signal

df['data_usage_trend_3m'] = ... # slope of data usage over 3 months
df['usage_declined_3_consecutive'] = ... # usage dropped 3 months in a row

# Engagement features
df['days_since_last_call'] = ... # inactivity
df['avg_daily_sessions'] = ...
df['active_days_per_month'] = ...
df['is_inactive_7d'] = (df['days_since_last_call'] > 7).astype(int)

# Support distress features
df['tickets_last_30d'] = ...
df['tickets_last_90d'] = ...
df['ticket_acceleration'] = df['tickets_last_30d'] - df['tickets_last_30d_shifted']
# More tickets than usual = escalating frustration

df['has_unresolved_ticket'] = ...
df['worst_satisfaction_last_90d'] = ...
df['pct_tickets_escalated'] = ...

# Payment behavior features
df['payment_delay_trend'] = ... # are they paying later each month?
df['missed_payments_3m'] = ...
df['autopay_removed_recently'] = ... # turned OFF autopay = red flag

# Network quality at customer's location
df['avg_signal_strength_30d'] = ...
df['dropped_call_pct_30d'] = ...
df['had_outage_last_30d'] = ...
df['network_quality_decline'] = ... # quality getting worse in their area

# Contract features
df['months_until_contract_end'] = ...
df['contract_ending_soon'] = (df['months_until_contract_end'] <= 2).astype(int)
# Contract ending = decision point

# Competitive pressure proxy
df['plan_is_overpriced'] = (df['monthly_charge'] > df['market_avg_for_usage']).astype(int)
# Paying more than they'd pay elsewhere

# Social network features
df['contacts_who_churned_recently'] = ... # friends leaving = social churn
df['unique_contacts_trend'] = ... # calling fewer people = disengagement

# Composite risk signals
df['churn_trajectory_score'] = (
    (df['tickets_last_30d'] >= 2).astype(int) +
    (df['minutes_change_1m'] < -0.2).astype(int) +
    (df['payment_delay_trend'] > 0).astype(int) +
    (df['contract_ending_soon']).astype(int)
)
# Score 3-4 = classic churn trajectory
```

---

## Step 6-10: Model Building

```python
# Two-level prediction:

# ================================================================
# LEVEL 1: Churn Risk Score (0-1 probability)
# ================================================================

# LightGBM with calibration (need accurate probabilities for ranking):
# AUC-ROC: 0.91
# At top 50K (1.2% of customer base):
#   Precision: 38% (19,000 actual churners found)
#   vs current system: 15% (7,500 found)
#   Improvement: 153% more churners identified within same call budget

# Per-risk-tier breakdown:
# Top 10K:    62% actually churn (call these FIRST)
# Next 10K:   45% actually churn
# Next 10K:   34% actually churn
# Next 10K:   27% actually churn
# Next 10K:   22% actually churn (still worth calling at $450 retention value)

# ================================================================
# LEVEL 2: Days Until Churn (Survival Model)
# ================================================================

# For the top 50K risk-scored customers, predict WHEN they'll churn
# This helps schedule the retention call optimally

# Cox Proportional Hazards (classical survival analysis):
# Concordance Index: 0.74

# Random Survival Forest:
# Concordance Index: 0.79

# LightGBM as survival model (custom objective):
# Concordance Index: 0.81

# Output: For each customer, a survival curve showing:
# P(still a customer at day 5) = 0.95
# P(still a customer at day 10) = 0.88
# P(still a customer at day 15) = 0.72
# P(still a customer at day 20) = 0.51
# P(still a customer at day 25) = 0.30
# P(still a customer at day 30) = 0.18

# Optimal call timing: when survival probability crosses 0.60
# (this is when churn risk is high but the customer hasn't mentally "left" yet)
# For this customer: call around day 13-15
```

---

## Step 11: Retention Offer Optimization

```python
# Three offers available:
# Offer A: 20% discount for 6 months (cost: ~$180)
# Offer B: Free device upgrade (cost: ~$300)
# Offer C: Loyalty points + priority support (cost: ~$50)

# Historical data shows:
# Offer A works best for: price-sensitive customers (usage decline + high charge)
# Offer B works best for: tech enthusiasts (high data usage, old device)
# Offer C works best for: long-tenure customers (emotional loyalty)

# Train a separate model: given a customer profile + offer type, predict P(retention)

# Results:
# Random offer:     35% retention rate among called churners
# Personalized offer: 52% retention rate among called churners
# Improvement: +49% retention effectiveness

# Net value calculation:
# 50,000 calls/month
# 19,000 are actual churners (38% precision)
# With personalized offers: 9,880 retained (52% of 19,000)
# Saved: 9,880 * $450 = $4,446,000
# Offer costs: avg $170 * 50,000 = $8,500,000... wait that's more than savings
#
# Correction: only offer to those likely to accept AND likely to churn
# Refined targeting:
# - Only call top 50K risk scores
# - Only offer to those where P(churn) > 0.25 AND P(retention|offer) > 0.40
# - This filters to ~22,000 calls with offers, rest get "just checking in" calls
# Offer costs: $170 * 22,000 = $3,740,000
# Retained: 11,440 customers
# Saved: 11,440 * $450 = $5,148,000
# Net benefit: $5,148,000 - $3,740,000 - $220,000 (call center costs) = $1,188,000/month
# Annual net benefit: $14.3M
```

---

## Step 12-14: Final Results

```python
# Test set: 1 month of unseen data (4.2M customers, ~88,200 actual churners)
#
# Current system (call whoever submitted ticket):
#   Calls: 50,000 | Churners found: 7,500 | Retained: 2,625 | Net: -$1.2M (losing money!)
#
# ML system:
#   Calls: 50,000 | Churners found: 19,000 | Retained: 11,440 | Net: +$1.19M/month
#
# Key metrics:
# AUC-ROC: 0.91
# Precision@50K: 38% (vs 15% current -- 2.5x improvement)
# Optimal call timing accuracy: within 5 days of actual churn 71% of time
# Personalized offer acceptance: 52% (vs 35% random offers)

# Top SHAP features:
# 1. minutes_change_1m (usage drop = churn signal)
# 2. contract_ending_soon
# 3. tickets_last_30d
# 4. churn_trajectory_score (composite feature)
# 5. payment_delay_trend
# 6. network_quality_decline
# 7. contacts_who_churned_recently (social contagion)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Problem framing | Binary churn yes/no | Three-level: who + when + which offer |
| Survivorship bias | Ignore past retention offers | Model counterfactual P(churn given no intervention) |
| Timing | "Will churn this month" only | Survival model predicting days-until-churn for optimal call timing |
| Ranking | Classify then sort | Optimize Precision@50K (matching call capacity) |
| Retention | Same offer to everyone | Personalized offers based on customer profile + uplift modeling |
| Usage features | Current month usage | 3-month trends, acceleration, consecutive declines |
| Social network | Ignore it | Contacts-who-churned (social contagion effect) |
| ROI | "AUC is 0.91" | Net benefit: $14.3M/year including offer costs and call center capacity |
| Network quality | Ignore infrastructure | Joined tower-level quality data to customer location |
