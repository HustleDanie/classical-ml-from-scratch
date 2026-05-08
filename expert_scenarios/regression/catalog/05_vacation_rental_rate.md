# Expert Scenario 5: Vacation Rental Nightly Rate (Calendar-Driven Pricing)

> **Complexity:** Calendar-driven (specific dates matter — not just day-of-week), heavy seasonality (peak summer vs winter), event-driven spikes (concerts, sports championships), competitive market response (host re-prices when competitors do), per-property variability with limited per-listing history.

---

## The Brief

A vacation-rental platform gives you 5 years of pricing and booking data for 800K listings. Hosts upload their property and the platform suggests a "smart price" per night. Hosts can override but most accept ~70% of suggestions. The brief:

- Predict the optimal nightly rate for a given listing × specific date.
- Maximize platform revenue (commission on bookings) — too high price = no booking; too low = lost revenue.
- Handle calendar-specific events: New Year's Eve in NYC is not a normal Sunday.
- Per-listing personalization (a 2BR in Park City has different dynamics than a 2BR in Pittsburgh).
- Continuously update as the market shifts (competitor prices, demand changes).

Constraints:

- MAPE within 12% on actual booked rates.
- Beat the host's manual pricing decisions in A/B test.
- Inference < 200ms per listing × date combination.
- Cold-start: handle listings with < 30 nights of history.

This is harder than typical regression because: pricing is endogenous (host's price affects whether a booking happens, which becomes training data); calendar effects compound (holiday × weekend × event); competition matters (when nearby listings drop price, this one should too); and the goal isn't just MAPE — it's revenue lift.

---

## Step 1: Define the Problem Type

```
Type:           Regression on USD per night
Primary Metric: MAPE on actually-booked rates
Secondary:      Revenue lift (vs manual host pricing) in A/B test
                Calibration on event nights (high-leverage dates)
Business Goal:  MAPE ≤ 12%; revenue lift ≥ 5% over manual pricing
Constraint:     < 200ms inference; cold-start with < 30 nights history
Target shape:   Log-normal-ish; range $30 - $5,000+
```

**Expert thinking:** revenue lift, not MAPE, is what the platform actually cares about. A model that under-prices by 10% but books 30% more nights might generate more revenue than one that hits MAPE perfectly. We optimize MAPE in training but evaluate revenue lift in A/B test.

---

## Step 2: Understand the Data

```
Shape: 800K listings × 1,825 days × pricing+booking events ≈ 1.5B observation points

Per (listing, date):
- listing_id (unique)
- date
- date_calendar_features (day_of_week, month, quarter)
- listing_features (bedrooms, capacity, location, rating)
- nightly_rate_set_by_host
- nightly_rate_recommended_by_platform
- was_booked (binary)
- booking_window_days (booked_date - check_in_date)

LISTING ATTRIBUTES (static-ish):
- city, region
- listing_lat, listing_lng
- num_bedrooms, num_bathrooms, max_capacity
- amenities_score (count of high-value amenities)
- has_pool, has_hot_tub, has_kitchen
- listing_age_days
- avg_rating (1-5)
- num_reviews
- host_response_rate

CALENDAR / EVENTS:
- holiday_flag (federal holidays)
- holiday_name
- school_break (binary, varies by region)
- known_event_within_50mi (concert, conference, sports championship)
- event_intensity_score
- season (summer / shoulder / winter)
- is_long_weekend (3+ day federal holiday)

MARKET / COMPETITIVE:
- median_rate_nearby_listings_30d
- num_competing_listings_within_5mi
- competitor_occupancy_rate_recent_30d
- price_elasticity_estimate (per-listing)

WEATHER:
- forecast_temperature_at_date
- forecast_precip_at_date

HISTORICAL RATES:
- listing_rate_lag_year (same date last year)
- listing_avg_rate_30d
- listing_avg_rate_dow (this day-of-week)

BOOKING DYNAMICS:
- days_to_check_in (how far in advance is the booking?)
- searches_for_this_listing_recent_7d (demand signal)
- view_to_book_ratio_30d
```

**Expert thinking:** the date IS a feature (not just day-of-week). New Year's Eve is structurally different from any other day. Christmas Eve, Independence Day, the day before the Super Bowl — each has its own pricing dynamics that don't generalize from "Saturday in winter."

---

## Step 3: EDA

```python
df['nightly_rate'].describe()
# count    1.5B
# mean     $174
# std      $312
# min      $30
# 25%      $89
# 50%      $135
# 75%      $215
# max     $5,000+

# Skewness 4.2 (heavy right tail)

# Calendar effect intensity
df['rate_vs_listing_avg'] = df['nightly_rate'] / df['listing_avg_rate_30d']
df.groupby('day_of_week')['rate_vs_listing_avg'].mean()
# Mon: 0.92  Tue: 0.91  Wed: 0.92  Thu: 0.95
# Fri: 1.06  Sat: 1.12  Sun: 0.98
# Friday-Saturday +10-15% premium typical

# Holiday lift
df.groupby('holiday_name')['rate_vs_listing_avg'].mean()
# Christmas Eve:    1.45
# New Year's Eve:   1.78
# Independence Day: 1.32
# Memorial Day Wknd: 1.28

# Event lift
df[df['event_intensity_score'] > 0.5].groupby('listing_id')['rate_vs_listing_avg'].mean().describe()
# Mean: 1.34 (events drive 34% premium)
# Variance high — depends on listing-event proximity
```

---

## Step 4: Feature Engineering

```python
# === CALENDAR FEATURES ===
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['week_of_year'] = df['date'].dt.isocalendar().week
df['day_of_year'] = df['date'].dt.dayofyear

# Cyclical encoding
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['woy_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
df['woy_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

# Specific high-leverage dates
df['is_christmas_eve'] = (df['date'].dt.month == 12) & (df['date'].dt.day == 24)
df['is_new_years_eve'] = (df['date'].dt.month == 12) & (df['date'].dt.day == 31)
df['is_july_4'] = (df['date'].dt.month == 7) & (df['date'].dt.day == 4)
df['is_super_bowl_sunday'] = compute_super_bowl_dates(df['date'])

# === LISTING-DATE INTERACTION ===
df['listing_avg_rate_log'] = np.log1p(df['listing_avg_rate_30d'])
df['listing_rate_lag_year_log'] = np.log1p(df['listing_rate_lag_year'])
df['listing_rate_dow_log'] = np.log1p(df['listing_avg_rate_dow'])

# === MARKET COMPARABLES ===
df['rate_vs_market_log'] = np.log1p(df['nightly_rate']) - np.log1p(df['median_rate_nearby_listings_30d'])

# === EVENT-AWARE FEATURES ===
df['event_lift_factor'] = df['event_intensity_score'] * 0.5  # rough multiplier hint
df['weekend_multiplier'] = (df['day_of_week'] >= 5).astype(int) * 0.1

# === BOOKING-WINDOW FEATURES ===
df['log_days_to_check_in'] = np.log1p(df['days_to_check_in'])
df['is_last_minute'] = (df['days_to_check_in'] <= 3).astype(int)
df['is_advance_booking'] = (df['days_to_check_in'] >= 60).astype(int)
```

---

## Step 5: Model Selection

```python
# === Model 1: Per-listing seasonal regression (one model per listing) ===
# Issue: most listings have only 1-2 years of data; over-fit to noise
# Skip; not feasible at 800K listings

# === Model 2: Global LightGBM with listing-as-categorical ===
import lightgbm as lgb
lgb_global = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=2000, max_depth=8, num_leaves=127,
    learning_rate=0.04,
    random_state=42
)
# Predict log(rate); inverse-transform
# MAPE: 11.4%  ✓ (target 12%)

# === Model 3: Per-region hierarchical model ===
# 50 regions; train per-region LightGBM on listings in that region
# Slightly better per-region MAPE but more complex deployment
# MAPE: 10.8% (~0.6pp improvement)
```

**Top performer:** Per-region LightGBM ensemble (MAPE 10.8%).

---

## Step 6: Cold-Start Handling

```python
# Listings with < 30 nights of history can't use lag features
# Fallback: use listing-attribute features only (size, location, amenities)
# Plus regional / calendar / market features

def predict_cold_start(listing_features, date_features, regional_features):
    # No lag features
    return cold_start_model.predict(features)

# Cold-start MAPE: 14.5% (worse than warm-start 11.4%)
# Acceptable; performance recovers as listing accumulates 30+ nights
```

---

## Step 7: Final Evaluation

```python
# Final model: per-region LightGBM with calendar + market + lag features
# Test set: 4 months held out
#
# Performance:
#   Overall MAPE:                 10.8%   ✓ (target 12%)
#   Per-segment:
#     Top 100 high-value listings: 8.2%
#     New / cold-start (< 30 nights): 14.5%
#     Holiday/event nights:        13.2%
#   Per-region MAPE: range 8.4% - 15%
#
# A/B test results (vs manual host pricing on 5% of listings, 30 days):
#   Bookings:                +8.4%
#   Revenue per listing:     +6.2%   ✓ (target 5%)
#   Listing utilization:     +4.1%
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Calendar features | day_of_week | Specific high-leverage dates (Xmas Eve, NYE, July 4) |
| Per-listing | 800K models or one model | Per-region ensemble (50 models — manageable) |
| Cold-start | "Not enough data, skip" | Fallback model on listing attributes only |
| Event handling | Treat as outliers | Explicit event_intensity feature; specific date flags |
| Booking dynamics | Static | Days_to_check_in feature; last-minute vs advance flags |
| Loss | MSE on raw rate | MSE on log(rate) — handles heavy tail |
| Market signals | Skip | Median nearby rate (competitive context) |
| Optimization target | MAPE | Optimize MAPE in training; evaluate REVENUE LIFT in A/B |
| Per-region drift | One global model | Per-region; some regions are vacation-heavy, others business-heavy |
