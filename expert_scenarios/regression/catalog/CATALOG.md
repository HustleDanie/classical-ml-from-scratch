# Regression Scenarios — Comprehensive Catalog

A catalog of ~80 realistic regression problems, organized into 18 archetype categories. Each entry summarizes the archetype, key brief signals, expected data shape, recommended approach, primary metric, and watch-outs. Entries marked **Deep dive available** link to a full ~500-line walkthrough. Entries marked **[planned]** are on the roadmap.

Use this catalog with the [methodology guide](../from_brief_to_solution/METHODOLOGY.md) — that doc walks you from a fresh brief + dataset to the matching catalog entry.

---

## How to Read an Entry

```
### N. Scenario Name
- Archetype: short structural description
- Brief signal: phrases in a brief that point at this scenario
- Data shape: typical rows × features × target distribution
- Recommended approach: model + key technique
- Primary metric: what to optimize
- Watch-out: the trap most people fall into
- Deep dive: link if available; otherwise [planned]
```

---

## The 18 Categories at a Glance

| # | Category | Count | What unites them |
|---|----------|-------|------------------|
| 1 | Property pricing | 5 | Price prediction; mix of numeric, ordinal, categorical |
| 2 | Financial / market | 5 | Noisy targets; temporal split mandatory; backtesting |
| 3 | Healthcare | 5 | Regulated; small-to-medium data; explainability required |
| 4 | Insurance | 5 | Heavy-tailed targets; outliers ARE signal |
| 5 | Manufacturing | 5 | Sensor-rich, multicollinear features |
| 6 | Energy | 5 | Time-series, weather-driven, exogenous regressors |
| 7 | Time-series forecasting | 5 | Lag features, seasonality, walk-forward validation |
| 8 | Dynamic pricing | 5 | Real-time, feedback loops, exploration/exploitation |
| 9 | Marketing / CLV | 5 | Long horizon, censored data, customer-level uncertainty |
| 10 | Logistics | 5 | Geospatial, time-of-day effects, hard latency budgets |
| 11 | Education | 3 | Small data, ordinal/bounded targets |
| 12 | HR / workforce | 3 | Small-to-medium data, mixed types, fairness concerns |
| 13 | Sports / events | 3 | Noisy, low signal-to-noise, league-specific features |
| 14 | Agriculture | 3 | Small data, geospatial, weather-driven |
| 15 | Environmental | 3 | Spatiotemporal, sensor noise, missing-by-design |
| 16 | Count regression | 3 | Poisson / Negative Binomial; integer targets |
| 17 | Quantile / interval regression | 4 | Predict P10/P50/P90, not just point estimate |
| 18 | Survival / time-to-event | 3 | Censored data; Cox PH or Random Survival Forest |

---

## Category 1 — Property Pricing

Predict price from features. Mix of numeric (sqft, beds), ordinal (condition), and categorical (neighborhood, zip).

### 1. House Price Prediction (Ames-style)
- **Archetype:** regression, mixed types, skewed target, multicollinear features
- **Brief signal:** "house price", "sale price", appraisal
- **Data shape:** 1K–500K listings × 30–100 features
- **Recommended approach:** ElasticNet baseline; XGBoost / LightGBM challenger; log-transform target
- **Primary metric:** RMSE on log(price); MAPE for business reporting
- **Watch-out:** outliers (mansions) skew RMSE; consider Huber loss or log-transform
- **Deep dive:** [01_house_price_prediction.md](01_house_price_prediction.md)

### 2. NYC Apartment Rent Prediction
- **Archetype:** regression, geospatial, mixed types, high-cardinality categoricals
- **Brief signal:** "rent", "monthly", "apartment"
- **Data shape:** 10K–500K listings × 30–80 features
- **Recommended approach:** LightGBM with target-encoded neighborhood; geospatial features (lat/long, distance to subway)
- **Primary metric:** RMSE on log(rent); MAE for stakeholder-friendly reporting
- **Watch-out:** rent control distorts the price-feature relationship in some neighborhoods
- **Deep dive:** [02_nyc_apartment_rent.md](02_nyc_apartment_rent.md)

### 3. Commercial Real Estate Valuation
- **Archetype:** regression, very high target variance, small-to-medium data
- **Brief signal:** "commercial appraisal", "office building", "warehouse"
- **Data shape:** 1K–100K properties × 20–60 features
- **Recommended approach:** Random Forest with log target; Lasso for feature selection
- **Primary metric:** MAPE (commercial valuations span 6 orders of magnitude)
- **Watch-out:** market segments (Class A office vs warehouse) need separate models
- **Deep dive:** [03_commercial_real_estate.md](03_commercial_real_estate.md)

### 4. Land / Lot Valuation
- **Archetype:** regression, sparse features, geospatial
- **Brief signal:** "land value", "lot price", undeveloped property
- **Data shape:** 10K–500K parcels × 10–30 features
- **Recommended approach:** Gradient Boosting; geospatial KNN as auxiliary feature
- **Primary metric:** MAPE; RMSE on log(price)
- **Watch-out:** zoning + utilities access dominate price; missing these kills accuracy
- **Deep dive:** [04_land_lot_valuation.md](04_land_lot_valuation.md)

### 5. Vacation Rental Nightly Rate
- **Archetype:** regression, calendar-driven, high seasonality
- **Brief signal:** "vacation rental", "Airbnb", "nightly rate"
- **Data shape:** 10K–10M listing-nights × 30–80 features
- **Recommended approach:** LightGBM with calendar features (day-of-week, holiday); per-city models
- **Primary metric:** MAPE; RMSE on log(rate)
- **Watch-out:** competitive pricing — your model competes with hosts who reprice daily
- **Deep dive:** [05_vacation_rental_rate.md](05_vacation_rental_rate.md)

---

## Category 2 — Financial / Market

Noisy targets, low signal-to-noise. Temporal split mandatory. Backtesting trumps CV.

### 6. Stock Return Forecast (Direction-Magnitude)
- **Archetype:** regression on daily/weekly returns, very low signal-to-noise
- **Brief signal:** "return forecast", "alpha"
- **Data shape:** 1K–10M instrument-days × 20–500 features
- **Recommended approach:** Lasso / ElasticNet (most signals are spurious); Gradient Boosting with strong regularization
- **Primary metric:** Out-of-sample IC (information coefficient); Sharpe on backtest
- **Watch-out:** look-ahead bias is everywhere — features must be available at decision time
- **Deep dive:** [06_stock_return_forecast.md](06_stock_return_forecast.md)

### 7. Bond Yield Prediction
- **Archetype:** regression, time-series, macro-driven
- **Brief signal:** "yield", "Treasury", "spread"
- **Data shape:** 1K–100K bond-days × 30–80 features
- **Recommended approach:** Linear baseline + XGBoost; macro features (Fed rate, CPI) dominate
- **Primary metric:** RMSE in basis points; directional accuracy
- **Watch-out:** regime changes (QE, rate hikes) break models trained on old regimes
- **Deep dive:** [07_bond_yield.md](07_bond_yield.md)

### 8. Options Pricing Residual
- **Archetype:** regression, predict Black-Scholes residual
- **Brief signal:** "implied volatility", "options pricing"
- **Data shape:** 100K–100M option-days × 20–50 features
- **Recommended approach:** Gradient Boosting; physics-informed (use BS as baseline)
- **Primary metric:** RMSE in pricing units; PnL on backtest
- **Watch-out:** liquidity drives bid-ask spread — illiquid options are noisy at any depth
- **Deep dive:** [08_options_pricing_residual.md](08_options_pricing_residual.md)

### 9. FX Rate Move Forecast
- **Archetype:** regression, time-series, macro-driven, small-effect signals
- **Brief signal:** "FX", "currency", "EURUSD"
- **Data shape:** 1K–1M ticks × 20–60 features
- **Recommended approach:** ElasticNet baseline; XGBoost with strong regularization
- **Primary metric:** Hit rate (directional); Sharpe on backtest
- **Watch-out:** central bank announcements dominate moves — model needs event handling
- **Deep dive:** [09_fx_rate_move.md](09_fx_rate_move.md)

### 10. Dividend Forecast
- **Archetype:** regression, low-frequency, fundamental-driven
- **Brief signal:** "dividend", "payout"
- **Data shape:** 1K–500K firm-quarters × 30–80 features
- **Recommended approach:** Linear regression with L2; XGBoost as challenger
- **Primary metric:** RMSE; classification-style accuracy on dividend-cut events
- **Watch-out:** dividend cuts are rare and asymmetric in cost; consider hybrid regression + classification
- **Deep dive:** [10_dividend_forecast.md](10_dividend_forecast.md)

---

## Category 3 — Healthcare

Regulated, small-to-medium data, explainability required.

### 11. Hospital Length of Stay (LOS)
- **Archetype:** regression, right-skewed target, mixed-type features
- **Brief signal:** "length of stay", "discharge date", "bed allocation"
- **Data shape:** 10K–1M admissions × 50–200 features
- **Recommended approach:** Gradient Boosting with log-transformed target + SHAP; Linear baseline for interpretability
- **Primary metric:** RMSE on log(LOS); MAE in days
- **Watch-out:** outlier stays (90+ days) need separate modeling or capped target
- **Deep dive:** [02_hospital_los.md](02_hospital_los.md)

### 12. Drug Dosage Recommendation
- **Archetype:** regression, regulated, small-to-medium data
- **Brief signal:** "dose", "pharmacokinetics", "personalized medicine"
- **Data shape:** 1K–100K patient-doses × 20–60 features
- **Recommended approach:** Linear regression with patient-specific features; XGBoost with monotonic constraints
- **Primary metric:** MAE; clinically-relevant deviation (within ±10% of optimal)
- **Watch-out:** dose-response is non-linear and patient-specific — Bayesian methods preferred
- **Deep dive:** [12_drug_dosage.md](12_drug_dosage.md)

### 13. Blood Pressure Prediction
- **Archetype:** regression, longitudinal, missing-at-random labs
- **Brief signal:** "BP", "systolic", "diastolic"
- **Data shape:** 10K–1M visits × 20–80 features
- **Recommended approach:** Random Forest baseline; longitudinal models if multi-visit data exists
- **Primary metric:** MAE in mmHg
- **Watch-out:** white-coat hypertension — measurement context matters
- **Deep dive:** [13_blood_pressure.md](13_blood_pressure.md)

### 14. Recovery Time Prediction
- **Archetype:** regression, right-skewed, small-to-medium data, censored records
- **Brief signal:** "recovery", "time to discharge", "rehab"
- **Data shape:** 1K–100K patients × 30–80 features
- **Recommended approach:** Gradient Boosting on log target; survival models if censoring is heavy
- **Primary metric:** MAE in days; calibration on long-tail
- **Watch-out:** censoring (patients who don't return) bias estimates
- **Deep dive:** [14_recovery_time.md](14_recovery_time.md)

### 15. Surgery Duration Prediction
- **Archetype:** regression, scheduling-driven
- **Brief signal:** "surgery duration", "OR scheduling"
- **Data shape:** 10K–500K surgeries × 30–80 features
- **Recommended approach:** Gradient Boosting; per-surgery-type models for high-volume procedures
- **Primary metric:** MAE in minutes
- **Watch-out:** surgeon-specific effects — random effects model or include surgeon ID
- **Deep dive:** [15_surgery_duration.md](15_surgery_duration.md)

---

## Category 4 — Insurance

Heavy-tailed targets. Outliers ARE the signal (catastrophic claims drive the math).

### 16. Insurance Claim Amount Prediction
- **Archetype:** regression, extremely heavy-tailed, two-part model preferred
- **Brief signal:** "claim amount", "loss severity"
- **Data shape:** 50K–10M claims × 20–80 features
- **Recommended approach:** Two-part model: classification (will there be a claim?) + Tweedie regression (amount given claim); Gradient Boosting works well
- **Primary metric:** Tweedie deviance; Gini index for ranking
- **Watch-out:** P99 claims dominate dollars — never cap them, never drop them
- **Deep dive:** [05_insurance_claims.md](05_insurance_claims.md)

### 17. Insurance Premium Pricing
- **Archetype:** regression, regulated, actuarial
- **Brief signal:** "premium", "rate quote", actuarial
- **Data shape:** 100K–10M policies × 30–80 features
- **Recommended approach:** Generalized Linear Model (GLM) with Gamma/Tweedie; XGBoost challenger with monotonic constraints
- **Primary metric:** Tweedie deviance; loss ratio on backtest
- **Watch-out:** state regulators require rate filings — model must be auditable
- **Deep dive:** [17_insurance_premium_pricing.md](17_insurance_premium_pricing.md)

### 18. Lifetime Claims Forecast
- **Archetype:** regression, long-horizon, censored
- **Brief signal:** "lifetime value", "cumulative claims"
- **Data shape:** 10K–10M policies × 30–80 features
- **Recommended approach:** Survival regression (Cox) for time-to-claim + claim-amount model
- **Primary metric:** Concordance for ranking; MAE on cumulative loss
- **Watch-out:** policy lapses bias estimates downward — censor-aware models required
- **Deep dive:** [18_lifetime_claims.md](18_lifetime_claims.md)

### 19. Fraud Loss Estimation
- **Archetype:** regression on fraud loss given fraud, heavy-tailed
- **Brief signal:** "fraud loss", "exposure"
- **Data shape:** 10K–500K fraud cases × 20–60 features
- **Recommended approach:** Gradient Boosting on log loss; Tweedie if zero-inflated
- **Primary metric:** RMSE on log loss; Gini on ranking
- **Watch-out:** investigation costs not in the loss number — ask if they should be
- **Deep dive:** [19_fraud_loss_estimation.md](19_fraud_loss_estimation.md)

### 20. Accident Severity Prediction
- **Archetype:** regression on injury cost; heavy-tailed
- **Brief signal:** "severity", "injury cost", auto insurance
- **Data shape:** 10K–1M accidents × 20–60 features
- **Recommended approach:** Gradient Boosting; ordinal classification as alternative
- **Primary metric:** MAPE; Gini for ranking
- **Watch-out:** medical cost inflation needs CPI-adjustment in historical data
- **Deep dive:** [20_accident_severity.md](20_accident_severity.md)

---

## Category 5 — Manufacturing

Sensor-rich, multicollinear features. Often time-series within each part.

### 21. Manufacturing Yield Prediction
- **Archetype:** regression on % yield, bounded [0, 1] target
- **Brief signal:** "yield", "process output"
- **Data shape:** 1K–500K runs × 50–500 features
- **Recommended approach:** Gradient Boosting; logit-transform target if accuracy near boundaries matters
- **Primary metric:** MAE in percentage points
- **Watch-out:** sensor drift over months — recent training data preferred
- **Deep dive:** [21_manufacturing_yield.md](21_manufacturing_yield.md)

### 22. Production Throughput Forecast
- **Archetype:** regression, time-series, capacity-constrained
- **Brief signal:** "throughput", "units per hour", "capacity"
- **Data shape:** 1K–1M time-windows × 30–80 features
- **Recommended approach:** LightGBM with time-of-day features; AR baseline
- **Primary metric:** MAE in units; MAPE
- **Watch-out:** capacity ceilings — model may predict above physically achievable; clip predictions
- **Deep dive:** [22_production_throughput.md](22_production_throughput.md)

### 23. Defect Rate Prediction
- **Archetype:** regression on defect rate, bounded, low-volume rare events
- **Brief signal:** "defect rate", "DPM", "PPM"
- **Data shape:** 1K–100K runs × 50–500 features
- **Recommended approach:** Lasso for feature selection in 200+ sensor space; LightGBM challenger
- **Primary metric:** MAE in DPM/PPM
- **Watch-out:** zero-defect runs dominate; consider Tweedie or log(rate + epsilon)
- **Deep dive:** [23_defect_rate.md](23_defect_rate.md)

### 24. Equipment Downtime Prediction
- **Archetype:** regression, right-skewed, censored
- **Brief signal:** "downtime", "MTBF", "uptime"
- **Data shape:** 1K–100K equipment-events × 30–80 features
- **Recommended approach:** Survival regression for time-to-failure; Gradient Boosting for raw downtime
- **Primary metric:** MAE in hours; concordance
- **Watch-out:** maintenance schedules confound — censor or model explicitly
- **Deep dive:** [24_equipment_downtime.md](24_equipment_downtime.md)

### 25. Quality Score Prediction
- **Archetype:** regression on quality score (continuous or ordinal)
- **Brief signal:** "quality", "grade", inline-test scores
- **Data shape:** 10K–1M parts × 50–500 features
- **Recommended approach:** Gradient Boosting; ordinal regression if quality is a discrete grade
- **Primary metric:** MAE; quadratic-weighted kappa for ordinal
- **Watch-out:** subjective quality scoring (humans grading) — inter-rater agreement bounds model accuracy
- **Deep dive:** [25_quality_score.md](25_quality_score.md)

---

## Category 6 — Energy

Time-series, weather-driven, exogenous regressors are dominant.

### 26. Building Electricity Consumption
- **Archetype:** regression, time-series, weather-driven, multi-seasonality
- **Brief signal:** "energy", "consumption", "kWh"
- **Data shape:** 1K–10M building-hours × 20–80 features
- **Recommended approach:** Gradient Boosting with weather + calendar features; SARIMA baseline
- **Primary metric:** MAPE; CV of RMSE
- **Watch-out:** holidays/weekends behave differently — explicit features required
- **Deep dive:** [10_energy_forecasting.md](10_energy_forecasting.md)

### 27. Power Grid Load Forecast
- **Archetype:** regression, time-series, very large data
- **Brief signal:** "grid load", "demand forecast", utility
- **Data shape:** 10K–10M region-hours × 30–80 features
- **Recommended approach:** Gradient Boosting + ensemble with NWP weather forecasts
- **Primary metric:** MAPE; peak-period accuracy
- **Watch-out:** peak hour prediction matters far more than off-peak — weight loss accordingly
- **Deep dive:** [27_power_grid_load_forecast.md](27_power_grid_load_forecast.md)

### 28. Oil / Gas Demand Forecast
- **Archetype:** regression, time-series, macro-driven
- **Brief signal:** "demand", "consumption", commodity
- **Data shape:** 1K–1M region-days × 30–80 features
- **Recommended approach:** SARIMA baseline + Gradient Boosting on macro features
- **Primary metric:** MAPE; per-region accuracy
- **Watch-out:** geopolitical events dominate — model needs event-handling pathway
- **Deep dive:** [28_oil_gas_demand.md](28_oil_gas_demand.md)

### 29. Solar Generation Forecast
- **Archetype:** regression, time-series, weather-driven
- **Brief signal:** "solar", "PV output", "generation"
- **Data shape:** 1K–1M panel-hours × 20–60 features
- **Recommended approach:** Gradient Boosting on weather (cloud cover, temperature, irradiance)
- **Primary metric:** MAPE; peak-hour accuracy
- **Watch-out:** physical limits (panel rating) — clip predictions
- **Deep dive:** [29_solar_generation.md](29_solar_generation.md)

### 30. EV Charging Demand
- **Archetype:** regression, time-series, behavioral + weather drivers
- **Brief signal:** "EV charging", "kWh demand"
- **Data shape:** 10K–10M station-hours × 20–60 features
- **Recommended approach:** Gradient Boosting with calendar + weather features
- **Primary metric:** MAPE; per-station accuracy
- **Watch-out:** rapidly growing market — historical data may not represent current state
- **Deep dive:** [30_ev_charging.md](30_ev_charging.md)

---

## Category 7 — Time-Series Forecasting

Lag features, seasonality, walk-forward validation.

### 31. Daily Sales Forecast
- **Archetype:** regression, time-series, multi-seasonality
- **Brief signal:** "sales forecast", "demand planning"
- **Data shape:** 1K–10M product-days × 20–80 features
- **Recommended approach:** Prophet or SARIMA baseline; LightGBM with lag features for hierarchical SKU forecasts
- **Primary metric:** MAPE; weighted-MAPE by SKU revenue
- **Watch-out:** promotional events (sales, holidays) dominate — explicit features required
- **Deep dive:** [31_daily_sales_forecast.md](31_daily_sales_forecast.md)

### 32. Inventory Demand Forecast
- **Archetype:** regression, time-series, count-like, intermittent
- **Brief signal:** "inventory", "demand", "stockout"
- **Data shape:** 100K–100M SKU-day × 20–60 features
- **Recommended approach:** Croston's method for intermittent; LightGBM for high-volume
- **Primary metric:** MAPE; service-level (P95)
- **Watch-out:** intermittent demand (zeros) breaks naive metrics — use mean absolute scaled error
- **Deep dive:** [32_inventory_demand_forecast.md](32_inventory_demand_forecast.md)

### 33. Ad Impression Forecast
- **Archetype:** regression, time-series, traffic-driven
- **Brief signal:** "impressions", "ad inventory"
- **Data shape:** 1K–100M placement-hours × 20–60 features
- **Recommended approach:** Prophet or SARIMA; LightGBM with calendar features
- **Primary metric:** MAPE; daily aggregate accuracy
- **Watch-out:** auction dynamics — supply changes when buyers change
- **Deep dive:** [33_ad_impressions.md](33_ad_impressions.md)

### 34. Web Traffic Forecast
- **Archetype:** regression, time-series, multi-seasonality, occasional spikes
- **Brief signal:** "traffic", "page views"
- **Data shape:** 10K–100M site-hours × 20–60 features
- **Recommended approach:** SARIMA baseline; Gradient Boosting for non-linear effects
- **Primary metric:** MAPE; P95 accuracy
- **Watch-out:** viral spikes are unpredictable — separate baseline + spike-handling logic
- **Deep dive:** [34_web_traffic_forecast.md](34_web_traffic_forecast.md)

### 35. Ride Volume Forecast
- **Archetype:** regression, time-series, geospatial, weather-driven
- **Brief signal:** "rides", "trips per hour"
- **Data shape:** 10K–10M zone-hours × 30–80 features
- **Recommended approach:** Gradient Boosting with geospatial + weather features; per-zone or hierarchical
- **Primary metric:** MAPE; per-zone accuracy
- **Watch-out:** events (concerts, sports) require explicit features
- **Deep dive:** [35_ride_volume_forecast.md](35_ride_volume_forecast.md)

---

## Category 8 — Dynamic Pricing

Real-time, feedback loops between price and demand. Exploration/exploitation matters.

### 36. Ride-Hail Surge Pricing
- **Archetype:** regression on price multiplier, real-time
- **Brief signal:** "surge", "dynamic pricing", "supply-demand"
- **Data shape:** 100K–1B trip-events × 20–60 features
- **Recommended approach:** LightGBM on real-time supply/demand features; bandit overlay for exploration
- **Primary metric:** Booking conversion; revenue-per-trip
- **Watch-out:** price elasticity is endogenous — historical data is pricing-policy-confounded
- **Deep dive:** [03_ride_pricing.md](03_ride_pricing.md)

### 37. Hotel Room Rate
- **Archetype:** regression on optimal rate, time-series + competitor signals
- **Brief signal:** "room rate", "ADR", "RevPAR"
- **Data shape:** 10K–10M room-days × 30–80 features
- **Recommended approach:** Gradient Boosting on calendar + occupancy + competitor rates; reinforcement learning for production
- **Primary metric:** RevPAR; booking conversion
- **Watch-out:** small inventory (10–500 rooms) means high variance — Bayesian shrinkage helps
- **Deep dive:** [37_hotel_room_rate.md](37_hotel_room_rate.md)

### 38. Dynamic Discount Optimization
- **Archetype:** regression on optimal discount %, real-time
- **Brief signal:** "discount", "promotion", "uplift"
- **Data shape:** 100K–10M offer-events × 20–80 features
- **Recommended approach:** Uplift modeling on response curve; LightGBM regression on revenue per offer
- **Primary metric:** Incremental revenue per offer
- **Watch-out:** training data is biased by past offer policy — randomization or off-policy correction needed
- **Deep dive:** [38_dynamic_discount.md](38_dynamic_discount.md)

### 39. Ad Bid Optimization
- **Archetype:** regression on optimal bid, real-time, very low latency
- **Brief signal:** "bid", "RTB", "CPM"
- **Data shape:** 10M–1B impressions × 30–80 features
- **Recommended approach:** Logistic regression with FTRL on click probability; bid = pCTR × value × discount
- **Primary metric:** ROI; CPC; conversion rate
- **Watch-out:** millisecond budgets — model size constrained by inference speed
- **Deep dive:** [39_ad_bid_optimization.md](39_ad_bid_optimization.md)

### 40. Subscription Renewal Pricing
- **Archetype:** regression on optimal renewal price, low-frequency
- **Brief signal:** "renewal price", "subscription", "churn risk"
- **Data shape:** 10K–10M customer-renewals × 30–80 features
- **Recommended approach:** Two-part: churn classifier (will they renew?) + price elasticity model
- **Primary metric:** Expected lifetime revenue
- **Watch-out:** confounds with churn modeling — joint training works better than two separate models
- **Deep dive:** [40_subscription_renewal.md](40_subscription_renewal.md)

---

## Category 9 — Marketing / CLV

Long-horizon predictions, censored data, customer-level uncertainty.

### 41. Customer Lifetime Value (CLV)
- **Archetype:** regression on cumulative spend, censored, heavy-tailed
- **Brief signal:** "CLV", "lifetime value", "LTV"
- **Data shape:** 10K–10M customers × 30–80 features
- **Recommended approach:** Two-part: retention model + spend model, or BG/NBD + Gamma-Gamma; Gradient Boosting alternative
- **Primary metric:** MAPE on holdout cohort; rank correlation
- **Watch-out:** "whale" customers (top 1%) drive 40%+ of revenue — never cap their predictions
- **Deep dive:** [41_customer_lifetime_value.md](41_customer_lifetime_value.md)

### 42. Customer Spend Forecast (Next 90 Days)
- **Archetype:** regression on near-term spend
- **Brief signal:** "next-90-days spend", "near-term LTV"
- **Data shape:** 100K–100M customers × 30–80 features
- **Recommended approach:** LightGBM with RFM features; per-segment models for high-volume cohorts
- **Primary metric:** MAPE; rank correlation
- **Watch-out:** seasonal patterns confound — use seasonal-naïve baseline
- **Deep dive:** [42_customer_spend.md](42_customer_spend.md)

### 43. Campaign ROI Prediction
- **Archetype:** regression on campaign return, small-to-medium data, causal
- **Brief signal:** "ROI", "campaign", "marketing mix"
- **Data shape:** 1K–100K campaigns × 30–80 features
- **Recommended approach:** Bayesian regression for uncertainty; Gradient Boosting for raw fit
- **Primary metric:** Out-of-sample MAPE; budget-allocation backtest
- **Watch-out:** causal vs correlational — confounding by self-selection of past campaigns
- **Deep dive:** [43_campaign_roi.md](43_campaign_roi.md)

### 44. Ad Lift / Incrementality
- **Archetype:** regression on incremental lift, causal-inference
- **Brief signal:** "lift", "incrementality", "uplift"
- **Data shape:** 100K–10M user-events × 20–80 features
- **Recommended approach:** Uplift trees / causal forests; experiments preferred over observational fit
- **Primary metric:** Qini coefficient; AUUC
- **Watch-out:** uplift is a difference of small numbers — high variance, large samples needed
- **Deep dive:** [44_ad_lift_incrementality.md](44_ad_lift_incrementality.md)

### 45. Cart Value Prediction (E-commerce)
- **Archetype:** regression, mid-skew, transactional
- **Brief signal:** "cart value", "AOV"
- **Data shape:** 100K–100M sessions × 20–60 features
- **Recommended approach:** LightGBM; log-transform target for skew
- **Primary metric:** MAPE; segmentwise accuracy
- **Watch-out:** browsing-only sessions have $0 carts — two-part model (purchase y/n + value)
- **Deep dive:** [45_cart_value.md](45_cart_value.md)

---

## Category 10 — Logistics

Geospatial, time-of-day effects, hard latency budgets.

### 46. Delivery Time Prediction
- **Archetype:** regression on minutes/hours, real-time, geospatial
- **Brief signal:** "ETA", "delivery time"
- **Data shape:** 100K–1B deliveries × 30–80 features
- **Recommended approach:** LightGBM with route features; haversine + actual road-distance from routing API
- **Primary metric:** MAE in minutes; on-time delivery rate
- **Watch-out:** traffic exogenous shocks (accidents) — model needs uncertainty estimates
- **Deep dive:** [46_delivery_time_prediction.md](46_delivery_time_prediction.md)

### 47. Fuel Cost Forecast
- **Archetype:** regression, time-series, macro-driven
- **Brief signal:** "fuel", "diesel", "gas price"
- **Data shape:** 1K–100K region-days × 20–60 features
- **Recommended approach:** SARIMA with macro features; XGBoost challenger
- **Primary metric:** MAPE
- **Watch-out:** geopolitical shocks dominate — model needs event-handling
- **Deep dive:** [47_fuel_cost_forecast.md](47_fuel_cost_forecast.md)

### 48. Route ETA Prediction
- **Archetype:** regression on travel time, real-time
- **Brief signal:** "ETA", "travel time", "routing"
- **Data shape:** 1M–1B route-events × 30–80 features
- **Recommended approach:** LightGBM; per-corridor models for high-volume routes
- **Primary metric:** MAE in seconds; P95 over-arrival rate
- **Watch-out:** time-of-day effects are non-linear — bin or Fourier-encode hour
- **Deep dive:** [48_route_eta.md](48_route_eta.md)

### 49. Warehouse Pick Time Prediction
- **Archetype:** regression on pick-task duration, operational
- **Brief signal:** "pick time", "fulfillment"
- **Data shape:** 100K–100M tasks × 20–60 features
- **Recommended approach:** Gradient Boosting with location + worker + item features
- **Primary metric:** MAE; P95 accuracy
- **Watch-out:** worker-specific effects — random effects model or worker ID as feature
- **Deep dive:** [49_warehouse_pick.md](49_warehouse_pick.md)

### 50. Package Volume / Weight Estimation
- **Archetype:** regression, geospatial + product-driven
- **Brief signal:** "package volume", "shipping weight"
- **Data shape:** 1M–1B packages × 10–30 features
- **Recommended approach:** Gradient Boosting; product-category × dimension lookup as baseline
- **Primary metric:** MAPE
- **Watch-out:** mis-declared weights (carrier surcharges) — labeled training data may have noise
- **Deep dive:** [50_package_volume.md](50_package_volume.md)

---

## Category 11 — Education

Small data, ordinal/bounded targets.

### 51. Student Grade Prediction
- **Archetype:** regression on letter-or-numeric grade, small-medium data
- **Brief signal:** "grade", "GPA", "exam score"
- **Data shape:** 1K–100K students × 20–80 features
- **Recommended approach:** Linear regression with L2; XGBoost challenger
- **Primary metric:** MAE on grade scale
- **Watch-out:** ceiling effects (max grade is bounded) — log-transform may distort
- **Deep dive:** [51_student_grade_prediction.md](51_student_grade_prediction.md)

### 52. Graduation Rate Forecast (cohort)
- **Archetype:** regression on % graduating, bounded target
- **Brief signal:** "graduation rate", "completion"
- **Data shape:** 100–10K cohorts × 20–60 features
- **Recommended approach:** Beta regression for bounded target; Random Forest as alternative
- **Primary metric:** MAE in percentage points
- **Watch-out:** Simpson's paradox — institution-level vs student-level confound
- **Deep dive:** [52_graduation_rate.md](52_graduation_rate.md)

### 53. Standardized Test Score Prediction
- **Archetype:** regression, mid-data, normal-ish distribution
- **Brief signal:** "SAT", "ACT", "test score"
- **Data shape:** 10K–1M students × 20–80 features
- **Recommended approach:** Linear regression with L2; XGBoost challenger
- **Primary metric:** RMSE on score scale
- **Watch-out:** test prep data confounds the prediction — controls for prep are essential
- **Deep dive:** [53_test_score.md](53_test_score.md)

---

## Category 12 — HR / Workforce

Small-to-medium data, mixed types, fairness concerns.

### 54. Salary Prediction
- **Archetype:** regression, mixed types, sometimes regulated (pay equity)
- **Brief signal:** "salary", "compensation"
- **Data shape:** 10K–1M employees × 30–80 features
- **Recommended approach:** Linear regression with L2 + SHAP; Random Forest challenger
- **Primary metric:** RMSE on log(salary); MAPE
- **Watch-out:** historical pay encodes bias — fairness audit by protected group required
- **Deep dive:** [54_salary_prediction.md](54_salary_prediction.md)

### 55. Time-to-Hire Prediction
- **Archetype:** regression on days-to-fill, right-skewed, censored
- **Brief signal:** "time to hire", "TTH"
- **Data shape:** 1K–100K reqs × 20–60 features
- **Recommended approach:** Gradient Boosting on log(days); survival regression if censoring is high
- **Primary metric:** MAE in days
- **Watch-out:** open reqs that never fill — censored data; ignoring biases predictions low
- **Deep dive:** [55_time_to_hire.md](55_time_to_hire.md)

### 56. Productivity Prediction
- **Archetype:** regression on output-per-period, mixed measurement
- **Brief signal:** "productivity", "output"
- **Data shape:** 1K–100K worker-periods × 20–80 features
- **Recommended approach:** Linear regression with L2; per-team random effects
- **Primary metric:** MAE; R²
- **Watch-out:** measurement bias — what counts as "productivity" varies by role
- **Deep dive:** [56_productivity.md](56_productivity.md)

---

## Category 13 — Sports / Events

Noisy targets, low signal-to-noise, league-specific features.

### 57. Athlete Performance Forecast
- **Archetype:** regression on stat (points, yards, etc.), high variance
- **Brief signal:** "performance", "season stats"
- **Data shape:** 100–100K athlete-seasons × 30–80 features
- **Recommended approach:** Bayesian shrinkage (regress to position mean); XGBoost challenger
- **Primary metric:** MAE; rank correlation
- **Watch-out:** small samples — single-season data is high variance, multi-season averaging helps
- **Deep dive:** [57_athlete_performance.md](57_athlete_performance.md)

### 58. Game Score Prediction
- **Archetype:** regression on point totals, paired with classification (win/loss)
- **Brief signal:** "game score", "totals", "spread"
- **Data shape:** 1K–100K games × 30–80 features
- **Recommended approach:** Linear regression with team + player features; XGBoost challenger
- **Primary metric:** RMSE on point totals
- **Watch-out:** time-based split — never use future games to predict past
- **Deep dive:** [58_game_score_prediction.md](58_game_score_prediction.md)

### 59. Fantasy Sports Points
- **Archetype:** regression on player points, high variance
- **Brief signal:** "fantasy points", "DFS"
- **Data shape:** 1K–1M player-games × 30–80 features
- **Recommended approach:** Gradient Boosting; per-position models
- **Primary metric:** MAE; correlation with actual outcomes
- **Watch-out:** game-script effects (blowout vs close game) change point distributions
- **Deep dive:** [59_fantasy_points.md](59_fantasy_points.md)

---

## Category 14 — Agriculture

Small data, geospatial, weather-driven.

### 60. Crop Yield Forecast
- **Archetype:** regression on tons/acre, small-medium data, weather-dominated
- **Brief signal:** "yield", "harvest"
- **Data shape:** 100–100K plot-years × 20–60 features
- **Recommended approach:** Linear regression with L2 + weather features; Gradient Boosting for non-linear effects
- **Primary metric:** MAPE; per-region accuracy
- **Watch-out:** climate change shifts the weather-yield relationship — recent data is more relevant
- **Deep dive:** [60_crop_yield_forecast.md](60_crop_yield_forecast.md)

### 61. Soil Moisture Prediction
- **Archetype:** regression on % moisture, geospatial + weather
- **Brief signal:** "soil moisture", "irrigation"
- **Data shape:** 1K–100K sensor-readings × 10–40 features
- **Recommended approach:** Linear regression on weather lags; Gradient Boosting challenger
- **Primary metric:** MAE in percentage points
- **Watch-out:** sensor calibration drift — recent calibration data only
- **Deep dive:** [61_soil_moisture.md](61_soil_moisture.md)

### 62. Livestock Weight Gain
- **Archetype:** regression on weight delta, longitudinal
- **Brief signal:** "weight gain", "ADG", livestock
- **Data shape:** 100–100K animal-periods × 10–40 features
- **Recommended approach:** Linear regression with random effects per animal; XGBoost challenger
- **Primary metric:** MAE in kg
- **Watch-out:** breed and feed effects dominate — explicit features required
- **Deep dive:** [62_livestock_weight.md](62_livestock_weight.md)

---

## Category 15 — Environmental

Spatiotemporal, sensor noise, missing-by-design.

### 63. Air Quality Index Prediction
- **Archetype:** regression on AQI, time-series + geospatial
- **Brief signal:** "AQI", "PM2.5", "air quality"
- **Data shape:** 10K–10M station-hours × 20–60 features
- **Recommended approach:** Gradient Boosting with weather + traffic features; per-station models
- **Primary metric:** MAE; P95 accuracy on high-AQI events
- **Watch-out:** sensor noise (cheap sensors) — outlier-robust models preferred
- **Deep dive:** [63_air_quality_index.md](63_air_quality_index.md)

### 64. Water Level Forecast (Rivers / Reservoirs)
- **Archetype:** regression, time-series, weather-driven
- **Brief signal:** "water level", "flood forecast"
- **Data shape:** 1K–1M station-hours × 20–60 features
- **Recommended approach:** SARIMA baseline + Gradient Boosting on weather
- **Primary metric:** MAE in meters; flood-event recall
- **Watch-out:** rare flood events drive impact — weighted loss
- **Deep dive:** [64_water_level.md](64_water_level.md)

### 65. Local Temperature Forecast
- **Archetype:** regression, time-series, atmospheric
- **Brief signal:** "temperature", "weather forecast"
- **Data shape:** 10K–10M station-hours × 30–80 features
- **Recommended approach:** Ensemble of NWP model output + Gradient Boosting on residuals
- **Primary metric:** MAE in °C; bias correction
- **Watch-out:** classical models can correct NWP bias but won't outperform — set expectations
- **Deep dive:** [65_temperature_forecast.md](65_temperature_forecast.md)

---

## Category 16 — Count Regression

Integer targets. Poisson / Negative Binomial loss preferred over Gaussian.

### 66. Website Visit Count
- **Archetype:** count regression, time-series
- **Brief signal:** "visits", "traffic count"
- **Data shape:** 1K–1M page-hours × 20–60 features
- **Recommended approach:** Poisson Regression baseline; LightGBM with `objective='poisson'`
- **Primary metric:** Poisson deviance; MAPE
- **Watch-out:** zero-inflation — many pages have zero visits in a given hour; ZIP model
- **Deep dive:** [66_website_visits.md](66_website_visits.md)

### 67. Call Center Volume
- **Archetype:** count regression, time-series, multi-seasonality
- **Brief signal:** "call volume", "contacts per hour"
- **Data shape:** 1K–1M queue-hours × 20–60 features
- **Recommended approach:** Poisson regression with calendar features; LightGBM challenger
- **Primary metric:** MAPE; service-level calibration
- **Watch-out:** abandoned-call accounting — "offered" vs "answered" volume
- **Deep dive:** [67_call_center_volume.md](67_call_center_volume.md)

### 68. Items Sold per SKU per Day
- **Archetype:** count regression, intermittent, very large data
- **Brief signal:** "units sold", "demand"
- **Data shape:** 1M–10B SKU-days × 20–60 features
- **Recommended approach:** LightGBM with `objective='tweedie'` (handles intermittent zeros + positive integers)
- **Primary metric:** Tweedie deviance; weighted-MAPE
- **Watch-out:** intermittent demand — zero-inflated models or Croston's method
- **Deep dive:** [68_items_sold_per_sku.md](68_items_sold_per_sku.md)

---

## Category 17 — Quantile / Interval Regression

Predict P10/P50/P90 instead of just point estimate. Useful when downstream decisions need uncertainty.

### 69. Worst-Case Demand (P95)
- **Archetype:** quantile regression, supply-chain planning
- **Brief signal:** "P95 demand", "service level"
- **Data shape:** 1K–10M SKU-windows × 20–60 features
- **Recommended approach:** Gradient Boosting with quantile loss (`alpha=0.95`); separate model per quantile
- **Primary metric:** Pinball loss; coverage
- **Watch-out:** non-crossing quantiles — train multiple quantiles jointly to avoid P10 > P50
- **Deep dive:** [69_worst_case_demand.md](69_worst_case_demand.md)

### 70. P99 Latency Prediction
- **Archetype:** quantile regression on latency tail
- **Brief signal:** "P99", "tail latency", SLO
- **Data shape:** 100K–100M request-events × 20–60 features
- **Recommended approach:** Gradient Boosting with quantile loss; system-feature engineering matters more than algo
- **Primary metric:** Pinball loss at 0.99
- **Watch-out:** tail predictions are noisy — large training sets required
- **Deep dive:** [70_p99_latency_prediction.md](70_p99_latency_prediction.md)

### 71. Value-at-Risk (VaR)
- **Archetype:** quantile regression, financial risk
- **Brief signal:** "VaR", "1% tail", "risk capital"
- **Data shape:** 1K–10M portfolio-days × 30–80 features
- **Recommended approach:** Quantile regression on macro factors; GARCH baseline
- **Primary metric:** Pinball loss; backtest exceptions count
- **Watch-out:** Basel III backtesting — a fixed test framework is required
- **Deep dive:** [71_value_at_risk.md](71_value_at_risk.md)

### 72. Prediction Interval (Generic)
- **Archetype:** simultaneous P10/P50/P90, deployable to any regression
- **Brief signal:** "uncertainty", "intervals"
- **Data shape:** any
- **Recommended approach:** Gradient Boosting with quantile loss for each quantile; conformal prediction wrapper
- **Primary metric:** Coverage; interval width
- **Watch-out:** conformal prediction needs a proper holdout — don't bake into training
- **Deep dive:** [72_prediction_intervals.md](72_prediction_intervals.md)

---

## Category 18 — Survival / Time-to-Event Regression

Censored data. Cox PH or Random Survival Forest for the proper treatment.

### 73. Customer Time-to-Churn
- **Archetype:** survival regression, censored
- **Brief signal:** "time to churn", "lifetime"
- **Data shape:** 10K–10M customers × 30–80 features
- **Recommended approach:** Cox Proportional Hazards baseline; Random Survival Forest challenger
- **Primary metric:** Concordance (C-index)
- **Watch-out:** treating "still active" as "0 churn" biases the model — censoring-aware fit required
- **Deep dive:** [73_customer_time_to_churn_survival.md](73_customer_time_to_churn_survival.md)

### 74. Equipment Time-to-Failure
- **Archetype:** survival regression, censored, predictive maintenance
- **Brief signal:** "remaining useful life", "MTBF", "RUL"
- **Data shape:** 1K–100K equipment-events × 50–500 features
- **Recommended approach:** Random Survival Forest or DeepSurv; Cox PH baseline
- **Primary metric:** Concordance; MAE on censored holdout
- **Watch-out:** preventive replacements (right-censored on purpose) — must be encoded as censored
- **Deep dive:** [74_equipment_time_to_failure.md](74_equipment_time_to_failure.md)

### 75. Drug Retention / Persistence
- **Archetype:** survival regression on time on treatment
- **Brief signal:** "persistence", "adherence", "stay on therapy"
- **Data shape:** 1K–100K patient-treatments × 20–60 features
- **Recommended approach:** Cox PH; Kaplan-Meier baseline
- **Primary metric:** Concordance; treatment-arm comparisons
- **Watch-out:** loss-to-followup is informative censoring — confounded with outcome
- **Deep dive:** [75_drug_retention.md](75_drug_retention.md)

---

## Where to Go Next

- Read the [methodology guide](../from_brief_to_solution/METHODOLOGY.md) to map a fresh brief into this catalog.
- Refer to [MODEL_SELECTION_GUIDE_REGRESSION.md](../../../MODEL_SELECTION_GUIDE_REGRESSION.md) for the 6 framing questions used inside each scenario.
- Refer to [ML_PIPELINE_GUIDE.md](../../../ML_PIPELINE_GUIDE.md) for the 12-phase pipeline every scenario instantiates.
