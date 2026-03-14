# Feature Selection Guide

How to determine which features (columns) to keep and which to drop. 3 stages: Common Sense, Statistical Methods, and Automated Selection.

---

## Stage 1: Common Sense — Columns to Always Check

### Obvious Drops (Almost Always Remove)

| Column Type | Examples You'll See | Why Drop | How to Detect |
|------------|-------------------|---------|--------------|
| Row identifier | `id`, `index`, `row_id`, `record_id`, `serial_no` | Every value unique — model memorizes instead of learning | `df[col].nunique() == len(df)` |
| Primary/foreign key | `customer_id`, `order_id`, `transaction_id`, `employee_id`, `patient_id` | Database keys, not meaningful features | Column name ends in `_id` and all values unique |
| Name fields | `first_name`, `last_name`, `full_name`, `username`, `email` | Too many unique values, no predictive pattern | Very high cardinality string column |
| Exact timestamps used as IDs | `created_at` (to the second), `timestamp`, `log_time` | Too granular — every row has different value | datetime with second/millisecond precision, mostly unique |
| Constants | A column where every row = same value (e.g., `country = "US"` for all rows) | Zero variance = zero information | `df[col].nunique() == 1` |
| Near-constants | A column where 99.9% of rows have the same value | Almost no information | `df[col].value_counts(normalize=True).iloc[0] > 0.999` |
| Exact duplicates of another column | `price_usd` and `price_dollars`, `age` and `age_years` | Same information twice adds nothing | `df[col1].equals(df[col2])` or correlation = 1.0 |
| Target leakers | Column that is derived FROM the target or IS the target in disguise | Gives artificially perfect accuracy, useless in production | Suspiciously high correlation (> 0.95) with target |
| Free text without processing | `comments`, `notes`, `description`, `feedback` (raw paragraphs) | Model can't use raw strings — needs TF-IDF or embedding first | String column with long text, high cardinality |
| URL / file path columns | `image_url`, `profile_link`, `file_path` | Not a meaningful feature as-is | Contains `http`, `/`, `.com`, file extensions |
| Internal system codes | `error_code_hex`, `log_hash`, `session_token`, `api_key` | System metadata, not related to what you're predicting | Looks random — hashes, hex codes, UUIDs |

### Columns to Investigate Before Deciding

| Column Type | Examples | Keep If... | Drop If... |
|------------|---------|-----------|-----------|
| Dates | `date_of_birth`, `signup_date`, `purchase_date` | You extract useful features (age, day_of_week, month, days_since) | You leave it as a raw date string |
| Phone numbers | `phone`, `mobile` | You extract area code or country code as a feature | You use the full number (it's basically an ID) |
| Zip codes / postal codes | `zip_code`, `postal_code` | You map to region, income bracket, or urban/rural | You use the raw 5-digit code (too many unique values) |
| Categorical with too many levels | `city` (500 unique), `product_sku` (10,000 unique) | You group into top-N + "other" or use target/frequency encoding | You one-hot encode it (creates thousands of sparse columns) |
| Mostly missing columns | Any column with > 70% missing values | The non-missing values are highly predictive | The non-missing values have no pattern |
| Highly correlated feature pairs | `height_cm` and `height_inches`, `total_price` and `unit_price * quantity` | You keep the one more correlated with target | You keep both (redundancy wastes model capacity) |
| Columns from the future | `outcome_date`, `resolution_status` (if predicting whether something resolves) | Never — this is data leakage | Always — not available at prediction time |
| Ordinal encoded as text | `"low"`, `"medium"`, `"high"` or `"poor"`, `"fair"`, `"good"`, `"excellent"` | You convert to numeric (1, 2, 3, 4) | You leave as unordered text |

### Common Columns in Typical Datasets

| Domain | Columns You'll Almost Always See | Keep/Drop/Transform |
|--------|--------------------------------|-------------------|
| **Any dataset** | `id`, `index` | DROP — always |
| **Any dataset** | `name`, `email` | DROP — identifier, not feature |
| **Customer data** | `customer_id`, `account_number` | DROP — identifier |
| **Customer data** | `age`, `gender`, `income`, `region` | KEEP — core demographics |
| **Customer data** | `signup_date` | TRANSFORM — extract tenure, day_of_week |
| **Customer data** | `last_login_date` | TRANSFORM — extract days_since_last_login |
| **E-commerce** | `order_id`, `product_id` | DROP — identifiers |
| **E-commerce** | `quantity`, `unit_price`, `total_price` | KEEP quantity + unit_price, DROP total_price (it's quantity * unit_price = leaker/redundant) |
| **Healthcare** | `patient_id`, `medical_record_number` | DROP — identifier |
| **Healthcare** | `admission_date`, `discharge_date` | TRANSFORM — extract length_of_stay, day_of_week |
| **Finance** | `transaction_id`, `account_id` | DROP — identifier |
| **Finance** | `transaction_amount`, `balance`, `credit_limit` | KEEP — core financial features |
| **HR / Employee** | `employee_id`, `ssn` | DROP — identifier / sensitive |
| **HR / Employee** | `department`, `salary`, `years_experience`, `education_level` | KEEP — core features |
| **Real estate** | `listing_id`, `address` | DROP listing_id; TRANSFORM address to region/zip |
| **Real estate** | `square_feet`, `bedrooms`, `bathrooms`, `lot_size` | KEEP — core property features |
| **IoT / Sensor** | `device_id`, `timestamp` | DROP device_id; TRANSFORM timestamp to hour, day_of_week |
| **IoT / Sensor** | `temperature`, `humidity`, `pressure`, `vibration` | KEEP — core sensor readings |

---

## Stage 2: Statistical Methods

### For Linear Relationships

Methods that detect features with a straight-line relationship to the target.

#### Method 1: Pearson Correlation

Measures the linear relationship between a feature and the target on a scale from -1 to +1.

| Pearson Correlation Value | Interpretation | Action |
|--------------------------|---------------|--------|
| 0.7 to 1.0 (or -0.7 to -1.0) | Strong linear relationship | Definitely keep |
| 0.4 to 0.7 (or -0.4 to -0.7) | Moderate linear relationship | Keep |
| 0.1 to 0.4 (or -0.1 to -0.4) | Weak linear relationship | Keep but monitor — may not help |
| -0.1 to 0.1 | No linear relationship | Candidate for removal (but check non-linear methods first!) |
| Exactly 0 | Zero linear correlation | Remove if non-linear methods also show no relationship |

| What to Watch For | Why |
|------------------|-----|
| High positive correlation (close to +1) | Feature and target move together — very useful |
| High negative correlation (close to -1) | Feature and target move in opposite directions — equally useful |
| Near zero but non-linear pattern exists | Pearson misses it — e.g., U-shaped relationship shows ~0 correlation |
| Two features correlated > 0.9 with each other | Multicollinearity — drop one of them |

**Limitations:** Only detects linear patterns. A perfectly U-shaped relationship (e.g., both very young and very old have high medical costs) shows Pearson correlation near 0, even though the feature is extremely useful.

---

#### Method 2: Spearman Rank Correlation

Measures monotonic (always increasing or always decreasing) relationships, not just straight lines. Converts values to ranks first.

| Spearman Correlation Value | Interpretation | Action |
|---------------------------|---------------|--------|
| 0.7 to 1.0 (or -0.7 to -1.0) | Strong monotonic relationship | Definitely keep |
| 0.4 to 0.7 (or -0.4 to -0.7) | Moderate monotonic relationship | Keep |
| 0.1 to 0.4 (or -0.1 to -0.4) | Weak monotonic relationship | Keep but monitor |
| -0.1 to 0.1 | No monotonic relationship | Candidate for removal |

| When Spearman Beats Pearson | Example |
|----------------------------|---------|
| Relationship is monotonic but curved | Income vs spending (increases but curves) |
| Outliers exist in the data | Pearson gets distorted, Spearman uses ranks so outliers don't matter |
| Ordinal features | Education level (1=high school, 2=bachelor, 3=master, 4=PhD) |

**Limitations:** Still misses non-monotonic patterns (U-shapes, periodic patterns).

---

#### Method 3: F-test (ANOVA F-statistic)

Tests whether the feature has a statistically significant linear relationship with the target. Used by SelectKBest with `f_regression` or `f_classif`.

| F-statistic | Interpretation | Action |
|------------|---------------|--------|
| Very high (> 100) | Extremely strong linear relationship | Definitely keep |
| High (10–100) | Strong relationship | Keep |
| Low (1–10) | Weak relationship | Keep if other methods agree |
| < 1 | No relationship | Remove |

| p-value | Interpretation | Action |
|---------|---------------|--------|
| < 0.001 | Highly significant — relationship is real | Keep |
| 0.001 – 0.05 | Significant | Keep |
| 0.05 – 0.1 | Marginally significant | Investigate further |
| > 0.1 | Not significant — relationship is likely noise | Remove |

**Limitations:** Assumes linear relationship and normally distributed features. Sensitive to outliers.

---

#### Method 4: Point-Biserial Correlation (for binary target)

Special case of Pearson correlation when the target is binary (0/1). Measures how well a continuous feature separates the two classes.

| Point-Biserial Value | Interpretation | Action |
|---------------------|---------------|--------|
| > 0.3 (or < -0.3) | Good class separation | Definitely keep |
| 0.1 – 0.3 | Moderate separation | Keep |
| < 0.1 | Poor separation | Candidate for removal |

---

### 10 Real Scenarios: Feature Selection Using Linear Methods

#### Scenario 1: Predicting House Prices (15 features)

| Feature | Pearson with Price | Spearman with Price | F-test p-value | Decision |
|---------|-------------------|--------------------|--------------|---------|
| `square_feet` | 0.82 | 0.79 | < 0.001 | **KEEP** — strong linear relationship |
| `bedrooms` | 0.53 | 0.51 | < 0.001 | **KEEP** — moderate linear |
| `bathrooms` | 0.61 | 0.58 | < 0.001 | **KEEP** — moderate-strong linear |
| `lot_size` | 0.38 | 0.35 | < 0.001 | **KEEP** — weak but significant |
| `garage_size` | 0.55 | 0.52 | < 0.001 | **KEEP** — moderate |
| `year_built` | 0.42 | 0.45 | < 0.001 | **KEEP** — Spearman slightly higher (monotonic but curved) |
| `distance_to_school` | -0.31 | -0.29 | 0.002 | **KEEP** — negative correlation (farther = cheaper) |
| `property_tax` | 0.91 | 0.88 | < 0.001 | **INVESTIGATE** — suspiciously high, may be derived from price (leaker) |
| `owner_age` | 0.03 | 0.02 | 0.72 | **REMOVE** — no relationship at all |
| `listing_month` | 0.01 | 0.01 | 0.85 | **REMOVE** — month of listing doesn't correlate |

**Outcome:** Kept 7 features, removed 2, investigated 1 for leakage. Ridge Regression with 7 features got R² = 0.87, same as all 15 features (R² = 0.86). Simpler model won.

---

#### Scenario 2: Predicting Student GPA (8 features)

| Feature | Pearson with GPA | F-test p-value | Decision |
|---------|-----------------|--------------|----------|
| `study_hours_per_week` | 0.74 | < 0.001 | **KEEP** — strongest predictor |
| `attendance_rate` | 0.68 | < 0.001 | **KEEP** — strong linear |
| `previous_gpa` | 0.81 | < 0.001 | **KEEP** — very strong |
| `sleep_hours` | 0.35 | 0.001 | **KEEP** — weak but significant |
| `part_time_job_hours` | -0.42 | < 0.001 | **KEEP** — negative (more work = lower GPA) |
| `parent_income` | 0.12 | 0.09 | **BORDERLINE** — p-value near 0.05, weak |
| `commute_time` | -0.08 | 0.31 | **REMOVE** — not significant |
| `student_id` | 0.01 | 0.89 | **REMOVE** — identifier, no relationship |

**Outcome:** Kept 5 features + borderline `parent_income`. Linear Regression with 5 features: R² = 0.72. With 6 features: R² = 0.73. Keeping `parent_income` gave marginal gain — included it since only 6 features total (no overfitting risk).

---

#### Scenario 3: Predicting Monthly Electricity Bill (12 features)

| Feature | Pearson | Spearman | Feature-Feature Correlation | Decision |
|---------|---------|----------|---------------------------|----------|
| `kwh_consumed` | 0.97 | 0.95 | — | **INVESTIGATE** — almost perfect correlation, is this a leaker? (Yes — bill = rate * kwh. Removed.) |
| `square_footage` | 0.62 | 0.59 | 0.85 with `num_rooms` | **KEEP** — kept this over `num_rooms` (higher Pearson with target) |
| `num_rooms` | 0.58 | 0.55 | 0.85 with `square_footage` | **REMOVE** — redundant with `square_footage` |
| `num_occupants` | 0.51 | 0.48 | — | **KEEP** — independent predictor |
| `avg_temperature` | 0.44 | 0.30 | — | **KEEP** — but Spearman lower than Pearson (possible non-linear) |
| `has_ac` | 0.47 | 0.45 | — | **KEEP** — binary, Point-Biserial = 0.47 |
| `appliance_count` | 0.39 | 0.37 | — | **KEEP** |
| `insulation_rating` | -0.33 | -0.35 | — | **KEEP** — better insulation = lower bill |
| `account_id` | 0.00 | 0.01 | — | **REMOVE** — identifier |
| `billing_cycle_day` | 0.02 | 0.01 | — | **REMOVE** — which day the bill is generated is irrelevant |

**Outcome:** Started with 12 features, ended with 7. Removed 1 leaker, 1 redundant pair member, 2 irrelevant. Ridge R² went from 0.65 (all 12, inflated by leaker) to 0.78 (7 features, honest model).

---

#### Scenario 4: Predicting Employee Salary (10 features)

| Feature | Pearson with Salary | Spearman with Salary | Decision |
|---------|--------------------|--------------------|----------|
| `years_experience` | 0.78 | 0.82 | **KEEP** — Spearman higher (salary growth curves, not purely linear) |
| `education_level` (1-4 ordinal) | 0.55 | 0.58 | **KEEP** — Spearman better for ordinal data |
| `department_size` | 0.08 | 0.07 | **REMOVE** — barely any relationship |
| `performance_rating` | 0.49 | 0.47 | **KEEP** — solid moderate correlation |
| `age` | 0.72 | 0.75 | **INVESTIGATE** — highly correlated with `years_experience` (r = 0.92). Drop one. |
| `certifications_count` | 0.31 | 0.29 | **KEEP** — weak but meaningful |
| `manager_rating` | 0.44 | 0.41 | **KEEP** |
| `commute_distance` | -0.05 | -0.04 | **REMOVE** — no relationship |
| `employee_id` | 0.02 | 0.01 | **REMOVE** — identifier |
| `hire_date` | 0.71 | 0.74 | **TRANSFORM** — extracted `tenure_years`, which correlated 0.76 with salary |

**Outcome:** Dropped `age` (kept `years_experience` — more directly causal), dropped `department_size` and `commute_distance`, transformed `hire_date`. Final 6 features. Lasso zeroed out `certifications_count` too, leaving 5. R² = 0.81.

---

#### Scenario 5: Predicting Crop Yield (6 features)

| Feature | Pearson with Yield | F-test p-value | Decision |
|---------|-------------------|--------------|----------|
| `rainfall_mm` | 0.71 | < 0.001 | **KEEP** — strong |
| `temperature_avg` | 0.45 | < 0.001 | **KEEP** — moderate |
| `fertilizer_kg` | 0.63 | < 0.001 | **KEEP** — strong |
| `soil_ph` | 0.22 | 0.04 | **KEEP** — weak but significant (and only 6 features total) |
| `altitude_m` | -0.18 | 0.08 | **BORDERLINE** — p near 0.05 |
| `farm_id` | 0.03 | 0.78 | **REMOVE** — identifier |

**Outcome:** Only 6 features — kept all 5 real ones (including borderline `altitude_m`). With this few features, removing any risks losing information. Linear Regression R² = 0.69. No improvement from dropping `altitude_m`, but no harm keeping it.

---

#### Scenario 6: Predicting Insurance Premium (20 features)

| Feature | Pearson | Spearman | Feature-Feature Corr Issue? | Decision |
|---------|---------|----------|---------------------------|----------|
| `age` | 0.65 | 0.63 | — | **KEEP** |
| `bmi` | 0.51 | 0.48 | — | **KEEP** |
| `smoker` (0/1) | 0.78 | 0.76 | — | **KEEP** — strongest predictor |
| `num_children` | 0.07 | 0.06 | — | **REMOVE** — negligible |
| `annual_income` | -0.04 | -0.03 | — | **REMOVE** — no relationship |
| `region_north` | 0.02 | 0.02 | — | **REMOVE** — one-hot encoded region, no effect |
| `region_south` | -0.01 | -0.01 | — | **REMOVE** — same |
| `region_east` | 0.03 | 0.03 | — | **REMOVE** — same |
| `region_west` | -0.04 | -0.03 | — | **REMOVE** — same |
| `exercise_frequency` | -0.38 | -0.41 | — | **KEEP** — moderate negative |
| `pre_existing_conditions` | 0.59 | 0.57 | — | **KEEP** |
| `weight_kg` | 0.49 | 0.46 | 0.93 with `bmi` | **REMOVE** — redundant with `bmi` |
| `height_cm` | -0.11 | -0.10 | Part of `bmi` calculation | **REMOVE** — `bmi` already captures this |

**Outcome:** 20 features reduced to 6. Four region dummies had no effect (premium doesn't vary by region). Weight/height redundant with BMI. Income and children irrelevant. Ridge R² = 0.85 with 6 features vs R² = 0.84 with all 20. Fewer features, slightly better.

---

#### Scenario 7: Predicting Delivery Time in Days (14 features)

| Feature | Pearson with Delivery Days | F-test p-value | Decision |
|---------|---------------------------|--------------|----------|
| `distance_km` | 0.82 | < 0.001 | **KEEP** — dominant predictor |
| `package_weight_kg` | 0.29 | < 0.001 | **KEEP** — weak but significant |
| `order_hour` | 0.04 | 0.52 | **REMOVE** — hour of day doesn't matter |
| `is_weekend_order` | 0.15 | 0.02 | **KEEP** — weekend orders take longer |
| `shipping_method` (1-3 ordinal) | -0.61 | < 0.001 | **KEEP** — express vs standard matters a lot |
| `warehouse_stock` (0/1) | -0.44 | < 0.001 | **KEEP** — in-stock items ship faster |
| `customer_loyalty_tier` | -0.08 | 0.22 | **REMOVE** — loyalty tier doesn't speed up delivery |
| `coupon_used` (0/1) | 0.01 | 0.91 | **REMOVE** — completely irrelevant |
| `order_id` | 0.00 | 0.98 | **REMOVE** — identifier |
| `item_category` | 0.06 | 0.35 | **REMOVE** — category doesn't affect delivery speed |

**Outcome:** 14 features down to 5. Linear Regression R² with 5 features = 0.76, with all 14 = 0.75. Exact same accuracy with fewer features.

---

#### Scenario 8: Predicting Car Fuel Efficiency (MPG) (9 features)

| Feature | Pearson with MPG | Spearman | Feature-Feature Issue? | Decision |
|---------|-----------------|----------|----------------------|----------|
| `engine_displacement` | -0.80 | -0.82 | 0.95 with `cylinders` | **KEEP** — stronger of the pair |
| `cylinders` | -0.78 | -0.80 | 0.95 with `displacement` | **REMOVE** — redundant |
| `horsepower` | -0.77 | -0.79 | 0.84 with `displacement` | **KEEP** — correlated but < 0.9, adds info |
| `weight_lbs` | -0.83 | -0.85 | — | **KEEP** — strongest predictor |
| `acceleration` | 0.42 | 0.38 | — | **KEEP** — moderate positive |
| `model_year` | 0.58 | 0.60 | — | **KEEP** — newer cars more efficient |
| `origin` (1-3) | 0.57 | 0.55 | — | **KEEP** — region of manufacture matters |
| `car_name` | 0.01 | 0.02 | — | **REMOVE** — text string, too many unique values |
| `vin_number` | 0.00 | 0.00 | — | **REMOVE** — identifier |

**Outcome:** 9 features to 6. Dropped `cylinders` (redundant with `displacement`), `car_name` and `vin_number` (identifiers). Ridge R² = 0.82 with 6 features.

---

#### Scenario 9: Predicting Hospital Readmission Days (18 features)

| Feature | Pearson | Spearman | F-test p-value | Decision |
|---------|---------|----------|--------------|----------|
| `age` | 0.41 | 0.44 | < 0.001 | **KEEP** |
| `length_of_stay` | 0.56 | 0.58 | < 0.001 | **KEEP** — longer stays = sicker patients = sooner readmission |
| `num_medications` | 0.47 | 0.50 | < 0.001 | **KEEP** |
| `num_procedures` | 0.33 | 0.35 | < 0.001 | **KEEP** |
| `num_diagnoses` | 0.39 | 0.42 | < 0.001 | **KEEP** |
| `num_lab_results` | 0.28 | 0.30 | < 0.001 | **KEEP** — proxy for severity |
| `had_emergency` (0/1) | 0.22 | 0.21 | 0.003 | **KEEP** |
| `insurance_type` | 0.08 | 0.07 | 0.18 | **REMOVE** — not significant |
| `admission_day_of_week` | 0.02 | 0.02 | 0.75 | **REMOVE** — day of week doesn't matter |
| `patient_id` | 0.00 | 0.01 | 0.95 | **REMOVE** — identifier |
| `hospital_id` | 0.03 | 0.03 | 0.55 | **REMOVE** — not significant with this sample |
| `zipcode` | 0.01 | 0.01 | 0.88 | **REMOVE** — too granular |
| `blood_pressure_sys` | 0.15 | 0.14 | 0.04 | **KEEP** — barely significant, but medically relevant |
| `blood_pressure_dia` | 0.13 | 0.12 | 0.06 | **BORDERLINE** — 0.89 corr with systolic, drop this one |
| `heart_rate` | 0.19 | 0.20 | 0.01 | **KEEP** |

**Outcome:** 18 features down to 9. Removed identifiers, non-significant features, and one of the correlated blood pressure pair. ElasticNet R² = 0.52 with 9 features vs 0.50 with all 18. Fewer features actually improved accuracy (less noise).

---

#### Scenario 10: Predicting Monthly Sales Revenue (11 features)

| Feature | Pearson with Revenue | Spearman | Feature-Feature Issue? | Decision |
|---------|---------------------|----------|----------------------|----------|
| `ad_spend` | 0.73 | 0.70 | — | **KEEP** — strong |
| `num_salespeople` | 0.58 | 0.55 | — | **KEEP** — moderate |
| `avg_product_price` | 0.41 | 0.39 | — | **KEEP** |
| `customer_count` | 0.85 | 0.83 | — | **INVESTIGATE** — could be a leaker (more customers = more revenue by definition) |
| `website_visits` | 0.62 | 0.60 | 0.88 with `ad_spend` | **KEEP** — correlated with ad_spend but < 0.9 |
| `social_media_followers` | 0.35 | 0.33 | — | **KEEP** — weak but useful |
| `competitor_price` | -0.28 | -0.26 | — | **KEEP** — higher competitor price = we gain sales |
| `month_number` | 0.05 | 0.04 | — | **TRANSFORM** — encode as seasonal (Q1-Q4) instead of 1-12, new Pearson = 0.22 |
| `company_id` | 0.00 | 0.00 | — | **REMOVE** — identifier |
| `report_date` | 0.01 | 0.01 | — | **REMOVE** — metadata |
| `currency` | 0.00 | 0.00 | — | **REMOVE** — constant (all USD) |

**Outcome:** 11 features to 7 (plus transformed `month` to `quarter`). Removed `customer_count` as partial leaker — it's available at prediction time but is almost definitionally tied to revenue. Without it, Ridge R² = 0.74. With it, R² = 0.91 (inflated). Chose the honest model.

---

### For Non-Linear Relationships

Methods that detect features with complex, curved, or interaction-based relationships.

#### Method 1: Mutual Information (MI)

Measures how much knowing a feature reduces uncertainty about the target. Works for ANY relationship shape — linear, curved, U-shaped, periodic, whatever.

| MI Score (regression) | Interpretation | Action |
|----------------------|---------------|--------|
| > 0.5 | Very high information | Definitely keep |
| 0.1 – 0.5 | Moderate information | Keep |
| 0.01 – 0.1 | Low information | Keep if other methods agree |
| < 0.01 | Near zero — feature tells nothing about target | Remove |

| MI Score (classification) | Interpretation | Action |
|--------------------------|---------------|--------|
| > 0.3 | Very high information | Definitely keep |
| 0.05 – 0.3 | Moderate information | Keep |
| 0.01 – 0.05 | Low information | Investigate |
| < 0.01 | Useless | Remove |

| What to Watch For | Why |
|------------------|-----|
| Feature with low Pearson but high MI | Non-linear relationship exists — Pearson missed it, MI caught it |
| Feature with high Pearson but low MI | Shouldn't happen — likely a data issue (check for constant regions) |
| MI = 0 exactly | Feature is statistically independent of target — safe to remove |

**Limitations:** Sensitive to how continuous features are binned. Results can vary with different random seeds. Needs enough data (> 200 rows) to be reliable.

---

#### Method 2: Random Forest Feature Importance (Gini / Mean Decrease Impurity)

Train a Random Forest and measure how much each feature reduces impurity across all tree splits. Captures any relationship shape.

| Feature Importance Score | Interpretation | Action |
|-------------------------|---------------|--------|
| > 0.1 (top feature) | Dominant predictor | Definitely keep |
| 0.01 – 0.1 | Contributing predictor | Keep |
| 0.001 – 0.01 | Minor contributor | Keep if dataset is small; drop if many features |
| < 0.001 | Essentially unused by the model | Remove |

| What to Watch For | Why |
|------------------|-----|
| High-cardinality features ranked too high | RF importance is biased toward features with more unique values — a random ID column can appear "important" |
| Correlated features splitting importance | If two features are correlated, importance splits between them — each looks less important than it actually is |
| Importance sums to 1.0 | It's a relative measure — adding more junk features makes good features look lower |

**Limitations:** Biased toward high-cardinality and continuous features. Two correlated features dilute each other's importance.

---

#### Method 3: Permutation Importance

After training any model, randomly shuffle one feature's values and measure how much accuracy drops. If it drops a lot, the feature was important. Works with any model, not just trees.

| Accuracy Drop After Shuffling | Interpretation | Action |
|------------------------------|---------------|--------|
| > 5% drop | Very important feature | Definitely keep |
| 1–5% drop | Important feature | Keep |
| 0.1–1% drop | Mildly useful | Keep if few features; consider dropping if many |
| < 0.1% drop | Unimportant | Remove |
| Accuracy increases after shuffling | Feature was actually hurting the model | Definitely remove |

| What to Watch For | Why |
|------------------|-----|
| Negative importance (accuracy goes UP when shuffled) | Feature is noise — model is better without it |
| High variance across runs | Feature importance is unstable — run multiple times and average |
| Correlated features show low importance | Shuffling one still leaves the other intact, masking true importance |

**Limitations:** Slow (must retrain or re-evaluate for each feature). Correlated features underestimate each other.

---

#### Method 4: Chi-Square Test (for categorical features vs categorical target)

Measures whether a categorical feature and the target are independent. Only works when both feature and target are categorical (or binned).

| Chi-Square Statistic | Interpretation | Action |
|---------------------|---------------|--------|
| Very high (relative to degrees of freedom) | Strong association | Keep |
| Low | Weak or no association | Candidate for removal |

| p-value | Interpretation | Action |
|---------|---------------|--------|
| < 0.01 | Feature and target are definitely not independent | Keep |
| 0.01 – 0.05 | Likely not independent | Keep |
| > 0.05 | Could be independent (no relationship) | Remove |

**Limitations:** Only works for categorical data. Assumes sufficiently large expected frequencies (> 5 per cell).

---

#### Method 5: Variance Threshold

Removes features with very low variance (they barely change across rows). Not strictly about relationship with target — more about whether the feature has any information at all.

| Variance | Interpretation | Action |
|---------|---------------|--------|
| 0 | Constant column — same value every row | Always remove |
| Near 0 (< 0.01 after scaling) | Almost constant — 99% of rows have same value | Usually remove |
| Reasonable variance | Feature varies across rows | Keep (then check relationship with target) |

**Limitations:** A feature with high variance can still be useless (random noise has high variance). Must be combined with other methods.

---

#### Method 6: SHAP Values (Post-Training)

After training a model, SHAP tells you the exact contribution of each feature to each prediction. The mean absolute SHAP value = overall feature importance.

| Mean Absolute SHAP Value | Interpretation | Action |
|--------------------------|---------------|--------|
| Among top 20% of features | Important contributor | Definitely keep |
| Middle 40% | Moderate contributor | Keep unless trying to simplify |
| Bottom 40% | Low contributor | Candidate for removal |
| Near 0 | Feature doesn't affect predictions | Remove |

| What to Watch For | Why |
|------------------|-----|
| SHAP importance differs from RF importance | SHAP is more accurate — it accounts for interactions and correlations |
| Feature has high SHAP for some predictions but 0 for others | Feature matters for a subgroup (keep it!) |

**Limitations:** Computationally expensive. Must train a model first (chicken-and-egg). Different models may give different SHAP rankings.

---

### 10 Real Scenarios: Feature Selection Using Non-Linear Methods

#### Scenario 1: Predicting Customer Churn (20 features, classification)

| Feature | Pearson with Churn | Mutual Information | RF Importance | SHAP Rank | Decision |
|---------|-------------------|-------------------|--------------|-----------|----------|
| `monthly_charges` | 0.19 | 0.08 | 0.11 | #3 | **KEEP** — all methods agree it matters |
| `tenure_months` | -0.35 | 0.15 | 0.18 | #1 | **KEEP** — top predictor across all methods |
| `total_charges` | -0.20 | 0.12 | 0.09 | #4 | **INVESTIGATE** — correlated 0.83 with `tenure * monthly`. RF splits importance. Kept because SHAP shows independent contribution |
| `contract_type` (month/1yr/2yr) | -0.40 | 0.09 | 0.14 | #2 | **KEEP** — Pearson high because ordinal, MI confirms |
| `num_support_tickets` | 0.10 | 0.06 | 0.05 | #6 | **KEEP** — moderate MI, useful non-linear (spike at 5+ tickets) |
| `internet_service_type` | 0.11 | 0.04 | 0.03 | #8 | **KEEP** — fiber optic users churn more (non-linear) |
| `has_online_backup` | -0.08 | 0.02 | 0.02 | #10 | **KEEP** — small but consistent across methods |
| `gender` | 0.01 | 0.00 | 0.001 | #18 | **REMOVE** — all methods agree: zero information |
| `phone_service` | 0.01 | 0.00 | 0.001 | #19 | **REMOVE** — 95% of customers have it (near-constant) |
| `customer_id` | 0.00 | 0.00 | 0.04 | #20 | **REMOVE** — RF importance is FAKE (high-cardinality bias!) |

**Key Insight:** `customer_id` tricked RF importance (ranked it as if useful because every value is unique — trees split easily). MI and SHAP correctly identified it as useless. **Always cross-check RF importance with MI.**

**Outcome:** 20 features to 12. XGBoost F1 improved from 0.79 (all 20) to 0.81 (12 features) because noise features were removed.

---

#### Scenario 2: Predicting Loan Default (25 features)

| Feature | Pearson | MI Score | RF Importance | Permutation Importance | Decision |
|---------|---------|---------|--------------|----------------------|----------|
| `credit_score` | -0.52 | 0.22 | 0.15 | 8.2% accuracy drop | **KEEP** — dominant predictor |
| `debt_to_income_ratio` | 0.38 | 0.14 | 0.10 | 5.1% drop | **KEEP** — strong non-linear (risk jumps above 0.4 ratio) |
| `num_late_payments` | 0.41 | 0.18 | 0.12 | 6.3% drop | **KEEP** — Pearson underestimates this (non-linear: 0 late payments is fine, 1-2 is moderate risk, 3+ is high risk) |
| `loan_amount` | 0.15 | 0.05 | 0.04 | 1.2% drop | **KEEP** — modest but real |
| `employment_years` | -0.22 | 0.08 | 0.06 | 2.8% drop | **KEEP** |
| `annual_income` | -0.18 | 0.06 | 0.05 | 2.1% drop | **KEEP** |
| `num_open_accounts` | 0.08 | 0.02 | 0.02 | 0.5% drop | **BORDERLINE** — small effect |
| `home_ownership` | 0.05 | 0.01 | 0.01 | 0.2% drop | **REMOVE** — all methods agree: negligible |
| `loan_purpose` (15 categories) | 0.03 | 0.01 | 0.03 | 0.3% drop | **REMOVE** — RF importance inflated by cardinality |
| `zipcode` (500 unique) | 0.01 | 0.00 | 0.06 | 0.1% drop | **REMOVE** — RF importance FAKE again (high cardinality). Permutation confirms useless. |
| `application_date` | 0.00 | 0.00 | 0.00 | 0.0% drop | **REMOVE** — identifier |

**Key Insight:** `num_late_payments` had moderate Pearson (0.41) but very high MI (0.18). This gap means the relationship is non-linear — and it is: risk doesn't increase linearly per late payment, it jumps at thresholds.

**Outcome:** 25 features to 7. LightGBM AUC improved from 0.83 to 0.85. Removing noisy features reduced overfitting.

---

#### Scenario 3: Predicting Medical Cost (15 features)

| Feature | Pearson | MI Score | RF Importance | Decision |
|---------|---------|---------|--------------|----------|
| `age` | 0.30 | 0.12 | 0.08 | **KEEP** — MI much higher than Pearson suggests: cost-age relationship is non-linear (sharp increase after 55) |
| `bmi` | 0.20 | 0.10 | 0.06 | **KEEP** — non-linear: costs spike above BMI 30 (obesity threshold) |
| `smoker` (0/1) | 0.78 | 0.35 | 0.45 | **KEEP** — dominant predictor by every metric |
| `smoker * bmi` (interaction) | 0.55 | 0.30 | 0.15 | **KEEP** — engineered feature, highest MI after `smoker`. Obese smokers cost 5x more than either alone |
| `num_children` | 0.07 | 0.01 | 0.01 | **REMOVE** — all methods agree: negligible |
| `region` | 0.02 | 0.00 | 0.01 | **REMOVE** — no effect |
| `exercise_hours` | -0.25 | 0.07 | 0.04 | **KEEP** — moderate MI, non-linear (exercise helps up to a point then plateaus) |
| `blood_pressure` | 0.18 | 0.08 | 0.05 | **KEEP** — MI higher than Pearson (costs spike above 140 systolic — a threshold effect) |
| `gender` | 0.06 | 0.01 | 0.01 | **REMOVE** — minimal effect |
| `income` | -0.03 | 0.00 | 0.01 | **REMOVE** — no relationship to medical costs |

**Key Insight:** `smoker * bmi` (an engineered interaction feature) was the second most informative feature by MI. Neither `smoker` nor `bmi` alone fully captured that obese smokers have dramatically higher costs. **Feature engineering found what statistical methods alone couldn't.**

**Outcome:** 15 features to 6 (including the interaction term). Random Forest RMSE improved by 12% after removing noise.

---

#### Scenario 4: Predicting Website Conversion (30 features, classification)

| Feature | Pearson | MI Score | Permutation Importance | Chi-Square p-value | Decision |
|---------|---------|---------|----------------------|-------------------|----------|
| `time_on_page` | 0.42 | 0.15 | 4.5% drop | — | **KEEP** — strong in all methods |
| `pages_viewed` | 0.38 | 0.12 | 3.8% drop | — | **KEEP** |
| `returned_visitor` (0/1) | 0.25 | 0.08 | 2.1% drop | < 0.001 | **KEEP** |
| `traffic_source` (5 categories) | 0.05 | 0.04 | 1.5% drop | 0.003 | **KEEP** — Pearson was low (categorical!), but Chi-Square and MI caught it |
| `device_type` (3 categories) | 0.08 | 0.03 | 1.0% drop | 0.01 | **KEEP** — mobile converts less |
| `day_of_week` | 0.02 | 0.01 | 0.2% drop | 0.45 | **REMOVE** |
| `browser_type` (8 categories) | 0.01 | 0.00 | 0.1% drop | 0.62 | **REMOVE** |
| `screen_resolution` | 0.03 | 0.01 | 0.1% drop | — | **REMOVE** |
| `operating_system` | 0.02 | 0.01 | 0.2% drop | 0.38 | **REMOVE** |
| `session_id` | 0.00 | 0.00 | 0.0% drop | — | **REMOVE** — identifier |

**Key Insight:** `traffic_source` had near-zero Pearson (0.05) because it's categorical — Pearson can't measure categorical relationships! Chi-Square (p = 0.003) and MI (0.04) correctly identified it as useful. **Always use Chi-Square or MI for categorical features, never Pearson.**

**Outcome:** 30 features to 8. XGBoost F1 stayed at 0.72 with 8 features vs 0.71 with all 30. Removed 22 features with zero accuracy loss.

---

#### Scenario 5: Predicting Equipment Failure (50 sensor features)

| Feature Group | Pearson Range | MI Range | RF Importance Range | Decision |
|--------------|--------------|---------|--------------------|---------|
| Temperature sensors (5 features) | 0.05–0.15 | 0.08–0.20 | 0.02–0.08 | **KEEP 2** — `temp_bearing` and `temp_motor` had highest MI. Others redundant (corr > 0.9 with each other) |
| Vibration sensors (5 features) | 0.10–0.25 | 0.12–0.28 | 0.03–0.10 | **KEEP 2** — `vibration_x` and `vibration_z` (Y was 0.95 correlated with X) |
| Pressure sensors (5 features) | 0.02–0.08 | 0.01–0.04 | 0.01–0.02 | **KEEP 1** — `pressure_main` only. Others near-zero MI |
| Current/voltage (5 features) | 0.08–0.18 | 0.06–0.15 | 0.02–0.06 | **KEEP 2** — `current_motor` and `voltage_input` |
| Humidity sensors (3 features) | 0.01–0.03 | 0.00–0.01 | 0.00–0.01 | **REMOVE ALL** — no relationship to failure |
| Speed sensors (5 features) | 0.15–0.30 | 0.10–0.22 | 0.04–0.09 | **KEEP 2** — `rpm_spindle` and `rpm_main` |
| Timestamps / IDs (7 features) | 0.00 | 0.00 | 0.00–0.02 | **REMOVE ALL** — metadata |
| Operational settings (5 features) | 0.20–0.40 | 0.10–0.18 | 0.05–0.12 | **KEEP 3** — `feed_rate`, `cutting_depth`, `tool_wear` |
| Noise sensors (5 features) | 0.03–0.08 | 0.01–0.03 | 0.01–0.02 | **REMOVE ALL** — ambient noise isn't predictive |
| Cycle counters (5 features) | 0.05–0.12 | 0.04–0.09 | 0.02–0.04 | **KEEP 1** — `total_cycles` only, others derived |

**Key Insight:** 50 sensor features reduced to 13 by: (1) removing entire sensor groups with no MI, (2) within useful groups, keeping only 1-2 least-correlated sensors. **For correlated sensor groups, pick representatives rather than dropping randomly.**

**Outcome:** 50 features to 13. LightGBM F1 went from 0.82 (all 50, overfitting) to 0.88 (13 features). Massive improvement from removing noise.

---

#### Scenario 6: Predicting Taxi Trip Duration (18 features)

| Feature | Pearson | MI Score | RF Importance | Decision |
|---------|---------|---------|--------------|----------|
| `trip_distance` | 0.85 | 0.42 | 0.35 | **KEEP** — dominant predictor |
| `pickup_hour` | 0.05 | 0.08 | 0.06 | **KEEP** — Low Pearson but decent MI! Rush hour trips take longer (non-linear: 8am and 5pm are slow, 2am is fast — cyclical pattern Pearson can't see) |
| `pickup_day_of_week` | 0.01 | 0.03 | 0.02 | **KEEP** — MI found weekend vs weekday difference |
| `pickup_latitude` | 0.02 | 0.06 | 0.04 | **KEEP** — MI detects Manhattan vs outer borough pattern |
| `pickup_longitude` | 0.03 | 0.07 | 0.05 | **KEEP** — same as latitude |
| `dropoff_latitude` | 0.01 | 0.05 | 0.04 | **KEEP** — destination matters |
| `dropoff_longitude` | 0.02 | 0.06 | 0.04 | **KEEP** |
| `passenger_count` | 0.01 | 0.00 | 0.01 | **REMOVE** — number of passengers doesn't affect trip time |
| `payment_type` | 0.00 | 0.00 | 0.00 | **REMOVE** — happens after the trip (leaker if correlated, useless if not) |
| `fare_amount` | 0.82 | 0.40 | 0.30 | **REMOVE** — this IS the target in disguise (fare = rate * time). Leaker! |
| `tip_amount` | 0.15 | 0.05 | 0.03 | **REMOVE** — happens after the trip, not available at prediction time |
| `store_and_fwd_flag` | 0.00 | 0.00 | 0.00 | **REMOVE** — technical flag about data storage |

**Key Insight:** `pickup_hour` had nearly zero Pearson (0.05) but useful MI (0.08). The relationship is cyclical — 8am (slow), 11am (fast), 5pm (slow), 11pm (fast). Pearson sees no *overall* trend, but MI detects the pattern. **Cyclical features always need MI, not Pearson.**

**Outcome:** 18 features to 8. Removed leakers (`fare_amount`, `tip_amount`), post-trip data (`payment_type`), and noise. XGBoost RMSE improved 15% because leakers were artificially inflating metrics.

---

#### Scenario 7: Predicting Customer Lifetime Value (22 features)

| Feature | Pearson | MI Score | SHAP Mean |SHAP Insight | Decision |
|---------|---------|---------|-----------|-------------|----------|
| `total_purchases` | 0.72 | 0.30 | 0.45 | Higher for loyal customers | **KEEP** |
| `avg_order_value` | 0.55 | 0.22 | 0.32 | Top 3 feature | **KEEP** |
| `days_since_first_purchase` | 0.48 | 0.18 | 0.25 | Longer tenure = higher CLV | **KEEP** |
| `purchase_frequency` | 0.60 | 0.25 | 0.38 | Non-linear: weekly buyers 10x more valuable than monthly | **KEEP** |
| `returns_rate` | -0.30 | 0.10 | 0.15 | High returns = low CLV, but effect plateaus above 30% | **KEEP** |
| `email_open_rate` | 0.25 | 0.08 | 0.10 | Engaged customers buy more | **KEEP** |
| `category_diversity` | 0.20 | 0.07 | 0.08 | Buying from more categories = higher CLV | **KEEP** |
| `discount_usage_rate` | 0.05 | 0.04 | 0.03 | SHAP shows it HURTS CLV for discount-only buyers but helps for occasional discount users | **KEEP** — non-linear! |
| `signup_source` (5 cats) | 0.03 | 0.02 | 0.02 | Organic search customers have 2x CLV vs paid ads | **KEEP** |
| `customer_service_calls` | 0.08 | 0.03 | 0.04 | Non-linear: 1-2 calls = engaged, 5+ calls = unhappy | **KEEP** |
| `social_media_follower` (0/1) | 0.02 | 0.00 | 0.01 | Negligible effect | **REMOVE** |
| `referral_code_used` (0/1) | 0.04 | 0.01 | 0.01 | Tiny effect | **REMOVE** |
| `account_id` | 0.00 | 0.00 | 0.00 | Identifier | **REMOVE** |

**Key Insight:** `discount_usage_rate` had near-zero Pearson (0.05) and low MI (0.04), but SHAP revealed it has a *conditional* non-linear effect: occasional discount users have higher CLV, while discount-only buyers have lower CLV. **SHAP catches feature effects that depend on other features (interactions).**

**Outcome:** 22 features to 10. Gradient Boosting R² = 0.78 with 10 features vs 0.76 with all 22.

---

#### Scenario 8: Predicting Flight Delay (35 features)

| Feature | Pearson | MI Score | RF Importance | Permutation Imp | Decision |
|---------|---------|---------|--------------|----------------|----------|
| `departure_hour` | 0.08 | 0.06 | 0.04 | 1.8% drop | **KEEP** — delays accumulate through the day (non-linear: 6am rarely late, 10pm often late) |
| `airline` (12 categories) | 0.02 | 0.03 | 0.05 | 1.5% drop | **KEEP** — some airlines consistently more delayed. Pearson can't measure categorical! |
| `origin_airport` (50 categories) | 0.01 | 0.04 | 0.08 | 2.1% drop | **KEEP** — congested airports (JFK, ORD, ATL) cause delays. RF importance inflated by cardinality but Permutation confirms real effect |
| `weather_severity` (1-5) | 0.35 | 0.15 | 0.12 | 5.5% drop | **KEEP** — strongest predictor after distance |
| `wind_speed` | 0.22 | 0.10 | 0.07 | 3.2% drop | **KEEP** |
| `visibility_miles` | -0.19 | 0.08 | 0.05 | 2.5% drop | **KEEP** |
| `temperature_f` | 0.05 | 0.03 | 0.02 | 0.8% drop | **KEEP** — MI found non-linear effect (extreme cold and extreme heat both cause delays) |
| `scheduled_departure_time` | 0.10 | 0.05 | 0.03 | 1.2% drop | **REMOVE** — redundant with `departure_hour` (0.97 correlation) |
| `tail_number` (500 planes) | 0.00 | 0.00 | 0.07 | 0.1% drop | **REMOVE** — RF importance is FAKE (high cardinality). Permutation proves useless |
| `flight_number` | 0.00 | 0.00 | 0.05 | 0.0% drop | **REMOVE** — same high-cardinality trap |
| `ticket_price` | 0.03 | 0.01 | 0.01 | 0.2% drop | **REMOVE** — price doesn't cause delays |

**Key Insight:** `temperature_f` had low Pearson (0.05) because the effect is U-shaped (very cold = de-icing delays, very hot = equipment issues). MI (0.03) caught this non-linear pattern. Also, `tail_number` and `flight_number` fooled RF importance with high-cardinality but Permutation Importance correctly scored them near 0%. **Always validate RF importance with Permutation Importance for high-cardinality features.**

**Outcome:** 35 features to 10. LightGBM MAE improved from 18.2 min (all features) to 16.5 min (10 features).

---

#### Scenario 9: Predicting Employee Attrition (28 features, classification)

| Feature | Pearson | MI Score | RF Importance | Chi-Square | Decision |
|---------|---------|---------|--------------|-----------|----------|
| `job_satisfaction` (1-4) | -0.16 | 0.04 | 0.05 | p < 0.001 | **KEEP** — ordinal, all methods agree |
| `monthly_income` | -0.16 | 0.05 | 0.06 | — | **KEEP** |
| `overtime` (0/1) | 0.25 | 0.06 | 0.07 | p < 0.001 | **KEEP** — strongest binary predictor |
| `years_at_company` | -0.17 | 0.06 | 0.08 | — | **KEEP** |
| `age` | -0.16 | 0.04 | 0.05 | — | **KEEP** but correlated 0.69 with `years_at_company` — both kept since < 0.9 |
| `work_life_balance` (1-4) | -0.06 | 0.01 | 0.02 | p = 0.04 | **KEEP** — Chi-Square significant, all methods agree it has small effect |
| `distance_from_home` | 0.08 | 0.02 | 0.03 | — | **KEEP** — modest MI |
| `num_companies_worked` | 0.04 | 0.02 | 0.02 | — | **BORDERLINE** — kept due to domain knowledge (job hoppers leave) |
| `education` (1-5) | -0.03 | 0.00 | 0.01 | p = 0.42 | **REMOVE** — education level doesn't predict attrition here |
| `gender` | 0.01 | 0.00 | 0.00 | p = 0.78 | **REMOVE** — no effect |
| `marital_status` (3 categories) | 0.06 | 0.01 | 0.01 | p = 0.08 | **REMOVE** — borderline Chi-Square, very low MI |
| `employee_number` | 0.01 | 0.00 | 0.02 | — | **REMOVE** — identifier (RF importance is fake) |
| `standard_hours` | 0.00 | 0.00 | 0.00 | — | **REMOVE** — constant (everyone has 80 hours) |
| `over_18` (Y/N) | 0.00 | 0.00 | 0.00 | — | **REMOVE** — constant (everyone is over 18) |

**Key Insight:** `standard_hours` and `over_18` were identified as constants at Stage 1 but survived into the dataset. MI = 0.00 for both confirms they should be removed. **Variance Threshold would catch these automatically.** Also, `num_companies_worked` had low metrics across the board but was kept because domain knowledge says job-hopping predicts attrition — sometimes domain expertise overrides statistical tests.

**Outcome:** 28 features to 8. Random Forest F1 went from 0.38 (all 28, overfitting badly) to 0.52 (8 features). Removing noise features had a massive impact.

---

#### Scenario 10: Predicting Energy Consumption (40 features, IoT building data)

| Feature | Pearson | MI Score | RF Importance | Permutation Imp | Feature-Feature Corr | Decision |
|---------|---------|---------|--------------|----------------|---------------------|----------|
| `outdoor_temperature` | 0.15 | 0.25 | 0.12 | 7.2% drop | — | **KEEP** — huge MI vs low Pearson! Relationship is U-shaped (heating in winter, cooling in summer) |
| `hour_of_day` | 0.02 | 0.18 | 0.09 | 5.5% drop | — | **KEEP** — MI found cyclical pattern (peak at 9am & 6pm, low at 3am). Pearson sees nothing |
| `day_of_week` | 0.01 | 0.08 | 0.04 | 2.8% drop | — | **KEEP** — weekday vs weekend pattern |
| `occupancy_count` | 0.55 | 0.20 | 0.15 | 6.8% drop | — | **KEEP** — strong in all methods |
| `humidity_indoor` | 0.12 | 0.05 | 0.03 | 1.5% drop | 0.72 with `outdoor_humidity` | **KEEP** — indoor more relevant than outdoor |
| `humidity_outdoor` | 0.08 | 0.03 | 0.02 | 0.8% drop | 0.72 with `indoor_humidity` | **REMOVE** — less predictive of the correlated pair |
| `solar_radiation` | 0.10 | 0.08 | 0.05 | 2.2% drop | — | **KEEP** — MI found non-linear (strong effect on sunny days, none on cloudy) |
| `hvac_setpoint` | 0.20 | 0.10 | 0.08 | 3.5% drop | — | **KEEP** — directly controls energy usage |
| `window_state` (open/closed) | 0.05 | 0.03 | 0.02 | 1.0% drop | — | **KEEP** — binary but meaningful |
| `building_id` | 0.00 | 0.00 | 0.06 | 0.0% drop | — | **REMOVE** — RF fooled by cardinality again |
| `sensor_id` (50 unique) | 0.00 | 0.00 | 0.04 | 0.0% drop | — | **REMOVE** — identifier |
| `floor_number` | 0.03 | 0.02 | 0.01 | 0.5% drop | — | **BORDERLINE** — top floors may use more AC |
| `room_area_sqft` | 0.30 | 0.12 | 0.08 | 3.2% drop | — | **KEEP** — bigger rooms use more energy |
| Redundant sensor readings (15 features) | 0.01–0.05 | 0.00–0.02 | 0.00–0.01 | 0.0–0.3% drop | Corr > 0.9 with kept sensors | **REMOVE ALL** — redundant with already-kept sensors |
| Maintenance logs (5 features) | 0.00–0.02 | 0.00–0.01 | 0.00–0.01 | 0.0–0.1% drop | — | **REMOVE ALL** — no predictive value |

**Key Insight:** `outdoor_temperature` had massive MI (0.25) but low Pearson (0.15). This is the textbook U-shaped relationship: cold weather needs heating, hot weather needs cooling, mild weather needs neither. If you only used Pearson, you'd think temperature barely matters. MI revealed it's the single most important feature. **This is why non-linear methods are mandatory — linear methods can completely miss the #1 feature.**

**Outcome:** 40 features to 10. XGBoost RMSE dropped from 145 kWh (all 40, overfitting) to 98 kWh (10 features). Removing 30 redundant/noisy sensor readings made a 32% improvement.

---

## Stage 3: Automated Selection Methods

### All Known Methods

| Method | What It Does | When to Use | When NOT to Use | Models It Works With |
|--------|-------------|------------|-----------------|---------------------|
| **SelectKBest** | Scores all features with a statistical test, keeps the top K | Quick first pass; when you have a target number of features in mind | When you don't know how many features to keep | Any — it's a preprocessing step |
| **RFE (Recursive Feature Elimination)** | Trains a model, removes weakest feature, repeats until K remain | When you want features that work well together (not just individually) | Very slow on large datasets or feature counts > 100 | Any model with `coef_` or `feature_importances_` (Ridge, Lasso, RF, SVM) |
| **RFECV** | Same as RFE but uses cross-validation to find the optimal K automatically | When you don't know how many features to keep | Very slow — tries many values of K | Same as RFE |
| **Lasso (L1 Regularization)** | Sets useless feature coefficients to exactly zero | When you suspect many features are irrelevant; feature selection + training in one step | When relationships are non-linear (Lasso is a linear model) | Lasso, LogisticRegression with `penalty='l1'` |
| **ElasticNet** | Like Lasso but also handles groups of correlated features | When features are correlated AND you want selection | Same as Lasso (linear only) | ElasticNet, LogisticRegression with `penalty='elasticnet'` |
| **Tree-Based Selection** | Train a tree model, drop features with importance below a threshold | Fast; works for non-linear relationships | Biased toward high-cardinality features | RandomForest, XGBoost, LightGBM |
| **Variance Threshold** | Drops features with variance below a cutoff | First-pass cleanup to remove constants/near-constants | Alone — a noisy feature can have high variance but be useless | Any — it's a preprocessing step |
| **Sequential Forward Selection (SFS)** | Starts with 0 features, adds the one that improves accuracy most, repeats | Small feature sets (< 30) where you want the truly optimal subset | Slow on many features; greedy (may miss optimal combinations) | Any model |
| **Sequential Backward Selection (SBS)** | Starts with all features, removes the one that hurts accuracy least, repeats | Same as SFS but starting from the other direction | Same limitations as SFS | Any model |
| **Boruta** | Compares each feature's importance against "shadow features" (randomized copies). Keeps only features that beat random. | When you want a statistically rigorous cutoff between useful and useless | Slow; needs Random Forest | Random Forest based |
| **PCA (as feature reduction)** | Combines correlated features into uncorrelated components | Many correlated features; you want dimensionality reduction, not selection | When you need to interpret individual features (PCA components aren't original features) | Any model after PCA transform |

### Detailed Method Comparison

| Method | Speed | Accuracy of Selection | Handles Non-Linear? | Handles Correlated Features? | Gives You Original Feature Names? |
|--------|-------|----------------------|---------------------|-----------------------------|---------------------------------|
| SelectKBest | Very fast | Moderate — checks features individually, ignores interactions | Only with `mutual_info_*` scoring | No — may keep redundant correlated features | Yes |
| RFE | Slow | High — considers feature combinations | Depends on base estimator (RF = yes, Ridge = no) | Somewhat — drops one of a correlated pair | Yes |
| RFECV | Very slow | Highest — auto-selects optimal count | Depends on base estimator | Somewhat | Yes |
| Lasso | Fast | Good for linear problems | No | Randomly picks one from correlated group | Yes (non-zero coefficients) |
| ElasticNet | Fast | Good for linear problems with correlations | No | Yes — keeps correlated features as a group | Yes |
| Tree-Based | Fast | Good for non-linear | Yes | Splits importance between correlated features | Yes |
| Variance Threshold | Very fast | Poor on its own | N/A — doesn't check target | N/A | Yes |
| SFS / SBS | Very slow | Very high | Depends on base estimator | Yes — evaluates combinations | Yes |
| Boruta | Slow | Very high — statistically rigorous | Yes | Somewhat | Yes |
| PCA | Fast | Good for redundancy removal | Partially (linear combinations only) | Excellent — designed for this | No — gives you components, not original names |

### When to Use Each — Decision Table

| Your Situation | Best Method | Why |
|---------------|-------------|-----|
| Quick first pass, reduce from 100 to 20 features | SelectKBest with MI | Fast, catches both linear and non-linear |
| Want the best subset, have time to wait | RFECV with Random Forest | Finds optimal set and optimal count |
| Linear model is your final model (Ridge, Lasso) | Lasso or ElasticNet | Selection and training in one step |
| Many correlated features (sensor data, financial) | ElasticNet or PCA | Handle multicollinearity properly |
| Non-linear relationships dominate | Tree-Based Importance then drop low-importance features | Fast, captures complex patterns |
| Need statistical rigor (academic, regulatory) | Boruta | Formal statistical test, not just a heuristic |
| Small feature set (< 30), want truly optimal | SFS or SBS | Evaluates actual combinations, not just individual features |
| First cleanup before other methods | Variance Threshold | Removes obvious junk in milliseconds |
| Need interpretable feature selection | Lasso | You can show: "these features have non-zero coefficients, the rest are irrelevant" |
| Need to explain why features were dropped | SHAP + threshold | "Feature contributed < 0.01 to predictions on average" |

---

## Recommended Workflow

```
GET A NEW DATASET
│
│  STAGE 1: COMMON SENSE (5 minutes)
│  ─────────────────────────────────
├── Drop IDs, names, emails, keys, constants
├── Drop columns from the future (data leakage)
├── Drop duplicates of other columns
├── Transform dates -> extract useful components (age, tenure, day_of_week)
├── Decide: keep or encode high-cardinality categoricals
│
│  STAGE 2: STATISTICAL ANALYSIS (15 minutes)
│  ──────────────────────────────────────────
├── Pearson correlation -> identify strong linear features
├── Spearman correlation -> catch monotonic non-linear features
├── Mutual Information -> catch any-shape relationships
├── Feature-to-feature correlation matrix -> find pairs > 0.9 and drop one
├── Quick Random Forest -> get feature importance ranking
├── Flag features that ALL methods rank as useless -> safe to remove
├── Flag features that SOME methods rank high -> keep them
│
│  STAGE 3: AUTOMATED SELECTION (10 minutes)
│  ─────────────────────────────────────────
├── Run Variance Threshold -> remove near-constants that slipped through
├── Run SelectKBest (MI) -> get a ranked list of top K features
├── Run Lasso/ElasticNet -> see which features get zeroed out
├── Compare results from SelectKBest vs Lasso vs RF importance
├── Features that ALL methods agree are important -> definitely keep
├── Features that ALL methods agree are useless -> definitely remove
├── Features that methods disagree on -> keep (better safe than sorry)
│
│  VALIDATION (5 minutes)
│  ─────────────────────
├── Train your model with ALL features -> record accuracy
├── Train your model with SELECTED features -> record accuracy
├── If accuracy is within 1-2% -> use fewer features (simpler model wins)
├── If accuracy drops significantly -> add back some removed features
│
└── DONE — You have your final feature set
```

### How Many Features to Keep (Rule of Thumb)

| Total Features | Rule of Thumb |
|---------------|---------------|
| < 10 | Keep all (after Stage 1 cleanup) |
| 10–30 | Drop obvious junk, usually keep 70–80% |
| 30–100 | Use SelectKBest or RFE to reduce to 15–30 |
| 100+ | Lasso or PCA first, then keep top 10–30 |

**Goal:** Use the fewest features that give you 95%+ of the accuracy of using all features. Fewer features = simpler model, faster training, less overfitting, easier to explain.

---

## How to Know When Your Features Are Not Enough

Even after careful feature selection, sometimes the problem isn't *which* features to keep — it's that you don't have *enough* information to solve the problem. Here's how to diagnose this and what to do about it.

### The 7 Diagnostic Signs

#### Sign 1: Poor Performance Across ALL Models

If you've tried Linear, Tree-Based, SVM, KNN, and ensemble methods, and they ALL perform poorly — the problem likely isn't the model. It's the features.

| Situation | Likely Cause |
|-----------|-------------|
| All models give ~60% accuracy on a binary task | Features don't carry enough signal (60% is barely above random 50%) |
| All regression models give R² < 0.3 | Features explain less than 30% of variance — missing key drivers |
| Every model overfits (train=95%, test=55%) | Features are noisy, models memorize noise instead of learning patterns |

**The test:** If your best model after tuning is within 5% of a dummy baseline (predicting mean/mode), your features are almost certainly insufficient.

#### Sign 2: Underfitting (High Bias)

| Metric | Healthy | Underfitting (feature problem) |
|--------|---------|-------------------------------|
| Training accuracy | 90%+ | 65% |
| Test accuracy | 85%+ | 63% |
| Gap | ~5% | ~2% (both are bad) |

When train AND test performance are both low and close together, the model can't even learn the training data. Adding more data won't help. Adding complexity won't help. You need better/more features.

#### Sign 3: Residual Patterns

After fitting a regression model, plot residuals (actual - predicted) vs predicted values:

- **Random scatter** = good, your features are sufficient
- **Curved pattern** = missing a non-linear feature or interaction
- **Clusters of errors** = missing a categorical feature that splits data into groups
- **Errors increase with magnitude** = missing a scaling/ratio feature

If you see ANY pattern in your residuals, there's information your features aren't capturing.

#### Sign 4: Feature Importance Is "Thin"

After training a Random Forest or XGBoost, check feature importances:

| Distribution | What It Means |
|-------------|---------------|
| Top feature = 60%, rest < 5% each | You basically have one useful feature. Need more signal. |
| Top 3 features = 80%, rest = noise | Okay for simple problems, but complex problems need richer features |
| Importances spread across 10+ features | Healthy — multiple features contribute |
| All features roughly equal (~5% each) | No single strong signal — could mean features are all weak |

**Red flag:** If your best feature has importance < 10%, and the best model R² is < 0.5, the features collectively don't carry enough predictive power.

#### Sign 5: Domain Knowledge Says You're Missing Something

Ask yourself: "If a human expert had to make this prediction, what would they look at?"

| Prediction Task | What an Expert Would Use | Common Missing Features |
|----------------|--------------------------|------------------------|
| House price | Location, size, condition, market trends | Neighborhood crime rate, school district rating, recent nearby sales |
| Customer churn | Usage patterns, complaints, contract terms | Customer sentiment (from support calls/texts), competitor offers |
| Loan default | Income, debt, credit score | Employment stability, spending patterns, recent credit inquiries |
| Disease diagnosis | Symptoms, test results, vitals | Family history, lifestyle factors, medication interactions |

If your features don't cover what an expert would use, you're missing information.

#### Sign 6: The Random Feature Test

Add a column of pure random noise to your dataset. Train your model.

| Result | Interpretation |
|--------|---------------|
| Random feature gets 0% importance | Good — your real features are stronger than noise |
| Random feature gets 5%+ importance | Bad — your real features are so weak that noise competes with them |
| Random feature makes accuracy go UP | Very bad — model is overfitting, features are essentially noise-level |

This is a quick sanity check. If a random column is as "useful" as your real features, you need fundamentally better features.

#### Sign 7: Learning Curve Plateaus Very Low

Plot accuracy vs training set size:

```
Accuracy
  |
  |     ___________________________  <-- Plateau at 65% (BAD - feature ceiling)
  |    /
  |   /
  |  /
  | /
  |/_________________________________
           Training Set Size
```

- **Plateau at high accuracy (90%+):** Features are sufficient, you have enough data
- **Plateau at medium accuracy (70-80%):** Features might be enough, try engineering more
- **Plateau at low accuracy (< 70%):** Features are almost certainly insufficient. More data won't help because you've already plateaued. You need new/better features.

### What To Do When Features Aren't Enough

| Strategy | How | Example |
|----------|-----|---------|
| **Engineer from existing** | Create ratios, differences, interactions | `price_per_sqft = price / sqft`, `age_at_purchase = purchase_date - birth_date` |
| **Aggregate features** | Summarize historical data into statistics | `avg_purchase_last_6months`, `max_late_payment_days`, `trend_slope` |
| **Domain transforms** | Apply domain-specific math | `BMI = weight / height²`, `debt_to_income = debt / income` |
| **Extract from text** | Pull structured features from text fields | Sentiment score, keyword counts, text length, TF-IDF features |
| **Extract from dates** | Pull components from timestamps | Day of week, month, is_weekend, days_since_last_event, holiday_flag |
| **Add external data** | Join with outside datasets | Weather data, census data, economic indicators, competitor pricing |
| **Polynomial features** | Capture non-linear relationships | `sqft²`, `age × income`, `feature1 × feature2` (caution: increases dimensionality) |
| **Binning** | Convert continuous to categorical | Age groups, income brackets, time-of-day buckets |

### Quick Diagnostic Checklist

Before concluding your features aren't enough, run through this:

```
[ ] 1. Tried at least 3 different model types? (linear, tree, SVM/KNN)
[ ] 2. Training accuracy is also low? (not just test — rules out overfitting)
[ ] 3. Residual plot shows patterns? (not random scatter)
[ ] 4. Feature importances are "thin"? (one feature dominates or all are weak)
[ ] 5. Random noise feature competes with real features?
[ ] 6. Learning curve plateaus below acceptable accuracy?
[ ] 7. Domain expert would say "but what about X?"
```

**If 3+ boxes are checked:** Your features are likely insufficient. Focus on feature engineering and external data before any more model tuning.

**If 1-2 boxes are checked:** Try feature engineering first, but the issue might also be model selection or hyperparameter tuning.

**If 0 boxes are checked:** Your features are probably fine — focus on model tuning and ensembling instead.
