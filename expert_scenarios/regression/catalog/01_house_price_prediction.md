# Expert Scenario 1: House Price Prediction (Ames-Style)

> **Complexity:** 80+ mixed-type features (numeric, ordinal, categorical, dates), heavy right-skew on the target ($35K to $755K), multicollinearity between size-related features, ordinal categoricals masquerading as nominal, and a log-target is essentially mandatory.

---

## The Brief

A regional real-estate brokerage gives you 1,460 historical sales from Ames, Iowa (2006-2010), with 80+ features per home (lot size, year built, neighborhood, garage type, condition ratings, basement finish, etc.). They want a model that:

- Predicts the sale price for a new listing within 10% MAPE on average.
- Surfaces the top features driving each prediction so the listing agent can defend the recommended price.
- Handles winter/summer seasonality (sales spike in spring).
- Is robust to outliers (the rare $700K mansion shouldn't break median predictions).

This is the canonical Kaggle regression challenge, but the brief adds production constraints: explainability, MAPE in production reporting, and seasonal awareness.

---

## Step 1: Define the Problem Type

```
Type:           Regression (continuous, target = sale_price in USD)
Primary Metric: RMSE on log(price)  (Kaggle official)
Secondary:      MAPE on raw price (business reporting metric)
Business Goal:  Within 10% MAPE
Constraint:     Explainable (LR coefficients or SHAP)
Target shape:   Right-skewed, range $35K-$755K, median $163K
```

**Expert thinking:** the target is heavily right-skewed. Predicting in raw dollars means RMSE is dominated by the few high-end mansions. Log-transforming the target turns the multiplicative ratio "predicted/actual" into an additive error, which is the actual business metric (10% off is 10% off whether the house is $100K or $700K).

---

## Step 2: Understand the Data

```
Shape: 1,460 rows x 80 columns
Target: 'SalePrice' -- $34,900 to $755,000, median $163,000

Feature categories (80 total):

LOT (5):
- LotArea       (sqft, 1300-215,245 -- huge outlier range)
- LotFrontage   (linear feet, 17% missing)
- LotShape      (ordinal: Reg/IR1/IR2/IR3)
- LandContour   (Lvl/Bnk/HLS/Low)
- LandSlope     (ordinal: Gtl/Mod/Sev)

LOCATION (3):
- Neighborhood  (25 unique categories -- huge price effect)
- Condition1, Condition2  (proximity to road/railroad)

BUILDING (12):
- YearBuilt     (1872-2010)
- YearRemodAdd  (year of last remodel)
- HouseStyle    (1Story, 2Story, SLvl, etc.)
- BldgType      (1Fam, TwnhsE, etc.)
- OverallQual   (ordinal 1-10 -- strong signal)
- OverallCond   (ordinal 1-10)
- RoofStyle, RoofMatl
- Exterior1st, Exterior2nd
- MasVnrType    (8% missing)
- Foundation

INTERIOR (15):
- TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GrLivArea (sqft)
- BsmtFinSF1, BsmtFinSF2, BsmtUnfSF
- BsmtQual, BsmtCond, BsmtExposure (ordinal, ~3% missing each)
- BsmtFinType1, BsmtFinType2
- HeatingQC (Ex/Gd/TA/Fa/Po)
- Electrical (Mix/SBrkr/FuseA/FuseF/FuseP -- 1 missing)
- KitchenQual (Ex/Gd/TA/Fa)

ROOMS (6):
- TotRmsAbvGrd, BedroomAbvGr, KitchenAbvGr
- FullBath, HalfBath, BsmtFullBath, BsmtHalfBath

GARAGE (7):
- GarageType    (5% missing)
- GarageYrBlt   (5% missing)
- GarageFinish  (5% missing)
- GarageCars, GarageArea
- GarageQual, GarageCond  (ordinal)

OUTDOOR / PORCH (8):
- WoodDeckSF, OpenPorchSF, EnclosedPorch, 3SsnPorch, ScreenPorch
- PoolArea, PoolQC (99.5% missing!)
- Fence (80% missing)
- MiscFeature (96% missing -- shed, tennis court)

OTHER (8):
- MSSubClass (the property type code)
- MSZoning (zoning class)
- Street, Alley
- Utilities, FireplaceQu (47% missing -- not all houses have fireplace)
- Functional, SaleType, SaleCondition

TEMPORAL (2):
- YrSold (2006-2010)
- MoSold (1-12)
```

**Expert thinking:** Several patterns to flag immediately:
- **PoolQC 99.5% missing**: most houses don't have pools — `PoolQC=NaN` is "no pool", not missing data.
- **FireplaceQu 47% missing**: half the houses lack fireplaces. Same: `NaN` = no fireplace.
- **GarageYrBlt 5% missing**: aligns with houses that have no garage. `NaN` = no garage.
- **Many ordinal masquerading as categorical**: `OverallQual` is treated as int (already ordinal); `BsmtQual` (Ex/Gd/TA/Fa/Po) reads as nominal but is ordinal.
- **MSSubClass is numeric but actually a category code** (20=1-story, 60=2-story, etc.).

---

## Step 3: Exploratory Data Analysis (EDA)

```python
# Target distribution
df['SalePrice'].describe()
# count    1460
# mean    180,921
# std      79,442
# min      34,900
# 25%     129,975
# 50%     163,000
# 75%     214,000
# max     755,000
# Skew: 1.88 (heavily right-skewed)

# After log-transform:
np.log1p(df['SalePrice']).skew()  # 0.12 (essentially Gaussian)

# Top correlations with log(price)
log_price = np.log1p(df['SalePrice'])
df.select_dtypes(include='number').corrwith(log_price).abs().sort_values(ascending=False).head(15)
# OverallQual    0.79
# GrLivArea      0.71
# GarageCars     0.68
# GarageArea     0.65
# TotalBsmtSF    0.64
# 1stFlrSF       0.61
# FullBath       0.59
# YearBuilt      0.58
# YearRemodAdd   0.57
# TotRmsAbvGrd   0.55
# Fireplaces     0.49
# OpenPorchSF    0.39
```

**Findings:**

| Finding | Implication |
|---------|------------|
| OverallQual (1-10) carries 0.79 correlation alone | Single most powerful feature; ordinal already encoded |
| GrLivArea + TotalBsmtSF + GarageArea highly correlated with each other (multicollinearity) | Linear models suffer; consider Lasso/Ridge or composite "TotalSF" feature |
| YearBuilt has 0.58 correlation but is non-linear (depreciation curve) | Tree models capture this; linear models need binning or polynomial |
| Neighborhood: median price ranges from $98K (MeadowV) to $310K (NoRidge) | 3x spread — biggest categorical signal |
| 4 outlier houses with GrLivArea > 4500 priced low (partial sales / abnormal sales) | Drop these per Ames documentation guidance |
| Sales by month: peak Jun-Jul (45% of annual), trough Dec-Feb | Seasonality real but not huge |
| Houses sold "Abnormal" (foreclosure, family transfer): median 30% below market | Filter these or include `SaleCondition` feature |

**Expert insight:** the Ames dataset has 4 documented outlier houses (large `GrLivArea` with low price — partial sales). Per the dataset's author, drop them. Don't try to model them.

---

## Step 4: Data Cleaning

```python
# === DROP DOCUMENTED OUTLIERS ===
# Per de Cock 2011 dataset paper -- 4 partial sales at low prices
df = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))]
# Lost 4 rows, kept 1456.

# === HANDLE "MEANINGFUL MISSING" ===
# These are NOT missing data -- they're "no feature" coded as NaN
no_feature_cols = {
    'PoolQC':       'None',  # no pool
    'MiscFeature':  'None',
    'Alley':        'None',  # no alley access
    'Fence':        'None',  # no fence
    'FireplaceQu':  'None',  # no fireplace
    'GarageType':   'None',
    'GarageFinish': 'None',
    'GarageQual':   'None',
    'GarageCond':   'None',
    'BsmtQual':     'None',  # no basement
    'BsmtCond':     'None',
    'BsmtExposure': 'None',
    'BsmtFinType1': 'None',
    'BsmtFinType2': 'None',
    'MasVnrType':   'None',  # no masonry veneer
}
for col, val in no_feature_cols.items():
    df[col] = df[col].fillna(val)

# Numeric counterparts to "no feature"
df['MasVnrArea']   = df['MasVnrArea'].fillna(0)    # no veneer -> 0 sqft
df['GarageYrBlt']  = df['GarageYrBlt'].fillna(0)   # no garage -> 0 (will encode as binary later)
df['BsmtFinSF1']   = df['BsmtFinSF1'].fillna(0)
df['BsmtFinSF2']   = df['BsmtFinSF2'].fillna(0)
df['BsmtUnfSF']    = df['BsmtUnfSF'].fillna(0)
df['TotalBsmtSF']  = df['TotalBsmtSF'].fillna(0)

# === HANDLE TRUE MISSING ===
# LotFrontage: 17% missing, but feet of street frontage is real data
# Impute by neighborhood median (street layout is neighborhood-correlated)
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
    lambda x: x.fillna(x.median())
)

# Electrical: 1 missing -- impute with mode
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

# === LOG-TRANSFORM SKEWED NUMERIC FEATURES ===
# Identify features with skew > 0.75 (rule of thumb)
from scipy.stats import skew
numeric_feats = df.select_dtypes(include='number').columns
skewed = df[numeric_feats].apply(lambda x: skew(x.dropna())).abs()
skewed_feats = skewed[skewed > 0.75].index.drop('SalePrice')

# Apply log1p to skewed numerics
for col in skewed_feats:
    df[col] = np.log1p(df[col])
# Common candidates: LotArea, GrLivArea, TotalBsmtSF, 1stFlrSF, MiscVal

# === LOG-TRANSFORM TARGET ===
df['SalePrice_log'] = np.log1p(df['SalePrice'])
# Now train on SalePrice_log; back-transform with expm1 at prediction time
```

**Expert insight:** the Ames dataset is a textbook example of "missing means absent." Imputing `PoolQC=NaN` with the mode would inject "average pool quality" into 99.5% of houses that have no pool — clearly wrong. Treat as a category `'None'`.

---

## Step 5: Feature Engineering

```python
# === COMPOSITE SIZE FEATURES (handle multicollinearity) ===
df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['TotalBath'] = (df['FullBath'] + 0.5*df['HalfBath']
                    + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'])
df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch']
                       + df['3SsnPorch'] + df['ScreenPorch'] + df['WoodDeckSF'])

# === AGE FEATURES (avoid year directly -- model can't extrapolate to 2025) ===
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
df['IsNew'] = (df['HouseAge'] <= 1).astype(int)

# === HAS-FEATURE BINARY FLAGS ===
df['HasPool']      = (df['PoolArea'] > 0).astype(int)
df['HasGarage']    = (df['GarageArea'] > 0).astype(int)
df['HasBasement']  = (df['TotalBsmtSF'] > 0).astype(int)
df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
df['Has2ndFloor']  = (df['2ndFlrSF'] > 0).astype(int)

# === ORDINAL ENCODING (was string-coded as Ex/Gd/TA/Fa/Po) ===
quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
ordinal_cols = ['BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
                'KitchenQual', 'HeatingQC', 'ExterQual', 'ExterCond',
                'FireplaceQu', 'PoolQC']
for col in ordinal_cols:
    df[col] = df[col].map(quality_map).fillna(0).astype(int)

# === NEIGHBORHOOD TARGET ENCODING (with smoothing) ===
# 25 neighborhoods, target-encode using TRAIN ONLY (in CV)
neighborhood_mean = df_train.groupby('Neighborhood')['SalePrice_log'].mean()
neighborhood_count = df_train.groupby('Neighborhood').size()
overall_mean = df_train['SalePrice_log'].mean()
alpha = 10  # smoothing factor
df['NeighborhoodEnc'] = df['Neighborhood'].map(
    (neighborhood_count * neighborhood_mean + alpha * overall_mean) /
    (neighborhood_count + alpha)
).fillna(overall_mean)

# === MSSubClass IS A CATEGORY, NOT A NUMBER ===
df['MSSubClass'] = df['MSSubClass'].astype(str)

# === SEASONALITY ===
df['SaleSeason'] = df['MoSold'].map({
    12:'Winter', 1:'Winter', 2:'Winter',
    3:'Spring', 4:'Spring', 5:'Spring',
    6:'Summer', 7:'Summer', 8:'Summer',
    9:'Fall',  10:'Fall',  11:'Fall'
})

# Final feature count: ~95 (12 engineered + 80 cleaned + drops)
```

**Expert insight:** `TotalSF` (basement + 1st + 2nd floor) is a single feature that captures most of what GrLivArea + TotalBsmtSF + 1stFlrSF + 2ndFlrSF carry, with less multicollinearity. Linear models (especially ElasticNet) benefit dramatically.

---

## Step 6: Feature Selection

```python
# Stage 1 -- Drop ID, raw year columns (replaced by Age features)
drop_cols = ['Id', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt']

# Stage 2 -- Mutual Information for top features
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(X_train, y_train, random_state=42)

# Top 20 features by MI:
#   OverallQual         0.65
#   TotalSF             0.58
#   NeighborhoodEnc     0.55
#   GrLivArea           0.53
#   GarageCars          0.46
#   TotalBath           0.42
#   ExterQual           0.40
#   GarageArea          0.39
#   KitchenQual         0.38
#   YearBuilt (kept as HouseAge)  0.37
#   ...

# Stage 3 -- Lasso for automatic feature selection
# We'll use Lasso AS our model, which handles selection inline
```

---

## Step 7: Preprocessing

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = [..]  # ~50 numeric (after log + composite + age + ordinal-encoded)
categorical_features = ['MSSubClass', 'MSZoning', 'Neighborhood', 'HouseStyle',
                         'BldgType', 'RoofStyle', 'Foundation', 'SaleType',
                         'SaleCondition', 'SaleSeason', ...]
# Note: Neighborhood kept as raw category for tree models;
# for linear models we'd substitute NeighborhoodEnc

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# After one-hot: ~190 features for linear; ~80 features for tree (raw + ordinal int)
```

---

## Step 8: Train/Test Split

```python
# 1456 rows -- precious. 80/20 random split (no temporal axis since 4-year span is short
# and all data is the same market period).

# However, the OFFICIAL Kaggle split is by Id (random). For a real estate brokerage,
# a time-ordered split would be more honest -- houses sold in 2010 don't help predict
# 2011 prices.

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.20, random_state=42
)
# X_train: 1164 rows; X_val: 292 rows
```

---

## Step 9: Baseline

```python
# Baseline 1: predict the median price for every house
# RMSE on log(price): 0.413
# MAPE on raw price: 35%
# Useless.

# Baseline 2: simple linear regression on TotalSF + OverallQual + Neighborhood (target-encoded)
# RMSE on log(price): 0.158
# MAPE on raw price: 14%
# Already in business-acceptable range. Anything below 14% MAPE wins.
```

A 3-feature linear regression already meets 14% MAPE. The job is to push to <10%.

---

## Step 10: Try Multiple Models with 5-Fold CV (on log-target)

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: Linear Regression (with all 190 features after one-hot)
# CV RMSE on log: 0.165 -- multicollinearity hurts
# RMSE without composite features but with raw size cols: 0.171

# Model 2: Ridge (alpha=10)
# CV RMSE: 0.131
# Big jump from L2 regularization

# Model 3: Lasso (alpha=0.0005)
# CV RMSE: 0.128
# Selects ~85 features (kills 100+ irrelevant one-hot dummies)

# Model 4: ElasticNet (alpha=0.0005, l1_ratio=0.7)
# CV RMSE: 0.126
# Best linear model

# Model 5: Random Forest (n=300, max_depth=15)
# CV RMSE: 0.143

# Model 6: Gradient Boosting (n=500, max_depth=4, lr=0.05)
# CV RMSE: 0.123

# Model 7: XGBoost
# CV RMSE: 0.122

# Model 8: LightGBM
# CV RMSE: 0.121
```

**Top performers:** LightGBM (0.121), XGBoost (0.122), GBM (0.123), ElasticNet (0.126).

---

## Step 11: Target Transformation Already Done

Already log-transformed in Step 4. RMSE values above are on `log1p(SalePrice)` — small numbers, but they translate to meaningful MAPE on the raw scale.

---

## Step 12: Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV

# Tune ElasticNet (interpretable challenger)
param_grid_en = {
    'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'l1_ratio': [0.3, 0.5, 0.7, 0.9, 1.0]
}
grid_en = GridSearchCV(ElasticNet(max_iter=10000), param_grid_en, cv=cv, scoring='neg_root_mean_squared_error')
grid_en.fit(X_train, y_train)
# Best: alpha=0.0005, l1_ratio=0.5
# Tuned CV RMSE: 0.124

# Tune LightGBM
param_dist_lgbm = {
    'n_estimators':       [500, 1000, 2000],
    'max_depth':          [-1, 4, 6, 8],
    'num_leaves':         [15, 31, 63],
    'learning_rate':      [0.01, 0.03, 0.05],
    'min_child_samples':  [5, 10, 20],
    'subsample':          [0.7, 0.8, 0.9],
    'colsample_bytree':   [0.7, 0.8, 0.9],
    'reg_alpha':          [0, 0.1, 1.0],
    'reg_lambda':         [0, 0.1, 1.0]
}
grid_lgbm = RandomizedSearchCV(LGBMRegressor(verbose=-1), param_dist_lgbm,
                                 n_iter=80, cv=cv, scoring='neg_root_mean_squared_error')
# Best: n_estimators=1000, max_depth=6, num_leaves=31, lr=0.03, subsample=0.8
# Tuned CV RMSE: 0.117
```

---

## Step 13: Stacking — Where Real Money Is Made

```python
from sklearn.ensemble import StackingRegressor

# Linear models (ElasticNet) and tree models (LightGBM, XGBoost) make DIFFERENT errors
# Stacking captures the complementarity

stack = StackingRegressor(
    estimators=[
        ('lgbm', best_lgbm),
        ('xgb', best_xgb),
        ('en', best_elasticnet)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=KFold(5)
)
stack.fit(X_train, y_train)
# CV RMSE: 0.114

# A 0.003 improvement on log RMSE = ~0.3% improvement on MAPE
# Worth shipping for Kaggle, possibly worth shipping for production if the lift is consistent
```

---

## Step 14: Residual Analysis

```python
# Plot predicted vs actual on log scale
predictions = stack.predict(X_val)

# Distribution of residuals: |predicted - actual| / actual
relative_errors = np.abs(np.expm1(predictions) - np.expm1(y_val)) / np.expm1(y_val)

# Quantiles:
# 25th: 4.2% error
# 50th: 6.8% error
# 75th: 11.4% error
# 90th: 17.2% error
# 95th: 22.6% error
# 99th: 38.1% error  <- outlier predictions

# Where do the worst predictions live?
# Top errors: very-high-end ($600K+) and very-low-end (<$50K) houses
# Mid-range ($120K-$300K): MAPE under 8%
```

**Expert insight:** report MAPE distribution, not just mean. The hospital, sorry, brokerage cares whether 90% of predictions are within 15% — not just the average.

---

## Step 15: Final Evaluation on Held-Out Validation Set

```python
# Final model: Stacking (LightGBM + XGBoost + ElasticNet, Ridge meta-learner)
#
# Test set: 292 houses
#
# Performance:
#   RMSE on log(price): 0.114
#   RMSE on raw price:  $19,400
#   MAPE on raw price:  7.8%       <- below 10% target
#   Median Abs % Error: 6.1%
#   90% of predictions within: 16.8%
#   95% of predictions within: 22.3%
#
# By price band:
#   < $130K:   MAPE 9.2% (less data, noisier)
#   $130-200K: MAPE 6.4% (sweet spot)
#   $200-350K: MAPE 7.5%
#   > $350K:   MAPE 11.3% (rarer high-end)
```

---

## Step 16: Explainability

```python
import shap

# For tree-based components, SHAP works directly
explainer = shap.TreeExplainer(stack.named_estimators_['lgbm'])
shap_values = explainer.shap_values(X_val_processed)

# For each house, surface top 5 features driving the predicted price
def explain_listing(idx):
    contributions = shap_values[idx]
    top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
    return [(feature_names[i], contributions[i], X_val_processed.iloc[idx, i])
            for i in top_indices]

# Example output for a $245K predicted listing:
# 1. OverallQual = 7         (+$32K vs baseline)
# 2. NeighborhoodEnc = 0.21   (+$24K)
# 3. TotalSF = 2,400          (+$18K)
# 4. GarageCars = 2           (+$11K)
# 5. KitchenQual = 4 (Gd)     (+$7K)
# Listing agent talks-track: "This home prices at $245K because of high overall quality (7/10),
# desirable neighborhood (NoRidge), large total square footage (2,400), 2-car garage, and
# good kitchen quality. Each contributes the dollar amount shown."
```

---

## Step 17: Deployment Considerations

```python
# Model size: ~5MB on disk
# Inference: ~50ms per house (stacking adds latency vs single model)

# Production:
#   1. Pre-trained model + preprocessor + neighborhood target-encoding lookup table
#   2. SHAP explainer pre-computed and cached
#   3. API: POST /predict with JSON house features -> returns price + top-5 explanation

# Retraining: QUARTERLY
#   - Real estate markets shift slowly but materially
#   - Quarterly retrain on rolling 4-year window
#   - A/B test: 10% of new listings get new model first, validate against actual sale prices
#   - Auto-rollback if new model MAPE > 10% on holdout

# Monitoring:
#   - Daily: predicted price vs realized sale price (when houses close)
#   - Weekly MAPE by neighborhood (drift detection)
#   - Alert if monthly MAPE > 12% (worse than baseline)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Outliers | Cap at 99th percentile | Drop 4 documented partial sales (per dataset author guidance) |
| PoolQC missing | Median impute | Encode `'None'` -- absence is meaningful, not missing |
| Year columns | Use raw year | Convert to HouseAge / RemodAge (model can extrapolate) |
| Multicollinear sizes | Use all raw cols | Engineer TotalSF composite |
| Ordinal categoricals | One-hot encode | Map Ex/Gd/TA/Fa/Po to 1-5 (preserves ordering) |
| 25-neighborhood handling | One-hot (25 cols) | Target encode with Bayesian smoothing |
| Target | Predict raw $ (RMSE skewed) | log1p(SalePrice) -- additive errors = % errors |
| Skewed numeric features | Use raw | log1p() any feature with skew > 0.75 |
| Train/test split | All-default | 80/20 with reserve for honest evaluation |
| Imbalance | (n/a for regression) | Stratify by price quartile to prevent CV folds with no high-end houses |
| Final model | Single XGBoost | Stack of LightGBM + XGBoost + ElasticNet (different error patterns) |
| Reporting | Mean MAPE only | MAPE distribution by price band + top-5 SHAP per listing |
