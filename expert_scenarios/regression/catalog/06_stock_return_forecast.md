# Expert Scenario 6: Stock Return Forecast (Financial Regression)

> **Complexity:** Extremely low signal-to-noise (~1-3% R² on cross-section), look-ahead bias is everywhere, train/test split MUST be temporal AND respect data availability lag, regime changes break models, evaluation must be backtest-based not just RMSE, transaction costs eat marginal alpha.

---

## The Brief

A quant fund gives you 8 years of daily data on the Russell 3000 universe (~3,000 US stocks). They want a model predicting next-day cross-sectional returns, used to construct a long-short portfolio (long top decile, short bottom decile).

The brief asks for:

- **Information Coefficient (IC) > 0.04** averaged over the test period (an IC of 0.04 is considered "good" in equities; 0.10+ is exceptional).
- **Sharpe ratio > 1.5** on a daily-rebalanced long-short portfolio after transaction costs.
- Maximum daily turnover of 50% (positions don't rotate too fast — costs eat returns).
- **Decay analysis**: signal must work over 1-day horizon AND should still have positive IC over 5-day horizon.
- Robust across regimes: separately backtest on 2015-2018 (low vol), 2019-2021 (COVID era), 2022-2023 (rate-hike regime).

This is fundamentally different from any other ML problem in this catalog because:
1. The signal you're chasing is at most 3% of variance (vs 70-90% in normal ML problems).
2. Most "great features" you'll find in cross-validation are spurious — they would have been arbitraged away in real markets.
3. **Look-ahead bias** is the single largest risk and it's invisible when you don't know to look for it.

---

## Step 1: Define the Problem Type

```
Type:           Cross-sectional regression (predict next-day return per stock)
Primary Metric: Information Coefficient (IC) — Spearman correlation of pred vs actual returns
Secondary:      IC over multiple horizons (1d, 5d, 21d) — should decay smoothly
                Sharpe ratio of resulting long-short portfolio after costs
                Backtest-implied turnover
Business Goal:  IC > 0.04, Sharpe > 1.5 net of costs
Constraint:     1-day prediction horizon; daily rebalancing; max 50% daily turnover
Target shape:   Returns approximately Gaussian after winsorization; tiny mean,
                std dominates
```

**Expert thinking:** RMSE is the wrong metric. Predicting raw returns precisely is impossible (market is too noisy). What matters is **ranking** — can you put stocks in the right order? IC (Spearman correlation) measures rank quality. A model with terrible RMSE but high IC is profitable; a model with great RMSE but no rank correlation is worthless.

---

## Step 2: Understand the Data

```
Shape: ~3,000 stocks × ~2,000 trading days × ~250 features = ~1.5B rows in panel form

Per stock-day, features available AT THE END OF DAY t (used to predict t+1 return):

PRICE / RETURNS (10):
- daily_return (yesterday's close-to-close return) -- AVAILABLE EOD t
- 5d_return                                          -- EOD t
- 21d_return                                         -- EOD t
- 63d_return (~quarter)                              -- EOD t
- volatility_20d (rolling std of returns)            -- EOD t
- volatility_60d                                     -- EOD t
- max_drawdown_60d                                   -- EOD t
- price_relative_to_52w_high                         -- EOD t
- log_dollar_volume                                  -- EOD t
- avg_volume_20d                                     -- EOD t

TECHNICAL (15):
- RSI_14                                             -- EOD t
- moving_avg_5_minus_50                              -- EOD t
- moving_avg_50_minus_200                            -- EOD t
- bollinger_band_position                            -- EOD t
- MACD                                               -- EOD t
- on_balance_volume_z                                -- EOD t
- ... ~9 more standard indicators

FUNDAMENTAL (per stock, updated quarterly with 1-quarter LAG):
- log_market_cap                                     -- EOD t
- price_to_earnings (NaN if E < 0)
- price_to_book
- earnings_yield
- dividend_yield
- debt_to_equity
- return_on_assets
- return_on_equity
- gross_margin
- revenue_growth_yoy
- ... ~30 more

CROSS-SECTIONAL Z-SCORES (computed daily across the universe):
- z_daily_return                                     -- relative rank of return
- z_volatility_20d
- z_log_market_cap
- z_price_to_earnings
- ... most features have a z-version

EARNINGS / NEWS:
- days_to_next_earnings (int, LAG-1; we know schedule)
- days_since_last_earnings_surprise
- days_since_last_news_event

UNIVERSE / SECTOR:
- sector (11 GICS sectors)
- industry_group (24)
- sub_industry (158)
- index_membership (binary: in_S&P_500, in_Russell_1000, etc.)

TARGET:
- next_day_return (close-to-close from t to t+1)  -- shifted forward, used as label only
```

**Expert thinking:** every feature has a "data availability lag" — a fundamental ratio reported in Q1 isn't available until ~6 weeks after the quarter ends. Including a feature based on data not yet released = look-ahead bias. The data must be **point-in-time** (PIT), meaning each feature's value at date t reflects ONLY what was knowable at date t.

---

## Step 3: EDA on the Tail

```python
df['next_day_return'].describe()
# count    6,000,000
# mean    0.0004    (~10 bps avg, mostly market drift)
# std     0.025     (2.5% daily volatility)
# min    -0.485     (-48% one day -- COVID crash etc)
# 25%    -0.011
# 50%     0.000
# 75%     0.012
# max     0.892    (extreme rebound)

# Cross-sectional cleanup
df['next_day_return_winsorized'] = df['next_day_return'].clip(
    df['next_day_return'].quantile(0.005),
    df['next_day_return'].quantile(0.995)
)
# Winsorize 0.5% tails to prevent outliers from blowing up training

# Cross-sectional Z-scoring (the key transformation)
# We don't care about absolute return; we care about RANKING TODAY
df['return_z'] = df.groupby('date')['daily_return'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Cross-sectional std of next-day returns ~2.5%; most "alpha signals" are 5-15bp at best | Tiny effect sizes; models that target this need extreme regularization |
| Top-correlation features have IC ~ 0.01-0.03 | Even strong individual signals are weak; ensembles help more than tuning |
| Many features have lookahead bias if naively computed | Audit every feature for PIT-ness |
| Returns have heavy tails on a daily basis | Winsorize for training; report on raw |
| 2020 returns 5x more volatile than 2018 | Regime change; backtest by sub-period |

---

## Step 4: Data Cleaning — Audit for Look-Ahead Bias

```python
# === LOOK-AHEAD BIAS AUDIT ===
# For every feature, ask: "what was the latest known value at EOD t?"

# Examples of look-ahead bugs:
#   1. Using "today's earnings announcement" as a feature for today's prediction
#      → unless the announcement is BEFORE market close, this is leakage
#   2. Using volume on day t in a 20-day rolling z-score for predicting t+1
#      → fine if you compute the z-score at EOD t (which is when day t closes)
#   3. Joining a quarterly fundamental that was filed 30 days after quarter-end
#      → must use lag-30-day version

# Corrected: require all features to use end-of-day-t data only
# Filing dates lookup: every fundamental has a "first_publish_date" in our database
df['fundamental_at_t'] = df.merge(filings, on='ticker').query('first_publish_date <= date')

# === HOLDING PERIOD STARTS AFTER EOD ===
# Position taken at EOD t; held overnight; closed EOD t+1
# Return = log(close_{t+1} / close_t)
# The CLOSE on day t+1 is the target -- so target needs t+1 close
# Make sure to NOT use any feature computed AFTER market open on day t+1

# === WINSORIZE WITH PIT BOUNDS ===
# DO NOT winsorize using the full-period min/max -- that's lookahead
# Compute bounds on a rolling 1-year window:
df['return_z_capped'] = df.groupby('date')['return_z'].transform(
    lambda x: x.clip(-3, 3)  # cross-sectional, not historical
)
```

**Expert insight:** the most common bug in equity ML is using fundamental data without filing-date lag. A naive join puts Q1 earnings into March 31 features — but the actual filing was May 5. That's 35 days of free information the model gets that real traders didn't. Once you fix this, your "0.10 IC backtest" usually drops to 0.04. That's reality.

---

## Step 5: Feature Engineering — Cross-Sectional Z-Scores

```python
# Most equity factor research uses cross-sectional z-scores
# Why: relative ranking matters more than absolute values
# A P/E of 20 is "high" or "low" relative to TODAY's market average

# === CROSS-SECTIONAL Z FOR ALL FEATURES ===
features_to_z = ['return_5d', 'return_21d', 'return_63d', 'volatility_20d',
                 'price_to_earnings', 'price_to_book', 'log_market_cap',
                 'RSI_14', 'log_dollar_volume', ...]

for f in features_to_z:
    df[f'{f}_z'] = df.groupby('date')[f].transform(
        lambda x: (x - x.median()) / x.std()  # median for robustness
    )

# === SECTOR-NEUTRAL Z-SCORES ===
# Some factors are sector-driven; energy stocks vs tech have different P/E ranges
# Subtract sector median first
for f in ['price_to_earnings', 'price_to_book', 'return_5d']:
    df[f'{f}_sec_z'] = df.groupby(['date', 'sector'])[f].transform(
        lambda x: (x - x.median()) / x.std()
    )

# === MOMENTUM RANKS ===
df['momentum_5d_rank'] = df.groupby('date')['return_5d'].rank(pct=True)
df['momentum_21d_rank'] = df.groupby('date')['return_21d'].rank(pct=True)

# === REVERSAL ===
# Short-term reversal: yesterday's loser tends to bounce back tomorrow
df['daily_return_neg'] = (-df['daily_return_z']).clip(0, None)

# === COMPOSITE QUALITY SCORE (Asness/Frazzini-style) ===
df['quality_score'] = (
    df['return_on_equity_z'].fillna(0) +
    df['gross_margin_z'].fillna(0) +
    df['debt_to_equity_z'].fillna(0).clip(upper=0) * (-1)  # less debt = higher
)

# Final feature set: ~60 cross-sectional z-scored features
```

---

## Step 6: Train/Test Split — Walk-Forward With Embargo

```python
# Cardinal rule of finance ML: walk-forward with EMBARGO

# Setup:
#   Train: rolling 4-year window
#   Embargo: 5 trading days (don't train on data immediately preceding test date)
#   Test: next 1 month

# Why embargo: if a feature has serial correlation, training data from day t-1
# can leak into day t test prediction. Embargo of 5 days ensures the model
# isn't trained on overlapping samples with the test set.

def walk_forward_splits(df, train_years=4, test_months=1, embargo_days=5):
    dates = sorted(df['date'].unique())
    splits = []
    for test_start in pd.date_range(dates[train_years*250], dates[-1], freq='MS'):
        train_end = test_start - pd.Timedelta(days=embargo_days)
        train_start = train_end - pd.DateOffset(years=train_years)
        test_end = test_start + pd.DateOffset(months=test_months)
        splits.append((train_start, train_end, test_start, test_end))
    return splits

# Each backtest fold: train on [t-4y, t-5d], test on [t, t+1mo]
# Total: 60+ folds across 5 years of test data
```

**Expert insight:** without embargo, a model can learn a feature that has 5-day autocorrelation and effectively "see" the test set through that correlation. Embargo of at least the longest feature lookback period is essential.

---

## Step 7: Modeling — Conservative

```python
# With 1.5B rows but only 1-3% R² target, complex models overfit
# Simple, regularized linear or shallow tree models are appropriate

# === Model 1: ElasticNet ===
from sklearn.linear_model import ElasticNet
en = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)
# Walk-forward IC: 0.038 ± 0.012 across 60 folds

# === Model 2: Lasso (more aggressive selection) ===
ls = Lasso(alpha=0.001, max_iter=2000)
# Walk-forward IC: 0.041 ± 0.011

# === Model 3: Ridge (no feature selection) ===
ridge = Ridge(alpha=1.0)
# Walk-forward IC: 0.037 ± 0.012

# === Model 4: LightGBM with quantile loss (predict ranking, not value) ===
# Use ranking-friendly objective; train on top-bottom decile margins
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=300,
    max_depth=4,           # SHALLOW
    num_leaves=15,
    learning_rate=0.05,
    subsample=0.8,
    reg_alpha=0.5,         # heavy regularization
    reg_lambda=1.0,
    random_state=42
)
# Walk-forward IC: 0.046 ± 0.011  <- BEST

# === Model 5: XGBoost ===
# Similar to LightGBM, IC: 0.044
```

**Top performer:** Shallow LightGBM (IC 0.046). Lasso is close (0.041) and more interpretable.

---

## Step 8: Backtest — The Real Test

```python
# IC is helpful but the brief asks for Sharpe of a long-short portfolio
# Construction:
#   Each EOD: rank stocks by predicted return
#   Long: top decile (300 stocks), equal-weighted
#   Short: bottom decile (300 stocks), equal-weighted
#   Daily rebalance

def backtest_long_short(predictions, returns, txn_cost_bps=5):
    daily_returns = []
    for date, group in predictions.groupby('date'):
        # Rank stocks
        long_stocks = group.nlargest(300, 'predicted_return')
        short_stocks = group.nsmallest(300, 'predicted_return')
        # Position weights (equal-weight)
        long_pos = group['ticker'].isin(long_stocks['ticker']).astype(float) / 300
        short_pos = group['ticker'].isin(short_stocks['ticker']).astype(float) / 300
        # Net daily return = long_return - short_return
        net_return = (long_pos * group['actual_return']).sum() - (short_pos * group['actual_return']).sum()
        # Subtract transaction costs (one-way turnover assumption)
        # Roughly: 50% of position turns over → 50% × 2 sides × 5bp = 5bp daily cost
        net_return -= 0.0005
        daily_returns.append((date, net_return))
    return pd.DataFrame(daily_returns, columns=['date', 'return'])

# After backtesting on 5-year out-of-sample period:
# Mean daily return:    0.045%
# Annualized return:    11.3%
# Annualized vol:       6.8%
# Sharpe ratio:         1.66    ✓ (target 1.5)
# Max drawdown:         -8.2%
# Mean turnover:        47%     ✓ (under 50% cap)
# Mean IC:              0.046
```

---

## Step 9: Sub-Period Robustness

```python
# Test in three regimes:
#   2015-2018: low vol, slow trend
#   2019-2021: COVID crash + rebound
#   2022-2023: rate-hike, value re-emergence

# Per-period Sharpe:
#   2015-2018: Sharpe 1.83 (best)
#   2019-2021: Sharpe 1.42 (vol spike hurt momentum factors)
#   2022-2023: Sharpe 1.51 (regime change reduced fundamental factor lift)

# Robustness: still profitable in every sub-period, but Sharpe varies 1.4 - 1.8
```

---

## Step 10: Decay Analysis

```python
# Test the model's IC at multiple horizons
horizons = [1, 2, 3, 5, 10, 21]
ic_by_horizon = {}
for h in horizons:
    df_test[f'return_{h}d_forward'] = df_test.groupby('ticker')['daily_return'].shift(-h).rolling(h).sum()
    ic = spearmanr(df_test['predicted_return'], df_test[f'return_{h}d_forward']).correlation
    ic_by_horizon[h] = ic

# 1d:  0.046
# 2d:  0.041 (still positive)
# 3d:  0.034
# 5d:  0.024 (still positive)
# 10d: 0.008
# 21d: -0.002 (no signal at month horizon)

# Smooth decay -- signal is real, doesn't reverse
# A reverse signal at horizon-N would suggest the 1d model is exploiting noise
```

---

## Step 11: Per-Sector / Per-Cap Performance

```python
# Audit IC across sectors and market-cap buckets
# Real alpha should be relatively sector-neutral

# Per sector IC:
#   Technology: 0.038
#   Financials: 0.052
#   Healthcare: 0.041
#   Industrials: 0.045
#   Consumer Discretionary: 0.044
#   Energy: 0.029  <- weakest; volatility regimes confound
#   Real Estate: 0.062  <- strongest; less efficient

# Per market cap:
#   Mega cap (>$200B):    IC 0.025  (efficient, less alpha)
#   Large cap ($10-200B): IC 0.040
#   Mid cap ($2-10B):     IC 0.052
#   Small cap (<$2B):     IC 0.061  <- most alpha but liquidity risk
```

**Expert insight:** small-cap alpha being highest is typical. But small caps have poor liquidity — if you put real money in, you'll move the price, and your "Sharpe 1.66" backtest evaporates. Real-money allocation needs to be capacity-aware.

---

## Step 12: Final Evaluation

```python
# Final model: LightGBM (shallow, regularized)
#
# 5-year out-of-sample backtest results:
#   Mean IC:              0.046
#   IC stability (% positive months): 73%
#   Sharpe ratio:         1.66  ✓
#   Max drawdown:         -8.2%
#   Annualized return:    11.3%
#   Mean turnover:        47%   ✓
#
# Decay analysis: smooth positive at horizons 1d-5d, dies at 10d+
#
# Per-sector / per-cap: alpha persistent in all sectors, all cap bands
#
# Sub-period: Sharpe 1.4-1.8 across regime changes (acceptable robustness)
```

---

## Step 13: Deployment

```python
# Production:
#   - Daily EOD: pull latest data, audit for any look-ahead, refresh features
#   - Generate predictions for all R3000 stocks
#   - Construct portfolio (top/bottom decile)
#   - Submit orders for next day's open execution
#
# Retraining: MONTHLY rolling
#   - Train on most recent 4 years (excluding 5-day embargo)
#   - Compare new model IC to incumbent on out-of-sample window
#   - Champion-challenger: deploy if new model beats by 0.005 IC for 2 consecutive months

# Monitoring:
#   - Daily IC on closed positions
#   - Weekly Sharpe drift
#   - Drawdown alerts (auto-reduce position size if drawdown > 5%)
#   - Regime detection: if VIX > 30 OR rates change > 50bp / month, flag for review

# Risk overlay:
#   - Beta-neutral (long-short hedges market risk)
#   - Sector-neutral (don't take large unintended sector bets)
#   - Position limit per stock: 0.5% of portfolio
#   - Liquidity check: never trade > 5% of average daily volume
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Metric | RMSE | Information Coefficient (Spearman); backtest Sharpe |
| Train/test split | Random | Walk-forward + embargo + multi-regime testing |
| Look-ahead | Use latest data | Audit every feature for point-in-time correctness |
| Fundamental data | Latest values | Lagged with proper filing dates (e.g. Q1 EPS not in features until ~May) |
| Target | Raw return | Cross-sectionally z-scored or ranked |
| Features | Raw values | Cross-sectional z-scores; sector-neutral z-scores |
| Model size | XGBoost 1000 trees max_depth=10 | Shallow LightGBM (max_depth=4); heavy regularization |
| Evaluation | CV RMSE | Walk-forward backtest with transaction costs |
| Profitability check | "model has good RMSE" | Long-short Sharpe > target after costs |
| Sub-period | One backtest | Multi-regime: low-vol, COVID, rate-hike eras |
| Decay | Single horizon | IC at 1d/2d/3d/5d/10d -- should decay smoothly |
| Capacity | Ignore | Small-cap signal is real but limited; account for liquidity |
| Risk | "max drawdown is fine" | Beta-neutral + sector-neutral + position limits + liquidity caps |
