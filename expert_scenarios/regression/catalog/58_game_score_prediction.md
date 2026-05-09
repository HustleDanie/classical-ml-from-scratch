# Expert Scenario 58: NBA Game Score Prediction (Sports Regression)

> **Complexity:** Low signal-to-noise (sports outcomes are inherently random), temporal split required, paired binary + regression framing (predict winner AND margin), home-court / rest-day / injury effects compound, evaluation must include against-the-spread ATS performance.

---

## The Brief

A sports media company gives you 12 NBA seasons of game data (~14,000 games). They want a model predicting:

1. The point spread (home team minus away team).
2. The total points scored (over/under).

These predictions feed into:

- Pre-game commentary ("expect a tight game") and in-game graphics.
- Advertising targeting (high-scoring games command higher CPMs).
- Optional: hedge fund desk uses for market-side predictions vs Vegas lines.

Constraints:

- RMSE on point spread ≤ 11 points.
- Beat Vegas closing line (or come within 0.5 RMSE) — Vegas is the ML benchmark in sports.
- Inference at game-time: 30 minutes before tip-off (last lineup adjustments visible).
- Robust across regular season + playoffs (different dynamics).
- Account for star player absence (Doncic out → spread shifts 4-7 points).

This is harder than typical regression because: each game is one data point per team-pair-day; the signal-to-noise ratio is low (Vegas is approximately optimal); injury reports change predictions by 2-7 points; the underlying market is competitive (someone else also has a good model).

---

## Step 1: Define the Problem Type

```
Type:           Regression on point spread (-30 to +30)
Primary Metric: RMSE on spread; secondary: ATS (against-the-spread) accuracy
                (model better than Vegas line in % of cases)
Secondary:      RMSE on total points
                Per-game-type: regular season vs playoff
                Pinball loss for prediction intervals
Business Goal:  RMSE ≤ 11 points; ATS ≥ 50.5% (slightly beat Vegas)
Constraint:     30-min pre-tip prediction; survives playoff regime
Target shape:   Roughly normal (-15 to +15), occasional blowouts
```

**Expert thinking:** RMSE alone isn't enough. In sports betting, "beat the closing line" matters — Vegas has efficient prediction, so the test is can your model beat their consensus. ATS (against-the-spread) accuracy compares predictions to the actual spread.

---

## Step 2: Understand the Data

```
Shape: 14,000 games × ~120 features per game

Per game:
- game_id (unique)
- date
- season (12 seasons)
- season_type (regular / playoff)
- home_team_id, away_team_id
- home_score (post-game; target component)
- away_score (post-game; target component)
- score_diff_home (target = home - away)
- total_points (secondary target)

PRE-GAME FEATURES (available 30 min before tip):

Team Form:
- home_team_record_l10 (wins in last 10)
- away_team_record_l10
- home_avg_score_l10
- away_avg_score_l10
- home_avg_allowed_l10
- away_avg_allowed_l10
- home_offensive_rating_l10
- away_offensive_rating_l10
- home_defensive_rating_l10
- away_defensive_rating_l10
- home_pace_l10
- away_pace_l10

Rest / Travel:
- home_rest_days
- away_rest_days
- away_travel_distance_miles
- home_back_to_back (binary)
- away_back_to_back (binary)
- home_3_in_5 (3 games in 5 days)
- away_3_in_5
- altitude_difference (Denver, Utah are high)

Injury / Lineup:
- num_starters_out_home
- num_starters_out_away
- star_player_out_home (binary; player worth > 5 win shares missing)
- star_player_out_away
- minutes_lost_home (estimated minutes for missing starters)
- minutes_lost_away

Head-to-Head:
- h2h_avg_score_diff_recent_5_meetings
- days_since_last_meeting
- season_series_score_diff_so_far

Venue / Crowd:
- is_home_court
- expected_attendance
- arena_factor (some venues favor offense due to crowd noise)

Market / Vegas:
- vegas_opening_spread
- vegas_closing_spread (available 30 min pre-tip)
- vegas_over_under
- public_betting_pct_home
- sharp_money_indicator (if money has moved against the public)
- line_movement_l24h (how much the line shifted)

Season Context:
- num_games_into_season
- standings_implication (high-stakes vs garbage time at end of season)
- is_playoff_chase (binary)
- coach_change_recent (within 30 days)
```

**Expert thinking:** the Vegas closing line is the strongest single feature. It encodes professional bookmakers' best estimate AND the wisdom of the crowd of bettors. Most sports models that don't use Vegas data underperform.

---

## Step 3: EDA

```python
df['score_diff_home'].describe()
# count    14,000
# mean      +2.4 (home team advantage)
# std        13.1
# min       -45
# 25%        -7
# 50%        +3
# 75%       +12
# max       +52
# Skewness: 0.05 (basically normal)

# Home advantage (the most well-known phenomenon in sports)
df['score_diff_home'].mean()  # +2.4 (home teams win by 2.4 on average)

# Effect of star player absence
df.groupby(['star_player_out_home', 'star_player_out_away']).agg(
    avg_diff=('score_diff_home', 'mean'),
    n=('score_diff_home', 'count')
)
# Both healthy:  avg_diff +2.5, n=11200
# Home star out: avg_diff -1.8, n=850 (drops by 4.3 vs healthy)
# Away star out: avg_diff +6.7, n=920 (gain 4.2 vs healthy)
# Both stars out: avg_diff +2.1, n=130

# Vegas line accuracy
df['vegas_residual'] = df['score_diff_home'] - df['vegas_closing_spread']
df['vegas_residual'].describe()
# mean: 0.05 (Vegas is almost unbiased)
# std:  10.8 (Vegas RMSE ~ 10.8)
# Vegas IS the benchmark
```

---

## Step 4: Data Cleaning

```python
# === HANDLE PLAYOFF DIFFERENCES ===
# Playoff games have different dynamics: smaller sample per team, higher stakes
# Don't drop them; flag with a binary feature
df['is_playoff'] = (df['season_type'] == 'playoff').astype(int)

# === HANDLE COACH CHANGES ===
# When a team changes coaches, prior data is less relevant
df['team_dynamics_unstable'] = (df['days_since_coach_change_home'] < 30).astype(int) | \
                                 (df['days_since_coach_change_away'] < 30).astype(int)

# === IMPUTE INJURY DATA FOR HISTORICAL GAMES ===
# Injury reporting wasn't as detailed in older seasons
# For pre-2015 games, use proxy: did star player play (DNP-CD vs INJURY)?
df['star_player_out_home'] = df['star_player_out_home'].fillna(0)
```

---

## Step 5: Feature Engineering

```python
# === RELATIVE STRENGTH ===
df['offensive_rating_diff'] = df['home_offensive_rating_l10'] - df['away_defensive_rating_l10']
df['defensive_rating_diff'] = df['home_defensive_rating_l10'] - df['away_offensive_rating_l10']
df['pace_avg'] = (df['home_pace_l10'] + df['away_pace_l10']) / 2

# === REST / FATIGUE ===
df['rest_disadvantage'] = df['home_rest_days'] - df['away_rest_days']
df['both_back_to_back'] = (df['home_back_to_back'] & df['away_back_to_back']).astype(int)
df['fatigue_score'] = df['home_3_in_5'].astype(int) - df['away_3_in_5'].astype(int)

# === INJURY IMPACT ===
df['injury_diff_minutes'] = df['minutes_lost_home'] - df['minutes_lost_away']
df['star_diff'] = df['star_player_out_home'].astype(int) - df['star_player_out_away'].astype(int)

# === MARKET FEATURES ===
df['line_movement'] = df['vegas_closing_spread'] - df['vegas_opening_spread']
df['public_vs_sharp'] = df['public_betting_pct_home'] - 50  # >0 = public on home
df['sharp_money_against_public'] = (df['sharp_money_indicator'] == 1).astype(int)

# === SEASON CONTEXT ===
df['games_remaining_in_season'] = 82 - df['num_games_into_season']
df['playoff_implication_weight'] = (df['standings_implication'] == 'high_stakes').astype(int)

# === HOME COURT FEATURES ===
df['home_court_advantage_l10'] = df['home_team_home_record_l10'] - df['home_team_away_record_l10']

# === ALTITUDE ADJUSTMENT ===
df['high_altitude_visit'] = ((df['away_team_id'].isin([DEN, UTA])) &
                              (~df['home_team_id'].isin([DEN, UTA]))).astype(int)
df['low_altitude_for_high_team'] = ((df['home_team_id'].isin([DEN, UTA])) &
                                      (~df['away_team_id'].isin([DEN, UTA]))).astype(int)
```

---

## Step 6: Train/Test Split — Temporal

```python
df = df.sort_values('date')

# Train: seasons 1-10
# Validation: season 11
# Test: season 12 (most recent, never seen during training)

# Walk-forward CV inside training (5 folds, season-based)
```

---

## Step 7: Try Multiple Models

```python
# === Baseline 1: predict +2.4 (mean home advantage) for every game ===
# RMSE: 13.0

# === Baseline 2: vegas_closing_spread directly ===
# RMSE: 10.8 (Vegas is the bar)

# === Model 1: Linear Regression on engineered features ===
# RMSE: 11.2 (worse than Vegas)
# Note: linear can't beat Vegas because it can't capture nonlinearities

# === Model 2: Random Forest ===
# RMSE: 10.9 (essentially tied with Vegas)

# === Model 3: Gradient Boosting ===
# RMSE: 10.7 (slightly beats Vegas)

# === Model 4: LightGBM ===
import lightgbm as lgb
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=500, max_depth=4, num_leaves=15,
    learning_rate=0.05, reg_alpha=1, reg_lambda=2,  # heavy regularization
    random_state=42
)
# RMSE: 10.4  ✓ (target ≤ 11; beats Vegas by 0.4)

# === Model 5: ElasticNet ===
# RMSE: 10.9
```

**Top performer:** LightGBM (RMSE 10.4 vs Vegas 10.8).

---

## Step 8: ATS Evaluation

```python
# Against-the-spread (ATS) is the actual betting metric
# A "winning bet" is when our prediction beats Vegas's spread

def compute_ats(predictions, actual_spread, vegas_spread):
    """If predicted_spread > vegas_spread, bet on home; else bet on away.
       Win condition: actual matches the side we bet on."""
    bet_on_home = predictions > vegas_spread
    bet_on_away = predictions < vegas_spread
    won = ((bet_on_home & (actual_spread > vegas_spread)) |
           (bet_on_away & (actual_spread < vegas_spread)))
    return won.mean()

# Our model: 51.4% ATS accuracy (need ≥ 52.4% to overcome -110 vig)
# Even though we beat Vegas RMSE, getting profitable ATS is harder
```

**Expert insight:** "beat Vegas RMSE" and "beat Vegas profit" are different. Even a 51-52% ATS hit rate is below break-even due to bookmaker fees. A profitable model needs ~52.4% ATS. Most sports models get 51-52% — close, but not enough for sustained profit. Beware overfitting in CV inflating the apparent ATS rate.

---

## Step 9: Per-Type Performance

```python
# Regular season vs playoff

reg_mask = ~df_test['is_playoff']
print(f"Regular season RMSE: {rmse(predictions[reg_mask], y_test[reg_mask]):.1f}")
# 10.2

playoff_mask = df_test['is_playoff']
print(f"Playoff RMSE: {rmse(predictions[playoff_mask], y_test[playoff_mask]):.1f}")
# 12.8 -- much harder to predict playoffs

# Why playoffs harder:
# 1. Smaller sample (16 teams)
# 2. Adjusted strategies between games of a series
# 3. Star player rest/load is more strategic
# 4. Higher emotional / psychological factor
```

---

## Step 10: Final Evaluation

```python
# Final model: LightGBM, regularized, on engineered features incl Vegas line
# Test set: season 12 (1,230 games)
#
# Performance:
#   RMSE on spread:                10.4 points  ✓ (target 11)
#   RMSE vs Vegas RMSE:           10.4 vs 10.8 (beat by 0.4)
#   ATS accuracy:                 51.6%  (slightly above Vegas, below break-even)
#   Mean home court margin:        2.5 points (matches data)
#   Per-segment:
#     Regular season RMSE:         10.2
#     Playoff RMSE:                12.8
#     Game with star out: RMSE     11.5 (more variable)
#     Game with both stars: RMSE   10.0
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Metric | RMSE only | RMSE + ATS (against-the-spread) — different optimization |
| Vegas line | Treat as one feature | Use as both feature AND benchmark |
| Train/test split | Random | Temporal (season-based) |
| Star player tracking | Skip | Critical feature; +/-4 point swing |
| Rest/travel | Skip | Modeled explicitly with multi-feature interactions |
| Pace | Use raw possession count | Pace_avg as composite of both teams |
| Regularization | Default | Heavy L1+L2 (low signal-to-noise; prevent overfitting) |
| Playoff handling | Train, ship | Per-segment evaluation; harder to predict |
| Profit threshold | "Beat RMSE" | Account for vig: need ≥ 52.4% ATS, not 51% |
| Backtest | None | Use test season; verify ATS rate matches in-sample |
| Coaching changes | Ignore | Flag teams with recent coach change as unstable |
| Altitude effects | Ignore | Explicit feature (Denver, Utah games are different) |
