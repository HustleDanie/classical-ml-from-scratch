# Expert Scenario 74: Sports Betting Calibrated Odds

> **Compact.** Calibrated probability for public-facing odds. See [73_loan_risk_scoring_calibrated.md](73_loan_risk_scoring_calibrated.md) and [58_game_score_prediction.md](../../regression/catalog/58_game_score_prediction.md).

```
Type: Binary, calibrated probability
Approach: Logistic Regression baseline; ensemble + isotonic recalibration
Metric: Log loss; Brier score; profit-at-Kelly-criterion on backtest
```

**Distinct:** market is competitive (sportsbooks have efficient pricing). Beating the market by 1% is hard. Account for the vig (need 52.4%+ ATS for profit).
