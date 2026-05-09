# Expert Scenario 8: Options Pricing Residual

> **Compact.** Financial regression on the residual after Black-Scholes baseline. See [06_stock_return_forecast.md](06_stock_return_forecast.md) and [07_bond_yield.md](07_bond_yield.md).

```
Type: Regression on (actual_price - BS_price)
Approach: Gradient Boosting; physics-informed (BS as baseline)
Metric: RMSE in pricing units; PnL on backtest
```

**Distinct:** liquidity drives bid-ask spread — illiquid options are noisy at any depth; filter by volume.
