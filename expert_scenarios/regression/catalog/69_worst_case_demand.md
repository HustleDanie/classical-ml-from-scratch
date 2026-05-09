# Expert Scenario 69: Worst-Case Demand (P95 Quantile)

> **Compact.** Quantile regression for supply-chain planning. See [70_p99_latency_prediction.md](70_p99_latency_prediction.md) (quantile + conformal).

```
Type: Quantile regression at q=0.95
Approach: Gradient Boosting with quantile loss; train multiple quantiles jointly to avoid crossing
Metric: Pinball loss; coverage
```

**Distinct:** non-crossing quantiles — train multiple quantiles jointly to avoid P10 > P50.
