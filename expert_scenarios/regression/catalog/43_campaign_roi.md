# Expert Scenario 43: Campaign ROI Prediction

> **Compact.** Causal-flavored marketing regression. See [44_ad_lift_incrementality.md](44_ad_lift_incrementality.md).

```
Type: Regression on campaign return; small-medium data
Approach: Bayesian regression for uncertainty; Gradient Boosting for raw fit
Metric: Out-of-sample MAPE; budget-allocation backtest
```

**Distinct:** causal vs correlational — confounding by self-selection of past campaigns; randomization helps.
