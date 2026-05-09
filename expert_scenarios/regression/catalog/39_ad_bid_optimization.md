# Expert Scenario 39: Ad Bid Optimization

> **Compact.** Real-time pricing; see [48_rtb_click_prediction.md](../../classification/catalog/48_rtb_click_prediction.md) (FTRL pattern).

```
Type: Regression on optimal bid, < 10ms latency
Approach: Logistic regression with FTRL on click probability; bid = pCTR × value × discount
Metric: ROI; CPC; conversion rate
```

**Distinct:** millisecond budgets — model size constrained by inference speed.
