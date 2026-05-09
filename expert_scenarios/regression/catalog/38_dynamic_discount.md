# Expert Scenario 38: Dynamic Discount Optimization

> **Compact.** Real-time dynamic pricing; see [03_ride_pricing.md](03_ride_pricing.md) and [44_ad_lift_incrementality.md](44_ad_lift_incrementality.md).

```
Type: Regression on optimal discount %
Approach: Uplift modeling on response curve; LightGBM regression on revenue per offer
Metric: Incremental revenue per offer
```

**Distinct:** training data is biased by past offer policy — randomization or off-policy correction needed.
