# Expert Scenario 40: Subscription Renewal Pricing

> **Compact.** Two-part churn + price elasticity. See [41_customer_lifetime_value.md](41_customer_lifetime_value.md).

```
Type: Regression on optimal renewal price
Approach: Two-part: churn classifier (will they renew?) + price elasticity model
Metric: Expected lifetime revenue
```

**Distinct:** confounds with churn modeling — joint training works better than two separate models.
