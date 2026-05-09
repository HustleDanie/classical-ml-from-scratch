# Expert Scenario 45: Cart Value Prediction (E-commerce)

> **Compact.** Mid-skew transactional regression. See [42_customer_spend.md](42_customer_spend.md).

```
Type: Regression, mid-skew
Approach: LightGBM; log-transform target for skew
Metric: MAPE; segmentwise accuracy
```

**Distinct:** browsing-only sessions have $0 carts — two-part model (purchase y/n + value).
