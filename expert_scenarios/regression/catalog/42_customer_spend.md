# Expert Scenario 42: Customer Spend Forecast (90-Day)

> **Compact.** Closest analog [41_customer_lifetime_value.md](41_customer_lifetime_value.md) (2-year CLV). Distinct: shorter horizon, less censoring concern.

```
Type:           Regression on 90-day forward spend
Metric:         MAPE; rank correlation
```

**Approach:** LightGBM with RFM features. Per-segment models for high-volume cohorts.

**Distinct vs CLV:** less censoring (90 days is mostly observable); seasonal patterns dominate; survival regression unnecessary.

| Beginner | Expert |
|----------|--------|
| Same as CLV | Skip survival; single regression model is enough |
| Mean predictions | Per-segment models for cohorts |
