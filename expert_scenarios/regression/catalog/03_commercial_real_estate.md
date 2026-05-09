# Expert Scenario 3: Commercial Real Estate Valuation

> **Compact.** See [01_house_price_prediction.md](01_house_price_prediction.md) for residential pattern. Distinct: market segments (Class A office vs warehouse) need separate models; small-medium dataset; very high target variance.

```
Type:           Regression on price per square foot
Metric:         MAPE (commercial valuations span 6 orders of magnitude)
```

**Approach:** Random Forest with log target; per-segment models for major asset classes (office, retail, industrial, multifamily).

| Beginner | Expert |
|----------|--------|
| One model | Per-segment models for major asset classes |
| RMSE on raw | MAPE on log(price/sqft) |
