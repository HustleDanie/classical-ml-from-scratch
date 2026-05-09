# Expert Scenario 2: NYC Apartment Rent Prediction

> **Compact.** Builds on [01_house_price_prediction.md](01_house_price_prediction.md) (property pricing). Distinct: rent (vs sale price), high-cardinality neighborhood encoding, rent-control distortion.

```
Type:           Regression on monthly rent
Target shape:   Right-skewed; log-transform
Metric:         MAPE; per-neighborhood MAPE
```

**Distinct features:**
- Subway proximity (distance to nearest line)
- Walk score
- Year of last renovation
- Rent-stabilized flag (huge price compression)

**Distinct concerns:**
- Rent-stabilization distorts the price-feature relationship — exclude or model separately.
- Per-neighborhood target encoding with smoothing (200+ neighborhoods).
- Lease-term seasonality (most leases start September → demand peak).

| Beginner | Expert |
|----------|--------|
| One-hot 200 neighborhoods | Target encode with Bayesian smoothing |
| Ignore rent-stabilization | Flag or exclude rent-stabilized listings |
| Use raw subway distance | Distance to nearest line + walk score composite |
