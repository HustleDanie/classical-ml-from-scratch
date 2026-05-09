# Expert Scenario 4: Land / Lot Valuation

> **Compact walkthrough.** Builds on [01_house_price_prediction.md](01_house_price_prediction.md) (property pricing). Distinct nuance: undeveloped land has fewer attribute features (no bedrooms, no bathrooms, no condition score) but more zoning / utility / access features. Geospatial context dominates.

---

## The Brief

A regional real estate analytics firm provides 60K sales of undeveloped land lots over 6 years (1-acre to 500-acre parcels). They want a model that values raw land for:

- Bank appraisal support.
- Real estate broker pricing recommendations.
- Speculative investor guidance.

Target: sale price per acre ($1K - $200K+ across regions).

---

## Problem Type

```
Type:           Regression, log-transformed price per acre
Primary Metric: MAPE
Secondary:      Per-zoning-class accuracy
Target shape:   Heavily right-skewed (urban-edge premium); log-transform mandatory
```

---

## Distinct Features (vs Houses)

1. **Zoning matters most** — Residential / Agricultural / Commercial / Mixed-Use, plus subtypes.
2. **Utilities** — Has water? Electric? Sewer? Each adds 20-40% to value when present.
3. **Access** — Road frontage, distance to paved road.
4. **Geospatial features** — distance to nearest city, school district, environmental designations (flood zone, wetlands).

Missing features that DO matter for houses but DON'T exist for raw land: bedrooms, bathrooms, year built, condition, finish quality.

---

## Recommended Approach

- **Random Forest with log target**
- Geospatial KNN as auxiliary feature (nearest 5 sold lots' average price)
- Strong regularization on zoning categorical (target encoding with smoothing)

---

## Summary

| Technique | Beginner | Expert |
|-----------|----------|--------|
| Feature set | Use house features | Use zoning + utilities + access + geospatial |
| Target | Per-lot price | Per-acre price (allows comparison across sizes) |
| Geo | One-hot zip | KNN-based nearest-comparable feature |
| Skew | RMSE on raw | MAPE on log(price/acre) |
