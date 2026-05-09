# Expert Scenario 37: Hotel Room Rate Pricing

> **Compact.** Dynamic pricing; closest analogs [05_vacation_rental_rate.md](05_vacation_rental_rate.md) and [03_ride_pricing.md](03_ride_pricing.md). Distinct: small fixed inventory; competitor rate visibility.

```
Type:           Regression on optimal rate per night per room type
Metric:         RevPAR (revenue per available room)
```

**Approach:** Gradient Boosting on calendar + occupancy + competitor rates. Reinforcement learning in mature production.

**Distinct vs vacation rental:**
- Hotels have small inventory (10-500 rooms) → high variance per-room
- Competitor rates publicly visible (better signal than rental market)
- Bayesian shrinkage helps for small-inventory rooms

| Beginner | Expert |
|----------|--------|
| Predict per room | Bayesian shrinkage for small-inventory rooms |
| Skip competitor data | Competitor rates as feature |
