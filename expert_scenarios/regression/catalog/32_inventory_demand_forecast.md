# Expert Scenario 32: Inventory Demand Forecast

> **Compact.** Time-series with intermittent demand; closest analogs [31_daily_sales_forecast.md](31_daily_sales_forecast.md) and [68_items_sold_per_sku.md](68_items_sold_per_sku.md). Distinct: focus on inventory turn / stockout cost rather than revenue.

```
Type:           Count regression, often intermittent
Metric:         Service Level (P95 demand coverage); MASE
Constraint:     Predict P95 demand; over-stock = capital cost; under = stockout
```

**Approach:** Quantile regression at q=0.95 (worst-case demand) + Croston's for intermittent SKUs. Inventory math then sets reorder point at predicted P95.

**Distinct vs scenario 31 (daily sales):**
- Optimize service level (P95 coverage), not revenue.
- Inventory cost ↔ stockout cost trade-off determines quantile (P90 vs P95 vs P99).

**Distinct vs scenario 68 (per-SKU sold):**
- Same intermittent demand techniques (Croston's).
- Different metric: scenario 68 is WAPE (revenue), scenario 32 is service level.

| Beginner | Expert |
|----------|--------|
| Predict mean demand | Predict P95 quantile |
| Stockout cost = ignore | Trade-off: inventory carrying vs stockout |
