# Expert Scenario 22: Production Throughput Forecast

> **Compact walkthrough.** Time-series adjacent to [31_daily_sales_forecast.md](31_daily_sales_forecast.md) and manufacturing-adjacent to [21_manufacturing_yield.md](21_manufacturing_yield.md). Distinct nuance: capacity ceilings (you can't produce more than the line allows), shift-pattern features, and downtime as a primary driver.

---

## The Brief

A multi-line factory wants 24-hour-ahead throughput forecasts (units per hour) per line. Forecasts feed labor scheduling, raw material orders, and shipping windows.

```
Per (line, hour):
- units_produced
- shift (Day/Swing/Night)
- has_planned_maintenance (next 24h)
- raw_materials_at_start_of_shift
- crew_size_actual
- machine_uptime_pct_recent_24h
- preceding_shift_units (lag)
```

---

## Problem Type

```
Type:           Regression on units per hour
Primary Metric: MAPE in units; Bias on peak-shift forecasts
Constraint:     Capacity ceiling (model predictions must respect physical limit)
```

---

## Distinct Concerns

1. **Capacity ceiling** — line has a max physical throughput. Predict above it = wrong by definition. Clip predictions at ceiling.
2. **Shift effects** — Day shift typically 1.1x of average; Night 0.85x. Strong feature.
3. **Downtime is asymmetric** — recovering from downtime takes longer than the downtime itself (warm-up effects).

---

## Recommended Approach

- **LightGBM** with lag features + shift dummies + capacity-aware feature
- Clip predictions at line capacity ceiling at inference
- MAPE-weighted loss (each shift weighted by labor cost)

---

## Summary

| Technique | Beginner | Expert |
|-----------|----------|--------|
| Capacity | Predict freely | Clip at ceiling |
| Shift | Treat as one-hot | Shift × hour interaction |
| Downtime recovery | Ignore | Add post-downtime warm-up feature |
| Crew impact | Skip | crew_size as feature; understaffed shifts predict lower |
