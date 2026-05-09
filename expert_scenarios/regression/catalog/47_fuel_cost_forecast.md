# Expert Scenario 47: Fuel Cost Forecast

> **Compact.** Time-series with macro drivers; analog [28_oil_gas_demand.md](28_oil_gas_demand.md).

```
Type:           Regression on $/gallon
Metric:         MAPE per region
```

**Approach:** SARIMA + macro features (crude oil futures, refinery margins, regional demand).

**Distinct:** geopolitical shocks dominate. Maintain event-handling pathway.

| Beginner | Expert |
|----------|--------|
| Pure time-series | + crude oil futures + macro |
