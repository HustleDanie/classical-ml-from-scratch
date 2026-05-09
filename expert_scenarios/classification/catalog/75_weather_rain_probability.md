# Expert Scenario 75: Weather Rain Probability

> **Compact.** Calibrated probability output for public weather services. See [73_loan_risk_scoring_calibrated.md](73_loan_risk_scoring_calibrated.md) (calibration patterns) and [27_power_grid_load_forecast.md](../../regression/catalog/27_power_grid_load_forecast.md) (NWP integration).

```
Type: Binary or multiclass, calibrated probability
Approach: Logistic Regression Softmax + isotonic recalibration; ensemble with NWP output
Metric: Brier score; reliability diagram
```

**Distinct:** "30% chance of rain" must mean rain happens 30% of the time on those days — calibration is the spec, not nice-to-have.
