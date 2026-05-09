# Expert Scenario 66: Website Visit Count

> **Compact.** Count regression. See [67_call_center_volume.md](67_call_center_volume.md).

```
Type: Count regression, time-series
Approach: Poisson Regression baseline; LightGBM with `objective='poisson'`
Metric: Poisson deviance; MAPE
```

**Distinct:** zero-inflation — many pages have zero visits in a given hour; ZIP model.
