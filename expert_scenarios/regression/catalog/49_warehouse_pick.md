# Expert Scenario 49: Warehouse Pick Time Prediction

> **Compact.** Operational logistics regression. See [46_delivery_time_prediction.md](46_delivery_time_prediction.md).

```
Type: Regression on pick-task duration
Approach: Gradient Boosting with location + worker + item features
Metric: MAE; P95 accuracy
```

**Distinct:** worker-specific effects — random effects model or worker ID with smoothing.
