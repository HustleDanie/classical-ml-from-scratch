# Expert Scenario 51: Real-Time Recommendation Click Prediction

> **Compact.** Closely related to [07_ctr_prediction.md](07_ctr_prediction.md) (advertiser-side) and [48_rtb_click_prediction.md](48_rtb_click_prediction.md) (RTB).

```
Type: Binary, mild imbalance, < 50ms latency
Approach: Factorization Machines or shallow LightGBM; precomputed user/item embeddings
```

**Distinct:** cold-start (new users/items) needs a fallback model; embedding cache freshness matters.
