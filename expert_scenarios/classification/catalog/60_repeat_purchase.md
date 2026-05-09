# Expert Scenario 60: Repeat-Purchase Prediction

> **Compact.** RFM-based binary classification. See [42_customer_spend.md](../../regression/catalog/42_customer_spend.md) for related continuous spend predictions.

```
Type: Binary, mild-to-moderate imbalance
Approach: XGBoost / LightGBM with RFM (Recency × Frequency × Monetary) features
```

**Distinct:** customer dormancy patterns vary by industry — e-commerce ≠ subscription. Tune dormancy window per business.
