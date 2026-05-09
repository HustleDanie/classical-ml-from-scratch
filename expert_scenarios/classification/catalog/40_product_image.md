# Expert Scenario 40: Product Category from Image (E-commerce)

> **Compact.** Hierarchical multiclass + image embeddings. Combines [67_ecommerce_product_taxonomy.md](67_ecommerce_product_taxonomy.md) (hierarchy) and [77_long_tail_ecommerce_category.md](77_long_tail_ecommerce_category.md) (long tail).

```
Type: Multiclass hierarchical, 100-10K classes
Approach: Hierarchical Logistic Regression on CNN embeddings; LightGBM challenger
```

**Distinct:** new categories appear monthly; FAISS index update without full retrain (per scenario 77).
