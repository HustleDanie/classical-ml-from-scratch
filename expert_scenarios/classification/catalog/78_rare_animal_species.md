# Expert Scenario 78: Rare Animal Species Identification

> **Compact.** Long-tail multiclass species ID. See [77_long_tail_ecommerce_category.md](77_long_tail_ecommerce_category.md) (head/tail strategy) and [68_bird_species.md](68_bird_species.md).

```
Type: Multiclass, very many classes, severe long-tail
Approach: Logistic Regression Softmax for top species + nearest-neighbor on embeddings for tail
Metric: Top-1 head accuracy; top-5 tail recall
```

**Distinct:** observation bias — common species over-represented in citizen-science datasets; geographic features help.
