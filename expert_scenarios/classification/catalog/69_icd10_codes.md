# Expert Scenario 69: ICD-10 Disease Code Assignment

> **Compact.** Hierarchical multilabel with very many classes. See [67_ecommerce_product_taxonomy.md](67_ecommerce_product_taxonomy.md) and [33_document_tags_multilabel.md](33_document_tags_multilabel.md).

```
Type: Hierarchical multilabel, 3-5 levels, 70K+ leaf codes
Approach: Per-chapter Logistic Regression; transformer fine-tuning in production
```

**Distinct:** label sparsity — many leaf codes have < 10 examples; route to chapter-level fallback.
