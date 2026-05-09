# Expert Scenario 39: Food Category from Photo

> **Compact.** Multiclass image-feature classification. See [37_mnist_digit_recognition.md](37_mnist_digit_recognition.md) and [40_product_image.md](40_product_image.md).

```
Type: Multiclass, 50-500 food classes, imbalanced
Approach: Logistic Regression Softmax on CNN embeddings (transfer learning)
Metric: Top-5 accuracy
```

**Distinct:** "salad" looks like 50 different things — visual ambiguity is structural.
