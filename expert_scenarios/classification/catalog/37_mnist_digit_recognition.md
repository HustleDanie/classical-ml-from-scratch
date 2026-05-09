# Expert Scenario 37: Handwritten Digit Recognition (MNIST)

> **Compact.** Educational benchmark; closest analog [38_traffic_sign_classification.md](38_traffic_sign_classification.md) (image-feature multiclass). Set realistic expectations: classical methods cap ~97-98% on MNIST; CNNs achieve 99.5%+.

```
Type:           Multiclass classification, 10 classes, balanced
Metric:         Top-1 accuracy
Realistic ceiling: ~97% with classical ML
```

**Approach:** Random Forest or XGBoost on raw 784 pixels OR HOG features (slight lift). SVM with RBF on a subset works well.

**Distinct concerns vs production:**
- Pure educational scenario; for production, use a CNN.
- Set stakeholder expectations: classical methods on MNIST = 96-97%, deep learning = 99.5%+.

| Beginner | Expert |
|----------|--------|
| Try to hit 99% with classical ML | Set ceiling expectation at 97% |
| Use raw pixels | HOG or PCA-compressed features |
| Compare to deep learning unfairly | Position as classical ML benchmark |
