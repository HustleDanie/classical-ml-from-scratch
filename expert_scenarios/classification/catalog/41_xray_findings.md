# Expert Scenario 41: X-ray Finding Classification (Multilabel Medical Imaging)

> **Compact walkthrough.** Combines patterns from [33_document_tags_multilabel.md](33_document_tags_multilabel.md) (multilabel architecture) and [71_cancer_screening.md](71_cancer_screening.md) (medical regulated context). Distinct nuance: 14+ co-occurring findings per image, severe per-finding imbalance, FDA SaMD regulated, radiologist disagreement on subtle findings.

---

## The Brief

A radiology imaging center provides 200K chest X-rays with multi-radiologist labels (3-5 radiologists per image). Each X-ray can have 0+ of 14 possible findings (Cardiomegaly, Pneumonia, Pleural Effusion, Atelectasis, Consolidation, Edema, Pneumothorax, Mass, Nodule, Infiltration, Emphysema, Fibrosis, Pleural Thickening, Hernia). The goal: per-finding probability for each X-ray.

---

## Problem Type

```
Type:           Multilabel binary classification, 14 findings
Primary Metric: Per-finding AUC; Macro-AUC weighted by support
Secondary:      Sensitivity at 95% specificity per finding
                Inter-rater agreement check
Imbalance:      Per-finding; some findings 0.5%, others 15%
Constraint:     FDA Class II SaMD; radiologist override mandatory
```

---

## Key Differences vs Similar Scenarios

| Aspect | This (X-ray multilabel) | 33 (Doc tags) | 71 (Cancer screening) |
|--------|------------------------|---------------|----------------------|
| Output structure | 14 binary | 32 binary | 1 binary |
| Domain | Medical imaging | NLP | Medical imaging |
| Regulation | FDA SaMD | None | FDA SaMD |
| Per-class imbalance | Wide range | Wide range | Single class |

---

## Recommended Approach

- **Per-finding Logistic Regression with L2** + per-finding threshold tuning
- Features: pre-extracted CNN embeddings (1024-d) + radiologist priors (age, sex, prior findings)
- Per-finding calibration via isotonic on a holdout
- Inter-radiologist agreement features (when they disagreed, model uncertainty should match)

---

## Distinct Techniques

1. **Annotator disagreement as feature** — when 3 of 5 radiologists agree (vs 5 of 5), label uncertainty is high; reflect in model output uncertainty.
2. **Multilabel calibration per finding** — common findings like "Atelectasis" need different threshold than rare "Hernia."
3. **FDA SaMD path** — clinical study with multi-reader, multi-case design.
4. **Radiologist-in-the-loop** — never autonomous; always advisory.

---

## Summary: Beginner vs Expert

| Technique | Beginner | Expert |
|-----------|----------|--------|
| Architecture | Multiclass softmax (wrong — they co-occur) | Per-finding binary (Binary Relevance) |
| Threshold | 0.5 default | Per-finding tuned threshold (range 0.10 - 0.45) |
| Inter-rater | Use majority vote | Treat disagreement as informative; reflect in confidence |
| Deployment | Direct integration | Advisory tier; radiologist always overrides |
| Calibration | Use raw output | Per-finding isotonic calibration |
