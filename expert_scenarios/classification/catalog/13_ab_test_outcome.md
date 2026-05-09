# Expert Scenario 13: A/B Test Outcome Classifier

> **Compact walkthrough.** Closely related to [44_ad_lift_incrementality.md](../../regression/catalog/44_ad_lift_incrementality.md) (causal regression) but framed as binary classification on the outcome. Distinct nuance: the model's role is descriptive (which users responded?) — not prescriptive (which to target?). Useful for post-hoc segmentation analysis.

---

## The Brief

A product team ran an A/B test on a new onboarding flow (treatment vs control, 500K users each, randomized). They want a classifier that predicts whether each user converted within 7 days. The downstream use:

- Identify which user segments responded best to the treatment.
- Inform next iteration of feature design.
- Validate randomization (treatment and control should have similar feature distributions).

---

## Problem Type

```
Type:           Binary classification on conversion outcome
Primary Metric: AUC; per-segment recall
Secondary:      Calibration (matters for downstream funnel modeling)
Imbalance:      Mild (12% conversion overall)
```

---

## Distinct Concerns

1. **Treatment is a feature** — including/excluding it answers different questions:
   - Include: model captures heterogeneous response (uplift-adjacent).
   - Exclude: model captures user propensity (descriptive).
2. **Don't confuse with uplift modeling** — this scenario is about who converted, not who would have converted differently. For causal questions, see scenario 44.
3. **Validate randomization** — feature distributions should be similar across treatment/control. Use a "treatment classifier" test: can a model distinguish treatment from control by features alone? AUC should be ~0.50 if randomization worked.

---

## Recommended Approach

- **LightGBM** with treatment as a feature
- Compute per-segment AUC and uplift on holdout
- Validate randomization first

---

## Summary

| Technique | Beginner | Expert |
|-----------|----------|--------|
| Causal vs descriptive | Conflate | Distinguish: this is descriptive; uplift is causal (scenario 44) |
| Randomization check | Skip | Treatment-classifier AUC ≈ 0.50 sanity check |
| Treatment as feature | Always include | Include if descriptive; exclude for uplift inference |
| Output | Single label | Probability vector + per-segment lift table |
