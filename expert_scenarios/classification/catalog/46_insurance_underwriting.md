# Expert Scenario 46: Insurance Underwriting Classification

> **Compact.** Combines [17_insurance_premium_pricing.md](../../regression/catalog/17_insurance_premium_pricing.md) (regulated GLM) and [42_loan_fairness](../../EXPERT_SCENARIO_7_LOAN_FAIRNESS.md). Distinct: state-by-state per-state filing requirements; multi-tier output (Approve/Decline/Refer).

```
Type: Multiclass (3 tiers); regulated
Approach: Logistic Regression Softmax with monotonic constraints; per-state model variants
```

**Distinct:** state-specific feature prohibitions; rate filings must defend each coefficient.
