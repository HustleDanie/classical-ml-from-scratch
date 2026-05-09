# Expert Scenario 43: Hiring Screen / Resume Pre-Filter

> **Compact.** Combines [44_recidivism_risk.md](44_recidivism_risk.md) (ethics-bounded refusal) and [07_loan_fairness.md](../../../EXPERT_SCENARIO_7_LOAN_FAIRNESS.md). Distinct: EEOC regulation; Amazon 2018 case is the canonical warning.

```
Type:           Binary classification, regulated (EEOC)
Metric:         Recall at fixed precision; demographic parity ratio
```

**Approach:** Logistic Regression with L2 on TF-IDF resume + structured features. Mandatory fairness audit. Treat as decision support, NEVER autonomous filter.

**Critical:** historical hiring data encodes historical bias (Amazon abandoned its resume model in 2018 for exactly this reason). Be willing to recommend AGAINST deployment.

| Beginner | Expert |
|----------|--------|
| Train on past hires | Audit demographic parity; consider whether to ship at all |
| Auto-filter candidates | Decision support tier; humans always retain veto |
