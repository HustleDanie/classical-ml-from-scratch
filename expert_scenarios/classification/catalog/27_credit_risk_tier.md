# Expert Scenario 27: Credit Risk Tier (Low/Medium/High)

> **Compact.** Combines [26_customer_tier_classification.md](26_customer_tier_classification.md) (ordinal multiclass) and [73_loan_risk_scoring_calibrated.md](73_loan_risk_scoring_calibrated.md) (regulated calibrated probability). Distinct: 3-tier ordinal output for lending decisions.

```
Type:           Ordinal multiclass classification (Low/Medium/High risk)
Metric:         Quadratic Weighted Kappa; per-tier AUC; ECOA fairness
Constraint:     Regulated; auditable
```

**Approach:** Same as scenario 73 (calibrated Logistic Regression with L2) but with ordinal output via either cumulative logit or regression-and-bin. Maintain calibration per tier.

**Distinct concerns:**
- ECOA-compliant adverse action codes for "denied" tier (High risk → declined).
- Calibration must hold within each tier (predicted Medium-risk applicants default at Medium-risk rate).
- DI ratio audit for tier assignment.

| Beginner | Expert |
|----------|--------|
| Multinomial softmax | Cumulative logit OR regression-and-bin |
| Single calibration | Per-tier calibration check |
| Skip fairness | DI ratio per protected group on tier assignment |
