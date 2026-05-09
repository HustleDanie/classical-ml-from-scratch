# Expert Scenario 10: Customer Complaint Escalation Prediction

> **Compact.** Combines text classification with imbalance; closest analog [17_email_spam_detection.md](17_email_spam_detection.md). Distinct: text + structured features fused; survivorship bias on de-escalated tickets.

```
Type:           Binary classification, 3-10% escalation rate
Metric:         Recall at fixed precision (rep workload tolerance)
```

**Approach:** Logistic Regression on TF-IDF + structured features. ColumnTransformer.

**Distinct:** Successful de-escalations look like "non-escalations" → biased labels. Track de-escalation actions explicitly.

| Beginner | Expert |
|----------|--------|
| Text-only | Text + structured fused |
| Trust labels | Track de-escalation as separate signal |
