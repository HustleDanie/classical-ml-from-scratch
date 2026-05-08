# Classification Scenarios — Comprehensive Catalog

A catalog of ~80 realistic classification problems, organized into 15 archetype categories. Each entry summarizes the archetype, key brief signals, expected data shape, recommended approach, primary metric, and watch-outs. Entries marked **Deep dive available** link to a full ~500-line walkthrough. Entries marked **[planned]** are on the roadmap.

Use this catalog with the [methodology guide](../from_brief_to_solution/METHODOLOGY.md) — that doc walks you from a fresh brief + dataset to the matching catalog entry.

---

## How to Read an Entry

```
### N. Scenario Name
- Archetype: short structural description
- Brief signal: phrases in a brief that point at this scenario
- Data shape: typical rows × features × target distribution
- Recommended approach: model + key technique
- Primary metric: what to optimize
- Watch-out: the trap most people fall into
- Deep dive: link if available; otherwise [planned]
```

---

## The 15 Categories at a Glance

| # | Category | Count | What unites them |
|---|----------|-------|------------------|
| 1 | Imbalanced binary tabular | 10 | Minority class < 10%; accuracy is misleading |
| 2 | Balanced binary tabular | 6 | Roughly 50/50; standard binary playbook applies |
| 3 | Text / NLP classification | 8 | TF-IDF or character n-grams; linear models or NB |
| 4 | Multiclass tabular | 8 | 3–20 classes; all-vs-all or softmax |
| 5 | Multilabel | 4 | Each row can have multiple labels simultaneously |
| 6 | Image-feature classification | 5 | Engineered features from images; classical models, not deep learning |
| 7 | Regulated / explainability-first | 5 | Coefficients required by law; SHAP at minimum |
| 8 | Real-time / low-latency | 5 | Inference budget < 100ms |
| 9 | Small-data classification | 5 | < 1,000 rows; simple models, heavy regularization |
| 10 | Mixed-type tabular | 5 | Numeric + categorical + missingness; pipelines required |
| 11 | Time-series event classification | 5 | Predict an event window from temporal sensor / log streams |
| 12 | Hierarchical classification | 3 | Class taxonomy with parent / child relationships |
| 13 | Cost-sensitive / asymmetric loss | 3 | FN cost ≫ FP cost (or vice versa) |
| 14 | Calibrated probability output | 3 | Need well-calibrated probabilities, not just labels |
| 15 | Long-tail / rare-class | 3 | One class is 50%+, dozens of classes are < 1% each |

---

## Category 1 — Imbalanced Binary Tabular

The minority class is < 10% of the data. Accuracy lies; you must use precision / recall / PR-AUC and tune the decision threshold.

### 1. Credit Card Fraud Detection
- **Archetype:** binary classification, extreme imbalance (0.1–0.5%), real-time scoring, asymmetric cost
- **Brief signal:** "rare", "0.x%", "fraud", "anomaly", missed-detection cost ≫ false-alarm cost
- **Data shape:** 100K–10M transactions, 20–50 features (some PCA-anonymized), positive rate 0.1–1%
- **Recommended approach:** XGBoost / LightGBM with `scale_pos_weight`, threshold tuning to a cost function, SHAP per-decision
- **Primary metric:** PR-AUC; recall at fixed precision; expected dollar cost
- **Watch-out:** outliers are the signal — never cap or remove them
- **Deep dive:** [01_fraud_detection.md](01_fraud_detection.md)

### 2. Telecom Customer Churn
- **Archetype:** binary classification, severe imbalance (~2–10%), batch scoring, retention campaign downstream
- **Brief signal:** "predict who will leave / cancel / unsubscribe", monthly business cycle
- **Data shape:** 100K–5M customers, 30–80 mixed-type features, 2–10% positive rate
- **Recommended approach:** LightGBM + class_weight, threshold tuned for top-k retention budget, SHAP for retention drivers
- **Primary metric:** Top-k recall (does the targeting list contain the actual churners?); lift at decile
- **Watch-out:** survival bias — customers still active are right-censored, not "non-churners"
- **Deep dive:** [06_telecom_churn.md](06_telecom_churn.md)

### 3. Loan Default Prediction
- **Archetype:** binary classification, moderate imbalance (5–15%), regulated, batch underwriting
- **Brief signal:** "default", "delinquency", "ECOA", "adverse action"
- **Data shape:** 100K–1M loans, 20–40 features (income, DTI, FICO, employment), 5–15% positive rate
- **Recommended approach:** Logistic Regression with L2 + monotonic-constraint XGBoost challenger, SHAP, threshold tuning
- **Primary metric:** AUC-ROC for ranking; calibration error; KS statistic; fairness metrics across protected groups
- **Watch-out:** sample selection bias — you only see loans that were approved
- **Deep dive:** [07_loan_fairness.md](07_loan_fairness.md)

### 4. Network Intrusion Detection
- **Archetype:** binary or multiclass, severe imbalance (~5–10% intrusion), near-real-time alerting
- **Brief signal:** "intrusion", "attack", "alert", "SOC", connection logs / packet metadata
- **Data shape:** 500K–50M flows, 30–80 features, 5–10% positive rate
- **Recommended approach:** Random Forest or LightGBM with `class_weight`; threshold tuned for analyst alert budget
- **Primary metric:** Recall at fixed daily alert volume (e.g., recall@100 alerts/day)
- **Watch-out:** label drift — attack patterns change weekly; weekly retraining needed
- **Status:** [planned]

### 5. Manufacturing Defect Detection (binary view)
- **Archetype:** binary classification, severe imbalance (1–5%), real-time inline inspection
- **Brief signal:** "defective parts", "yield loss", inline-sensor data
- **Data shape:** 30K–1M parts, 50–500 sensor features, 1–5% defect rate
- **Recommended approach:** LightGBM with `is_unbalance=True`, SHAP for which sensor flagged it
- **Primary metric:** Recall at fixed false-alarm rate (production line tolerance)
- **Watch-out:** correlated sensors create multicollinearity — Lasso to pick a representative subset
- **Deep dive:** [04_manufacturing_defect.md](04_manufacturing_defect.md) (multiclass variant)

### 6. Insurance Fraud Claim Detection
- **Archetype:** binary classification, severe imbalance (1–3%), batch investigation downstream
- **Brief signal:** "suspicious claim", "investigation queue"
- **Data shape:** 50K–500K claims, 30–60 features (claim, policy, claimant, history), 1–3% positive rate
- **Recommended approach:** XGBoost + SHAP; investigators see top-k flagged claims daily
- **Primary metric:** Precision at top-k; investigator hit rate
- **Watch-out:** investigators only confirm flagged claims, so labels for "non-fraud" are uncertain
- **Status:** [planned]

### 7. Click-Through Rate (CTR) Prediction
- **Archetype:** binary classification, severe imbalance (1–5%), real-time bidding, very large data
- **Brief signal:** "CTR", "click", "RTB", millisecond latency budget
- **Data shape:** 10M–1B impressions, 30–100 features (user × ad × context), 1–5% positive rate
- **Recommended approach:** Logistic Regression with hashing trick (FTRL) or LightGBM, calibrated probabilities for bidding
- **Primary metric:** Log loss (calibration matters for bidding); AUC for ranking
- **Watch-out:** distribution shift between training and serving; need online learning
- **Status:** [planned]

### 8. Anomaly Detection in IoT Sensor Streams
- **Archetype:** binary classification or one-class, extreme imbalance (< 0.1%)
- **Brief signal:** "anomaly", "outlier", sensor stream from devices
- **Data shape:** 1M–100M readings, 5–30 features, < 0.1% labeled anomalies
- **Recommended approach:** Isolation Forest or one-class SVM; supervised LightGBM if labels are reliable
- **Primary metric:** PR-AUC; precision at top-k
- **Watch-out:** unlabeled anomalies in "normal" data poison supervised models
- **Status:** [planned]

### 9. Bot / Fake-Account Detection
- **Archetype:** binary classification, moderate imbalance (5–10%), adversarial, high-frequency retraining
- **Brief signal:** "bot", "fake account", "automated", "adversary"
- **Data shape:** 100K–10M accounts, 50–200 features (behavioral, device, network), 5–10% positive rate
- **Recommended approach:** XGBoost with frequent retraining; adversarial probes in CV
- **Primary metric:** Recall at fixed precision; week-over-week recall drop = sign attackers adapted
- **Watch-out:** adversaries adapt; static thresholds rot; build a feedback loop from analyst confirmations
- **Status:** [planned]

### 10. Customer Complaint Escalation Prediction
- **Archetype:** binary classification, mild-to-severe imbalance (3–10%), real-time chat / call routing
- **Brief signal:** "escalation", "supervisor request", "negative sentiment"
- **Data shape:** 100K–5M tickets, mix of structured + text features, 3–10% positive rate
- **Recommended approach:** Logistic Regression on TF-IDF + structured features, or DistilBERT-encoded text + LightGBM
- **Primary metric:** Recall at fixed precision (rep workload tolerance)
- **Watch-out:** survivorship — successful de-escalations look like "non-escalations" in the data
- **Status:** [planned]

---

## Category 2 — Balanced Binary Tabular

50/50 or close. Standard playbook applies. Accuracy is OK as a starting metric.

### 11. Titanic Survival
- **Archetype:** binary classification, moderately balanced (38% positive), small data, mixed-type, missing values
- **Brief signal:** Kaggle starter, demographic + ticket features
- **Data shape:** 891 rows × 15 features
- **Recommended approach:** Random Forest or Gradient Boosting with early stopping; impute Age, encode Cabin presence as a feature
- **Primary metric:** Accuracy or F1 (data is balanced enough)
- **Watch-out:** Cabin column is ~77% missing — missingness itself is informative
- **Status:** [planned]

### 12. Sentiment Classification (positive vs negative)
- **Archetype:** binary text classification, balanced
- **Brief signal:** "review sentiment", "positive vs negative", text + label
- **Data shape:** 25K–500K reviews, 5K–50K TF-IDF features
- **Recommended approach:** Logistic Regression with L2 on TF-IDF, or fine-tuned transformer for more nuance
- **Primary metric:** Accuracy or F1
- **Watch-out:** sarcasm and negation flip sentiment; bigrams help
- **Status:** [planned]

### 13. A/B Test Outcome Classifier
- **Archetype:** binary, balanced by design, downstream of an experiment
- **Brief signal:** "treatment vs control outcome", causal-inference-adjacent
- **Data shape:** 10K–10M users, 5–30 features
- **Recommended approach:** Logistic Regression for transparency; uplift modeling if causal effect is the goal
- **Primary metric:** Treatment effect lift; AUUC for uplift models
- **Watch-out:** correlation ≠ causation; randomization quality matters more than model
- **Status:** [planned]

### 14. Customer Satisfaction (CSAT) Binary
- **Archetype:** binary, balanced (45/55), survey-driven
- **Brief signal:** "satisfied vs not", post-purchase survey
- **Data shape:** 5K–500K responses, 10–40 features
- **Recommended approach:** Logistic Regression or Random Forest; SHAP for top dissatisfaction drivers
- **Primary metric:** Accuracy or F1
- **Watch-out:** non-response bias — happy customers respond more
- **Status:** [planned]

### 15. Sports Match Outcome (win/loss, no ties)
- **Archetype:** binary, balanced
- **Brief signal:** "predict win", historical stats
- **Data shape:** 1K–100K games, 20–80 features
- **Recommended approach:** Logistic Regression baseline, Gradient Boosting for raw accuracy
- **Primary metric:** Accuracy or log loss (if betting odds are downstream)
- **Watch-out:** temporal split required — never use future games to predict past
- **Status:** [planned]

### 16. Cat vs Dog (image-feature based)
- **Archetype:** binary image-feature classification, balanced
- **Brief signal:** "image classify", small-data educational
- **Data shape:** 1K–10K images, 100–500 engineered features (HOG, color histograms)
- **Recommended approach:** SVM with RBF kernel; Random Forest as alternative
- **Primary metric:** Accuracy
- **Watch-out:** for production, deep learning beats classical by 10–20%; this is for education
- **Status:** [planned]

---

## Category 3 — Text / NLP Classification

High-dimensional sparse features (TF-IDF, character n-grams). Linear models and Naive Bayes dominate.

### 17. Email Spam Detection
- **Archetype:** binary text classification, mild imbalance (~30%), real-time
- **Brief signal:** "spam", inbox, email body + headers
- **Data shape:** 5K–500K emails, 5K–50K TF-IDF features, ~30% spam
- **Recommended approach:** Multinomial Naive Bayes or Logistic Regression with L2; calibrated probabilities for "definitely spam" vs "uncertain"
- **Primary metric:** F1; precision at high threshold (don't lose legitimate mail)
- **Watch-out:** adversarial spammers — features rot fast
- **Status:** [planned]

### 18. News Topic Classification (multiclass)
- **Archetype:** multiclass text, ~4–20 topics, balanced
- **Brief signal:** "categorize articles", topic taxonomy
- **Data shape:** 5K–500K articles, 5K–50K TF-IDF features
- **Recommended approach:** Logistic Regression Softmax or Linear SVM with OvR
- **Primary metric:** Macro-F1
- **Watch-out:** topic drift — election season changes topic distribution
- **Status:** [planned]

### 19. Customer Support Intent Detection
- **Archetype:** multiclass text, ~10–30 intents, imbalanced
- **Brief signal:** "intent", "route ticket", chatbot
- **Data shape:** 50K–5M tickets, char + word n-grams
- **Recommended approach:** Logistic Regression Softmax with `class_weight`; for short utterances, transformer
- **Primary metric:** Macro-F1; per-intent recall
- **Watch-out:** rare intents (< 1%) need oversampling or a "fallback to human" route
- **Status:** [planned]

### 20. Sarcasm / Irony Detection
- **Archetype:** binary text, balanced or mild imbalance, hard task
- **Brief signal:** "sarcasm", "irony", subtle linguistic cues
- **Data shape:** 50K–500K tweets/headlines, character + word n-grams
- **Recommended approach:** Logistic Regression baseline; transformer-based for production accuracy
- **Primary metric:** F1; human agreement ceiling matters (often 70–80%)
- **Watch-out:** dataset bias — many "sarcasm" datasets are headline-based, not real conversation
- **Status:** [planned]

### 21. Customer Support Ticket Routing (12+ departments)
- **Archetype:** multiclass text, 10–30 departments, imbalanced
- **Brief signal:** "route to team", department names
- **Data shape:** 80K–5M tickets, 8K+ TF-IDF features
- **Recommended approach:** Logistic Regression Softmax with `class_weight`; Multinomial NB as fast baseline
- **Primary metric:** Macro-F1; routing accuracy weighted by ticket volume
- **Watch-out:** SVM with OvO is impractical (66+ binary models for 12 classes)
- **Status:** [planned]

### 22. Document Type Classification (invoice / contract / resume / etc.)
- **Archetype:** multiclass text + structural features, 5–15 classes, balanced
- **Brief signal:** "document type", OCR'd text
- **Data shape:** 10K–1M documents, 5K+ TF-IDF + structural features (page count, layout)
- **Recommended approach:** Logistic Regression Softmax; layout features help disambiguate
- **Primary metric:** Macro-F1
- **Watch-out:** OCR errors create noise; clean the text pipeline before tuning the model
- **Status:** [planned]

### 23. Document Language Identification (25+ languages)
- **Archetype:** multiclass text, very many classes, character n-gram features
- **Brief signal:** "language ID", i18n routing
- **Data shape:** 200K–10M documents, 3K+ char n-gram features, 25–100 classes
- **Recommended approach:** Multinomial Naive Bayes (gold standard); Logistic Regression Softmax challenger
- **Primary metric:** Per-language accuracy; macro-F1
- **Watch-out:** related languages (Spanish vs Portuguese) are hard; need character n-grams of order 3–4
- **Status:** [planned]

### 24. Toxicity / Content Moderation (binary)
- **Archetype:** binary text, mild imbalance (3–10%), high precision required (false flag = censorship)
- **Brief signal:** "toxic", "hate speech", "moderation"
- **Data shape:** 100K–10M comments, char + word n-grams
- **Recommended approach:** Logistic Regression with L2; transformer for production
- **Primary metric:** Precision at fixed recall; per-protected-group fairness
- **Watch-out:** dialect bias — AAVE flagged as toxic by naive models
- **Status:** [planned]

---

## Category 4 — Multiclass Tabular

3–20 classes on structured (non-text) data. Native multiclass models dominate.

### 25. Manufacturing Defect Type Classification
- **Archetype:** multiclass, 5–10 defect types + OK class, imbalanced
- **Brief signal:** "which defect type", "corrective action varies"
- **Data shape:** 30K–1M parts, 50–500 sensor features
- **Recommended approach:** XGBoost with `multi:softprob`, per-class threshold tuning
- **Primary metric:** Macro-F1; per-class recall (catch each rare defect)
- **Watch-out:** rarest defect class may have < 100 examples — collect more before fine-tuning
- **Deep dive:** [04_manufacturing_defect.md](04_manufacturing_defect.md)

### 26. Customer Tier Classification (Bronze / Silver / Gold / Platinum)
- **Archetype:** ordinal multiclass, 3–5 tiers
- **Brief signal:** "tier", "bracket", ordinal levels
- **Data shape:** 50K–10M customers, 30–80 features
- **Recommended approach:** XGBoost or Random Forest; ordinal regression as alternative for the ordering
- **Primary metric:** Quadratic-weighted kappa (penalizes off-by-many predictions)
- **Watch-out:** business rules often define tiers — model should respect them
- **Status:** [planned]

### 27. Credit Risk Tier (Low / Medium / High)
- **Archetype:** ordinal multiclass, 3 tiers, regulated
- **Brief signal:** "risk tier", lending
- **Data shape:** 50K–1M applications, 30–60 features
- **Recommended approach:** Logistic Regression Softmax (regulated baseline); XGBoost challenger with SHAP
- **Primary metric:** Quadratic-weighted kappa; per-tier recall
- **Watch-out:** must be auditable for ECOA — black-box ensembles need SHAP at minimum
- **Status:** [planned]

### 28. Customer Segment Label (Marketing 5–10 segments)
- **Archetype:** multiclass, 5–10 nominal segments
- **Brief signal:** "segment", "persona", marketing
- **Data shape:** 100K–10M customers, 30–80 features
- **Recommended approach:** XGBoost or LightGBM multiclass; SHAP for segment definition
- **Primary metric:** Macro-F1; segment-volume weighted accuracy
- **Watch-out:** segments often come from prior unsupervised clustering — beware label leakage
- **Status:** [planned]

### 29. ICU Severity Triage (3–5 levels)
- **Archetype:** ordinal multiclass, 4–5 severity levels, real-time, regulated
- **Brief signal:** "severity", "triage", "ICU", "ED"
- **Data shape:** 10K–500K admissions, 50–200 features
- **Recommended approach:** Random Forest or XGBoost with per-class threshold; Logistic Regression baseline for audit
- **Primary metric:** Per-severity recall; misclassification cost matrix
- **Watch-out:** missing labs are informative ("not ordered" = "not concerning") — encode missingness
- **Status:** [planned]

### 30. Equipment Fault Type (multiclass diagnostic)
- **Archetype:** multiclass, 5–15 fault types + OK class
- **Brief signal:** "fault type", "diagnostic code"
- **Data shape:** 10K–1M readings, 20–100 features (vibration, temp, current)
- **Recommended approach:** Random Forest or LightGBM multiclass
- **Primary metric:** Per-fault recall (each fault has a different repair playbook)
- **Watch-out:** new fault types appear — keep an "unknown" path
- **Status:** [planned]

### 31. Weather Class (sunny / rainy / cloudy / stormy)
- **Archetype:** multiclass, 4–8 classes, balanced or mild imbalance, geographic
- **Brief signal:** "forecast class", "weather state"
- **Data shape:** 10K–1M observations, 20–50 features
- **Recommended approach:** XGBoost or LightGBM multiclass; geospatial features (lat, long, elevation)
- **Primary metric:** Accuracy; per-class F1
- **Watch-out:** temporal split required; rare classes (stormy) cluster in time and space
- **Status:** [planned]

### 32. Vehicle Class (sedan / SUV / truck / motorcycle / etc.)
- **Archetype:** multiclass, 5–10 classes, balanced
- **Brief signal:** "vehicle type", insurance / DMV
- **Data shape:** 100K–10M vehicles, 10–30 features
- **Recommended approach:** Random Forest or LightGBM multiclass
- **Primary metric:** Accuracy
- **Watch-out:** trim levels and crossovers blur class boundaries
- **Status:** [planned]

---

## Category 5 — Multilabel

Each row can have multiple labels simultaneously. Different from multiclass.

### 33. Document Tags (e.g. arXiv categories)
- **Archetype:** multilabel text, 5–50 possible tags, average 2–5 tags per doc
- **Brief signal:** "multiple tags", "categories" (plural)
- **Data shape:** 10K–10M documents, 5K+ TF-IDF features
- **Recommended approach:** Binary Relevance (one-vs-rest Logistic per tag); Classifier Chains for tag correlation
- **Primary metric:** Micro-F1; subset accuracy (exact match)
- **Watch-out:** tag co-occurrence patterns matter — independent OvR loses information
- **Status:** [planned]

### 34. Image Tags (multilabel from image features)
- **Archetype:** multilabel image, 10–500 tag vocabulary
- **Brief signal:** "tag images", auto-tagging
- **Data shape:** 10K–1M images, 100–1K engineered or CNN-embedding features
- **Recommended approach:** Binary Relevance with Logistic Regression on each tag (deep learning is better in production)
- **Primary metric:** Micro-F1; per-tag F1 distribution
- **Watch-out:** rare tags (< 100 occurrences) are unlearnable from classical features — drop or fold up
- **Status:** [planned]

### 35. Gene Function Prediction
- **Archetype:** multilabel, 100–10,000 functional categories, very sparse positives per gene
- **Brief signal:** "gene ontology", "function", "annotation"
- **Data shape:** 10K–100K genes, 100+ features (sequence, expression, network)
- **Recommended approach:** Per-label Logistic Regression with L2; structured-output methods if hierarchy is rich
- **Primary metric:** Per-class AUC (rank-based); micro-AUPRC
- **Watch-out:** label hierarchy (parent functions imply children) — must be respected
- **Status:** [planned]

### 36. Content Moderation (multi-violation)
- **Archetype:** multilabel binary, 5–30 violation types per item
- **Brief signal:** "violations", "policy", multiple categories per piece of content
- **Data shape:** 100K–100M items, mixed text + image features
- **Recommended approach:** Binary Relevance with Logistic Regression per violation; calibrated thresholds per class
- **Primary metric:** Per-violation precision at fixed recall
- **Watch-out:** policies change quarterly — bake in retraining cadence
- **Status:** [planned]

---

## Category 6 — Image-Feature Classification

Engineered features from images (HOG, color histograms, CNN embeddings). Classical models on these features. For production, deep learning usually wins — these scenarios are educational or constrained.

### 37. Handwritten Digit Recognition (MNIST)
- **Archetype:** multiclass, 10 classes, balanced, dense pixel features
- **Brief signal:** "digit recognition", classic benchmark
- **Data shape:** 60K images × 784 pixel features
- **Recommended approach:** Random Forest or XGBoost; SVM with RBF on a subset
- **Primary metric:** Accuracy
- **Watch-out:** classical methods cap around 97%; CNNs get 99.5%+ — set expectations
- **Status:** [planned]

### 38. Traffic Sign Classification
- **Archetype:** multiclass, 30–50 sign classes, imbalanced (some signs are rare)
- **Brief signal:** "sign recognition", autonomous driving
- **Data shape:** 50K–500K images × 100–1K engineered features
- **Recommended approach:** SVM with RBF or Random Forest on HOG features
- **Primary metric:** Per-class recall (missing a stop sign matters)
- **Watch-out:** lighting, occlusion, weather drift — augment training data
- **Status:** [planned]

### 39. Food Category from Photo
- **Archetype:** multiclass, 50–500 classes, imbalanced, food photos
- **Brief signal:** "food classify", calorie tracking
- **Data shape:** 100K–1M images × 1K+ features (CNN embeddings)
- **Recommended approach:** Logistic Regression Softmax on CNN embeddings (transfer learning + classical head)
- **Primary metric:** Top-5 accuracy
- **Watch-out:** "salad" looks like 50 different things — ambiguity is structural
- **Status:** [planned]

### 40. Product Category from Image (e-commerce)
- **Archetype:** multiclass hierarchical, 100–10K classes, very imbalanced
- **Brief signal:** "product category", catalog
- **Data shape:** 1M–100M images × 1K+ embeddings
- **Recommended approach:** Hierarchical Logistic Regression on embeddings; LightGBM challenger
- **Primary metric:** Top-1 and Top-5 accuracy at each hierarchy level
- **Watch-out:** category drift — new categories appear monthly
- **Status:** [planned]

### 41. X-ray Finding Classification
- **Archetype:** multilabel binary (each finding present/absent), 10–30 findings, regulated
- **Brief signal:** "radiology", "findings", "abnormality"
- **Data shape:** 50K–1M X-rays × 1K+ features (CNN embeddings + radiologist priors)
- **Recommended approach:** Per-finding Logistic Regression with L2 + SHAP; calibrated probabilities
- **Primary metric:** Per-finding AUC; sensitivity at fixed specificity
- **Watch-out:** label noise — radiologists disagree on subtle findings
- **Status:** [planned]

---

## Category 7 — Regulated / Explainability-First

Coefficients required by law (lending, hiring, healthcare). Model choice is constrained by audit needs.

### 42. Loan Approval (Fair Lending)
- **Archetype:** binary classification, regulated, mixed types, fairness audit
- **Brief signal:** "ECOA", "fair lending", "adverse action notice", protected groups
- **Data shape:** 100K–10M applications, 30–60 features
- **Recommended approach:** Logistic Regression with L2 (primary), XGBoost with monotonic constraints + SHAP (challenger)
- **Primary metric:** AUC + fairness metrics (demographic parity, equal opportunity); KS statistic
- **Watch-out:** removing race/gender doesn't remove bias — proxy features (zip code, name) carry it
- **Deep dive:** [07_loan_fairness.md](07_loan_fairness.md)

### 43. Hiring Screen (Resume Pre-Filter)
- **Archetype:** binary classification, regulated (EEOC), text + structured
- **Brief signal:** "resume", "screen", "hire"
- **Data shape:** 10K–1M resumes
- **Recommended approach:** Logistic Regression with L2 on TF-IDF + structured fields; mandatory fairness audit
- **Primary metric:** Recall at fixed precision; demographic parity
- **Watch-out:** historical hiring data encodes historical bias — Amazon's 2018 case is the canonical warning
- **Status:** [planned]

### 44. Parole / Recidivism Risk
- **Archetype:** binary classification, severely regulated, ethical landmines
- **Brief signal:** "recidivism", "risk score", criminal justice
- **Data shape:** 10K–500K cases, 20–50 features
- **Recommended approach:** Logistic Regression with L2 only (interpretability is a legal requirement); fairness audit non-negotiable
- **Primary metric:** AUC + group-wise calibration; false-positive parity
- **Watch-out:** COMPAS controversy — historical data reflects systemic bias; consider whether the use case is ethical at all
- **Status:** [planned]

### 45. Medical Diagnosis (Disease Detection)
- **Archetype:** binary or multiclass, regulated, real-time, mild-to-severe imbalance
- **Brief signal:** "disease", "diagnosis", medical imaging or labs
- **Data shape:** 5K–500K patient records, 20–200 features
- **Recommended approach:** Logistic Regression with L2 + SHAP; ensemble challenger; calibrated probabilities
- **Primary metric:** Sensitivity at fixed specificity; per-disease AUC
- **Watch-out:** "patient dies" outcome means FN cost ≫ FP cost — threshold tune accordingly
- **Status:** [planned]

### 46. Insurance Underwriting
- **Archetype:** binary or multiclass tier, regulated, batch
- **Brief signal:** "underwriting", "policy approval", actuarial
- **Data shape:** 50K–1M applications, 30–80 features
- **Recommended approach:** Logistic Regression Softmax (multi-tier); XGBoost with monotonic constraints + SHAP
- **Primary metric:** AUC; calibration; per-state fairness (insurance is state-regulated in the US)
- **Watch-out:** state-specific rules (some prohibit credit-based features); model needs per-state variants
- **Status:** [planned]

---

## Category 8 — Real-Time / Low-Latency

Inference budget < 100ms (often < 10ms). Big trees and deep KNN are eliminated.

### 47. Real-Time Fraud Scoring
- **Archetype:** binary, severe imbalance, < 50ms latency
- **Brief signal:** "real-time", "transaction approval", "100ms"
- **Data shape:** 1M–1B transactions, 30–100 features (must be precomputable)
- **Recommended approach:** LightGBM with limited tree depth (8–10), feature precomputation in feature store
- **Primary metric:** PR-AUC; p99 latency
- **Watch-out:** real-time features (last-5-transaction velocity) require streaming infrastructure
- **Status:** [planned]

### 48. Real-Time Bidding (RTB) Click Prediction
- **Archetype:** binary, severe imbalance, < 10ms latency, billions of impressions
- **Brief signal:** "RTB", "ad bidding", millisecond auction
- **Data shape:** 1B+ impressions, 50–200 features (mostly hashed)
- **Recommended approach:** Logistic Regression with FTRL (online learning) + feature hashing
- **Primary metric:** Log loss (calibration drives bidding); AUC
- **Watch-out:** every ms of latency = lost ad spend; budget the model to fit inference budget
- **Status:** [planned]

### 49. Network Intrusion Real-Time Alert
- **Archetype:** binary or multiclass, severe imbalance, < 100ms latency
- **Brief signal:** "live SIEM", "alerting"
- **Data shape:** 10K–10M connections/sec, 30–80 features
- **Recommended approach:** Random Forest with bounded depth or single calibrated Decision Tree
- **Primary metric:** Recall at fixed alert volume; p99 latency
- **Watch-out:** alert fatigue — analysts ignore high-FP feeds
- **Status:** [planned]

### 50. Real-Time Content Moderation
- **Archetype:** binary or multilabel, < 100ms latency
- **Brief signal:** "live moderation", "publish flow"
- **Data shape:** 10K–10M posts/min
- **Recommended approach:** Logistic Regression with L2 on TF-IDF (ms-fast); transformer for borderline cases (escalation tier)
- **Primary metric:** Precision at fixed throughput; p99 latency
- **Watch-out:** two-tier system (fast cheap classifier + slow expensive escalation) is the standard pattern
- **Status:** [planned]

### 51. Real-Time Recommendation Click Classification
- **Archetype:** binary, mild imbalance, < 50ms latency
- **Brief signal:** "click prediction", "recommendation", real-time
- **Data shape:** 1M–10B impressions, 30–100 features
- **Recommended approach:** Factorization Machines or shallow LightGBM; precomputed user/item embeddings
- **Primary metric:** AUC; log loss
- **Watch-out:** cold-start (new users/items) needs a fallback model
- **Status:** [planned]

---

## Category 9 — Small-Data Classification

< 1,000 rows. Heavy regularization, simple models, tiny CV folds.

### 52. Rare Disease Diagnosis
- **Archetype:** binary or multiclass, very small data, severe imbalance
- **Brief signal:** "rare disease", "<100 cases", clinical
- **Data shape:** 100–5,000 patients, 20–100 features
- **Recommended approach:** Logistic Regression with L1, leave-one-out CV; SVM with linear kernel as alternative
- **Primary metric:** Sensitivity (with confidence interval); per-fold variance
- **Watch-out:** report uncertainty; with 50 cases your AUC could be ±0.1
- **Status:** [planned]

### 53. Survey Response Classification
- **Archetype:** multiclass, < 2,000 responses, ordinal
- **Brief signal:** "survey", "Likert", small N
- **Data shape:** 200–2,000 responses, 10–50 features
- **Recommended approach:** Ordinal Logistic Regression; Random Forest with limited depth
- **Primary metric:** Quadratic-weighted kappa
- **Watch-out:** response bias — opt-in surveys skew positive
- **Status:** [planned]

### 54. Niche-Product Repeat-Purchase Prediction
- **Archetype:** binary, small data, mild imbalance
- **Brief signal:** "specialty", "niche", small customer base
- **Data shape:** 200–5,000 customers, 10–30 features
- **Recommended approach:** Logistic Regression with L1; Decision Tree for interpretability
- **Primary metric:** Recall at high precision (small marketing budget)
- **Watch-out:** look-alike features more useful than past purchases when N is tiny
- **Status:** [planned]

### 55. Startup Funding Outcome Classification
- **Archetype:** binary or multiclass (no funding / seed / Series A+), small data, imbalanced
- **Brief signal:** "funding", "investor screen"
- **Data shape:** 100–5,000 startups, 20–80 features
- **Recommended approach:** Logistic Regression with L1; simple Decision Tree as a sanity check
- **Primary metric:** Per-class recall; ROC-AUC
- **Watch-out:** survivorship bias — failed startups disappear from datasets
- **Status:** [planned]

### 56. Lab Experiment Outcome Classification
- **Archetype:** binary, very small data
- **Brief signal:** "experiment", "trial", < 500 runs
- **Data shape:** 50–500 experiments, 5–30 features
- **Recommended approach:** Logistic Regression; report effect sizes with confidence intervals
- **Primary metric:** Coefficient confidence intervals; AUC if there are enough rows
- **Watch-out:** with N=50, almost any model will look good in CV — Bayesian methods help
- **Status:** [planned]

---

## Category 10 — Mixed-Type Tabular

Numeric + categorical + missing values. Pipelines (`ColumnTransformer`) are mandatory.

### 57. Hospital Readmission Risk
- **Archetype:** binary, moderate imbalance (~11%), regulated, mixed types
- **Brief signal:** "30-day readmission", HIPAA
- **Data shape:** 100K–1M discharges, 50–80 features
- **Recommended approach:** Logistic Regression with L2 + SHAP; XGBoost with `enable_categorical=True`
- **Primary metric:** AUC; per-DRG fairness
- **Watch-out:** DRG codes are high cardinality — target encoding, not one-hot
- **Status:** [planned]

### 58. Mortgage Default
- **Archetype:** binary, moderate imbalance, regulated, mixed types
- **Brief signal:** "mortgage", "default", regulated
- **Data shape:** 100K–10M loans, 30–80 features
- **Recommended approach:** Logistic Regression with L2; XGBoost with monotonic constraints
- **Primary metric:** AUC; KS; calibration
- **Watch-out:** time-based split mandatory — economic regime changes
- **Status:** [planned]

### 59. Customer Upsell Probability
- **Archetype:** binary, mild imbalance, mixed types
- **Brief signal:** "upsell", "cross-sell", offer-eligibility
- **Data shape:** 50K–10M customers, 30–80 features
- **Recommended approach:** Uplift modeling (causal) preferred over plain classification; LightGBM as baseline
- **Primary metric:** AUUC (uplift) or top-decile lift; offer ROI
- **Watch-out:** "would have bought anyway" overlap dilutes lift — measure incremental, not absolute
- **Status:** [planned]

### 60. Repeat-Purchase Prediction
- **Archetype:** binary, mild-to-moderate imbalance, mixed types
- **Brief signal:** "repeat customer", "RFM"
- **Data shape:** 100K–10M customers, 20–60 features
- **Recommended approach:** XGBoost or LightGBM; RFM-engineered features (Recency × Frequency × Monetary)
- **Primary metric:** AUC; precision at top-k
- **Watch-out:** customer dormancy patterns vary by industry — e-commerce ≠ subscription
- **Status:** [planned]

### 61. Employee Attrition (6-month risk)
- **Archetype:** binary, moderate imbalance (~9%), small-to-medium data
- **Brief signal:** "turnover", "attrition", HR
- **Data shape:** 1K–50K employees, 50–80 features
- **Recommended approach:** Logistic Regression with L1 (auto feature selection); Random Forest challenger
- **Primary metric:** Recall at fixed precision (HR retention budget); SHAP for top drivers
- **Watch-out:** survivorship — employees still active aren't "non-attriters", they're censored
- **Deep dive:** [08_employee_attrition.md](08_employee_attrition.md)

---

## Category 11 — Time-Series Event Classification

Predict an event window from temporal sensor or log streams.

### 62. EEG Seizure Event Detection
- **Archetype:** binary, severe imbalance, sliding-window time-series
- **Brief signal:** "seizure", "event detection", EEG
- **Data shape:** 1K–100K windows × 100+ features (statistical, frequency-domain)
- **Recommended approach:** Random Forest or LightGBM on engineered window features; CNN-LSTM in production
- **Primary metric:** Per-event sensitivity; false alarms per hour
- **Watch-out:** subject-level CV mandatory — random split leaks across same patient
- **Status:** [planned]

### 63. ECG Arrhythmia Classification
- **Archetype:** multiclass, 5–10 arrhythmia types, imbalanced
- **Brief signal:** "ECG", "arrhythmia", "rhythm"
- **Data shape:** 10K–1M beats × 50–500 features
- **Recommended approach:** Random Forest on engineered ECG features; deep learning in production
- **Primary metric:** Per-class recall; macro-F1
- **Watch-out:** patient-level CV; arrhythmias are correlated within a patient
- **Status:** [planned]

### 64. Equipment Failure Window Prediction
- **Archetype:** binary, severe imbalance, sliding-window
- **Brief signal:** "predict failure in next N hours", PdM
- **Data shape:** 1K–1M windows × 50–500 sensor features
- **Recommended approach:** LightGBM with `is_unbalance`; class_weight tuned to maintenance cost
- **Primary metric:** Recall at fixed precision (maintenance crew capacity)
- **Watch-out:** many "failures" are recovered by automation — labels need careful definition
- **Status:** [planned]

### 65. Network Attack Window Detection
- **Archetype:** multiclass, sliding-window from packet streams
- **Brief signal:** "attack window", "exfil", multi-stage attack
- **Data shape:** 100K–10M windows × 30–80 features
- **Recommended approach:** Random Forest or LightGBM; HMM for sequential patterns
- **Primary metric:** Per-attack-type recall; analyst alert budget
- **Watch-out:** attackers move slowly — windows must overlap multiple hours
- **Status:** [planned]

### 66. Manufacturing Run-State Classification
- **Archetype:** multiclass (startup / steady / wind-down / fault), balanced
- **Brief signal:** "run state", "process state"
- **Data shape:** 10K–1M windows × 50–200 sensor features
- **Recommended approach:** Random Forest or HMM
- **Primary metric:** Per-state recall; transition-time delay
- **Watch-out:** state transitions are smooth, not crisp — boundary windows are inherently ambiguous
- **Status:** [planned]

---

## Category 12 — Hierarchical Classification

Class taxonomy with parent / child relationships. Models must respect the hierarchy.

### 67. E-commerce Product Taxonomy
- **Archetype:** hierarchical multiclass, 3–5 levels, 100–10K leaf classes
- **Brief signal:** "category > subcategory > product"
- **Data shape:** 1M–100M products × 1K+ features (text + image embeddings)
- **Recommended approach:** Per-level Logistic Regression Softmax; flat model with hierarchy-aware loss
- **Primary metric:** Per-level top-k accuracy
- **Watch-out:** hierarchy violations (e.g., "Electronics > Books") are unacceptable — hard-constrain decoding
- **Status:** [planned]

### 68. Bird / Animal Species Classification
- **Archetype:** hierarchical multiclass, 3–4 taxonomic levels, 100–10K leaf species
- **Brief signal:** "species", "taxonomy", iNaturalist-style
- **Data shape:** 100K–10M observations × 1K+ features (image embeddings + location)
- **Recommended approach:** Hierarchical Logistic Regression on embeddings; deep learning in production
- **Primary metric:** Top-1 species; top-5; per-genus accuracy
- **Watch-out:** geographic bias — species distributions vary by region
- **Status:** [planned]

### 69. ICD-10 Disease Code Assignment
- **Archetype:** hierarchical multilabel, 3–5 levels, 70K+ leaf codes
- **Brief signal:** "ICD-10", "diagnosis codes"
- **Data shape:** 100K–100M visits × free text + structured features
- **Recommended approach:** Per-chapter Logistic Regression; transformer fine-tuning in production
- **Primary metric:** Per-chapter F1; root-to-leaf path accuracy
- **Watch-out:** label sparsity — many leaf codes have < 10 examples
- **Status:** [planned]

---

## Category 13 — Cost-Sensitive / Asymmetric Loss

FN cost ≫ FP cost (or vice versa). Threshold tuning to a cost function is the central act.

### 70. Security Alert Triage (FN catastrophic)
- **Archetype:** binary, severe imbalance, FN cost ≫ FP cost
- **Brief signal:** "missed breach = $1M+ cost"
- **Data shape:** 100K–10M events × 50+ features
- **Recommended approach:** XGBoost; threshold = `argmin(cost)` from PR curve
- **Primary metric:** Expected cost per scoring window
- **Watch-out:** cost ratio shifts when business changes — re-tune threshold quarterly
- **Status:** [planned]

### 71. Cancer Screening (FN catastrophic)
- **Archetype:** binary, regulated, FN cost ≫ FP cost
- **Brief signal:** "screening", "early detection", "cancer"
- **Data shape:** 10K–1M patients × 30–200 features
- **Recommended approach:** Logistic Regression with L2 + SHAP; threshold tuned for sensitivity at fixed specificity
- **Primary metric:** Sensitivity at 95%+ specificity
- **Watch-out:** false negatives cost lives; false positives cost biopsies — get clinician input on the tradeoff
- **Status:** [planned]

### 72. Fraud Cost-Aware Threshold
- **Archetype:** binary, severe imbalance, explicit cost matrix
- **Brief signal:** "missed fraud costs $X, false alarm costs $Y"
- **Data shape:** 1M–1B transactions × 30–60 features
- **Recommended approach:** XGBoost; threshold = `argmin(FN×cost_FN + FP×cost_FP)`
- **Primary metric:** Total expected dollar cost
- **Watch-out:** static cost ratio is wrong — costs vary by transaction amount; weight per-row
- **Status:** [planned]

---

## Category 14 — Calibrated Probability Output

Need well-calibrated probabilities, not just labels (downstream business logic uses the probability directly).

### 73. Loan Risk Scoring (FICO-style)
- **Archetype:** binary, calibrated probability, regulated
- **Brief signal:** "risk score", "calibrated probability"
- **Data shape:** 100K–10M loans × 30–60 features
- **Recommended approach:** Logistic Regression (natively calibrated) or XGBoost with Platt scaling / isotonic regression
- **Primary metric:** Brier score; calibration plot; expected calibration error
- **Watch-out:** XGBoost probabilities are NOT calibrated by default — apply Platt or isotonic on a held-out set
- **Status:** [planned]

### 74. Sports Betting Odds (Calibrated)
- **Archetype:** binary, calibrated probability, public-facing
- **Brief signal:** "odds", "win probability", betting
- **Data shape:** 1K–1M games × 20–80 features
- **Recommended approach:** Logistic Regression baseline; ensemble + isotonic recalibration
- **Primary metric:** Log loss; Brier score; profit-at-Kelly-criterion
- **Watch-out:** public lines are competitive — beating the market by 1% is hard
- **Status:** [planned]

### 75. Weather Rain Probability
- **Archetype:** binary or multiclass, calibrated probability, public-facing
- **Brief signal:** "X% chance of rain"
- **Data shape:** 10K–10M observations × 30–80 features
- **Recommended approach:** Logistic Regression Softmax + isotonic recalibration; ensemble with NWP model output
- **Primary metric:** Brier score; reliability diagram
- **Watch-out:** "30% chance of rain" must mean rain happens 30% of the time on those days — calibration is the spec, not a nice-to-have
- **Status:** [planned]

---

## Category 15 — Long-Tail / Rare-Class

One class is 50%+, dozens of classes are < 1% each. Long-tail evaluation matters more than overall accuracy.

### 76. Rare Exam Findings (Radiology)
- **Archetype:** multilabel binary, long-tail (a few common findings, 100+ rare ones)
- **Brief signal:** "rare findings", "incidentaloma"
- **Data shape:** 100K–10M exams × 1K+ features
- **Recommended approach:** Per-finding Logistic Regression with L2 + per-class threshold; freeze rare-finding training when data is too sparse
- **Primary metric:** Per-finding AUC; macro-AUC weighted toward rare findings
- **Watch-out:** rare finding with 5 examples is unlearnable — keep an "alert radiologist" path
- **Status:** [planned]

### 77. Long-Tail E-commerce Category
- **Archetype:** multiclass, very many classes, head/tail split (90% of volume in 10% of categories)
- **Brief signal:** "long-tail", "marketplace categories"
- **Data shape:** 10M–10B products × 1K+ features
- **Recommended approach:** Two-stage: head classifier (top categories) + tail classifier (KNN over embeddings for remainder)
- **Primary metric:** Top-1 accuracy on head; recall on tail
- **Watch-out:** revenue concentration in head — but tail is where new growth comes from
- **Status:** [planned]

### 78. Rare Animal Species Identification
- **Archetype:** multiclass, very many classes, severe long-tail (many species with < 10 training images)
- **Brief signal:** "species ID", "biodiversity"
- **Data shape:** 100K–10M observations × 1K+ embedding features
- **Recommended approach:** Logistic Regression Softmax for top species; nearest-neighbor on embeddings for tail
- **Primary metric:** Top-1 head accuracy; top-5 tail recall
- **Watch-out:** observation bias — common species over-represented in citizen-science datasets
- **Status:** [planned]

---

## Where to Go Next

- Read the [methodology guide](../from_brief_to_solution/METHODOLOGY.md) to map a fresh brief into this catalog.
- Refer to [MODEL_SELECTION_GUIDE_CLASSIFICATION.md](../../../MODEL_SELECTION_GUIDE_CLASSIFICATION.md) for the 8 framing questions used inside each scenario.
- Refer to [ML_PIPELINE_GUIDE.md](../../../ML_PIPELINE_GUIDE.md) for the 12-phase pipeline every scenario instantiates.
