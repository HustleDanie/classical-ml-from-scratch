# Expert Scenario 11: Titanic Survival Prediction

> **Complexity:** Small messy dataset, mixed types, heavy missing values, hidden interaction features (sex × class), survivorship bias absent (we have full ground truth) — but the dataset is famously easy to overfit.

---

## The Brief

A maritime history museum wants a predictive model that explains, given a passenger's demographic and ticket information, whether they survived the 1912 Titanic disaster. The goal is not just accuracy — they want a model whose decisions can be explained on a museum exhibit panel ("women in first class survived 96% of the time...").

You receive 891 labeled passenger records and a held-out 418 unlabeled records (the classic Kaggle competition split). Target balance: 38% survived. The data is famously messy — Age has 20% missing values, Cabin has 77% missing, ticket numbers have multiple formats.

This is harder than it looks because: 891 rows is not a lot, the strongest features (Sex × Pclass interaction) require manual engineering, missing values are sometimes informative (no Cabin entry = lower-class passenger), and the natural overfitting pressure is enormous.

---

## Step 1: Define the Problem Type

```
Type:           Binary Classification (Survived = 1, Not Survived = 0)
Primary Metric: Accuracy (data is balanced enough; museum-friendly metric)
Secondary:      F1; per-group recall (women, men, by class)
Business Goal:  Interpretable model for museum panel + Kaggle leaderboard score
Constraint:     Must be interpretable (no SHAP-only outputs)
Imbalance:      62/38 (mild, no special handling)
```

**Expert thinking:** Titanic is a "balanced enough" dataset. Accuracy is fine here. The constraint that makes this scenario non-trivial is the *interpretability* — a museum panel requires the model to be explainable in plain English, which steers us toward Logistic Regression or a Decision Tree as the primary, with ensemble models as challengers we DON'T ultimately deploy.

---

## Step 2: Understand the Data

```
Shape: 891 rows x 12 columns

Features received:
- PassengerId  (int, unique)
- Survived     (int, target: 0 or 1)
- Pclass       (int, 1/2/3 -- ticket class)
- Name         (string, e.g. "Braund, Mr. Owen Harris")
- Sex          (string, male/female)
- Age          (float, 0.42-80, 20% missing)
- SibSp        (int, # siblings/spouses aboard)
- Parch        (int, # parents/children aboard)
- Ticket       (string, e.g. "A/5 21171" -- mixed format)
- Fare         (float, 0-512, 1 missing in test set)
- Cabin        (string, e.g. "C85" -- 77% missing)
- Embarked     (string, S/C/Q port codes, 2 missing)
```

**Expert thinking:** I see immediately:
- **Name** is not just a string — it contains the title (Mr/Mrs/Miss/Master/Rev/Dr) which is a HUGE signal (Master = young boy; Rev = clergy with very different survival).
- **Cabin** at 77% missing is tempting to drop — but missing means "we don't have a recorded cabin" which strongly correlates with steerage class.
- **Ticket** looks useless but groups of passengers shared tickets — a feature for "traveled with companions" is hidden inside.
- **Age** missing 20% — not random; correlates with passengers without family ties.

---

## Step 3: Exploratory Data Analysis (EDA)

```python
df['Survived'].value_counts(normalize=True)
# 0    0.6162
# 1    0.3838

# The famous insight: Sex is THE strongest feature
df.groupby('Sex')['Survived'].mean()
# female    0.742
# male      0.189

# But it interacts strongly with Pclass:
df.groupby(['Sex', 'Pclass'])['Survived'].mean()
# female  1    0.968
# female  2    0.921
# female  3    0.500
# male    1    0.369
# male    2    0.157
# male    3    0.135

# 1st-class women: 97% survived. 3rd-class men: 14% survived.
# A 7x survival gap.
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Sex × Pclass dominates everything else | This single interaction explains ~75% of survival variance |
| Cabin missing → 76% died vs Cabin present → 33% died | Missingness IS the signal (no cabin = steerage) |
| Title in Name: Master (boys) → 57% survived; Mr → 16% | Title is a pseudo-age + role feature |
| Age missing rows had 29% survival (close to overall) | Age missingness is mostly random, not informative |
| Fare > $50 → 65% survival; Fare < $10 → 20% | Wealth proxy; correlated with Pclass |
| FamilySize = SibSp + Parch + 1: alone (1) → 30%, small (2-4) → 56%, large (5+) → 16% | Inverted U shape; large families couldn't move together |
| Embarked C → 55% survived; Embarked S → 34% | Embarkation point correlates with class composition |

**Expert insight:** the Sex × Pclass interaction is the dominant signal. If the model can't learn it, no amount of other features will save it. Linear models like Logistic Regression CANNOT learn interactions automatically — we MUST create the interaction feature manually. Tree models learn it for free.

---

## Step 4: Data Cleaning

```python
import numpy as np

# Cabin: 77% missing -- DON'T impute; encode missingness
df['HasCabin'] = df['Cabin'].notna().astype(int)
df['CabinDeck'] = df['Cabin'].str[0].fillna('Unknown')  # First letter = deck

# Drop the raw Cabin column (too high cardinality, mostly missing)
# Keep HasCabin and CabinDeck

# Age: 20% missing -- impute by Title group (better than median)
def extract_title(name):
    return name.split(',')[1].split('.')[0].strip()

df['Title'] = df['Name'].apply(extract_title)
# Map rare titles to common groups
title_map = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Capt': 'Officer', 'Col': 'Officer', 'Major': 'Officer',
    'Dr': 'Officer', 'Rev': 'Officer',
    'Lady': 'Royalty', 'Sir': 'Royalty', 'the Countess': 'Royalty',
    'Jonkheer': 'Royalty', 'Don': 'Royalty', 'Dona': 'Royalty'
}
df['Title'] = df['Title'].replace(title_map)
# Final categories: Mr, Mrs, Miss, Master, Officer, Royalty

# Now impute Age by the median age of each title group
df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

# Embarked: 2 missing values -- impute with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Fare: 1 missing in test set -- impute with median by Pclass
df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

# Drop columns we won't use further
df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
```

**Expert insight:** Imputing Age by the overall median gives 28 to everyone. Imputing by Title gives 4 to "Master", 22 to "Miss", 32 to "Mr", 36 to "Mrs", 49 to "Officer". The title-based imputation captures real signal — a "Master" with median 28 would corrupt the under-15 male survival rate.

---

## Step 5: Feature Engineering

```python
# === FAMILY SIZE ===
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['LargeFamily'] = (df['FamilySize'] >= 5).astype(int)

# === SEX x PCLASS INTERACTION (THE dominant feature) ===
df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
# 6 categories: female_1, female_2, female_3, male_1, male_2, male_3

# === AGE BANDS ===
df['AgeBand'] = pd.cut(
    df['Age'],
    bins=[0, 12, 18, 35, 60, 100],
    labels=['Child', 'Teen', 'YoungAdult', 'Adult', 'Senior']
)
# Children survived disproportionately ("women and children first")

# === FARE BANDS ===
df['FareBand'] = pd.qcut(df['Fare'], 4, labels=['VeryLow', 'Low', 'Medium', 'High'])
# Quartile-based -- robust to outliers

# === TITLE × IS_MARRIED proxy ===
df['IsMarried'] = (df['Title'] == 'Mrs').astype(int)
# Mrs in 1912 implied wife/mother; survived more than Miss in same class

# Final feature set: 14 features
# - Pclass, Sex, Age, SibSp, Parch, Fare, Embarked  (raw, kept some)
# - HasCabin, CabinDeck, Title, FamilySize, IsAlone, LargeFamily
# - Sex_Pclass, AgeBand, FareBand, IsMarried
```

**Expert insight:** The Sex_Pclass interaction is the most important engineered feature. For Logistic Regression, this interaction would otherwise be unlearnable. For tree models, it's redundant but doesn't hurt.

---

## Step 6: Feature Selection

```python
# Stage 1 -- already dropped: PassengerId, Name, Ticket, Cabin

# Stage 2 -- Mutual Information
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)

# Top features by MI:
#   Sex            0.21
#   Sex_Pclass     0.20
#   Title          0.19
#   Pclass         0.10
#   Fare           0.09
#   FareBand       0.09
#   HasCabin       0.07
#   FamilySize     0.04
#   Age            0.03
#   AgeBand        0.03
#   Embarked       0.02
#   IsAlone        0.01
#   SibSp          0.01
#   Parch          0.01
#   CabinDeck      0.01

# Sex, Sex_Pclass, Title carry 65% of total MI -- the "big three" survivors features

# Stage 3 -- Drop features with MI < 0.005
# All features clear that bar; keep all 16
```

---

## Step 7: Preprocessing

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
categorical = ['Sex', 'Pclass', 'Embarked', 'Title', 'CabinDeck',
               'Sex_Pclass', 'AgeBand', 'FareBand']
binary = ['HasCabin', 'IsAlone', 'LargeFamily', 'IsMarried']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('bin', 'passthrough', binary)
])

# Final feature count after one-hot: ~35 features
```

---

## Step 8: Train/Test Split

```python
# 891 rows -- precious. Use stratified 5-fold CV; reserve final 178 (20%) as a HOLD-OUT
# we never touch until the very end.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
# X_train: 712 rows; X_test: 179 rows
```

**Expert insight:** with 891 rows, every split decision matters. Stratify=y ensures both halves preserve the 62/38 class balance.

---

## Step 9: Baseline

```python
# Baseline 1: predict majority class (everyone died)
# Accuracy: 61.6%

# Baseline 2: simple rule -- women survive, men die
# Accuracy: 79% (!)
# This rule alone beats most over-engineered models
```

The "women survive" rule sets the bar at 79%. Anything below that is broken.

---

## Step 10: Try Multiple Models with Stratified 5-Fold CV

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

# Model 1: Logistic Regression
results['LR'] = cross_val_score(LogisticRegression(max_iter=500), X_train, y_train, cv=cv, scoring='accuracy').mean()
# 0.825

# Model 2: Decision Tree (max_depth=4 -- crucial for small data)
results['DT'] = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=42), X_train, y_train, cv=cv).mean()
# 0.819

# Model 3: Random Forest
results['RF'] = cross_val_score(RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42), X_train, y_train, cv=cv).mean()
# 0.831

# Model 4: Gradient Boosting
results['GB'] = cross_val_score(GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42), X_train, y_train, cv=cv).mean()
# 0.838

# Model 5: SVM with RBF
results['SVM'] = cross_val_score(SVC(C=1.0, kernel='rbf', random_state=42), X_train, y_train, cv=cv).mean()
# 0.821

# Model 6: KNN (k=5)
results['KNN'] = cross_val_score(KNeighborsClassifier(n_neighbors=5), X_train, y_train, cv=cv).mean()
# 0.792 -- worst, KNN suffers in mixed-type space even after one-hot

# Model 7: Gaussian Naive Bayes
results['GNB'] = cross_val_score(GaussianNB(), X_train, y_train, cv=cv).mean()
# 0.770 -- independence assumption violated by Sex_Pclass interaction

# Model 8: XGBoost
results['XGB'] = cross_val_score(XGBClassifier(n_estimators=200, max_depth=3, random_state=42), X_train, y_train, cv=cv).mean()
# 0.835
```

**Top performers:** Gradient Boosting (0.838), XGBoost (0.835), Random Forest (0.831).
**For interpretability:** Logistic Regression (0.825) and Decision Tree (0.819) — only 1-2% behind.

---

## Step 11: Imbalance Handling — Skip

Class balance is 62/38 — `class_weight='balanced'` doesn't help here. CV results above are without rebalancing.

---

## Step 12: Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Tune Gradient Boosting (highest CV score)
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [2, 3, 4],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1
)
grid.fit(X_train, y_train)
# Best: n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9
# Tuned CV: 0.844

# Tune Logistic Regression (interpretable challenger)
grid_lr = GridSearchCV(
    LogisticRegression(max_iter=500),
    {'C': [0.01, 0.1, 0.5, 1.0, 2.0, 10.0], 'penalty': ['l1', 'l2']},
    cv=5, scoring='accuracy', n_jobs=-1
)
grid_lr.fit(X_train, y_train)
# Best: C=1.0, penalty='l2'
# Tuned CV: 0.829
```

---

## Step 13: Interpretability Audit Before Choosing the Final Model

```python
# The museum requires interpretable. Compare:
#
# Gradient Boosting (CV 0.844): black box. Need SHAP. Each prediction = 200 trees voting.
# Decision Tree (CV 0.819, max_depth=4): fully transparent. 16 leaf rules.
# Logistic Regression (CV 0.829): fully transparent. ~35 coefficients.
#
# Trade-off: 1.5 percentage points of accuracy for full interpretability.
#
# DECISION: ship Logistic Regression as the primary museum model.
# Use Gradient Boosting only as a "challenger" for the Kaggle submission.
```

**Expert insight:** the 1.5% accuracy difference between Logistic Regression (0.829) and Gradient Boosting (0.844) is worth reading carefully. On 891 rows, that's ~13 passengers correctly classified. For a museum exhibit panel, 0 SHAP outputs is worth more than 13 marginal predictions. For Kaggle leaderboard, GB wins.

---

## Step 14: Stacking (Optional, Kaggle Only)

```python
from sklearn.ensemble import StackingClassifier

stack = StackingClassifier(
    estimators=[
        ('gb', GradientBoostingClassifier(n_estimators=200, max_depth=3)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=6)),
        ('lr', LogisticRegression(C=1.0, max_iter=500))
    ],
    final_estimator=LogisticRegression(max_iter=500),
    cv=5
)
# CV: 0.847 (+0.003 over single GB)
# Marginal gain, complexity not worth it for museum but useful for Kaggle
```

---

## Step 15: Final Evaluation on Held-Out Test Set

```python
# Final museum model: Logistic Regression with C=1.0, L2 penalty
# Final Kaggle model: Gradient Boosting

# Logistic Regression on test (179 rows):
#   Accuracy: 0.832
#   Female accuracy: 0.943
#   Male accuracy: 0.778
#   Confusion matrix:
#                 Predicted 0  Predicted 1
#   Actual 0         93           17
#   Actual 1         13           56

# Gradient Boosting on test:
#   Accuracy: 0.849
#   Female accuracy: 0.957
#   Male accuracy: 0.793

# Both models converge on similar per-group accuracy.
# The 1.7% accuracy difference comes from edge cases in the male / 3rd class group.
```

---

## Step 16: Explainability — The Museum Panel

The key deliverable is a museum panel. Logistic Regression coefficients tell the story:

```python
# Top coefficients (Logistic Regression, scaled to standardized features)
# Sex_Pclass = female_1     +2.85   "1st-class women: huge survival advantage"
# Sex_Pclass = female_2     +2.40
# Title = Master           +1.62   "Boys under 12: 'children first' worked"
# Sex_Pclass = female_3     +1.10   "3rd-class women: still survived more"
# HasCabin                  +0.51   "Having a recorded cabin"
# Fare                      +0.40   "Higher fare ~ better access to lifeboats"
# IsAlone                  -0.31   "Solo travelers had lower priority"
# Sex_Pclass = male_3      -1.85
# Sex_Pclass = male_2      -2.10
# Sex_Pclass = male_1      -1.50   "Even 1st-class men survived only 37%"

# Museum panel text (auto-generated from coefficients):
#   "Of all factors, sex and class together explained survival best.
#    1st-class women survived 96% of the time, while 3rd-class men survived only 14%.
#    Children (titled 'Master') had a similar boost regardless of class.
#    Solo travelers were less likely to survive than those with small families."
```

**Expert insight:** Logistic Regression is the only classical model that gives you literal English explanations. SHAP can produce something similar from Gradient Boosting, but a museum visitor doesn't want a SHAP plot — they want a sentence.

---

## Step 17: Deployment / Delivery

```python
# Model size: 5KB (tiny)
# Inference: < 1ms per passenger

# Deliverables:
#   1. Joblib pickle of fitted preprocessor + LR model
#   2. Static HTML museum panel with the top-10 coefficient bars
#   3. CSV submission for Kaggle (using GB challenger model)
#
# This is a one-shot artifact. No retraining, no monitoring needed.
# The museum exhibit is a static display.
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Title in Name | Drop Name column | Extract title -> 6-category feature |
| Cabin missing | Drop column (77% missing) | Encode HasCabin (missingness IS the signal) |
| Age imputation | Median over all rows | Median by Title group (Master=4, Mr=32, Mrs=36) |
| Sex × Pclass interaction | Hope the model finds it | Manually create Sex_Pclass feature |
| Fare/Age bands | Use raw continuous | qcut (Fare) and binned ranges (Age) for robustness |
| Train/test split | 80/20 random | 80/20 stratified by Survived |
| Model choice | Pick highest CV (GB at 0.844) | Pick LR (0.829) for interpretability — museum brief |
| Tuning | n_estimators=1000 | max_depth=3 with low n_estimators (small data!) |
| Imbalance | SMOTE | Skip — 62/38 doesn't need it |
| Explanation | "model said so" | LR coefficients turned into museum-panel sentences |
| Hold-out | None | 20% reserve never touched until final eval |
