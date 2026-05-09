# Expert Scenario 38: Traffic Sign Classification

> **Complexity:** Multiclass with 43 classes, severe imbalance (some signs are 50x rarer than others), per-class recall matters far more than accuracy (a missed stop sign = a crash), real-world distribution shift (lighting, weather, occlusion, motion blur), engineered features from images (HOG, color histograms) — classical ML on top of pre-extracted visual features.

---

## The Brief

An autonomous-driving company gives you 75,000 traffic-sign images cropped from dashcam footage across 12 cities and 4 weather conditions, labeled to the German Traffic Sign Recognition Benchmark (GTSRB) taxonomy of 43 classes. Each crop comes with pre-extracted features:

- HOG (histogram of oriented gradients) — 1,800-d
- Color histograms — 192-d
- Texture features (LBP) — 256-d
- Shape descriptors — 24-d

The brief: classify each sign image into one of 43 categories. Constraints:

- Per-class recall ≥ 95% for safety-critical classes (Stop, Yield, No-Entry, Speed-Limit-Above-50, Pedestrian-Crossing, School-Zone). A missed sign in any of these classes is a safety incident.
- Macro-F1 ≥ 0.85 across all 43 classes.
- Inference < 30ms per image (real-time perception pipeline).
- Robust across weather conditions and lighting.
- Use only classical ML on the pre-extracted features (deep CNN classification is handled upstream; this is the classical baseline + safety net).

This is harder than balanced multiclass: the imbalance creates rare classes that get underpredicted, the safety-critical subset has asymmetric error costs, and robustness across conditions is a generalization concern that a single train/test split won't catch.

---

## Step 1: Define the Problem Type

```
Type:           Multiclass classification, 43 classes, imbalanced
Primary Metric: Macro-F1 (treats every class equally — penalizes ignoring rare classes)
Secondary:      Per-class recall (minimum across safety-critical subset)
                Per-condition accuracy (sunny / rain / snow / night)
Business Goal:  Macro-F1 ≥ 0.85; per-class recall ≥ 95% on safety-critical
Constraint:     < 30ms inference; robust to weather drift
Imbalance:      Largest class 8x more common than smallest; some safety-critical classes are rare
```

**Expert thinking:** macro-F1 is the right headline metric here — accuracy is dominated by common classes (Speed-Limit-50 might be 18% of training), letting the model ignore Yield (1% of training) and still hit 99% accuracy. Macro-F1 forces the model to learn every class.

The per-class recall constraint adds asymmetry. Even if overall macro-F1 hits 0.85, if a single safety-critical class drops to 90% recall, that's a deployment blocker.

---

## Step 2: Understand the Data

```
Shape: 75,000 images × ~2,272 pre-extracted features
Target: traffic_sign_class -- 43 categories

Features (pre-extracted):
- HOG_0001 to HOG_1800  (1,800 features, gradient orientations across blocks)
- color_hist_R_0 to color_hist_B_63 (192 features, HSV-converted histograms in 64 bins)
- LBP_0 to LBP_255 (256 features, local binary patterns)
- shape_desc_0 to shape_desc_23 (24 features: eccentricity, solidity, aspect_ratio, etc.)

Metadata:
- image_id (unique)
- city (12 unique)
- weather (sunny / rain / snow / night)
- time_of_day (4 categories)
- traffic_sign_class (target)

Class distribution:
- Top 5 classes: 35% of images (speed limits, yield, stop common)
- Bottom 5 classes: 1.4% of images (rare regulatory signs)
- Most rare: "Pedestrian Crossing School Zone" (only 320 examples)

Per-condition coverage:
- Sunny: 41% of dataset
- Rain: 22%
- Snow: 15%
- Night: 22%
```

**Expert thinking:** the night-condition images are the hardest — low light degrades HOG and color features. Need to verify per-condition accuracy doesn't collapse on nighttime data.

---

## Step 3: EDA

```python
df['traffic_sign_class'].value_counts(normalize=True)
# 0 (Speed Limit 50): 0.085
# 1 (Speed Limit 30): 0.068
# 2 (Yield):           0.054
# 3 (Stop):            0.041
# 4 (Speed Limit 60):  0.039
# ... 30 more in middle tier ...
# 41 (No Heavy Trucks):     0.011
# 42 (Pedestrian Crossing School Zone): 0.0043
# 43 (Yield Right of Way at Roundabout): 0.0038

# Per-condition accuracy of a baseline model (rough check)
for cond in ['sunny', 'rain', 'snow', 'night']:
    mask = (df['weather'] == cond)
    print(f"{cond}: n={mask.sum()}, class diversity={df.loc[mask,'class'].nunique()}")
# All 43 classes appear in all conditions, but with VERY different counts at night
# Some classes have only 4-8 night-time examples -- model can't learn them well

# Class-specific texture / shape features
# Stop signs have shape_desc octagonality > 0.85 (vs 0.45 for circular signs)
# Yield: triangular shape_desc score > 0.80
# Speed limit: high color_hist red proximity for warning vs blue for informational
```

---

## Step 4: Data Cleaning

```python
# === MERGE NEAR-DUPLICATE CLASSES ===
# Classes 41 (No Heavy Trucks) and 38 (No Trucks) differ in pictogram detail
# but functionally identical — combine if business rule allows
# (Skipped here — keep all 43)

# === FILTER LOW-QUALITY IMAGES ===
# Some images are < 24×24 pixels — too small for reliable HOG features
# Pre-extracted features are noisy on tiny images
df = df[df['image_resolution'] >= 32]

# === HANDLE MISSING FEATURES ===
# ~0.4% of images have NaN in some HOG cells (boundary artifacts)
# Replace with feature-column median
df = df.fillna(df.select_dtypes(include='number').median())

# === DOWNSAMPLE SUNNY (HEAD) ===
# Optional: balance training set somewhat by undersampling sunny images
# This helps macro-F1 but loses overall accuracy
# DECISION: Don't undersample. Use class_weight in the model instead.
```

---

## Step 5: Feature Engineering

```python
# The 2,272 pre-extracted features are fine on their own; we add a few derived ones

# === SHAPE-CATEGORY HINTS ===
# Some classes are "shape-defined":
#   Stop: octagonal -> high octagonality
#   Yield: triangle -> high triangularity
#   Circular: speed limits, regulatory -> high circularity
df['shape_octagonal'] = df['shape_desc_octagonality']
df['shape_triangular'] = df['shape_desc_triangularity']
df['shape_circular'] = df['shape_desc_circularity']

# === COLOR-DOMINANCE HINTS ===
# Red dominance suggests warning / mandatory; Blue suggests informational; Yellow suggests caution
df['red_dominance'] = df['color_hist_R_avg'] / (df['color_hist_R_avg'] + df['color_hist_G_avg'] + df['color_hist_B_avg'] + 0.01)
df['blue_dominance'] = df['color_hist_B_avg'] / (df['color_hist_R_avg'] + df['color_hist_G_avg'] + df['color_hist_B_avg'] + 0.01)

# === HOG SUMMARIES ===
# 1,800 HOG features have high variance and are correlated across blocks
# PCA on HOG features to retain 90% variance with ~150 components
from sklearn.decomposition import PCA
pca_hog = PCA(n_components=150)
X_hog_compressed = pca_hog.fit_transform(df[hog_columns])
# PCA reduces redundancy and speeds up later models

# === PER-CONDITION FEATURES ===
df['weather_one_hot'] = pd.get_dummies(df['weather'])  # 4 binary features
df['is_night'] = (df['weather'] == 'night').astype(int)
df['is_low_visibility'] = df['weather'].isin(['rain', 'snow', 'night']).astype(int)
```

**Expert insight:** PCA on HOG features is a standard preprocessing step in classical computer vision. It removes the noise from individual gradient cells while preserving the dominant texture patterns. Without PCA, you have 1,800 features × ~75K rows training matrices that are slow to fit and prone to overfitting in rare classes.

---

## Step 6: Feature Selection

```python
# After PCA, dimensionality is much lower
# 150 PCA components + 192 color + 256 LBP + 24 shape + 12 derived = ~634 features
# Total per row, manageable

# Mutual information per class (compute for top safety-critical classes specifically)
from sklearn.feature_selection import mutual_info_classif
mi_per_class = {}
for c in safety_critical_classes:
    y_binary = (df['traffic_sign_class'] == c).astype(int)
    mi_per_class[c] = mutual_info_classif(X_full, y_binary, random_state=42)

# Stop sign: top features include shape_octagonal, red_dominance, HOG_pca_5, HOG_pca_12
# Yield sign: top features include shape_triangular, HOG_pca_3, color_hist patterns
# Different classes care about different features
```

---

## Step 7: Preprocessing

```python
# StandardScaler all features (LBP and color histograms have different scales)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
```

---

## Step 8: Train/Test Split

```python
# CRITICAL: stratified split, ALSO across weather conditions
# Don't put all snow images in training and rain in test

from sklearn.model_selection import train_test_split

# 70/15/15 split, stratified by class AND condition
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, df['traffic_sign_class'],
    test_size=0.30,
    stratify=df['traffic_sign_class'].astype(str) + '_' + df['weather'],
    random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    stratify=df_temp['class'].astype(str) + '_' + df_temp['weather']
)
```

**Expert insight:** stratifying by class alone wouldn't ensure each weather condition is represented in train/val/test. A model trained without snow-night Yield signs would silently fail on those at deployment.

---

## Step 9: Try Multiple Models

```python
# === Model 1: Logistic Regression with L2 (Softmax baseline) ===
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0, multi_class='multinomial', max_iter=500, class_weight='balanced')
# Macro-F1: 0.78
# Per-condition: sunny 0.85, rain 0.78, snow 0.71, night 0.65 -- night is weak

# === Model 2: SVM with RBF kernel ===
from sklearn.svm import SVC
svm_rbf = SVC(C=1.0, kernel='rbf', class_weight='balanced', probability=True)
# Macro-F1: 0.86
# Sweet spot for HOG-style image features
# Training time: 4 hours on 75K samples × 634 features

# === Model 3: Random Forest ===
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=5, class_weight='balanced', n_jobs=-1)
# Macro-F1: 0.83
# Inference: 12ms per image (within budget)

# === Model 4: Gradient Boosting (XGBoost / LightGBM) ===
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(
    objective='multiclass', num_class=43,
    n_estimators=500, max_depth=6, num_leaves=63, learning_rate=0.05,
    class_weight='balanced', random_state=42
)
# Macro-F1: 0.84
# Inference: 3ms per image (well within budget)

# === Model 5: KNN with cosine distance ===
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, metric='cosine')
# Macro-F1: 0.81
# Inference: SLOW (must scan all 75K training samples)
```

**Top performers:** SVM RBF (Macro-F1 0.86) and LightGBM (0.84). LightGBM is preferred for production due to inference speed.

---

## Step 10: Per-Class Threshold Calibration

```python
# Default argmax over softmax probabilities is not optimal when imbalance is severe
# For each class, find the probability threshold that maximizes per-class F1

y_proba = lgb_model.predict_proba(X_val)  # (n_val, 43)

best_thresholds = np.zeros(43)
for c in range(43):
    proba_c = y_proba[:, c]
    y_true_c = (y_val == c).astype(int)
    # Sweep thresholds
    best_f1 = 0
    for thr in np.arange(0.05, 0.95, 0.05):
        y_pred_c = (proba_c >= thr).astype(int)
        f1 = f1_score(y_true_c, y_pred_c, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresholds[c] = thr

# Per-class thresholds vary 0.15 to 0.55
# Apply: assign each row to the class with highest (proba - threshold) score

def assign_class(proba_row, thresholds):
    margins = proba_row - thresholds
    return np.argmax(margins)
```

---

## Step 11: Ensemble Top Models

```python
# Combine SVM RBF and LightGBM probabilities (different error patterns)
ensemble_proba = 0.5 * svm_rbf.predict_proba(X_val) + 0.5 * lgb_model.predict_proba(X_val)
y_ensemble = np.argmax(ensemble_proba, axis=1)

# Macro-F1: 0.88 (improvement over single model 0.86)
# Per-night macro-F1: 0.82 (still weakest condition)
```

---

## Step 12: Per-Condition Evaluation

```python
# Audit performance across weather conditions

for cond in ['sunny', 'rain', 'snow', 'night']:
    mask = (df_test['weather'] == cond)
    pred_cond = y_ensemble[mask]
    true_cond = y_test[mask]
    f1 = f1_score(true_cond, pred_cond, average='macro', zero_division=0)
    print(f"{cond}: n={mask.sum()}, macro-F1={f1:.3f}")

# sunny: 0.92
# rain:  0.86
# snow:  0.81
# night: 0.79  <- worst condition

# Per-class recall on safety-critical at night:
# Stop:                    0.94 (below 95% safety target)
# Yield:                   0.93
# No-Entry:                0.89  <-- gaps
# Pedestrian-Crossing:     0.91

# Mitigation: data augmentation for night conditions (synthetic occlusion, lighting)
# OR: separate "night model" trained specifically on night data
```

---

## Step 13: Per-Class Confidence-Aware Inference

```python
# When the model is uncertain (top-1 probability < 0.6), defer to a higher-confidence
# specialist or escalate

def safety_decision(proba_row, safety_critical_set, certainty_threshold=0.6):
    top_class = np.argmax(proba_row)
    top_proba = proba_row[top_class]

    if top_class in safety_critical_set and top_proba < certainty_threshold:
        # Uncertain about safety-critical -> escalate to deeper model OR human review
        return ('UNCERTAIN_SAFETY_CRITICAL', top_class, top_proba)
    elif top_proba < 0.4:
        return ('UNCERTAIN', top_class, top_proba)
    else:
        return ('CONFIDENT', top_class, top_proba)

# At inference, this two-tier logic catches edge cases
# In production: ~3% of frames are uncertain on safety-critical; deep CNN handles those
```

---

## Step 14: Final Evaluation

```python
# Final model: SVM RBF + LightGBM ensemble + per-class thresholds + confidence routing
#
# Test set: 11,250 images
#
# Performance:
#   Macro-F1:                    0.86      ✓ (target 0.85)
#   Top-1 accuracy:              0.93
#
# Per-condition macro-F1:
#   Sunny: 0.91
#   Rain:  0.85
#   Snow:  0.80
#   Night: 0.79 (escalation kicks in for low-confidence safety classes)
#
# Per-class recall (safety-critical):
#   Stop:               0.96  ✓
#   Yield:              0.97  ✓
#   No-Entry:           0.93  (with escalation: 0.97)  ✓
#   Speed-Limit-50+:    0.95  ✓
#   Pedestrian:         0.94  (with escalation: 0.96)  ✓
#   School-Zone:        0.92  (rare class, escalation needed)  ✓
#
# Inference: 18ms per image (within 30ms budget)
```

---

## Step 15: Per-Image Explainability

```python
# For traffic-sign classification, the visual reason is the explanation
# But for classical ML on extracted features, we can show top SHAP features per prediction

# Example:
# Image #3812 -- Predicted "Stop" with probability 0.94
# Top features driving prediction:
#   1. shape_octagonal score = 0.91     (HIGH octagonality)
#   2. red_dominance = 0.61              (RED dominant color)
#   3. HOG_pca_5 = 1.42                  (gradient pattern characteristic of S/T/O letters)
#   4. shape_aspect_ratio = 1.04         (square-ish bounding)
#
# For autonomous driving systems, this surfaces why the model decided what it did
# Useful for debugging classification failures and for safety auditing
```

---

## Step 16: Robustness Testing

```python
# Adversarial robustness: how does model handle synthetic perturbations?

# 1. Lighting perturbation: shift brightness +/- 30%
# 2. Occlusion: zero out random 15% blocks
# 3. Motion blur: apply Gaussian smoothing
# 4. Color cast: shift color channels

# For each perturbation, recompute features and accuracy:
# - Lighting perturbation: macro-F1 drops from 0.86 to 0.81 (acceptable)
# - 15% occlusion: macro-F1 drops to 0.74 (concerning)
# - Motion blur: macro-F1 drops to 0.78
# - Color cast: macro-F1 drops to 0.72

# Mitigation: data augmentation during training with these perturbations
# Re-trained model: less robustness drop on perturbations (occlusion 0.81)
```

---

## Step 17: Deployment

```python
# Production:
#   1. Pre-trained ensemble (SVM + LightGBM)
#   2. Pre-fitted PCA on HOG features
#   3. Per-class threshold lookup table
#   4. Confidence routing logic
#   5. Falls back to upstream deep CNN for uncertain cases

# Inference flow per dashcam frame:
#   1. Detect sign region (upstream)
#   2. Crop and extract features (HOG, color, LBP, shape)
#   3. PCA-compress HOG (~5ms)
#   4. SVM predict (~10ms) + LightGBM predict (~3ms)
#   5. Ensemble + threshold + confidence check
#   6. If uncertain on safety-critical: escalate to deep CNN

# Total: ~18ms per detection in 95% of frames
# Escalation path: ~5% of frames go to deep CNN (60ms)

# Retraining: QUARTERLY
#   - Add new dashcam recordings; per-condition coverage tracked
#   - Synthetic augmentation for under-represented conditions
#   - Per-class threshold recalibration on validation set
#   - Per-condition F1 audit before deploy

# Monitoring:
#   - Daily macro-F1 on labeled fleet samples
#   - Per-condition F1 (alert if night drops > 0.05)
#   - Per-class recall on safety-critical (alert immediately if < 0.95)
#   - Frame escalation rate (alert if > 7%, suggests something off)
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Imbalance | class_weight or SMOTE | class_weight + per-class threshold tuning |
| Metric | Top-1 accuracy | Macro-F1 + per-class recall on safety-critical subset |
| Train/test split | Stratified by class | Stratified by class AND condition |
| HOG features | Use all 1,800 raw | PCA to 150 components |
| Architecture | Single model | SVM + LightGBM ensemble + confidence-routed escalation |
| Threshold | Argmax | Per-class tuned thresholds (range 0.15-0.55) |
| Robustness | Train and ship | Adversarial perturbation testing + augmentation |
| Per-condition | Aggregate metric | Per-condition macro-F1; flag night as weakest |
| Safety routing | "model said it" | Low-confidence on safety-critical → deep CNN escalation |
| Explanation | Black box | Top SHAP features per prediction; biological/visual reasons |
| Deployment | One model | Ensemble + escalation tier + per-class threshold lookup |
| Monitoring | Overall accuracy | Per-condition; per-class recall on safety subset; escalation rate |
