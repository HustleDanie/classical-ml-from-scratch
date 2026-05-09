# Expert Scenario 67: E-commerce Product Taxonomy (Hierarchical Classification)

> **Complexity:** 4-level taxonomy with ~3,500 leaf categories, severe long-tail (top 100 leaf categories cover 60% of products), per-level vs flat trade-off, hierarchy-aware loss, hard-constrained decoding to prevent invalid paths, mixed text + image-embedding features.

---

## The Brief

An e-commerce marketplace lists 12M products across a 4-level taxonomy:

```
Level 1 (Department):     22 categories      e.g. "Electronics"
Level 2 (Category):       180 categories     e.g. "Electronics > Audio"
Level 3 (Subcategory):    980 categories     e.g. "Electronics > Audio > Headphones"
Level 4 (Leaf):          ~3,500 categories   e.g. "Electronics > Audio > Headphones > Wireless Over-Ear"
```

Sellers upload products with title + 5-image bundle + sometimes a description. They CHOOSE a category — often wrongly. The marketplace operations team estimates 18% of products are mis-categorized, hurting search relevance.

The brief: auto-classify products to the correct leaf category. Constraints:

- Top-1 leaf accuracy >= 60% (random would be ~0.03%).
- Top-5 leaf accuracy >= 85%.
- Hierarchy violations (e.g., predict "Electronics > Toys") are NEVER allowed.
- Inference < 200ms per product (catalog onboarding flow).
- Must work on cold-start products (no sales history yet).
- Use only classical ML (no deep learning training; pre-trained CNN embeddings allowed for images, pre-trained text embeddings allowed).

This is harder than flat multiclass because: hierarchical structure must be respected, the leaf level has 3,500 classes (long-tail), and per-level error compounds (a Level-2 mistake makes Level-4 unreachable).

---

## Step 1: Define the Problem Type

```
Type:           Hierarchical multiclass classification (4 levels)
Primary Metric: Top-1 leaf accuracy; Hierarchy-Consistency rate (always 1.0 by construction)
Secondary:      Top-5 leaf accuracy; per-level top-1 (L1, L2, L3, L4)
                Macro-F1 by category to ensure long-tail catches
Business Goal:  Top-1 leaf >= 60%; Top-5 >= 85%
Constraint:     < 200ms inference; cold-start ready; classical ML only
Class count:    3,500 leaf classes; 22 root classes; 180 mid; 980 sub
Imbalance:      Severe (top 100 leaves = 60% of products)
```

**Expert thinking:** the architectural decision is "flat softmax over 3,500 classes" vs "per-level cascade" vs "hybrid." Pure flat softmax with 3,500 classes doesn't respect the hierarchy and lets the model predict invalid paths. Per-level cascade (predict Level 1 → predict Level 2 conditional on Level 1, etc.) respects hierarchy but compounds errors.

We'll use a **hybrid**: train a flat-leaf classifier but constrain the decoding to valid taxonomy paths only.

---

## Step 2: Understand the Data

```
Shape: 12,000,000 products with labels
Features per product:

TEXT (3):
- title (string, 5-200 chars)
- description (string, 0-3000 chars; 38% missing for cold-start)
- brand (string, 12K unique)

IMAGE-DERIVED (1):
- image_embedding (CNN-pretrained, 512-d float vector)

OTHER:
- price ($1-$50000)
- weight_grams (10% missing)
- num_attributes_filled (int, 0-30)
- list_of_attribute_names (variable-length)

LABEL (target):
- category_path (4-tuple: [L1_id, L2_id, L3_id, L4_id])

Hierarchy table:
- Edge list of valid (parent, child) relationships
- Each L2 has exactly 1 L1 parent; each L3 has 1 L2 parent; each L4 has 1 L3 parent
- 22 → 180 → 980 → 3,500
```

**Expert thinking:** image embeddings (512-d dense) and text features (TF-IDF 50K-d sparse) live in very different spaces. We need to combine them for the final classifier — but trees handle dense well, linear models handle sparse well. A common pattern: train two sub-classifiers (text-only and image-only), then ensemble their outputs.

---

## Step 3: EDA

```python
# Class distribution at leaf level
leaf_counts = df['L4_id'].value_counts()
# Top 10 leaves: 28% of products
# Top 100 leaves: 60%
# Bottom 1000 leaves: < 0.5% combined
# 87 leaf categories have < 50 training examples each (effectively unlearnable)

# Per-level marginal accuracy of "predict the most common"
print(f"Most common L1 share: {df['L1_id'].value_counts(normalize=True).iloc[0]}")  # 0.21 (Home & Garden)
print(f"Most common L2 share: {df['L2_id'].value_counts(normalize=True).iloc[0]}")  # 0.041
print(f"Most common L4 share: {df['L4_id'].value_counts(normalize=True).iloc[0]}")  # 0.014

# Mis-categorization audit (sellers self-tag wrong)
# Get a sample of "ground truth" labels from manual ops review (~5,000 products)
# Compare to seller-tags
seller_tag_accuracy = (df_audit['seller_tag'] == df_audit['ops_truth']).mean()
# 0.82 -- 18% of seller tags wrong (matches reported estimate)
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Top-100 leaves cover 60%, bottom 1000 cover < 1% | Long-tail problem — head/tail strategy |
| 87 leaves have < 50 examples | Unlearnable directly; fold them up to L3 prediction or use nearest-neighbor on embeddings |
| Seller self-tags are 82% accurate | Training labels themselves have 18% noise — must clean OR be robust |
| Image embeddings cluster well by category (PCA viz shows clear clumps) | Image is a strong signal |
| Brand correlates with leaf category for 35% of brands (e.g., "Nike" → mostly footwear) | Brand-as-feature with target encoding |

---

## Step 4: Data Cleaning

```python
# === CLEAN SELLER LABEL NOISE ===
# We have a 5K manual-audit set. Use it for a "label noise correction" pass.
# Train a classifier on the audit set; apply to all 12M; rows where the model strongly
# disagrees with seller's self-label get flagged and either re-labeled or excluded.

# Lightweight label noise correction:
# 1. Train preliminary model on seller-tag data
# 2. For each row, if model's top-1 prediction ≠ seller_tag AND model confidence > 0.7:
#    flag for re-review
# 3. Manually resolve top 50K flagged (highest-confidence disagreements)
# 4. Drop or relabel

# After correction: estimated label noise ~5%

# === HIERARCHICAL CONSISTENCY CHECK ===
# Some seller tags are inconsistent with the official taxonomy
df['valid_path'] = df.apply(lambda r: is_valid_taxonomy_path(r['L1'], r['L2'], r['L3'], r['L4']), axis=1)
df = df[df['valid_path']]  # drop ~0.3% with invalid hierarchy

# === FOLD UP RARE LEAVES ===
# Leaves with < 50 examples are unlearnable
# Strategy: model the L3 (subcategory) for those products and let users browse to leaf
rare_leaves = df['L4_id'].value_counts()[df['L4_id'].value_counts() < 50].index
df['L4_modeled'] = df['L4_id'].where(~df['L4_id'].isin(rare_leaves), -1)  # -1 = "rare/unmodeled"

# Now we predict L4_modeled (3,413 leaves + 1 "rare" bucket)
```

---

## Step 5: Feature Engineering

```python
# === TEXT FEATURES ===
from sklearn.feature_extraction.text import TfidfVectorizer

df['text_combined'] = (
    df['title'].fillna('') + ' [TITLE-END] ' +
    df['brand'].fillna('') + ' [BRAND-END] ' +
    df['description'].fillna('').str[:1000]
)

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=100000,
    min_df=10,
    max_df=0.85,
    sublinear_tf=True
)
X_text = tfidf.fit_transform(df['text_combined'])

# === IMAGE EMBEDDINGS (already pre-extracted) ===
X_image = np.stack(df['image_embedding'].values)  # (n, 512)
X_image_scaled = StandardScaler().fit_transform(X_image)

# === BRAND TARGET ENCODING ===
# For each brand, the dominant L4 in training data
brand_dominant_L4 = df_train.groupby('brand')['L4_id'].agg(lambda x: x.mode().iloc[0])
df['brand_dominant_L4'] = df['brand'].map(brand_dominant_L4).fillna(-1)
# Convert to one-hot or as a feature for trees

# === STRUCTURED FEATURES ===
df['log_price'] = np.log1p(df['price'])
df['has_description'] = df['description'].notna().astype(int)
df['title_length'] = df['title'].str.len()
df['num_attributes_filled_norm'] = df['num_attributes_filled'] / 30
```

---

## Step 6: Architecture — Hybrid Per-Level + Embedding Lookup

```python
# Plan:
# 1. Flat-leaf logistic regression on TF-IDF text (high recall)
# 2. Image-embedding KNN per-leaf (fallback for visual similarity)
# 3. Combine: weighted sum of probabilities, then constrained-decode to valid taxonomy path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# === MODEL 1: Flat-leaf Logistic Regression on text ===
# 100K text features × 3,400 classes -- big model
flat_lr = LogisticRegression(
    C=1.0, penalty='l2', solver='saga',
    multi_class='multinomial',
    max_iter=200,
    n_jobs=-1
)
flat_lr.fit(X_text_train, y_L4_train)
# Top-1 leaf accuracy: 0.51

# === MODEL 2: Per-level Logistic Regression cascade ===
# Train L1, L2, L3, L4 classifiers separately
# At inference: predict L1, then L2 conditional on predicted L1, etc.
lr_L1 = LogisticRegression(C=1.0, multi_class='multinomial', max_iter=200, n_jobs=-1)
lr_L2 = ... (one per L1 parent)  # OR a single L2 model
lr_L3 = ...
lr_L4 = ...
# Per-level top-1: L1 0.93, L2 0.78, L3 0.66, L4 0.49
# Cumulative top-1 (correct path): 0.41 (errors compound)

# Per-level cascade gives WORSE end-to-end accuracy than flat
# because errors compound. Skip cascade.

# === MODEL 3: Flat softmax + constrained decoding ===
# Take flat-LR probabilities; mask out leaves whose L1...L3 path is unlikely
# This combines flat training with hierarchy-aware decoding

# === MODEL 4: KNN on image embeddings ===
from sklearn.neighbors import KNeighborsClassifier
knn_image = KNeighborsClassifier(n_neighbors=20, metric='cosine', n_jobs=-1)
knn_image.fit(X_image_train, y_L4_train)
# Top-1 image-only: 0.42

# === ENSEMBLE: text + image, weighted ===
# At inference:
# proba_text  = flat_lr.predict_proba(X_text_query)         # (3400,)
# proba_image = knn_image.predict_proba(X_image_query)      # (3400,)
# proba_combined = 0.65 * proba_text + 0.35 * proba_image
# Top-1 combined: 0.61
```

---

## Step 7: Constrained Decoding for Hierarchy Consistency

```python
# Take the flat-classifier probabilities and constrain to valid taxonomy paths

def hierarchy_consistent_predict(proba_L4, taxonomy_tree):
    """
    proba_L4: array of probabilities per L4 leaf
    taxonomy_tree: dict of valid (L1, L2, L3, L4) tuples
    """
    # For each candidate L4, parent path is determined; aggregate probabilities up the tree
    best_path = None
    best_score = -1
    for path in taxonomy_tree:
        score = proba_L4[path.L4_idx]  # leaf probability
        if score > best_score:
            best_score = score
            best_path = path
    return best_path

# Wait -- this just picks argmax over a flat softmax (which is automatically
# valid since each L4 implies a unique path). The hierarchy-violation problem
# only appears if you predict each level INDEPENDENTLY (cascade). Flat softmax
# already gives you hierarchy consistency for free.

# REAL hierarchy constraint matters when scoring each level independently then combining
# We don't do that. We use flat-leaf softmax. Decoded path is always valid.
```

**Expert insight:** the hierarchy constraint problem typically arises when you train per-level classifiers and combine. With flat-leaf softmax, you predict the leaf directly and the path is implied, always valid. We chose flat for this reason.

---

## Step 8: Training the Flat-Leaf Model

```python
# 12M rows × 3,400 classes × 100K features -- BIG
# Logistic regression with multinomial loss is feasible with saga solver

from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack

# Combine text + scaled image embedding
X_combined = hstack([X_text, csr_matrix(X_image_scaled)])

flat_model = LogisticRegression(
    C=0.5,
    penalty='l2',
    solver='saga',
    multi_class='multinomial',
    max_iter=300,
    n_jobs=-1,
    verbose=1
)
flat_model.fit(X_combined_train, y_L4_train)
# Training time: 4 hours on 32-core machine

# Save model: 8GB -- LARGE
# Inference is OK because each row only computes 3,400 dot products with sparse text
```

---

## Step 9: Long-Tail Strategy

```python
# 87 leaves have < 50 examples
# Strategy: route to "L3 prediction + manual disambiguation" for these

# After flat softmax:
# 1. If predicted leaf has ≥ 100 training examples: use prediction
# 2. If predicted leaf has < 100 examples: drop to L3 (subcategory) prediction
#    User then disambiguates manually

# For the head 1000 leaves: model accuracy 0.68
# For the mid 2000 leaves:  model accuracy 0.42
# For the tail 500 leaves:  model accuracy 0.21 (route to L3)
```

---

## Step 10: Final Evaluation

```python
# Test set: 1.2M held-out products with manual-truth labels
#
# Performance:
#   Top-1 leaf accuracy:      0.62     ✓ (target 0.60)
#   Top-5 leaf accuracy:      0.87     ✓ (target 0.85)
#   Top-1 L1 accuracy:        0.96
#   Top-1 L2 accuracy:        0.83
#   Top-1 L3 accuracy:        0.74
#   Hierarchy violations:     0.0%     (by construction, flat softmax)
#
# Per-segment:
#   Top 100 leaves:    Top-1 0.71
#   Mid 1000 leaves:   Top-1 0.45
#   Bottom 2000:       Top-1 0.18 (these route to L3)
#
# Inference: 120ms per product (TF-IDF + flat LR + ensemble math)  ✓
```

---

## Step 11: Cold-Start Handling

```python
# Cold-start = new products without sales history
# All our features are content-based (title, image, brand, description)
# So cold-start works the same as warm products

# Edge case: brand we've never seen
# Fallback: model still predicts based on text + image alone
# Brand_dominant_L4 feature defaults to -1; trees handle this naturally
```

---

## Step 12: Deployment

```python
# Model artifacts:
#   - 100K-term TF-IDF vocabulary
#   - 8GB flat-leaf Logistic Regression
#   - Pre-computed image-embedding KNN index (FAISS)
#   - Taxonomy tree dict
#   - Brand → dominant-L4 lookup

# Inference flow:
#   1. Vectorize text (50ms)
#   2. Logistic Regression top-K (40ms; only return top-50 leaves)
#   3. KNN image lookup (20ms; pre-built index)
#   4. Weighted combine (5ms)
#   5. Return top-5 leaves with probabilities
# Total: ~120ms

# Retraining: MONTHLY
#   - New products and corrected labels accumulate
#   - Retrain flat softmax on rolling 18-month window
#   - Re-fit image-KNN index

# Monitoring:
#   - Daily top-1 accuracy on next-day's manual-corrected sample (~1K products)
#   - Per-leaf accuracy drift (alert if any leaf with > 100 examples drops > 0.10)
#   - New leaves: when ops adds a new L4 category, model has 0% recall until retrained
#     Mitigation: surface as "unknown leaf, predicted L3 confidently"
```

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Architecture | Per-level cascade (errors compound) | Flat softmax over leaves; hierarchy implied |
| Class count handling | One-hot 3,500 classes | Flat multinomial logistic with sparse text |
| Long-tail leaves | Train as usual | Fold leaves with < 50 examples; route to L3 |
| Image features | Concatenate to text | Separate KNN model; weighted ensemble |
| Label noise | Train on seller-tag as truth | Audit-set + flag-and-review for high-confidence disagreements |
| Cold-start | "We can't predict for new products" | Content-based features (title/image/brand) work without sales |
| Hierarchy enforcement | Manual rule checking | Built-in via flat-leaf softmax |
| Brand handling | One-hot 12K brands | Target encode brand → dominant L4; fall back gracefully |
| New leaves | Retrain periodically | Active learning loop: route unknown leaves to ops review |
| Tail evaluation | Overall accuracy | Head/mid/tail breakdown with separate routing logic |
