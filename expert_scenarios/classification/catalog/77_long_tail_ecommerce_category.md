# Expert Scenario 77: Long-Tail E-commerce Category Classification

> **Complexity:** 2,800 leaf categories with extreme head/tail distribution (top 50 cover 65% of products, tail 1,500 cover < 2%), per-tail accuracy is the business priority (new growth lives in the tail), two-stage head/tail architecture, embedding-based KNN for tail, periodic catalog growth (new tail categories appear monthly).

---

## The Brief

A large marketplace classifies products into 2,800 leaf categories. The catalog is heavily skewed:

- Top 50 categories: ~65% of all products (head)
- Mid 1,250 categories: ~33% (mid)
- Tail 1,500 categories: ~2% (long tail; many with < 50 training examples)

The brief: build a category classifier that:

- Top-1 accuracy on head categories: ≥ 85%.
- Top-3 accuracy on mid categories: ≥ 75%.
- Top-5 recall on tail categories: ≥ 60% (ranking-based, since absolute accuracy is impossible at tail).
- Handles cold-start tail categories (new categories with < 10 examples).
- Inference budget: < 200ms per product.

The business reason for tail accuracy: tail categories are where new growth happens. A model that achieves 95% overall accuracy by ignoring the tail leaves money on the table — those products can't be discovered if they're misclassified.

This is harder than balanced multiclass because: classical models train poorly on classes with < 50 examples; cosine-similarity in embedding space generalizes better in the tail; the head/tail boundary moves over time as categories grow or shrink; new categories are added monthly without retraining; and the metric (top-K) requires probability calibration.

---

## Step 1: Define the Problem Type

```
Type:           Multiclass classification, 2,800 classes, severe head/tail imbalance
Primary Metrics by tier:
  - Head (top 50): Top-1 accuracy ≥ 85%
  - Mid (1,250):    Top-3 accuracy ≥ 75%
  - Tail (1,500):   Top-5 recall ≥ 60% (rank-based)
Secondary:        Macro-F1 weighted by category support
                  Coverage (% of products that get a non-tail prediction)
Constraint:       < 200ms inference; cold-start tail support; monthly category drift
```

**Expert thinking:** different tiers need different metrics. Head is "be right." Tail is "be in the right neighborhood." This is a head-tail TRADE-OFF problem; optimizing only Top-1 accuracy will trash tail performance. Two-stage architecture solves it: confident head classifier + KNN fallback for the tail.

---

## Step 2: Understand the Data

```
Shape: 25M products × ~600 features
Target: leaf_category (2,800 unique)

Feature blocks:

TEXT (TF-IDF on title + brief description):
- 50,000+ TF-IDF features (sparse)

EMBEDDINGS (pre-computed):
- text_embedding: 384-d (sentence-transformer; pre-extracted)
- image_embedding: 512-d (CNN; pre-extracted)

STRUCTURED:
- price_log
- num_attributes_filled
- brand_id (12K+ unique brands)
- top_attribute_value_indicators (sparse)

Category metadata:
- category_path (taxonomy parent → leaf)
- category_age_months (when added to catalog)
- category_volume_30d (rolling product count)
```

---

## Step 3: EDA on Head/Tail

```python
cat_counts = df['leaf_category'].value_counts()

# Head (top 50): 16.2M products
# Mid (1,250):   8.3M products
# Tail (1,500):  0.5M products  -- one-fifth of categories have <50 products each

# Per-tail-category training data
tail_categories = cat_counts.tail(1500).index
tail_per_category = df[df['leaf_category'].isin(tail_categories)].groupby('leaf_category').size()
print(tail_per_category.describe())
# count    1500
# mean       33
# std        18
# min         3   -- this category has only 3 examples!
# 25%        18
# 50%        29
# 75%        45
# max        99

# 412 categories have < 20 training examples
# Direct multinomial training will overfit catastrophically on these
```

**Findings:**

| Finding | Implication |
|---------|------------|
| Top 50 = 65% of products; tail = 2% | Head/tail strategy required |
| 412 tail categories have < 20 examples | Multiclass softmax can't learn these — alternative needed |
| Cold-start: ~30 new tail categories added monthly | Architecture must support new categories without retraining |
| Embedding distance preserves category for tail (PCA visualization clusters work) | KNN on embeddings is viable for tail prediction |

---

## Step 4: Two-Stage Architecture

```
ARCHITECTURE PLAN:

Stage 1: Head Classifier
- Input: product features
- Output: top-1 head category OR "REJECT" (means: not in top 50)
- Trained on: top 50 categories only (head data)

Stage 2: Mid/Tail KNN
- For products that Stage 1 rejected, search nearest neighbors in embedding space
- Use mid + tail products combined; rank top-5 categories
- Cold-start: new categories with > 1 example are immediately searchable

This gives:
- Head products: fast, accurate softmax
- Mid: handled either by Stage 1 if confident, or Stage 2 KNN
- Tail: KNN provides graceful degradation; cold-start works
```

---

## Step 5: Stage 1 — Head Classifier

```python
# Train on top 50 categories
head_categories = cat_counts.head(50).index
df_head = df[df['leaf_category'].isin(head_categories)].copy()

# Add a "REJECT" class: products from mid+tail (sample down to 50K examples)
mid_tail_sample = df[~df['leaf_category'].isin(head_categories)].sample(50000)
mid_tail_sample['leaf_category'] = 'REJECT'
df_train = pd.concat([df_head, mid_tail_sample])

# Train Logistic Regression Softmax (51 classes: 50 head + REJECT)
from sklearn.linear_model import LogisticRegression
head_classifier = LogisticRegression(
    multi_class='multinomial',
    C=1.0, penalty='l2',
    max_iter=200, n_jobs=-1
)
head_classifier.fit(X_train_combined, y_train_with_reject)

# At inference: get top-1 prediction
# If predicted = REJECT, send to Stage 2
# Otherwise: use the head prediction
```

---

## Step 6: Stage 2 — Mid/Tail KNN

```python
# Build a FAISS index of all mid + tail product embeddings
# At inference: query with new product's embedding; retrieve top-50 neighbors
# Aggregate neighbor categories by inverse-distance weighting

import faiss
X_mid_tail = np.stack(df_mid_tail['text_embedding'].values + df_mid_tail['image_embedding'].values, axis=1)
# Concatenate text + image embeddings -> 896-d vector per product

index = faiss.IndexFlatIP(896)  # inner product (with normalized embeddings = cosine)
faiss.normalize_L2(X_mid_tail)
index.add(X_mid_tail)

def knn_predict(query_embedding, k=50):
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    distances, neighbor_idx = index.search(query_embedding.reshape(1, -1), k)
    
    # Weighted vote by inverse distance
    neighbor_cats = df_mid_tail.iloc[neighbor_idx[0]]['leaf_category']
    weights = 1.0 / (1.0 + distances[0])
    
    cat_scores = pd.Series(weights, index=neighbor_cats).groupby(level=0).sum()
    return cat_scores.sort_values(ascending=False).head(5)

# At inference for tail product:
top_5_categories = knn_predict(product_embedding)
```

---

## Step 7: Confidence-Aware Stage Routing

```python
# When Stage 1 predicts a head category but with LOW confidence, also consult Stage 2
# This catches cases where the head classifier is uncertain

def classify_product(features, embedding, head_clf, mid_tail_index, confidence_threshold=0.55):
    # Stage 1
    proba = head_clf.predict_proba(features.reshape(1, -1))[0]
    top_class_idx = np.argmax(proba)
    top_class = head_clf.classes_[top_class_idx]
    top_proba = proba[top_class_idx]
    
    if top_class == 'REJECT':
        # Stage 2: tail KNN
        return knn_predict(embedding, k=50)
    elif top_proba < confidence_threshold:
        # Low-confidence head prediction; combine with KNN top-5
        head_top = pd.Series([top_proba], index=[top_class])
        knn_top = knn_predict(embedding, k=50) * 0.5  # de-weight
        combined = pd.concat([head_top, knn_top]).groupby(level=0).sum()
        return combined.sort_values(ascending=False).head(5)
    else:
        # Confident head prediction
        top_5 = pd.Series(proba, index=head_clf.classes_).sort_values(ascending=False).head(5)
        return top_5
```

---

## Step 8: Cold-Start Handling

```python
# When a new tail category is added (e.g., "Electric Bike Helmets" launches):
# - Initial: 1-3 products
# - Need: classify NEW incoming products into this category from day 1

# Solution: KNN doesn't need retraining when new categories are added
# We just add new product embeddings to the FAISS index
# The category appears in top-5 if any of its 1-3 products are nearest neighbors

# Practical: add new category products to the index nightly
def update_index_nightly(index, new_products):
    new_embeddings = np.stack(new_products['text_embedding'].values + new_products['image_embedding'].values, axis=1)
    faiss.normalize_L2(new_embeddings)
    index.add(new_embeddings)

# Cold-start performance: with 3 example products, top-5 recall ≈ 30-40%
# Bootstrap: as the category grows to 10+ examples, performance climbs to ≥ 60%
```

---

## Step 9: Final Evaluation

```python
# Test set: 2.5M held-out products
# 
# Performance by tier:
#
# HEAD (top 50 categories, ~1.6M products):
#   Top-1 accuracy: 0.87  ✓ (target 0.85)
#   Stage 1 confident prediction: 92% of head products
#   Stage 2 fallback: 8%
#
# MID (1,250 categories, ~830K products):
#   Top-3 accuracy: 0.78  ✓ (target 0.75)
#   Stage 1 captures: 12% (most went to Stage 2 via REJECT)
#   Stage 2 KNN captures: 88%
#
# TAIL (1,500 categories, ~50K products):
#   Top-5 recall: 0.62  ✓ (target 0.60)
#   Stage 1 only routes to Stage 2 (REJECT)
#   Stage 2 KNN captures all
#
# Cold-start (new categories with 1-3 examples):
#   Top-5 recall: 0.34
#   With 5+ examples: 0.51
#   With 10+ examples: 0.66
#
# Inference latency:
#   Stage 1 only: ~30ms p99
#   Stage 2 (KNN): ~80ms p99
#   Combined (when both run): ~110ms p99 (well within 200ms budget)
```

---

## Step 10: Per-Tier Calibration

```python
# Different tiers have different probability dynamics
# Head: well-calibrated (lots of data per class)
# Tail: KNN inverse-distance scores aren't probabilities -- need calibration

# Apply per-tier isotonic calibration:
from sklearn.isotonic import IsotonicRegression

# For head: probability calibration on holdout
head_calibrator = IsotonicRegression(out_of_bounds='clip')
head_calibrator.fit(predictions_head_train, y_train_head)

# For tail: similar approach but on rank-based scores
tail_calibrator = IsotonicRegression(out_of_bounds='clip')
tail_calibrator.fit(rank_scores_tail_train, y_tail_train)
```

---

## Step 11: Deployment Considerations

```python
# Components:
#   1. Head softmax model (~50MB)
#   2. FAISS index of 25M embeddings (~5GB)
#   3. Cosine similarity calibrators
#   4. Per-tier confidence thresholds
#   5. Hourly product upload pipeline appends to FAISS index

# Inference path:
#   1. Receive new product (title, image)
#   2. Compute features (TF-IDF) and embeddings (model server lookup)
#   3. Run Stage 1 classifier
#   4. Conditionally run Stage 2 (if REJECT or low confidence)
#   5. Return top-5 categories with calibrated probabilities

# Daily monitoring:
#   - Per-tier accuracy on labeled new products
#   - Coverage: % of products getting top-5 with confident category
#   - New category propagation: time from category creation to first prediction in it
#   - FAISS index staleness (alert if backlog > 24h)

# Quarterly retraining:
#   - Refresh head classifier on rolling 90-day window of head products
#   - Re-evaluate head/mid/tail boundaries (categories may have moved tiers)
#   - Recompute FAISS index with the new product set

# Continuous adaptation:
#   - New categories: add to FAISS index nightly without retraining
#   - Category renames / merges: update labels in index nightly
#   - Category retirement: archive index entries; products redirect to active categories
```

---

## Step 12: Key Trade-offs

| Decision | Pros | Cons |
|----------|------|------|
| Two-stage head/tail | Specialized models per regime; cold-start works | Architectural complexity; latency higher when both stages run |
| KNN for tail | Generalizes to small / cold-start categories | Latency 50-80ms; FAISS index 5GB memory |
| Embedding-based | Captures semantic similarity | Embeddings come from upstream models — dependency on those |
| Confidence thresholds | Avoids Stage 1 false certainties | More UI work to handle "uncertain head" cases |
| REJECT class | Clean Stage 1 output | Adds a class with high heterogeneity (catch-all) |

---

## Summary: What Made This Expert-Level

| Technique | Beginner Would Do | Expert Did |
|-----------|------------------|------------|
| Architecture | Flat softmax over 2,800 classes | Two-stage: head softmax + tail KNN |
| Imbalance | class_weight | Tier-specific architecture; head classifier doesn't try to predict tail at all |
| Cold-start | Manual retraining | KNN index update nightly; no retraining needed |
| Per-tier metric | Top-1 accuracy overall | Head Top-1 + Mid Top-3 + Tail Top-5 (rank-based) |
| Tail prediction | Skip / reject | Cosine KNN with inverse-distance weighted voting |
| Confidence routing | Single argmax | Combine Stage 1 + Stage 2 when Stage 1 confidence < 0.55 |
| Calibration | Use raw scores | Per-tier isotonic calibration |
| Tier boundary | Fixed | Quarterly re-evaluation; categories move tiers as volume changes |
| Cold-start performance | "Wait for retraining" | Bootstrap KNN-from-day-1 for new categories |
| Latency | Train one model, ship | Profile both stages; total ~110ms; well within 200ms budget |
| FAISS index | Single static dump | Nightly incremental updates + quarterly full rebuild |
| Coverage metric | Ignore | Track % of products getting top-5; new categories = lower coverage temporarily |
