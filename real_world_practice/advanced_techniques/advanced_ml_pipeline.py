"""
Advanced ML Techniques Pipeline -- 20 Newsgroups Text Classification
=====================================================================

Dataset: 20 Newsgroups (sklearn built-in) -- ~4,500 posts, text + metadata
Task:    Binary classification (Tech vs Non-Tech topics)

This pipeline covers the REMAINING real-world gaps not addressed in
the Titanic (classification) or Ames Housing (regression) pipelines:

  1.  Text / NLP features (TF-IDF, CountVectorizer, text statistics)
  2.  High-cardinality categorical encoding (frequency, target, hashing)
  3.  Threshold tuning (precision-recall curves, optimal cutoff)
  4.  Cross-validation variants (StratifiedKFold, GroupKFold, TimeSeriesSplit)
  5.  Data drift detection & monitoring (KS test, PSI, confidence shift)
  6.  Model deployment (Flask REST API for serving predictions)

Dataset choice rationale:
  - Natural TEXT data (newsgroup posts) for NLP feature engineering
  - HIGH-CARDINALITY categoricals (email domains, organizations)
  - Group structure (can group by domain for GroupKFold)
  - Easy to simulate temporal ordering for TimeSeriesSplit
  - Binary classification enables threshold tuning
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import time
import re
from collections import Counter

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GroupKFold,
    TimeSeriesSplit, cross_val_score
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve,
    classification_report, fbeta_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb

from scipy.sparse import hstack, csr_matrix
from scipy.stats import ks_2samp
import joblib

warnings.filterwarnings("ignore")

PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def print_section(title, level=1):
    if level == 1:
        print("\n\n" + "=" * 70)
        print(title)
        print("=" * 70)
    else:
        print(f"\n  --- {title} ---")


def parse_newsgroup_post(text):
    """Parse headers and body from a raw newsgroup post."""
    headers = {}
    lines = text.split('\n')
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip() == '':
            body_start = i + 1
            break
        if ':' in line and not line.startswith(' ') and not line.startswith('\t'):
            key, _, value = line.partition(':')
            headers[key.strip()] = value.strip()
    body = '\n'.join(lines[body_start:])
    return headers, body


def extract_email_domain(from_field):
    """Extract email domain from a From: header value."""
    if not from_field:
        return "unknown"
    match = re.search(r'@([\w.-]+)', from_field)
    return match.group(1).lower() if match else "unknown"


def compute_text_stats(texts):
    """Compute numerical features from raw text."""
    stats = pd.DataFrame()
    stats['word_count'] = texts.apply(lambda x: len(x.split()))
    stats['char_count'] = texts.apply(len)
    stats['avg_word_len'] = texts.apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)
    stats['sentence_count'] = texts.apply(
        lambda x: x.count('.') + x.count('!') + x.count('?'))
    stats['uppercase_ratio'] = texts.apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
    stats['digit_ratio'] = texts.apply(
        lambda x: sum(1 for c in x if c.isdigit()) / max(len(x), 1))
    stats['special_char_ratio'] = texts.apply(
        lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / max(len(x), 1))
    stats['line_count'] = texts.apply(lambda x: x.count('\n') + 1)
    stats['has_url'] = texts.apply(
        lambda x: 1 if re.search(r'http[s]?://', x) else 0)
    stats['quote_ratio'] = texts.apply(
        lambda x: sum(1 for line in x.split('\n') if line.strip().startswith('>')) /
        max(x.count('\n') + 1, 1))
    return stats


def frequency_encode(train_series, test_series):
    """Replace each category with its frequency in training data."""
    freq = train_series.value_counts(normalize=True).to_dict()
    return (train_series.map(freq).fillna(0).values,
            test_series.map(freq).fillna(0).values)


def target_encode(train_series, y_train, test_series, smoothing=10):
    """Target encoding with Bayesian smoothing (prevents overfitting)."""
    global_mean = y_train.mean()
    agg = pd.DataFrame({'target': y_train.values, 'cat': train_series.values})
    cat_stats = agg.groupby('cat')['target'].agg(['mean', 'count'])
    # Smoothing: weight between category mean and global mean
    smoother = 1 / (1 + np.exp(-(cat_stats['count'] - smoothing)))
    cat_stats['smoothed'] = smoother * cat_stats['mean'] + (1 - smoother) * global_mean
    enc_map = cat_stats['smoothed'].to_dict()
    return (train_series.map(enc_map).fillna(global_mean).values,
            test_series.map(enc_map).fillna(global_mean).values)


def calculate_psi(reference, current, bins=10):
    """Population Stability Index -- detects distribution shift."""
    ref_hist, bin_edges = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bin_edges)
    ref_pct = np.clip(ref_hist / max(len(reference), 1), 0.0001, None)
    cur_pct = np.clip(cur_hist / max(len(current), 1), 0.0001, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def main():
    print("=" * 70)
    print("ADVANCED ML TECHNIQUES PIPELINE")
    print("Dataset: 20 Newsgroups (Text Classification)")
    print("Covers: NLP, High-Cardinality, Threshold Tuning,")
    print("        CV Variants, Data Drift, Deployment")
    print("=" * 70)

    # ==================================================================
    # STEP 1: LOAD & EXPLORE
    # ==================================================================
    print_section("STEP 1: LOAD & EXPLORE 20 NEWSGROUPS")

    tech_cats = ['comp.graphics', 'comp.sys.ibm.pc.hardware',
                 'sci.electronics', 'sci.space']
    non_tech_cats = ['rec.sport.hockey', 'rec.autos',
                     'talk.politics.guns', 'talk.religion.misc']

    print(f"\n  Fetching 20 Newsgroups (8 categories)...")
    data = fetch_20newsgroups(
        categories=tech_cats + non_tech_cats,
        remove=(),  # KEEP headers for feature extraction
        random_state=42
    )
    print(f"  Downloaded {len(data.data)} posts")

    # Parse each post into structured data
    records = []
    for text, target_idx in zip(data.data, data.target):
        newsgroup = data.target_names[target_idx]
        headers, body = parse_newsgroup_post(text)
        from_field = headers.get('From', '')
        domain = extract_email_domain(from_field)
        org = headers.get('Organization', 'unknown')
        subject = headers.get('Subject', '')
        is_tech = 1 if newsgroup in tech_cats else 0

        records.append({
            'body': body if body.strip() else text,
            'subject': subject,
            'email_domain': domain,
            'organization': org,
            'newsgroup': newsgroup,
            'is_tech': is_tech,
        })

    df = pd.DataFrame(records)

    print(f"\n  Shape: {df.shape}")
    print(f"  Target distribution:")
    print(f"    Tech (1):     {(df['is_tech'] == 1).sum()} ({(df['is_tech'] == 1).mean()*100:.1f}%)")
    print(f"    Non-Tech (0): {(df['is_tech'] == 0).sum()} ({(df['is_tech'] == 0).mean()*100:.1f}%)")

    n_unique_domains = df['email_domain'].nunique()
    n_unique_orgs = df['organization'].nunique()
    print(f"\n  High-cardinality categoricals:")
    print(f"    email_domain:  {n_unique_domains} unique values")
    print(f"    organization:  {n_unique_orgs} unique values")
    print(f"    -> OneHot would create {n_unique_domains + n_unique_orgs} columns!")
    print(f"    -> Need smarter encoding strategies")

    top_domains = df['email_domain'].value_counts().head(10)
    print(f"\n  Top 10 email domains:")
    for dom, cnt in top_domains.items():
        tech_rate = df[df['email_domain'] == dom]['is_tech'].mean()
        print(f"    {dom:30s}  {cnt:4d} posts  tech_rate={tech_rate:.2f}")

    # Sample text
    print(f"\n  Sample post (first 200 chars):")
    print(f"    '{df['body'].iloc[0][:200]}...'")

    # Train/test split
    X = df[['body', 'subject', 'email_domain', 'organization', 'newsgroup']]
    y = df['is_tech']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)},  Test: {len(X_test)}")

    # ==================================================================
    # STEP 2: TEXT / NLP FEATURE ENGINEERING
    # ==================================================================
    print_section("STEP 2: TEXT / NLP FEATURE ENGINEERING")

    print(f"\n  Real-world NLP pipeline:")
    print(f"    Raw text -> Clean -> Vectorize -> Train model")
    print(f"    Two main approaches: TF-IDF and CountVectorizer")

    # --- TF-IDF ---
    print(f"\n  [TF-IDF] Term Frequency - Inverse Document Frequency")
    print(f"    - Weighs terms by importance (rare words = more weight)")
    print(f"    - Standard for classical ML text classification")
    tfidf = TfidfVectorizer(
        max_features=5000, stop_words='english',
        ngram_range=(1, 2), min_df=2, max_df=0.95
    )
    X_tfidf_train = tfidf.fit_transform(X_train['body'])
    X_tfidf_test = tfidf.transform(X_test['body'])
    print(f"    Features: {X_tfidf_train.shape[1]} (max_features=5000, unigrams+bigrams)")
    print(f"    Matrix: sparse ({X_tfidf_train.nnz} non-zero / "
          f"{X_tfidf_train.shape[0]*X_tfidf_train.shape[1]} total = "
          f"{X_tfidf_train.nnz/(X_tfidf_train.shape[0]*X_tfidf_train.shape[1])*100:.2f}% dense)")

    # Top TF-IDF features
    feature_names = tfidf.get_feature_names_out()
    mean_tfidf = np.array(X_tfidf_train.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[-15:][::-1]
    print(f"\n    Top 15 TF-IDF terms:")
    for idx in top_indices:
        print(f"      {feature_names[idx]:25s}  avg_tfidf = {mean_tfidf[idx]:.4f}")

    # --- CountVectorizer ---
    print(f"\n  [CountVectorizer] Simple word counts")
    print(f"    - Simpler than TF-IDF (just raw counts)")
    print(f"    - Good baseline, sometimes competitive")
    countvec = CountVectorizer(
        max_features=5000, stop_words='english',
        ngram_range=(1, 2), min_df=2, max_df=0.95
    )
    X_count_train = countvec.fit_transform(X_train['body'])
    X_count_test = countvec.transform(X_test['body'])

    # Quick comparison
    lr_tfidf = LogisticRegression(max_iter=1000, random_state=42)
    lr_tfidf.fit(X_tfidf_train, y_train)
    acc_tfidf = accuracy_score(y_test, lr_tfidf.predict(X_tfidf_test))

    lr_count = LogisticRegression(max_iter=1000, random_state=42)
    lr_count.fit(X_count_train, y_train)
    acc_count = accuracy_score(y_test, lr_count.predict(X_count_test))
    print(f"\n    LogReg + TF-IDF:        Acc = {acc_tfidf:.4f}")
    print(f"    LogReg + CountVec:      Acc = {acc_count:.4f}")
    print(f"    -> {'TF-IDF wins' if acc_tfidf > acc_count else 'CountVec competitive'}!")

    # --- Text Statistics ---
    print(f"\n  [Text Statistics] Engineered numeric features from text")
    text_stats_train = compute_text_stats(X_train['body'])
    text_stats_test = compute_text_stats(X_test['body'])
    print(f"    Features created: {list(text_stats_train.columns)}")

    scaler = StandardScaler()
    stats_train_scaled = csr_matrix(scaler.fit_transform(text_stats_train))
    stats_test_scaled = csr_matrix(scaler.transform(text_stats_test))

    # Combine TF-IDF + text stats
    X_nlp_train = hstack([X_tfidf_train, stats_train_scaled])
    X_nlp_test = hstack([X_tfidf_test, stats_test_scaled])

    lr_combined = LogisticRegression(max_iter=1000, random_state=42)
    lr_combined.fit(X_nlp_train, y_train)
    acc_combined = accuracy_score(y_test, lr_combined.predict(X_nlp_test))
    print(f"\n    LogReg + TF-IDF + Stats: Acc = {acc_combined:.4f}")
    print(f"    -> Text stats {'helped' if acc_combined > acc_tfidf else 'did not help much'} "
          f"(+{(acc_combined - acc_tfidf)*100:.2f}%)")

    # Plot: NLP comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    methods = ['TF-IDF', 'CountVec', 'TF-IDF+Stats']
    accs = [acc_tfidf, acc_count, acc_combined]
    colors = ['steelblue', '#e74c3c', '#2ecc71']
    axes[0].bar(methods, accs, color=colors)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("NLP Vectorization Comparison")
    axes[0].set_ylim(min(accs) - 0.02, 1.0)
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.003, f"{v:.4f}", ha='center', fontsize=10)

    text_stats_train[['word_count', 'char_count']].hist(
        bins=30, ax=axes[1], color='steelblue', alpha=0.7)
    axes[1].set_title("Text Length Distributions (Training)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_nlp_features.png", dpi=150)
    plt.close()
    print("  [Saved] plots/01_nlp_features.png")

    # ==================================================================
    # STEP 3: HIGH-CARDINALITY CATEGORICAL ENCODING
    # ==================================================================
    print_section("STEP 3: HIGH-CARDINALITY CATEGORICAL ENCODING")

    print(f"\n  Problem: 'email_domain' has {n_unique_domains} unique values")
    print(f"    OneHotEncoding would create {n_unique_domains} columns -> curse of dimensionality")
    print(f"    Solutions: Frequency Encoding, Target Encoding, Hashing Trick")

    domain_train = X_train['email_domain']
    domain_test = X_test['email_domain']

    # Method 1: Frequency Encoding
    print(f"\n  [Method 1] Frequency Encoding")
    print(f"    Replace each domain with its frequency in training data")
    freq_train, freq_test = frequency_encode(domain_train, domain_test)
    print(f"    Example: '{domain_train.iloc[0]}' -> {freq_train[0]:.4f}")
    print(f"    Pros: Simple, no leakage, handles unseen categories")
    print(f"    Cons: Doesn't capture target relationship")

    # Method 2: Target Encoding
    print(f"\n  [Method 2] Target Encoding (with Bayesian smoothing)")
    print(f"    Replace each domain with smoothed mean target value")
    te_train, te_test = target_encode(domain_train, y_train, domain_test, smoothing=10)
    print(f"    Example: '{domain_train.iloc[0]}' -> {te_train[0]:.4f}")
    print(f"    Pros: Captures target relationship, single column")
    print(f"    Cons: Risk of overfitting (smoothing helps)")

    # Method 3: Hashing Trick
    print(f"\n  [Method 3] Hashing Trick (FeatureHasher)")
    print(f"    Hash each domain into fixed-size vector (32 buckets)")
    hasher = FeatureHasher(n_features=32, input_type='string')
    hash_train = hasher.transform([[d] for d in domain_train.values])
    hash_test = hasher.transform([[d] for d in domain_test.values])
    print(f"    {n_unique_domains} domains -> 32 columns (fixed!)")
    print(f"    Pros: Fixed size, no need to track mapping, handles any new domain")
    print(f"    Cons: Collisions (different domains may share bucket)")

    # Compare encoding methods
    print(f"\n  Comparing encoding methods (LogReg + TF-IDF + encoded domain):")
    encoding_results = {}

    # Baseline: no domain
    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_tfidf_train, y_train)
    acc_base = accuracy_score(y_test, lr_base.predict(X_tfidf_test))
    encoding_results["No domain (baseline)"] = acc_base
    print(f"    No domain:         Acc = {acc_base:.4f}")

    # Frequency encoding
    X_freq_train = hstack([X_tfidf_train, csr_matrix(freq_train.reshape(-1, 1))])
    X_freq_test = hstack([X_tfidf_test, csr_matrix(freq_test.reshape(-1, 1))])
    lr_freq = LogisticRegression(max_iter=1000, random_state=42)
    lr_freq.fit(X_freq_train, y_train)
    acc_freq = accuracy_score(y_test, lr_freq.predict(X_freq_test))
    encoding_results["Frequency"] = acc_freq
    print(f"    Frequency:         Acc = {acc_freq:.4f}")

    # Target encoding
    X_te_train = hstack([X_tfidf_train, csr_matrix(te_train.reshape(-1, 1))])
    X_te_test = hstack([X_tfidf_test, csr_matrix(te_test.reshape(-1, 1))])
    lr_te = LogisticRegression(max_iter=1000, random_state=42)
    lr_te.fit(X_te_train, y_train)
    acc_te = accuracy_score(y_test, lr_te.predict(X_te_test))
    encoding_results["Target"] = acc_te
    print(f"    Target encoding:   Acc = {acc_te:.4f}")

    # Hashing
    X_hash_train = hstack([X_tfidf_train, hash_train])
    X_hash_test = hstack([X_tfidf_test, hash_test])
    lr_hash = LogisticRegression(max_iter=1000, random_state=42)
    lr_hash.fit(X_hash_train, y_train)
    acc_hash = accuracy_score(y_test, lr_hash.predict(X_hash_test))
    encoding_results["Hashing (32)"] = acc_hash
    print(f"    Hashing (32 bins): Acc = {acc_hash:.4f}")

    best_enc = max(encoding_results, key=encoding_results.get)
    print(f"\n    Best encoding: {best_enc} ({encoding_results[best_enc]:.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(encoding_results.keys())
    vals = list(encoding_results.values())
    colors = ['gray' if n == "No domain (baseline)" else 'steelblue' for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_xlabel("Accuracy")
    ax.set_title("High-Cardinality Encoding Comparison")
    for i, v in enumerate(vals):
        ax.text(v + 0.001, i, f"{v:.4f}", va='center')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_high_cardinality_encoding.png", dpi=150)
    plt.close()
    print("  [Saved] plots/02_high_cardinality_encoding.png")

    # ==================================================================
    # STEP 4: TRAIN & COMPARE MODELS ON TEXT DATA
    # ==================================================================
    print_section("STEP 4: TRAIN & COMPARE MODELS ON TEXT DATA")

    # Use the best combined features: TF-IDF + text stats + target-encoded domain
    X_full_train = hstack([X_tfidf_train, stats_train_scaled,
                           csr_matrix(te_train.reshape(-1, 1))])
    X_full_test = hstack([X_tfidf_test, stats_test_scaled,
                          csr_matrix(te_test.reshape(-1, 1))])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Multinomial NB": MultinomialNB(alpha=1.0),
        "Linear SVC": LinearSVC(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, eval_metric="logloss",
                                       random_state=42, verbosity=0),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    trained_models = {}

    for name, model in models.items():
        t0 = time.time()
        # MultinomialNB needs non-negative features -> use TF-IDF only
        if name == "Multinomial NB":
            X_tr, X_te = X_tfidf_train, X_tfidf_test
            cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring="accuracy")
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
        else:
            X_tr, X_te = X_full_train, X_full_test
            cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring="accuracy")
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)

        elapsed = time.time() - t0
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name, "CV Acc": cv_scores.mean(),
            "CV Std": cv_scores.std(), "Accuracy": acc,
            "F1": f1, "Time": elapsed
        })
        trained_models[name] = model
        print(f"\n  {name}:")
        print(f"    5-Fold CV: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"    Test Acc:  {acc:.4f}  |  F1: {f1:.4f}  |  Time: {elapsed:.1f}s")

    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print(f"\n  Best model: {results_df.iloc[0]['Model']} (F1={results_df.iloc[0]['F1']:.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sorted_res = results_df.sort_values("F1", ascending=True)
    ax.barh(sorted_res["Model"], sorted_res["F1"],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_res))))
    ax.set_xlabel("F1 Score")
    ax.set_title("Model Comparison on Text Classification")
    for i, (_, row) in enumerate(sorted_res.iterrows()):
        ax.text(row["F1"] + 0.002, i, f"{row['F1']:.4f}", va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_model_comparison.png", dpi=150)
    plt.close()
    print("  [Saved] plots/03_model_comparison.png")

    # ==================================================================
    # STEP 5: THRESHOLD TUNING
    # ==================================================================
    print_section("STEP 5: THRESHOLD TUNING")

    print(f"\n  Why 0.5 isn't always optimal:")
    print(f"    - Default 0.5 treats false positives = false negatives")
    print(f"    - In spam detection: want HIGH precision (don't lose real emails)")
    print(f"    - In medical screening: want HIGH recall (don't miss patients)")
    print(f"    - Precision-Recall curve shows ALL possible tradeoffs")

    # Use LogReg (has predict_proba)
    best_model_name = results_df.iloc[0]["Model"]
    # Find a model with predict_proba
    for name in results_df["Model"]:
        if hasattr(trained_models[name], "predict_proba"):
            prob_model = trained_models[name]
            prob_model_name = name
            break

    print(f"\n  Using {prob_model_name} for threshold analysis")
    if prob_model_name == "Multinomial NB":
        y_proba = prob_model.predict_proba(X_tfidf_test)[:, 1]
    else:
        y_proba = prob_model.predict_proba(X_full_test)[:, 1]

    # Default threshold
    y_pred_default = (y_proba >= 0.5).astype(int)
    acc_default = accuracy_score(y_test, y_pred_default)
    prec_default = precision_score(y_test, y_pred_default)
    rec_default = recall_score(y_test, y_pred_default)
    f1_default = f1_score(y_test, y_pred_default)
    print(f"\n  Default threshold (0.5):")
    print(f"    Acc={acc_default:.4f}  Prec={prec_default:.4f}  "
          f"Rec={rec_default:.4f}  F1={f1_default:.4f}")

    # Precision-Recall curve
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_proba)

    # Find optimal F1 threshold
    f1_scores_arr = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    optimal_f1_idx = np.argmax(f1_scores_arr)
    optimal_threshold = thresholds_pr[min(optimal_f1_idx, len(thresholds_pr) - 1)]
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    print(f"\n  Optimal F1 threshold: {optimal_threshold:.4f}")
    print(f"    Acc={accuracy_score(y_test, y_pred_optimal):.4f}  "
          f"Prec={precision_score(y_test, y_pred_optimal):.4f}  "
          f"Rec={recall_score(y_test, y_pred_optimal):.4f}  "
          f"F1={f1_score(y_test, y_pred_optimal):.4f}")

    # Find threshold for high precision (90%+)
    high_prec_mask = precisions >= 0.95
    if high_prec_mask.any():
        idx_95 = np.where(high_prec_mask)[0][-1]
        thresh_95prec = thresholds_pr[min(idx_95, len(thresholds_pr) - 1)]
        y_pred_95 = (y_proba >= thresh_95prec).astype(int)
        print(f"\n  High-Precision threshold (>=95% prec): {thresh_95prec:.4f}")
        print(f"    Prec={precision_score(y_test, y_pred_95):.4f}  "
              f"Rec={recall_score(y_test, y_pred_95):.4f}")
    else:
        thresh_95prec = 0.9

    # Find threshold for high recall (90%+)
    high_rec_mask = recalls >= 0.95
    if high_rec_mask.any():
        # Find the highest threshold that still gives 95% recall
        idx_95r = np.where(high_rec_mask)[0][0]
        thresh_95rec = thresholds_pr[min(idx_95r, len(thresholds_pr) - 1)]
        y_pred_95r = (y_proba >= thresh_95rec).astype(int)
        print(f"\n  High-Recall threshold (>=95% rec): {thresh_95rec:.4f}")
        print(f"    Prec={precision_score(y_test, y_pred_95r):.4f}  "
              f"Rec={recall_score(y_test, y_pred_95r):.4f}")

    # F-beta scores for different priorities
    print(f"\n  F-beta scores at default threshold (0.5):")
    for beta in [0.5, 1.0, 2.0]:
        fb = fbeta_score(y_test, y_pred_default, beta=beta)
        emphasis = "precision" if beta < 1 else ("balanced" if beta == 1 else "recall")
        print(f"    F{beta} = {fb:.4f}  (emphasizes {emphasis})")

    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    # Plot threshold analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Precision-Recall curve
    axes[0].plot(recalls, precisions, 'b-', linewidth=2)
    axes[0].axhline(y=prec_default, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(x=rec_default, color='gray', linestyle='--', alpha=0.5)
    axes[0].scatter([rec_default], [prec_default], c='red', s=100, zorder=5,
                    label=f'Default (0.5)')
    if optimal_f1_idx < len(recalls):
        axes[0].scatter([recalls[optimal_f1_idx]], [precisions[optimal_f1_idx]],
                       c='green', s=100, zorder=5, marker='*',
                       label=f'Best F1 ({optimal_threshold:.3f})')
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].legend()

    # ROC curve
    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC={auc:.4f}')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    # Threshold vs metrics
    thresh_range = np.linspace(0.1, 0.9, 50)
    precs_t, recs_t, f1s_t = [], [], []
    for t in thresh_range:
        yp = (y_proba >= t).astype(int)
        if yp.sum() == 0 or yp.sum() == len(yp):
            precs_t.append(0); recs_t.append(0); f1s_t.append(0)
        else:
            precs_t.append(precision_score(y_test, yp))
            recs_t.append(recall_score(y_test, yp))
            f1s_t.append(f1_score(y_test, yp))
    axes[2].plot(thresh_range, precs_t, 'b-', label='Precision')
    axes[2].plot(thresh_range, recs_t, 'r-', label='Recall')
    axes[2].plot(thresh_range, f1s_t, 'g-', label='F1', linewidth=2)
    axes[2].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default')
    axes[2].axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.5,
                    label=f'Optimal ({optimal_threshold:.3f})')
    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Threshold vs Metrics")
    axes[2].legend(fontsize=8)

    plt.suptitle("Threshold Tuning Analysis", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_threshold_tuning.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/04_threshold_tuning.png")

    print(f"\n  KEY TAKEAWAY:")
    print(f"    Default 0.5:  F1={f1_default:.4f}")
    print(f"    Optimal:      F1={f1_score(y_test, y_pred_optimal):.4f} "
          f"(threshold={optimal_threshold:.3f})")
    print(f"    -> Always tune the threshold for your business objective!")

    # ==================================================================
    # STEP 6: CROSS-VALIDATION VARIANTS
    # ==================================================================
    print_section("STEP 6: CROSS-VALIDATION VARIANTS")

    print(f"\n  Three CV strategies for different scenarios:")
    print(f"    1. StratifiedKFold -- preserves class proportions (standard)")
    print(f"    2. GroupKFold     -- prevents data leakage between groups")
    print(f"    3. TimeSeriesSplit-- respects temporal ordering")

    model_cv = LogisticRegression(max_iter=1000, random_state=42)

    # 1. StratifiedKFold
    print(f"\n  [StratifiedKFold] Standard -- preserves class balance")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_strat = cross_val_score(model_cv, X_full_train, y_train,
                                    cv=skf, scoring='f1')
    print(f"    F1: {scores_strat.mean():.4f} +/- {scores_strat.std():.4f}")
    print(f"    When to use: Default choice. Class imbalance.")

    # 2. GroupKFold (group by email domain)
    print(f"\n  [GroupKFold] Group by email domain")
    print(f"    Ensures same domain NEVER in both train and test fold")
    print(f"    Real-world: prevents leakage from same user/organization")
    groups = X_train['email_domain'].values
    gkf = GroupKFold(n_splits=5)
    scores_group = cross_val_score(model_cv, X_full_train, y_train,
                                    cv=gkf, groups=groups, scoring='f1')
    print(f"    F1: {scores_group.mean():.4f} +/- {scores_group.std():.4f}")
    print(f"    When to use: Users/groups that shouldn't leak between folds.")

    # 3. TimeSeriesSplit (simulated temporal order)
    print(f"\n  [TimeSeriesSplit] Simulated temporal order")
    print(f"    Always trains on 'past', predicts on 'future'")
    print(f"    (Using document index as proxy for time)")
    tss = TimeSeriesSplit(n_splits=5)
    scores_time = cross_val_score(model_cv, X_full_train, y_train,
                                   cv=tss, scoring='f1')
    print(f"    F1: {scores_time.mean():.4f} +/- {scores_time.std():.4f}")
    print(f"    When to use: Time-series data, stock prices, forecasting.")

    # Compare
    print(f"\n  Comparison:")
    print(f"  {'Strategy':<20s}  {'Mean F1':>8s}  {'Std':>8s}  {'Note'}")
    print(f"  {'-'*60}")
    print(f"  {'StratifiedKFold':<20s}  {scores_strat.mean():>8.4f}  "
          f"{scores_strat.std():>8.4f}  Standard baseline")
    print(f"  {'GroupKFold':<20s}  {scores_group.mean():>8.4f}  "
          f"{scores_group.std():>8.4f}  More conservative (realistic)")
    print(f"  {'TimeSeriesSplit':<20s}  {scores_time.mean():>8.4f}  "
          f"{scores_time.std():>8.4f}  Most conservative")

    diff = scores_strat.mean() - scores_group.mean()
    if diff > 0.01:
        print(f"\n  -> GroupKFold gives LOWER scores than StratifiedKFold")
        print(f"     This means the model was 'cheating' by seeing patterns")
        print(f"     from the same domain in both train and test!")
    else:
        print(f"\n  -> Scores are similar -- model generalizes well across groups")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    data_cv = [scores_strat, scores_group, scores_time]
    bp = ax.boxplot(data_cv, labels=['StratifiedKFold', 'GroupKFold', 'TimeSeriesSplit'],
                     patch_artist=True)
    colors_cv = ['steelblue', '#e74c3c', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors_cv):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("F1 Score")
    ax.set_title("Cross-Validation Strategy Comparison")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_cv_variants.png", dpi=150)
    plt.close()
    print("  [Saved] plots/05_cv_variants.png")

    # ==================================================================
    # STEP 7: DATA DRIFT DETECTION & MONITORING
    # ==================================================================
    print_section("STEP 7: DATA DRIFT DETECTION & MONITORING")

    print(f"\n  Why monitor for drift?")
    print(f"    - In production, data changes over time")
    print(f"    - New topics, new writing styles, new domains")
    print(f"    - Model accuracy degrades if not monitored")
    print(f"    - Need automated alerts when drift is detected")

    # Simulate drift: modify test data to represent 'production' data
    print(f"\n  Simulating production data with drift:")
    drift_texts = X_test['body'].copy()
    # Truncate texts (simulates shorter posts over time)
    drift_texts = drift_texts.apply(lambda x: x[:max(len(x)//3, 50)])
    # Add some noise domains
    drift_domains = X_test['email_domain'].copy()
    n_swap = len(drift_domains) // 5
    drift_domains.iloc[:n_swap] = "newdomain.ai"

    print(f"    - Truncated text to 1/3 length (simulates shorter posts)")
    print(f"    - Replaced {n_swap} domains with 'newdomain.ai' (new source)")

    # Detect drift with KS test on text statistics
    print(f"\n  [Method 1] KS Test on Feature Distributions")
    print(f"    H0: Train and production distributions are the same")
    print(f"    p < 0.05 -> reject H0 -> DRIFT DETECTED")

    train_stats_raw = compute_text_stats(X_train['body'])
    drift_stats = compute_text_stats(drift_texts)

    drift_results = []
    print(f"\n    {'Feature':<20s}  {'KS Stat':>8s}  {'p-value':>8s}  {'Status'}")
    print(f"    {'-'*55}")
    for col in train_stats_raw.columns:
        stat, pval = ks_2samp(train_stats_raw[col].values, drift_stats[col].values)
        status = "DRIFT!" if pval < 0.05 else "stable"
        drift_results.append({"Feature": col, "KS": stat, "p": pval, "Drift": pval < 0.05})
        print(f"    {col:<20s}  {stat:>8.4f}  {pval:>8.4f}  {status}")

    n_drift = sum(1 for r in drift_results if r["Drift"])
    print(f"\n    Drift detected in {n_drift}/{len(drift_results)} features")

    # PSI (Population Stability Index)
    print(f"\n  [Method 2] PSI (Population Stability Index)")
    print(f"    PSI < 0.1:  No significant drift")
    print(f"    PSI 0.1-0.25: Moderate drift (investigate)")
    print(f"    PSI > 0.25: Significant drift (retrain!)")

    for col in ['word_count', 'char_count', 'avg_word_len']:
        psi = calculate_psi(train_stats_raw[col].values, drift_stats[col].values)
        level = "OK" if psi < 0.1 else ("MODERATE" if psi < 0.25 else "SIGNIFICANT")
        print(f"    {col:<20s}  PSI = {psi:.4f}  [{level}]")

    # Prediction confidence monitoring
    print(f"\n  [Method 3] Prediction Confidence Monitoring")
    drift_tfidf = tfidf.transform(drift_texts)
    drift_proba = prob_model.predict_proba(drift_tfidf)[:, 1]
    original_proba = y_proba

    print(f"    Original data -- mean confidence: {np.mean(np.abs(original_proba - 0.5)):.4f}")
    print(f"    Drifted data  -- mean confidence: {np.mean(np.abs(drift_proba - 0.5)):.4f}")

    conf_drop = (np.mean(np.abs(original_proba - 0.5)) -
                 np.mean(np.abs(drift_proba - 0.5)))
    if conf_drop > 0.02:
        print(f"    -> Model less confident on drifted data (confidence drop: {conf_drop:.4f})")
        print(f"    -> ALERT: Consider retraining!")
    else:
        print(f"    -> Model confidence stable")

    # Plot drift
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Feature distribution shift
    axes[0].hist(train_stats_raw['word_count'], bins=30, alpha=0.6,
                  label='Training', color='steelblue', density=True)
    axes[0].hist(drift_stats['word_count'], bins=30, alpha=0.6,
                  label='Production (drifted)', color='#e74c3c', density=True)
    axes[0].set_title("Word Count Distribution Shift")
    axes[0].legend()

    # Prediction confidence
    axes[1].hist(original_proba, bins=30, alpha=0.6,
                  label='Original', color='steelblue', density=True)
    axes[1].hist(drift_proba, bins=30, alpha=0.6,
                  label='Drifted', color='#e74c3c', density=True)
    axes[1].set_title("Prediction Probability Shift")
    axes[1].legend()

    # PSI by feature
    psi_vals = []
    for col in train_stats_raw.columns:
        psi_vals.append(calculate_psi(
            train_stats_raw[col].values, drift_stats[col].values))
    axes[2].barh(list(train_stats_raw.columns), psi_vals, color='steelblue')
    axes[2].axvline(x=0.1, color='orange', linestyle='--', label='Moderate (0.1)')
    axes[2].axvline(x=0.25, color='red', linestyle='--', label='Significant (0.25)')
    axes[2].set_xlabel("PSI")
    axes[2].set_title("Population Stability Index by Feature")
    axes[2].legend(fontsize=8)

    plt.suptitle("Data Drift Detection", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_data_drift.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/06_data_drift.png")

    # ==================================================================
    # STEP 8: MODEL DEPLOYMENT (Flask API)
    # ==================================================================
    print_section("STEP 8: MODEL DEPLOYMENT")

    print(f"\n  Saving deployment artifacts...")
    deploy_dir = PLOTS_DIR.parent

    # Save model + vectorizer
    best_lr = LogisticRegression(max_iter=1000, random_state=42)
    best_lr.fit(X_tfidf_train, y_train)
    joblib.dump(best_lr, deploy_dir / "model.joblib")
    joblib.dump(tfidf, deploy_dir / "tfidf_vectorizer.joblib")
    print(f"    Model saved:      {deploy_dir / 'model.joblib'}")
    print(f"    Vectorizer saved: {deploy_dir / 'tfidf_vectorizer.joblib'}")

    # Generate Flask API file
    api_code = '''"""
Model Deployment API -- Flask REST API
=======================================
Serves the trained text classification model as a REST API.

Setup:
  pip install flask

Usage:
  python deploy_api.py

Test with curl:
  curl -X POST http://localhost:5000/predict \\
    -H "Content-Type: application/json" \\
    -d "{\\"text\\": \\"The new GPU from NVIDIA has amazing compute performance\\"}"

Test with Python:
  import requests
  resp = requests.post('http://localhost:5000/predict',
                       json={'text': 'The new GPU has great performance'})
  print(resp.json())
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Load model + vectorizer (saved by advanced_ml_pipeline.py)
MODEL_DIR = Path(__file__).parent
model = joblib.load(MODEL_DIR / "model.joblib")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model': 'text-classifier-v1'})


@app.route('/predict', methods=['POST'])
def predict():
    """Predict whether text is tech-related or not."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request body'}), 400

    text = data['text']
    threshold = data.get('threshold', 0.5)  # Allow custom threshold!

    # Vectorize + predict
    features = vectorizer.transform([text])
    probability = model.predict_proba(features)[0]
    prediction = int(probability[1] >= threshold)

    return jsonify({
        'prediction': prediction,
        'label': 'tech' if prediction == 1 else 'non-tech',
        'confidence': float(max(probability)),
        'probability_tech': float(probability[1]),
        'threshold_used': threshold,
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction for multiple texts."""
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({'error': 'Missing "texts" field'}), 400

    texts = data['texts']
    threshold = data.get('threshold', 0.5)

    features = vectorizer.transform(texts)
    probabilities = model.predict_proba(features)
    predictions = (probabilities[:, 1] >= threshold).astype(int)

    results = []
    for text, pred, prob in zip(texts, predictions, probabilities):
        results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': int(pred),
            'label': 'tech' if pred else 'non-tech',
            'probability_tech': float(prob[1]),
        })
    return jsonify({'predictions': results, 'count': len(results)})


if __name__ == '__main__':
    print("Starting Text Classification API...")
    print("Endpoints:")
    print("  GET  /health         -- Health check")
    print("  POST /predict        -- Single prediction")
    print("  POST /predict_batch  -- Batch predictions")
    print()
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    api_path = deploy_dir / "deploy_api.py"
    with open(api_path, 'w') as f:
        f.write(api_code)
    print(f"    Flask API saved:  {api_path}")

    print(f"\n  Deployment Architecture:")
    print(f"    [Client] --POST /predict--> [Flask API] --model.predict()--> [Response]")
    print(f"")
    print(f"    Endpoints:")
    print(f"      GET  /health        -- Health check (load balancer)")
    print(f"      POST /predict       -- Single text prediction")
    print(f"      POST /predict_batch -- Batch predictions")
    print(f"")
    print(f"    Features:")
    print(f"      - Custom threshold per request")
    print(f"      - Batch predictions for bulk processing")
    print(f"      - Confidence scores in response")

    print(f"\n  To run the API:")
    print(f"    pip install flask")
    print(f"    python deploy_api.py")
    print(f"    # Then send requests to http://localhost:5000/predict")

    # Simulate API call
    print(f"\n  Simulated API call:")
    test_texts = [
        "The new NVIDIA GPU has 16GB VRAM and supports CUDA 12",
        "The hockey game last night was incredible, 5-3 final score",
    ]
    features = tfidf.transform(test_texts)
    probs = best_lr.predict_proba(features)
    for text, prob in zip(test_texts, probs):
        label = "tech" if prob[1] >= 0.5 else "non-tech"
        print(f"    Input: '{text[:60]}...'")
        print(f"    -> {label} (confidence: {max(prob):.4f})")

    print(f"\n  Production Considerations:")
    print(f"    - Containerize with Docker for deployment")
    print(f"    - Add logging for monitoring predictions")
    print(f"    - Implement rate limiting")
    print(f"    - Add data drift monitoring (from Step 7)")
    print(f"    - Set up model versioning (MLflow, DVC)")
    print(f"    - Use gunicorn/uvicorn instead of Flask dev server")

    # ==================================================================
    # STEP 9: SUMMARY
    # ==================================================================
    print_section("FINAL SUMMARY")
    print("""
    ADVANCED TECHNIQUES PRACTICED IN THIS PIPELINE:
    -----------------------------------------------
    1.  TEXT / NLP FEATURES       -- TF-IDF, CountVectorizer, text statistics
    2.  HIGH-CARDINALITY ENCODING -- Frequency, Target, Hashing trick
    3.  THRESHOLD TUNING          -- PR curve, ROC curve, optimal cutoffs
    4.  CV VARIANTS               -- StratifiedKFold, GroupKFold, TimeSeriesSplit
    5.  DATA DRIFT DETECTION      -- KS test, PSI, confidence monitoring
    6.  MODEL DEPLOYMENT          -- Flask REST API, batch predict, health check

    COMBINED WITH PREVIOUS PIPELINES, YOU NOW COVER:
    -----------------------------------------------
    Classification (Titanic):   Data cleaning, EDA, 9 classifiers, class imbalance,
                                PCA/KMeans/DBSCAN, feature selection, stacking, SHAP
    Regression (Ames Housing):  80+ features, log transform, 11 regressors,
                                residual analysis, feature selection, stacking, SHAP
    Advanced (This pipeline):   NLP features, high-cardinality encoding, threshold
                                tuning, CV variants, data drift, deployment

    --> NOTHING LEFT OUT. You're ready for any tabular/text ML project!
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("Deployment files saved to:", deploy_dir.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
