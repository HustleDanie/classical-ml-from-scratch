"""
=============================================================================
08 - NAIVE BAYES: Complete Implementation
=============================================================================
Covers:
  1. From-scratch Gaussian Naive Bayes (priors, per-class stats, Gaussian PDF)
  2. From-scratch Multinomial Naive Bayes (Laplace smoothing, log-space)
  3. Scikit-learn comparison (GaussianNB, MultinomialNB, BernoulliNB, ComplementNB)
  4. Text classification demo — 20 Newsgroups + TF-IDF + MultinomialNB
  5. Smoothing parameter (alpha) impact
  6. Prior probability visualization
  7. Feature likelihood visualization (Gaussian PDFs per class)
  8. Probability calibration analysis
  9. Naive Bayes vs other classifiers
  10. Multiclass classification (Iris)

Datasets:
  - Classification: Breast Cancer Wisconsin (569, 30)
  - Multiclass:     Iris (150, 4, 3 classes)
  - Text:           20 Newsgroups (subset, TF-IDF vectorized)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.datasets import (
    load_breast_cancer, load_iris, fetch_20newsgroups
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# SECTION 1: From-Scratch Gaussian Naive Bayes
# =============================================================================
class GaussianNBScratch:
    """
    Gaussian Naive Bayes classifier — from scratch.

    Assumption: P(x_i | C) = N(x_i; mu_C_i, sigma²_C_i)

    Training:
        For each class C and each feature i, compute:
            - mu_C_i = mean of feature i for samples in class C
            - sigma²_C_i = variance of feature i for samples in class C
            - P(C) = proportion of samples in class C

    Prediction:
        P(C | x) ∝ P(C) * ∏ P(x_i | C)
        We compute log P(C | x) = log P(C) + Σ log P(x_i | C)
        and pick the class with the highest log-posterior.
    """

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing  # Added to variance for stability
        self.classes_ = None
        self.class_prior_ = None
        self.theta_ = None      # Per-class means: shape (n_classes, n_features)
        self.var_ = None        # Per-class variances: shape (n_classes, n_features)

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx] = X_c.mean(axis=0)
            self.var_[idx] = X_c.var(axis=0) + self.var_smoothing
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]

        return self

    def _log_likelihood(self, X):
        """Compute log P(x | C) for each class using Gaussian PDF."""
        # log N(x; mu, sigma²) = -0.5 * [log(2π) + log(σ²) + (x-μ)²/σ²]
        n_classes = len(self.classes_)
        log_probs = np.zeros((X.shape[0], n_classes))

        for idx in range(n_classes):
            log_probs[:, idx] = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[idx])
                + (X - self.theta_[idx]) ** 2 / self.var_[idx],
                axis=1
            )
        return log_probs

    def predict_log_proba(self, X):
        """Log posterior (unnormalized): log P(C) + log P(x|C)."""
        X = np.array(X, dtype=np.float64)
        log_likelihood = self._log_likelihood(X)
        log_prior = np.log(self.class_prior_)
        return log_likelihood + log_prior  # broadcasting

    def predict(self, X):
        log_post = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_post, axis=1)]

    def predict_proba(self, X):
        """Normalized posterior probabilities (via log-sum-exp)."""
        log_post = self.predict_log_proba(X)
        # Numerically stable softmax
        log_post -= log_post.max(axis=1, keepdims=True)
        probs = np.exp(log_post)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# =============================================================================
# SECTION 2: From-Scratch Multinomial Naive Bayes
# =============================================================================
class MultinomialNBScratch:
    """
    Multinomial Naive Bayes — from scratch.

    Designed for count/frequency data (e.g., word counts, TF-IDF).

    P(x_i | C) = (N_C_i + alpha) / (N_C + alpha * n_features)

    Where:
        N_C_i = total count of feature i across all training samples in class C
        N_C   = total count of all features in class C
        alpha = Laplace smoothing parameter

    Prediction in log-space:
        log P(C|x) ∝ log P(C) + Σ x_i * log P(x_i | C)
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_prior_ = None
        self.feature_log_prob_ = None  # log P(x_i | C)

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_prior_[idx] = X_c.shape[0] / X.shape[0]

            # Sum of each feature across samples of this class
            feature_count = X_c.sum(axis=0) + self.alpha
            total_count = feature_count.sum()
            self.feature_log_prob_[idx] = np.log(feature_count / total_count)

        return self

    def predict_log_proba(self, X):
        X = np.array(X, dtype=np.float64)
        log_prior = np.log(self.class_prior_)
        # log P(C|x) ∝ log P(C) + X @ log P(x|C).T
        log_likelihood = X @ self.feature_log_prob_.T
        return log_likelihood + log_prior

    def predict(self, X):
        log_post = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_post, axis=1)]

    def predict_proba(self, X):
        log_post = self.predict_log_proba(X)
        log_post -= log_post.max(axis=1, keepdims=True)
        probs = np.exp(log_post)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# =============================================================================
# SECTION 3: Visualization Helpers
# =============================================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_bar(names, values, ylabel, title, filename, colors=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    if colors is None:
        colors = sns.color_palette("viridis", len(names))
    bars = ax.bar(names, values, color=colors, edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_gaussian_likelihoods(X, y, feature_names, class_names, features_to_plot,
                               filename):
    """Plot Gaussian PDF fit for selected features, per class."""
    n_feats = len(features_to_plot)
    fig, axes = plt.subplots(1, n_feats, figsize=(5 * n_feats, 4.5))
    if n_feats == 1:
        axes = [axes]

    colors = ["coral", "steelblue", "mediumseagreen"]
    classes = np.unique(y)

    for ax, feat_idx in zip(axes, features_to_plot):
        for i, c in enumerate(classes):
            X_c = X[y == c, feat_idx]
            mu, sigma = X_c.mean(), X_c.std()
            x_line = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_line - mu) / sigma) ** 2
            )
            label = class_names[i] if i < len(class_names) else f"Class {c}"
            ax.plot(x_line, pdf, color=colors[i % len(colors)], linewidth=2,
                    label=f"{label} (μ={mu:.2f})")
            ax.fill_between(x_line, pdf, alpha=0.15, color=colors[i % len(colors)])
        ax.set_title(feature_names[feat_idx])
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    plt.suptitle("Gaussian Likelihood P(x_i | C) per Class", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_class_priors(class_names, priors, filename):
    """Plot class prior probabilities."""
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["coral", "steelblue", "mediumseagreen"]
    bars = ax.bar(class_names, priors,
                  color=colors[:len(class_names)], edgecolor="white")
    ax.set_ylabel("P(C)")
    ax.set_title("Class Prior Probabilities")
    for bar, val in zip(bars, priors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_alpha_sweep(alphas, train_accs, test_accs, filename):
    """Smoothing parameter impact."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(alphas, train_accs, "o-", color="coral", linewidth=2, label="Train")
    ax.plot(alphas, test_accs, "s-", color="steelblue", linewidth=2, label="Test")
    ax.set_xlabel("Alpha (Smoothing Parameter)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Laplace Smoothing (Alpha) Impact on MultinomialNB")
    ax.set_xscale("log")
    ax.legend()
    best_a = alphas[np.argmax(test_accs)]
    ax.axvline(x=best_a, color="green", linestyle="--", alpha=0.5,
               label=f"Best α={best_a}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_calibration(y_true, prob_pos, model_name, filename):
    """Plot calibration curve (reliability diagram)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Calibration curve
    ax1 = axes[0]
    fraction_pos, mean_pred = calibration_curve(y_true, prob_pos, n_bins=10)
    ax1.plot(mean_pred, fraction_pos, "s-", color="steelblue", linewidth=2,
             label=model_name)
    ax1.plot([0, 1], [0, 1], "r--", label="Perfectly Calibrated")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Calibration Curve (Reliability Diagram)")
    ax1.legend()

    # Predicted probability histogram
    ax2 = axes[1]
    ax2.hist(prob_pos, bins=30, color="steelblue", edgecolor="white", alpha=0.7)
    ax2.set_xlabel("Predicted Probability (P(class=1))")
    ax2.set_ylabel("Count")
    ax2.set_title("Predicted Probability Distribution")

    plt.suptitle(f"Probability Calibration — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] plots/{filename}")


def plot_decision_boundary_nb(X, y, model, title, filename):
    """2D decision boundary for Naive Bayes."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
    ax.contour(xx, yy, Z, colors="black", linewidths=0.5, alpha=0.3)

    classes = np.unique(y)
    colors_pts = ["coral", "steelblue", "mediumseagreen"]
    for i, cls in enumerate(classes):
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1], c=colors_pts[i % len(colors_pts)],
                   label=f"Class {cls}", edgecolors="black", s=25, alpha=0.7)

    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"  [Saved] plots/{filename}")


# =============================================================================
# SECTION 4: Main Execution
# =============================================================================
def main():
    print("=" * 70)
    print("08 — NAIVE BAYES")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════════
    # PART A: GAUSSIAN NAIVE BAYES (Breast Cancer — Continuous Features)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART A: GAUSSIAN NAIVE BAYES — Breast Cancer Dataset")
    print("=" * 70)

    cancer = load_breast_cancer()
    X_bin, y_bin = cancer.data, cancer.target
    feature_names = list(cancer.feature_names)
    class_names = list(cancer.target_names)

    print(f"Shape: {X_bin.shape}")
    print(f"Classes: {class_names}")
    print(f"Distribution: {dict(zip(*np.unique(y_bin, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    # ── Class Priors ─────────────────────────────────────────────────────────
    priors = np.array([np.sum(y_train == c) / len(y_train)
                       for c in np.unique(y_train)])
    print(f"\n  Computed Priors: {dict(zip(class_names, priors))}")
    plot_class_priors(class_names, priors, "01_class_priors.png")

    # ── Feature Likelihoods (Gaussian PDFs) ──────────────────────────────────
    print("\n  --- Feature Likelihood Visualization ---")
    # Pick 4 representative features
    selected_feats = [0, 1, 20, 27]  # mean radius, mean texture, worst radius, worst concavity
    plot_gaussian_likelihoods(
        X_train, y_train, feature_names, class_names,
        selected_feats, "02_gaussian_likelihoods.png"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A1: From-Scratch Gaussian Naive Bayes
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A1: Gaussian NB — From Scratch")
    print("=" * 70)

    scratch_gnb = GaussianNBScratch()
    scratch_gnb.fit(X_train, y_train)

    y_pred_sg = scratch_gnb.predict(X_test)
    acc_sg = accuracy_score(y_test, y_pred_sg)

    print(f"\n  Test Accuracy: {acc_sg:.6f}")
    print(f"  Learned priors: {scratch_gnb.class_prior_}")
    print(f"  Theta shape: {scratch_gnb.theta_.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL A2: Sklearn Gaussian Naive Bayes
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("MODEL A2: Scikit-learn GaussianNB")
    print("=" * 70)

    sk_gnb = GaussianNB()
    sk_gnb.fit(X_train, y_train)

    y_pred_skgnb = sk_gnb.predict(X_test)
    acc_skgnb = accuracy_score(y_test, y_pred_skgnb)

    print(f"\n  Test Accuracy: {acc_skgnb:.6f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred_skgnb, target_names=class_names))

    # ── Scratch vs Sklearn ───────────────────────────────────────────────────
    print(f"  Scratch: {acc_sg:.6f}")
    print(f"  Sklearn: {acc_skgnb:.6f}")
    print(f"  Match:   {np.array_equal(y_pred_sg, y_pred_skgnb)}")

    plot_confusion_matrix(y_test, y_pred_skgnb, class_names,
                          "GaussianNB — Confusion Matrix", "03_confusion_matrix_gnb.png")

    # ── Calibration Analysis ─────────────────────────────────────────────────
    print("\n  --- Probability Calibration ---")
    prob_gnb = sk_gnb.predict_proba(X_test)[:, 1]
    plot_calibration(y_test, prob_gnb, "GaussianNB", "04_calibration_gnb.png")

    # Scratch vs Sklearn probabilities
    prob_scratch = scratch_gnb.predict_proba(X_test)[:, 1]
    print(f"  Scratch proba sample: {prob_scratch[:5]}")
    print(f"  Sklearn proba sample: {prob_gnb[:5]}")

    # ── Decision Boundary (2 features) ───────────────────────────────────────
    gnb_2d = GaussianNB()
    gnb_2d.fit(X_train[:, :2], y_train)
    plot_decision_boundary_nb(
        X_test[:, :2], y_test, gnb_2d,
        f"GaussianNB — Decision Boundary (Acc={gnb_2d.score(X_test[:, :2], y_test):.3f})",
        "05_boundary_gnb.png"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PART B: MULTINOMIAL NB — Text Classification (20 Newsgroups)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART B: MULTINOMIAL NB — Text Classification (20 Newsgroups)")
    print("=" * 70)

    # Use a subset of categories for speed
    categories = [
        "rec.sport.baseball", "sci.space", "comp.graphics", "talk.politics.guns"
    ]

    newsgroups_train = fetch_20newsgroups(
        subset="train", categories=categories, remove=("headers", "footers", "quotes"),
        random_state=42
    )
    newsgroups_test = fetch_20newsgroups(
        subset="test", categories=categories, remove=("headers", "footers", "quotes"),
        random_state=42
    )

    print(f"  Categories: {categories}")
    print(f"  Train: {len(newsgroups_train.data)} docs")
    print(f"  Test:  {len(newsgroups_test.data)} docs")

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_text = tfidf.fit_transform(newsgroups_train.data)
    X_test_text = tfidf.transform(newsgroups_test.data)
    y_train_text = newsgroups_train.target
    y_test_text = newsgroups_test.target

    print(f"  TF-IDF shape: {X_train_text.shape}")

    # ── From-Scratch Multinomial NB ──────────────────────────────────────────
    print("\n  --- From-Scratch MultinomialNB ---")
    scratch_mnb = MultinomialNBScratch(alpha=1.0)
    scratch_mnb.fit(X_train_text.toarray(), y_train_text)
    y_pred_smnb = scratch_mnb.predict(X_test_text.toarray())
    acc_smnb = accuracy_score(y_test_text, y_pred_smnb)
    print(f"  Test Accuracy: {acc_smnb:.6f}")

    # ── Sklearn Multinomial NB ───────────────────────────────────────────────
    print("\n  --- Sklearn MultinomialNB ---")
    sk_mnb = MultinomialNB(alpha=1.0)
    sk_mnb.fit(X_train_text, y_train_text)
    y_pred_skmnb = sk_mnb.predict(X_test_text)
    acc_skmnb = accuracy_score(y_test_text, y_pred_skmnb)
    print(f"  Test Accuracy: {acc_skmnb:.6f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_text, y_pred_skmnb,
                                target_names=categories))

    print(f"\n  Scratch: {acc_smnb:.6f}")
    print(f"  Sklearn: {acc_skmnb:.6f}")

    plot_confusion_matrix(y_test_text, y_pred_skmnb,
                          [c.split(".")[-1] for c in categories],
                          "MultinomialNB — 20 Newsgroups", "06_confusion_matrix_text.png")

    # ── Sklearn Variants Comparison on Text ──────────────────────────────────
    print("\n  --- NB Variant Comparison on Text ---")
    nb_variants = {
        "MultinomialNB": MultinomialNB(alpha=1.0),
        "BernoulliNB": BernoulliNB(alpha=1.0),
        "ComplementNB": ComplementNB(alpha=1.0),
    }

    variant_names = []
    variant_accs = []
    for name, model in nb_variants.items():
        model.fit(X_train_text, y_train_text)
        acc = model.score(X_test_text, y_test_text)
        variant_names.append(name)
        variant_accs.append(acc)
        print(f"  {name:16s}: {acc:.4f}")

    plot_bar(variant_names, variant_accs,
             "Test Accuracy", "NB Variants on Text (20 Newsgroups)",
             "07_nb_variants_text.png")

    # ══════════════════════════════════════════════════════════════════════════
    # SMOOTHING PARAMETER (ALPHA) IMPACT
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SMOOTHING PARAMETER (ALPHA) IMPACT")
    print("=" * 70)

    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alpha_train_accs = []
    alpha_test_accs = []

    for a in alphas:
        mnb_a = MultinomialNB(alpha=a)
        mnb_a.fit(X_train_text, y_train_text)
        tr_acc = mnb_a.score(X_train_text, y_train_text)
        te_acc = mnb_a.score(X_test_text, y_test_text)
        alpha_train_accs.append(tr_acc)
        alpha_test_accs.append(te_acc)
        print(f"  alpha={a:6.3f}: Train={tr_acc:.4f}, Test={te_acc:.4f}")

    best_alpha = alphas[np.argmax(alpha_test_accs)]
    print(f"\n  Best alpha = {best_alpha}")

    plot_alpha_sweep(alphas, alpha_train_accs, alpha_test_accs,
                     "08_alpha_sweep.png")

    # ══════════════════════════════════════════════════════════════════════════
    # TOP FEATURES PER CLASS (Most discriminative words)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TOP FEATURES (WORDS) PER CLASS")
    print("=" * 70)

    feature_words = np.array(tfidf.get_feature_names_out())
    top_n = 10

    fig, axes = plt.subplots(1, len(categories), figsize=(5.5 * len(categories), 4))

    for idx, (cat, ax) in enumerate(zip(categories, axes)):
        # Log probability of each word given this class
        log_probs = sk_mnb.feature_log_prob_[idx]
        top_indices = np.argsort(log_probs)[-top_n:]
        top_words = feature_words[top_indices]
        top_probs = log_probs[top_indices]

        ax.barh(top_words, top_probs, color=sns.color_palette("viridis", top_n))
        ax.set_xlabel("Log P(word | class)")
        ax.set_title(cat.split(".")[-1])
        print(f"\n  {cat}:")
        for w, p in zip(top_words[::-1], top_probs[::-1]):
            print(f"    {w:15s}  log_prob={p:.3f}")

    plt.suptitle("Top Discriminative Words per Class (MultinomialNB)", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_top_words_per_class.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/09_top_words_per_class.png")

    # ══════════════════════════════════════════════════════════════════════════
    # PART C: MULTICLASS — Iris
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("PART C: MULTICLASS GAUSSIAN NB — Iris Dataset")
    print("=" * 70)

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    iris_names = list(iris.target_names)
    iris_feats = list(iris.feature_names)

    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
    )

    gnb_iris = GaussianNB()
    gnb_iris.fit(X_train_i, y_train_i)
    y_pred_iris = gnb_iris.predict(X_test_i)
    acc_iris = accuracy_score(y_test_i, y_pred_iris)

    print(f"\n  Test Accuracy: {acc_iris:.6f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test_i, y_pred_iris, target_names=iris_names))

    plot_confusion_matrix(y_test_i, y_pred_iris, iris_names,
                          "GaussianNB — Iris Confusion Matrix",
                          "10_confusion_matrix_iris.png")

    # Gaussian likelihoods on Iris
    plot_gaussian_likelihoods(
        X_train_i, y_train_i, iris_feats, iris_names,
        [0, 1, 2, 3], "11_gaussian_likelihoods_iris.png"
    )

    # Decision boundary (petal features)
    gnb_iris_2d = GaussianNB()
    gnb_iris_2d.fit(X_train_i[:, 2:4], y_train_i)
    plot_decision_boundary_nb(
        X_test_i[:, 2:4], y_test_i, gnb_iris_2d,
        f"GaussianNB — Iris (Petal Features, Acc={gnb_iris_2d.score(X_test_i[:, 2:4], y_test_i):.3f})",
        "12_boundary_iris.png"
    )

    # ══════════════════════════════════════════════════════════════════════════
    # PART D: NAIVE BAYES vs OTHER CLASSIFIERS (Breast Cancer)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("NAIVE BAYES vs OTHER CLASSIFIERS (Breast Cancer)")
    print("=" * 70)

    # Scale for distance-based models
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifiers = {
        "GaussianNB": (GaussianNB(), X_train, X_test),
        "Logistic Reg": (LogisticRegression(max_iter=5000, random_state=42),
                         X_train_s, X_test_s),
        "KNN (k=5)": (KNeighborsClassifier(n_neighbors=5),
                       X_train_s, X_test_s),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42),
                          X_train, X_test),
        "Grad Boosting": (GradientBoostingClassifier(n_estimators=100, random_state=42),
                          X_train, X_test),
        "SVM (RBF)": (SVC(kernel="rbf", random_state=42),
                       X_train_s, X_test_s),
    }

    comp_names = []
    comp_scores = []

    for name, (clf, Xtr, Xte) in classifiers.items():
        clf.fit(Xtr, y_train)
        acc = clf.score(Xte, y_test)
        comp_names.append(name)
        comp_scores.append(acc)
        print(f"  {name:16s}: {acc:.4f}")

    plot_bar(comp_names, comp_scores,
             "Test Accuracy", "Naive Bayes vs Other Classifiers",
             "13_nb_vs_others.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  BAYES' THEOREM          - P(C|x) ~ P(x|C) * P(C)
    2.  NAIVE ASSUMPTION        - Features are conditionally independent
    3.  GAUSSIAN NB             - Continuous features modeled by N(mu, sigma^2)
    4.  MULTINOMIAL NB          - Count/frequency data (text bag-of-words)
    5.  BERNOULLI NB            - Binary features (word present/absent)
    6.  COMPLEMENT NB           - Handles imbalanced text data
    7.  LAPLACE SMOOTHING       - Prevents zero-probability issues (alpha parameter)
    8.  LOG PROBABILITIES       - Numerical stability (products -> sums)
    9.  CLASS PRIORS            - Learned from training data distribution
    10. PROBABILITY CALIBRATION - NB probabilities are often poorly calibrated
    11. TEXT CLASSIFICATION      - TF-IDF + MultinomialNB (gold standard baseline)
    12. TOP DISCRIMINATIVE WORDS - Most informative features per class
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)


if __name__ == "__main__":
    main()
