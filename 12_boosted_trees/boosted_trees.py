"""
=============================================================================
12 - Boosted Trees (XGBoost & LightGBM): Complete Implementation
=============================================================================
Covers:
  1.  XGBoost Classifier     — Breast Cancer binary classification
  2.  LightGBM Classifier    — Same dataset for head-to-head comparison
  3.  Early Stopping          — Learning curves: loss vs boosting round
  4.  XGBoost Regressor       — California Housing regression
  5.  LightGBM Regressor      — Same dataset for comparison
  6.  Feature Importance       — gain, split, SHAP values
  7.  Hyperparameter sweep     — max_depth / num_leaves effect
  8.  Learning Rate trade-off  — eta vs n_estimators
  9.  Framework Comparison     — XGBoost vs LightGBM vs sklearn GBM
  10. Regularisation sweep     — L1 / L2 effect on overfitting

Datasets:
  - Breast Cancer Wisconsin (569, 30) — binary classification
  - California Housing (20640, 8) — regression
=============================================================================
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

# ── Setup ────────────────────────────────────────────────────────────────────
PLOTS_DIR = Path(__file__).parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# =============================================================================
# Helpers
# =============================================================================
def plot_confusion(y_true, y_pred, class_names, title, filename):
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


# =============================================================================
# SECTION 1: CLASSIFICATION — Breast Cancer
# =============================================================================
def classification_experiments():
    print("=" * 70)
    print("CLASSIFICATION — Breast Cancer Wisconsin")
    print("=" * 70)

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    feature_names = list(cancer.feature_names)
    class_names = list(cancer.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"  Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 1: XGBoost Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("MODEL 1: XGBoost Classifier")
    print("-" * 50)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        early_stopping_rounds=30,
    )

    t0 = time.time()
    xgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
    )
    xgb_time = time.time() - t0

    y_pred_xgb = xgb_clf.predict(X_test)
    y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_auc = roc_auc_score(y_test, y_proba_xgb)

    print(f"  Best iteration: {xgb_clf.best_iteration}")
    print(f"  Accuracy:  {xgb_acc:.4f}")
    print(f"  AUC-ROC:   {xgb_auc:.4f}")
    print(f"  Time:      {xgb_time:.3f}s")

    plot_confusion(y_test, y_pred_xgb, class_names,
                   f"XGBoost — Acc={xgb_acc:.4f}", "01_xgb_confusion.png")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 2: LightGBM Classifier
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("MODEL 2: LightGBM Classifier")
    print("-" * 50)

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    callbacks_lgb = [
        lgb.early_stopping(30, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    t0 = time.time()
    lgb_clf.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_metric="logloss",
        callbacks=callbacks_lgb,
    )
    lgb_time = time.time() - t0

    y_pred_lgb = lgb_clf.predict(X_test)
    y_proba_lgb = lgb_clf.predict_proba(X_test)[:, 1]
    lgb_acc = accuracy_score(y_test, y_pred_lgb)
    lgb_auc = roc_auc_score(y_test, y_proba_lgb)

    print(f"  Best iteration: {lgb_clf.best_iteration_}")
    print(f"  Accuracy:  {lgb_acc:.4f}")
    print(f"  AUC-ROC:   {lgb_auc:.4f}")
    print(f"  Time:      {lgb_time:.3f}s")

    plot_confusion(y_test, y_pred_lgb, class_names,
                   f"LightGBM — Acc={lgb_acc:.4f}", "02_lgb_confusion.png")

    # ══════════════════════════════════════════════════════════════════════════
    # ROC Curves: XGBoost vs LightGBM
    # ══════════════════════════════════════════════════════════════════════════
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
    fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba_lgb)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_xgb, tpr_xgb, linewidth=2, label=f"XGBoost (AUC={xgb_auc:.4f})")
    ax.plot(fpr_lgb, tpr_lgb, linewidth=2, label=f"LightGBM (AUC={lgb_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — XGBoost vs LightGBM")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_roc_comparison.png", dpi=150)
    plt.close()
    print("  [Saved] plots/03_roc_comparison.png")

    # ══════════════════════════════════════════════════════════════════════════
    # EARLY STOPPING: Learning Curves
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("EARLY STOPPING — Learning Curves")
    print("-" * 50)

    # XGBoost learning curve
    xgb_results = xgb_clf.evals_result()
    xgb_train_loss = xgb_results["validation_0"]["logloss"]
    xgb_val_loss = xgb_results["validation_1"]["logloss"]

    # LightGBM learning curve
    lgb_results = lgb_clf.evals_result_
    lgb_train_loss = lgb_results["training"]["binary_logloss"]
    lgb_val_loss = lgb_results["valid_1"]["binary_logloss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(xgb_train_loss, label="Train", alpha=0.8)
    axes[0].plot(xgb_val_loss, label="Validation", alpha=0.8)
    axes[0].axvline(x=xgb_clf.best_iteration, color="red", linestyle="--",
                    alpha=0.6, label=f"Best iter={xgb_clf.best_iteration}")
    axes[0].set_xlabel("Boosting Round")
    axes[0].set_ylabel("Log Loss")
    axes[0].set_title("XGBoost — Learning Curve")
    axes[0].legend()

    axes[1].plot(lgb_train_loss, label="Train", alpha=0.8)
    axes[1].plot(lgb_val_loss, label="Validation", alpha=0.8)
    axes[1].axvline(x=lgb_clf.best_iteration_ - 1, color="red", linestyle="--",
                    alpha=0.6, label=f"Best iter={lgb_clf.best_iteration_}")
    axes[1].set_xlabel("Boosting Round")
    axes[1].set_ylabel("Log Loss")
    axes[1].set_title("LightGBM — Learning Curve")
    axes[1].legend()

    plt.suptitle("Early Stopping — Loss vs Boosting Round", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/04_learning_curves.png")

    # ══════════════════════════════════════════════════════════════════════════
    # FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("FEATURE IMPORTANCE")
    print("-" * 50)

    # XGBoost — gain-based importance
    xgb_imp = xgb_clf.feature_importances_
    xgb_idx = np.argsort(xgb_imp)[-15:]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(range(15), xgb_imp[xgb_idx], color="steelblue")
    axes[0].set_yticks(range(15))
    axes[0].set_yticklabels([feature_names[i] for i in xgb_idx], fontsize=8)
    axes[0].set_xlabel("Importance (gain)")
    axes[0].set_title("XGBoost — Top 15 Features")

    lgb_imp = lgb_clf.feature_importances_
    lgb_idx = np.argsort(lgb_imp)[-15:]

    axes[1].barh(range(15), lgb_imp[lgb_idx], color="coral")
    axes[1].set_yticks(range(15))
    axes[1].set_yticklabels([feature_names[i] for i in lgb_idx], fontsize=8)
    axes[1].set_xlabel("Importance (split count)")
    axes[1].set_title("LightGBM — Top 15 Features")

    plt.suptitle("Feature Importance Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/05_feature_importance.png")

    # XGBoost — multiple importance types
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, imp_type in zip(axes, ["weight", "gain", "cover"]):
        xgb.plot_importance(xgb_clf, ax=ax, importance_type=imp_type,
                            max_num_features=12, title=f"XGBoost — {imp_type}",
                            show_values=True)
        ax.tick_params(axis="y", labelsize=7)
    plt.suptitle("XGBoost — Importance Types (weight / gain / cover)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_xgb_importance_types.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/06_xgb_importance_types.png")

    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "X_tr": X_tr, "X_val": X_val, "y_tr": y_tr, "y_val": y_val,
        "feature_names": feature_names, "class_names": class_names,
        "xgb_acc": xgb_acc, "lgb_acc": lgb_acc,
        "xgb_time": xgb_time, "lgb_time": lgb_time,
    }


# =============================================================================
# SECTION 2: REGRESSION — California Housing
# =============================================================================
def regression_experiments():
    print("\n\n" + "=" * 70)
    print("REGRESSION — California Housing")
    print("=" * 70)

    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = list(housing.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"  Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 3: XGBoost Regressor
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("MODEL 3: XGBoost Regressor")
    print("-" * 50)

    xgb_reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
    )

    t0 = time.time()
    xgb_reg.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
    )
    xgb_reg_time = time.time() - t0

    y_pred_xgb = xgb_reg.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    xgb_r2 = r2_score(y_test, y_pred_xgb)

    print(f"  Best iteration: {xgb_reg.best_iteration}")
    print(f"  RMSE:  {xgb_rmse:.4f}")
    print(f"  R²:    {xgb_r2:.4f}")
    print(f"  Time:  {xgb_reg_time:.3f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL 4: LightGBM Regressor
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 50)
    print("MODEL 4: LightGBM Regressor")
    print("-" * 50)

    lgb_reg = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    callbacks_lgb = [
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    t0 = time.time()
    lgb_reg.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_metric="rmse",
        callbacks=callbacks_lgb,
    )
    lgb_reg_time = time.time() - t0

    y_pred_lgb = lgb_reg.predict(X_test)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))
    lgb_r2 = r2_score(y_test, y_pred_lgb)

    print(f"  Best iteration: {lgb_reg.best_iteration_}")
    print(f"  RMSE:  {lgb_rmse:.4f}")
    print(f"  R²:    {lgb_r2:.4f}")
    print(f"  Time:  {lgb_reg_time:.3f}s")

    # ── Actual vs Predicted ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(y_test, y_pred_xgb, s=6, alpha=0.3, color="steelblue")
    axes[0].plot([0, 5.5], [0, 5.5], "r--", linewidth=2)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"XGBoost — RMSE={xgb_rmse:.4f}, R²={xgb_r2:.4f}")

    axes[1].scatter(y_test, y_pred_lgb, s=6, alpha=0.3, color="coral")
    axes[1].plot([0, 5.5], [0, 5.5], "r--", linewidth=2)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"LightGBM — RMSE={lgb_rmse:.4f}, R²={lgb_r2:.4f}")

    plt.suptitle("Regression — Actual vs Predicted (California Housing)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_regression_pred_vs_actual.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/07_regression_pred_vs_actual.png")

    # Regression learning curves
    xgb_res = xgb_reg.evals_result()
    lgb_res = lgb_reg.evals_result_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(xgb_res["validation_0"]["rmse"], label="Train", alpha=0.8)
    axes[0].plot(xgb_res["validation_1"]["rmse"], label="Validation", alpha=0.8)
    axes[0].axvline(x=xgb_reg.best_iteration, color="red", linestyle="--",
                    alpha=0.6, label=f"Best={xgb_reg.best_iteration}")
    axes[0].set_xlabel("Boosting Round")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("XGBoost Regressor — Learning Curve")
    axes[0].legend()

    axes[1].plot(lgb_res["training"]["rmse"], label="Train", alpha=0.8)
    axes[1].plot(lgb_res["valid_1"]["rmse"], label="Validation", alpha=0.8)
    axes[1].axvline(x=lgb_reg.best_iteration_ - 1, color="red", linestyle="--",
                    alpha=0.6, label=f"Best={lgb_reg.best_iteration_}")
    axes[1].set_xlabel("Boosting Round")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("LightGBM Regressor — Learning Curve")
    axes[1].legend()

    plt.suptitle("Regression Learning Curves", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_regression_learning_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/08_regression_learning_curves.png")

    # Feature importance — regression
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    xgb_imp = xgb_reg.feature_importances_
    lgb_imp = lgb_reg.feature_importances_
    xidx = np.argsort(xgb_imp)
    lidx = np.argsort(lgb_imp)

    axes[0].barh(range(len(feature_names)), xgb_imp[xidx], color="steelblue")
    axes[0].set_yticks(range(len(feature_names)))
    axes[0].set_yticklabels([feature_names[i] for i in xidx])
    axes[0].set_title("XGBoost Regressor — Feature Importance")
    axes[0].set_xlabel("Importance (gain)")

    axes[1].barh(range(len(feature_names)), lgb_imp[lidx], color="coral")
    axes[1].set_yticks(range(len(feature_names)))
    axes[1].set_yticklabels([feature_names[i] for i in lidx])
    axes[1].set_title("LightGBM Regressor — Feature Importance")
    axes[1].set_xlabel("Importance (split count)")

    plt.suptitle("Regression Feature Importance — California Housing", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_regression_feature_importance.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/09_regression_feature_importance.png")

    return {
        "xgb_rmse": xgb_rmse, "lgb_rmse": lgb_rmse,
        "xgb_r2": xgb_r2, "lgb_r2": lgb_r2,
        "xgb_time": xgb_reg_time, "lgb_time": lgb_reg_time,
        "X_tr": X_tr, "X_val": X_val, "y_tr": y_tr, "y_val": y_val,
        "X_test": X_test, "y_test": y_test, "feature_names": feature_names,
    }


# =============================================================================
# SECTION 3: HYPERPARAMETER EXPERIMENTS
# =============================================================================
def hyperparameter_experiments(clf_data, reg_data):
    print("\n\n" + "=" * 70)
    print("HYPERPARAMETER EXPERIMENTS")
    print("=" * 70)

    X_tr = clf_data["X_tr"]
    X_val = clf_data["X_val"]
    y_tr = clf_data["y_tr"]
    y_val = clf_data["y_val"]
    X_test = clf_data["X_test"]
    y_test = clf_data["y_test"]

    # ── Experiment 1: max_depth / num_leaves ─────────────────────────────────
    print("\n  --- max_depth / num_leaves sweep ---")

    depths = [2, 3, 4, 5, 6, 8, 10, 15]
    xgb_depth_acc = []
    for d in depths:
        m = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=d,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            early_stopping_rounds=20,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        xgb_depth_acc.append(m.score(X_test, y_test))
        print(f"    XGB max_depth={d:2d}: Acc={xgb_depth_acc[-1]:.4f} "
              f"(trees={m.best_iteration})")

    leaves_list = [8, 15, 31, 50, 80, 127, 200, 400]
    lgb_leaf_acc = []
    for nl in leaves_list:
        m = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=nl,
            random_state=42, verbose=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric="logloss",
              callbacks=[lgb.early_stopping(20, verbose=False),
                         lgb.log_evaluation(period=0)])
        lgb_leaf_acc.append(m.score(X_test, y_test))
        print(f"    LGB num_leaves={nl:3d}: Acc={lgb_leaf_acc[-1]:.4f} "
              f"(trees={m.best_iteration_})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(depths, xgb_depth_acc, "o-", color="steelblue", linewidth=2,
                 markersize=7)
    axes[0].set_xlabel("max_depth")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_title("XGBoost — max_depth Sweep")
    axes[0].set_xticks(depths)

    axes[1].plot(leaves_list, lgb_leaf_acc, "s-", color="coral", linewidth=2,
                 markersize=7)
    axes[1].set_xlabel("num_leaves")
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("LightGBM — num_leaves Sweep")

    plt.suptitle("Tree Complexity — Effect on Accuracy", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_depth_leaves_sweep.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/10_depth_leaves_sweep.png")

    # ── Experiment 2: Learning Rate vs N Estimators ──────────────────────────
    print("\n  --- Learning Rate vs N Estimators ---")

    lr_configs = [
        (0.01, 2000),
        (0.05, 800),
        (0.1,  400),
        (0.2,  200),
        (0.5,  100),
    ]

    xgb_lr_results = []
    for lr, n_est in lr_configs:
        m = xgb.XGBClassifier(
            n_estimators=n_est, learning_rate=lr, max_depth=5,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            early_stopping_rounds=30,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        acc = m.score(X_test, y_test)
        xgb_lr_results.append((lr, m.best_iteration, acc))
        print(f"    lr={lr:.2f}, max_est={n_est}, best_iter={m.best_iteration}, "
              f"Acc={acc:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    lrs = [r[0] for r in xgb_lr_results]
    iters = [r[1] for r in xgb_lr_results]
    accs = [r[2] for r in xgb_lr_results]

    scatter = ax.scatter(lrs, iters, c=accs, cmap="RdYlGn", s=200,
                         edgecolors="black", linewidths=1, zorder=5)
    for lr_val, it, acc in xgb_lr_results:
        ax.annotate(f"Acc={acc:.3f}", (lr_val, it),
                    textcoords="offset points", xytext=(10, 5), fontsize=8)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Best N Estimators (early stopped)")
    ax.set_title("XGBoost — Learning Rate vs N Estimators Trade-off")
    ax.set_xscale("log")
    plt.colorbar(scatter, label="Test Accuracy")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_lr_vs_nestimators.png", dpi=150)
    plt.close()
    print("  [Saved] plots/11_lr_vs_nestimators.png")

    # ── Experiment 3: Regularisation Sweep ───────────────────────────────────
    print("\n  --- Regularisation Sweep ---")

    reg_values = [0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
    train_accs_l2 = []
    test_accs_l2 = []
    train_accs_l1 = []
    test_accs_l1 = []

    for rv in reg_values:
        # L2 sweep
        m = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            reg_lambda=rv, reg_alpha=0,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            early_stopping_rounds=20,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        train_accs_l2.append(m.score(X_tr, y_tr))
        test_accs_l2.append(m.score(X_test, y_test))

        # L1 sweep
        m2 = xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=5,
            reg_lambda=0, reg_alpha=rv,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            early_stopping_rounds=20,
        )
        m2.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        train_accs_l1.append(m2.score(X_tr, y_tr))
        test_accs_l1.append(m2.score(X_test, y_test))

        print(f"    reg={rv:5.2f}  L2: train={train_accs_l2[-1]:.4f} "
              f"test={test_accs_l2[-1]:.4f}  |  "
              f"L1: train={train_accs_l1[-1]:.4f} test={test_accs_l1[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(reg_values, train_accs_l2, "o--", label="Train", alpha=0.7)
    axes[0].plot(reg_values, test_accs_l2, "s-", label="Test", linewidth=2)
    axes[0].set_xlabel("reg_lambda (L2)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("L2 Regularisation Sweep")
    axes[0].legend()

    axes[1].plot(reg_values, train_accs_l1, "o--", label="Train", alpha=0.7)
    axes[1].plot(reg_values, test_accs_l1, "s-", label="Test", linewidth=2)
    axes[1].set_xlabel("reg_alpha (L1)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("L1 Regularisation Sweep")
    axes[1].legend()

    plt.suptitle("XGBoost — Regularisation Effect on Overfitting", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_regularisation_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/12_regularisation_sweep.png")


# =============================================================================
# SECTION 4: FRAMEWORK COMPARISON — XGBoost vs LightGBM vs Sklearn GBM
# =============================================================================
def framework_comparison():
    print("\n\n" + "=" * 70)
    print("FRAMEWORK COMPARISON — XGBoost vs LightGBM vs Sklearn GBM")
    print("=" * 70)

    # Use California Housing (larger dataset) for meaningful timing
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    # --- XGBoost ---
    t0 = time.time()
    m_xgb = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.1, max_depth=6,
        random_state=42,
    )
    m_xgb.fit(X_train, y_train)
    xgb_time = time.time() - t0
    y_pred = m_xgb.predict(X_test)
    results["XGBoost"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred),
        "Time (s)": xgb_time,
    }

    # --- LightGBM ---
    t0 = time.time()
    m_lgb = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.1, num_leaves=31,
        random_state=42, verbose=-1,
    )
    m_lgb.fit(X_train, y_train)
    lgb_time = time.time() - t0
    y_pred = m_lgb.predict(X_test)
    results["LightGBM"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred),
        "Time (s)": lgb_time,
    }

    # --- Sklearn GBM ---
    t0 = time.time()
    m_sk = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.1, max_depth=6,
        random_state=42,
    )
    m_sk.fit(X_train, y_train)
    sk_time = time.time() - t0
    y_pred = m_sk.predict(X_test)
    results["Sklearn GBM"] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred),
        "Time (s)": sk_time,
    }

    # Print
    df = pd.DataFrame(results).T
    print(f"\n{df.to_string()}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    frameworks = list(results.keys())
    colors = ["steelblue", "coral", "mediumseagreen"]

    # Time
    times = [results[f]["Time (s)"] for f in frameworks]
    axes[0].bar(frameworks, times, color=colors)
    axes[0].set_ylabel("Training Time (seconds)")
    axes[0].set_title("Training Speed")
    for i, t in enumerate(times):
        axes[0].text(i, t + 0.01, f"{t:.2f}s", ha="center", fontsize=10)

    # RMSE
    rmses = [results[f]["RMSE"] for f in frameworks]
    axes[1].bar(frameworks, rmses, color=colors)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE (lower is better)")
    for i, r in enumerate(rmses):
        axes[1].text(i, r + 0.005, f"{r:.4f}", ha="center", fontsize=10)

    # R²
    r2s = [results[f]["R²"] for f in frameworks]
    axes[2].bar(frameworks, r2s, color=colors)
    axes[2].set_ylabel("R²")
    axes[2].set_title("R² Score (higher is better)")
    for i, r in enumerate(r2s):
        axes[2].text(i, r + 0.002, f"{r:.4f}", ha="center", fontsize=10)

    plt.suptitle("Framework Comparison — California Housing (500 trees)", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "13_framework_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [Saved] plots/13_framework_comparison.png")

    # Classification comparison too
    print("\n  --- Classification comparison (Breast Cancer) ---")
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf_results = {}

    for name, model in [
        ("XGBoost", xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=42, use_label_encoder=False, eval_metric="logloss")),
        ("LightGBM", lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=31,
            random_state=42, verbose=-1)),
        ("Sklearn GBM", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=42)),
    ]:
        t0 = time.time()
        model.fit(X_train, y_train)
        t1 = time.time() - t0
        acc = model.score(X_test, y_test)
        cv = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
        clf_results[name] = {"Accuracy": acc, "CV Mean": cv, "Time (s)": t1}
        print(f"    {name:12s}: Acc={acc:.4f}, 5-CV={cv:.4f}, Time={t1:.3f}s")

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(clf_results))
    accs = [clf_results[f]["Accuracy"] for f in clf_results]
    cvs = [clf_results[f]["CV Mean"] for f in clf_results]
    width = 0.35

    bars1 = ax.bar(x_pos - width / 2, accs, width, color="steelblue", label="Test Acc")
    bars2 = ax.bar(x_pos + width / 2, cvs, width, color="coral", label="5-CV Mean")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(clf_results.keys()))
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification — XGBoost vs LightGBM vs Sklearn GBM")
    ax.legend()
    ax.set_ylim(0.9, 1.0)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "14_clf_framework_comparison.png", dpi=150)
    plt.close()
    print("  [Saved] plots/14_clf_framework_comparison.png")

    return results, clf_results


# =============================================================================
# SECTION 5: MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("12 — Boosted Trees (XGBoost & LightGBM)")
    print("=" * 70)

    clf_data = classification_experiments()
    reg_data = regression_experiments()
    hyperparameter_experiments(clf_data, reg_data)
    fw_reg, fw_clf = framework_comparison()

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print("""
    1.  XGBOOST CLASSIFIER          — Early stopping, regularisation, boosting
    2.  LIGHTGBM CLASSIFIER         — Leaf-wise growth, GOSS, histogram splits
    3.  EARLY STOPPING              — Train vs val loss, best iteration
    4.  XGBOOST REGRESSOR           — California Housing continuous prediction
    5.  LIGHTGBM REGRESSOR          — Same task, framework comparison
    6.  FEATURE IMPORTANCE           — gain / weight / cover importance types
    7.  MAX_DEPTH / NUM_LEAVES       — Tree complexity effect on performance
    8.  LEARNING RATE TRADE-OFF      — Lower LR → more trees → same accuracy
    9.  FRAMEWORK COMPARISON         — Speed & accuracy: XGBoost vs LightGBM vs Sklearn
    10. REGULARISATION SWEEP         — L1 / L2 effect on train-test gap
    """)
    print("All plots saved to:", PLOTS_DIR.resolve())
    print("=" * 70)
    print("\n🎉  PROJECT COMPLETE — All 12 classical ML algorithms implemented!")


if __name__ == "__main__":
    main()
