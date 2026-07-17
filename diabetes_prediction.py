#!/usr/bin/env python3
"""
🩺 Diabetes Prediction — Full ML Pipeline
============================================
A comprehensive, standalone pipeline for predicting diabetes
(Pima Indians Diabetes Dataset).

Features:
  • Exploratory Data Analysis (6 visualizations)
  • SMOTE for class imbalance
  • 9 individual classifiers + 6 ensemble methods
  • Stratified cross-validation
  • F1‑score as primary metric (+ accuracy, precision, recall)
  • Confusion matrix
  • PCA visualization
  • Feature importance analysis
  • Logging & argparse CLI

Author : Ali Kazemi
Source : https://github.com/Arefmtl/Diabetes-Prediction
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("diabetes")


# ── Helpers ──────────────────────────────────────────────────────────────────
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  1  LOAD & EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset."""
    df = pd.read_csv(path)
    log.info("Dataset shape: %s", df.shape)
    log.info("Columns: %s", list(df.columns))
    log.info("Outcome distribution:\n%s", df["Outcome"].value_counts().to_string())
    return df


def eda(df: pd.DataFrame, out_dir: str) -> None:
    """Generate 6 EDA plots."""
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    # 1 — Outcome count
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Outcome"].value_counts().plot(kind="bar", color=["#4382DF", "#112E81"], ax=ax)
    ax.set_title("Outcome Distribution", fontweight="bold")
    ax.set_xticklabels(["No Diabetes (0)", "Diabetes (1)"])
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "01_outcome_distribution.png"), dpi=150)
    plt.close(fig)

    # 2 — Feature histograms
    features = [c for c in df.columns if c != "Outcome"]
    n = len(features)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(14, rows * 3.5))
    axes = axes.flatten()
    for i, col in enumerate(features):
        ax = axes[i]
        df[col].hist(bins=30, color="#4382DF", edgecolor="white", ax=ax)
        ax.set_title(col, fontsize=10, fontweight="bold")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "02_feature_distributions.png"), dpi=150)
    plt.close(fig)

    # 3 — Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="Blues", square=True, ax=ax)
    ax.set_title("Correlation Matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "03_correlation_heatmap.png"), dpi=150)
    plt.close(fig)

    # 4 — Box plots (Outcome vs features)
    top_cols = sorted(
        df.corr()["Outcome"].abs().drop("Outcome", errors="ignore").nlargest(6).index
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(top_cols[:6]):
        df.boxplot(column=col, by="Outcome", ax=axes[i])
        axes[i].set_title(col, fontweight="bold")
    fig.suptitle("Feature vs Outcome (Top 6 by Correlation)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "04_boxplots_vs_outcome.png"), dpi=150)
    plt.close(fig)

    # 5 — Pair-plot (top 4 features)
    top4 = top_cols[:4]
    if top4:
        fig = sns.pairplot(df, vars=top4, hue="Outcome", palette={0: "#AACCD6", 1: "#112E81"}, diag_kind="kde")
        fig.fig.suptitle("Pairplot (Top 4 Features)", y=1.02, fontweight="bold")
        fig.savefig(os.path.join(fig_dir, "05_pairplot.png"), dpi=150)
        plt.close(fig.fig)

    # 6 — Null / zero check
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    fig, ax = plt.subplots(figsize=(7, 5))
    (df[zero_cols] == 0).sum().plot(kind="bar", color="#112E81", ax=ax)
    ax.set_title("Zero Values per Feature (likely missing)", fontweight="bold")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "06_zero_values.png"), dpi=150)
    plt.close(fig)

    log.info("EDA complete — 6 figures saved to %s", fig_dir)


# ══════════════════════════════════════════════════════════════════════════════
#  2  PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame) -> tuple:
    """Handle missing-like zeros, split, scale, SMOTE."""
    # Replace plausible zeros with median
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        median_val = df.loc[df[col] > 0, col].median()
        df[col] = df[col].replace(0, median_val)

    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    log.info("After SMOTE — train shape: %s, class balance: %s", X_train_res.shape, np.bincount(y_train_res))

    return X_train_res, X_test, y_train_res, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  3  MODELS
# ══════════════════════════════════════════════════════════════════════════════
def get_models() -> dict:
    """Return dict of 9 individual + 6 ensemble classifiers."""
    return {
        # ── Individual ──
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K‑Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", gamma="scale", probability=True, random_state=42),
        "Naïve Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=42),
        "MLP (NN)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        # ── Ensembles ──
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Bagging (DT)": BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=42
        ),
        "Soft Voting": VotingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000)),
                ("rf", RandomForestClassifier(n_estimators=100)),
                ("gb", GradientBoostingClassifier(n_estimators=100)),
            ],
            voting="soft",
        ),
        "Stack (LR meta)": StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=100)),
                ("svm", SVC(probability=True, random_state=42)),
                ("gb", GradientBoostingClassifier(n_estimators=100)),
            ],
            final_estimator=LogisticRegression(),
        ),
        "Stack (RF meta)": StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000)),
                ("svm", SVC(probability=True, random_state=42)),
                ("gb", GradientBoostingClassifier(n_estimators=100)),
            ],
            final_estimator=RandomForestClassifier(n_estimators=100),
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  4  TRAIN & EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_models(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: str,
) -> pd.DataFrame:
    """
    Train each model with stratified 5‑fold CV, evaluate on hold‑out,
    log metrics and produce a confusion-matrix grid.
    """
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))
    results = []
    best_f1 = 0.0
    best_name = ""
    best_model = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        log.info("─" * 50)
        log.info("Training: %s", name)

        # Cross‑validation
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        log.info("  CV F1:  %.4f (±%.4f)", cv_f1.mean(), cv_f1.std())
        log.info("  CV Acc: %.4f (±%.4f)", cv_acc.mean(), cv_acc.std())

        # Train on full train set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Hold‑out metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        log.info("  Hold‑out  →  Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f", acc, prec, rec, f1)
        results.append(
            {
                "Model": name,
                "CV F1 (mean)": round(cv_f1.mean(), 4),
                "CV F1 (std)": round(cv_f1.std(), 4),
                "CV Acc": round(cv_acc.mean(), 4),
                "Test Acc": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1‑Score": round(f1, 4),
            }
        )

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    # ── Results table ──
    df_res = pd.DataFrame(results).sort_values("F1‑Score", ascending=False)
    df_res.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    log.info("\n" + "=" * 65)
    log.info("RESULTS (sorted by F1‑Score)")
    log.info("\n%s", df_res.to_string(index=False))

    # ── Confusion-matrix grid ──
    n_models = len(models)
    cols = 4
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = axes.flatten()
    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["No Diabetes", "Diabetes"])
        disp.plot(ax=axes[i], cmap="Blues", colorbar=False, values_format="d")
        axes[i].set_title(name, fontsize=9, fontweight="bold")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Confusion Matrices — All Classifiers", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "07_confusion_matrices.png"), dpi=150)
    plt.close(fig)

    log.info("Best model: %s  (F1 = %.4f)", best_name, best_f1)
    return df_res, best_model, best_name


# ══════════════════════════════════════════════════════════════════════════════
#  5  PCA VIZ
# ══════════════════════════════════════════════════════════════════════════════
def plot_pca(X_train: np.ndarray, y_train: np.ndarray, out_dir: str) -> None:
    """2‑D PCA scatter coloured by class."""
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_train,
        cmap="coolwarm",
        alpha=0.6,
        edgecolors="k",
        linewidth=0.3,
    )
    ax.set_title("PCA — 2‑Component Projection", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    legend = ax.legend(*scatter.legend_elements(), title="Outcome")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "08_pca_projection.png"), dpi=150)
    plt.close(fig)
    log.info("PCA plot saved — explained variance: %.1f%%", pca.explained_variance_ratio_.sum() * 100)


# ══════════════════════════════════════════════════════════════════════════════
#  6  FEATURE IMPORTANCE (Tree‑based)
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(
    model, feature_names: list, out_dir: str, top_n: int = 8
) -> None:
    """Feature‑importance bar-plot for tree‑based models."""
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))
    if not hasattr(model, "feature_importances_"):
        log.info("Model %s has no feature_importances_ — skipping", type(model).__name__)
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        importances[indices[::-1]],
        color="#4382DF",
        edgecolor="#112E81",
    )
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "09_feature_importance.png"), dpi=150)
    plt.close(fig)
    log.info("Feature importance plot saved")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🩺 Diabetes Prediction — Full ML Pipeline"
    )
    parser.add_argument(
        "--data",
        default="Dataset/diabetes.csv",
        help="Path to diabetes.csv (default: Dataset/diabetes.csv)",
    )
    parser.add_argument(
        "--out",
        default="output",
        help="Output directory for figures & results (default: output)",
    )
    parser.add_argument(
        "--no-eda",
        action="store_true",
        help="Skip exploratory data analysis plots",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Skip PCA visualization",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out)
    log.info("=" * 60)
    log.info("🩺  DIABETES PREDICTION PIPELINE")
    log.info("=" * 60)

    # 1 — Load
    df = load_data(args.data)

    # 2 — EDA
    if not args.no_eda:
        eda(df, out_dir)

    # 3 — Preprocess
    feature_names = [c for c in df.columns if c != "Outcome"]
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    # 4 — Train & evaluate
    models = get_models()
    results, best_model, best_name = evaluate_models(
        models, X_train, y_train, X_test, y_test, out_dir
    )

    # 5 — PCA
    if not args.no_pca:
        plot_pca(X_train, y_train, out_dir)

    # 6 — Feature importance (from best tree‑based model)
    plot_feature_importance(best_model, feature_names, out_dir)

    log.info("=" * 60)
    log.info("✅ Pipeline complete → output saved to: %s", os.path.abspath(out_dir))
    log.info("🏆 Best model: %s  (F1 = %.4f)", best_name, results.iloc[0]["F1‑Score"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
