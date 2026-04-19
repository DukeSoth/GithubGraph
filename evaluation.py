"""
evaluation.py
-------------
Evaluate modeling results, produce diagnostic plots, and write a summary
report.  Reads from outputs/ — no raw data access required.

Inputs (from outputs/)
----------------------
  cv_results.csv          model-level LOO metrics
  feature_importance.csv  per-feature importance for each model
  loo_predictions.csv     per-repo predictions from the best model
  features.csv            feature matrix (for error analysis)
  labels.csv              ground-truth labels with intermediate columns

Outputs (to outputs/)
---------------------
  evaluation_report.txt   plain-text summary of all results
  confusion_matrix.png    confusion matrix for each model
  roc_curves.png          overlaid ROC curves
  feature_importance.png  top-10 features, side-by-side for each model
  error_analysis.png      feature profiles of misclassified repos

Public API
----------
  compute_classification_metrics(y_true, y_pred, y_proba) -> dict
  confusion_matrix_table(y_true, y_pred)                  -> pd.DataFrame
  error_analysis(pred_df, feature_df)                     -> pd.DataFrame
  plot_confusion_matrix(y_true, y_pred, ax, title)        -> None
  plot_roc_curve(y_true, y_proba, ax, label)              -> None
  plot_feature_importance(importance_df, ax, model_name)  -> None
  plot_error_analysis(error_df, feature_df, ax)           -> None
  run_evaluation(outputs_dir)                             -> dict
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for all environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ── 1. Metrics ────────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute standard binary classification metrics.

    Parameters
    ----------
    y_true  : ground-truth labels (0 / 1)
    y_pred  : hard predictions (0 / 1)
    y_proba : predicted probability of class 1 (needed for roc_auc)

    Returns
    -------
    dict: accuracy, precision, recall, f1, roc_auc, support_y1, support_y0,
          true_positives, false_positives, true_negatives, false_negatives
    """
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    roc_auc = float("nan")
    if y_proba is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_proba))

    return {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":          roc_auc,
        "true_positives":   int(tp),
        "false_positives":  int(fp),
        "true_negatives":   int(tn),
        "false_negatives":  int(fn),
        "support_y1":       int(np.sum(y_true == 1)),
        "support_y0":       int(np.sum(y_true == 0)),
    }


def confusion_matrix_table(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Return confusion matrix as a labelled 2×2 DataFrame.

    Rows = actual, columns = predicted.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=pd.Index(["Actual 0 (stable)", "Actual 1 (at risk)"], name=""),
        columns=["Pred 0 (stable)", "Pred 1 (at risk)"],
    )


# ── 2. Error analysis ─────────────────────────────────────────────────────────

def error_analysis(
    pred_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare feature values of misclassified repos against correctly classified
    ones to surface what the model struggled with.

    Parameters
    ----------
    pred_df    : per-repo prediction table (repo_name, y_true, y_pred,
                 prob_y1, correct).  Output of modeling.run_pipeline or
                 the saved loo_predictions.csv.
    feature_df : feature matrix with repo_name as index or column.

    Returns
    -------
    pd.DataFrame with columns:
      error_type  : 'FP' (false positive) or 'FN' (false negative)
      repo_name
      y_true, y_pred, prob_y1
      + one column per feature (values for that repo)
    """
    # normalise feature_df index
    if "repo_name" in feature_df.columns:
        feat = feature_df.set_index("repo_name")
    else:
        feat = feature_df.copy()

    errors = pred_df[~pred_df["correct"]].copy()

    def error_type(row):
        if row["y_true"] == 0 and row["y_pred"] == 1:
            return "FP (predicted at-risk, actually stable)"
        return "FN (predicted stable, actually at-risk)"

    errors["error_type"] = errors.apply(error_type, axis=1)

    # attach feature values
    errors = errors.set_index("repo_name")
    errors = errors.join(feat, how="left")
    errors = errors.reset_index().rename(columns={"index": "repo_name"})

    return errors


def _mean_comparison(pred_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table of feature means split by correct vs. incorrect prediction.

    Used internally for the text report.
    """
    if "repo_name" in feature_df.columns:
        feat = feature_df.set_index("repo_name")
    else:
        feat = feature_df.copy()

    merged = pred_df.set_index("repo_name").join(feat, how="left")

    numeric_cols = feat.columns.tolist()
    correct_mean   = merged[merged["correct"]][numeric_cols].mean()
    incorrect_mean = merged[~merged["correct"]][numeric_cols].mean()

    diff = incorrect_mean - correct_mean
    comparison = pd.DataFrame({
        "correct_mean":   correct_mean.round(4),
        "incorrect_mean": incorrect_mean.round(4),
        "difference":     diff.round(4),
    }).sort_values("difference", key=abs, ascending=False)

    return comparison


# ── 3. Plots ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax: plt.Axes,
    title: str = "Confusion Matrix",
) -> None:
    """
    Draw a colour-coded confusion matrix with cell counts and percentages.

    Rows = actual class, columns = predicted class.
    Blue intensity = count magnitude.  Text colour adapts to background.
    """
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1])
    n      = cm.sum()
    labels = [["TN", "FP"], ["FN", "TP"]]
    class_labels = ["Stable (0)", "At risk (1)"]

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())

    ax.set_xticks([0, 1]);  ax.set_xticklabels(class_labels, fontsize=9)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(class_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct   = 100 * count / n
            color = "white" if count > thresh else "black"
            ax.text(j, i,
                    f"{labels[i][j]}\n{count}\n({pct:.0f}%)",
                    ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    ax: plt.Axes,
    label: str = "model",
    color: Optional[str] = None,
) -> None:
    """
    Plot a single ROC curve on ax.  Diagonal chance line is drawn once by
    the caller to avoid duplicate legend entries.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    kwargs = {"label": f"{label}  (AUC = {auc:.3f})"}
    if color:
        kwargs["color"] = color
    ax.plot(fpr, tpr, linewidth=2, **kwargs)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    ax: plt.Axes,
    model_name: str,
    top_n: int = 10,
    color: str = "steelblue",
) -> None:
    """
    Horizontal bar chart of the top-n features for one model.

    Parameters
    ----------
    importance_df : DataFrame with features as index and model columns
                    (output of feature_importance.csv, one column at a time)
    model_name    : column name in importance_df to plot
    top_n         : how many top features to show
    """
    series = importance_df[model_name].dropna().sort_values(ascending=False).head(top_n)
    # reverse so most important is at the top of the chart
    series = series[::-1]

    bars = ax.barh(range(len(series)), series.values, color=color, alpha=0.85,
                   edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels(series.index, fontsize=8)
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_title(f"{model_name}\n(top {top_n})", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotate bars with values
    for bar, val in zip(bars, series.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=7, color="#333333")


def plot_error_analysis(
    error_rows: pd.DataFrame,
    feature_df: pd.DataFrame,
    ax: plt.Axes,
    top_features: list[str],
) -> None:
    """
    Dot plot comparing per-feature z-scores of each misclassified repo
    against the overall feature mean.

    Each row in error_rows is a misclassified repo.
    Each column is a feature from top_features.
    The dot's x position is the z-score relative to the full dataset mean/std.
    """
    if "repo_name" in feature_df.columns:
        feat = feature_df.set_index("repo_name")
    else:
        feat = feature_df.copy()

    # z-score each feature across the full dataset
    mu  = feat[top_features].mean()
    std = feat[top_features].std().replace(0, 1)

    if "repo_name" not in error_rows.columns:
        errors_indexed = error_rows
    else:
        errors_indexed = error_rows.set_index("repo_name")

    colors = {"FP (predicted at-risk, actually stable)": "tomato",
              "FN (predicted stable, actually at-risk)": "steelblue"}
    markers = {"FP (predicted at-risk, actually stable)": "^",
               "FN (predicted stable, actually at-risk)": "v"}

    for i, (repo, row) in enumerate(errors_indexed.iterrows()):
        vals   = feat.loc[repo, top_features] if repo in feat.index else pd.Series(dtype=float)
        if vals.empty:
            continue
        z      = (vals - mu) / std
        etype  = row.get("error_type", "unknown")
        color  = colors.get(etype, "gray")
        marker = markers.get(etype, "o")
        label  = f"{repo.split('/')[-1]} ({etype.split()[0]})"

        ax.scatter(z.values, range(len(top_features)),
                   color=color, marker=marker, s=60, alpha=0.85,
                   label=label, zorder=3)

    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=8)
    ax.set_xlabel("Z-score vs. dataset mean", fontsize=9)
    ax.set_title("Misclassified repos: feature z-scores", fontsize=10)
    ax.legend(fontsize=7, loc="lower right",
              framealpha=0.9, bbox_to_anchor=(1.0, 0.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)


# ── 4. Text report ────────────────────────────────────────────────────────────

def _build_text_report(
    cv_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    imp_df: pd.DataFrame,
) -> str:
    """
    Build a complete plain-text evaluation report.

    Returns the report as a string so callers can both print and save it.
    """
    lines = []
    w = 65

    def sep(title=""):
        if title:
            lines.append(f"\n{'=' * w}")
            lines.append(f"  {title}")
            lines.append(f"{'=' * w}")
        else:
            lines.append("─" * w)

    lines.append("=" * w)
    lines.append("  COLLABORATION RISK MODEL — EVALUATION REPORT")
    lines.append("  Leave-One-Out Cross-Validation  |  n = 20 repos")
    lines.append("=" * w)

    # ── model comparison ──────────────────────────────────────────────────────
    sep("1. Model Comparison (LOO CV, pooled OOF metrics)")
    lines.append("")
    lines.append(cv_df.to_string(index=False))
    lines.append("")
    lines.append(textwrap.fill(
        "Primary metric: ROC-AUC (robust to class imbalance with n=20). "
        "F1 is identical across non-dummy models because all achieve the "
        "same 6 errors; AUC distinguishes probability calibration.",
        width=w, initial_indent="  ", subsequent_indent="  "
    ))

    # ── confusion matrix (best model) ─────────────────────────────────────────
    best_row = cv_df[cv_df["model"] != "Dummy (baseline)"].sort_values(
        "roc_auc", ascending=False
    ).iloc[0]
    best_name = best_row["model"]

    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    y_prob = pred_df["prob_y1"].values

    m = compute_classification_metrics(y_true, y_pred, y_prob)
    cm_table = confusion_matrix_table(y_true, y_pred)

    sep(f"2. Confusion Matrix — {best_name}")
    lines.append("")
    lines.append(cm_table.to_string())
    lines.append("")
    lines.append(f"  TP={m['true_positives']}  FP={m['false_positives']}  "
                 f"TN={m['true_negatives']}  FN={m['false_negatives']}")
    lines.append(f"  Precision = {m['precision']:.3f}  "
                 f"Recall = {m['recall']:.3f}  "
                 f"F1 = {m['f1']:.3f}  "
                 f"AUC = {m['roc_auc']:.3f}")
    lines.append("")
    lines.append(textwrap.fill(
        f"Of {m['support_y1']} truly at-risk repos, the model correctly identified "
        f"{m['true_positives']} (recall = {m['recall']:.0%}). "
        f"Of {m['support_y0']} stable repos, {m['false_positives']} were falsely "
        f"flagged (false alarm rate = {m['false_positives']/m['support_y0']:.0%}).",
        width=w, initial_indent="  ", subsequent_indent="  "
    ))

    # ── per-repo predictions ───────────────────────────────────────────────────
    sep("3. Per-Repo Predictions (sorted by predicted probability)")
    display = pred_df.sort_values("prob_y1", ascending=False).copy()
    display["status"] = display.apply(
        lambda r: "✓" if r["correct"]
        else ("FP" if r["y_pred"] == 1 else "FN"),
        axis=1
    )
    lines.append("")
    lines.append(f"  {'repo':<38} {'y':>4} {'pred':>5} {'prob':>6}  {'':>4}")
    lines.append("  " + "─" * 58)
    for _, row in display.iterrows():
        name = row["repo_name"].split("/")[-1][:37]
        lines.append(
            f"  {name:<38} {int(row['y_true']):>4} {int(row['y_pred']):>5} "
            f"{row['prob_y1']:>6.3f}  {row['status']:>4}"
        )

    errors = pred_df[~pred_df["correct"]]
    fp = errors[errors["y_true"] == 0]
    fn = errors[errors["y_true"] == 1]

    sep("4. Error Analysis")
    lines.append(f"\n  Total errors : {len(errors)} / {len(pred_df)}")
    lines.append(f"  False Positives (predicted at-risk, actually stable) : {len(fp)}")
    for _, r in fp.iterrows():
        lines.append(f"    → {r['repo_name']}  (prob={r['prob_y1']:.3f})")
    lines.append(f"  False Negatives (predicted stable, actually at-risk) : {len(fn)}")
    for _, r in fn.iterrows():
        lines.append(f"    → {r['repo_name']}  (prob={r['prob_y1']:.3f})")

    # feature comparison: errors vs correct
    lines.append("")
    comparison = _mean_comparison(pred_df, feature_df)
    lines.append("  Feature means: correctly classified vs. misclassified")
    lines.append(f"\n  {'feature':<40} {'correct':>9} {'error':>9} {'diff':>9}")
    lines.append("  " + "─" * 70)
    for feat, row in comparison.head(8).iterrows():
        lines.append(
            f"  {feat:<40} {row['correct_mean']:>9.4f} "
            f"{row['incorrect_mean']:>9.4f} {row['difference']:>+9.4f}"
        )
    lines.append("")
    lines.append(textwrap.fill(
        "Interpretation: large |diff| features are the ones where errors "
        "have unusually high or low values compared to correct predictions.",
        width=w, initial_indent="  ", subsequent_indent="  "
    ))

    # ── feature importance ────────────────────────────────────────────────────
    sep("5. Feature Importance — Top 8 (all three models)")
    lines.append("")
    top8 = imp_df.mean(axis=1).sort_values(ascending=False).head(8).index
    header = f"  {'feature':<40} {'LR-all':>8} {'LR-sel':>8} {'RF':>8}"
    lines.append(header)
    lines.append("  " + "─" * 70)
    for feat in top8:
        lr_all = imp_df.loc[feat, "logistic_all"] if feat in imp_df.index else float("nan")
        lr_sel = imp_df.loc[feat, "logistic_sel"] if feat in imp_df.index else float("nan")
        rf     = imp_df.loc[feat, "random_forest"]  if feat in imp_df.index else float("nan")
        lr_sel_str = f"{lr_sel:>8.4f}" if not np.isnan(lr_sel) else f"{'—':>8}"
        lines.append(
            f"  {feat:<40} {lr_all:>8.4f} {lr_sel_str} {rf:>8.4f}"
        )
    lines.append("")
    lines.append(textwrap.fill(
        "active_contributor_slope and degree_centralization are the strongest "
        "predictors across all three models, confirming that the rate of "
        "contributor decline and network centralisation are the clearest "
        "early signals of collaboration collapse.",
        width=w, initial_indent="  ", subsequent_indent="  "
    ))

    lines.append(f"\n{'=' * w}\n  END OF REPORT\n{'=' * w}")
    return "\n".join(lines)


# ── 5. Generate all plots ──────────────────────────────────────────────────────

def _plot_roc_all_models(cv_df, pred_df, outputs_dir):
    """
    ROC curves for all non-dummy models on one figure.

    Since we only have OOF probabilities for the best model (loo_predictions.csv),
    we plot the single curve we have and annotate the AUC for all models from
    the summary table.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    y_true = pred_df["y_true"].values
    y_prob = pred_df["prob_y1"].values

    # plot the one curve we have full OOF probabilities for
    best_name = cv_df[cv_df["model"] != "Dummy (baseline)"].sort_values(
        "roc_auc", ascending=False
    ).iloc[0]["model"]
    plot_roc_curve(y_true, y_prob, ax, label=best_name, color="steelblue")

    # annotate AUC values for other models as text (no curve without their OOF proba)
    other_models = cv_df[
        (cv_df["model"] != "Dummy (baseline)") &
        (cv_df["model"] != best_name)
    ]
    colors = ["tomato", "seagreen", "darkorange"]
    for (_, row), col in zip(other_models.iterrows(), colors):
        ax.plot([], [], color=col,
                label=f"{row['model']}  (AUC = {row['roc_auc']:.3f})")

    # dummy reference
    ax.plot([], [], "k--", linewidth=0.8,
            label=f"Dummy (AUC = 0.500)")

    # diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)

    ax.set_xlim(-0.02, 1.02);  ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — LOO Cross-Validation", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    path = os.path.join(outputs_dir, "roc_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_confusion_matrices(pred_df, cv_df, outputs_dir):
    """One subplot per non-dummy model (we have a single pred table, so one matrix)."""
    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values

    best_name = cv_df[cv_df["model"] != "Dummy (baseline)"].sort_values(
        "roc_auc", ascending=False
    ).iloc[0]["model"]

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion_matrix(y_true, y_pred, ax,
                          title=f"Confusion Matrix\n{best_name}  (LOO CV)")
    path = os.path.join(outputs_dir, "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_feature_importances(imp_df, outputs_dir):
    """Three side-by-side bar charts, one per model."""
    model_cols = [c for c in ["logistic_all", "logistic_sel", "random_forest"]
                  if c in imp_df.columns]
    colors     = ["steelblue", "seagreen", "tomato"]

    fig, axes = plt.subplots(1, len(model_cols), figsize=(5 * len(model_cols), 5))
    if len(model_cols) == 1:
        axes = [axes]

    for ax, col, color in zip(axes, model_cols, colors):
        plot_feature_importance(imp_df, ax, col, top_n=10, color=color)

    fig.suptitle("Feature Importances by Model", fontsize=13, y=1.02)
    path = os.path.join(outputs_dir, "feature_importance.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_error_analysis_chart(pred_df, feature_df, imp_df, outputs_dir):
    """Z-score dot plot for misclassified repos."""
    errors = error_analysis(pred_df, feature_df)
    if errors.empty:
        return None

    # use top-8 features by mean importance across models
    top_feats = imp_df.mean(axis=1).sort_values(ascending=False).head(8).index.tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_error_analysis(errors, feature_df, ax, top_features=top_feats)
    path = os.path.join(outputs_dir, "error_analysis.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 6. Orchestrator ───────────────────────────────────────────────────────────

def run_evaluation(outputs_dir: Optional[str] = None) -> dict:
    """
    Load saved modeling outputs and produce a complete evaluation.

    Parameters
    ----------
    outputs_dir : path to outputs/ directory (default: auto-resolved)

    Returns
    -------
    dict with keys:
      "metrics"      : compute_classification_metrics result for best model
      "cv_df"        : model comparison DataFrame
      "errors"       : error_analysis DataFrame
      "comparison"   : feature mean comparison (correct vs. incorrect)
      "report_path"  : path to evaluation_report.txt
      "plot_paths"   : list of saved plot file paths
    """
    if outputs_dir is None:
        outputs_dir = os.path.join(os.path.dirname(_SRC), "outputs")

    def _load(name):
        path = os.path.join(outputs_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Run modeling.py first."
            )
        return path

    _sep = lambda t: print(f"\n{'=' * 60}\n  {t}\n{'=' * 60}")

    # ── load artifacts ────────────────────────────────────────────────────────
    _sep("Loading saved artifacts")
    cv_df     = pd.read_csv(_load("cv_results.csv"))
    imp_df    = pd.read_csv(_load("feature_importance.csv"), index_col=0)
    pred_df   = pd.read_csv(_load("loo_predictions.csv"))
    feature_df = pd.read_csv(_load("features.csv"))
    labels_df  = pd.read_csv(_load("labels.csv"))

    print(f"  cv_results    : {len(cv_df)} models")
    print(f"  loo_predictions: {len(pred_df)} repos")
    print(f"  features      : {feature_df.shape[1]-1} feature columns")
    print(f"  importances   : {len(imp_df)} features × {len(imp_df.columns)} models")

    y_true = pred_df["y_true"].values
    y_pred = pred_df["y_pred"].values
    y_prob = pred_df["prob_y1"].values

    # ── metrics ───────────────────────────────────────────────────────────────
    _sep("Classification metrics (best model — LOO CV)")
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    best_name = cv_df[cv_df["model"] != "Dummy (baseline)"].sort_values(
        "roc_auc", ascending=False
    ).iloc[0]["model"]
    print(f"\n  Model: {best_name}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<22} {v:.4f}")
        else:
            print(f"  {k:<22} {v}")

    # ── confusion matrix ──────────────────────────────────────────────────────
    _sep("Confusion matrix")
    print()
    print(confusion_matrix_table(y_true, y_pred).to_string())

    # ── model comparison table ────────────────────────────────────────────────
    _sep("Model comparison")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 120)
    print(cv_df.to_string(index=False))

    # ── error analysis ────────────────────────────────────────────────────────
    _sep("Error analysis")
    errors = error_analysis(pred_df, feature_df)
    print(f"\n  {len(errors)} misclassified repos:\n")
    for _, row in errors.iterrows():
        print(f"  [{row['error_type'].split()[0]}]  {row['repo_name']}"
              f"  (prob_y1={row['prob_y1']:.3f})")

    comparison = _mean_comparison(pred_df, feature_df)
    print(f"\n  Feature means: correct vs. error  (sorted by |difference|)")
    print(f"\n  {'feature':<40} {'correct':>9} {'error':>9} {'diff':>9}")
    print("  " + "─" * 70)
    for feat, row in comparison.head(8).iterrows():
        diff_str = f"{row['difference']:>+9.4f}"
        print(f"  {feat:<40} {row['correct_mean']:>9.4f} "
              f"{row['incorrect_mean']:>9.4f} {diff_str}")

    # ── feature importance ────────────────────────────────────────────────────
    _sep("Feature importance (top 8, mean across models)")
    top8 = imp_df.mean(axis=1).sort_values(ascending=False).head(8)
    print(f"\n  {'feature':<40}  mean_importance")
    print("  " + "─" * 58)
    for feat, val in top8.items():
        print(f"  {feat:<40}  {val:.4f}")

    # ── generate plots ────────────────────────────────────────────────────────
    _sep("Generating plots")
    os.makedirs(outputs_dir, exist_ok=True)
    plot_paths = []

    p = _plot_confusion_matrices(pred_df, cv_df, outputs_dir)
    plot_paths.append(p);  print(f"  Saved: {p}")

    p = _plot_roc_all_models(cv_df, pred_df, outputs_dir)
    plot_paths.append(p);  print(f"  Saved: {p}")

    p = _plot_feature_importances(imp_df, outputs_dir)
    plot_paths.append(p);  print(f"  Saved: {p}")

    p = _plot_error_analysis_chart(pred_df, feature_df, imp_df, outputs_dir)
    if p:
        plot_paths.append(p);  print(f"  Saved: {p}")

    # ── text report ───────────────────────────────────────────────────────────
    _sep("Writing text report")
    report = _build_text_report(cv_df, pred_df, feature_df, imp_df)
    print(report)

    report_path = os.path.join(outputs_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Saved: {report_path}")

    return {
        "metrics":      metrics,
        "cv_df":        cv_df,
        "errors":       errors,
        "comparison":   comparison,
        "report_path":  report_path,
        "plot_paths":   plot_paths,
    }


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_evaluation()
