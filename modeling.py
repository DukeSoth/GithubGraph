"""
modeling.py
-----------
Train and evaluate classification models that predict collaboration risk (y=1)
from the repository-level feature matrix.

Dataset context
---------------
n = 20 labeled repositories, 14 numeric features, y ∈ {0, 1}.
Class distribution: y=1 (9 repos, 45%), y=0 (11 repos, 55%) — near-balanced.

With n=20:
  - Leave-One-Out CV (LOO) is used to maximise training data per fold.
  - AUC and F1 are computed over POOLED out-of-fold predictions, not as means
    of per-fold scores, because each fold has exactly one test sample.
  - Only simple, regularised models are used to avoid overfitting.

Models
------
  Dummy            : majority-class baseline
  Logistic (LR)    : L2-regularised, z-scored features, balanced class weights
  Random Forest    : shallow (max_depth=3), balanced class weights

Feature selection
-----------------
  SelectKBest (mutual information) reduces 14 features to k=8 before LR.
  RF uses all features and reports built-in importance.

Public API
----------
  load_features_and_labels()                  -> (feature_df, label_series)
  prepare_xy(feature_df, label_series)        -> (X, y, feature_names)
  select_features(X, y, feature_names, k)     -> (X_sel, selected_names)
  loo_cross_validate(model, X, y)             -> CVResult
  train_final_model(model, X, y)              -> fitted model
  get_feature_importance(model, feature_names)-> pd.Series
  run_pipeline()                              -> dict of results
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data_loading import load_datasets, load_candidate_repos
from feature_engineering import build_feature_matrix, build_all_repo_graphs
from labeling import compute_labels, get_label_series


# ── result container ───────────────────────────────────────────────────────────

@dataclass
class CVResult:
    """
    Holds cross-validation outcomes for one model.

    Metrics are computed over POOLED out-of-fold predictions so that F1,
    precision, recall, and AUC are well-defined even with LOO (one test
    sample per fold).

    Attributes
    ----------
    model_name      : display name
    n_samples       : number of repos evaluated
    oof_preds       : binary predictions for each repo (in original order)
    oof_proba       : predicted probability of y=1 for each repo
    accuracy        : fraction of correctly predicted repos
    f1              : F1 score (binary, positive class = 1)
    precision       : precision for y=1
    recall          : recall for y=1
    roc_auc         : ROC-AUC over pooled OOF probabilities
    per_fold_correct: bool array — was repo i predicted correctly?
    """
    model_name:       str
    n_samples:        int
    oof_preds:        np.ndarray
    oof_proba:        np.ndarray
    accuracy:         float
    f1:               float
    precision:        float
    recall:           float
    roc_auc:          float
    per_fold_correct: np.ndarray = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "model":     self.model_name,
            "n":         self.n_samples,
            "accuracy":  round(self.accuracy,  4),
            "f1":        round(self.f1,        4),
            "precision": round(self.precision, 4),
            "recall":    round(self.recall,    4),
            "roc_auc":   round(self.roc_auc,   4),
        }


# ── 1. Load from saved artifacts ──────────────────────────────────────────────

def load_features_and_labels(
    outputs_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load features.csv and labels.csv from outputs/ and return aligned objects.

    Returns
    -------
    feature_df   : DataFrame indexed by repo_name, numeric features only
    label_series : Series indexed by repo_name, values ∈ {0, 1}

    Raises
    ------
    FileNotFoundError if either file is missing.
    """
    if outputs_dir is None:
        outputs_dir = os.path.join(os.path.dirname(_SRC), "outputs")

    feat_path  = os.path.join(outputs_dir, "features.csv")
    label_path = os.path.join(outputs_dir, "labels.csv")

    for p in (feat_path, label_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found.\n"
                "Run feature_engineering.py and labeling.py first."
            )

    feature_df = pd.read_csv(feat_path).set_index("repo_name")
    labels_df  = pd.read_csv(label_path)
    label_series = labels_df.set_index("repo_name")["y"].rename("y")

    return feature_df, label_series


# ── 2. Prepare arrays ─────────────────────────────────────────────────────────

def prepare_xy(
    feature_df: pd.DataFrame,
    label_series: pd.Series,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Align feature matrix and labels by repo_name, return numpy arrays.

    feature_df must be indexed by repo_name (use load_features_and_labels or
    call df.set_index("repo_name") before passing here).

    Drops repos present in one but not the other with a printed warning.
    Fills residual NaN values with 0.

    Returns
    -------
    X            : float array  (n_repos, n_features)
    y            : int array    (n_repos,)
    feature_names: list of column names matching X's columns
    """
    common   = feature_df.index.intersection(label_series.index)
    n_feat   = len(feature_df)
    n_label  = len(label_series)
    dropped  = (n_feat - len(common)) + (n_label - len(common))
    if dropped:
        print(f"[prepare_xy] {dropped} repos dropped (index mismatch)")

    X_df  = feature_df.loc[common].fillna(0.0)
    y_arr = label_series.loc[common].values.astype(int)

    return X_df.values.astype(float), y_arr, list(X_df.columns)


# ── 3. Feature selection ──────────────────────────────────────────────────────

def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    k: int = 8,
) -> tuple[np.ndarray, list[str]]:
    """
    Select the top-k features by mutual information with y.

    Mutual information is used instead of ANOVA F-score because it captures
    non-linear relationships and makes no distributional assumptions — both
    appropriate for a small, mixed-scale feature set.

    Caution: selector is fit on the FULL dataset before LOO CV, which is a
    mild form of data leakage (feature ranks could shift if one sample is
    removed).  For n=20 this is acceptable; for larger datasets, embed
    selection inside the CV loop via a Pipeline.

    Parameters
    ----------
    X            : feature matrix  (n, p)
    y            : labels          (n,)
    feature_names: column names matching X
    k            : number of features to keep (capped at p)

    Returns
    -------
    X_sel          : (n, k) array
    selected_names : list of k feature names, sorted by importance descending
    """
    k = min(k, X.shape[1])
    selector  = SelectKBest(mutual_info_classif, k=k)
    X_sel     = selector.fit_transform(X, y)
    mask      = selector.get_support()
    scores    = selector.scores_

    selected  = [(name, scores[i]) for i, (name, keep) in
                 enumerate(zip(feature_names, mask)) if keep]
    selected.sort(key=lambda t: t[1], reverse=True)

    selected_names = [name for name, _ in selected]
    # reorder X_sel columns to match descending importance order
    name_to_col    = {name: i for i, name in enumerate(selected_names)}
    orig_indices   = [i for i, (name, keep) in
                      enumerate(zip(feature_names, mask)) if keep]
    # map original column order to importance order
    importance_order = sorted(range(len(orig_indices)),
                              key=lambda j: scores[orig_indices[j]],
                              reverse=True)
    X_sel = X_sel[:, importance_order]

    print(f"[select_features] top {k} features (mutual info):")
    for name, score in selected:
        print(f"  {name:<40}  MI = {score:.4f}")

    return X_sel, selected_names


# ── 4. LOO cross-validation ───────────────────────────────────────────────────

def loo_cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "model",
) -> CVResult:
    """
    Leave-One-Out cross-validation with pooled out-of-fold metric computation.

    In each of the n=20 folds, one repo is held out and the model is trained
    fresh on the remaining 19.  Predictions and probabilities for the held-out
    repo are collected into OOF arrays.

    Metrics are then computed ONCE on the full OOF arrays rather than averaged
    over folds.  This matters for:
      - roc_auc: impossible per-fold with 1 test sample; valid over 20 pooled
      - precision/recall: per-fold values are 0 or 1, which would give a noisy
        mean; pooled gives the true confusion-matrix-derived values

    Parameters
    ----------
    model      : unfitted sklearn-compatible estimator (will be cloned per fold)
    X          : feature matrix  (n, p)
    y          : labels          (n,)
    model_name : display string

    Returns
    -------
    CVResult
    """
    from sklearn.base import clone

    loo       = LeaveOneOut()
    oof_preds = np.zeros(len(y), dtype=int)
    oof_proba = np.zeros(len(y), dtype=float)

    supports_proba = hasattr(model, "predict_proba")

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        oof_preds[test_idx] = fold_model.predict(X_test)

        if supports_proba:
            proba = fold_model.predict_proba(X_test)
            # column index for class 1 — handle models where class ordering varies
            class_list = list(fold_model.classes_)
            pos_idx    = class_list.index(1) if 1 in class_list else -1
            oof_proba[test_idx] = proba[:, pos_idx] if pos_idx >= 0 else 0.5
        else:
            # for models without predict_proba, use hard prediction as proxy
            oof_proba[test_idx] = float(oof_preds[test_idx])

    # ── compute pooled metrics ────────────────────────────────────────────────
    accuracy  = accuracy_score(y, oof_preds)
    f1        = f1_score(y, oof_preds,        zero_division=0)
    precision = precision_score(y, oof_preds, zero_division=0)
    recall    = recall_score(y, oof_preds,    zero_division=0)

    # roc_auc requires both classes present in y (they are: 9/11 split)
    try:
        roc_auc = roc_auc_score(y, oof_proba)
    except ValueError:
        roc_auc = float("nan")

    per_fold_correct = (oof_preds == y)

    return CVResult(
        model_name       = model_name,
        n_samples        = len(y),
        oof_preds        = oof_preds,
        oof_proba        = oof_proba,
        accuracy         = accuracy,
        f1               = f1,
        precision        = precision,
        recall           = recall,
        roc_auc          = roc_auc,
        per_fold_correct = per_fold_correct,
    )


# ── 5. Final model training ───────────────────────────────────────────────────

def train_final_model(model, X: np.ndarray, y: np.ndarray):
    """
    Fit model on the full dataset after cross-validation is complete.

    This gives a model for inspection (coefficients, feature importances) and
    future use on new data.  It is NOT used to report generalization metrics —
    those come from LOO CV.

    Returns the fitted model.
    """
    from sklearn.base import clone
    fitted = clone(model)
    fitted.fit(X, y)
    return fitted


# ── 6. Feature importance ─────────────────────────────────────────────────────

def get_feature_importance(
    fitted_model,
    feature_names: list[str],
) -> pd.Series:
    """
    Extract feature importance from a fitted model as a named Series.

    Handles:
      - Pipeline(StandardScaler, LogisticRegression): uses |coefficient| values
      - RandomForestClassifier: uses mean decrease impurity
      - DummyClassifier: returns uniform zeros

    Parameters
    ----------
    fitted_model  : model trained on the full dataset
    feature_names : column names matching the features passed to fit()

    Returns
    -------
    pd.Series — index = feature name, values = importance, sorted descending
    """
    # unwrap Pipeline to get the final estimator
    estimator = fitted_model
    if isinstance(fitted_model, Pipeline):
        estimator = fitted_model.named_steps[fitted_model.steps[-1][0]]

    if hasattr(estimator, "coef_"):
        # logistic regression: use absolute value of z-scored coefficients
        importances = np.abs(estimator.coef_[0])
    elif hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    else:
        importances = np.zeros(len(feature_names))

    series = pd.Series(importances, index=feature_names, name="importance")
    return series.sort_values(ascending=False)


# ── 7. Model definitions ──────────────────────────────────────────────────────

def _make_logistic(C: float = 0.1) -> Pipeline:
    """
    L2-regularised logistic regression inside a scaling pipeline.

    C=0.1 (strong regularisation) is appropriate for n=20.  class_weight
    ='balanced' down-weights the majority class so the model doesn't ignore
    the minority class (y=1).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        )),
    ])


def _make_random_forest(n_estimators: int = 200, max_depth: int = 3) -> RandomForestClassifier:
    """
    Shallow random forest.

    max_depth=3 prevents individual trees from memorising the training set.
    n_estimators=200 gives stable feature importances despite the small n.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _make_dummy() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent", random_state=42)


# ── 8. Full pipeline ──────────────────────────────────────────────────────────

def run_pipeline(
    outputs_dir: Optional[str] = None,
    k_features:  int = 8,
) -> dict:
    """
    End-to-end modeling pipeline.

    Steps
    -----
    1. Load features.csv and labels.csv
    2. Prepare (X, y) arrays; print class distribution
    3. Select top-k features by mutual information
    4. LOO cross-validate three models on the full 14-feature matrix
    5. LOO cross-validate LR on the k-selected features (for comparison)
    6. Train final models on the full dataset for inspection
    7. Print results table and feature importances
    8. Save outputs/cv_results.csv and outputs/feature_importance.csv

    Parameters
    ----------
    outputs_dir : directory for outputs (default: project_root/outputs)
    k_features  : number of features to select for the LR+selection variant

    Returns
    -------
    dict with keys:
      "cv_results"          : pd.DataFrame — one row per model
      "feature_importances" : dict[str, pd.Series]
      "final_models"        : dict[str, fitted model]
      "selected_features"   : list[str]
      "X"                   : full feature matrix
      "y"                   : label array
      "feature_names"       : all feature names
    """
    if outputs_dir is None:
        outputs_dir = os.path.join(os.path.dirname(_SRC), "outputs")

    _sep = lambda title: print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

    # ── Step 1: load ──────────────────────────────────────────────────────────
    _sep("Step 1: Load features and labels")
    feature_df, label_series = load_features_and_labels(outputs_dir)
    print(f"  Features : {feature_df.shape[1]} columns, {len(feature_df)} repos")
    print(f"  Labels   : {len(label_series)} repos")

    # ── Step 2: prepare arrays ────────────────────────────────────────────────
    _sep("Step 2: Prepare X, y")
    X, y, feature_names = prepare_xy(feature_df, label_series)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    print(f"  n = {len(y)},  y=1 (at risk): {n_pos},  y=0 (stable): {n_neg}")
    print(f"  Features ({len(feature_names)}): {feature_names}")

    # ── Step 3: feature selection ─────────────────────────────────────────────
    _sep(f"Step 3: Select top-{k_features} features (mutual information)")
    X_sel, selected_names = select_features(X, y, feature_names, k=k_features)

    # ── Step 4: LOO CV on all models ──────────────────────────────────────────
    _sep("Step 4: LOO cross-validation")

    models_to_cv = [
        ("Dummy (baseline)",           _make_dummy(),                  X),
        ("Logistic (all features)",    _make_logistic(C=0.1),          X),
        ("Logistic (top-8 features)",  _make_logistic(C=0.1),          X_sel),
        ("Random Forest",              _make_random_forest(),           X),
    ]

    cv_results: list[CVResult] = []
    for name, model, X_input in models_to_cv:
        print(f"\n  → {name}")
        result = loo_cross_validate(model, X_input, y, model_name=name)
        cv_results.append(result)
        r = result.as_dict()
        print(f"    accuracy={r['accuracy']:.3f}  f1={r['f1']:.3f}  "
              f"precision={r['precision']:.3f}  recall={r['recall']:.3f}  "
              f"roc_auc={r['roc_auc']:.3f}")

    # ── Step 5: results table ─────────────────────────────────────────────────
    _sep("Step 5: Results summary")
    cv_df = pd.DataFrame([r.as_dict() for r in cv_results])
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 120)
    print(cv_df.to_string(index=False))

    # ── Step 6: train final models on full data ───────────────────────────────
    _sep("Step 6: Final models (trained on full dataset)")
    final_models = {
        "logistic_all":  train_final_model(_make_logistic(C=0.1),     X,     y),
        "logistic_sel":  train_final_model(_make_logistic(C=0.1),     X_sel, y),
        "random_forest": train_final_model(_make_random_forest(),      X,     y),
        "dummy":         train_final_model(_make_dummy(),              X,     y),
    }
    print("  Final models trained.")

    # ── Step 7: feature importances ───────────────────────────────────────────
    _sep("Step 7: Feature importances")
    importances = {
        "logistic_all":  get_feature_importance(final_models["logistic_all"],
                                                feature_names),
        "logistic_sel":  get_feature_importance(final_models["logistic_sel"],
                                                selected_names),
        "random_forest": get_feature_importance(final_models["random_forest"],
                                                feature_names),
    }

    for mname, imp in importances.items():
        print(f"\n  {mname}:")
        for feat, val in imp.head(8).items():
            print(f"    {feat:<40}  {val:.4f}")

    # ── Step 8: per-repo prediction detail ───────────────────────────────────
    _sep("Step 8: Per-repo LOO predictions")

    # use the best non-dummy model by roc_auc for the detail view
    non_dummy = [r for r in cv_results if "Dummy" not in r.model_name]
    best      = max(non_dummy, key=lambda r: r.roc_auc)
    print(f"\n  Showing predictions from: '{best.model_name}'\n")

    repo_names = feature_df.index.tolist()
    pred_df = pd.DataFrame({
        "repo_name":  repo_names,
        "y_true":     y,
        "y_pred":     best.oof_preds,
        "prob_y1":    best.oof_proba.round(3),
        "correct":    best.per_fold_correct,
    })
    errors = pred_df[~pred_df["correct"]]
    print(pred_df.to_string(index=False))
    print(f"\n  Misclassified ({len(errors)}):")
    if errors.empty:
        print("    None")
    else:
        print(errors[["repo_name", "y_true", "y_pred", "prob_y1"]].to_string(index=False))

    # ── Step 9: save ──────────────────────────────────────────────────────────
    _sep("Step 9: Save outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    cv_path = os.path.join(outputs_dir, "cv_results.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"  Saved: {cv_path}")

    # combine all importances into one wide table
    imp_df = pd.concat(
        {name: imp for name, imp in importances.items()},
        axis=1,
    ).fillna(0.0)
    imp_path = os.path.join(outputs_dir, "feature_importance.csv")
    imp_df.to_csv(imp_path)
    print(f"  Saved: {imp_path}")

    pred_path = os.path.join(outputs_dir, "loo_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved: {pred_path}")

    return {
        "cv_results":           cv_df,
        "cv_result_objects":    cv_results,
        "feature_importances":  importances,
        "final_models":         final_models,
        "selected_features":    selected_names,
        "X":                    X,
        "y":                    y,
        "feature_names":        feature_names,
    }


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_pipeline()
