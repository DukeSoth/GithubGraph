"""
risk_score.py
-------------
Compute interpretable, rule-based risk scores for each candidate repository
as a complement to the learned model's predicted probability.

Design philosophy
-----------------
The composite score makes explicit WHICH dimension of health is concerning and
by how much, whereas the model's prob_y1 is a black-box real number.  Both
outputs are meaningful; they should be read together.

Three risk components
---------------------
  Activity risk  (weight 0.35)
    Captures whether contribution activity was declining across the
    observation window.
    Signals: active_contributor_slope, contribution_volume_slope

  Contributor risk  (weight 0.25)
    Captures bus-factor fragility: few contributors, one dominant actor,
    unequal distribution of work.
    Signals: num_contributors, top1_share, gini_coefficient

  Structural risk  (weight 0.40)
    Captures collaboration network fragility: star topology, network
    collapse when the top contributor is removed.
    Signals: degree_centralization, density_change_after_remove_top1,
             lcc_ratio_after_remove_top1

Normalization
-------------
Unbounded features (slopes, contributor counts) are normalized to [0, 1]
using robust dataset-level percentile bounds (p5 / p95), so every score
expresses "how risky is this repo relative to the observed cohort?".
Features that are already in [0, 1] (gini, top1_share, etc.) are used
directly.

This means scores are only meaningful relative to the 20-repo candidate set.
If the candidate set changes, call score_all_repos() again — no manual
recalibration is needed.

Output columns (score_all_repos)
---------------------------------
  repo_name
  activity_risk         float [0, 1]
  contributor_risk      float [0, 1]
  structural_risk       float [0, 1]
  composite_risk        float [0, 1]
  risk_tier             str   Low / Medium / High / Critical
  activity_signal       str   human-readable dominant activity signal
  contributor_signal    str   human-readable dominant contributor signal
  structural_signal     str   human-readable dominant structural signal

Public API
----------
  score_all_repos(feature_df, weights)           -> pd.DataFrame
  rank_repos_by_risk(scored_df, score_col)       -> pd.DataFrame
  score_to_tier(score)                           -> str
  explain_repo(repo_name, scored_df, feature_df) -> None
  run_scoring(outputs_dir)                       -> pd.DataFrame
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ── constants ──────────────────────────────────────────────────────────────────

# Component weights must sum to 1.0.
# Reflect model-derived importances: structural features dominate (degree_centralization
# + density_change together account for ~36% of mean importance), activity comes next
# (active_contributor_slope = 32%), contributor features are informative but weaker.
DEFAULT_WEIGHTS: dict[str, float] = {
    "activity":    0.35,
    "contributor": 0.25,
    "structural":  0.40,
}

# Risk tier boundaries (composite_risk)
TIER_THRESHOLDS = {
    "Critical": 0.70,
    "High":     0.55,
    "Medium":   0.40,
    # below 0.40 → "Low"
}


# ── normalization helpers ──────────────────────────────────────────────────────

class _DatasetNorms:
    """
    Pre-computed dataset-level normalization parameters.

    Built once from the full feature_df so every per-row risk function uses
    consistent, cohort-relative bounds.

    For slopes and counts: robust [p5, p95] percentile bounds.
    For already-bounded [0,1] features: passthrough (lo=0, hi=1).
    """

    def __init__(self, feature_df: pd.DataFrame):
        df = feature_df.copy()
        if "repo_name" in df.columns:
            df = df.set_index("repo_name")

        def _bounds(col: str, lo_pct: float = 5, hi_pct: float = 95):
            lo = float(np.percentile(df[col].dropna(), lo_pct))
            hi = float(np.percentile(df[col].dropna(), hi_pct))
            if hi == lo:
                hi = lo + 1e-9   # avoid divide-by-zero for constant columns
            return lo, hi

        # activity
        self.slope_contrib_lo, self.slope_contrib_hi = _bounds("active_contributor_slope")
        self.slope_volume_lo,  self.slope_volume_hi  = _bounds("contribution_volume_slope")

        # contributor
        self.n_contrib_lo, self.n_contrib_hi = _bounds("num_contributors")

        # structural: density_change is always ≤ 0; lcc_ratio ∈ [0, 1]
        # Use p5 of density_change as the most-extreme risk end
        self.density_change_lo, _ = _bounds("density_change_after_remove_top1", 5, 95)
        # density_change_lo is the most-negative (most fragile) value

    def norm(self, value: float, lo: float, hi: float) -> float:
        """Clip-and-scale value to [0, 1] given [lo, hi] bounds."""
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def norm_inv(self, value: float, lo: float, hi: float) -> float:
        """Same as norm(), but inverted: high raw value → low risk score."""
        return 1.0 - self.norm(value, lo, hi)


# ── 1. Activity risk ───────────────────────────────────────────────────────────

def _activity_risk(row: pd.Series, norms: _DatasetNorms) -> tuple[float, str]:
    """
    Estimate activity-based risk from observation-window trend features.

    Signals
    -------
    active_contributor_slope  — declining contributor base (strongest signal)
    contribution_volume_slope — declining event volume

    Both slopes are inverted (lower slope = higher risk).  The two signals
    are weighted 2:1 in favour of the contributor slope, matching its higher
    model importance.

    Returns (risk_score, dominant_signal_description).
    """
    # lower slope → higher risk; normalize relative to dataset cohort
    slope_c = row.get("active_contributor_slope", 0.0)
    slope_v = row.get("contribution_volume_slope", 0.0)

    risk_c = norms.norm_inv(slope_c, norms.slope_contrib_lo, norms.slope_contrib_hi)
    risk_v = norms.norm_inv(slope_v, norms.slope_volume_lo,  norms.slope_volume_hi)

    # 2:1 weight towards contributor slope (higher model importance)
    score = (2 * risk_c + 1 * risk_v) / 3.0

    if slope_c < 0 and slope_v < 0:
        signal = "both contributor count and volume declining"
    elif slope_c < 0:
        signal = f"contributor base shrinking (slope={slope_c:.2f})"
    elif slope_v < 0:
        signal = f"contribution volume declining (slope={slope_v:.0f})"
    else:
        signal = "activity stable or growing"

    return float(score), signal


# ── 2. Contributor risk ────────────────────────────────────────────────────────

def _contributor_risk(row: pd.Series, norms: _DatasetNorms) -> tuple[float, str]:
    """
    Estimate contributor-fragility risk.

    Signals
    -------
    num_contributors  — few contributors = high bus-factor risk
    top1_share        — dominant single contributor (already ∈ [0,1])
    gini_coefficient  — unequal contribution distribution (already ∈ [0,1])

    Weighted equally; all three independently capture different aspects of
    concentration risk.

    Returns (risk_score, dominant_signal_description).
    """
    n_c   = row.get("num_contributors", 1.0)
    top1  = row.get("top1_share",       0.0)
    gini  = row.get("gini_coefficient", 0.0)

    # fewer contributors → higher risk; invert and normalize
    risk_n    = norms.norm_inv(n_c, norms.n_contrib_lo, norms.n_contrib_hi)
    risk_top1 = float(np.clip(top1, 0.0, 1.0))
    risk_gini = float(np.clip(gini, 0.0, 1.0))

    score = float(np.mean([risk_n, risk_top1, risk_gini]))

    # pick the dominant signal for the human-readable explanation
    signals = [
        (risk_n,    f"only {int(n_c)} contributor(s)"),
        (risk_top1, f"top contributor holds {top1:.0%} of all events"),
        (risk_gini, f"contribution Gini = {gini:.2f}"),
    ]
    dominant = max(signals, key=lambda t: t[0])[1]

    return score, dominant


# ── 3. Structural risk ─────────────────────────────────────────────────────────

def _structural_risk(row: pd.Series, norms: _DatasetNorms) -> tuple[float, str]:
    """
    Estimate collaboration-network fragility risk.

    Signals
    -------
    degree_centralization            — star topology; hub = single point of failure
    density_change_after_remove_top1 — network density drop when top contributor
                                       is removed (more negative = more fragile)
    lcc_ratio_after_remove_top1      — fraction of nodes still connected after
                                       top contributor removed (lower = fragile)

    density_change_after_remove_top1 normalization:
      The most-negative value in the dataset (p5) maps to risk=1;
      0.0 (no density change) maps to risk=0.

    Returns (risk_score, dominant_signal_description).
    """
    deg_cent = row.get("degree_centralization",            0.0)
    d_change = row.get("density_change_after_remove_top1", 0.0)
    lcc      = row.get("lcc_ratio_after_remove_top1",      1.0)

    # degree_centralization is already ∈ [0, 1]
    risk_deg = float(np.clip(deg_cent, 0.0, 1.0))

    # density_change ≤ 0; most-negative = highest risk
    # norms.density_change_lo is the most-extreme (most-negative) p5 value
    # map [lo, 0] → [1, 0]
    dc_lo = min(norms.density_change_lo, -1e-9)  # ensure lo < 0
    risk_dc = float(np.clip((d_change - 0.0) / (dc_lo - 0.0), 0.0, 1.0))

    # lcc_ratio ∈ [0, 1]; lower = more fragile = higher risk
    risk_lcc = 1.0 - float(np.clip(lcc, 0.0, 1.0))

    score = float(np.mean([risk_deg, risk_dc, risk_lcc]))

    signals = [
        (risk_deg, f"degree centralization = {deg_cent:.2f} (star topology)"),
        (risk_dc,  f"density drops {abs(d_change):.2f} when top contributor leaves"),
        (risk_lcc, f"only {lcc:.0%} of contributors stay connected after top-1 removed"),
    ]
    dominant = max(signals, key=lambda t: t[0])[1]

    return score, dominant


# ── 4. Composite score and tier ────────────────────────────────────────────────

def score_to_tier(score: float) -> str:
    """
    Map a composite risk score in [0, 1] to a human-readable tier.

    Tiers
    -----
    Critical : score >= 0.70  — multiple strong warning signals
    High     : score >= 0.55  — clear risk, warrants attention
    Medium   : score >= 0.40  — moderate signals, monitor
    Low      : score <  0.40  — no strong indicators of risk
    """
    if score >= TIER_THRESHOLDS["Critical"]:
        return "Critical"
    if score >= TIER_THRESHOLDS["High"]:
        return "High"
    if score >= TIER_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


# ── 5. Score all repos ────────────────────────────────────────────────────────

def score_all_repos(
    feature_df: pd.DataFrame,
    weights: dict[str, float] = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    """
    Compute all risk components and the composite score for every repository.

    Normalization is fitted on feature_df itself (cohort-relative scoring).

    Parameters
    ----------
    feature_df : feature matrix from feature_engineering.build_feature_matrix()
                 or loaded from outputs/features.csv.
                 repo_name may be a column or the index.
    weights    : component weights; must have keys 'activity', 'contributor',
                 'structural'.  Values should sum to 1.0.

    Returns
    -------
    pd.DataFrame — one row per repo with columns:
        repo_name, activity_risk, contributor_risk, structural_risk,
        composite_risk, risk_tier,
        activity_signal, contributor_signal, structural_signal
    """
    df = feature_df.copy()
    if "repo_name" in df.columns:
        df = df.set_index("repo_name")

    norms = _DatasetNorms(df)

    w_act  = weights.get("activity",    DEFAULT_WEIGHTS["activity"])
    w_cont = weights.get("contributor", DEFAULT_WEIGHTS["contributor"])
    w_str  = weights.get("structural",  DEFAULT_WEIGHTS["structural"])

    rows = []
    for repo, row in df.iterrows():
        act_score,  act_sig  = _activity_risk(row,     norms)
        cont_score, cont_sig = _contributor_risk(row,  norms)
        str_score,  str_sig  = _structural_risk(row,   norms)

        composite = float(np.clip(
            w_act * act_score + w_cont * cont_score + w_str * str_score,
            0.0, 1.0
        ))

        rows.append({
            "repo_name":          repo,
            "activity_risk":      round(act_score,  4),
            "contributor_risk":   round(cont_score, 4),
            "structural_risk":    round(str_score,  4),
            "composite_risk":     round(composite,  4),
            "risk_tier":          score_to_tier(composite),
            "activity_signal":    act_sig,
            "contributor_signal": cont_sig,
            "structural_signal":  str_sig,
        })

    return pd.DataFrame(rows)


# ── 6. Rank repos ─────────────────────────────────────────────────────────────

def rank_repos_by_risk(
    scored_df: pd.DataFrame,
    score_col: str = "composite_risk",
) -> pd.DataFrame:
    """
    Return scored_df sorted by score_col descending (highest risk first).
    """
    return scored_df.sort_values(score_col, ascending=False).reset_index(drop=True)


# ── 7. Per-repo explanation ───────────────────────────────────────────────────

def explain_repo(
    repo_name: str,
    scored_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model_proba: Optional[pd.Series] = None,
) -> None:
    """
    Print a human-readable risk explanation for one repository.

    Parameters
    ----------
    repo_name   : the repository to explain
    scored_df   : output of score_all_repos()
    feature_df  : the original feature matrix (for raw feature values)
    model_proba : optional Series (index=repo_name) of model prob_y1 values
    """
    row = scored_df[scored_df["repo_name"] == repo_name]
    if row.empty:
        print(f"[explain] '{repo_name}' not found in scored_df")
        return

    r = row.iloc[0]

    feat_df = feature_df.copy()
    if "repo_name" in feat_df.columns:
        feat_df = feat_df.set_index("repo_name")

    w = 60
    print(f"\n{'═' * w}")
    print(f"  Risk Report: {repo_name}")
    print(f"{'═' * w}")
    print(f"  Composite risk : {r['composite_risk']:.3f}  [{r['risk_tier']}]")
    if model_proba is not None and repo_name in model_proba.index:
        print(f"  Model prob_y1  : {model_proba[repo_name]:.3f}  "
              f"({'at risk' if model_proba[repo_name] >= 0.5 else 'stable'})")
    print()

    print(f"  {'Component':<22} {'Score':>7}  {'Dominant signal'}")
    print(f"  {'─' * 56}")
    components = [
        ("Activity",    "activity_risk",    "activity_signal"),
        ("Contributor", "contributor_risk", "contributor_signal"),
        ("Structural",  "structural_risk",  "structural_signal"),
    ]
    for name, score_col, sig_col in components:
        bar = "█" * int(r[score_col] * 10) + "░" * (10 - int(r[score_col] * 10))
        print(f"  {name:<22} {r[score_col]:>5.3f}  {bar}  {r[sig_col]}")

    print()
    print("  Key feature values:")
    show_features = [
        "active_contributor_slope", "contribution_volume_slope",
        "num_contributors", "top1_share", "gini_coefficient",
        "degree_centralization", "density_change_after_remove_top1",
        "lcc_ratio_after_remove_top1",
    ]
    if repo_name in feat_df.index:
        for feat in show_features:
            if feat in feat_df.columns:
                val = feat_df.loc[repo_name, feat]
                print(f"    {feat:<42}  {val:.4f}")
    print(f"{'═' * w}\n")


# ── 8. Orchestrator ───────────────────────────────────────────────────────────

def run_scoring(outputs_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load features.csv and loo_predictions.csv, compute all risk scores,
    print the ranked table, explain each tier, and save outputs/risk_scores.csv.

    Returns the ranked scored DataFrame.
    """
    if outputs_dir is None:
        outputs_dir = os.path.join(os.path.dirname(_SRC), "outputs")

    def _load(name: str) -> pd.DataFrame:
        p = os.path.join(outputs_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run feature_engineering.py first.")
        return pd.read_csv(p)

    _sep = lambda t: print(f"\n{'=' * 62}\n  {t}\n{'=' * 62}")

    # ── load ──────────────────────────────────────────────────────────────────
    _sep("Loading artifacts")
    feature_df = _load("features.csv")
    labels_df  = _load("labels.csv")
    pred_df    = _load("loo_predictions.csv")

    model_proba = pred_df.set_index("repo_name")["prob_y1"]
    y_true      = labels_df.set_index("repo_name")["y"]

    print(f"  {len(feature_df)} repos, {feature_df.shape[1]-1} features")

    # ── score ─────────────────────────────────────────────────────────────────
    _sep("Computing risk scores")
    scored = score_all_repos(feature_df)
    ranked = rank_repos_by_risk(scored)

    # attach ground-truth and model probability for comparison
    ranked = ranked.merge(
        y_true.reset_index().rename(columns={"y": "y_true"}),
        on="repo_name", how="left"
    ).merge(
        model_proba.reset_index().rename(columns={"prob_y1": "model_prob_y1"}),
        on="repo_name", how="left"
    )

    # ── ranked table ──────────────────────────────────────────────────────────
    _sep("Risk ranking (highest → lowest)")
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 130)
    pd.set_option("display.max_columns", None)

    display_cols = [
        "repo_name", "composite_risk", "risk_tier",
        "activity_risk", "contributor_risk", "structural_risk",
        "model_prob_y1", "y_true",
    ]
    print(ranked[display_cols].to_string(index=False))

    # ── tier summary ──────────────────────────────────────────────────────────
    _sep("Tier distribution")
    tier_order = ["Critical", "High", "Medium", "Low"]
    for tier in tier_order:
        repos = ranked[ranked["risk_tier"] == tier]["repo_name"].tolist()
        print(f"\n  {tier} ({len(repos)} repos):")
        for r in repos:
            row   = ranked[ranked["repo_name"] == r].iloc[0]
            label = "y=1" if row["y_true"] == 1 else "y=0"
            print(f"    [{label}]  {r:<45}  score={row['composite_risk']:.3f}  "
                  f"model={row['model_prob_y1']:.3f}")

    # ── alignment with model and labels ───────────────────────────────────────
    _sep("Rule score vs. model vs. ground truth")
    ranked["score_flag"] = (ranked["composite_risk"] >= 0.55).astype(int)
    ranked["model_flag"] = (ranked["model_prob_y1"]  >= 0.50).astype(int)

    agree = (ranked["score_flag"] == ranked["y_true"]).sum()
    model_agree = (ranked["model_flag"] == ranked["y_true"]).sum()
    both_agree  = (
        (ranked["score_flag"] == ranked["y_true"]) &
        (ranked["model_flag"] == ranked["y_true"])
    ).sum()

    print(f"\n  Threshold: score≥0.55 → at risk,  model_prob≥0.50 → at risk")
    print(f"  Rule score agrees with y_true : {agree}/{len(ranked)}")
    print(f"  Model prob  agrees with y_true : {model_agree}/{len(ranked)}")
    print(f"  Both agree                     : {both_agree}/{len(ranked)}")

    # disagreements
    disagree = ranked[ranked["score_flag"] != ranked["model_flag"]]
    if not disagree.empty:
        print(f"\n  Score vs. model disagreements ({len(disagree)}):")
        for _, row in disagree.iterrows():
            print(f"    {row['repo_name']:<45} "
                  f"score={row['composite_risk']:.3f}({['stable','at-risk'][row['score_flag']]})  "
                  f"model={row['model_prob_y1']:.3f}({['stable','at-risk'][row['model_flag']]})  "
                  f"y={int(row['y_true'])}")

    # ── per-repo explanations for highest-risk repos ──────────────────────────
    _sep("Detailed explanations — Critical and High tier")
    high_risk = ranked[ranked["risk_tier"].isin(["Critical", "High"])]["repo_name"].tolist()
    for repo in high_risk:
        explain_repo(repo, scored, feature_df, model_proba)

    # ── save ──────────────────────────────────────────────────────────────────
    _sep("Saving")
    os.makedirs(outputs_dir, exist_ok=True)

    save_cols = [
        "repo_name", "composite_risk", "risk_tier",
        "activity_risk", "contributor_risk", "structural_risk",
        "activity_signal", "contributor_signal", "structural_signal",
        "model_prob_y1", "y_true",
    ]
    out_path = os.path.join(outputs_dir, "risk_scores.csv")
    ranked[save_cols].to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    return ranked


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_scoring()
