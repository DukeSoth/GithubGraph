"""
labeling.py
-----------
Derive binary collaboration-risk labels for each candidate repository using
only future-window events (2015-07-01 to 2015-09-30).

Label rule
----------
  y = 1  ("at risk of collaboration collapse")  if EITHER:

    Condition A — volume drop:
      future_to_observation_volume_ratio < 0.5
      i.e. the repo produced less than half the event volume in the future
      window compared to what it produced in the observation window.

    Condition B — sustained inactivity:
      has_90_day_future_inactivity_flag == True
      i.e. there exists a consecutive stretch of ≥ 90 days inside the
      future window with no qualifying contribution events.
      The future window is 92 days (Jul 1 – Sep 30 inclusive), so this
      flag fires when the repo is essentially silent for the entire period.

  y = 0  otherwise.

Leakage discipline
------------------
  This module reads from obs_df ONLY to obtain observation-period event totals
  as the denominator for the volume ratio.  That count is used solely to
  define the label threshold — it is not passed to the model as a feature.

  All inactivity and volume computations are derived from future_df only.

  Call-site contract:
    - obs_df must contain events from [OBS_START, OBS_END] only.
    - future_df must contain events from [FUTURE_START, FUTURE_END] only.
    - _validate_no_leakage() enforces these bounds with printed warnings.
    - Never pass future_df rows into the feature matrix.

Output columns
--------------
  repo_name                           str
  obs_total_events                    int    denominator for ratio; diagnostic only
  future_total_events                 int    total events in future window
  future_max_inactive_gap_days        int    longest consecutive inactive stretch (days)
  future_to_observation_volume_ratio  float  future_total / obs_total  (NaN if obs=0)
  has_90_day_future_inactivity_flag   bool   future_max_inactive_gap_days >= 90
  y                                   int    binary label: 1 = at risk, 0 = stable

Public API
----------
  compute_labels(obs_df, future_df, repo_names)  -> pd.DataFrame
  label_summary(labeled_df)                      -> None
  get_label_series(labeled_df)                   -> pd.Series
"""

from __future__ import annotations

import os
import sys
from datetime import date
from typing import Optional

import pandas as pd

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data_loading import (
    FUTURE_START, FUTURE_END,
    OBS_START, OBS_END,
    load_datasets, load_candidate_repos,
)
from preprocessing import max_gap_days


# ── constants ──────────────────────────────────────────────────────────────────

VOLUME_DROP_THRESHOLD: float = 0.5
INACTIVITY_THRESHOLD_DAYS: int = 90

# pre-parsed date objects for max_gap_days (avoid re-parsing in the hot loop)
_FUT_LO: date = date.fromisoformat(FUTURE_START)
_FUT_HI: date = date.fromisoformat(FUTURE_END)
_OBS_LO: date = date.fromisoformat(OBS_START)
_OBS_HI: date = date.fromisoformat(OBS_END)

# canonical column names used throughout this module
COL_REPO        = "repo_name"
COL_OBS_TOTAL   = "obs_total_events"
COL_FUT_TOTAL   = "future_total_events"
COL_MAX_GAP     = "future_max_inactive_gap_days"
COL_RATIO       = "future_to_observation_volume_ratio"
COL_INACTIVITY  = "has_90_day_future_inactivity_flag"
COL_LABEL       = "y"

# output column order
_COLUMN_ORDER = [
    COL_REPO,
    COL_OBS_TOTAL,
    COL_FUT_TOTAL,
    COL_MAX_GAP,
    COL_RATIO,
    COL_INACTIVITY,
    COL_LABEL,
]


# ── leakage guard ──────────────────────────────────────────────────────────────

def _validate_no_leakage(obs_df: pd.DataFrame, future_df: pd.DataFrame) -> None:
    """
    Warn if either DataFrame contains events outside its intended time window.

    This is a defensive check, not a hard error — the pipeline should already
    have split events correctly via preprocessing.split_windows().  Violations
    are printed so they are visible without crashing a longer pipeline run.
    """
    fut_boundary = pd.Timestamp(FUTURE_START, tz="UTC")
    obs_boundary = pd.Timestamp(OBS_END,      tz="UTC") + pd.Timedelta(days=1)

    # obs_df should contain nothing in the future window
    obs_leaks = obs_df[obs_df["created_at"] >= fut_boundary]
    if not obs_leaks.empty:
        print(
            f"[labeling] WARNING: obs_df contains {len(obs_leaks)} event(s) "
            f"with created_at >= {FUTURE_START}. This may indicate a split error."
        )

    # future_df should contain nothing in the observation window
    fut_leaks = future_df[future_df["created_at"] < fut_boundary]
    if not fut_leaks.empty:
        print(
            f"[labeling] WARNING: future_df contains {len(fut_leaks)} event(s) "
            f"with created_at < {FUTURE_START}. This may indicate a split error."
        )


# ── per-repo future statistics ─────────────────────────────────────────────────

def _future_stats_one_repo(
    repo_events: pd.DataFrame,
) -> tuple[int, int, bool]:
    """
    Compute future-window statistics for a single repository.

    Uses only the rows of future_df that belong to this repository.

    Parameters
    ----------
    repo_events : future_df filtered to one repo (may be empty)

    Returns
    -------
    (future_total_events, future_max_inactive_gap_days, has_90_day_flag)
    """
    total = len(repo_events)

    # extract unique active calendar dates as Python date objects
    # .dt.date works on both tz-aware and tz-naive datetime columns
    event_dates: list[date] = (
        repo_events["created_at"].dt.date.unique().tolist()
        if total > 0
        else []
    )

    # max_gap_days accounts for gaps at both ends of the window:
    #   gap from _FUT_LO to first event date
    #   gaps between consecutive event dates
    #   gap from last event date to _FUT_HI
    # An empty event_dates list returns the full window length (92 days here).
    gap  = max_gap_days(event_dates, _FUT_LO, _FUT_HI)
    flag = gap >= INACTIVITY_THRESHOLD_DAYS

    return total, gap, flag


def _build_future_stats(
    future_df: pd.DataFrame,
    repo_names: list[str],
) -> pd.DataFrame:
    """
    Compute future-window statistics for every repository in repo_names.

    Repositories absent from future_df (went completely silent) receive:
      future_total_events = 0
      future_max_inactive_gap_days = 92   (full window)
      has_90_day_future_inactivity_flag = True

    Parameters
    ----------
    future_df  : cleaned future-window event DataFrame
    repo_names : ordered list of repositories to compute stats for

    Returns
    -------
    pd.DataFrame with columns COL_REPO, COL_FUT_TOTAL, COL_MAX_GAP, COL_INACTIVITY
    """
    rows = []
    for repo in repo_names:
        repo_events = future_df[future_df[COL_REPO] == repo]
        total, gap, flag = _future_stats_one_repo(repo_events)
        rows.append({
            COL_REPO:       repo,
            COL_FUT_TOTAL:  total,
            COL_MAX_GAP:    gap,
            COL_INACTIVITY: flag,
        })

    return pd.DataFrame(rows)


# ── main label computation ─────────────────────────────────────────────────────

def compute_labels(
    obs_df: pd.DataFrame,
    future_df: pd.DataFrame,
    repo_names: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute binary risk labels from observation and future event DataFrames.

    This is the authoritative implementation of the label rule.  All
    downstream modules (modeling, evaluation) should call this function rather
    than re-implementing the conditions.

    Parameters
    ----------
    obs_df      : cleaned observation-window events (Jan–Jun 2015).
                  Used ONLY to derive obs_total_events as the ratio denominator.
    future_df   : cleaned future-window events (Jul–Sep 2015).
                  All label signals are derived from this DataFrame.
    repo_names  : repositories to label.  Defaults to all repos present in
                  obs_df (the set we have features for).

    Returns
    -------
    pd.DataFrame with one row per repo and columns (in order):
      repo_name, obs_total_events, future_total_events,
      future_max_inactive_gap_days, future_to_observation_volume_ratio,
      has_90_day_future_inactivity_flag, y

    Leakage note
    ------------
    obs_total_events is an intermediate column derived from the observation
    period.  It is included here for transparency and diagnostic purposes but
    must NOT be passed to the model as a feature — it is equivalent to
    total_contributions in the feature matrix and would constitute leakage of
    label-construction logic into the features.
    """
    _validate_no_leakage(obs_df, future_df)

    if repo_names is None:
        repo_names = sorted(obs_df[COL_REPO].unique().tolist())

    # ── observation totals (denominator only) ─────────────────────────────────
    # groupby.size() counts every qualifying event in obs_df for each repo.
    # This count is NOT a feature; it only anchors what "50% volume drop" means.
    obs_totals: pd.Series = obs_df.groupby(COL_REPO).size().rename(COL_OBS_TOTAL)

    # ── future statistics ──────────────────────────────────────────────────────
    df = _build_future_stats(future_df, repo_names)

    # attach obs totals; fill 0 for any repo absent from obs_df (shouldn't
    # occur in a well-filtered pipeline, but be safe)
    df[COL_OBS_TOTAL] = df[COL_REPO].map(obs_totals).fillna(0).astype(int)

    # ── volume ratio ──────────────────────────────────────────────────────────
    # NaN when obs_total == 0 (no denominator); treated as 0.0 in label rule
    df[COL_RATIO] = df.apply(
        lambda r: r[COL_FUT_TOTAL] / r[COL_OBS_TOTAL]
        if r[COL_OBS_TOTAL] > 0
        else float("nan"),
        axis=1,
    )

    # ── label rule ────────────────────────────────────────────────────────────
    # Condition A: future volume < 50% of observation volume
    volume_drop = df[COL_RATIO].fillna(0.0) < VOLUME_DROP_THRESHOLD

    # Condition B: at least one 90-day consecutive gap in the future window
    inactivity  = df[COL_INACTIVITY].astype(bool)

    df[COL_LABEL] = (volume_drop | inactivity).astype(int)

    return df[_COLUMN_ORDER]


# ── reporting helpers ──────────────────────────────────────────────────────────

def label_summary(labeled_df: pd.DataFrame) -> None:
    """
    Print label distribution and per-condition breakdown to stdout.

    Parameters
    ----------
    labeled_df : output of compute_labels()
    """
    n         = len(labeled_df)
    n_at_risk = int(labeled_df[COL_LABEL].sum())
    n_stable  = n - n_at_risk

    vol_flag   = (labeled_df[COL_RATIO].fillna(0.0) < VOLUME_DROP_THRESHOLD)
    inact_flag = labeled_df[COL_INACTIVITY].astype(bool)
    both       = vol_flag & inact_flag
    either     = vol_flag | inact_flag

    print("── Label distribution ─────────────────────────────────────")
    print(f"  Total repos            : {n}")
    print(f"  y = 1  (at risk)       : {n_at_risk}  ({100 * n_at_risk / n:.1f}%)")
    print(f"  y = 0  (stable)        : {n_stable}   ({100 * n_stable / n:.1f}%)")
    print()
    print("── Condition breakdown ────────────────────────────────────")
    print(f"  A only  (volume drop)  : {int((vol_flag & ~inact_flag).sum())}")
    print(f"  B only  (inactivity)   : {int((inact_flag & ~vol_flag).sum())}")
    print(f"  A and B (both)         : {int(both.sum())}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  A total (volume drop)  : {int(vol_flag.sum())}")
    print(f"  B total (inactivity)   : {int(inact_flag.sum())}")
    print(f"  y = 1  (A or B)        : {int(either.sum())}")
    print()
    print(f"  Volume threshold       : < {VOLUME_DROP_THRESHOLD:.0%} of obs volume")
    print(f"  Inactivity threshold   : >= {INACTIVITY_THRESHOLD_DAYS} consecutive days")
    print(f"  Future window length   : {(_FUT_HI - _FUT_LO).days + 1} days  "
          f"({FUTURE_START} – {FUTURE_END})")


def get_label_series(
    labeled_df: pd.DataFrame,
    index_col: str = COL_REPO,
) -> pd.Series:
    """
    Return the y label Series indexed by repo_name.

    Parameters
    ----------
    labeled_df : output of compute_labels()
    index_col  : column to use as the Series index

    Returns
    -------
    pd.Series — index = repo_name, values ∈ {0, 1}, name = 'y'
    """
    return labeled_df.set_index(index_col)[COL_LABEL].rename(COL_LABEL)


# backward-compatibility alias for code that used the old stub's API
assign_labels = compute_labels


# ── main / smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  labeling.py — compute repository risk labels")
    print("=" * 65)

    candidates = load_candidate_repos()
    repo_names = candidates["repo_name"].tolist()

    obs_df, future_df = load_datasets(repo_names=set(repo_names))
    print(f"\nobs_df    : {len(obs_df):,} events across {obs_df[COL_REPO].nunique()} repos")
    print(f"future_df : {len(future_df):,} events across {future_df[COL_REPO].nunique()} repos")

    # ── compute labels ────────────────────────────────────────────────────────
    print()
    labeled = compute_labels(obs_df, future_df, repo_names)

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    label_summary(labeled)

    # ── full table ────────────────────────────────────────────────────────────
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 130)
    pd.set_option("display.max_columns", None)

    print("\n── Full label table ───────────────────────────────────────")
    print(labeled.to_string(index=False))

    # ── cross-check against pre-computed values in the candidates CSV ─────────
    # The CSV was produced by run_preprocessing.py during the initial scan.
    # Checking against it confirms our per-event computation matches.
    print("\n── Cross-check vs. pre-computed candidates CSV ────────────")

    ref_cols = [
        "repo_name",
        "future_total_events",
        "future_to_observation_volume_ratio",
        "has_90_day_future_inactivity_flag",
    ]
    ref = candidates[ref_cols].rename(columns={
        "future_total_events":                "ref_fut_total",
        "future_to_observation_volume_ratio":  "ref_ratio",
        "has_90_day_future_inactivity_flag":   "ref_inactivity",
    })

    check = labeled.merge(ref, on="repo_name", how="left")
    check["total_match"]     = check[COL_FUT_TOTAL] == check["ref_fut_total"]
    check["inactivity_match"] = check[COL_INACTIVITY] == check["ref_inactivity"]
    # ratio may differ slightly due to floating-point; check within tolerance
    check["ratio_match"] = (
        (check[COL_RATIO] - check["ref_ratio"]).abs() < 1e-6
    ) | (check[COL_RATIO].isna() & check["ref_ratio"].isna())

    all_match = check[["total_match", "ratio_match", "inactivity_match"]].all(axis=1)
    print(f"  Repos checked                  : {len(check)}")
    print(f"  future_total_events match      : {check['total_match'].sum()}/{len(check)}")
    print(f"  volume_ratio match (1e-6 tol)  : {check['ratio_match'].sum()}/{len(check)}")
    print(f"  inactivity_flag match          : {check['inactivity_match'].sum()}/{len(check)}")
    print(f"  All fields match               : {all_match.sum()}/{len(check)}")

    if not all_match.all():
        mismatches = check[~all_match][["repo_name", COL_FUT_TOTAL, "ref_fut_total",
                                        COL_RATIO, "ref_ratio",
                                        COL_INACTIVITY, "ref_inactivity"]]
        print("\n  Mismatched rows:")
        print(mismatches.to_string(index=False))

    # ── save ──────────────────────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(_SRC), "outputs")
    if os.path.isdir(output_dir):
        path = os.path.join(output_dir, "labels.csv")
        labeled.to_csv(path, index=False)
        print(f"\n[saved] {path}")
    else:
        print(f"\n[save] 'outputs/' not found — skipping.")

    print("\n[done]")
