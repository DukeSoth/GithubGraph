"""
run_preprocessing.py
--------------------
Orchestrates the full preprocessing pipeline:

  Step 1 — Streaming scan of all raw files → repo summary table
  Step 2 — Apply inclusion filter
  Step 3 — Recommend candidate repo list (~18-20 repos)
  Step 4 — Targeted reload of only candidate repos' events
  Step 5 — Clean and split into obs / future DataFrames
  Step 6 — Print schema and summary statistics
  Step 7 — Save outputs to data/processed/

Run from the project root:
    python src/run_preprocessing.py

Outputs saved to data/processed/:
    repo_summary_all.parquet      -- summary for every repo seen in the data
    repo_summary_candidates.csv   -- filtered + recommended repos
    obs_events.parquet            -- clean observation events for candidate repos
    future_events.parquet         -- clean future events for candidate repos
"""

import os
import sys

# allow imports from src/ when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import pandas as pd
import numpy as np

from data_loading import (
    OBS_START, OBS_END, FUTURE_START, FUTURE_END,
    load_events_for_repos,
)
from preprocessing import (
    scan_and_build_repo_summary,
    filter_candidate_repos,
    recommend_repo_list,
    clean_events,
    split_windows,
    assign_subwindow,
)

# ── configuration ──────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "data", "processed")

MIN_OBS_EVENTS       = 20
MIN_OBS_CONTRIBUTORS = 3
N_RECOMMENDED        = 20


def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_schema(df: pd.DataFrame, name: str):
    print(f"\n--- schema: {name} ---")
    for col in df.columns:
        dtype = df[col].dtype
        n_null = df[col].isna().sum()
        print(f"  {col:<45} {str(dtype):<20} nulls={n_null}")
    print(f"  TOTAL rows: {len(df):,}")


def print_summary_stats(summary_df: pd.DataFrame):
    """Print descriptive statistics for the repo summary table."""
    print("\n--- observation window stats (candidate repos only) ---")
    num_cols = [
        "observation_total_events",
        "observation_unique_contributors",
        "observation_push_events",
        "observation_pull_request_events",
        "observation_issues_events",
        "observation_release_events",
    ]
    print(summary_df[num_cols].describe().round(1).to_string())

    print("\n--- future window stats ---")
    fut_cols = [
        "future_total_events",
        "future_unique_contributors",
        "future_to_observation_volume_ratio",
    ]
    print(summary_df[fut_cols].describe().round(3).to_string())

    print("\n--- label distribution ---")
    flag_col = "has_90_day_future_inactivity_flag"
    vol_flag = summary_df["future_to_observation_volume_ratio"].fillna(0) < 0.5
    inact    = summary_df[flag_col].sum()
    vol_drop = vol_flag.sum()
    y1       = (summary_df[flag_col] | vol_flag).sum()
    n        = len(summary_df)
    print(f"  90-day inactivity flag = True  : {inact}/{n}")
    print(f"  volume < 50%% of obs         : {vol_drop}/{n}")
    print(f"  y=1 (either condition)       : {y1}/{n}  ({100*y1/n:.1f}%)")
    print(f"  y=0                          : {n - y1}/{n}  ({100*(n-y1)/n:.1f}%)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── STEP 1: streaming scan ─────────────────────────────────────────────────
    print_separator("STEP 1: Streaming scan → repo summary")

    # Use pickle for the summary cache — it's large (~4M rows) and we want a
    # fast, reliable save without a parquet engine dependency.
    summary_path = os.path.join(OUTPUT_DIR, "repo_summary_all.pkl")

    if os.path.exists(summary_path):
        print(f"[cache] loading existing summary from {summary_path}")
        summary_all = pd.read_pickle(summary_path)
    else:
        summary_all = scan_and_build_repo_summary(
            data_dir              = DATA_DIR,
            obs_start             = OBS_START,
            obs_end               = OBS_END,
            future_start          = FUTURE_START,
            future_end            = FUTURE_END,
            min_obs_events        = MIN_OBS_EVENTS,
            min_obs_contributors  = MIN_OBS_CONTRIBUTORS,
            verbose               = True,
        )
        summary_all.to_pickle(summary_path)
        print(f"[saved] {summary_path}")

    print_schema(summary_all, "repo_summary_all")

    print(f"\n  Total unique repos seen: {len(summary_all):,}")
    print(f"  Repos with ANY obs events: "
          f"{(summary_all['observation_total_events'] > 0).sum():,}")
    print(f"  Repos with ANY future events: "
          f"{(summary_all['future_total_events'] > 0).sum():,}")

    # ── STEP 2: apply inclusion filter ────────────────────────────────────────
    print_separator("STEP 2: Apply inclusion filter")

    candidates = filter_candidate_repos(
        summary_all,
        min_obs_events       = MIN_OBS_EVENTS,
        min_obs_contributors = MIN_OBS_CONTRIBUTORS,
    )

    # ── STEP 3: recommend ~20 repos ───────────────────────────────────────────
    print_separator("STEP 3: Recommend candidate repo list")

    recommended = recommend_repo_list(candidates, n_total=N_RECOMMENDED)

    print(f"\n  Recommended {len(recommended)} repositories:\n")
    display_cols = [
        "repo_name", "tier",
        "observation_total_events", "observation_unique_contributors",
        "future_total_events", "future_to_observation_volume_ratio",
        "has_90_day_future_inactivity_flag",
    ]
    print(recommended[display_cols].to_string(index=True))

    # compute and attach the binary label for display
    recommended["y"] = (
        recommended["has_90_day_future_inactivity_flag"] |
        (recommended["future_to_observation_volume_ratio"].fillna(0) < 0.5)
    ).astype(int)

    print(f"\n  Label distribution in recommended set:")
    print(f"  y=1: {recommended['y'].sum()}   y=0: {(recommended['y']==0).sum()}")

    # save candidate summary
    cand_path = os.path.join(OUTPUT_DIR, "repo_summary_candidates.csv")
    recommended.to_csv(cand_path, index=False)
    print(f"\n[saved] {cand_path}")

    # ── STEP 4: reload events for candidate repos only ────────────────────────
    print_separator("STEP 4: Load events for candidate repos")

    repo_set = set(recommended["repo_name"].tolist())

    raw_df = load_events_for_repos(
        data_dir   = DATA_DIR,
        start_date = OBS_START,
        end_date   = FUTURE_END,
        repo_names = repo_set,
        verbose    = True,
    )

    # ── STEP 5: clean and split ────────────────────────────────────────────────
    print_separator("STEP 5: Clean and split events")

    clean_df = clean_events(raw_df)
    obs_df, future_df = split_windows(clean_df)

    # add subwindow labels to observation events for use in feature engineering
    obs_df = assign_subwindow(obs_df)

    # save — candidate event tables are small (~thousands of rows), use parquet
    # with a CSV fallback so the pipeline never crashes on a missing engine
    def save_df(df, stem):
        try:
            path = os.path.join(OUTPUT_DIR, f"{stem}.parquet")
            df.to_parquet(path, index=False)
        except ImportError:
            path = os.path.join(OUTPUT_DIR, f"{stem}.csv")
            df.to_csv(path, index=False)
        print(f"[saved] {path}")
        return path

    obs_path    = save_df(obs_df,    "obs_events")
    future_path = save_df(future_df, "future_events")

    # ── STEP 6: schemas and statistics ────────────────────────────────────────
    print_separator("STEP 6: Output schema and statistics")

    print_schema(obs_df,    "obs_events")
    print_schema(future_df, "future_events")
    print_schema(recommended, "repo_summary_candidates")

    print_summary_stats(recommended)

    # per-repo event breakdown
    print("\n--- per-repo event counts (obs window) ---")
    per_repo = (
        obs_df.groupby("repo_name")
        .agg(
            total_events    = ("id",          "count"),
            unique_contribs = ("actor_login", "nunique"),
            subwindows_seen = ("subwindow",   "nunique"),
        )
        .sort_values("total_events", ascending=False)
    )
    print(per_repo.to_string())

    print("\n--- subwindow coverage (obs events per subwindow, all repos) ---")
    sw = obs_df.groupby("subwindow").size().rename("event_count")
    print(sw.to_string())

    print_separator("DONE")
    print(f"\n  All outputs written to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
