"""
preprocessing.py
----------------
Responsibility: transform raw event records into analysis-ready structures.

Public API:
  clean_events()                -- dedup, drop nulls, confirm datetime
  split_windows()               -- slice into observation vs. future DataFrames
  max_gap_days()                -- detect longest consecutive inactive stretch
  scan_and_build_repo_summary() -- streaming single-pass over ALL files to
                                   produce the per-repo summary table without
                                   loading all events into memory
  filter_candidate_repos()      -- apply minimum inclusion criteria
  recommend_repo_list()         -- tiered selection of ~18-20 diverse repos
"""

import gzip
import json
import os
from collections import defaultdict
from datetime import date, timedelta, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from data_loading import (
    ALLOWED_TYPES,
    OBS_START, OBS_END,
    FUTURE_START, FUTURE_END,
    list_files_in_range,
    iter_events,
)

# ── 1. Clean event DataFrame ───────────────────────────────────────────────────

def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the mandatory cleaning steps to a raw event DataFrame produced by
    load_events_for_repos().

    Steps (in order):
      1. Drop rows with null actor_login, repo_name, or created_at
      2. Deduplicate on event id (keep first occurrence)
      3. Drop rows where created_at could not be parsed
      4. Sort by created_at ascending

    Parameters
    ----------
    df : raw DataFrame from load_events_for_repos()

    Returns
    -------
    Cleaned DataFrame with the same column schema.
    """
    if df.empty:
        return df.copy()

    n_raw = len(df)

    # 1. drop missing required fields
    df = df.dropna(subset=["actor_login", "repo_name", "created_at"])
    df = df[df["actor_login"].str.strip() != ""]
    df = df[df["repo_name"].str.strip()   != ""]

    # 2. deduplicate on id (cross-file duplicates are possible in GH Archive)
    df = df.drop_duplicates(subset=["id"], keep="first")

    # 3. ensure created_at is UTC-aware datetime; drop unparseable rows
    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"])

    # 4. sort chronologically
    df = df.sort_values("created_at").reset_index(drop=True)

    print(
        f"[clean] {n_raw:,} → {len(df):,} events "
        f"({n_raw - len(df):,} removed by cleaning)"
    )
    return df


# ── 2. Window split ────────────────────────────────────────────────────────────

def split_windows(
    df: pd.DataFrame,
    obs_start:    str = OBS_START,
    obs_end:      str = OBS_END,
    future_start: str = FUTURE_START,
    future_end:   str = FUTURE_END,
) -> tuple:
    """
    Split a clean event DataFrame into observation and future slices.

    Date boundaries are inclusive on both ends and treated as calendar-day
    boundaries in UTC (i.e. obs_end means up to 23:59:59 of that day).

    Returns
    -------
    (obs_df, future_df) — two DataFrames, same column schema as input.
    """
    # build UTC-aware Timestamps at the start and end of each day
    obs_lo  = pd.Timestamp(obs_start,    tz="UTC")
    obs_hi  = pd.Timestamp(obs_end,      tz="UTC") + pd.Timedelta(days=1)
    fut_lo  = pd.Timestamp(future_start, tz="UTC")
    fut_hi  = pd.Timestamp(future_end,   tz="UTC") + pd.Timedelta(days=1)

    ts = df["created_at"]
    obs_df    = df[(ts >= obs_lo)  & (ts < obs_hi)].copy()
    future_df = df[(ts >= fut_lo)  & (ts < fut_hi)].copy()

    print(
        f"[split] observation: {len(obs_df):,} events  "
        f"({obs_start} – {obs_end})"
    )
    print(
        f"[split] future:      {len(future_df):,} events  "
        f"({future_start} – {future_end})"
    )
    return obs_df, future_df


# ── 3. Consecutive-inactivity helper ──────────────────────────────────────────

def max_gap_days(
    event_dates: list,
    window_start: date,
    window_end:   date,
) -> int:
    """
    Return the length (in days) of the longest consecutive stretch with no
    events within [window_start, window_end] (both inclusive).

    Gap calculation includes:
      - gap from window_start to the first event date
      - gaps between consecutive event dates
      - gap from the last event date to window_end

    If event_dates is empty, the entire window is inactive (gap = window
    length in days).

    Parameters
    ----------
    event_dates  : list or set of date objects (or anything date-comparable)
    window_start : date
    window_end   : date

    Returns
    -------
    int — maximum gap length in whole days
    """
    if not event_dates:
        return (window_end - window_start).days + 1

    sorted_dates = sorted(set(event_dates))  # unique days, ascending

    gaps = []

    # gap from window start to first event
    gaps.append((sorted_dates[0] - window_start).days)

    # gaps between consecutive events
    for i in range(1, len(sorted_dates)):
        gaps.append((sorted_dates[i] - sorted_dates[i - 1]).days - 1)

    # gap from last event to window end
    gaps.append((window_end - sorted_dates[-1]).days)

    return max(gaps)


# ── 4. Streaming repo summary — two-pass, memory-efficient ────────────────────

def _scan_obs_counts(
    obs_files: list,
    obs_lo: date,
    obs_hi: date,
    verbose: bool,
) -> tuple:
    """
    Pass 1 (cheap): stream obs files and accumulate only:
      - total event count per repo       (int counter)
      - unique contributor count per repo (approximate via a frozen set)

    We store contributor logins only as hashes (integers) to halve string
    memory.  The count is exact; the hash collision probability is negligible
    for O(100) contributors per repo.

    Returns (obs_total, obs_contrib_hashes) — two plain dicts.
    """
    obs_total   = defaultdict(int)
    obs_contribs = defaultdict(set)   # stores hash(login), not the string

    for i, path in enumerate(obs_files):
        if verbose and i > 0 and i % 500 == 0:
            print(f"  pass-1 obs: {i}/{len(obs_files)} files  |  "
                  f"{len(obs_total):,} repos")
        fname     = os.path.basename(path)
        day_str   = fname.rsplit("-", 1)[0]
        try:
            file_date = date.fromisoformat(day_str)
        except ValueError:
            continue
        if not (obs_lo <= file_date <= obs_hi):
            continue

        for rec in iter_events(path):
            try:
                event_date = date.fromisoformat(rec["created_at"][:10])
            except (ValueError, TypeError):
                continue
            if obs_lo <= event_date <= obs_hi:
                repo = rec["repo_name"]
                obs_total[repo]    += 1
                obs_contribs[repo].add(hash(rec["actor_login"]))

    return obs_total, obs_contribs


def scan_and_build_repo_summary(
    data_dir: str,
    obs_start:    str = OBS_START,
    obs_end:      str = OBS_END,
    future_start: str = FUTURE_START,
    future_end:   str = FUTURE_END,
    min_obs_events:       int = 20,
    min_obs_contributors: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Two-pass memory-efficient streaming scan.

    Pass 1 — obs window only, counts only.
      Accumulates obs event counts and hashed contributor sets for ALL repos.
      Memory: two dicts with int values — lightweight even for 4M repos.

    After pass 1 we identify the repos that pass the minimum inclusion criteria
    (min_obs_events, min_obs_contributors).  For the full dataset this drops
    from ~4M repos to a few thousand, dramatically reducing pass-2 memory.

    Pass 2 — both windows, detailed stats only for filtered repos.
      Collects: event-type breakdown, min/max obs date, future counts,
      future contributor set (strings now, only for filtered repos),
      future event date list (for 90-day gap check).

    Columns in the returned DataFrame
    ----------------------------------
    repo_name
    observation_total_events
    observation_unique_contributors
    observation_push_events
    observation_pull_request_events
    observation_issues_events
    observation_release_events
    observation_first_event_date
    observation_last_event_date
    future_total_events
    future_unique_contributors
    future_first_event_date
    future_last_event_date
    future_to_observation_volume_ratio
    has_90_day_future_inactivity_flag
    """

    obs_lo  = date.fromisoformat(obs_start)
    obs_hi  = date.fromisoformat(obs_end)
    fut_lo  = date.fromisoformat(future_start)
    fut_hi  = date.fromisoformat(future_end)

    obs_files    = list_files_in_range(data_dir, obs_start,    obs_end)
    future_files = list_files_in_range(data_dir, future_start, future_end)

    if verbose:
        print(f"[scan-p1] {len(obs_files)} obs files — counting events & contributors ...")

    # ── Pass 1: cheap obs counts ──────────────────────────────────────────────
    obs_total_all, obs_contribs_all = _scan_obs_counts(
        obs_files, obs_lo, obs_hi, verbose
    )

    if verbose:
        print(f"[scan-p1] done.  {len(obs_total_all):,} repos with obs events.")

    # identify repos that pass the minimum filter
    eligible = {
        repo
        for repo, cnt in obs_total_all.items()
        if cnt >= min_obs_events
        and len(obs_contribs_all.get(repo, set())) >= min_obs_contributors
    }
    if verbose:
        print(f"[scan-p1] {len(eligible):,} repos pass minimum filter "
              f"(≥{min_obs_events} events, ≥{min_obs_contributors} contributors)")

    # free pass-1 data for all non-eligible repos
    obs_total_eligible    = {r: obs_total_all[r]   for r in eligible}
    obs_contribs_eligible = {r: obs_contribs_all[r] for r in eligible}
    del obs_total_all, obs_contribs_all

    # ── Pass 2: detailed stats for eligible repos only ────────────────────────
    if verbose:
        n = len(obs_files) + len(future_files)
        print(f"[scan-p2] {n} files — detailed stats for {len(eligible):,} repos ...")

    obs_type_count = {r: defaultdict(int) for r in eligible}
    obs_first_date = {}
    obs_last_date  = {}

    fut_total    = defaultdict(int)
    fut_contribs = defaultdict(set)   # full strings now (only eligible repos)
    fut_dates    = defaultdict(list)  # all unique future event dates per repo

    all_files_p2 = obs_files + future_files
    for i, path in enumerate(all_files_p2):
        if verbose and i > 0 and i % 500 == 0:
            print(f"  pass-2: {i}/{len(all_files_p2)} files")
        fname     = os.path.basename(path)
        day_str   = fname.rsplit("-", 1)[0]
        try:
            file_date = date.fromisoformat(day_str)
        except ValueError:
            continue

        in_obs    = obs_lo  <= file_date <= obs_hi
        in_future = fut_lo  <= file_date <= fut_hi

        for rec in iter_events(path, repo_filter=eligible):
            repo = rec["repo_name"]
            try:
                event_date = date.fromisoformat(rec["created_at"][:10])
            except (ValueError, TypeError):
                continue

            if in_obs and obs_lo <= event_date <= obs_hi:
                obs_type_count[repo][rec["type"]] += 1
                if repo not in obs_first_date or event_date < obs_first_date[repo]:
                    obs_first_date[repo] = event_date
                if repo not in obs_last_date or event_date > obs_last_date[repo]:
                    obs_last_date[repo] = event_date

            elif in_future and fut_lo <= event_date <= fut_hi:
                fut_total[repo]    += 1
                fut_contribs[repo].add(rec["actor_login"])
                fut_dates[repo].append(event_date)

    if verbose:
        print(f"[scan-p2] done.")

    # ── assemble summary rows ─────────────────────────────────────────────────
    rows = []
    for repo in eligible:
        o_total   = obs_total_eligible[repo]
        o_contrib = len(obs_contribs_eligible[repo])   # count of hashed logins

        o_first   = obs_first_date.get(repo)
        o_last    = obs_last_date.get(repo)

        f_total       = fut_total.get(repo, 0)
        f_contrib     = len(fut_contribs.get(repo, set()))
        f_dates_list  = list(set(fut_dates.get(repo, [])))
        f_first       = min(f_dates_list) if f_dates_list else None
        f_last        = max(f_dates_list) if f_dates_list else None

        ratio           = (f_total / o_total) if o_total > 0 else float("nan")
        gap             = max_gap_days(f_dates_list, fut_lo, fut_hi)
        inactivity_flag = gap >= 90

        tc = obs_type_count[repo]
        rows.append({
            "repo_name":                         repo,
            "observation_total_events":           o_total,
            "observation_unique_contributors":    o_contrib,
            "observation_push_events":            tc.get("PushEvent",        0),
            "observation_pull_request_events":    tc.get("PullRequestEvent", 0),
            "observation_issues_events":          tc.get("IssuesEvent",      0),
            "observation_release_events":         tc.get("ReleaseEvent",     0),
            "observation_first_event_date":       o_first,
            "observation_last_event_date":        o_last,
            "future_total_events":                f_total,
            "future_unique_contributors":         f_contrib,
            "future_first_event_date":            f_first,
            "future_last_event_date":             f_last,
            "future_to_observation_volume_ratio": ratio,
            "has_90_day_future_inactivity_flag":  inactivity_flag,
        })

    summary_df = pd.DataFrame(rows).sort_values(
        "observation_total_events", ascending=False
    ).reset_index(drop=True)

    return summary_df


# ── 5. Inclusion filter ────────────────────────────────────────────────────────

def filter_candidate_repos(
    summary_df: pd.DataFrame,
    min_obs_events:       int = 20,
    min_obs_contributors: int = 3,
) -> pd.DataFrame:
    """
    Apply the project-specified minimum inclusion criteria.

    Keeps only repos that had:
      - at least min_obs_events   contribution events in the observation window
      - at least min_obs_contributors unique contributors in the observation window

    Returns a filtered copy of summary_df (sorted by obs events descending).
    """
    mask = (
        (summary_df["observation_total_events"]        >= min_obs_events) &
        (summary_df["observation_unique_contributors"] >= min_obs_contributors)
    )
    out = summary_df[mask].copy().reset_index(drop=True)
    print(
        f"[filter] {len(summary_df):,} total repos → "
        f"{len(out):,} pass inclusion filter "
        f"(≥{min_obs_events} events, ≥{min_obs_contributors} contributors)"
    )
    return out


# ── 6. Repo list recommendation ────────────────────────────────────────────────

def recommend_repo_list(
    candidates: pd.DataFrame,
    n_total: int = 20,
) -> pd.DataFrame:
    """
    Select a diverse set of ~n_total repositories from the candidate pool.

    Since star counts are not in GH Archive data, we use observation-period
    activity as a size proxy:
      - "large"  : top tier by obs event count  (target ~5 repos)
      - "medium" : mid tier                      (target ~10 repos)
      - "small"  : lower tier, still ≥ thresholds (target ~5 repos)

    Within each tier we also aim for a mix of:
      - repos likely to become inactive (future_to_obs_ratio < 0.5 → y=1 proxy)
      - repos likely to remain active   (future_to_obs_ratio ≥ 0.5 → y=0 proxy)

    This avoids selecting only high-activity or only active repos, which would
    produce severe class imbalance.

    Returns
    -------
    DataFrame — subset of candidates with an added 'tier' column.
    The number of returned repos may be slightly below n_total if the candidate
    pool is small.
    """
    df = candidates.copy()

    # compute activity-proxy percentiles to define tiers
    p66 = df["observation_total_events"].quantile(0.66)
    p33 = df["observation_total_events"].quantile(0.33)

    def assign_tier(x):
        if x >= p66:
            return "large"
        elif x >= p33:
            return "medium"
        else:
            return "small"

    df["tier"] = df["observation_total_events"].apply(assign_tier)

    # within each tier, aim for ~50/50 label balance using the ratio proxy
    df["proxy_y1"] = df["future_to_observation_volume_ratio"].fillna(0) < 0.5

    tier_targets = {"large": 5, "medium": 10, "small": 5}
    selected = []

    for tier, target in tier_targets.items():
        pool = df[df["tier"] == tier].copy()
        if pool.empty:
            continue

        half = max(1, target // 2)

        # pick from likely-risky repos first
        risky   = pool[pool["proxy_y1"]].nlargest(half, "observation_total_events")
        stable  = pool[~pool["proxy_y1"]].nlargest(target - len(risky), "observation_total_events")

        # if one class is short, fill from the other
        if len(risky) < half:
            stable = pool[~pool["proxy_y1"]].nlargest(target - len(risky), "observation_total_events")
        if len(stable) < (target - half):
            risky  = pool[pool["proxy_y1"]].nlargest(target - len(stable), "observation_total_events")

        selected.append(pd.concat([risky, stable]))

    result = pd.concat(selected).drop_duplicates(subset=["repo_name"])

    # trim or pad to n_total
    if len(result) > n_total:
        result = result.head(n_total)

    # sort for display
    result = result.sort_values(
        ["tier", "observation_total_events"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return result


# ── 7. Subwindow helper (needed later by feature engineering) ─────────────────

def assign_subwindow(
    df: pd.DataFrame,
    obs_start: str = OBS_START,
    n_subwindows: int = 6,
    subwindow_days: int = 30,
) -> pd.DataFrame:
    """
    Add a 'subwindow' column (0-indexed integer, 0 = first 30 days) to an
    observation-period event DataFrame.

    Events outside the n_subwindows * subwindow_days range are labelled -1
    and should be investigated but should not occur in a properly split dataset.
    """
    obs_lo = pd.Timestamp(obs_start, tz="UTC")
    delta  = (df["created_at"] - obs_lo).dt.days
    df     = df.copy()
    df["subwindow"] = (delta // subwindow_days).astype(int)
    # clamp: events exactly on the boundary of the last subwindow
    df.loc[df["subwindow"] >= n_subwindows, "subwindow"] = n_subwindows - 1
    df.loc[df["subwindow"] < 0,             "subwindow"] = -1
    return df
