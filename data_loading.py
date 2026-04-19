"""
data_loading.py
---------------
Two layers of data access:

Layer 1 — Raw file I/O (used by preprocessing.py)
  list_files_in_range()   -- enumerate .json.gz files for a date range
  iter_events()           -- generator over qualifying events in a single file
  load_events_for_repos() -- targeted DataFrame load for a repo set

Layer 2 — Processed artifact loaders (used by all downstream modules)
  load_candidate_repos()  -- repo_summary_candidates.csv → DataFrame
  load_obs_events()       -- obs_events.parquet → filtered DataFrame
  load_future_events()    -- future_events.parquet → filtered DataFrame
  load_datasets()         -- convenience: returns (obs_df, future_df) in one call
"""

import gzip
import json
import os
from datetime import date, timedelta
from typing import Generator, Optional

import pandas as pd

# ── project-level constants ────────────────────────────────────────────────────

ALLOWED_TYPES: set = {
    "PushEvent",
    "PullRequestEvent",
    "IssuesEvent",
    "ReleaseEvent",
    # PullRequestReviewEvent intentionally omitted: not present in 2015 GH Archive
}

OBS_START  = "2015-01-01"
OBS_END    = "2015-06-30"
FUTURE_START = "2015-07-01"
FUTURE_END   = "2015-09-30"

# ── file enumeration ───────────────────────────────────────────────────────────

def list_files_in_range(
    data_dir: str,
    start_date: str,
    end_date: str,
) -> list:
    """
    Return a sorted list of .json.gz paths whose calendar date falls within
    [start_date, end_date] (inclusive, ISO format strings).

    GH Archive filenames are YYYY-MM-DD-H.json.gz where H is 0-23 (not
    zero-padded).  Files for missing hours or days are silently skipped.
    """
    start = date.fromisoformat(start_date)
    end   = date.fromisoformat(end_date)

    paths = []
    current = start
    while current <= end:
        day_str = current.strftime("%Y-%m-%d")
        for hour in range(24):
            fname = f"{day_str}-{hour}.json.gz"
            full_path = os.path.join(data_dir, fname)
            if os.path.exists(full_path):
                paths.append(full_path)
        current += timedelta(days=1)

    return paths


# ── single-file event generator ────────────────────────────────────────────────

def iter_events(
    path: str,
    repo_filter: Optional[set] = None,
) -> Generator[dict, None, None]:
    """
    Yield one minimal dict per qualifying event from a single .json.gz file.

    Qualifying means:
      - event type is in ALLOWED_TYPES
      - actor.login is present and non-empty
      - repo.name is present and non-empty
      - if repo_filter is given, repo.name must be in that set

    Yielded dict keys: id, actor_login, repo_name, type, created_at (raw str).
    Datetime parsing is left to preprocessing.clean_events() so this generator
    stays fast and allocation-light.
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # type filter (cheapest check first)
                event_type = obj.get("type")
                if event_type not in ALLOWED_TYPES:
                    continue

                # extract nested fields
                actor = obj.get("actor")
                repo  = obj.get("repo")
                if not actor or not repo:
                    continue

                login    = actor.get("login")
                reponame = repo.get("name")
                if not login or not reponame:
                    continue

                # optional repo filter
                if repo_filter is not None and reponame not in repo_filter:
                    continue

                yield {
                    "id":          obj.get("id", ""),
                    "actor_login": login,
                    "repo_name":   reponame,
                    "type":        event_type,
                    "created_at":  obj.get("created_at", ""),
                }
    except (OSError, EOFError):
        # skip corrupt or truncated files
        pass


# ── targeted DataFrame load (second pass) ─────────────────────────────────────

def load_events_for_repos(
    data_dir: str,
    start_date: str,
    end_date: str,
    repo_names: set,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load all qualifying events for the given repos into a DataFrame.

    Parameters
    ----------
    data_dir   : directory containing the .json.gz files
    start_date : ISO date string, inclusive
    end_date   : ISO date string, inclusive
    repo_names : set of repo.name strings to keep (e.g. {"torvalds/linux"})
    verbose    : print progress every 500 files if True

    Returns
    -------
    DataFrame with columns:
        id, actor_login, repo_name, type, created_at (datetime64[ns, UTC])

    Notes
    -----
    - created_at is parsed to UTC-aware datetime.
    - id is kept as string (GH Archive ids are large integers stored as strings).
    - Deduplication by id is NOT done here; call preprocessing.clean_events()
      on the result.
    """
    files = list_files_in_range(data_dir, start_date, end_date)
    if verbose:
        print(f"[load] scanning {len(files)} files for {len(repo_names)} repos ...")

    records = []
    for i, path in enumerate(files):
        if verbose and i > 0 and i % 500 == 0:
            print(f"  {i}/{len(files)} files processed, {len(records):,} events so far")
        for rec in iter_events(path, repo_filter=repo_names):
            records.append(rec)

    if not records:
        return pd.DataFrame(
            columns=["id", "actor_login", "repo_name", "type", "created_at"]
        )

    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    if verbose:
        print(f"[load] done. {len(df):,} raw events loaded.")

    return df


# ── Layer 2: processed artifact loaders ───────────────────────────────────────
#
# These functions read from data/processed/, the outputs written by
# run_preprocessing.py.  They are the entry point for all modules downstream
# of preprocessing (feature_engineering, labeling, modeling, etc.).

def _resolve_processed_dir(processed_dir: Optional[str]) -> str:
    """
    Return the absolute path to data/processed/.

    If processed_dir is None, infer it as <project_root>/data/processed/
    where project_root is two directories above this file.
    """
    if processed_dir is not None:
        return processed_dir
    src_dir      = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    return os.path.join(project_root, "data", "processed")


def _ensure_utc_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the created_at column is a UTC-aware datetime64.

    Parquet files preserve the timezone; CSV files lose it.  This function
    handles both cases and coerces unparseable values to NaT.

    Parameters
    ----------
    df : DataFrame that must contain a 'created_at' column.

    Returns
    -------
    Same DataFrame with created_at cast to datetime64[ns, UTC].
    """
    col = df["created_at"]
    if pd.api.types.is_datetime64_any_dtype(col):
        # already datetime — make sure it is timezone-aware
        if col.dt.tz is None:
            df = df.copy()
            df["created_at"] = col.dt.tz_localize("UTC")
    else:
        df = df.copy()
        df["created_at"] = pd.to_datetime(col, utc=True, errors="coerce")
    return df


def _validate_events_df(df: pd.DataFrame, name: str) -> None:
    """
    Run basic sanity checks on a loaded event DataFrame and print a warning
    for each issue found.  Never raises — callers decide whether to abort.

    Checks
    ------
    - Required columns are present.
    - No null values in id, actor_login, repo_name, created_at, type.
    - created_at is UTC-aware datetime.
    - event types are within ALLOWED_TYPES.
    - created_at range matches the expected window (obs or future).
    """
    required = {"id", "actor_login", "repo_name", "type", "created_at"}
    missing  = required - set(df.columns)
    if missing:
        print(f"[validate:{name}] MISSING columns: {missing}")
        return  # can't check further

    for col in required:
        n_null = int(df[col].isna().sum())
        if n_null:
            print(f"[validate:{name}] {n_null} null values in '{col}'")

    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        print(f"[validate:{name}] 'created_at' is not datetime dtype")
    elif df["created_at"].dt.tz is None:
        print(f"[validate:{name}] 'created_at' is timezone-naive (expected UTC)")

    unknown_types = set(df["type"].dropna().unique()) - ALLOWED_TYPES
    if unknown_types:
        print(f"[validate:{name}] unexpected event types: {unknown_types}")


def load_candidate_repos(
    processed_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the saved candidate repository summary from repo_summary_candidates.csv.

    This file is produced by run_preprocessing.py (Step 3) and contains one
    row per selected repository with observation/future statistics and the
    binary risk label 'y'.

    Parameters
    ----------
    processed_dir : path to data/processed/, or None to auto-resolve.

    Returns
    -------
    DataFrame with columns including:
        repo_name, tier, observation_total_events,
        observation_unique_contributors, future_total_events,
        future_to_observation_volume_ratio,
        has_90_day_future_inactivity_flag, y

    Raises
    ------
    FileNotFoundError if the CSV does not exist.
    """
    pdir = _resolve_processed_dir(processed_dir)
    path = os.path.join(pdir, "repo_summary_candidates.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Candidate repo file not found: {path}\n"
            "Run src/run_preprocessing.py first to generate it."
        )

    df = pd.read_csv(path)

    # coerce date-like columns that round-trip as strings through CSV
    date_cols = [
        "observation_first_event_date", "observation_last_event_date",
        "future_first_event_date",      "future_last_event_date",
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    # basic validation
    if "repo_name" not in df.columns:
        print("[validate:candidates] MISSING 'repo_name' column")
    if "y" not in df.columns:
        print("[validate:candidates] 'y' label column absent; "
              "run run_preprocessing.py to regenerate")

    print(f"[load_candidate_repos] {len(df)} repos loaded from {path}")
    return df


def _load_event_parquet(
    stem: str,
    processed_dir: str,
    repo_names: Optional[set],
    validate: bool,
) -> pd.DataFrame:
    """
    Internal helper: load <stem>.parquet (with CSV fallback), ensure UTC
    datetime, optionally filter to repo_names, optionally validate.
    """
    parquet_path = os.path.join(processed_dir, f"{stem}.parquet")
    csv_path     = os.path.join(processed_dir, f"{stem}.csv")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"[load] .parquet not found; reading CSV fallback: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Neither {parquet_path} nor {csv_path} found.\n"
            "Run src/run_preprocessing.py first."
        )

    df = _ensure_utc_datetime(df)

    if repo_names is not None:
        before = len(df)
        df = df[df["repo_name"].isin(repo_names)].reset_index(drop=True)
        print(
            f"[load:{stem}] filtered {before:,} → {len(df):,} events "
            f"({len(repo_names)} repos requested)"
        )

    if validate:
        _validate_events_df(df, stem)

    return df


def load_obs_events(
    processed_dir: Optional[str] = None,
    repo_names: Optional[set] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load the cleaned observation-window events (2015-01-01 to 2015-06-30).

    Reads obs_events.parquet from data/processed/ (CSV fallback if parquet
    is absent).  Optionally restricts the result to a specific set of repos.

    Parameters
    ----------
    processed_dir : path to data/processed/, or None to auto-resolve.
    repo_names    : if given, keep only rows whose repo_name is in this set.
    validate      : if True, run basic sanity checks and print warnings.

    Returns
    -------
    DataFrame with columns:
        id, actor_login, repo_name, type,
        created_at (datetime64[ns, UTC]), subwindow (int)

    Raises
    ------
    FileNotFoundError if neither obs_events.parquet nor obs_events.csv exists.
    """
    pdir = _resolve_processed_dir(processed_dir)
    df   = _load_event_parquet("obs_events", pdir, repo_names, validate)

    print(
        f"[load_obs_events] {len(df):,} events | "
        f"repos={df['repo_name'].nunique()} | "
        f"date range: {df['created_at'].min().date()} – {df['created_at'].max().date()}"
    )
    return df


def load_future_events(
    processed_dir: Optional[str] = None,
    repo_names: Optional[set] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Load the cleaned future-window events (2015-07-01 to 2015-09-30).

    Reads future_events.parquet from data/processed/ (CSV fallback if absent).
    Optionally restricts the result to a specific set of repos.

    Parameters
    ----------
    processed_dir : path to data/processed/, or None to auto-resolve.
    repo_names    : if given, keep only rows whose repo_name is in this set.
    validate      : if True, run basic sanity checks and print warnings.

    Returns
    -------
    DataFrame with columns:
        id, actor_login, repo_name, type,
        created_at (datetime64[ns, UTC])

    Raises
    ------
    FileNotFoundError if neither future_events.parquet nor future_events.csv exists.
    """
    pdir = _resolve_processed_dir(processed_dir)
    df   = _load_event_parquet("future_events", pdir, repo_names, validate)

    print(
        f"[load_future_events] {len(df):,} events | "
        f"repos={df['repo_name'].nunique()} | "
        f"date range: {df['created_at'].min().date()} – {df['created_at'].max().date()}"
    )
    return df


def load_datasets(
    processed_dir: Optional[str] = None,
    repo_names: Optional[set] = None,
    validate: bool = True,
) -> tuple:
    """
    Load both processed event windows in a single call.

    This is the primary entry point for downstream modules.  It combines
    load_obs_events() and load_future_events() and ensures both DataFrames
    cover the same set of repositories.

    Parameters
    ----------
    processed_dir : path to data/processed/, or None to auto-resolve.
    repo_names    : if given, restrict both DataFrames to this repo set.
                    If None, uses all repos found in obs_events; any future
                    events for repos not in obs_events are dropped.
    validate      : if True, run sanity checks on both DataFrames.

    Returns
    -------
    (obs_df, future_df) — two DataFrames with the same column schema.
        obs_df    : events from 2015-01-01 to 2015-06-30
        future_df : events from 2015-07-01 to 2015-09-30

    Raises
    ------
    FileNotFoundError  if either parquet file is missing.
    ValueError         if obs_df is empty after loading.
    """
    pdir    = _resolve_processed_dir(processed_dir)
    obs_df  = load_obs_events(pdir, repo_names=repo_names, validate=validate)
    fut_df  = load_future_events(pdir, repo_names=repo_names, validate=validate)

    # if no explicit filter was given, align future to the obs repo universe
    if repo_names is None:
        obs_repos = set(obs_df["repo_name"].unique())
        if not obs_repos:
            raise ValueError(
                "obs_df is empty — nothing to align future events to. "
                "Check that run_preprocessing.py completed successfully."
            )
        before = len(fut_df)
        fut_df = fut_df[fut_df["repo_name"].isin(obs_repos)].reset_index(drop=True)
        dropped = before - len(fut_df)
        if dropped:
            print(
                f"[load_datasets] dropped {dropped:,} future events for repos "
                "absent from obs window"
            )

    # cross-window repo consistency check
    obs_repos = set(obs_df["repo_name"].unique())
    fut_repos = set(fut_df["repo_name"].unique())
    obs_only  = obs_repos - fut_repos
    fut_only  = fut_repos - obs_repos
    if obs_only:
        print(
            f"[load_datasets] {len(obs_only)} repos have obs events but no "
            f"future events: {sorted(obs_only)}"
        )
    if fut_only:
        print(
            f"[load_datasets] {len(fut_only)} repos appear in future but not "
            f"in obs (unexpected): {sorted(fut_only)}"
        )

    print(
        f"\n[load_datasets] ready — obs: {len(obs_df):,} events across "
        f"{len(obs_repos)} repos | future: {len(fut_df):,} events across "
        f"{len(fut_repos)} repos"
    )
    return obs_df, fut_df


# ── module entry point / smoke test ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # allow running as  python src/data_loading.py  from project root
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("  data_loading.py — processed artifact smoke test")
    print("=" * 60)

    # 1. candidate repos
    candidates = load_candidate_repos()
    repo_set   = set(candidates["repo_name"].tolist())
    print(f"\nCandidate repos ({len(repo_set)}):")
    for r in sorted(repo_set):
        print(f"  {r}")

    # 2. observation events
    print()
    obs_df = load_obs_events(repo_names=repo_set)

    # 3. future events
    print()
    fut_df = load_future_events(repo_names=repo_set)

    # 4. combined load (alignment check)
    print()
    obs2, fut2 = load_datasets(repo_names=repo_set)

    # 5. per-repo event counts
    print("\n--- per-repo event counts ---")
    summary = (
        obs2.groupby("repo_name")
        .agg(obs_events=("id", "count"), contributors=("actor_login", "nunique"))
        .join(
            fut2.groupby("repo_name")
            .agg(fut_events=("id", "count")),
            how="left",
        )
        .fillna({"fut_events": 0})
        .astype({"fut_events": int})
        .sort_values("obs_events", ascending=False)
    )
    print(summary.to_string())

    print("\n[done]")
