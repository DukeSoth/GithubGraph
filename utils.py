"""
utils.py
--------
Responsibility: shared helpers used across multiple modules.

Covers:
  - I/O convenience wrappers (load/save parquet, load candidates CSV)
  - Path resolution relative to project root
  - Reproducibility (seed setting)
  - Lightweight pretty-printing

Nothing domain-specific lives here — keep this module dependency-free from
other src/ modules to avoid circular imports.

Public API
----------
  get_project_paths()             -> dict[str, str]
  load_parquet(path)              -> pd.DataFrame
  save_parquet(df, path)          -> None
  load_candidates(path)           -> pd.DataFrame
  set_random_seed(seed)           -> None
  print_separator(title, width)   -> None
  print_df_summary(df, name)      -> None
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd


# ── 1. Project path resolution ────────────────────────────────────────────────

def get_project_paths(
    project_root: str | None = None,
) -> dict[str, str]:
    """
    Return a dict of canonical project paths derived from the project root.

    If project_root is None, it is inferred as two directories above this file
    (i.e., collab_risk_proj/).

    Returns
    -------
    dict with keys:
      root        : absolute path to project root
      data        : data/
      processed   : data/processed/
      src         : src/
    """
    if project_root is None:
        # src/utils.py → two levels up → project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return {
        "root":      project_root,
        "data":      os.path.join(project_root, "data"),
        "processed": os.path.join(project_root, "data", "processed"),
        "src":       os.path.join(project_root, "src"),
    }


# ── 2. I/O helpers ────────────────────────────────────────────────────────────

def load_parquet(path: str) -> pd.DataFrame:
    """
    Load a parquet file and return a DataFrame.

    Falls back to CSV (same stem, .csv extension) if the parquet file is not
    found, so callers don't break if the parquet engine was unavailable during
    preprocessing.

    Parameters
    ----------
    path : path to the .parquet file (absolute or relative)

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError if neither the .parquet nor the fallback .csv exists.
    """
    if os.path.exists(path):
        return pd.read_parquet(path)

    # try CSV fallback
    csv_path = os.path.splitext(path)[0] + ".csv"
    if os.path.exists(csv_path):
        print(f"[load_parquet] .parquet not found; loading CSV fallback: {csv_path}")
        return pd.read_csv(csv_path)

    raise FileNotFoundError(f"Neither {path} nor {csv_path} found.")


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to parquet, creating parent directories as needed.

    Falls back to CSV if pyarrow / fastparquet is not installed.

    Parameters
    ----------
    df   : DataFrame to save
    path : destination .parquet path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except ImportError:
        csv_path = os.path.splitext(path)[0] + ".csv"
        df.to_csv(csv_path, index=False)
        print(f"[save_parquet] pyarrow not available; saved as CSV: {csv_path}")


def load_candidates(path: str | None = None) -> pd.DataFrame:
    """
    Load the candidate repo summary CSV produced by preprocessing.

    If path is None, looks for data/processed/repo_summary_candidates.csv
    relative to the project root.

    Parameters
    ----------
    path : explicit file path, or None to use the default location

    Returns
    -------
    pd.DataFrame with all columns from repo_summary_candidates.csv,
    including the 'y' label column if it was saved there.
    """
    if path is None:
        paths = get_project_paths()
        path  = os.path.join(paths["processed"], "repo_summary_candidates.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Candidates file not found: {path}")

    df = pd.read_csv(path)

    # coerce date columns if present
    for col in ["observation_first_event_date", "observation_last_event_date",
                "future_first_event_date",      "future_last_event_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    return df


# ── 3. Reproducibility ────────────────────────────────────────────────────────

def set_random_seed(seed: int = 42) -> None:
    """
    Set the random seed for Python, NumPy, and (if available) PyTorch.

    Call once at the start of a pipeline script for reproducible results.

    Parameters
    ----------
    seed : integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass  # torch is not required


# ── 4. Pretty-printing helpers ────────────────────────────────────────────────

def print_separator(title: str = "", width: int = 60) -> None:
    """Print a visual separator line with an optional centered title."""
    if title:
        pad   = max(0, width - len(title) - 4)
        left  = pad // 2
        right = pad - left
        print("\n" + "=" * left + f"  {title}  " + "=" * right)
    else:
        print("\n" + "=" * width)


def print_df_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Print a compact schema and null-count summary for a DataFrame.

    Mirrors the print_schema() helper in run_preprocessing.py, but lives here
    so any module can call it without importing the run script.

    Parameters
    ----------
    df   : DataFrame to describe
    name : display name for the header
    """
    print(f"\n── schema: {name} ({'×'.join(str(x) for x in df.shape)}) ──")
    for col in df.columns:
        dtype  = df[col].dtype
        n_null = int(df[col].isna().sum())
        marker = "  ← has nulls" if n_null > 0 else ""
        print(f"  {col:<45} {str(dtype):<15} nulls={n_null}{marker}")
