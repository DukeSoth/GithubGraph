"""
feature_engineering.py
-----------------------
Compute repository-level features from observation-window events and
repository-specific contribution graphs.

Features produced (one row per repository)
-------------------------------------------
  Activity / contributor features (from obs_df)
  -----------------------------------------------
  repo_name                  str    repository identifier
  num_contributors           int    unique developers active in obs window
  total_contributions        int    total qualifying events in obs window
  top1_share                 float  fraction of events by the single most active contributor
  top3_share                 float  fraction of events by the top-3 contributors combined
  gini_coefficient           float  Gini of per-contributor event counts
                                    0 = perfectly equal, 1 = one contributor does everything
  active_contributor_slope   float  OLS slope of unique-contributor counts across the 6
                                    monthly subwindows (positive = growing contributor base)
  contribution_volume_slope  float  OLS slope of total event counts across the 6 monthly
                                    subwindows (positive = increasing activity over time)

  Graph resilience features (from contributor projection graph)
  --------------------------------------------------------------
  degree_centralization            float  Freeman degree centralization of the co-contributor
                                          graph; 0 = uniform degree, 1 = perfect star topology
  kcore_max                        int    degeneracy (max k such that a non-empty k-core exists);
                                          higher = denser inner collaboration core
  lcc_ratio_after_remove_top1      float  fraction of remaining nodes in the largest connected
                                          component after removing the top-1 contributor
  lcc_ratio_after_remove_top2      float  same after removing the top-2 contributors
  density_before_removal           float  edge density of the co-contributor graph
  density_after_remove_top1        float  edge density after removing the top-1 contributor
  density_change_after_remove_top1 float  density_after_remove_top1 − density_before_removal
                                          (negative = top contributor was a structural hub)

  "Top contributor" definition
  ----------------------------
  Consistently ranked by total contribution weight (event count) during the
  observation period.  This is the same ranking used for top1_share / top3_share,
  and matches the edge weight in the bipartite contribution graph.

Subwindow convention
--------------------
  sw0 = Jan 2015, sw1 = Feb, ..., sw5 = Jun 2015 (30-day buckets).
  The 'subwindow' column is expected to already be present on obs_df
  (produced by preprocessing.assign_subwindow).

Public API
----------
  extract_repo_features(obs_df, repo_name, repo_graph_dict)      -> dict
  extract_graph_resilience_features(cg, actor_counts)            -> dict
  build_feature_matrix(obs_df, repo_names, graph_data)           -> pd.DataFrame
  save_features(df, output_dir)                                   -> str
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

# support  python src/feature_engineering.py  from project root
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data_loading import OBS_START, load_datasets, load_candidate_repos
from graph_building import build_all_repo_graphs, build_contributor_graph
from preprocessing import assign_subwindow

N_SUBWINDOWS: int = 6  # Jan–Jun 2015, one 30-day bucket per month


# ── helpers ────────────────────────────────────────────────────────────────────

def _gini(values: list[float]) -> float:
    """
    Gini coefficient of a list of non-negative values.

    Returns 0.0 when the list is empty or all zeros (treat as perfectly equal —
    a conservative default for repos with a single contributor).

    Formula: sorted cumulative-sum approach, O(n log n).
    """
    if not values or sum(values) == 0:
        return 0.0
    n  = len(values)
    sv = sorted(values)
    s  = sum(sv)
    weighted_sum = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sv))
    return weighted_sum / (n * s)


def _ols_slope(y: list[float]) -> float:
    """
    Least-squares slope of y over x = [0, 1, ..., n-1].

    Using the closed-form solution avoids a numpy.polyfit call and is clear
    about what is being computed:

        slope = (n * sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)

    Returns 0.0 for a single-element series (slope is undefined).
    """
    n = len(y)
    if n < 2:
        return 0.0
    x    = list(range(n))
    sx   = sum(x)
    sy   = sum(y)
    sxy  = sum(xi * yi for xi, yi in zip(x, y))
    sxx  = sum(xi * xi for xi in x)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    return (n * sxy - sx * sy) / denom


def _per_subwindow_counts(
    repo_df: pd.DataFrame,
) -> tuple[list[int], list[int]]:
    """
    Return (events_per_sw, unique_contributors_per_sw) for subwindows 0–5.

    Subwindows with no events contribute 0 to both lists, which correctly
    anchors the slope regression — a repo that went quiet in May is penalised
    by the zero, not by omission.

    Parameters
    ----------
    repo_df : obs_df filtered to a single repository, with 'subwindow' column.

    Returns
    -------
    events_per_sw       : list of 6 ints
    contributors_per_sw : list of 6 ints
    """
    events_per_sw       = []
    contributors_per_sw = []

    for sw in range(N_SUBWINDOWS):
        sw_mask = repo_df["subwindow"] == sw
        events_per_sw.append(int(sw_mask.sum()))
        contributors_per_sw.append(int(repo_df.loc[sw_mask, "actor_login"].nunique()))

    return events_per_sw, contributors_per_sw


def _lcc_ratio(G: "nx.Graph") -> float:  # type: ignore[name-defined]
    """
    Fraction of nodes in the largest connected component of G.

    Returns 0.0 for an empty graph.  A result of 1.0 means all remaining
    nodes are in a single connected component (no fragmentation).
    """
    import networkx as nx
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    largest = max(len(c) for c in nx.connected_components(G))
    return largest / n


def _degree_centralization(G: "nx.Graph") -> float:  # type: ignore[name-defined]
    """
    Freeman degree centralization for an undirected graph.

    Formula: Σᵢ (d_max − dᵢ) / ((n−1)(n−2))

    The denominator is the theoretical maximum of the numerator, achieved by
    a perfect star graph (one hub connected to all others, no other edges).

    Returns 0.0 when n ≤ 2 (denominator is zero; centralization undefined).
    """
    n = G.number_of_nodes()
    if n <= 2:
        return 0.0
    degrees = [d for _, d in G.degree()]
    d_max   = max(degrees)
    denom   = (n - 1) * (n - 2)
    return sum(d_max - d for d in degrees) / denom


# ── graph resilience features ──────────────────────────────────────────────────

def extract_graph_resilience_features(
    cg: "nx.Graph",  # type: ignore[name-defined]
    actor_counts: pd.Series,
) -> dict:
    """
    Compute structural robustness features from the contributor projection graph.

    All features that involve node removal operate on a copy of cg; the input
    graph is never mutated.

    "Top contributor" ranking
    -------------------------
    Determined by actor_counts (a pd.Series mapping actor_login → total events,
    sorted descending).  This is the same ordering used for top1_share /
    top3_share, so all features share a single, consistent definition of
    contributor importance.

    Parameters
    ----------
    cg           : contributor projection graph (nx.Graph, unipartite actors).
                   Built by graph_building.build_contributor_graph().
    actor_counts : pd.Series from repo_df["actor_login"].value_counts().
                   Index = actor names, values = event counts, descending order.

    Returns
    -------
    dict with keys:
      degree_centralization, kcore_max,
      lcc_ratio_after_remove_top1, lcc_ratio_after_remove_top2,
      density_before_removal, density_after_remove_top1,
      density_change_after_remove_top1
    """
    import networkx as nx

    _EMPTY = {
        "degree_centralization":            0.0,
        "kcore_max":                        0,
        "lcc_ratio_after_remove_top1":      0.0,
        "lcc_ratio_after_remove_top2":      0.0,
        "density_before_removal":           0.0,
        "density_after_remove_top1":        0.0,
        "density_change_after_remove_top1": 0.0,
    }

    if cg.number_of_nodes() == 0:
        return _EMPTY

    # ── degree centralization ─────────────────────────────────────────────────
    deg_central = _degree_centralization(cg)

    # ── k-core max (degeneracy) ───────────────────────────────────────────────
    # nx.core_number assigns each node its shell index; the max is the
    # degeneracy.  Isolated nodes get core number 0.
    core_nums = nx.core_number(cg)
    kcore_max = int(max(core_nums.values())) if core_nums else 0

    # ── density before any removal ────────────────────────────────────────────
    density_before = nx.density(cg)

    # ── identify top contributors present in the graph ────────────────────────
    # actor_counts is already sorted descending by value_counts(); re-sort
    # defensively to guarantee correct ordering regardless of how it was built.
    ranked = actor_counts.sort_values(ascending=False).index.tolist()
    graph_nodes   = set(cg.nodes())
    ranked_in_cg  = [a for a in ranked if a in graph_nodes]

    top1 = ranked_in_cg[0] if len(ranked_in_cg) >= 1 else None
    top2 = ranked_in_cg[1] if len(ranked_in_cg) >= 2 else None

    # ── remove top-1 ──────────────────────────────────────────────────────────
    if top1 is not None:
        cg_minus1 = cg.copy()
        cg_minus1.remove_node(top1)
        lcc_top1     = _lcc_ratio(cg_minus1)
        # nx.density returns 0.0 for graphs with 0 or 1 node (no edges possible)
        density_top1 = nx.density(cg_minus1)
    else:
        # graph has no nodes in actor_counts (shouldn't happen in practice)
        lcc_top1     = _lcc_ratio(cg)
        density_top1 = density_before

    # ── remove top-1 and top-2 ────────────────────────────────────────────────
    if top1 is not None and top2 is not None:
        cg_minus2 = cg.copy()
        cg_minus2.remove_nodes_from([top1, top2])
        lcc_top2 = _lcc_ratio(cg_minus2)
    elif top1 is not None:
        # only one contributor existed; removing "top-2" = same as top-1
        lcc_top2 = lcc_top1
    else:
        lcc_top2 = _lcc_ratio(cg)

    density_change = density_top1 - density_before

    return {
        "degree_centralization":            round(deg_central,    4),
        "kcore_max":                        kcore_max,
        "lcc_ratio_after_remove_top1":      round(lcc_top1,       4),
        "lcc_ratio_after_remove_top2":      round(lcc_top2,       4),
        "density_before_removal":           round(density_before, 4),
        "density_after_remove_top1":        round(density_top1,   4),
        "density_change_after_remove_top1": round(density_change, 4),
    }


# ── per-repository feature extraction ─────────────────────────────────────────

def extract_repo_features(
    obs_df: pd.DataFrame,
    repo_name: str,
    repo_graph_dict: Optional[dict] = None,
) -> dict:
    """
    Extract the 7 repository-level features for a single repository.

    Data sources
    ------------
    - obs_df is the primary source for contributor counts, contribution
      volumes, shares, and subwindow time-series.
    - repo_graph_dict (optional) provides pre-computed graph metrics.  When
      present, gini_coefficient is taken from the bipartite graph's
      weight_gini (identical in value to what we would compute from obs_df,
      but consistent with the graph representation used elsewhere).

    Parameters
    ----------
    obs_df          : cleaned observation-window DataFrame with 'subwindow'.
    repo_name       : the repository to featurise.
    repo_graph_dict : single-repo entry from build_all_repo_graphs(), or None.

    Returns
    -------
    dict with keys: repo_name, num_contributors, total_contributions,
    top1_share, top3_share, gini_coefficient,
    active_contributor_slope, contribution_volume_slope.
    """
    repo_df = obs_df[obs_df["repo_name"] == repo_name]

    # ── guard: empty repo ─────────────────────────────────────────────────────
    if repo_df.empty:
        import networkx as nx
        return {
            "repo_name":                 repo_name,
            "num_contributors":          0,
            "total_contributions":       0,
            "top1_share":                0.0,
            "top3_share":                0.0,
            "gini_coefficient":          0.0,
            "active_contributor_slope":  0.0,
            "contribution_volume_slope": 0.0,
            **extract_graph_resilience_features(nx.Graph(), pd.Series(dtype=float)),
        }

    # ensure subwindow column — compute on-the-fly if obs_df was loaded
    # without it (should not happen after run_preprocessing, but defensive)
    if "subwindow" not in repo_df.columns:
        repo_df = assign_subwindow(
            repo_df.copy(), obs_start=OBS_START, n_subwindows=N_SUBWINDOWS
        )

    # ── contributor counts and event volume ───────────────────────────────────
    actor_counts = repo_df["actor_login"].value_counts()   # sorted descending
    total        = len(repo_df)
    n_contrib    = len(actor_counts)

    # ── contribution shares ───────────────────────────────────────────────────
    top1_share = float(actor_counts.iloc[0]) / total if n_contrib >= 1 else 0.0
    top3_share = float(actor_counts.iloc[:3].sum()) / total if n_contrib >= 1 else 0.0

    # ── Gini coefficient ──────────────────────────────────────────────────────
    # Prefer the graph's pre-computed value for consistency; fall back to
    # computing inline from the same raw counts.
    if repo_graph_dict is not None:
        gini = float(repo_graph_dict["bipartite_metrics"].get("weight_gini", 0.0))
    else:
        gini = _gini(actor_counts.tolist())

    # ── subwindow time-series features ───────────────────────────────────────
    events_per_sw, contributors_per_sw = _per_subwindow_counts(repo_df)

    vol_slope     = _ols_slope([float(c) for c in events_per_sw])
    contrib_slope = _ols_slope([float(c) for c in contributors_per_sw])

    # ── graph resilience features ─────────────────────────────────────────────
    # Use the pre-built contributor graph when available (graph_data was passed
    # to build_feature_matrix).  Build on the fly as a fallback so this
    # function remains usable in isolation.
    if repo_graph_dict is not None:
        cg = repo_graph_dict["collaboration_graph"]
    else:
        cg = build_contributor_graph(obs_df, repo_name)

    # actor_counts is sorted descending by value_counts() — pass directly so
    # extract_graph_resilience_features uses the same "top contributor" ranking
    # as top1_share / top3_share above.
    resilience = extract_graph_resilience_features(cg, actor_counts)

    return {
        "repo_name":                 repo_name,
        "num_contributors":          n_contrib,
        "total_contributions":       total,
        "top1_share":                round(top1_share,   4),
        "top3_share":                round(top3_share,   4),
        "gini_coefficient":          round(gini,         4),
        "active_contributor_slope":  round(contrib_slope, 4),
        "contribution_volume_slope": round(vol_slope,    4),
        **resilience,
    }


# ── assemble the feature matrix ────────────────────────────────────────────────

def build_feature_matrix(
    obs_df: pd.DataFrame,
    repo_names: list[str],
    graph_data: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build one feature row per repository and return the result as a DataFrame.

    Parameters
    ----------
    obs_df     : cleaned observation-window event DataFrame (with 'subwindow').
    repo_names : ordered list of repositories to include.
    graph_data : output of graph_building.build_all_repo_graphs(), or None.
                 When provided, gini_coefficient is sourced from the graph.

    Returns
    -------
    pd.DataFrame — one row per repo, repo_name as a plain column (not index)
    so the table is self-contained for saving and merging.

    Column order matches the feature documentation at the top of this module.
    """
    rows = []
    for repo in repo_names:
        graph_dict = graph_data.get(repo) if graph_data else None
        rows.append(extract_repo_features(obs_df, repo, graph_dict))

    column_order = [
        # identity
        "repo_name",
        # activity / contributor
        "num_contributors",
        "total_contributions",
        "top1_share",
        "top3_share",
        "gini_coefficient",
        "active_contributor_slope",
        "contribution_volume_slope",
        # graph resilience
        "degree_centralization",
        "kcore_max",
        "lcc_ratio_after_remove_top1",
        "lcc_ratio_after_remove_top2",
        "density_before_removal",
        "density_after_remove_top1",
        "density_change_after_remove_top1",
    ]
    return pd.DataFrame(rows)[column_order]


# ── save helper ────────────────────────────────────────────────────────────────

def save_features(
    df: pd.DataFrame,
    output_dir: str,
    filename: str = "features.csv",
) -> str:
    """
    Save the feature matrix to output_dir/<filename> as CSV.

    Creates output_dir if it does not exist.  Returns the path written.

    Parameters
    ----------
    df         : feature DataFrame from build_feature_matrix().
    output_dir : destination directory path.
    filename   : file name (default: 'features.csv').

    Returns
    -------
    str — absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"[save_features] wrote {len(df)} rows to {path}")
    return path


# ── main / smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  feature_engineering.py — build feature matrix")
    print("=" * 65)

    # ── load data ─────────────────────────────────────────────────────────────
    candidates = load_candidate_repos()
    repo_names = candidates["repo_name"].tolist()

    obs_df, _ = load_datasets(repo_names=set(repo_names))
    print(f"\nobs_df  : {len(obs_df):,} events  |  {obs_df['repo_name'].nunique()} repos")

    # ── build graphs ──────────────────────────────────────────────────────────
    print("\nBuilding repository graphs …")
    graph_data = build_all_repo_graphs(obs_df, repo_names)
    print(f"Built graphs for {len(graph_data)} repos.")

    # ── extract features ──────────────────────────────────────────────────────
    print("\nExtracting features …")
    features = build_feature_matrix(obs_df, repo_names, graph_data)

    # ── display: activity features ────────────────────────────────────────────
    activity_cols = [
        "repo_name", "num_contributors", "total_contributions",
        "top1_share", "top3_share", "gini_coefficient",
        "active_contributor_slope", "contribution_volume_slope",
    ]
    resilience_cols = [
        "repo_name",
        "degree_centralization", "kcore_max",
        "lcc_ratio_after_remove_top1", "lcc_ratio_after_remove_top2",
        "density_before_removal", "density_after_remove_top1",
        "density_change_after_remove_top1",
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width",       130)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\nFeature matrix  ({features.shape[0]} rows × {features.shape[1]} cols)")
    print("\n-- Activity / contributor features --")
    print(features[activity_cols].to_string(index=False))
    print("\n-- Graph resilience features --")
    print(features[resilience_cols].to_string(index=False))

    # ── worked example: viljamis/responsive-nav.js ────────────────────────────
    example = "viljamis/responsive-nav.js"
    if example in set(repo_names):
        import networkx as nx

        ex_df  = obs_df[obs_df["repo_name"] == example]
        ex_cg  = graph_data[example]["collaboration_graph"]
        ex_ac  = ex_df["actor_login"].value_counts()

        print(f"\n-- Resilience walkthrough: {example} --")
        print(f"  Contributors in obs window : {ex_ac.sum()} events, {len(ex_ac)} actors")
        print(f"  Top-1 by events : {ex_ac.index[0]}  ({ex_ac.iloc[0]} events)")
        if len(ex_ac) > 1:
            print(f"  Top-2 by events : {ex_ac.index[1]}  ({ex_ac.iloc[1]} events)")

        # show what happens to LCC after each removal
        ranked_in_cg = [a for a in ex_ac.index if a in set(ex_cg.nodes())]
        print(f"\n  Co-contributor graph: {ex_cg.number_of_nodes()} nodes, "
              f"{ex_cg.number_of_edges()} edges, "
              f"density={nx.density(ex_cg):.4f}")
        print(f"  k-core numbers: {dict(sorted(nx.core_number(ex_cg).items(), key=lambda x: -x[1])[:5])} ...")

        if ranked_in_cg:
            top1 = ranked_in_cg[0]
            cg1  = ex_cg.copy(); cg1.remove_node(top1)
            print(f"\n  After removing '{top1}' (top-1):")
            print(f"    remaining nodes : {cg1.number_of_nodes()}")
            print(f"    remaining edges : {cg1.number_of_edges()}")
            print(f"    density         : {nx.density(cg1):.4f}")
            print(f"    lcc_ratio       : {_lcc_ratio(cg1):.4f}")

        events_sw, contribs_sw = _per_subwindow_counts(ex_df)
        print(f"\n  Subwindow breakdown:")
        print(f"  {'sw':<4}  {'events':>8}  {'contributors':>14}")
        for sw, (ev, co) in enumerate(zip(events_sw, contribs_sw)):
            print(f"    {sw}    {ev:>8}    {co:>14}")

    # ── save ──────────────────────────────────────────────────────────────────
    project_root = os.path.dirname(_SRC)
    output_dir   = os.path.join(project_root, "outputs")

    if os.path.isdir(output_dir):
        save_features(features, output_dir)
    else:
        print(f"\n[save] 'outputs/' directory not found — skipping save.")
        print(f"       Create it with: mkdir {output_dir}")
        print(f"       Then re-run this script to persist the feature table.")

    print("\n[done]")
