"""
graph_building.py
-----------------
Responsibility: construct collaboration graphs from observation-period events.

Two graph types are supported:

  Global bipartite contribution graph
    One graph covering ALL candidate repositories.
    Left partition  : developer nodes (actor_login)
    Right partition : repository nodes (repo_name)
    Edge            : unique (developer, repo) pair
    Edge weight     : total number of contribution events in the obs window

  Repository-specific contributor projection graph  (co-contributor graph)
    One graph per repository.
    Nodes  : developers who contributed to this repo in the obs window
    Edge   : two developers share an edge if they were BOTH active during
             at least one of the same monthly 30-day subwindows (sw0–sw5)
    Edge weight : number of subwindows in which both were active together

    Assumption: "active in a subwindow" means ≥1 qualifying event in that
    30-day period.  The 'subwindow' column (produced by
    preprocessing.assign_subwindow) must be present on obs_df, or it is
    computed on-the-fly using the default OBS_START constant.

Public API
----------
  build_global_bipartite_graph(obs_df)               -> nx.Graph
  build_bipartite_graph(obs_df, repo_name)            -> nx.Graph
  build_contributor_graph(obs_df, repo_name)          -> nx.Graph
  compute_graph_metrics(G, graph_type)                -> dict
  build_all_repo_graphs(obs_df, repo_names)           -> dict[str, dict]
"""

from __future__ import annotations

import sys
import os
from itertools import combinations
from typing import Optional

import networkx as nx
import pandas as pd

# ── import project helpers ─────────────────────────────────────────────────────
# Support both  python src/graph_building.py  and  import graph_building
_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from data_loading import OBS_START
from preprocessing import assign_subwindow

# Number of 30-day subwindows in the observation period (Jan–Jun 2015)
N_SUBWINDOWS: int = 6


# ── 1. Global bipartite contribution graph ────────────────────────────────────

def build_global_bipartite_graph(obs_df: pd.DataFrame) -> nx.Graph:
    """
    Build a single bipartite graph that covers ALL candidate repositories.

    Partitions
    ----------
    bipartite=0 : developer (actor_login) nodes
    bipartite=1 : repository (repo_name) nodes

    Edges
    -----
    One edge per unique (developer, repository) pair.
    weight = total number of qualifying contribution events between that
             developer and that repository during the observation window.

    Notes
    -----
    - This is an undirected weighted graph; direction (dev→repo) is implied
      by the bipartite attribute and is not encoded as a DiGraph because
      NetworkX's bipartite projection utilities expect nx.Graph.
    - Developers who contributed to multiple repos have degree > 1.
      Repositories with many contributors have high degree.
    - No self-loops are possible (partitions are disjoint by construction).

    Parameters
    ----------
    obs_df : cleaned observation-window event DataFrame with columns
             actor_login, repo_name (and any others).

    Returns
    -------
    nx.Graph — bipartite, weighted.
    """
    G = nx.Graph()

    # add all repo nodes first so even zero-contributor repos are present
    # (shouldn't happen after preprocessing, but defensive)
    for repo in obs_df["repo_name"].unique():
        G.add_node(repo, bipartite=1, node_type="repo")

    # aggregate (actor, repo) event counts in a single groupby — far faster
    # than iterating row-by-row for large DataFrames
    edge_weights = (
        obs_df.groupby(["actor_login", "repo_name"], sort=False)
        .size()
        .reset_index(name="weight")
    )

    for _, row in edge_weights.iterrows():
        actor = row["actor_login"]
        repo  = row["repo_name"]
        w     = int(row["weight"])

        if not G.has_node(actor):
            G.add_node(actor, bipartite=0, node_type="actor")
        G.add_edge(actor, repo, weight=w)

    return G


# ── 2. Per-repo bipartite contribution graph ──────────────────────────────────

def build_bipartite_graph(
    obs_df: pd.DataFrame,
    repo_name: str,
) -> nx.Graph:
    """
    Build a bipartite contribution graph restricted to a single repository.

    Identical schema to build_global_bipartite_graph() but contains only the
    nodes and edges relevant to repo_name.  Used for per-repo structural
    metrics (e.g. contribution weight Gini).

    Parameters
    ----------
    obs_df    : cleaned observation-window event DataFrame (all repos OK)
    repo_name : the repository to build the graph for

    Returns
    -------
    nx.Graph — bipartite, weighted.  The repo node is always present; if no
    events exist the graph contains only the repo node with no edges.
    """
    repo_df = obs_df[obs_df["repo_name"] == repo_name]

    G = nx.Graph()
    G.add_node(repo_name, bipartite=1, node_type="repo")

    if repo_df.empty:
        return G

    actor_counts = repo_df["actor_login"].value_counts()
    for actor, count in actor_counts.items():
        G.add_node(actor, bipartite=0, node_type="actor")
        G.add_edge(actor, repo_name, weight=int(count))

    return G


# ── 3. Per-repo contributor projection graph (subwindow-weighted) ─────────────

def build_contributor_graph(
    obs_df: pd.DataFrame,
    repo_name: str,
) -> nx.Graph:
    """
    Build a co-contributor graph for a single repository.

    Two developers share an edge if they were BOTH active in at least one
    of the same monthly 30-day subwindows (0 = Jan, 1 = Feb, …, 5 = Jun).
    The edge weight equals the number of subwindows in which both were active.

    Nodes
    -----
    All developers who contributed to repo_name in the obs window, including
    those active in only one subwindow with no co-active peers (degree 0).

    Edge weight semantics
    ---------------------
    weight=1 : developers co-occurred in exactly one subwindow
    weight=6 : developers were simultaneously active every month (max)

    A higher weight indicates a more sustained shared collaboration history,
    which is a stronger signal than a single co-occurrence.

    Assumption
    ----------
    The 'subwindow' column is expected on obs_df (added by
    preprocessing.assign_subwindow).  If absent it is computed here using
    the project default OBS_START = "2015-01-01".

    Parameters
    ----------
    obs_df    : cleaned observation-window event DataFrame
    repo_name : the repository to project

    Returns
    -------
    nx.Graph — unipartite (actors only), weighted.
    """
    repo_df = obs_df[obs_df["repo_name"] == repo_name].copy()

    G = nx.Graph()

    if repo_df.empty:
        return G

    # ensure subwindow column — compute on-the-fly if preprocessing omitted it
    if "subwindow" not in repo_df.columns:
        repo_df = assign_subwindow(
            repo_df, obs_start=OBS_START, n_subwindows=N_SUBWINDOWS
        )

    # drop any sentinel -1 rows (events outside the window; shouldn't exist
    # after clean split, but guard defensively)
    repo_df = repo_df[repo_df["subwindow"] >= 0]

    # add all contributor nodes (even those with no co-active peers)
    all_actors = repo_df["actor_login"].unique()
    G.add_nodes_from(all_actors)

    # build: subwindow -> frozenset of active developers
    sw_actor_sets: dict[int, set[str]] = {}
    for sw in range(N_SUBWINDOWS):
        actors_in_sw = set(
            repo_df.loc[repo_df["subwindow"] == sw, "actor_login"].unique()
        )
        if actors_in_sw:
            sw_actor_sets[sw] = actors_in_sw

    # for each subwindow, add 1 to the edge weight for every co-active pair
    # itertools.combinations gives each unordered pair exactly once per window
    for sw, actors in sw_actor_sets.items():
        for a, b in combinations(sorted(actors), 2):
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)

    return G


# backward-compatibility alias used by feature_engineering.py
build_collaboration_graph = build_contributor_graph


# ── 4. Gini helper ────────────────────────────────────────────────────────────

def _gini(values: list[float]) -> float:
    """
    Compute the Gini coefficient of a non-negative value list.

    Returns 0.0 for empty lists or all-zero lists (perfect equality
    is the conservative default when there is no data).

    The standard sorted-cumsum formula is used:
        G = (2 * sum_i( rank_i * x_i ) - (n+1) * sum(x)) / (n * sum(x))
    equivalently written as the form below for numerical stability.
    """
    if not values or sum(values) == 0:
        return 0.0
    n   = len(values)
    sv  = sorted(values)
    s   = sum(sv)
    num = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(sv))
    return num / (n * s)


# ── 5. Graph metrics ──────────────────────────────────────────────────────────

def compute_graph_metrics(
    G: nx.Graph,
    graph_type: str = "collaboration",
) -> dict:
    """
    Compute structural metrics from a graph and return them as a flat dict.

    Parameters
    ----------
    G          : graph produced by build_bipartite_graph or
                 build_contributor_graph (or build_global_bipartite_graph)
    graph_type : "collaboration"  — use contributor-graph metrics
                 "bipartite"      — use bipartite-graph metrics

    Returns
    -------
    dict of metric_name -> float or int  (see sub-functions for full schema)
    """
    if graph_type == "bipartite":
        return _bipartite_metrics(G)
    return _collaboration_metrics(G)


def _collaboration_metrics(G: nx.Graph) -> dict:
    """
    Compute metrics for the contributor projection graph.

    Metrics
    -------
    n_nodes          : number of developer nodes
    n_edges          : number of co-active-subwindow edges
    density          : edge density in [0, 1]
    avg_degree       : mean developer degree
    max_degree       : degree of the most-connected developer
    degree_gini      : Gini of degree distribution (0=equal, 1=one dominates)
    avg_edge_weight  : mean edge weight (mean shared-subwindow count)
    max_edge_weight  : maximum edge weight (most sustained collaboration pair)
    n_components     : number of connected components (>1 → isolated groups)
    largest_cc_frac  : fraction of nodes in the largest connected component
    avg_clustering   : mean clustering coefficient (triangle density)
    """
    n = G.number_of_nodes()
    if n == 0:
        return {
            "n_nodes": 0, "n_edges": 0, "density": 0.0,
            "avg_degree": 0.0, "max_degree": 0, "degree_gini": 0.0,
            "avg_edge_weight": 0.0, "max_edge_weight": 0,
            "n_components": 0, "largest_cc_frac": 0.0, "avg_clustering": 0.0,
        }

    degrees    = [d for _, d in G.degree()]
    components = list(nx.connected_components(G))
    weights    = [d.get("weight", 1) for _, _, d in G.edges(data=True)]

    return {
        "n_nodes":         n,
        "n_edges":         G.number_of_edges(),
        "density":         nx.density(G),
        "avg_degree":      sum(degrees) / n,
        "max_degree":      max(degrees),
        "degree_gini":     _gini(degrees),
        "avg_edge_weight": sum(weights) / len(weights) if weights else 0.0,
        "max_edge_weight": max(weights) if weights else 0,
        "n_components":    len(components),
        "largest_cc_frac": max(len(c) for c in components) / n,
        "avg_clustering":  nx.average_clustering(G),
    }


def _bipartite_metrics(G: nx.Graph) -> dict:
    """
    Compute metrics for a bipartite contribution graph.

    Metrics
    -------
    n_actors         : number of developer nodes (bipartite=0)
    n_edges          : total number of developer→repo edges
    total_weight     : sum of all edge weights (= total event count)
    max_actor_weight : highest single-developer event count
    weight_gini      : Gini of per-developer contribution weights
                       (0=uniform, 1=one developer does everything)
    """
    actors  = [nd for nd, d in G.nodes(data=True) if d.get("bipartite") == 0]
    weights = [G[a][r]["weight"] for a in actors for r in G.neighbors(a)]

    return {
        "n_actors":         len(actors),
        "n_edges":          G.number_of_edges(),
        "total_weight":     sum(weights),
        "max_actor_weight": max(weights) if weights else 0,
        "weight_gini":      _gini(weights),
    }


# ── 6. Build graphs and metrics for all candidate repos ───────────────────────

def build_all_repo_graphs(
    obs_df: pd.DataFrame,
    repo_names: Optional[list[str]] = None,
) -> dict[str, dict]:
    """
    Build both graph types and compute all metrics for every candidate repo.

    This is the primary entry point for feature_engineering.py.

    Parameters
    ----------
    obs_df     : cleaned observation-window event DataFrame (with 'subwindow')
    repo_names : list of repos to process.  Defaults to all unique repos in
                 obs_df if None.

    Returns
    -------
    dict mapping repo_name -> {
        "bipartite_graph":       nx.Graph   (per-repo bipartite)
        "collaboration_graph":   nx.Graph   (subwindow-weighted contributor graph)
        "bipartite_metrics":     dict       (from _bipartite_metrics)
        "collaboration_metrics": dict       (from _collaboration_metrics)
    }

    Note: the global bipartite graph is NOT included here because it spans all
    repos and is built separately with build_global_bipartite_graph().
    """
    if repo_names is None:
        repo_names = obs_df["repo_name"].unique().tolist()

    result: dict[str, dict] = {}
    for repo in repo_names:
        bg = build_bipartite_graph(obs_df, repo)
        cg = build_contributor_graph(obs_df, repo)
        result[repo] = {
            "bipartite_graph":       bg,
            "collaboration_graph":   cg,           # alias kept for feature_engineering
            "bipartite_metrics":     compute_graph_metrics(bg, graph_type="bipartite"),
            "collaboration_metrics": compute_graph_metrics(cg, graph_type="collaboration"),
        }

    return result


# ── 7. Example / smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from data_loading import load_datasets, load_candidate_repos

    print("=" * 65)
    print("  graph_building.py — example for one repository")
    print("=" * 65)

    # ── load data ─────────────────────────────────────────────────────────────
    candidates = load_candidate_repos()
    repo_set   = set(candidates["repo_name"].tolist())

    obs_df, _ = load_datasets(repo_names=repo_set)
    print(f"\nobs_df: {len(obs_df):,} events across {obs_df['repo_name'].nunique()} repos\n")

    # ── pick a small repo so the output is readable ───────────────────────────
    # viljamis/responsive-nav.js has ~88 events and 31 contributors — good size
    example_repo = "viljamis/responsive-nav.js"
    if example_repo not in repo_set:
        example_repo = sorted(repo_set)[0]
    print(f"Example repository: {example_repo}\n")

    # ── global bipartite graph ────────────────────────────────────────────────
    print("── Global bipartite graph ──────────────────────────────────────")
    G_global = build_global_bipartite_graph(obs_df)

    actor_nodes = [n for n, d in G_global.nodes(data=True) if d.get("bipartite") == 0]
    repo_nodes  = [n for n, d in G_global.nodes(data=True) if d.get("bipartite") == 1]
    print(f"  nodes  : {G_global.number_of_nodes():,}  "
          f"({len(actor_nodes):,} developers, {len(repo_nodes):,} repos)")
    print(f"  edges  : {G_global.number_of_edges():,}  (unique dev-repo pairs)")

    total_weight = sum(d["weight"] for _, _, d in G_global.edges(data=True))
    print(f"  total weight (events) : {total_weight:,}")

    # degree of the example repo node in the global graph
    if G_global.has_node(example_repo):
        repo_degree = G_global.degree(example_repo)
        repo_weight = sum(
            G_global[example_repo][nb]["weight"]
            for nb in G_global.neighbors(example_repo)
        )
        print(f"\n  {example_repo} in global graph:")
        print(f"    degree (unique contributors) : {repo_degree}")
        print(f"    total events                 : {repo_weight}")

    # ── per-repo bipartite graph ───────────────────────────────────────────────
    print("\n── Per-repo bipartite graph ────────────────────────────────────")
    G_bip = build_bipartite_graph(obs_df, example_repo)
    bip_m = compute_graph_metrics(G_bip, graph_type="bipartite")

    print(f"  nodes  : {G_bip.number_of_nodes()}  "
          f"({bip_m['n_actors']} actors + 1 repo)")
    print(f"  edges  : {bip_m['n_edges']}")
    print(f"  metrics:")
    for k, v in bip_m.items():
        print(f"    {k:<22} {v:.4f}" if isinstance(v, float) else f"    {k:<22} {v}")

    # top-3 contributors by event weight
    actors_sorted = sorted(
        [(a, G_bip[a][example_repo]["weight"])
         for a in G_bip.neighbors(example_repo)],
        key=lambda x: x[1], reverse=True
    )
    print(f"\n  top contributors:")
    for actor, w in actors_sorted[:5]:
        print(f"    {actor:<35}  {w} events")

    # ── per-repo contributor graph ─────────────────────────────────────────────
    print("\n── Per-repo contributor graph (subwindow-weighted) ─────────────")
    G_cg  = build_contributor_graph(obs_df, example_repo)
    cg_m  = compute_graph_metrics(G_cg, graph_type="collaboration")

    print(f"  nodes  : {G_cg.number_of_nodes()} developers")
    print(f"  edges  : {G_cg.number_of_edges()} co-active pairs")
    print(f"  metrics:")
    for k, v in cg_m.items():
        print(f"    {k:<22} {v:.4f}" if isinstance(v, float) else f"    {k:<22} {v}")

    # show edge weight distribution (how many pairs share 1 vs 2 vs ... subwindows)
    if G_cg.number_of_edges() > 0:
        from collections import Counter
        weight_dist = Counter(d["weight"] for _, _, d in G_cg.edges(data=True))
        print(f"\n  edge weight distribution (shared subwindows):")
        for w in sorted(weight_dist):
            print(f"    weight={w}  :  {weight_dist[w]} pairs")

    # ── build all repos ────────────────────────────────────────────────────────
    print("\n── build_all_repo_graphs() — all candidate repos ───────────────")
    all_graphs = build_all_repo_graphs(obs_df, list(repo_set))

    print(f"\n  {'repo':<45}  {'contributors':>12}  {'density':>8}  "
          f"{'n_components':>12}  {'weight_gini':>11}")
    print("  " + "-" * 95)
    for repo in sorted(all_graphs):
        cm = all_graphs[repo]["collaboration_metrics"]
        bm = all_graphs[repo]["bipartite_metrics"]
        print(
            f"  {repo:<45}  {cm['n_nodes']:>12}  {cm['density']:>8.3f}  "
            f"{cm['n_components']:>12}  {bm['weight_gini']:>11.3f}"
        )

    print("\n[done]")
