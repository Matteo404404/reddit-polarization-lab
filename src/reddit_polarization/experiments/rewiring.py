"""
experiments/rewiring.py
Counterfactual edge rewiring on a political interaction graph.

Two treatments:
  - diversify    : add synthetic cross-group edges (simulate diverse feed)
  - homophilize  : remove existing cross-group edges (simulate filter bubble)

Returns a tidy DataFrame of metrics per (treatment, fraction).
"""
from __future__ import annotations
import logging
import random
from typing import Sequence

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

POL_LABELS = {"left", "right"}


# ── Metric helpers ────────────────────────────────────────────────────────────

def _assortativity(G: nx.DiGraph) -> float:
    try:
        return nx.attribute_assortativity_coefficient(G, "ideology_label")
    except Exception:
        return float("nan")


def _cross_cutting_frac(G: nx.DiGraph) -> float:
    pol_edges = [
        (u, v) for u, v in G.edges()
        if G.nodes[u].get("ideology_label") in POL_LABELS
        and G.nodes[v].get("ideology_label") in POL_LABELS
    ]
    if not pol_edges:
        return float("nan")
    cross = sum(
        1 for u, v in pol_edges
        if G.nodes[u].get("ideology_label") != G.nodes[v].get("ideology_label")
    )
    return cross / len(pol_edges)


def _modularity_partition(G: nx.DiGraph) -> float:
    try:
        partition = {
            n: G.nodes[n].get("ideology_label", "unknown")
            for n in G.nodes()
        }
        communities = {}
        for node, label in partition.items():
            communities.setdefault(label, set()).add(node)
        return nx.algorithms.community.quality.modularity(
            G, list(communities.values()), weight="weight"
        )
    except Exception:
        return float("nan")


def _snapshot_metrics(G: nx.DiGraph, treatment: str, fraction: float) -> dict:
    cc = _cross_cutting_frac(G)
    return {
        "treatment":           treatment,
        "fraction":            fraction,
        "n_nodes":             G.number_of_nodes(),
        "n_edges":             G.number_of_edges(),
        "assortativity":       _assortativity(G),
        "cross_cutting_frac":  cc,
        "echo_index":          1.0 - cc if cc == cc else float("nan"),
        "modularity_partition": _modularity_partition(G),
    }


# ── Treatment A: diversify ────────────────────────────────────────────────────

def _diversify(
    G: nx.DiGraph,
    fraction: float,
    seed: int,
) -> nx.DiGraph:
    """
    Add synthetic cross-group edges.
    For each left node, sample right-leaning candidates of similar degree
    and add a directed edge with median weight.
    fraction = fraction of existing edges to add as new cross-group edges.
    """
    rng = random.Random(seed)
    H   = G.copy()

    left_nodes  = [n for n, d in H.nodes(data=True) if d.get("ideology_label") == "left"]
    right_nodes = [n for n, d in H.nodes(data=True) if d.get("ideology_label") == "right"]

    if not left_nodes or not right_nodes:
        return H

    n_add = max(1, int(H.number_of_edges() * fraction))
    weights = [d.get("weight", 1) for _, _, d in H.edges(data=True)]
    median_w = sorted(weights)[len(weights) // 2]

    added = 0
    attempts = 0
    max_attempts = n_add * 10

    while added < n_add and attempts < max_attempts:
        src = rng.choice(left_nodes)
        tgt = rng.choice(right_nodes)
        if src != tgt and not H.has_edge(src, tgt):
            H.add_edge(src, tgt, weight=median_w)
            added += 1
        attempts += 1

    logger.info("Diversify (frac=%.2f): added %d cross-group edges", fraction, added)
    return H


# ── Treatment B: homophilize ──────────────────────────────────────────────────

def _homophilize(
    G: nx.DiGraph,
    fraction: float,
    seed: int,
) -> nx.DiGraph:
    """
    Remove a fraction of existing cross-group edges.
    Simulates an algorithm that suppresses cross-cutting content.
    """
    rng = random.Random(seed)
    H   = G.copy()

    cross_edges = [
        (u, v) for u, v in H.edges()
        if H.nodes[u].get("ideology_label") in POL_LABELS
        and H.nodes[v].get("ideology_label") in POL_LABELS
        and H.nodes[u].get("ideology_label") != H.nodes[v].get("ideology_label")
    ]

    n_remove = max(1, int(len(cross_edges) * fraction))
    to_remove = rng.sample(cross_edges, min(n_remove, len(cross_edges)))
    H.remove_edges_from(to_remove)

    logger.info(
        "Homophilize (frac=%.2f): removed %d cross-group edges (of %d total)",
        fraction, len(to_remove), len(cross_edges),
    )
    return H


# ── Main experiment runner ────────────────────────────────────────────────────

def run_rewiring_experiment(
    G: nx.DiGraph,
    *,
    diversify_fractions:  Sequence[float] = (0.05, 0.10, 0.20, 0.35, 0.50),
    homophilize_fractions: Sequence[float] = (0.10, 0.20, 0.30, 0.50, 0.70),
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run baseline + both treatments at multiple fractions.
    Returns a tidy DataFrame with one row per (treatment, fraction).
    """
    rows = []

    # Baseline
    logger.info("Computing baseline metrics…")
    rows.append(_snapshot_metrics(G, "baseline", 0.0))

    # Diversify
    for frac in diversify_fractions:
        logger.info("Treatment: diversify, fraction=%.2f", frac)
        H = _diversify(G, frac, seed=seed)
        rows.append(_snapshot_metrics(H, "diversify", frac))

    # Homophilize
    for frac in homophilize_fractions:
        logger.info("Treatment: homophilize, fraction=%.2f", frac)
        H = _homophilize(G, frac, seed=seed)
        rows.append(_snapshot_metrics(H, "homophilize", frac))

    df = pd.DataFrame(rows)
    logger.info("Experiment complete: %d scenarios", len(df))
    return df