"""
graphs/attributes.py
Compute and attach centrality + cross-group exposure to G_user nodes.
"""
from __future__ import annotations
import logging
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

POL_LABELS = {"left", "right"}


def add_degree(G: nx.DiGraph) -> nx.DiGraph:
    nx.set_node_attributes(G, dict(G.in_degree()),  "in_degree")
    nx.set_node_attributes(G, dict(G.out_degree()), "out_degree")
    nx.set_node_attributes(G, dict(G.degree()),     "total_degree")
    return G


def add_pagerank(G: nx.DiGraph, *, alpha: float = 0.85) -> nx.DiGraph:
    pr = nx.pagerank(G, alpha=alpha, weight="weight")
    nx.set_node_attributes(G, pr, "pagerank")
    logger.info("PageRank computed")
    return G


def add_betweenness(
    G: nx.DiGraph,
    *,
    k: int = 500,
    normalized: bool = True,
) -> nx.DiGraph:
    """
    Approximated betweenness centrality (k-sample).
    k=500 is fast even on large graphs; increase for more precision.
    """
    bc = nx.betweenness_centrality(G, k=k, normalized=normalized, weight="weight")
    nx.set_node_attributes(G, bc, "betweenness")
    logger.info("Betweenness centrality computed (k=%d)", k)
    return G


def add_cross_group_exposure(G: nx.DiGraph) -> nx.DiGraph:
    """
    For each node, compute:
      - cross_group_neighbors : count of neighbors with a DIFFERENT political label
      - cross_group_share     : fraction of neighbors that are cross-group
      - same_group_neighbors  : count of same-label neighbors
    Only counts neighbors with a clear political label (left/right).
    """
    exposure = {}
    for node in G.nodes():
        my_label = G.nodes[node].get("ideology_label")
        if my_label not in POL_LABELS:
            exposure[node] = {
                "cross_group_neighbors": 0,
                "same_group_neighbors":  0,
                "cross_group_share":     float("nan"),
            }
            continue

        neighbors = list(G.predecessors(node)) + list(G.successors(node))
        pol_neighbors = [
            n for n in neighbors
            if G.nodes[n].get("ideology_label") in POL_LABELS
        ]

        if not pol_neighbors:
            exposure[node] = {
                "cross_group_neighbors": 0,
                "same_group_neighbors":  0,
                "cross_group_share":     float("nan"),
            }
            continue

        cross = sum(1 for n in pol_neighbors
                    if G.nodes[n].get("ideology_label") != my_label)
        same  = len(pol_neighbors) - cross

        exposure[node] = {
            "cross_group_neighbors": cross,
            "same_group_neighbors":  same,
            "cross_group_share":     cross / len(pol_neighbors),
        }

    for attr in ("cross_group_neighbors", "same_group_neighbors", "cross_group_share"):
        nx.set_node_attributes(G, {n: v[attr] for n, v in exposure.items()}, attr)

    logger.info("Cross-group exposure computed for %d nodes", len(exposure))
    return G


def build_node_table(G: nx.DiGraph) -> pd.DataFrame:
    """
    Export all node attributes to a tidy DataFrame.
    One row per node, columns = all attributes.
    """
    rows = []
    for node, data in G.nodes(data=True):
        row = {"author": node}
        row.update(data)
        rows.append(row)
    return pd.DataFrame(rows)


def add_all(
    G: nx.DiGraph,
    *,
    compute_betweenness: bool = True,
    k_betweenness: int = 500,
) -> nx.DiGraph:
    add_degree(G)
    add_cross_group_exposure(G)
    if compute_betweenness:
        add_betweenness(G, k=k_betweenness)
    return G