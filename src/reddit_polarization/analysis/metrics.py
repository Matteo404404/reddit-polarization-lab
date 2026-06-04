"""
analysis/metrics.py — assortativity, modularity, cross-cutting, echo index.
"""
from __future__ import annotations
import logging
from typing import Any
import networkx as nx

logger = logging.getLogger(__name__)

def _label(G, node, attr="ideology_label"):
    return G.nodes[node].get(attr, "unknown")

def _political_sub(G, labels=None):
    if labels is None: labels = {"left","right"}
    return G.subgraph([n for n in G if _label(G,n) in labels]).copy()

def basic_stats(G: nx.DiGraph) -> dict:
    n, e = G.number_of_nodes(), G.number_of_edges()
    wcc  = max(nx.weakly_connected_components(G), key=len) if n > 0 else set()
    return {
        "n_nodes": n, "n_edges": e,
        "density": nx.density(G) if n > 1 else 0.0,
        "giant_n": len(wcc),
        "giant_frac": len(wcc)/n if n > 0 else 0.0,
    }

def assortativity_by_ideology(G: nx.DiGraph, *, political_only=True) -> float:
    H = _political_sub(G) if political_only else G
    if H.number_of_edges() == 0: return float("nan")
    try:
        return float(nx.attribute_assortativity_coefficient(H, "ideology_label"))
    except Exception as e:
        logger.warning("assortativity failed: %s", e); return float("nan")

def cross_cutting_fraction(G: nx.DiGraph, *, political_only=True) -> float:
    H = _political_sub(G) if political_only else G
    if H.number_of_edges() == 0: return float("nan")
    cross = sum(1 for u,v in H.edges() if _label(H,u) != _label(H,v))
    return cross / H.number_of_edges()

def echo_index(G: nx.DiGraph, **kw) -> float:
    cc = cross_cutting_fraction(G, **kw)
    return float("nan") if cc != cc else 1.0 - cc

def modularity_partition(G: nx.DiGraph, *, political_only=True) -> float:
    import community as community_louvain
    H = _political_sub(G) if political_only else G
    Hu = H.to_undirected()
    if Hu.number_of_edges() == 0: return float("nan")
    raw  = {n: _label(Hu, n) for n in Hu}
    uniq = sorted(set(raw.values()))
    lti  = {l: i for i,l in enumerate(uniq)}
    ipart = {n: lti[l] for n,l in raw.items()}
    return float(community_louvain.modularity(ipart, Hu, weight="weight"))

def modularity_louvain(G: nx.DiGraph, *, random_state=42) -> tuple[float, dict]:
    import community as community_louvain
    Hu = G.to_undirected()
    if Hu.number_of_edges() == 0: return float("nan"), {}
    part = community_louvain.best_partition(Hu, weight="weight", random_state=random_state)
    Q    = community_louvain.modularity(part, Hu, weight="weight")
    return float(Q), part

def compute_all(G: nx.DiGraph, *, run_louvain=True) -> dict[str, Any]:
    m = basic_stats(G)
    m["assortativity"]        = assortativity_by_ideology(G)
    m["cross_cutting_frac"]   = cross_cutting_fraction(G)
    m["echo_index"]           = echo_index(G)
    m["modularity_partition"] = modularity_partition(G)
    if run_louvain:
        Q, _ = modularity_louvain(G)
        m["modularity_louvain"] = Q
    logger.info("Metrics: %s", {k: round(v,4) if isinstance(v,float) else v for k,v in m.items()})
    return m