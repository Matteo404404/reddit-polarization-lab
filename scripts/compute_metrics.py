"""
scripts/compute_metrics.py  — fast version
Works on the politically-labelled subgraph only.
Full graph stats are computed cheaply without Louvain.
"""
import sys, logging, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx
from reddit_polarization.graphs.build_graphs import RedditGraphBundle

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

bundle = RedditGraphBundle.load("data/processed/bundle.pkl")
G      = bundle.G_user

print(f"\nFull graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# ── 1. Fast stats on full graph (no community detection) ─────────────────────
t0 = time.time()
wcc  = max(nx.weakly_connected_components(G), key=len)
print(f"Giant component: {len(wcc):,} nodes ({len(wcc)/G.number_of_nodes()*100:.1f}%)")
print(f"Density: {nx.density(G):.6f}")
print(f"Full graph stats done in {time.time()-t0:.1f}s")

# ── 2. Extract political subgraph (left + right only) ────────────────────────
POL_LABELS = {"left", "right"}
pol_nodes  = [n for n, d in G.nodes(data=True)
              if d.get("ideology_label") in POL_LABELS]
H = G.subgraph(pol_nodes).copy()
print(f"\nPolitical subgraph (left+right): {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

# ── 3. All metrics on political subgraph ─────────────────────────────────────
from reddit_polarization.analysis.metrics import (
    assortativity_by_ideology,
    cross_cutting_fraction,
    echo_index,
    modularity_partition,
    modularity_louvain,
)

print("\nComputing assortativity…")
t0 = time.time()
r  = assortativity_by_ideology(H, political_only=False)
print(f"  assortativity          {r:.4f}   ({time.time()-t0:.1f}s)")

print("Computing cross-cutting fraction…")
t0 = time.time()
cc = cross_cutting_fraction(H, political_only=False)
print(f"  cross_cutting_frac     {cc:.4f}   ({time.time()-t0:.1f}s)")
print(f"  echo_index             {1-cc:.4f}")

print("Computing modularity of left/right partition…")
t0 = time.time()
Q_part = modularity_partition(H, political_only=False)
print(f"  modularity_partition   {Q_part:.4f}   ({time.time()-t0:.1f}s)")

print("Computing Louvain on political subgraph (much faster now)…")
t0 = time.time()
Q_louv, partition = modularity_louvain(H)
print(f"  modularity_louvain     {Q_louv:.4f}   ({time.time()-t0:.1f}s)")

# ── 4. Label distribution in political subgraph ──────────────────────────────
import collections
label_counts = collections.Counter(
    d.get("ideology_label","unknown")
    for _, d in H.nodes(data=True)
)
print(f"\nPolitical subgraph node labels: {dict(label_counts)}")

# ── 5. Cross-group edge breakdown ─────────────────────────────────────────────
left_right = sum(
    1 for u, v in H.edges()
    if {H.nodes[u].get("ideology_label"), H.nodes[v].get("ideology_label")} == {"left","right"}
)
same_left  = sum(1 for u,v in H.edges()
                 if H.nodes[u].get("ideology_label")=="left"
                 and H.nodes[v].get("ideology_label")=="left")
same_right = sum(1 for u,v in H.edges()
                 if H.nodes[u].get("ideology_label")=="right"
                 and H.nodes[v].get("ideology_label")=="right")

total = H.number_of_edges()
print(f"\nEdge breakdown (political subgraph):")
print(f"  left  → left   {same_left:>8,}  ({same_left/total*100:.1f}%)")
print(f"  right → right  {same_right:>8,}  ({same_right/total*100:.1f}%)")
print(f"  cross-group    {left_right:>8,}  ({left_right/total*100:.1f}%)")

print("\n✅ Done.")