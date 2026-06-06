"""
scripts/make_figures.py
Generate all static figures from the saved bundle.
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx
from reddit_polarization.graphs.build_graphs import RedditGraphBundle
from reddit_polarization.viz.plots import (
    plot_metrics_summary,
    plot_edge_breakdown,
    plot_network_sample,
    plot_degree_by_ideology,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

bundle = RedditGraphBundle.load("data/processed/bundle.pkl")
G      = bundle.G_user

# Political subgraph
POL    = {"left", "right"}
pol_nodes = [n for n, d in G.nodes(data=True) if d.get("ideology_label") in POL]
H      = G.subgraph(pol_nodes).copy()

# Edge counts
same_left  = sum(1 for u,v in H.edges()
                 if H.nodes[u].get("ideology_label")=="left"
                 and H.nodes[v].get("ideology_label")=="left")
same_right = sum(1 for u,v in H.edges()
                 if H.nodes[u].get("ideology_label")=="right"
                 and H.nodes[v].get("ideology_label")=="right")
cross      = H.number_of_edges() - same_left - same_right

metrics = {
    "assortativity":        0.7591,
    "cross_cutting_frac":   0.1189,
    "echo_index":           0.8811,
    "modularity_partition": 0.3866,
    "modularity_louvain":   0.7370,
}

print("Generating figures…")
plot_metrics_summary(metrics)
plot_edge_breakdown(same_left, same_right, cross)
plot_network_sample(H, n_sample=600)
plot_degree_by_ideology(H)

print("\n✅ All figures saved to results/figures/")