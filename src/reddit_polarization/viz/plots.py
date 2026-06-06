"""
viz/plots.py
Static plots: metrics bar chart, edge breakdown pie, network sample.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

COLORS = {
    "left":          "#4e79a7",
    "right":         "#e15759",
    "mixed":         "#f28e2b",
    "general":       "#76b7b2",
    "non_political": "#bab0ac",
    "unknown":       "#d3d3d3",
}

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)


# ── 1. Metrics summary bar chart ─────────────────────────────────────────────

def plot_metrics_summary(metrics: dict, *, save: bool = True) -> plt.Figure:
    keys  = ["assortativity", "cross_cutting_frac", "echo_index",
             "modularity_partition", "modularity_louvain"]
    vals  = [metrics.get(k, 0) for k in keys]
    labels = ["Assortativity\n(r)", "Cross-cutting\nfraction",
              "Echo\nindex", "Modularity\n(partition)", "Modularity\n(Louvain)"]

    bar_colors = ["#e15759" if v > 0.5 else "#4e79a7" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=bar_colors, edgecolor="white", linewidth=1.2, width=0.55)

    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_ylabel("Value (0–1)", fontsize=11)
    ax.set_title("Reddit Political Polarization — Structural Metrics", fontsize=13, pad=14)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(OUT / "metrics_summary.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {OUT / 'metrics_summary.png'}")
    return fig


# ── 2. Edge breakdown donut ──────────────────────────────────────────────────

def plot_edge_breakdown(
    same_left: int, same_right: int, cross: int,
    *, save: bool = True
) -> plt.Figure:
    total  = same_left + same_right + cross
    sizes  = [same_left, same_right, cross]
    labels = [
        f"Left → Left\n{same_left:,} ({same_left/total*100:.1f}%)",
        f"Right → Right\n{same_right:,} ({same_right/total*100:.1f}%)",
        f"Cross-group\n{cross:,} ({cross/total*100:.1f}%)",
    ]
    colors = [COLORS["left"], COLORS["right"], COLORS["mixed"]]

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
    )
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, fontsize=10)
    ax.set_title("Political Interaction Structure\n(political subgraph edges)",
                 fontsize=12, pad=14)

    fig.tight_layout()
    if save:
        fig.savefig(OUT / "edge_breakdown.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {OUT / 'edge_breakdown.png'}")
    return fig


# ── 3. Network sample visualization ──────────────────────────────────────────

def plot_network_sample(
    G: nx.DiGraph,
    *,
    n_sample: int = 600,
    seed: int = 42,
    save: bool = True,
) -> plt.Figure:
    """
    Draw a sampled subgraph colored by ideology.
    Samples top-degree nodes for visual clarity.
    """
    # Take top-degree nodes for a readable sample
    top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:n_sample]
    H = G.subgraph(top_nodes).copy()

    node_colors = [
        COLORS.get(H.nodes[n].get("ideology_label", "unknown"), COLORS["unknown"])
        for n in H.nodes()
    ]

    pos = nx.spring_layout(H, seed=seed, k=0.4)

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(
        H, pos, ax=ax, alpha=0.15, width=0.4,
        edge_color="gray", arrows=False,
    )
    nx.draw_networkx_nodes(
        H, pos, ax=ax,
        node_color=node_colors, node_size=25, alpha=0.85,
    )
    ax.set_title(
        f"Political Interaction Network — top {n_sample} nodes by degree\n"
        f"Blue = Left  |  Red = Right  |  Orange = Mixed/General",
        fontsize=11, pad=12,
    )
    ax.axis("off")

    patches = [mpatches.Patch(color=COLORS["left"],    label="Left"),
               mpatches.Patch(color=COLORS["right"],   label="Right"),
               mpatches.Patch(color=COLORS["mixed"],   label="Mixed/General"),
               mpatches.Patch(color=COLORS["unknown"], label="Unknown")]
    ax.legend(handles=patches, loc="lower left", frameon=False, fontsize=10)

    fig.tight_layout()
    if save:
        fig.savefig(OUT / "network_sample.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {OUT / 'network_sample.png'}")
    return fig


# ── 4. Ideology degree distribution ──────────────────────────────────────────

def plot_degree_by_ideology(G: nx.DiGraph, *, save: bool = True) -> plt.Figure:
    import pandas as pd

    rows = [
        {"node": n, "degree": G.degree(n),
         "ideology": d.get("ideology_label", "unknown")}
        for n, d in G.nodes(data=True)
    ]
    df = pd.DataFrame(rows)
    pol = df[df["ideology"].isin(["left", "right"])]

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color in [("left", COLORS["left"]), ("right", COLORS["right"])]:
        vals = pol[pol["ideology"] == label]["degree"]
        ax.hist(vals, bins=50, alpha=0.65, color=color, label=label.capitalize(),
                edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Node degree (total interactions)", fontsize=11)
    ax.set_ylabel("Number of users", fontsize=11)
    ax.set_title("Degree Distribution by Ideology\n(political subgraph)", fontsize=12)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if save:
        fig.savefig(OUT / "degree_distribution.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {OUT / 'degree_distribution.png'}")
    return fig