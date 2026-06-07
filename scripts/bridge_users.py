"""
scripts/bridge_users.py
Identify bridge users — those connecting left and right communities.
Outputs:
  results/logs/bridge_users.csv   — full ranked table
  results/figures/bridge_scatter.png
  results/figures/bridge_network.png
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from reddit_polarization.graphs.build_graphs import RedditGraphBundle
from reddit_polarization.graphs.attributes   import (
    add_degree, add_betweenness, add_cross_group_exposure, build_node_table
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
OUT_FIG = Path("results/figures")
OUT_LOG = Path("results/logs")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_LOG.mkdir(parents=True, exist_ok=True)

COLORS = {"left": "#4e79a7", "right": "#e15759",
          "mixed": "#f28e2b", "general": "#76b7b2",
          "non_political": "#bab0ac", "unknown": "#d3d3d3"}
POL    = {"left", "right"}

# ── Load ─────────────────────────────────────────────────────────────────────
bundle = RedditGraphBundle.load("data/processed/bundle.pkl")
G      = bundle.G_user

# Work on political subgraph only for speed
pol_nodes = [n for n, d in G.nodes(data=True) if d.get("ideology_label") in POL]
H = G.subgraph(pol_nodes).copy()
print(f"Political subgraph: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

# ── Compute attributes ────────────────────────────────────────────────────────
print("Computing degree…")
add_degree(H)

print("Computing cross-group exposure…")
add_cross_group_exposure(H)

print("Computing betweenness (k=300, ~30s)…")
add_betweenness(H, k=300)

# ── Build node table ──────────────────────────────────────────────────────────
df = build_node_table(H)
df = df[df["ideology_label"].isin(POL)].copy()
df = df.dropna(subset=["cross_group_share", "betweenness"])
df = df.sort_values("betweenness", ascending=False)

# Bridge score: combination of betweenness + cross_group_share
df["bridge_score"] = (
    df["betweenness"]       / df["betweenness"].max() * 0.6 +
    df["cross_group_share"] / 1.0                     * 0.4
)
df = df.sort_values("bridge_score", ascending=False)

df.to_csv(OUT_LOG / "bridge_users.csv", index=False)
print(f"\nTop 20 bridge users:")
print(df[["author","ideology_label","bridge_score",
          "betweenness","cross_group_share","total_degree"]].head(20).to_string())

# ── Plot 1: Scatter betweenness vs cross-group share ─────────────────────────
print("\nGenerating scatter plot…")
fig, ax = plt.subplots(figsize=(10, 7))

for label in ["left", "right"]:
    sub = df[df["ideology_label"] == label]
    ax.scatter(
        sub["cross_group_share"],
        sub["betweenness"],
        c=COLORS[label], label=label.capitalize(),
        alpha=0.5, s=18, edgecolors="none",
    )

# Annotate top 10 bridge users
top10 = df.head(10)
for _, row in top10.iterrows():
    ax.annotate(
        row["author"],
        xy=(row["cross_group_share"], row["betweenness"]),
        xytext=(6, 3), textcoords="offset points",
        fontsize=7, color="#333333",
    )

ax.set_xlabel("Cross-group neighbor share", fontsize=12)
ax.set_ylabel("Betweenness centrality (normalised)", fontsize=12)
ax.set_title("Bridge Users: Betweenness vs Cross-group Exposure\n"
             "Top-right = structural bridges between left and right",
             fontsize=12, pad=12)
ax.legend(frameon=False, fontsize=11)
ax.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_FIG / "bridge_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG / 'bridge_scatter.png'}")


# ── Plot 2: Cross-group share distribution by ideology ───────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))

for label in ["left", "right"]:
    sub = df[df["ideology_label"] == label]["cross_group_share"].dropna()
    ax2.hist(sub, bins=30, alpha=0.65, color=COLORS[label],
             label=f"{label.capitalize()} (n={len(sub):,})",
             edgecolor="white", linewidth=0.5)

ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1,
            label="50% cross-group threshold")
ax2.set_xlabel("Cross-group neighbor share", fontsize=12)
ax2.set_ylabel("Number of users", fontsize=12)
ax2.set_title("Distribution of Cross-group Exposure by Ideology\n"
              "Users to the right of 0.5 interact more outside their group",
              fontsize=12, pad=12)
ax2.legend(frameon=False, fontsize=10)
ax2.spines[["top","right"]].set_visible(False)
fig2.tight_layout()
fig2.savefig(OUT_FIG / "cross_group_distribution.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG / 'cross_group_distribution.png'}")


# ── Plot 3: Bridge network — top bridge users highlighted ────────────────────
print("Generating bridge network visualization…")

TOP_N_BRIDGE = 20
bridge_names = set(df.head(TOP_N_BRIDGE)["author"])

# Sample subgraph: top 800 nodes by degree + all bridge users
top_by_deg = sorted(H.nodes(), key=lambda n: H.degree(n), reverse=True)[:800]
sample_nodes = list(set(top_by_deg) | bridge_names)
S = H.subgraph(sample_nodes).copy()

node_colors = [COLORS.get(S.nodes[n].get("ideology_label","unknown"), COLORS["unknown"])
               for n in S.nodes()]
node_sizes  = [120 if n in bridge_names else 15 for n in S.nodes()]
node_borders= [1.5 if n in bridge_names else 0  for n in S.nodes()]

pos = nx.spring_layout(S, seed=42, k=0.45)

fig3, ax3 = plt.subplots(figsize=(13, 10))
nx.draw_networkx_edges(S, pos, ax=ax3, alpha=0.12, width=0.4,
                       edge_color="gray", arrows=False)
nx.draw_networkx_nodes(S, pos, ax=ax3, node_color=node_colors,
                       node_size=node_sizes, linewidths=node_borders,
                       edgecolors="#222222")

# Label only bridge users
bridge_in_sample = {n: n for n in S.nodes() if n in bridge_names}
nx.draw_networkx_labels(S, pos, labels=bridge_in_sample, ax=ax3,
                        font_size=6.5, font_color="#111111")

patches = [
    mpatches.Patch(color=COLORS["left"],  label="Left"),
    mpatches.Patch(color=COLORS["right"], label="Right"),
    mpatches.Patch(color="white", ec="#222", label=f"Top-{TOP_N_BRIDGE} bridge users (outlined)"),
]
ax3.legend(handles=patches, loc="lower left", frameon=False, fontsize=10)
ax3.set_title(f"Political Network — Top {TOP_N_BRIDGE} Bridge Users highlighted\n"
              f"(larger outlined nodes = highest bridge score)",
              fontsize=11, pad=12)
ax3.axis("off")
fig3.tight_layout()
fig3.savefig(OUT_FIG / "bridge_network.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG / 'bridge_network.png'}")

print("\n✅ A1 complete. Output:")
print(f"   results/logs/bridge_users.csv")
print(f"   results/figures/bridge_scatter.png")
print(f"   results/figures/cross_group_distribution.png")
print(f"   results/figures/bridge_network.png")