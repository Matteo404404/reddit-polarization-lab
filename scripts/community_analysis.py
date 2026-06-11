"""
scripts/community_analysis.py
A5: Analyse Louvain communities vs ideology labels.
Key questions:
  - How many Louvain communities exist, and what is their size distribution?
  - How pure are they ideologically (left/right composition)?
  - Are there sub-communities within left or right?
  - Sankey-style alluvial: ideology label → Louvain community
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

try:
    from community import best_partition          # python-louvain
except ImportError:
    from networkx.algorithms.community import louvain_communities

from reddit_polarization.graphs.build_graphs import RedditGraphBundle

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
OUT_FIG = Path("results/figures")
OUT_LOG = Path("results/logs")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_LOG.mkdir(parents=True, exist_ok=True)

COLORS = {"left": "#4e79a7", "right": "#e15759"}
POL    = {"left", "right"}

# ── Load ──────────────────────────────────────────────────────────────────────
bundle    = RedditGraphBundle.load("data/processed/bundle.pkl")
G         = bundle.G_user
pol_nodes = [n for n, d in G.nodes(data=True) if d.get("ideology_label") in POL]
H         = G.subgraph(pol_nodes).copy()
U         = H.to_undirected()
print(f"Political subgraph: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

# ── Louvain ───────────────────────────────────────────────────────────────────
print("Running Louvain…")
try:
    partition = best_partition(U, weight="weight", random_state=42)
    community_map = partition   # node → community_id
except NameError:
    communities = louvain_communities(U, weight="weight", seed=42)
    community_map = {}
    for cid, members in enumerate(communities):
        for node in members:
            community_map[node] = cid

nx.set_node_attributes(H, community_map, "louvain_community")
n_communities = len(set(community_map.values()))
print(f"Found {n_communities} Louvain communities")

# ── Build node table ──────────────────────────────────────────────────────────
rows = [
    {
        "author":    n,
        "ideology":  H.nodes[n].get("ideology_label"),
        "community": community_map.get(n, -1),
        "degree":    H.degree(n),
    }
    for n in H.nodes()
]
df = pd.DataFrame(rows)
df.to_csv(OUT_LOG / "community_membership.csv", index=False)

# ── Community purity ──────────────────────────────────────────────────────────
comm_stats = []
for cid, group in df.groupby("community"):
    total = len(group)
    left  = (group["ideology"] == "left").sum()
    right = (group["ideology"] == "right").sum()
    purity = max(left, right) / total
    dominant = "left" if left >= right else "right"
    comm_stats.append({
        "community": cid,
        "size":      total,
        "left":      left,
        "right":     right,
        "purity":    purity,
        "dominant":  dominant,
    })

comm_df = pd.DataFrame(comm_stats).sort_values("size", ascending=False)
comm_df.to_csv(OUT_LOG / "community_stats.csv", index=False)

print("\n── Top 15 communities by size ──")
print(comm_df.head(15)[["community","size","left","right","purity","dominant"]].to_string())

# ── Plot 1: Community size distribution coloured by dominant ideology ─────────
top15 = comm_df.head(15)
bar_colors = [COLORS.get(d, "#aaa") for d in top15["dominant"]]

fig1, ax1 = plt.subplots(figsize=(12, 5))
bars = ax1.bar(
    range(len(top15)), top15["size"],
    color=bar_colors, edgecolor="white", linewidth=0.8,
)
ax1.set_xticks(range(len(top15)))
ax1.set_xticklabels([f"C{int(r['community'])}" for _, r in top15.iterrows()],
                    fontsize=9)
ax1.set_ylabel("Community size (nodes)", fontsize=11)
ax1.set_title("Top 15 Louvain Communities by Size\n"
              "Blue = Left-dominant   Red = Right-dominant",
              fontsize=12, pad=10)

# Add purity label
for bar, (_, row) in zip(bars, top15.iterrows()):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 10,
             f"{row['purity']:.0%}", ha="center", fontsize=8, color="#333")

ax1.spines[["top","right"]].set_visible(False)
fig1.tight_layout()
fig1.savefig(OUT_FIG / "community_sizes.png", dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FIG / 'community_sizes.png'}")

# ── Plot 2: Stacked bar — ideology composition per community ──────────────────
fig2, ax2 = plt.subplots(figsize=(12, 5))
top10 = comm_df.head(10)
x     = range(len(top10))
ax2.bar(x, top10["left"],  color=COLORS["left"],  label="Left",  edgecolor="white")
ax2.bar(x, top10["right"], bottom=top10["left"],
        color=COLORS["right"], label="Right", edgecolor="white")
ax2.set_xticks(list(x))
ax2.set_xticklabels([f"C{int(r['community'])}" for _, r in top10.iterrows()], fontsize=10)
ax2.set_ylabel("Number of users", fontsize=11)
ax2.set_title("Ideology Composition of Top 10 Louvain Communities",
              fontsize=12, pad=10)
ax2.legend(frameon=False, fontsize=10)
ax2.spines[["top","right"]].set_visible(False)
fig2.tight_layout()
fig2.savefig(OUT_FIG / "community_composition.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG / 'community_composition.png'}")

# ── Plot 3: Network coloured by Louvain community ─────────────────────────────
print("Generating Louvain network visualization…")
top_nodes = sorted(H.nodes(), key=lambda n: H.degree(n), reverse=True)[:800]
S         = H.subgraph(top_nodes).copy()

# Assign colors to top-N communities, grey for small ones
top_comm_ids = set(comm_df.head(8)["community"].tolist())
palette      = plt.cm.tab10.colors
comm_color_map = {cid: palette[i % 10]
                  for i, cid in enumerate(comm_df.head(8)["community"])}

node_colors = []
for n in S.nodes():
    cid = community_map.get(n, -1)
    if cid in comm_color_map:
        node_colors.append(comm_color_map[cid])
    else:
        node_colors.append("#cccccc")

pos = nx.spring_layout(S, seed=42, k=0.45)

fig3, ax3 = plt.subplots(figsize=(13, 10))
nx.draw_networkx_edges(S, pos, ax=ax3, alpha=0.10, width=0.4,
                       edge_color="gray", arrows=False)
nx.draw_networkx_nodes(S, pos, ax=ax3, node_color=node_colors,
                       node_size=20, alpha=0.9)

# Legend: community id + dominant ideology + size
import matplotlib.patches as mpatches
legend_patches = []
for i, (_, row) in enumerate(comm_df.head(8).iterrows()):
    cid   = int(row["community"])
    color = comm_color_map[cid]
    dom   = row["dominant"]
    label = f"C{cid} — {dom} ({row['size']} users, {row['purity']:.0%} pure)"
    legend_patches.append(mpatches.Patch(color=color, label=label))
legend_patches.append(mpatches.Patch(color="#cccccc", label="Small communities"))

ax3.legend(handles=legend_patches, loc="lower left", frameon=False,
           fontsize=8.5, title="Louvain Communities", title_fontsize=9)
ax3.set_title("Political Network — Nodes coloured by Louvain Community\n"
              "(top 800 nodes by degree)", fontsize=11, pad=12)
ax3.axis("off")
fig3.tight_layout()
fig3.savefig(OUT_FIG / "louvain_network.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG / 'louvain_network.png'}")

# ── Plot 4: Purity distribution ───────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.hist(comm_df["purity"], bins=20, color="#76b7b2",
         edgecolor="white", linewidth=0.8)
ax4.axvline(0.9, color="#e15759", linestyle="--", linewidth=1.5,
            label="90% purity threshold")
ax4.axvline(comm_df["purity"].mean(), color="#4e79a7", linestyle="-", linewidth=1.5,
            label=f"Mean purity ({comm_df['purity'].mean():.2f})")
ax4.set_xlabel("Community ideological purity", fontsize=11)
ax4.set_ylabel("Number of communities", fontsize=11)
ax4.set_title("Distribution of Ideological Purity Across Louvain Communities\n"
              "1.0 = all members share same ideology", fontsize=11, pad=10)
ax4.legend(frameon=False, fontsize=10)
ax4.spines[["top","right"]].set_visible(False)
fig4.tight_layout()
fig4.savefig(OUT_FIG / "community_purity.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG / 'community_purity.png'}")

print(f"\n── Summary ──")
print(f"Total communities       : {n_communities}")
print(f"Communities >90% pure   : {(comm_df['purity'] > 0.9).sum()}")
print(f"Communities >50% left   : {(comm_df['dominant'] == 'left').sum()}")
print(f"Communities >50% right  : {(comm_df['dominant'] == 'right').sum()}")
print(f"Mean community purity   : {comm_df['purity'].mean():.3f}")
print(f"Largest community size  : {comm_df['size'].max()}")

print("\n✅ A5 complete.")