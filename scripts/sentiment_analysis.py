"""
scripts/sentiment_analysis.py
A4: Analyse sentiment_sum on edges of the political subgraph.
Key question: are cross-group interactions more hostile than within-group?
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from scipy import stats

from reddit_polarization.graphs.build_graphs import RedditGraphBundle

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
OUT_FIG = Path("results/figures")
OUT_LOG = Path("results/logs")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_LOG.mkdir(parents=True, exist_ok=True)

COLORS = {"left": "#4e79a7", "right": "#e15759",
          "cross": "#f28e2b", "within": "#76b7b2"}
POL = {"left", "right"}

# ── Load ──────────────────────────────────────────────────────────────────────
bundle    = RedditGraphBundle.load("data/processed/bundle.pkl")
G         = bundle.G_user
pol_nodes = [n for n, d in G.nodes(data=True) if d.get("ideology_label") in POL]
H         = G.subgraph(pol_nodes).copy()
print(f"Political subgraph: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

# ── Build edge table ──────────────────────────────────────────────────────────
rows = []
for u, v, data in H.edges(data=True):
    label_u = H.nodes[u].get("ideology_label", "unknown")
    label_v = H.nodes[v].get("ideology_label", "unknown")
    sentiment = data.get("sentiment", data.get("sentiment_sum", np.nan))
    weight    = data.get("weight", 1)

    if label_u not in POL or label_v not in POL:
        continue

    edge_type = "cross" if label_u != label_v else "within"
    pair      = f"{label_u}→{label_v}"

    rows.append({
        "source":       u,
        "target":       v,
        "label_source": label_u,
        "label_target": label_v,
        "edge_type":    edge_type,
        "pair":         pair,
        "sentiment":    float(sentiment) if sentiment == sentiment else np.nan,
        "weight":       int(weight),
        "sentiment_per_interaction": (
            float(sentiment) / int(weight)
            if (sentiment == sentiment and weight > 0)
            else np.nan
        ),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_LOG / "edge_sentiment.csv", index=False)
print(f"Edge table: {len(df):,} rows, "
      f"{df['sentiment'].isna().sum():,} missing sentiment")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n── Sentiment by edge type ──")
summary = (
    df.groupby("edge_type")["sentiment_per_interaction"]
    .agg(["count", "mean", "median", "std"])
    .round(4)
)
print(summary)

print("\n── Sentiment by pair direction ──")
pair_summary = (
    df.groupby("pair")["sentiment_per_interaction"]
    .agg(["count", "mean", "median"])
    .round(4)
)
print(pair_summary)

# ── Statistical test: cross vs within ────────────────────────────────────────
cross_s  = df[df["edge_type"] == "cross"]["sentiment_per_interaction"].dropna()
within_s = df[df["edge_type"] == "within"]["sentiment_per_interaction"].dropna()

if len(cross_s) > 10 and len(within_s) > 10:
    t_stat, p_val = stats.mannwhitneyu(cross_s, within_s, alternative="two-sided")
    print(f"\nMann-Whitney U test (cross vs within sentiment):")
    print(f"  U={t_stat:.1f}, p={p_val:.4f} "
          f"({'significant' if p_val < 0.05 else 'not significant'} at α=0.05)")
    print(f"  Cross mean:  {cross_s.mean():.4f} ± {cross_s.std():.4f}")
    print(f"  Within mean: {within_s.mean():.4f} ± {within_s.std():.4f}")


# ── Plot 1: Boxplot cross vs within sentiment ─────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 6))

data_plot = [
    within_s.clip(-5, 5).values,
    cross_s.clip(-5, 5).values,
]
bp = ax1.boxplot(
    data_plot,
    labels=["Within-group\n(left→left, right→right)", "Cross-group\n(left→right, right→left)"],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    flierprops=dict(marker=".", markersize=2, alpha=0.3),
    widths=0.45,
)
bp["boxes"][0].set_facecolor(COLORS["within"])
bp["boxes"][1].set_facecolor(COLORS["cross"])

ax1.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax1.set_ylabel("Sentiment per interaction (clipped ±5)", fontsize=11)
ax1.set_title("Sentiment Distribution: Cross-group vs Within-group Interactions",
              fontsize=12, pad=12)
ax1.spines[["top","right"]].set_visible(False)
fig1.tight_layout()
fig1.savefig(OUT_FIG / "sentiment_boxplot.png", dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FIG / 'sentiment_boxplot.png'}")


# ── Plot 2: Sentiment heatmap by pair direction ───────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

for ax, metric, label in zip(
    axes2,
    ["mean", "median"],
    ["Mean sentiment per interaction", "Median sentiment per interaction"],
):
    pivot = (
        df.groupby(["label_source", "label_target"])["sentiment_per_interaction"]
        .agg(metric)
        .unstack()
    )
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([c.capitalize() for c in pivot.columns], fontsize=11)
    ax.set_yticklabels([i.capitalize() for i in pivot.index], fontsize=11)
    ax.set_xlabel("Target user ideology", fontsize=10)
    ax.set_ylabel("Source user ideology", fontsize=10)
    ax.set_title(label, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if abs(val) > 0.5 else "#222222")

fig2.suptitle("Sentiment Heatmap: Ideology of Source → Target",
              fontsize=13, y=1.02)
fig2.tight_layout()
fig2.savefig(OUT_FIG / "sentiment_heatmap.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG / 'sentiment_heatmap.png'}")


# ── Plot 3: Sentiment vs interaction count (cross vs within) ──────────────────
fig3, ax3 = plt.subplots(figsize=(9, 6))

for etype, color in [("within", COLORS["within"]), ("cross", COLORS["cross"])]:
    sub = df[df["edge_type"] == etype].dropna(subset=["sentiment_per_interaction"])
    ax3.scatter(
        sub["weight"].clip(upper=50),
        sub["sentiment_per_interaction"].clip(-3, 3),
        c=color, alpha=0.25, s=10, label=etype.capitalize(),
    )

ax3.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax3.set_xlabel("Interaction count (weight, clipped at 50)", fontsize=11)
ax3.set_ylabel("Sentiment per interaction (clipped ±3)", fontsize=11)
ax3.set_title("Sentiment vs Interaction Frequency\nCross-group vs Within-group",
              fontsize=12, pad=10)
ax3.legend(frameon=False, fontsize=10)
ax3.spines[["top","right"]].set_visible(False)
fig3.tight_layout()
fig3.savefig(OUT_FIG / "sentiment_vs_weight.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG / 'sentiment_vs_weight.png'}")

print("\n✅ A4 complete.")