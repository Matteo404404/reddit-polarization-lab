"""
scripts/rewiring_experiment.py
Counterfactual edge rewiring on the political subgraph.
Tests how assortativity, echo index, and modularity change
when cross-group edges are added (diversify) or removed (homophilize).
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from reddit_polarization.graphs.build_graphs import RedditGraphBundle
from reddit_polarization.experiments.rewiring import run_rewiring_experiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
OUT_FIG = Path("results/figures")
OUT_LOG = Path("results/logs")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_LOG.mkdir(parents=True, exist_ok=True)

# ── Load political subgraph ───────────────────────────────────────────────────
bundle = RedditGraphBundle.load("data/processed/bundle.pkl")
G      = bundle.G_user

POL    = {"left", "right"}
pol_nodes = [n for n, d in G.nodes(data=True) if d.get("ideology_label") in POL]
H = G.subgraph(pol_nodes).copy()
print(f"Political subgraph: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

# ── Run experiment ────────────────────────────────────────────────────────────
print("Running rewiring experiment…")
results = run_rewiring_experiment(
    H,
    diversify_fractions   =[0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
    homophilize_fractions =[0.10, 0.20, 0.30, 0.50, 0.70],
    seed=42,
)

results.to_csv(OUT_LOG / "rewiring_results.csv", index=False)
print("\nRewiring results:")
print(results[["treatment","fraction","assortativity",
               "cross_cutting_frac","echo_index",
               "modularity_partition"]].to_string())

# ── Plotting ──────────────────────────────────────────────────────────────────
METRICS = [
    ("assortativity",        "Assortativity (r)",          True),
    ("cross_cutting_frac",   "Cross-cutting fraction",     False),
    ("echo_index",           "Echo index",                 True),
    ("modularity_partition", "Modularity (partition)",     True),
]

baseline = results[results["treatment"] == "baseline"].iloc[0]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes = axes.flatten()

for ax, (col, title, high_is_bad) in zip(axes, METRICS):
    div  = results[results["treatment"] == "diversify"].sort_values("fraction")
    hom  = results[results["treatment"] == "homophilize"].sort_values("fraction")
    base_val = float(baseline[col])

    # Baseline as horizontal dashed line
    ax.axhline(base_val, color="gray", linestyle="--", linewidth=1.2,
               label=f"Baseline ({base_val:.3f})")

    # Diversify arm — should reduce polarization
    ax.plot(div["fraction"], div[col],
            color="#4e79a7", marker="o", linewidth=2, markersize=6,
            label="Diversify (add cross-group edges)")

    # Homophilize arm — should increase polarization
    ax.plot(hom["fraction"], hom[col],
            color="#e15759", marker="s", linewidth=2, markersize=6,
            label="Homophilize (remove cross-group edges)")

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Intervention fraction", fontsize=9)
    ax.set_ylabel(title, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top","right"]].set_visible(False)

    # Shade the "bad" zone
    y_top = ax.get_ylim()[1]
    y_bot = ax.get_ylim()[0]
    if high_is_bad:
        ax.axhspan(base_val, y_top, alpha=0.04, color="#e15759")
        ax.axhspan(y_bot, base_val, alpha=0.04, color="#4e79a7")
    else:
        ax.axhspan(base_val, y_top, alpha=0.04, color="#4e79a7")
        ax.axhspan(y_bot, base_val, alpha=0.04, color="#e15759")

fig.suptitle("Rewiring Experiment — Effect of Platform Interventions on Polarization",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(OUT_FIG / "rewiring_experiment.png", dpi=150, bbox_inches="tight")
print(f"\nSaved → {OUT_FIG / 'rewiring_experiment.png'}")

# ── Delta table: % change from baseline ──────────────────────────────────────
print("\n── Effect size (% change from baseline) ──")
for _, row in results[results["treatment"] != "baseline"].iterrows():
    delta_assort = (row["assortativity"] - baseline["assortativity"]) / baseline["assortativity"] * 100
    delta_echo   = (row["echo_index"]    - baseline["echo_index"])    / baseline["echo_index"]    * 100
    print(f"  {row['treatment']:15s} frac={row['fraction']:.2f}  "
          f"Δassortativity={delta_assort:+.1f}%  Δecho={delta_echo:+.1f}%")

print("\n✅ A2 complete.")