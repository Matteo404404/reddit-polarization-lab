"""
scripts/opinion_dynamics.py
Run and compare base vs diversity opinion dynamics scenarios.
"""
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from reddit_polarization.graphs.build_graphs import RedditGraphBundle
from reddit_polarization.experiments.feed_sim import run_opinion_dynamics

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
OUT_FIG = Path("results/figures")
OUT_LOG = Path("results/logs")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_LOG.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
bundle = RedditGraphBundle.load("data/processed/bundle.pkl")
G      = bundle.G_user

POL       = {"left", "right"}
pol_nodes = [n for n, d in G.nodes(data=True) if d.get("ideology_label") in POL]
H         = G.subgraph(pol_nodes).copy()
print(f"Political subgraph: {H.number_of_nodes():,} nodes, {H.number_of_edges():,} edges")

# ── Run both scenarios ────────────────────────────────────────────────────────
print("Running base scenario…")
df_base = run_opinion_dynamics(
    H, scenario="base",
    n_steps=200, k_neighbors=5, epsilon=0.5, mu=0.3,
    record_every=5, seed=42,
)

print("Running diversity scenario…")
df_div = run_opinion_dynamics(
    H, scenario="diversity",
    n_steps=200, k_neighbors=5, epsilon=0.5, mu=0.3,
    diversity_boost=3.0, record_every=5, seed=42,
)

results = pd.concat([df_base, df_div], ignore_index=True)
results.to_csv(OUT_LOG / "opinion_dynamics.csv", index=False)

# ── Plots ─────────────────────────────────────────────────────────────────────
METRICS = [
    ("group_distance",  "Group distance (right mean − left mean)",  True),
    ("variance",        "Opinion variance",                          True),
    ("extremity",       "Mean extremity (|b_i|)",                    True),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (col, title, high_is_bad) in zip(axes, METRICS):
    for label, color, ls in [
        ("base",      "#e15759", "-"),
        ("diversity", "#4e79a7", "--"),
    ]:
        sub = results[results["scenario"] == label]
        ax.plot(sub["step"], sub[col],
                color=color, linestyle=ls, linewidth=2,
                label=label.capitalize())

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("Simulation step", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

fig.suptitle(
    "Opinion Dynamics on Real Reddit Graph\n"
    "Base (red) vs Diversity feed boost (blue dashed)",
    fontsize=12, y=1.02,
)
fig.tight_layout()
fig.savefig(OUT_FIG / "opinion_dynamics.png", dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_FIG / 'opinion_dynamics.png'}")

# ── Final state summary ───────────────────────────────────────────────────────
print("\n── Final state (step 200) ──")
for scenario in ["base", "diversity"]:
    final = results[(results["scenario"] == scenario)].iloc[-1]
    print(f"\n  {scenario.upper()}")
    print(f"    group_distance : {final['group_distance']:.4f}")
    print(f"    variance       : {final['variance']:.4f}")
    print(f"    extremity      : {final['extremity']:.4f}")

print("\n✅ A3 complete.")