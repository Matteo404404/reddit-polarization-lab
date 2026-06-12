# reddit-polarization-lab

> **Network-based measurement and visualization of political polarization on Reddit.**

A serious research prototype — not a toy. Built on real Reddit data, real interaction graphs, and validated structural polarization metrics. No fabricated networks, no synthetic opinions.

---

## What this is

`reddit-polarization-lab` is a self-contained Python research lab that:

- Ingests a large-scale open Reddit dump (12.67M submissions, 85.9M user relations)
- Constructs directed user–user interaction networks and user–subreddit affiliation graphs
- Assigns ideology labels via subreddit mapping (left / right / general / non-political)
- Computes structural polarization metrics: assortativity, modularity, echo index, cross-cutting ties
- Detects echo-chamber communities via the Louvain algorithm
- Identifies bridge users — the rare intermediaries between ideological clusters
- Runs counterfactual rewiring experiments (diversity vs homophily platform interventions)
- Simulates bounded-confidence opinion dynamics on the real graph

Everything runs on a laptop CPU. No paid APIs. No proprietary datasets. Only open-source Python.

---

## Key findings (quick summary)

| Metric | Value |
|---|---|
| Assortativity (r) | **0.759** — strong ideological homophily |
| Echo Index | **0.881** — 88.1% of interactions are within-group |
| Cross-cutting Fraction | **0.119** — only 11.9% of ties cross ideological lines |
| Modularity (Louvain) | **0.736** — near-maximal community separation |
| Community Purity (mean) | **0.993** — 98.3% of communities >90% ideologically pure |

**Central policy finding:** reducing polarization via diversity interventions requires **3.6× more structural effort** than the equivalent increase via homophilization. The network is significantly easier to polarize than to de-polarize.

---

## Repository structure

```
reddit-polarization-lab/
  src/reddit_polarization/
    __init__.py
    data/
      loader.py        # load raw/processed data, cache as parquet
      filters.py       # select subreddits, time windows, min activity
      labeling.py      # subreddit → ideology, user label assignment
    graphs/
      build_graphs.py  # construct user-user & bipartite graphs (NetworkX)
      attributes.py    # centrality, ideology scores, bridge metrics
    analysis/
      metrics.py       # assortativity, modularity, echo index, cross-cutting
      temporal.py      # metrics over time windows
    viz/
      plots.py         # time-series, histograms, bar charts
      net_viz.py       # graph drawings (NetworkX + matplotlib / pyvis)
    experiments/
      rewiring.py      # counterfactual edge rewiring (diversify / homophilize)
      feed_sim.py      # bounded-confidence opinion dynamics on real graph
  scripts/
    preprocess.py
    build_graphs.py
    compute_metrics.py
    run_rewiring.py
  notebooks/
    01_explore_data.ipynb        # data summary, subreddit distribution, user labels
    02_build_networks.ipynb      # graph construction, components, degree distribution
    03_polarization_metrics.ipynb # structural metrics, communities, bridge users, sentiment
    04_rewiring_experiments.ipynb # rewiring + opinion dynamics counterfactuals
  results/
    figures/                     # all generated plots (PNG)
    graphs/                      # pickled NetworkX objects
    logs/                        # metrics per time window (CSV)
  REPORT.md                      # full research-style write-up
  README.md
  pyproject.toml
```

---

## Dataset

This project uses an open Reddit dump available on Kaggle (search: *Reddit comments social network*).

Required schema:

| Table | Key columns |
|---|---|
| Submissions | `post_id`, `author`, `subreddit`, `score` |
| User Relations | `source_author`, `target_author`, `sentiment_sum`, `interaction_count` |
| Users | `username` |

Place raw files in `data/raw/`. Processed parquet tables go to `data/processed/`.

The dataset used in this project:

- **12,670,489** submissions across **197,870** subreddits
- **85,910,259** user interaction pairs
- **6,562,881** unique users

---

## Installation

```bash
git clone https://github.com/yourname/reddit-polarization-lab.git
cd reddit-polarization-lab
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Dependencies

```
pandas >= 2.0
numpy
scipy
networkx >= 3.0
matplotlib
seaborn
scikit-learn
python-louvain          # community detection
pyarrow                 # parquet I/O
jupyter
```

All open-source. No paid APIs required.

---

## Quickstart

### 1. Preprocess raw data

```bash
python scripts/preprocess.py --input data/raw/ --output data/processed/
```

### 2. Build graphs

```bash
python scripts/build_graphs.py --processed data/processed/ --output results/graphs/
```

### 3. Compute polarization metrics

```bash
python scripts/compute_metrics.py --graphs results/graphs/ --output results/logs/
```

### 4. Run rewiring experiment

```bash
python scripts/run_rewiring.py --graphs results/graphs/ --output results/figures/
```

### 5. Explore interactively

```bash
jupyter lab
# open notebooks/ in order: 01 → 02 → 03 → 04
```

---

## Notebooks

| Notebook | What you get |
|---|---|
| `01_explore_data` | Top subreddits, user activity distribution, ideology label breakdown, political vs non-political split |
| `02_build_networks` | Graph construction, connected components, degree distribution (linear + log-log), political subgraph, network visualization |
| `03_polarization_metrics` | All 5 structural metrics, edge breakdown donut, Louvain communities, community purity, bridge user scatter, sentiment analysis |
| `04_rewiring_experiments` | Rewiring 4-panel plot, asymmetry curve, opinion dynamics time series, policy interpretation |

---

## Metrics reference

| Metric | Definition | High value means |
|---|---|---|
| **Assortativity r** | Pearson correlation of ideology labels across edges (Newman 2003) | Strong homophily / echo chambers |
| **Echo Index** | Fraction of edges within the same ideology group | High enclosure |
| **Cross-cutting Fraction** | Fraction of edges bridging different ideology groups | Open, diverse network |
| **Modularity Q (partition)** | How well ideology labels predict community structure | Labels align with graph communities |
| **Modularity Q (Louvain)** | Maximum modularity achievable by greedy community detection | Strong community structure |
| **Community Purity** | Fraction of nodes in a community belonging to the majority ideology | Mono-ideological communities |
| **Bridge Score** | Combined betweenness centrality + cross-group neighbor share | User bridges ideological divide |

---

## Experiments

### Rewiring counterfactuals

Two platform intervention treatments applied at varying fractions \`f\`:

- **Diversify:** Add cross-group edges (models "recommend diverse content")
- **Homophilize:** Remove cross-group edges (models algorithmic drift toward homophily)

Result: de-polarization requires **3.6× more effort** than equivalent re-polarization.

### Opinion dynamics

Bounded-confidence model (BCM) on the real graph:
- Initial opinions: \`b_i ∈ {-1, +1}\` (ideology label)
- Confidence bound: \`ε = 0.5\`
- Neighbor sampling: base (edge weights) vs diversity (cross-group boost ×2)
- 200 simulation steps

Result: modular graph topology traps opinion clusters — diversity feed boosts produce small but consistent reductions in group distance and variance.

---

## Results summary

All output figures are saved to `results/figures/`. Key plots:

| File | Content |
|---|---|
| `nb03_metrics_summary.png` | Bar chart of all 5 structural metrics |
| `nb03_edge_breakdown.png` | Donut: Left→Left / Right→Right / Cross-group edge distribution |
| `nb03_communities.png` | Community purity histogram + top 10 ideology composition |
| `nb03_bridge_scatter.png` | Betweenness centrality vs cross-group exposure scatter |
| `nb03_sentiment_dist.png` | Sentiment distribution: within-group vs cross-group |
| `nb04_rewiring.png` | 4-panel rewiring experiment (assortativity, cross-cutting, echo index, modularity) |
| `nb04_asymmetry.png` | Asymmetry curve: diversify vs homophilize echo index |
| `nb04_opinion_dynamics.png` | Opinion dynamics: base vs diversity feed (3 metrics × 200 steps) |

---

## Theoretical background

This project builds on:

- **Newman (2003)** — assortativity coefficient for ideology-labeled networks
- **Blondel et al. (2008)** — Louvain modularity optimization
- **Deffuant et al. (2000)** — bounded-confidence opinion dynamics
- **Artime et al. (2025)** — empirical Reddit polarization across the 2016 US election
- **Barbera (2020)** — social media, echo chambers, and political polarization
- **Thurner et al. (2025)** — homophily and increased social interactions as polarization drivers

See `REPORT.md` for the full theoretical background, empirical findings, and complete references.

---

## Limitations

- Ideology labeling collapses multidimensional political space to a single left/right axis
- Political users are ~0.93% of total users; findings may not generalize to the full Reddit population
- Static graph snapshot — no temporal dynamics
- Bipartite user–subreddit graph (G_bip) not yet populated in current pipeline version
- BCM uses binary initial opinions; continuous text-based ideology scores planned as extension

---

## Roadmap

- [ ] Temporal slicing: track polarization metrics month by month
- [ ] Fix bipartite graph pipeline (G_bip)
- [ ] Text-based ideology scoring (classifier trained on partisan subreddits)
- [ ] Comment-level sentiment dynamics
- [ ] Cross-platform extension (Twitter/X, Mastodon)
- [ ] Natural experiment: structural effects of community bans (e.g. r/The_Donald)

---

## License

MIT

---

## Citation

If you use this project in your research:

```bibtex
@misc{melis2026reddit,
  author    = {Matteo Melis},
  title     = {reddit-polarization-lab: Network-based measurement and visualization of polarization on Reddit},
  year      = {2026},
  url       = {https://github.com/yourname/reddit-polarization-lab}
}
```
