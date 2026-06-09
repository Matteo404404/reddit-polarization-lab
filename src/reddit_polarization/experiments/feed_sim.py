"""
experiments/feed_sim.py
Bounded-confidence opinion dynamics on the real G_user political subgraph.
Each user has an ideology score b_i in [-1, 1]:
  left  → initialised around -0.7
  right → initialised around +0.7

At each step, each user samples k neighbours.
If |b_i - b_j| < epsilon, they move toward each other (averaging).
Otherwise, opinions are unchanged (bounded confidence, Deffuant-Weisbuch variant).

Two scenarios:
  - base      : neighbour sampling proportional to observed edge weights
  - diversity : cross-group neighbours sampled 3x more likely (diversity boost)
"""
from __future__ import annotations
import logging
import random
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _init_opinions(
    G: nx.DiGraph,
    *,
    left_mean:  float = -0.7,
    right_mean: float = +0.7,
    std:        float = 0.15,
    seed:       int   = 42,
) -> dict[str, float]:
    """Initialise ideology score b_i in [-1, 1] for each political node."""
    rng = np.random.default_rng(seed)
    opinions = {}
    for node, data in G.nodes(data=True):
        label = data.get("ideology_label")
        if label == "left":
            b = rng.normal(left_mean, std)
        elif label == "right":
            b = rng.normal(right_mean, std)
        else:
            b = rng.uniform(-0.2, 0.2)
        opinions[node] = float(np.clip(b, -1.0, 1.0))
    return opinions


def _polarization_metrics(opinions: dict[str, float]) -> dict[str, float]:
    vals = np.array(list(opinions.values()))
    left_mean  = vals[vals < 0].mean() if (vals < 0).any() else 0.0
    right_mean = vals[vals > 0].mean() if (vals > 0).any() else 0.0
    return {
        "variance":     float(np.var(vals)),
        "group_distance": float(right_mean - left_mean),
        "extremity":    float(np.mean(np.abs(vals))),
        "bimodality":   float(np.mean(vals ** 2) - np.mean(np.abs(vals)) ** 2),
    }


def run_opinion_dynamics(
    G: nx.DiGraph,
    *,
    scenario: Literal["base", "diversity"] = "base",
    n_steps: int = 100,
    k_neighbors: int = 5,
    epsilon: float = 0.5,
    mu: float = 0.3,
    diversity_boost: float = 3.0,
    seed: int = 42,
    record_every: int = 5,
) -> pd.DataFrame:
    """
    Run Deffuant-Weisbuch bounded-confidence dynamics.

    Parameters
    ----------
    G               : political subgraph with ideology_label node attributes
    scenario        : 'base' uses edge weights; 'diversity' boosts cross-group sampling
    n_steps         : number of simulation steps
    k_neighbors     : neighbours sampled per user per step
    epsilon         : confidence bound — interact only if |b_i - b_j| < epsilon
    mu              : convergence rate (0 < mu <= 0.5)
    diversity_boost : multiplier on cross-group edge weights in diversity scenario
    record_every    : record metrics every N steps
    seed            : random seed
    """
    rng = random.Random(seed)
    opinions = _init_opinions(G, seed=seed)
    nodes    = list(G.nodes())

    records = []

    for step in range(n_steps + 1):
        # Record metrics
        if step % record_every == 0:
            m = _polarization_metrics(opinions)
            m["step"]     = step
            m["scenario"] = scenario
            records.append(m)

        if step == n_steps:
            break

        # Shuffle update order
        rng.shuffle(nodes)

        for node in nodes:
            # Get outgoing + incoming neighbours
            nbrs = list(G.predecessors(node)) + list(G.successors(node))
            if not nbrs:
                continue

            # Build sampling weights
            if scenario == "base":
                weights = []
                for n in nbrs:
                    w_out = G[node][n]["weight"] if G.has_edge(node, n) else 0
                    w_in  = G[n][node]["weight"] if G.has_edge(n, node) else 0
                    weights.append(max(w_out + w_in, 1))
            else:  # diversity: boost cross-group neighbours
                my_label = G.nodes[node].get("ideology_label")
                weights  = []
                for n in nbrs:
                    w_out = G[node][n]["weight"] if G.has_edge(node, n) else 0
                    w_in  = G[n][node]["weight"] if G.has_edge(n, node) else 0
                    base_w = max(w_out + w_in, 1)
                    if G.nodes[n].get("ideology_label") != my_label:
                        base_w *= diversity_boost
                    weights.append(base_w)

            # Sample k neighbours (with replacement)
            total_w = sum(weights)
            probs   = [w / total_w for w in weights]
            k       = min(k_neighbors, len(nbrs))
            sampled = rng.choices(nbrs, weights=probs, k=k)

            # Deffuant update
            for nbr in sampled:
                if abs(opinions[node] - opinions[nbr]) < epsilon:
                    opinions[node] += mu * (opinions[nbr] - opinions[node])
                    opinions[node]  = max(-1.0, min(1.0, opinions[node]))

    df = pd.DataFrame(records)
    logger.info("Simulation '%s' complete: %d steps", scenario, n_steps)
    return df