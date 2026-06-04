"""
graphs/build_graphs.py
Costruisce G_user, G_bip e RedditGraphBundle.
Supporta sia il formato classico (parent_id) sia il formato a 3 CSV.
"""
from __future__ import annotations
import logging, pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# ── Container ────────────────────────────────────────────────────────────────

@dataclass
class RedditGraphBundle:
    G_user: nx.DiGraph
    G_bip:  nx.Graph
    G_sim:  nx.Graph | None = None
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"G_user : {self.G_user.number_of_nodes()} nodes, "
            f"{self.G_user.number_of_edges()} edges\n"
            f"G_bip  : {self.G_bip.number_of_nodes()} nodes, "
            f"{self.G_bip.number_of_edges()} edges"
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Bundle salvato → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "RedditGraphBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Formato classico: da commenti con parent_id ──────────────────────────────

def build_user_graph(
    df: pd.DataFrame,
    *,
    weight_mode: Literal["count", "score"] = "count",
    directed: bool = True,
) -> nx.DiGraph:
    logger.info("Building G_user from comments…")
    id_to_author = dict(zip(df["comment_id"].astype(str), df["author"].astype(str)))
    G = nx.DiGraph() if directed else nx.Graph()

    for _, row in df.iterrows():
        i = str(row["author"])
        j = id_to_author.get(str(row["parent_id"]))
        if not j or i == j:
            continue
        w = 1 if weight_mode == "count" else max(int(row.get("score", 1)), 1)
        if G.has_edge(i, j):
            G[i][j]["weight"] += w
        else:
            G.add_edge(i, j, weight=w)

    logger.info("G_user: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def build_bipartite_graph(df: pd.DataFrame) -> nx.Graph:
    logger.info("Building G_bip…")
    G  = nx.Graph()
    ew = df.groupby(["author", "subreddit"]).size().reset_index(name="weight")
    for _, row in ew.iterrows():
        u, s, w = row["author"], row["subreddit"], row["weight"]
        if not G.has_node(u):
            G.add_node(u, bipartite="user")
        if not G.has_node(s):
            G.add_node(s, bipartite="subreddit")
        G.add_edge(u, s, weight=int(w))
    logger.info("G_bip: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def build_all(
    df: pd.DataFrame,
    user_labels: pd.DataFrame,
    *,
    weight_mode: Literal["count", "score"] = "count",
    metadata: dict | None = None,
) -> RedditGraphBundle:
    G_user = build_user_graph(df, weight_mode=weight_mode)
    G_bip  = build_bipartite_graph(df)
    label_map = dict(zip(user_labels["author"], user_labels["user_label"]))
    share_map = dict(zip(user_labels["author"], user_labels["political_share"]))
    nx.set_node_attributes(G_user, label_map, "ideology_label")
    nx.set_node_attributes(G_user, share_map, "political_share")
    return RedditGraphBundle(G_user=G_user, G_bip=G_bip, metadata=metadata or {})


# ── Formato 3-CSV: da user_relations direttamente ───────────────────────────

def build_user_graph_from_relations(
    relations: pd.DataFrame,
    user_labels: pd.DataFrame,
    *,
    weight_col: str = "interaction_count",
    sentiment_col: str = "sentiment_sum",
    min_interactions: int = 2,
    max_edges: int = 5_000_000,
    metadata: dict | None = None,
) -> RedditGraphBundle:
    """
    Builds G_user directly from user_relations.csv.
    Vectorised — no iterrows, safe on large datasets.
    """
    logger.info("Filtering relations (min_interactions=%d)…", min_interactions)

    # 1. Filter low-weight edges first (reduces RAM massively)
    if weight_col in relations.columns:
        rel = relations[relations[weight_col] >= min_interactions].copy()
    else:
        rel = relations.copy()
        rel[weight_col] = 1

    logger.info("Edges after min_interactions filter: %d", len(rel))

    # 2. Remove self-loops
    rel = rel[rel["source_author"] != rel["target_author"]]

    # 3. Cap to max_edges (take highest-weight edges)
    if len(rel) > max_edges:
        logger.warning("Capping to top %d edges by weight", max_edges)
        rel = rel.nlargest(max_edges, weight_col)

    # 4. Build graph vectorially via from_pandas_edgelist
    logger.info("Building DiGraph from %d edges…", len(rel))
    
    edge_attrs = [weight_col]
    if sentiment_col in rel.columns:
        edge_attrs.append(sentiment_col)

    G = nx.from_pandas_edgelist(
        rel,
        source="source_author",
        target="target_author",
        edge_attr=edge_attrs,
        create_using=nx.DiGraph(),
    )

    # Rename weight column to "weight" (networkx convention)
    if weight_col != "weight":
        nx.set_edge_attributes(
            G,
            {(u, v): d[weight_col] for u, v, d in G.edges(data=True)},
            "weight",
        )

    # 5. Attach ideology labels
    label_map = dict(zip(user_labels["author"], user_labels["user_label"]))
    share_map = dict(zip(user_labels["author"], user_labels["political_share"]))
    nx.set_node_attributes(G, label_map, "ideology_label")
    nx.set_node_attributes(G, share_map, "political_share")

    logger.info(
        "G_user: %d nodes, %d edges",
        G.number_of_nodes(), G.number_of_edges(),
    )

    return RedditGraphBundle(
        G_user=G,
        G_bip=nx.Graph(),
        metadata=metadata or {},
    )