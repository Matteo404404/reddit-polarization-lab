"""
scripts/preprocess_3csv.py
--------------------------
Pipeline completa per il formato a 3 CSV (submissions / user_relations / users).

Uso:
  python scripts/preprocess_3csv.py \
      --raw   data/raw/ \
      --out   data/processed/ \
      --labels subreddit_labels.yaml
"""
import argparse, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from reddit_polarization.data.loader   import load_three_csv, build_activity_table
from reddit_polarization.data.labeling import load_subreddit_map, label_comments, compute_user_ideology
from reddit_polarization.graphs.build_graphs import build_user_graph_from_relations, RedditGraphBundle

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",    default="data/raw/")
    ap.add_argument("--out",    default="data/processed/")
    ap.add_argument("--labels", default="subreddit_labels.yaml")
    ap.add_argument("--min-posts", type=int, default=2,
                    help="min post per utente per essere incluso")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Carica i 3 CSV
    dfs = load_three_csv(args.raw)

    # 2. Tabella attività (author × subreddit da submissions)
    activity = build_activity_table(dfs)

    # 3. Carica mappa ideologica e labella
    sub_map  = load_subreddit_map(args.labels)
    labelled = label_comments(activity, sub_map, subreddit_col="subreddit")
    labelled.to_parquet(out / "activity_labelled.parquet", index=False)

    # 4. User ideology labels
    user_labels = compute_user_ideology(labelled, author_col="author")
    user_labels.to_parquet(out / "user_labels.parquet", index=False)

    # 5. Costruisci grafo da user_relations
    relations = dfs["user_relations"]
    bundle    = build_user_graph_from_relations(relations, user_labels, min_interactions=2,   max_edges=3_000_000)
    bundle.save(out / "bundle.pkl")

    # ── Summary ──
    print("\n" + "─"*50)
    print(f"  Submissions   : {len(dfs['submissions']):,}")
    print(f"  User relations: {len(relations):,}")
    print(f"  Utenti labellati: {len(user_labels):,}")
    print()
    print("  User label distribution:")
    print(user_labels["user_label"].value_counts().to_string())
    print()
    print(bundle.summary())
    print("─"*50)
    print(f"\n✅ Tutto salvato in {out}")


if __name__ == "__main__":
    main()