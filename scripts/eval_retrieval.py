"""Retrieval evaluation: Recall@5 and MRR for BM25 vs dense vs hybrid on hand-labelled queries.

data/retrieval_queries.csv columns: anime,query,relevant_titles   (titles separated by '|')
A hit = a retrieved chunk from one of the relevant pages.
"""
import json

import pandas as pd

import _path  # noqa: F401
from canonical.config import DATA, RESULTS_DIR
from canonical.retrieval import Index

q = pd.read_csv(DATA / "retrieval_queries.csv")
res = {}
have_dense = all(Index.load(a).emb is not None for a in q["anime"].unique())
if not have_dense:
    print("(no embeddings yet - run scripts/build_index.py with a key to evaluate dense and hybrid)")
for mode in (["bm25", "dense", "hybrid"] if have_dense else ["bm25"]):
    rec, mrr = [], []
    for _, r in q.iterrows():
        rel = set(t.strip() for t in str(r["relevant_titles"]).split("|"))
        hits = [c["title"] in rel for c in Index.load(r["anime"]).search(r["query"], k=10, mode=mode)]
        rec.append(float(any(hits[:5])))
        mrr.append(next((1.0 / (i + 1) for i, h in enumerate(hits) if h), 0.0))
    res[mode] = {"recall@5": sum(rec) / len(rec), "mrr": sum(mrr) / len(mrr), "n": len(rec)}
    print(f"{mode:7} Recall@5 {res[mode]['recall@5']:.3f}  MRR {res[mode]['mrr']:.3f}  (n={len(rec)})")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "retrieval_eval.json").write_text(json.dumps(res, indent=1))
