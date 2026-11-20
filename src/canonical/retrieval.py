"""Canon retrieval: section-aware chunking, BM25, dense embeddings, and hybrid fusion.

Hybrid = reciprocal rank fusion (RRF) of the BM25 and dense rankings. We report Recall@5 and MRR
for all three on hand-labelled queries (scripts/eval_retrieval.py) instead of assuming hybrid wins.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from .config import ANIME_DIR
from .wiki import load_pages

CHUNK_WORDS = 140
OVERLAP = 30
_TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("""a an the and or of to in on for with by is are was were be been being it its this that
these those as at from his her their he she they them which who whom what when where how not no but if
then than so such can could would should will may might has have had do does did into out up down over
under again also only own same very just about after before during while because each other some""".split())


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def chunk_pages(anime: str) -> list[dict]:
    chunks = []
    for p in load_pages(anime):
        n = 0
        for sec in p["sections"]:
            words = sec["text"].split()
            step = CHUNK_WORDS - OVERLAP
            for s in range(0, max(len(words) - OVERLAP, 1), step):
                piece = words[s:s + CHUNK_WORDS]
                if len(piece) < 25 and s > 0:
                    break
                chunks.append({
                    "id": f"{anime}:{slug(p['title'])}:{n}",
                    "anime": anime,
                    "title": p["title"],
                    "heading": sec["heading"],
                    "url": p["url"],
                    "revid": p["revid"],
                    "text": " ".join(piece),
                })
                n += 1
    return chunks


def build_index(anime: str, embed: bool = True, log=print) -> dict:
    from . import llm

    d = ANIME_DIR / anime
    chunks = chunk_pages(anime)
    (d / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n")
    log(f"  [{anime}] {len(chunks)} chunks")
    if embed:
        texts = [f"{c['title']} - {c['heading']}: {c['text']}" for c in chunks]
        vecs = np.asarray(llm.embed(texts, "RETRIEVAL_DOCUMENT"), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        np.save(d / "emb.npy", vecs)
        log(f"  [{anime}] embeddings {vecs.shape}")
    Index.load.cache_clear()
    return {"chunks": len(chunks)}


class Index:
    def __init__(self, anime: str):
        d = ANIME_DIR / anime
        self.anime = anime
        self.chunks = [json.loads(l) for l in (d / "chunks.jsonl").read_text().splitlines() if l.strip()]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokens(f"{c['title']} {c['heading']} {c['text']}") for c in self.chunks])
        e = d / "emb.npy"
        self.emb = np.load(e) if e.exists() else None

    @staticmethod
    @lru_cache(maxsize=8)
    def load(anime: str) -> "Index":
        return Index(anime)

    def bm25_rank(self, query: str, k: int = 20) -> list[str]:
        s = self.bm25.get_scores(tokens(query))
        return [self.chunks[i]["id"] for i in np.argsort(-s)[:k]]

    def dense_rank(self, query: str, k: int = 20) -> list[str]:
        if self.emb is None:
            return []
        from . import llm
        q = np.asarray(llm.embed([query], "RETRIEVAL_QUERY")[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-9
        s = self.emb @ q
        return [self.chunks[i]["id"] for i in np.argsort(-s)[:k]]

    def search(self, query: str, k: int = 6, mode: str = "hybrid") -> list[dict]:
        if mode == "bm25" or self.emb is None:
            ids = self.bm25_rank(query, k)
        elif mode == "dense":
            ids = self.dense_rank(query, k)
        else:
            ids = rrf([self.bm25_rank(query, 30), self.dense_rank(query, 30)])[:k]
        return [self.by_id[i] for i in ids]


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    score: dict[str, float] = {}
    for r in rankings:
        for pos, cid in enumerate(r):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + pos + 1)
    return sorted(score, key=lambda c: -score[c])
