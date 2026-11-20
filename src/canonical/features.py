"""Hand-designed, explainable features of an analogy candidate (structure-mapping inspired).

Each feature is something a judge can understand from its name. Two are kept aside as BASELINES
the model must beat: `topic_sim` (embedding similarity only) and `self_score` (the LLM grading itself).
"""
from __future__ import annotations

import math

import numpy as np

from .retrieval import tokens

HOLDS = {"yes": 1.0, "partial": 0.5, "no": 0.0}

FEATURES = [
    "role_coverage",      # share of the card's roles that have an anime counterpart
    "relation_holds",     # mean of yes=1 / partial=.5 / no=0 over the card's relations
    "relation_explained", # share of relations with a real (>= 8 word) explanation
    "citation_valid",     # share of mapping entries citing at least one retrieved passage id
    "claim_support_lex",  # mean share of a canon claim's words found in its cited passages
    "claim_support_sem",  # mean max cosine(claim, cited passage)
    "misconception_sim",  # max cosine(analogy text, a known misconception) - higher is worse
    "breaks_words",       # log(1 + words in the 'where it breaks' statement)
    "n_claims",           # how many canon facts the analogy leans on
]
BASELINES = ["topic_sim", "self_score"]


def _cos(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def analogy_text(cand: dict) -> str:
    parts = [cand.get("summary", "")]
    parts += [f"{r.get('anime')}" for r in cand.get("role_map", []) if r.get("anime")]
    parts += [r.get("explanation", "") for r in cand.get("relation_map", [])]
    return " ".join(p for p in parts if p)


def compute(cand: dict, card: dict, chunks: list[dict], embed=None) -> dict:
    """`embed(texts)->vectors`; pass None to skip the semantic features (they become NaN)."""
    by_id = {c["id"]: c for c in chunks}
    role_ids = [r["id"] for r in card["roles"]]
    rel_ids = [r["id"] for r in card["relations"]]
    rmap = {r.get("role_id"): r for r in cand.get("role_map", [])}
    lmap = {r.get("relation_id"): r for r in cand.get("relation_map", [])}

    role_cov = sum(1 for r in role_ids if (rmap.get(r) or {}).get("anime")) / len(role_ids)
    rel_holds = sum(HOLDS.get(str((lmap.get(r) or {}).get("holds", "no")).lower(), 0.0) for r in rel_ids) / len(rel_ids)
    rel_expl = sum(1 for r in rel_ids if len(str((lmap.get(r) or {}).get("explanation", "")).split()) >= 8) / len(rel_ids)

    entries = [e for e in cand.get("role_map", []) if e.get("anime")] + cand.get("relation_map", []) \
        + cand.get("canon_claims", [])
    cite_ok = [any(cid in by_id for cid in e.get("chunk_ids", []) or []) for e in entries]
    citation_valid = sum(cite_ok) / len(cite_ok) if cite_ok else 0.0

    claims = cand.get("canon_claims", []) or []
    lex = []
    for cl in claims:
        toks = set(tokens(cl.get("claim", "")))
        text = " ".join(by_id[c]["text"] for c in cl.get("chunk_ids", []) or [] if c in by_id)
        have = set(tokens(text))
        lex.append(len(toks & have) / len(toks) if toks and have else 0.0)
    claim_lex = float(np.mean(lex)) if lex else 0.0

    f = {
        "role_coverage": role_cov,
        "relation_holds": rel_holds,
        "relation_explained": rel_expl,
        "citation_valid": citation_valid,
        "claim_support_lex": claim_lex,
        "claim_support_sem": math.nan,
        "misconception_sim": math.nan,
        "breaks_words": math.log1p(len(str(cand.get("breaks", "")).split())),
        "n_claims": float(len(claims)),
        "topic_sim": math.nan,
        "self_score": float(cand.get("self_score") or 0),
    }
    if embed is not None:
        atext = analogy_text(cand)
        mis = [m["wrong"] for m in card["misconceptions"]]
        claim_texts = [cl.get("claim", "") for cl in claims]
        cited = sorted({c for cl in claims for c in (cl.get("chunk_ids") or []) if c in by_id})
        vecs = embed([card["definition"], atext, *mis, *claim_texts, *[by_id[c]["text"] for c in cited]])
        dvec, avec = vecs[0], vecs[1]
        mvecs = vecs[2:2 + len(mis)]
        cvecs = vecs[2 + len(mis):2 + len(mis) + len(claim_texts)]
        pvecs = dict(zip(cited, vecs[2 + len(mis) + len(claim_texts):]))
        f["topic_sim"] = _cos(dvec, avec)
        f["misconception_sim"] = max((_cos(avec, m) for m in mvecs), default=0.0)
        sem = []
        for cl, cv in zip(claims, cvecs):
            ps = [pvecs[c] for c in (cl.get("chunk_ids") or []) if c in pvecs]
            sem.append(max((_cos(cv, p) for p in ps), default=0.0))
        f["claim_support_sem"] = float(np.mean(sem)) if sem else 0.0
    return f


def heuristic_score(f: dict) -> float:
    """Ranking used before the model is trained (and as a third baseline)."""
    return (0.35 * f["relation_holds"] + 0.25 * f["role_coverage"] + 0.2 * f["citation_valid"]
            + 0.1 * f["claim_support_lex"] + 0.1 * min(f["breaks_words"] / 3.0, 1.0))
