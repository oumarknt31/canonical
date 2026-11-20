"""End-to-end: (anime, topic, condition) -> lesson + mapping + checks. Everything is logged and
cached, so the demo can replay showcase pairs with the API switched off."""
from __future__ import annotations

import time

from . import generate as G
from . import llm, store
from .config import load_card
from .features import compute
from .ranker import score


def make_candidates(anime: str, topic: str, round_: int = 0, with_embed: bool = True) -> list[dict]:
    """Generate, featurise and store five candidates for one pair (idempotent via the LLM cache)."""
    card = load_card(topic)
    cands, chunks = G.candidates(anime, card, round_=round_)
    out = []
    for i, c in enumerate(cands):
        f = compute(c, card, chunks, embed=(lambda t: llm.embed(t, "SEMANTIC_SIMILARITY")) if with_embed else None)
        cid = f"{anime}|{topic}|r{round_}|{i}"
        store.save_candidate(cid, anime, topic, round_, i, {"cand": c, "chunk_ids": [x["id"] for x in chunks]}, f)
        out.append({"id": cid, "cand": c, "features": f, "chunks": chunks})
    return out


def teach(anime: str, topic: str, condition: str = "D", sample: int = 0, log=lambda *_: None) -> dict:
    card = load_card(topic)
    t0 = time.time()
    res: dict = {"anime": anime, "topic": topic, "condition": condition, "sample": sample}
    if condition == "A":
        les = G.lesson_plain(anime, card, sample=sample)
        res.update(markdown=les["markdown"], chunk_ids=[])
    elif condition == "B":
        les = G.lesson_rag(anime, card, sample=sample)
        res.update(markdown=les["markdown"], chunk_ids=les["chunks"])
    else:
        log("Proposing 5 structured analogies from canon...")
        cands = make_candidates(anime, topic, round_=sample)
        if not cands:
            raise llm.LLMError("no candidates returned")
        for c in cands:
            c["score"] = score(c["features"])
        cands.sort(key=lambda c: -c["score"])
        best = cands[0]
        log(f"Ranked: best is '{best['cand'].get('element')}' (score {best['score']:.2f})")
        md = G.lesson_from_candidate(anime, card, best["cand"], best["chunks"], sample=sample)
        res.update(markdown=md, chunk_ids=[c["id"] for c in best["chunks"]], chosen=best["cand"],
                   ranking=[{"id": c["id"], "element": c["cand"].get("element"), "score": c["score"],
                             "features": c["features"]} for c in cands])
    if condition in ("A", "B", "C", "D"):
        log("Fact-checking every claim against canon and the CS card...")
        chk = G.check(anime, card, res["markdown"], sample=sample)
        if condition == "D" and chk["canon_unsupported"] + chk["cs_contradicted"] > 0:
            bad = [c for c in chk["claims"] if (c["type"] == "canon" and c["verdict"] != "supported")
                   or c["verdict"] == "contradicted"]
            fb = "\n".join(f"- {c['text']} ({c['verdict']}: {c.get('note', '')})" for c in bad)
            log(f"Gate: {len(bad)} claim(s) rejected, regenerating once...")
            md2 = G.lesson_from_candidate(anime, card, res["chosen"], cands[0]["chunks"], sample=sample, feedback=fb)
            chk2 = G.check(anime, card, md2, sample=sample + 100)
            res["first_draft"] = {"markdown": res["markdown"], "check": chk}
            res["markdown"], chk = md2, chk2
        res["check"] = chk
    res["seconds"] = round(time.time() - t0, 1)
    store.save_lesson(f"{anime}|{topic}|{condition}|{sample}", anime, topic, condition, sample, res)
    return res
