"""LLM steps. The LLM proposes and writes; it never grades itself in the final system.

Conditions (the ablation):
  A  plain prompt: "teach <topic> through <anime>" (what every existing tool does)
  B  + canon retrieval, lesson must cite chunk ids
  C  B + CS concept card + 5 structured candidates ranked by the analogy-quality model
  D  C + faithfulness gate (unsupported claims are regenerated once, then flagged)
"""
from __future__ import annotations

import json

from . import llm
from .config import ANIME
from .retrieval import Index

TUTOR = ("You are a precise computer science tutor for first-year CS students who love anime. "
         "You never invent anime facts. Accuracy beats flourish.")


def _chunks_block(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['id']}] ({c['title']} / {c['heading']})\n{c['text']}" for c in chunks)


def _card_block(card: dict) -> str:
    roles = "\n".join(f"- {r['id']}: {r['name']} - {r['desc']}" for r in card["roles"])
    rels = "\n".join(f"- {r['id']}: {r['desc']}" for r in card["relations"])
    mis = "\n".join(f"- {m['wrong']} (correct: {m['correct']})" for m in card["misconceptions"])
    return (f"CS CONCEPT: {card['name']}\nDefinition: {card['definition']}\n\nRoles an analogy must map:\n{roles}"
            f"\n\nStructural relations an analogy must preserve:\n{rels}\n\nKnown misconceptions to avoid:\n{mis}")


# ---------------------------------------------------------------- brainstorm + retrieval

def brainstorm(anime: str, card: dict, *, round_: int = 0) -> list[dict]:
    """Ask for 5 distinct story elements (abilities, rules, events, relationships) that could
    model the concept, each with a wiki search query. Used to retrieve canon for B, C, D."""
    name = ANIME[anime]["name"]
    prompt = (f"{_card_block(card)}\n\nList 5 DIFFERENT elements from the anime {name} (abilities, rules, "
              f"systems, events or relationships) whose STRUCTURE could model this concept. Prefer elements with "
              f"explicit rules. Return JSON: {{\"elements\": [{{\"element\": str, \"why\": str (one sentence), "
              f"\"search\": str (a short wiki search query naming the technique/character/event)}}]}}")
    out = llm.generate_json(prompt, TUTOR, temperature=0.9, sample=round_)
    return out.get("elements", [])[:5]


def retrieve_for(anime: str, queries: list[str], per_query: int = 4, cap: int = 24) -> list[dict]:
    idx = Index.load(anime)
    seen, out = set(), []
    for q in queries:
        for c in idx.search(q, k=per_query):
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
    return out[:cap]


# ---------------------------------------------------------------- condition A / B lessons

LESSON_SPEC = ("Write a lesson of 300-400 words in Markdown: a hook, the concept explained through the anime, "
               "a worked example, and one sentence on where the analogy breaks. ")


def lesson_plain(anime: str, card: dict, *, sample: int = 0) -> dict:
    name = ANIME[anime]["name"]
    prompt = f"Teach the computer science concept '{card['name']}' to a CS1 student through the anime {name}. " \
             f"Use its characters and abilities as the analogy. {LESSON_SPEC}"
    text = llm.generate(prompt, TUTOR, temperature=0.7, sample=sample)
    return {"condition": "A", "markdown": text, "chunks": []}


def lesson_rag(anime: str, card: dict, *, sample: int = 0) -> dict:
    name = ANIME[anime]["name"]
    elems = brainstorm(anime, card, round_=sample)
    chunks = retrieve_for(anime, [e["search"] for e in elems])
    prompt = (f"CANON PASSAGES from the {name} wiki:\n\n{_chunks_block(chunks)}\n\n"
              f"Teach the computer science concept '{card['name']}' to a CS1 student through {name}. {LESSON_SPEC}"
              f"Every statement about the anime must be supported by a passage above; put the passage id in square "
              f"brackets right after it, e.g. [{chunks[0]['id'] if chunks else 'anime:page:0'}].")
    text = llm.generate(prompt, TUTOR, temperature=0.7, sample=sample)
    return {"condition": "B", "markdown": text, "chunks": [c["id"] for c in chunks]}


# ---------------------------------------------------------------- condition C candidates

CAND_SCHEMA = """{"candidates": [{
  "element": str,                      // the anime element
  "summary": str,                      // one-sentence pitch of the analogy
  "role_map": [{"role_id": str, "anime": str or null, "chunk_ids": [str]}],        // one entry per CS role; null if nothing maps
  "relation_map": [{"relation_id": str, "holds": "yes"|"partial"|"no", "explanation": str, "chunk_ids": [str]}],
  "breaks": str,                       // where the analogy fails, concretely
  "canon_claims": [{"claim": str, "chunk_ids": [str]}],                             // every anime fact the analogy relies on
  "self_score": int                    // 1-5, your own rating of this analogy
}]}"""


def candidates(anime: str, card: dict, *, round_: int = 0) -> tuple[list[dict], list[dict]]:
    """Five structured analogy candidates with explicit role/relation mappings and citations."""
    name = ANIME[anime]["name"]
    elems = brainstorm(anime, card, round_=round_)
    chunks = retrieve_for(anime, [e["search"] for e in elems], per_query=5, cap=28)
    listing = "\n".join(f"- {e['element']}: {e['why']}" for e in elems)
    prompt = (f"{_card_block(card)}\n\nCANON PASSAGES from the {name} wiki:\n\n{_chunks_block(chunks)}\n\n"
              f"Candidate elements:\n{listing}\n\nFor EACH candidate element, build a structured analogy for "
              f"{card['name']}. Map every role and judge every relation honestly: if a relation does not hold in "
              f"canon, say 'no'. Only cite passage ids shown above. Do not use facts that are not in the passages "
              f"as canon_claims. Return JSON exactly in this shape:\n{CAND_SCHEMA}")
    out = llm.generate_json(prompt, TUTOR, temperature=0.8, sample=round_)
    cands = out.get("candidates", [])[:5]
    return cands, chunks


def lesson_from_candidate(anime: str, card: dict, cand: dict, chunks: list[dict], *, sample: int = 0,
                          feedback: str = "") -> str:
    name = ANIME[anime]["name"]
    cited = {cid for e in cand.get("role_map", []) + cand.get("relation_map", []) + cand.get("canon_claims", [])
             for cid in e.get("chunk_ids", [])}
    use = [c for c in chunks if c["id"] in cited] or chunks[:8]
    prompt = (f"{_card_block(card)}\n\nCANON PASSAGES from the {name} wiki:\n\n{_chunks_block(use)}\n\n"
              f"CHOSEN ANALOGY (structured):\n{json.dumps(cand, ensure_ascii=False, indent=1)}\n\n"
              f"Teach {card['name']} through this analogy. {LESSON_SPEC}Make the mapping explicit (which anime "
              f"element plays which CS role), cover every relation that holds, and state clearly where it breaks "
              f"using the 'breaks' field. Cite passage ids in square brackets after every anime fact. "
              f"Do not add anime facts that the passages do not support.")
    if feedback:
        prompt += f"\n\nA fact-checker rejected these statements in your previous draft; fix or remove them:\n{feedback}"
    return llm.generate(prompt, TUTOR, temperature=0.5, sample=sample)


# ---------------------------------------------------------------- faithfulness check

def extract_claims(markdown: str, *, sample: int = 0) -> list[dict]:
    prompt = ("Split this lesson into atomic factual claims. Label each claim 'canon' if it states something about "
              "the anime's story, characters, abilities or rules, or 'cs' if it states something about computer "
              "science. Skip opinions, questions and pure analogy framing ('X is like Y') unless it asserts a fact. "
              "Return JSON: {\"claims\": [{\"text\": str, \"type\": \"canon\"|\"cs\"}]}\n\nLESSON:\n" + markdown)
    return llm.generate_json(prompt, "You are a meticulous fact-check editor.", temperature=0.0,
                             sample=sample).get("claims", [])


def verify_claims(anime: str, card: dict, claims: list[dict], *, sample: int = 0) -> list[dict]:
    """Canon claims are checked against retrieved wiki passages (retrieved per claim, so the plain
    condition is judged on the same evidence as the others); CS claims against the concept card."""
    name = ANIME[anime]["name"]
    idx = Index.load(anime)
    evidence: dict[str, dict] = {}
    items = []
    for i, c in enumerate(claims):
        if c.get("type") == "canon":
            ev = idx.search(c["text"], k=4)
            for e in ev:
                evidence[e["id"]] = e
            items.append(f"{i}. [canon] {c['text']}  (evidence ids: {', '.join(e['id'] for e in ev)})")
        else:
            items.append(f"{i}. [cs] {c['text']}")
    if not items:
        return []
    prompt = (f"Verify each claim. Canon claims about {name} must be checked ONLY against the passages below "
              f"(if the passages do not say it, it is 'unsupported', even if you believe it is true). CS claims must "
              f"be checked against the CS reference.\n\nCS REFERENCE:\n{_card_block(card)}\n\nPASSAGES:\n\n"
              f"{_chunks_block(list(evidence.values()))}\n\nCLAIMS:\n" + "\n".join(items) +
              "\n\nReturn JSON: {\"verdicts\": [{\"i\": int, \"verdict\": \"supported\"|\"unsupported\"|"
              "\"contradicted\", \"evidence\": [str], \"note\": str}]}")
    out = llm.generate_json(prompt, "You are a strict fact-checker. Absence of evidence means unsupported.",
                            temperature=0.0, sample=sample)
    by_i = {v.get("i"): v for v in out.get("verdicts", [])}
    res = []
    for i, c in enumerate(claims):
        v = by_i.get(i, {})
        res.append({**c, "verdict": v.get("verdict", "unsupported"), "evidence": v.get("evidence", []),
                    "note": v.get("note", "")})
    return res


def check(anime: str, card: dict, markdown: str, *, sample: int = 0) -> dict:
    claims = verify_claims(anime, card, extract_claims(markdown, sample=sample), sample=sample)
    canon = [c for c in claims if c["type"] == "canon"]
    cs = [c for c in claims if c["type"] == "cs"]
    bad = lambda cs_: sum(c["verdict"] != "supported" for c in cs_)  # noqa: E731
    # The card cannot cover every true CS statement, so only contradictions count as CS errors.
    return {
        "claims": claims,
        "n_canon": len(canon), "canon_unsupported": bad(canon),
        "n_cs": len(cs), "cs_contradicted": sum(c["verdict"] == "contradicted" for c in cs),
        "canon_unsupported_rate": bad(canon) / len(canon) if canon else 0.0,
    }
