"""Offline tests (no API key): features, fusion, citation rendering, card integrity."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from canonical.config import CARDS_DIR, TOPICS, load_card  # noqa: E402
from canonical.features import FEATURES, compute, heuristic_score  # noqa: E402
from canonical.retrieval import rrf, tokens  # noqa: E402

CHUNKS = [
    {"id": "naruto:shadow-clone-technique:0", "title": "Shadow Clone Technique", "heading": "Overview",
     "text": "The user creates clones that are real copies; when a clone is dispelled its experience returns to the user."},
    {"id": "naruto:rasengan:0", "title": "Rasengan", "heading": "Overview", "text": "A spinning sphere of chakra."},
]


def cand(holds="yes", cite=True, breaks="Clones do not form a call stack; the original never waits."):
    card = load_card("recursion")
    cid = [CHUNKS[0]["id"]] if cite else ["naruto:made-up:9"]
    return {
        "element": "Shadow Clone Technique", "summary": "Clones dispel and report back",
        "role_map": [{"role_id": r["id"], "anime": "a clone", "chunk_ids": cid} for r in card["roles"]],
        "relation_map": [{"relation_id": r["id"], "holds": holds,
                          "explanation": "the experience of each clone returns to the user when it is dispelled",
                          "chunk_ids": cid} for r in card["relations"]],
        "breaks": breaks,
        "canon_claims": [{"claim": "When a clone is dispelled its experience returns to the user", "chunk_ids": cid}],
        "self_score": 5,
    }


def test_cards_valid():
    for t in TOPICS:
        c = json.loads((CARDS_DIR / f"{t}.json").read_text())
        roles = {r["id"] for r in c["roles"]}
        assert len(c["quiz"]) == 5
        assert all(0 <= q["answer"] < len(q["choices"]) for q in c["quiz"])
        assert all(set(r["involves"]) <= roles for r in c["relations"])


def test_features_reward_structure_and_citations():
    card = load_card("recursion")
    good = compute(cand(), card, CHUNKS)
    shallow = compute(cand(holds="no", cite=False, breaks=""), card, CHUNKS)
    assert set(FEATURES) <= set(good)
    assert good["relation_holds"] == 1.0 and shallow["relation_holds"] == 0.0
    assert good["citation_valid"] == 1.0 and shallow["citation_valid"] == 0.0
    assert good["claim_support_lex"] > 0.6 and shallow["claim_support_lex"] == 0.0
    assert math.isnan(good["topic_sim"])  # semantic features need embeddings
    assert heuristic_score(good) > heuristic_score(shallow)


def test_rrf_prefers_agreement():
    assert rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf([["x", "y"], ["y", "z"]])[0] == "y"


def test_tokens_drop_stopwords():
    assert tokens("The Rasengan is a sphere of chakra") == ["rasengan", "sphere", "chakra"]
