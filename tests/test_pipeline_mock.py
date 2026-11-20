"""End-to-end pipeline test with a fake LLM (no key, no network): conditions A and D, the gate rewrite,
candidate storage, features, and the classifier training path."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canonical import llm, store  # noqa: E402
from canonical.config import load_card  # noqa: E402

CID = "naruto:shadow-clone-technique:4"


def fake_json(prompt, system="", **kw):
    if "List 5 DIFFERENT elements" in prompt:
        return {"elements": [{"element": f"Shadow Clone {i}", "why": "clones report back",
                              "search": "shadow clone dispelled experience"} for i in range(5)]}
    if "build a structured analogy" in prompt:
        from canonical.config import load_cards
        name = prompt.split("CS CONCEPT: ")[1].split("\n")[0]
        card = next(c for c in load_cards().values() if c["name"] == name)
        out = []
        for i in range(5):
            out.append({
                "element": f"Shadow Clone {i}", "summary": "clones dispel and report back",
                "role_map": [{"role_id": r["id"], "anime": "clone" if j <= i else None, "chunk_ids": [CID]}
                             for j, r in enumerate(card["roles"])],
                "relation_map": [{"relation_id": r["id"], "holds": ["no", "partial", "yes"][min(i, 2)],
                                  "explanation": "the clone's experience goes back to the user when dispelled",
                                  "chunk_ids": [CID]} for r in card["relations"]],
                "breaks": "the original does not wait for its clones" if i else "",
                "canon_claims": [{"claim": "a dispelled clone transfers what it learned", "chunk_ids": [CID]}],
                "self_score": 5,
            })
        return {"candidates": out}
    if "Split this lesson into atomic factual claims" in prompt:
        claims = [{"text": "a dispelled clone transfers what it learned", "type": "canon"},
                  {"text": "a base case stops recursion", "type": "cs"}]
        if "FIRST DRAFT" in prompt:
            claims.append({"text": "Naruto invented the Rasengan", "type": "canon"})
        return {"claims": claims}
    if "Verify each claim" in prompt:
        v = []
        for line in prompt.split("CLAIMS:\n")[1].splitlines():
            if line[:1].isdigit():
                i = int(line.split(".")[0])
                v.append({"i": i, "verdict": "unsupported" if "invented" in line else "supported",
                          "evidence": [CID], "note": ""})
        return {"verdicts": v}
    raise AssertionError("unexpected prompt: " + prompt[:80])


def fake_text(prompt, system="", **kw):
    if "A fact-checker rejected" in prompt:
        return f"Clones transfer what they learned [{CID}]. A base case stops recursion."
    if "CHOSEN ANALOGY" in prompt:
        return f"FIRST DRAFT Clones transfer what they learned [{CID}]. Naruto invented the Rasengan."
    return "Plain lesson: shadow clones are like recursion."


def fake_embed(texts, task="x"):
    out = []
    for t in texts:
        rng = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
        out.append(rng.normal(size=16).tolist())
    return out


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.sqlite")
    monkeypatch.setattr(store, "_CON", None)
    monkeypatch.setattr(llm, "generate_json", fake_json)
    monkeypatch.setattr(llm, "generate", fake_text)
    monkeypatch.setattr(llm, "embed", fake_embed)
    from canonical import ranker
    monkeypatch.setattr(ranker, "LABELS_CSV", tmp_path / "labels.csv")
    monkeypatch.setattr(ranker, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(ranker, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(ranker, "_loaded", None)
    yield


def test_condition_d_ranks_and_gates():
    from canonical.pipeline import teach
    r = teach("naruto", "recursion", "D")
    # the heuristic ranker should prefer the candidate with full coverage and "yes" relations
    assert r["ranking"][0]["element"] == "Shadow Clone 4"
    assert r["chosen"]["element"] == "Shadow Clone 4"
    # the first draft had an unsupported claim; the gate rewrote it and the rewrite is clean
    assert r["first_draft"]["check"]["canon_unsupported"] == 1
    assert r["check"]["canon_unsupported"] == 0
    assert len(store.load_candidates()) == 5
    assert store.get_lesson("naruto|recursion|D|0") is not None


def test_condition_a_is_checked_too():
    from canonical.pipeline import teach
    r = teach("naruto", "recursion", "A")
    assert r["chunk_ids"] == [] and "check" in r


def test_classifier_trains_on_labels():
    from canonical import ranker
    from canonical.pipeline import make_candidates
    for t in ["recursion", "stack", "queue", "bfs", "dfs", "hash_table", "binary_search", "deadlock"]:
        for rnd in range(2):
            make_candidates("naruto", t, round_=rnd)
    for c in store.load_candidates():
        good = c["features"]["relation_holds"] == 1.0
        ranker.save_label({"candidate_id": c["id"], "overall": "good" if good else "bad", "canon_error": 0,
                           "cs_error": 0, "shallow": int(not good), "rater": "primary", "note": ""})
    res = ranker.evaluate()
    assert "error" not in res, res
    assert res["auc"]["logistic_regression"] > 0.9
    assert 0.0 <= ranker.score(store.load_candidates()[0]["features"]) <= 1.0
