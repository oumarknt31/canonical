"""Step 5: run conditions A-D on all 24 pairs (x samples) and write the ablation table.

    .venv/bin/python scripts/run_ablation.py --samples 0 1 2
"""
import argparse
import json

import pandas as pd

import _path  # noqa: F401
from canonical import store
from canonical.config import ANIME, RESULTS_DIR, TOPICS, load_card
from canonical.pipeline import teach

ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, nargs="+", default=[0])
ap.add_argument("--conditions", nargs="+", default=["A", "B", "C", "D"])
ap.add_argument("--anime", nargs="*", default=list(ANIME))
ap.add_argument("--topics", nargs="*", default=TOPICS)
a = ap.parse_args()

for s in a.samples:
    for an in a.anime:
        for t in a.topics:
            for c in a.conditions:
                if store.get_lesson(f"{an}|{t}|{c}|{s}"):
                    continue
                try:
                    r = teach(an, t, c, sample=s)
                    k = r["check"]
                    print(f"s{s} {an:7} {t:14} {c}: canon unsupported {k['canon_unsupported']}/{k['n_canon']}, "
                          f"cs contradicted {k['cs_contradicted']}/{k['n_cs']}  ({r['seconds']}s)")
                except Exception as e:
                    print(f"s{s} {an:7} {t:14} {c}: FAILED {str(e)[:200]}")


def relation_coverage(lesson, topic):
    ch = lesson.get("chosen")
    if not ch:
        return None
    card = load_card(topic)
    holds = {r.get("relation_id"): r.get("holds") for r in ch.get("relation_map", [])}
    return sum(holds.get(r["id"]) in ("yes", "partial") for r in card["relations"]) / len(card["relations"])


rows = []
for L in store.load_lessons():
    d, k = L["data"], L["data"].get("check", {})
    rows.append({"condition": L["condition"], "anime": L["anime"], "topic": L["topic"], "sample": L["sample"],
                 "canon_claims": k.get("n_canon", 0), "canon_unsupported": k.get("canon_unsupported", 0),
                 "cs_claims": k.get("n_cs", 0), "cs_contradicted": k.get("cs_contradicted", 0),
                 "relation_coverage": relation_coverage(d, L["topic"]),
                 "breaks_mentioned": ("break" in d.get("markdown", "").lower()),
                 "seconds": d.get("seconds")})
df = pd.DataFrame(rows)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(RESULTS_DIR / "ablation_lessons.csv", index=False)
g = df.groupby("condition")
table = pd.DataFrame({
    "lessons": g.size(),
    "unsupported_canon_pct": 100 * g["canon_unsupported"].sum() / g["canon_claims"].sum().clip(lower=1),
    "cs_contradictions_per_lesson": g["cs_contradicted"].mean(),
    "relation_coverage_pct": 100 * g["relation_coverage"].mean(),
    "breaks_mentioned_pct": 100 * g["breaks_mentioned"].mean(),
}).round(1)
print(table.to_string())
(RESULTS_DIR / "ablation_table.json").write_text(table.reset_index().to_json(orient="records", indent=1))
