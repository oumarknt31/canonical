"""Step 3: generate 5 structured analogy candidates per (anime, topic) pair, per round.
3 anime x 8 topics x 5 = 120 candidates per round; run rounds 0 and 1 for ~240 to label.

    .venv/bin/python scripts/make_candidates.py --rounds 0 1
    .venv/bin/python scripts/make_candidates.py --rounds 0 1 --no-embed   # text calls only
"""
import argparse

import _path  # noqa: F401
from canonical.config import ANIME, TOPICS
from canonical.llm import DailyQuotaError
from canonical.pipeline import make_candidates

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", type=int, nargs="+", default=[0])
ap.add_argument("--anime", nargs="*", default=list(ANIME))
ap.add_argument("--topics", nargs="*", default=TOPICS)
ap.add_argument("--no-embed", action="store_true",
                help="skip the embedding-based features (fill them in later); saves the daily embedding quota")
a = ap.parse_args()
n = 0
for r in a.rounds:
    for an in a.anime:
        for t in a.topics:
            try:
                cs = make_candidates(an, t, round_=r, with_embed=not a.no_embed)
                n += len(cs)
                print(f"r{r} {an:7} {t:14} {len(cs)} candidates  e.g. {cs[0]['cand'].get('element') if cs else '-'}")
            except DailyQuotaError:
                print(f"daily quota used up after {n} candidates; everything so far is cached. Rerun tomorrow.")
                raise SystemExit(3)
            except Exception as e:
                print(f"r{r} {an:7} {t:14} FAILED: {str(e)[:200]}")
print(f"total {n}")
