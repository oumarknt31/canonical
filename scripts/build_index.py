"""Step 2: chunk the corpus and embed it (needs GEMINI_API_KEY for embeddings).

    .venv/bin/python scripts/build_index.py            # all three, with embeddings
    .venv/bin/python scripts/build_index.py --no-embed # BM25 only (no key needed)

The free tier caps embeddings per day. Every finished batch is cached, so when the cap is hit the script
stops cleanly; rerun it after the reset (midnight Pacific) and it continues where it left off.
"""
import sys

import _path  # noqa: F401
from canonical.config import ANIME
from canonical.llm import DailyQuotaError
from canonical.retrieval import build_index

embed = "--no-embed" not in sys.argv
keys = [a for a in sys.argv[1:] if not a.startswith("--")] or list(ANIME)
for k in keys:
    try:
        build_index(k, embed=embed)
    except DailyQuotaError:
        print(f"  [{k}] daily embedding quota used up; progress is cached. Rerun after midnight Pacific.")
        sys.exit(3)
