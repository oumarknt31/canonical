"""Step 2: chunk the corpus and embed it (needs GEMINI_API_KEY for embeddings).

    .venv/bin/python scripts/build_index.py            # all three, with embeddings
    .venv/bin/python scripts/build_index.py --no-embed # BM25 only (no key needed)
"""
import sys

import _path  # noqa: F401
from canonical.config import ANIME
from canonical.retrieval import build_index

embed = "--no-embed" not in sys.argv
keys = [a for a in sys.argv[1:] if not a.startswith("--")] or list(ANIME)
for k in keys:
    build_index(k, embed=embed)
