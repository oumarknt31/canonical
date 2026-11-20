"""Step 1: pull the canon corpus for the three showcase anime (resumable).

    .venv/bin/python scripts/fetch_corpus.py            # all three
    .venv/bin/python scripts/fetch_corpus.py naruto     # one
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canonical.config import ANIME  # noqa: E402
from canonical.wiki import fetch_anime, load_pages  # noqa: E402

keys = sys.argv[1:] or list(ANIME)
for k in keys:
    print(f"== {ANIME[k]['name']}")
    fetch_anime(k)
    pages = load_pages(k)
    words = sum(len(s["text"].split()) for p in pages for s in p["sections"])
    print(f"   {len(pages)} unique pages, {words:,} words")
