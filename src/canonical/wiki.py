"""Fetch a small, attributed canon corpus from Fandom wikis through the MediaWiki API.

Fandom text is CC BY-SA; images are not covered, so we keep text only. Volume is deliberately
tiny (about 25-30 pages per anime, one request per second) and every page keeps its URL and
revision id for attribution. The corpus is for local use only and is never redistributed.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .config import ANIME, ANIME_DIR, USER_AGENT

SESSION = requests.Session()
SESSION.headers["User-Agent"] = USER_AGENT
DELAY = 1.0

DROP_SELECTORS = [
    "aside", "table", ".navbox", ".reference", "sup", ".toc", "figure", ".mw-editsection",
    ".portable-infobox", ".gallery", ".wikia-gallery", "style", "script", ".quote", ".dablink",
    ".hatnote", ".notice", "noscript", ".mw-references-wrap", "ol.references",
]
STOP_HEADINGS = {"references", "see also", "trivia", "navigation", "external links", "notes",
                 "gallery", "quotes", "in other media", "site navigation"}


def _api(wiki: str, **params) -> dict:
    params.setdefault("format", "json")
    r = SESSION.get(f"https://{wiki}.fandom.com/api.php", params=params, timeout=30)
    r.raise_for_status()
    time.sleep(DELAY)
    return r.json()


def resolve_title(wiki: str, query: str) -> str | None:
    """Exact page if it exists (following redirects), otherwise the top search hit."""
    d = _api(wiki, action="query", titles=query, redirects=1)
    pages = d.get("query", {}).get("pages", {})
    for p in pages.values():
        if "missing" not in p and int(p.get("ns", 0)) == 0:
            return p["title"]
    d = _api(wiki, action="query", list="search", srsearch=query, srlimit=1, srnamespace=0)
    hits = d.get("query", {}).get("search", [])
    return hits[0]["title"] if hits else None


def clean_sections(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for sel in DROP_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    sections, heading, buf = [], "Introduction", []

    def flush():
        text = " ".join(buf).strip()
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"\s+", " ", text)
        if len(text.split()) >= 15:
            sections.append({"heading": heading, "text": text})

    root = soup.select_one(".mw-parser-output") or soup
    stopped = False
    for el in root.find_all(["h2", "h3", "h4", "p", "li", "dd"], recursive=True):
        if el.name in ("h2", "h3", "h4"):
            flush()
            buf = []
            heading = el.get_text(" ", strip=True)
            if el.name == "h2" and heading.lower() in STOP_HEADINGS:
                stopped = True
            elif el.name == "h2":
                stopped = False
            continue
        if stopped:
            continue
        # skip list items that only contain navigation/links collections
        t = el.get_text(" ", strip=True)
        if t:
            buf.append(t)
    flush()
    return sections


def fetch_page(wiki: str, title: str) -> dict:
    d = _api(wiki, action="parse", page=title, prop="text|revid|displaytitle", redirects=1)
    p = d["parse"]
    real = p["title"]
    return {
        "title": real,
        "url": f"https://{wiki}.fandom.com/wiki/{real.replace(' ', '_')}",
        "revid": p.get("revid"),
        "license": "CC BY-SA (Fandom)",
        "sections": clean_sections(p["text"]["*"]),
    }


def fetch_anime(key: str, log=print) -> Path:
    cfg = ANIME[key]
    out_dir = ANIME_DIR / key
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pages.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            done.add(json.loads(line)["query"])
    with out.open("a") as f:
        for q in cfg["pages"]:
            if q in done:
                continue
            try:
                title = resolve_title(cfg["wiki"], q)
                if not title:
                    log(f"  [{key}] no page for {q!r}")
                    continue
                page = fetch_page(cfg["wiki"], title)
                page["query"] = q
                page["anime"] = key
                words = sum(len(s["text"].split()) for s in page["sections"])
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
                f.flush()
                log(f"  [{key}] {q!r} -> {page['title']!r}: {len(page['sections'])} sections, {words} words")
            except Exception as e:  # keep going; a missing page is not fatal
                log(f"  [{key}] failed {q!r}: {e}")
    return out


def load_pages(key: str) -> list[dict]:
    p = ANIME_DIR / key / "pages.jsonl"
    if not p.exists():
        return []
    pages, seen = [], set()
    for line in p.read_text().splitlines():
        d = json.loads(line)
        if d["title"] in seen:  # two queries can resolve to the same page
            continue
        seen.add(d["title"])
        pages.append(d)
    return pages
