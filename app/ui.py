"""Shared Streamlit helpers: paths, theme CSS, citation rendering, mapping table, quiz."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from canonical.config import ANIME, GEMINI_API_KEY, load_card  # noqa: E402
from canonical.retrieval import Index  # noqa: E402

CSS = """
<style>
:root { --ok:#1f9d55; --warn:#d69e2e; --bad:#e53e3e; --ink:#1a202c; --muted:#718096; }
.block-container { max-width: 1100px; padding-top: 2rem; }
h1, h2, h3 { letter-spacing: -0.01em; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; margin-right:6px; }
.pill.ok { background:#e6f6ec; color:var(--ok); } .pill.bad { background:#fdecec; color:var(--bad); }
.pill.warn { background:#fdf5e6; color:#b7791f; } .pill.mute { background:#edf2f7; color:#4a5568; }
.cite { font-size:0.7em; vertical-align: super; text-decoration:none; color:#3182ce; font-weight:600; }
.breaks { border-left:4px solid var(--warn); background:#fffaf0; padding:0.8rem 1rem; border-radius:6px; color:#1a202c; }
.claim { padding:6px 10px; border-radius:6px; margin:4px 0; font-size:0.92rem; color:#1a202c; }
.claim.ok { background:#f0fff4; border-left:3px solid var(--ok); }
.claim.bad { background:#fff5f5; border-left:3px solid var(--bad); }
.hero-sub { color: var(--muted); font-size:1.05rem; margin-top:-0.6rem; }
</style>
"""

CITE_RE = re.compile(r"\[((?:naruto|hxh|jjk):[^\]\s,]+)(?:,\s*((?:naruto|hxh|jjk):[^\]\s,]+))*\]")
ANY_ID = re.compile(r"(?:naruto|hxh|jjk):[A-Za-z0-9\-]+:\d+")


def setup(title: str):
    st.set_page_config(page_title=title, page_icon="🍥", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def has_key() -> bool:
    return bool(GEMINI_API_KEY)


def chunk(anime: str, cid: str) -> dict | None:
    try:
        return Index.load(anime).by_id.get(cid)
    except Exception:
        return None


def render_citations(md: str, anime: str) -> tuple[str, list[dict]]:
    """Replace [anime:page:n] markers with numbered superscript links; return the source list."""
    order: dict[str, int] = {}
    sources = []

    def sub(m):
        ids = ANY_ID.findall(m.group(0))
        out = []
        for cid in ids:
            if cid not in order:
                order[cid] = len(order) + 1
                c = chunk(anime, cid)
                sources.append({"n": order[cid], "id": cid, "chunk": c})
            c = chunk(anime, cid)
            url = c["url"] if c else "#"
            out.append(f'<a class="cite" href="{url}" target="_blank" title="{cid}">[{order[cid]}]</a>')
        return "".join(out)

    return re.sub(r"\[[^\]]*(?:naruto|hxh|jjk):[^\]]*\]", sub, md), sources


def verdict_pill(v: str) -> str:
    cls = {"supported": "ok", "unsupported": "warn", "contradicted": "bad", "yes": "ok", "partial": "warn",
           "no": "bad"}.get(str(v).lower(), "mute")
    return f'<span class="pill {cls}">{v}</span>'


def mapping_table(chosen: dict, topic: str):
    card = load_card(topic)
    roles = {r.get("role_id"): r for r in chosen.get("role_map", [])}
    rels = {r.get("relation_id"): r for r in chosen.get("relation_map", [])}
    rows = []
    for r in card["roles"]:
        a = (roles.get(r["id"]) or {}).get("anime")
        rows.append(f"<tr><td><b>{r['name']}</b></td><td>{a or '<i>no counterpart</i>'}</td>"
                    f"<td>{verdict_pill('mapped' if a else 'missing')}</td></tr>")
    st.markdown("<table style='width:100%'><tr><th>CS role</th><th>Anime counterpart</th><th></th></tr>"
                + "".join(rows) + "</table>", unsafe_allow_html=True)
    st.markdown("")
    rrows = []
    for r in card["relations"]:
        x = rels.get(r["id"]) or {}
        rrows.append(f"<tr><td>{r['desc']}</td><td>{verdict_pill(x.get('holds', 'no'))}</td>"
                     f"<td style='color:#4a5568'>{x.get('explanation', '')}</td></tr>")
    st.markdown("<table style='width:100%'><tr><th>Structural rule</th><th>Holds?</th><th>How</th></tr>"
                + "".join(rrows) + "</table>", unsafe_allow_html=True)


def quiz(topic: str, key: str):
    card = load_card(topic)
    for q in card["quiz"]:
        ans = st.radio(f"**{q['q']}**", q["choices"], index=None, key=f"{key}-{q['id']}")
        if ans is not None:
            if q["choices"].index(ans) == q["answer"]:
                st.success("Correct. " + q["explain"])
            else:
                st.error("Not quite. " + q["explain"])


def anime_name(k: str) -> str:
    return ANIME[k]["name"]
