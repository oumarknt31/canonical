"""Canonical - learn CS through anime, with every analogy checked.

    .venv/bin/streamlit run app/Home.py
"""
import html

import streamlit as st

import ui
from canonical import store
from canonical.config import ANIME, TOPICS, load_card
from canonical.llm import LLMError

ui.setup("Canonical")

st.title("Canonical")
st.markdown('<div class="hero-sub">Learn computer science through the anime you love, '
            'with every analogy checked against canon and against the CS.</div>', unsafe_allow_html=True)
st.markdown("")

cards = {t: load_card(t) for t in TOPICS}
c1, c2, c3 = st.columns([1.2, 1.2, 1])
anime = c1.selectbox("Anime", list(ANIME), format_func=ui.anime_name)
topic = c2.selectbox("CS topic", TOPICS, format_func=lambda t: cards[t]["name"])
mode = c3.radio("Mode", ["Checked (Canonical)", "Compare with a plain prompt"], index=0)

cached = {c: store.get_lesson(f"{anime}|{topic}|{c}|0") for c in "AD"}
go = st.button("Teach me", type="primary", width="stretch")
if go:
    want = ["D"] if mode.startswith("Checked") else ["A", "D"]
    for c in want:
        if cached[c]:
            continue
        if not ui.has_key():
            st.warning("This pair is not cached and no GEMINI_API_KEY is set, so it cannot be generated here.")
            st.stop()
        from canonical.pipeline import teach
        with st.status(f"Building the {'checked' if c == 'D' else 'plain'} lesson...", expanded=True) as s:
            try:
                cached[c] = {"data": teach(anime, topic, c, sample=0, log=s.write)}
                s.update(label="Done", state="complete")
            except LLMError as e:
                s.update(label="The model call failed", state="error")
                st.error(str(e))
                st.stop()


def show(lesson: dict, title: str):
    d = lesson["data"]
    chk = d.get("check", {})
    nc, nu = chk.get("n_canon", 0), chk.get("canon_unsupported", 0)
    ncs, ncx = chk.get("n_cs", 0), chk.get("cs_contradicted", 0)
    st.subheader(title)
    st.markdown(
        f'<span class="pill {"ok" if nu == 0 else "bad"}">{nc - nu}/{nc} anime facts verified</span>'
        f'<span class="pill {"ok" if ncx == 0 else "bad"}">{ncs - ncx}/{ncs} CS statements consistent</span>'
        + (f'<span class="pill mute">fixed {d["first_draft"]["check"]["canon_unsupported"]} claims in a rewrite</span>'
           if d.get("first_draft") else ""),
        unsafe_allow_html=True)
    body, sources = ui.render_citations(d["markdown"], lesson.get("anime", anime) if "anime" in lesson else anime)
    st.markdown(body, unsafe_allow_html=True)
    with st.expander(f"Fact check: {len(chk.get('claims', []))} claims"):
        for c in chk.get("claims", []):
            ok = c["verdict"] == "supported"
            st.markdown(f'<div class="claim {"ok" if ok else "bad"}">{ui.verdict_pill(c["verdict"])}'
                        f'<span class="pill mute">{c["type"]}</span>{html.escape(c["text"])}'
                        + (f'<br><small style="color:#718096">{html.escape(c.get("note", ""))}</small>'
                           if c.get("note") and not ok else "") + "</div>", unsafe_allow_html=True)
    if sources:
        with st.expander(f"Sources ({len(sources)} wiki passages, CC BY-SA)"):
            for s in sources:
                c = s["chunk"]
                if c:
                    st.markdown(f"**[{s['n']}]** [{c['title']} / {c['heading']}]({c['url']}) (rev {c['revid']})  \n"
                                f"<small>{html.escape(c['text'][:300])}...</small>", unsafe_allow_html=True)


if mode.startswith("Checked") and cached["D"]:
    d = cached["D"]["data"]
    left, right = st.columns([1.6, 1])
    with left:
        show(cached["D"], f"{cards[topic]['name']} through {ui.anime_name(anime)}")
    with right:
        ch = d.get("chosen", {})
        st.subheader("The analogy")
        st.markdown(f"**{ch.get('element', '')}**: {ch.get('summary', '')}")
        st.markdown(f'<div class="breaks"><b>Where the analogy breaks.</b> {html.escape(ch.get("breaks", ""))}</div>',
                    unsafe_allow_html=True)
        st.markdown("")
        if d.get("ranking"):
            st.markdown("**Why this one?** Five candidates, ranked by the analogy-quality model:")
            for r in d["ranking"]:
                st.progress(min(max(r["score"], 0.0), 1.0), text=f"{r['element']} ({r['score']:.2f})")
    st.divider()
    t1, t2 = st.tabs(["Mapping: CS role to anime element", "Quiz yourself"])
    with t1:
        ui.mapping_table(d.get("chosen", {}), topic)
    with t2:
        ui.quiz(topic, f"{anime}-{topic}")
elif not mode.startswith("Checked") and cached["A"] and cached["D"]:
    l, r = st.columns(2)
    with l:
        show(cached["A"], "Plain prompt (what a chatbot gives you)")
    with r:
        show(cached["D"], "Canonical (grounded, ranked, checked)")
else:
    st.info("Pick an anime and a topic, then press **Teach me**." +
            ("" if ui.has_key() else " No API key is set, so only cached pairs work."))
