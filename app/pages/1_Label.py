"""Labelling tool: rate analogy candidates blind (no model score or self-rating shown).
Labels go to data/labels.csv and train the analogy-quality classifier."""
import html
import random

import streamlit as st

import ui
from canonical import store
from canonical.config import load_card
from canonical.ranker import load_labels, save_label

ui.setup("Label analogies")
st.title("Label analogies")
st.caption("Rate each candidate as a CS tutor who knows the show. Judge structure, not style: does each anime "
           "element really behave like the CS role it is mapped to, and are the anime facts true?")

rater = st.sidebar.text_input("Rater name", "primary",
                              help="Use 'primary' for your own labels. A second rater uses their own name.")
cands = store.load_candidates()
done = set(load_labels(rater)["candidate_id"]) if rater else set()
st.sidebar.metric("Labelled", f"{len(done)} / {len(cands)}")
if rater != "primary":
    prim = set(load_labels("primary")["candidate_id"])
    cands = [c for c in cands if c["id"] in prim]  # second raters only see items the primary labelled
todo = [c for c in cands if c["id"] not in done]
random.Random(7).shuffle(todo)
if not todo:
    st.success("Nothing left to label." if cands else "No candidates yet: run scripts/make_candidates.py first.")
    st.stop()

c = todo[0]
cand, card = c["data"]["cand"], load_card(c["topic"])
st.markdown(f"<span class='pill mute'>{ui.anime_name(c['anime'])}</span><span class='pill mute'>{card['name']}</span>"
            f"<span class='pill mute'>{c['id']}</span>", unsafe_allow_html=True)
st.subheader(cand.get("element", "?"))
st.write(cand.get("summary", ""))
ui.mapping_table(cand, c["topic"])
st.markdown(f'<div class="breaks"><b>Claimed break.</b> {html.escape(str(cand.get("breaks", "")))}</div>',
            unsafe_allow_html=True)

with st.expander(f"Canon claims ({len(cand.get('canon_claims', []))}) and the passages they cite"):
    for cl in cand.get("canon_claims", []):
        st.markdown(f"- {cl.get('claim')}")
        for cid in cl.get("chunk_ids", []) or []:
            ch = ui.chunk(c["anime"], cid)
            if ch:
                st.markdown(f"<small style='color:#4a5568'>&nbsp;&nbsp;[{cid}] {html.escape(ch['text'][:400])}...</small>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<small style='color:#e53e3e'>&nbsp;&nbsp;[{cid}] not a real passage id</small>",
                            unsafe_allow_html=True)

with st.form(f"f-{c['id']}"):
    overall = st.radio("Overall", ["good", "ok", "bad"], horizontal=True, index=None,
                       help="good = you would teach with it as is; ok = usable after edits; bad = misleading or wrong")
    a, b, d = st.columns(3)
    canon_err = a.checkbox("Canon error (an anime fact is wrong)")
    cs_err = b.checkbox("CS error (teaches something false)")
    shallow = d.checkbox("Shallow (surface match, structure does not hold)")
    note = st.text_input("Note (optional)")
    if st.form_submit_button("Save and next", type="primary"):
        if overall is None:
            st.warning("Pick good / ok / bad first.")
        else:
            save_label({"candidate_id": c["id"], "overall": overall, "canon_error": int(canon_err),
                        "cs_error": int(cs_err), "shallow": int(shallow), "rater": rater, "note": note})
            st.rerun()
