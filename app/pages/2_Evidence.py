"""Evidence: the corpus, the labels, the classifier vs baselines, the ablation, retrieval quality.
Every number on this page is computed from files in data/ - nothing is typed in by hand."""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui
from canonical import store
from canonical.config import ANIME, LABELS_CSV, RESULTS_DIR
from canonical.features import FEATURES
from canonical.ranker import dataset, load_labels
from canonical.wiki import load_pages

ui.setup("Evidence")
st.title("Evidence")
st.caption("Every number here is computed from the project's data files. Empty sections mean the experiment "
           "has not been run yet, not that it failed.")

# ---------------------------------------------------------------- corpus
st.header("1. The canon corpus")
rows = []
for k in ANIME:
    for p in load_pages(k):
        rows.append({"anime": ANIME[k]["name"], "page": p["title"],
                     "words": sum(len(s["text"].split()) for s in p["sections"]), "sections": len(p["sections"])})
corpus = pd.DataFrame(rows)
if corpus.empty:
    st.info("No corpus yet: run scripts/fetch_corpus.py")
else:
    a, b, c = st.columns(3)
    a.metric("Wiki pages", len(corpus))
    b.metric("Words of canon", f"{corpus['words'].sum():,}")
    c.metric("Anime", corpus["anime"].nunique())
    st.plotly_chart(px.bar(corpus.sort_values("words"), x="words", y="page", color="anime", orientation="h",
                           height=max(400, 14 * len(corpus)), title="Words per wiki page"), width="stretch")

# ---------------------------------------------------------------- labels
st.header("2. Hand-labelled analogies")
lab = load_labels()
cands = store.load_candidates()
st.write(f"{len(cands)} candidates generated, {len(lab)} labelled by the primary rater.")
if not lab.empty:
    df = dataset()
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(lab, x="overall", category_orders={"overall": ["good", "ok", "bad"]},
                                title="Label balance"), width="stretch")
    if not df.empty:
        f = b.selectbox("Feature vs label", FEATURES)
        b.plotly_chart(px.box(df, x="overall", y=f, points="all", category_orders={"overall": ["good", "ok", "bad"]},
                              title=f"{f} by label"), width="stretch")
    flags = lab[["canon_error", "cs_error", "shallow"]].mean().mul(100).round(1)
    st.write("Share of labelled candidates with each problem (%):", flags.to_dict())
    others = sorted(set(pd.read_csv(LABELS_CSV)["rater"]) - {"primary"})
    if others:
        from sklearn.metrics import cohen_kappa_score
        p = lab.set_index("candidate_id")["overall"]
        for r in others:
            o = load_labels(r).set_index("candidate_id")["overall"]
            both = p.index.intersection(o.index)
            if len(both) >= 10:
                m = {"bad": 0, "ok": 1, "good": 2}
                k = cohen_kappa_score(p[both].map(m), o[both].map(m), weights="quadratic")
                st.metric(f"Agreement with rater '{r}' (weighted Cohen's kappa, n={len(both)})", f"{k:.2f}")

# ---------------------------------------------------------------- classifier
st.header("3. Analogy-quality classifier")
ev = RESULTS_DIR / "ranker_eval.json"
if not ev.exists():
    st.info("Not trained yet: label at least 30 candidates, then run scripts/train_ranker.py")
else:
    r = json.loads(ev.read_text())
    st.write(f"Grouped {r['folds']}-fold cross-validation over {r['n_groups']} (anime, topic) pairs; "
             f"n = {r['n']} labelled candidates, {r['n_good']} rated good. 95% CIs from a cluster bootstrap.")
    names = {"logistic_regression": "Logistic regression", "random_forest": "Random forest",
             "topic_sim": "Baseline: embedding similarity", "self_score": "Baseline: LLM self-rating",
             "heuristic": "Baseline: hand-weighted heuristic"}
    fig = go.Figure()
    for k, v in r["roc"].items():
        fig.add_trace(go.Scatter(x=v["fpr"], y=v["tpr"], mode="lines",
                                 name=f"{names.get(k, k)} (AUC {r['auc'][k]:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="gray"), name="chance"))
    fig.update_layout(title="ROC curves (out-of-fold)", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", height=480)
    st.plotly_chart(fig, width="stretch")
    t = pd.DataFrame({"model": [names.get(k, k) for k in r["auc"]], "ROC-AUC": list(r["auc"].values()),
                      "95% CI": [f"{a:.2f} - {b:.2f}" for a, b in r["auc_ci95"].values()]}).sort_values("ROC-AUC",
                                                                                                      ascending=False)
    st.dataframe(t, hide_index=True, width="stretch")
    a, b = st.columns(2)
    co = pd.DataFrame({"feature": list(r["lr_coef"]), "coef": list(r["lr_coef"].values())}).sort_values("coef")
    a.plotly_chart(px.bar(co, x="coef", y="feature", orientation="h",
                          title="Logistic regression coefficients (standardised)"), width="stretch")
    im = pd.DataFrame({"feature": list(r["rf_importance"]), "importance": list(r["rf_importance"].values())})
    b.plotly_chart(px.bar(im.sort_values("importance"), x="importance", y="feature", orientation="h",
                          title="Random forest feature importance"), width="stretch")

# ---------------------------------------------------------------- ablation
st.header("4. Ablation: does each layer help?")
ab = RESULTS_DIR / "ablation_table.json"
if not ab.exists():
    st.info("Not run yet: scripts/run_ablation.py")
else:
    t = pd.DataFrame(json.loads(ab.read_text()))
    labels = {"A": "A: plain prompt", "B": "B: + canon retrieval", "C": "C: + CS card + ranker",
              "D": "D: + faithfulness gate"}
    t["condition"] = t["condition"].map(labels)
    st.dataframe(t, hide_index=True, width="stretch")
    st.plotly_chart(px.bar(t, x="condition", y="unsupported_canon_pct",
                           title="Unsupported anime claims (%), lower is better"), width="stretch")
    st.caption("Unsupported = the fact-checker found no wiki passage backing the claim. The checker is itself "
               "an LLM; its accuracy against hand checks is reported separately.")

# ---------------------------------------------------------------- retrieval
st.header("5. Retrieval quality")
rv = RESULTS_DIR / "retrieval_eval.json"
if not rv.exists():
    st.info("Not run yet: scripts/eval_retrieval.py")
else:
    r = json.loads(rv.read_text())
    st.dataframe(pd.DataFrame(r).T.reset_index(names="method"), hide_index=True, width="stretch")
