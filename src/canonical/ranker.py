"""The analogy-quality classifier: logistic regression (Week 7) and random forest (Week 9),
trained on hand labels, evaluated with grouped cross-validation so the same (anime, topic) pair
never appears in both train and test, and compared against baselines that do not need labels.
"""
from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import store
from .config import LABELS_CSV, MODELS_DIR, RESULTS_DIR
from .features import BASELINES, FEATURES, heuristic_score

LABEL_COLS = ["candidate_id", "overall", "canon_error", "cs_error", "shallow", "rater", "note"]


def load_labels(rater: str | None = "primary") -> pd.DataFrame:
    if not LABELS_CSV.exists():
        return pd.DataFrame(columns=LABEL_COLS)
    df = pd.read_csv(LABELS_CSV)
    if rater:
        df = df[df["rater"] == rater]
    return df.drop_duplicates("candidate_id", keep="last")


def save_label(row: dict) -> None:
    df = pd.read_csv(LABELS_CSV) if LABELS_CSV.exists() else pd.DataFrame(columns=LABEL_COLS)
    df = df[~((df["candidate_id"] == row["candidate_id"]) & (df["rater"] == row["rater"]))]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELS_CSV, index=False)


def dataset() -> pd.DataFrame:
    labels = load_labels()
    rows = []
    for c in store.load_candidates():
        f = c["features"]
        if not f:
            continue
        rows.append({"candidate_id": c["id"], "anime": c["anime"], "topic": c["topic"],
                     "group": f"{c['anime']}|{c['topic']}", **{k: f.get(k, math.nan) for k in FEATURES + BASELINES},
                     "heuristic": heuristic_score(f)})
    df = pd.DataFrame(rows)
    if df.empty or labels.empty:
        return df.iloc[0:0]
    df = df.merge(labels, on="candidate_id")
    df["y"] = (df["overall"] == "good").astype(int)
    return df


def models():
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"))
    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced",
                                              random_state=0))
    return {"logistic_regression": lr, "random_forest": rf}


def evaluate(n_splits: int = 5) -> dict:
    df = dataset()
    if len(df) < 30 or df["y"].nunique() < 2:
        return {"error": f"need at least 30 labelled candidates with both classes; have {len(df)}"}
    X, y, g = df[FEATURES].to_numpy(float), df["y"].to_numpy(), df["group"].to_numpy()
    k = min(n_splits, df["group"].nunique())
    oof = {name: np.zeros(len(df)) for name in models()}
    for tr, te in GroupKFold(n_splits=k).split(X, y, g):
        for name, m in models().items():
            if len(np.unique(y[tr])) < 2:  # a fold whose training part has one class: predict its base rate
                oof[name][te] = y[tr].mean()
                continue
            m.fit(X[tr], y[tr])
            oof[name][te] = m.predict_proba(X[te])[:, 1]
    scores = {name: p for name, p in oof.items()}
    for b in BASELINES + ["heuristic"]:
        scores[b] = df[b].fillna(df[b].median()).to_numpy(float)

    def auc_ci(p, n_boot=1000, seed=0):
        rng = np.random.default_rng(seed)
        groups = np.unique(g)
        vals = []
        for _ in range(n_boot):  # cluster bootstrap over (anime, topic) pairs
            pick = rng.choice(groups, len(groups), replace=True)
            idx = np.concatenate([np.where(g == q)[0] for q in pick])
            if len(np.unique(y[idx])) == 2:
                vals.append(roc_auc_score(y[idx], p[idx]))
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [math.nan, math.nan]

    res = {"n": int(len(df)), "n_good": int(y.sum()), "n_groups": int(df["group"].nunique()), "folds": int(k),
           "auc": {}, "auc_ci95": {}, "roc": {}}
    for name, p in scores.items():
        res["auc"][name] = float(roc_auc_score(y, p))
        res["auc_ci95"][name] = auc_ci(p)
        fpr, tpr, _ = roc_curve(y, p)
        res["roc"][name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    # fit on everything for the app + explanations
    final = models()
    for m in final.values():
        m.fit(X, y)
    lr = final["logistic_regression"][-1]
    rf = final["random_forest"][-1]
    res["lr_coef"] = dict(zip(FEATURES, lr.coef_[0].tolist()))
    res["rf_importance"] = dict(zip(FEATURES, rf.feature_importances_.tolist()))
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS_DIR / "ranker.joblib")
    (RESULTS_DIR / "ranker_eval.json").write_text(json.dumps(res, indent=1))
    return res


_loaded = None


def score(features: dict, which: str = "logistic_regression") -> float:
    """Probability the analogy is 'good'. Falls back to the heuristic before a model exists."""
    global _loaded
    path = MODELS_DIR / "ranker.joblib"
    if _loaded is None and path.exists():
        # Safe: this file is only ever written by evaluate() above on this machine; never load
        # a ranker.joblib downloaded from elsewhere (joblib/pickle can execute code).
        _loaded = joblib.load(path)
    if _loaded is None:
        return heuristic_score(features)
    x = np.array([[features.get(k, math.nan) for k in FEATURES]], dtype=float)
    return float(_loaded[which].predict_proba(x)[0, 1])


def model_ready() -> bool:
    return (MODELS_DIR / "ranker.joblib").exists()
