"""SQLite store: an LLM response cache (so every experiment is reproducible and the demo works
offline for cached pairs) plus run logs for candidates, lessons and checks."""
from __future__ import annotations

import json
import sqlite3
import threading
import time

from .config import DB_PATH

_LOCK = threading.Lock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY, model TEXT, prompt TEXT, response TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS embed_cache (
    key TEXT PRIMARY KEY, model TEXT, vector TEXT
);
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY, anime TEXT, topic TEXT, round INTEGER, idx INTEGER,
    data TEXT, features TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS lessons (
    id TEXT PRIMARY KEY, anime TEXT, topic TEXT, condition TEXT, sample INTEGER,
    data TEXT, created REAL
);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.executescript(SCHEMA)
    return con


_CON = None


def con() -> sqlite3.Connection:
    global _CON
    if _CON is None:
        _CON = connect()
    return _CON


def cache_get(key: str) -> str | None:
    with _LOCK:
        row = con().execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def cache_put(key: str, model: str, prompt: str, response: str) -> None:
    with _LOCK:
        con().execute("INSERT OR REPLACE INTO llm_cache VALUES (?,?,?,?,?)",
                      (key, model, prompt, response, time.time()))
        con().commit()


def embed_get(key: str):
    with _LOCK:
        row = con().execute("SELECT vector FROM embed_cache WHERE key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def embed_put(key: str, model: str, vec) -> None:
    with _LOCK:
        con().execute("INSERT OR REPLACE INTO embed_cache VALUES (?,?,?)", (key, model, json.dumps(vec)))
        con().commit()


def save_candidate(cid, anime, topic, rnd, idx, data, features=None):
    with _LOCK:
        con().execute("INSERT OR REPLACE INTO candidates VALUES (?,?,?,?,?,?,?,?)",
                      (cid, anime, topic, rnd, idx, json.dumps(data), json.dumps(features or {}), time.time()))
        con().commit()


def update_features(cid, features):
    with _LOCK:
        con().execute("UPDATE candidates SET features=? WHERE id=?", (json.dumps(features), cid))
        con().commit()


def load_candidates(anime=None, topic=None) -> list[dict]:
    q, args = "SELECT id, anime, topic, round, idx, data, features FROM candidates WHERE 1=1", []
    if anime:
        q += " AND anime=?"
        args.append(anime)
    if topic:
        q += " AND topic=?"
        args.append(topic)
    with _LOCK:
        rows = con().execute(q + " ORDER BY anime, topic, round, idx", args).fetchall()
    return [dict(id=r[0], anime=r[1], topic=r[2], round=r[3], idx=r[4], data=json.loads(r[5]),
                 features=json.loads(r[6] or "{}")) for r in rows]


def save_lesson(lid, anime, topic, condition, sample, data):
    with _LOCK:
        con().execute("INSERT OR REPLACE INTO lessons VALUES (?,?,?,?,?,?,?)",
                      (lid, anime, topic, condition, sample, json.dumps(data), time.time()))
        con().commit()


def load_lessons(condition=None) -> list[dict]:
    q, args = "SELECT id, anime, topic, condition, sample, data FROM lessons", []
    if condition:
        q += " WHERE condition=?"
        args.append(condition)
    with _LOCK:
        rows = con().execute(q + " ORDER BY anime, topic, condition, sample", args).fetchall()
    return [dict(id=r[0], anime=r[1], topic=r[2], condition=r[3], sample=r[4], data=json.loads(r[5]))
            for r in rows]


def get_lesson(lid) -> dict | None:
    with _LOCK:
        r = con().execute("SELECT id, anime, topic, condition, sample, data FROM lessons WHERE id=?",
                          (lid,)).fetchone()
    return dict(id=r[0], anime=r[1], topic=r[2], condition=r[3], sample=r[4], data=json.loads(r[5])) if r else None
