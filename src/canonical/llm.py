"""Thin Gemini wrapper: picks an available model, spaces calls under the free-tier rate limit,
retries on 429/5xx, returns parsed JSON, and caches every response in SQLite by a hash of
(model, system, prompt, sample). Re-running an experiment costs nothing and gives identical output.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time

from . import store
from .config import DATA, GEMINI_API_KEY, GEMINI_EMBED_MODEL, GEMINI_MODEL, GEMINI_RPM

# The auto-picked model names are remembered here so cached results stay usable offline.
CHOICE = DATA / "model_choice.json"


def _remember(kind: str, name: str) -> None:
    d = json.loads(CHOICE.read_text()) if CHOICE.exists() else {}
    d[kind] = name
    CHOICE.parent.mkdir(parents=True, exist_ok=True)
    CHOICE.write_text(json.dumps(d, indent=1))


def _recall(kind: str) -> str | None:
    return json.loads(CHOICE.read_text()).get(kind) if CHOICE.exists() else None


class LLMError(RuntimeError):
    pass


_client = None
_model = GEMINI_MODEL or None
_embed_model = GEMINI_EMBED_MODEL or None
_last_call = 0.0
_rate_lock = threading.Lock()


def client():
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set. Put it in canonical/.env (see .env.example).")
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _pick(names: list[str], prefer: list[str], avoid: list[str]) -> str | None:
    for p in prefer:
        for n in names:
            if p in n and not any(a in n for a in avoid):
                return n
    return names[0] if names else None


def model_name() -> str:
    """Best available Flash text model for this key (override with GEMINI_MODEL)."""
    global _model
    if _model:
        return _model
    _model = _recall("text")
    if _model:
        return _model
    names = []
    for m in client().models.list():
        acts = getattr(m, "supported_actions", None) or []
        if "generateContent" in acts and "gemini" in m.name:
            names.append(m.name.removeprefix("models/"))
    # newest first by version number in the name
    def ver(n):
        v = re.findall(r"gemini-(\d+(?:\.\d+)?)", n)
        return float(v[0]) if v else 0.0
    names.sort(key=ver, reverse=True)
    _model = _pick(names, ["flash"], ["lite", "image", "tts", "audio", "live", "thinking", "preview", "exp"]) \
        or _pick(names, ["flash"], ["image", "tts", "audio", "live"])
    if not _model:
        raise LLMError("No Gemini text model available for this key.")
    _remember("text", _model)
    return _model


def embed_model_name() -> str:
    global _embed_model
    if _embed_model:
        return _embed_model
    _embed_model = _recall("embed")
    if _embed_model:
        return _embed_model
    names = [m.name.removeprefix("models/") for m in client().models.list()
             if "embedContent" in (getattr(m, "supported_actions", None) or [])]
    _embed_model = _pick(names, ["gemini-embedding", "text-embedding"], ["exp"]) or _pick(names, ["embedding"], [])
    if not _embed_model:
        raise LLMError("No Gemini embedding model available for this key.")
    _remember("embed", _embed_model)
    return _embed_model


def _wait_turn():
    global _last_call
    with _rate_lock:
        gap = 60.0 / max(GEMINI_RPM, 0.1)
        now = time.time()
        if now - _last_call < gap:
            time.sleep(gap - (now - _last_call))
        _last_call = time.time()


def _key(*parts) -> str:
    return hashlib.sha256("\x1f".join(str(p) for p in parts).encode()).hexdigest()


def generate(prompt: str, system: str = "", *, json_mode: bool = False, temperature: float = 0.7,
             sample: int = 0, cache_only: bool = False) -> str:
    """Text (or JSON text) from Gemini, cached. `sample` distinguishes repeated draws."""
    from google.genai import types

    model = model_name()
    key = _key(model, system, prompt, json_mode, temperature, sample)
    hit = store.cache_get(key)
    if hit is not None:
        return hit
    if cache_only:
        raise LLMError("not cached")
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system or None,
        response_mime_type="application/json" if json_mode else None,
    )
    delay = 5.0
    for attempt in range(6):
        _wait_turn()
        try:
            r = client().models.generate_content(model=model, contents=prompt, config=cfg)
            text = r.text or ""
            if not text.strip():
                raise LLMError("empty response")
            store.cache_put(key, model, prompt, text)
            return text
        except Exception as e:  # rate limits and transient server errors: back off and retry
            msg = str(e)
            transient = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE",
                                               "DEADLINE", "empty response", "overloaded"))
            if not transient or attempt == 5:
                raise LLMError(msg[:500]) from e
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise LLMError("unreachable")


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if m:
            return json.loads(m.group(1))
        raise


def generate_json(prompt: str, system: str = "", **kw):
    return parse_json(generate(prompt, system, json_mode=True, **kw))


def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embeddings with a per-text cache; batches of up to 100."""
    from google.genai import types

    model = embed_model_name()
    out: list = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        v = store.embed_get(_key(model, task, t))
        if v is None:
            todo.append(i)
        else:
            out[i] = v
    for s in range(0, len(todo), 100):
        idx = todo[s:s + 100]
        batch = [texts[i] for i in idx]
        delay = 5.0
        for attempt in range(6):
            _wait_turn()
            try:
                r = client().models.embed_content(model=model, contents=batch,
                                                  config=types.EmbedContentConfig(task_type=task))
                break
            except Exception as e:
                if attempt == 5 or not any(s_ in str(e) for s_ in ("429", "RESOURCE_EXHAUSTED", "500", "503")):
                    raise LLMError(str(e)[:500]) from e
                time.sleep(delay)
                delay = min(delay * 2, 60)
        for i, emb in zip(idx, r.embeddings):
            out[i] = list(emb.values)
            store.embed_put(_key(model, task, texts[i]), model, out[i])
    return out
