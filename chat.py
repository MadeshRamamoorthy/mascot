"""LLM chat with OpenAI -> Ollama fallback.

Two backends. `ChatClient.ask(emp_id, question)` enforces a per-session
question budget and dispatches to whichever backend in
config.CHAT_BACKEND_ORDER is currently reachable.

OpenAI: requires `OPENAI_API_KEY` env var and internet.
Ollama:  reaches a local server (default http://localhost:11434)
         configured per https://www.raspberrypi.com/documentation/computers/ai.html.
         Used as a backup so the kiosk can answer questions offline.

Calls run on a dedicated worker thread so the camera loop never blocks.
The caller submits a question via `submit(emp_id, question)` which
returns a `concurrent.futures.Future`; poll it from the main loop and
read `result()` once done.

`available()` results are cached for AVAILABILITY_TTL_SEC so the camera
loop's per-frame `status()` calls don't hammer the network.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import socket
import time
from typing import Iterable

import requests


# ---- reply post-processing -----------------------------------------------
# LLM answers -- especially from OpenAI's web_search_preview -- come back
# with markdown formatting, citation markers (【1】, [source†1]) and inline
# URLs. Passed to the SPA that renders as raw text (asterisks, brackets,
# huge http:// blobs) AND to Piper TTS which literally reads "star star
# bold star star". Strip everything to plain speakable prose before the
# reply leaves the client.

_MD_IMG_RE     = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_MD_LINK_RE    = re.compile(r"\[([^\]]+)\]\([^)]+\)")            # [text](url)
_CITATION_RE   = re.compile(r"【[^】]*】|\[\^\d+\]|\[\d+\]")     # 【1】, [^1], [1]
_MD_BOLD_RE    = re.compile(r"\*\*([^*\n]+)\*\*|__([^_\n]+)__")
_MD_ITAL_RE    = re.compile(
    r"(?<![*_\w])\*([^*\n]+)\*(?!\*)|(?<![*_\w])_([^_\n]+)_(?!_)"
)
_MD_CODE_RE    = re.compile(r"`([^`\n]+)`")
_MD_HEADER_RE  = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", flags=re.MULTILINE)
_MD_BULLET_RE  = re.compile(r"^[ \t]{0,3}(?:[-*+]|\d+\.)[ \t]+", flags=re.MULTILINE)
_MD_HRULE_RE   = re.compile(r"^[ \t]{0,3}(?:---|\*\*\*|___)[ \t]*$", flags=re.MULTILINE)
_SOURCES_TAIL_RE = re.compile(
    r"\n[ \t]*(?:sources?|references?|citations?|further reading)"
    r"[ \t]*[:\n].*",
    flags=re.IGNORECASE | re.DOTALL,
)
_URL_RE        = re.compile(r"https?://\S+")
_MULTI_WS_RE   = re.compile(r"[ \t]{2,}")
_MULTI_NL_RE   = re.compile(r"\n{3,}")


def _strip_markdown(text: str) -> str:
    """Turn a markdown-formatted LLM reply into plain speakable text."""
    if not text:
        return text
    s = text
    s = _MD_IMG_RE.sub("", s)
    s = _MD_LINK_RE.sub(r"\1", s)          # keep link text, drop URL
    s = _CITATION_RE.sub("", s)
    s = _MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), s)
    s = _MD_ITAL_RE.sub(lambda m: m.group(1) or m.group(2), s)
    s = _MD_CODE_RE.sub(r"\1", s)
    s = _MD_HEADER_RE.sub("", s)
    s = _MD_BULLET_RE.sub("", s)
    s = _MD_HRULE_RE.sub("", s)
    s = _SOURCES_TAIL_RE.sub("", s)
    s = _URL_RE.sub("", s)                 # bare URLs -> gone (TTS-unfriendly)
    s = _MULTI_WS_RE.sub(" ", s)
    s = _MULTI_NL_RE.sub("\n\n", s)
    return s.strip()

import config


BASE_SYSTEM_PROMPT = (
    "You are ECHO SCOPE, a friendly AI kiosk at the Infosys Calgary AI "
    "Club. Your answers are spoken aloud, so ALWAYS keep them to 2-3 "
    "short sentences. Never produce a wall of text, bullet lists, or "
    "long explanations. Be concise and conversational.\n"
    "- Weather: ONE line -- temperature + condition (e.g. 'It's 18°C "
    "and partly cloudy in Calgary right now.'). No hourly breakdowns.\n"
    "- News / current events: 2-3 sentences, just the key facts.\n"
    "- Anything else: 2-3 sentences max.\n"
    "Sound like a person, not a brochure."
)


def build_system_prompt(db=None) -> str:
    """Return the system prompt with the current projects + next session
    folded in. When the user asks "what's on display today?" or "what's
    the next session?" the LLM can answer from this context instead of
    hallucinating. Other questions go through normally."""
    if db is None:
        return BASE_SYSTEM_PROMPT
    try:
        projects = db.list_projects() or []
        session_row = db.next_session()
    except Exception:
        return BASE_SYSTEM_PROMPT

    parts = [BASE_SYSTEM_PROMPT]
    if projects:
        lines = []
        for r in projects:
            title = (r[1] or "").strip()
            desc = (r[2] or "").strip()
            if not title:
                continue
            lines.append(f"  - {title}" + (f": {desc}" if desc else ""))
        if lines:
            parts.append(
                "PROJECTS ON DISPLAY TODAY. When the user asks what "
                "projects / AI files / demos are available, on display, "
                "or what they can see today, list these by name -- and "
                "ONLY these. Do not invent or add any projects that "
                "aren't in this list:\n" + "\n".join(lines)
            )
    if session_row:
        sid, title, starts_at, ends_at, notes = session_row
        when = (starts_at or "").replace("T", " ")
        if ends_at:
            when += f" - {ends_at}"
        ses = (
            f"Next upcoming session (read this out if the user asks "
            f"'what's the next session' / 'what's coming up' / 'when is "
            f"the next event'):\n  Title: {title}\n  When: {when}"
        )
        if notes:
            ses += f"\n  Notes: {notes}"
        parts.append(ses)
    parts.append(
        "Only use the lists above for questions about projects or "
        "sessions. For any other question, answer normally and "
        "conversationally."
    )
    return "\n\n".join(parts)


# Backwards compat for any external importer.
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

AVAILABILITY_TTL_SEC = 5.0


class ChatBudgetError(RuntimeError):
    pass


class ChatBackendError(RuntimeError):
    pass


# ---------------- backends ----------------


class _Backend:
    label = "?"

    def __init__(self):
        self._cache_ok: bool | None = None
        self._cache_at: float = 0.0

    def available(self) -> bool:
        now = time.time()
        if self._cache_ok is not None and now - self._cache_at < AVAILABILITY_TTL_SEC:
            return self._cache_ok
        ok = self._probe()
        self._cache_ok = ok
        self._cache_at = now
        return ok

    def invalidate(self) -> None:
        self._cache_ok = None

    def _probe(self) -> bool:
        raise NotImplementedError

    def complete(self, question: str, system: str) -> str:
        raise NotImplementedError


class OpenAIBackend(_Backend):
    label = "OpenAI"

    def __init__(self, api_key: str, model: str):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.label = f"OpenAI · {model}"
        self._client = None

    def _probe(self) -> bool:
        if not self.api_key:
            return False
        try:
            with socket.create_connection(("api.openai.com", 443), timeout=1.5):
                return True
        except OSError:
            return False

    def complete(self, question: str, system: str, history=None) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        history = history or []
        if getattr(config, "CHAT_WEB_SEARCH", False):
            text = self._complete_websearch(system, history, question)
            if text:
                return text
            # web_search unavailable / empty -> fall through to plain chat
        return self._complete_chat(system, history, question)

    def _messages(self, system, history, question):
        msgs = [{"role": "system", "content": system}]
        for role, text in history:
            if role in ("user", "assistant") and text:
                msgs.append({"role": role, "content": text})
        msgs.append({"role": "user", "content": question})
        return msgs

    def _complete_chat(self, system, history, question) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, history, question),
            max_tokens=getattr(config, "CHAT_MAX_REPLY_TOKENS", 300),
        )
        return resp.choices[0].message.content.strip()

    def _complete_websearch(self, system, history, question) -> str:
        """Use the Responses API with the web_search tool so the model
        can pull current info. Returns '' if the API/tool isn't
        available (caller then falls back to plain chat)."""
        try:
            resp = self._client.responses.create(
                model=self.model,
                tools=[{"type": "web_search_preview"}],
                input=self._messages(system, history, question),
                max_output_tokens=getattr(config, "CHAT_MAX_REPLY_TOKENS", 300),
            )
            return (getattr(resp, "output_text", "") or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"[chat] web_search unavailable, falling back: {exc!r}")
            return ""


class OllamaBackend(_Backend):
    label = "Ollama (local)"

    def __init__(self, url: str, model: str):
        super().__init__()
        self.url = url.rstrip("/")
        self.model = model
        self.label = f"Ollama (local) · {model}"

    def _probe(self) -> bool:
        # Cheap reachability probe via socket (no HTTP). Avoids parsing
        # /api/tags JSON, which Hailo-Ollama can return with a slightly
        # different schema. The actual model error (if any) surfaces
        # when complete() runs.
        try:
            from urllib.parse import urlparse
            u = urlparse(self.url)
            host = u.hostname or "localhost"
            port = u.port or (443 if u.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            return False

    def complete(self, question: str, system: str, history=None) -> str:
        # Fold recent turns into the prompt so the local model has some
        # conversational context too. Ollama /api/generate is single-turn,
        # so we inline the history as plain text.
        history = history or []
        if history:
            convo = "\n".join(
                f"{'User' if role == 'user' else 'Assistant'}: {text}"
                for role, text in history if role in ("user", "assistant") and text
            )
            prompt = f"{convo}\nUser: {question}"
        else:
            prompt = question
        r = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {"num_predict": getattr(config, "CHAT_MAX_REPLY_TOKENS", 300)},
            },
            timeout=60,
        )
        r.raise_for_status()
        return (r.json().get("response") or "").strip()


# ---------------- client + budget ----------------


class ChatBudget:
    """Per-emp_id question count. Reset on session start."""

    def __init__(self, max_questions: int):
        self.max_questions = max_questions
        self._used: dict[str, int] = {}

    def reset(self) -> None:
        self._used.clear()

    def remaining(self, emp_id: str) -> int:
        return max(0, self.max_questions - self._used.get(emp_id, 0))

    def consume(self, emp_id: str) -> int:
        used = self._used.get(emp_id, 0)
        if used >= self.max_questions:
            raise ChatBudgetError(
                f"Lovely chatting! That's {self.max_questions} questions for "
                "now — come say hi again anytime."
            )
        self._used[emp_id] = used + 1
        return self._used[emp_id]


class ChatClient:
    def __init__(self, db=None):
        # When `db` is set, every chat call builds a system prompt that
        # includes the current projects + next session row -- so the
        # LLM can answer those questions from local truth instead of
        # hallucinating. Other questions go through normally.
        self.db = db
        self.budget = ChatBudget(config.CHAT_MAX_QUESTIONS_PER_SESSION)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="chat",
        )
        self._backends: dict[str, _Backend] = {
            "openai": OpenAIBackend(
                os.environ.get("OPENAI_API_KEY", ""),
                config.OPENAI_MODEL,
            ),
            "ollama": OllamaBackend(config.OLLAMA_URL, config.OLLAMA_MODEL),
        }
        self._last_label: str | None = None

    @property
    def last_backend_label(self) -> str | None:
        return self._last_label

    def pick_backend(self) -> _Backend | None:
        for name in config.CHAT_BACKEND_ORDER:
            b = self._backends.get(name)
            if b is not None and b.available():
                return b
        return None

    def status(self) -> str:
        b = self.pick_backend()
        return (b.label if b
                else "Chat is taking a quick break — back online shortly.")

    def submit(self, emp_id: str, question: str, history=None
               ) -> "concurrent.futures.Future[str]":
        # Trim to the most recent N turns so token cost stays bounded.
        turns = getattr(config, "CHAT_HISTORY_TURNS", 0)
        if history and turns > 0:
            history = list(history)[-turns:]
        else:
            history = []
        return self._executor.submit(
            self._ask_blocking, emp_id, question, history,
        )

    # internal -----------------------------------------------------------

    def _ask_blocking(self, emp_id: str, question: str, history=None) -> str:
        backend = self.pick_backend()
        if backend is None:
            raise ChatBackendError("No chat backend reachable.")
        # Reserve a slot before calling the backend so we don't get charged
        # for a network round-trip the user can't actually use.
        self.budget.consume(emp_id)
        system = build_system_prompt(self.db)
        try:
            answer = backend.complete(question, system, history=history)
        except Exception as exc:  # roll back budget on error
            self.budget._used[emp_id] = max(0, self.budget._used.get(emp_id, 1) - 1)
            raise ChatBackendError(f"{backend.label}: {exc}") from exc
        # Turn markdown/citations/URLs into plain speakable prose. Applies
        # to both OpenAI (esp. web_search_preview output) and Ollama, since
        # any backend can decide to format the reply.
        answer = _strip_markdown(answer)
        self._last_label = backend.label
        return answer

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
