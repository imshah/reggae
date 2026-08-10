"""Persistent chat history for the UI.

Each session is a JSON file under data/chats/<id>.json. Pure local-file
persistence — no external services — so chats survive refreshes and restarts.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field

from docmind.config import CHATS_DIR, ensure_dirs


@dataclass
class Message:
    role: str                      # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)
    route: str | None = None       # "local" | "remote" | task name
    provider: str | None = None
    model: str | None = None
    cost: float | None = None
    sources: list[str] = field(default_factory=list)
    artifact: str | None = None    # e.g. rendered diagram SVG path (for history re-render)


@dataclass
class Session:
    id: str
    title: str
    created_at: float
    updated_at: float
    messages: list[Message] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Session":
        msgs = [Message(**m) for m in d.get("messages", [])]
        return cls(
            id=d["id"],
            title=d.get("title", "Untitled"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            messages=msgs,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": [asdict(m) for m in self.messages],
        }


class ChatStore:
    def __init__(self) -> None:
        ensure_dirs()

    def _path(self, session_id: str):
        return CHATS_DIR / f"{session_id}.json"

    # --- lifecycle --------------------------------------------------------
    def new_session(self, title: str | None = None) -> Session:
        now = time.time()
        sess = Session(
            id=uuid.uuid4().hex[:12],
            title=title or "New chat",
            created_at=now,
            updated_at=now,
        )
        self._save(sess)
        return sess

    def load(self, session_id: str) -> Session | None:
        p = self._path(session_id)
        if not p.exists():
            return None
        try:
            return Session.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            return None

    def _save(self, sess: Session) -> None:
        sess.updated_at = time.time()
        self._path(sess.id).write_text(json.dumps(sess.to_dict(), indent=2))

    def append(self, session_id: str, message: Message) -> Session | None:
        sess = self.load(session_id)
        if sess is None:
            return None
        # auto-title from the first user message
        if sess.title in ("New chat", "", "Untitled") and message.role == "user":
            sess.title = (message.content[:60] + "…") if len(message.content) > 60 else message.content
        sess.messages.append(message)
        self._save(sess)
        return sess

    def rename(self, session_id: str, title: str) -> None:
        sess = self.load(session_id)
        if sess:
            sess.title = title
            self._save(sess)

    def delete(self, session_id: str) -> None:
        p = self._path(session_id)
        if p.exists():
            p.unlink()

    # --- listing / export -------------------------------------------------
    def list_sessions(self) -> list[Session]:
        sessions: list[Session] = []
        for p in CHATS_DIR.glob("*.json"):
            try:
                sessions.append(Session.from_dict(json.loads(p.read_text())))
            except (json.JSONDecodeError, OSError, KeyError, TypeError):
                continue
        return sorted(sessions, key=lambda s: s.updated_at, reverse=True)

    def export_markdown(self, session_id: str) -> str:
        sess = self.load(session_id)
        if sess is None:
            return ""
        lines = [f"# {sess.title}", ""]
        for m in sess.messages:
            who = "You" if m.role == "user" else "docmind"
            meta = ""
            if m.role == "assistant" and (m.model or m.route):
                bits = [b for b in (m.route, m.model) if b]
                if m.cost:
                    bits.append(f"${m.cost:.4f}")
                meta = f"  _({' · '.join(bits)})_"
            lines.append(f"## {who}{meta}")
            lines.append("")
            lines.append(m.content)
            if m.sources:
                lines.append("")
                lines.append("Sources: " + ", ".join(m.sources))
            lines.append("")
        return "\n".join(lines)
