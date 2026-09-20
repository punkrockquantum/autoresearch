"""Per-channel conversation state, persisted so a restart keeps context.

A "channel key" is something like ``whatsapp:4477…`` or ``mic:local`` — one
running conversation per place you speak from.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class Session:
    key: str
    target: str
    turns: list[Turn] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    last_reply: str = ""

    def history(self, limit: int) -> list[Turn]:
        return self.turns[-limit:] if limit > 0 else []


class SessionStore:
    """Small JSON-backed store. Safe for the handful of writers we have."""

    def __init__(self, path: str | os.PathLike[str], *, default_target: str, ttl_minutes: int, max_turns: int) -> None:
        self.path = Path(path).expanduser()
        self.default_target = default_target
        self.ttl_seconds = max(ttl_minutes, 0) * 60
        self.max_turns = max_turns
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for key, blob in raw.get("sessions", {}).items():
            self._sessions[key] = Session(
                key=key,
                target=blob.get("target", self.default_target),
                turns=[Turn(**turn) for turn in blob.get("turns", [])],
                updated_at=blob.get("updated_at", time.time()),
                last_reply=blob.get("last_reply", ""),
            )
        self._expire()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sessions": {
                key: {
                    "target": session.target,
                    "turns": [asdict(turn) for turn in session.turns],
                    "updated_at": session.updated_at,
                    "last_reply": session.last_reply,
                }
                for key, session in self._sessions.items()
            }
        }
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, prefix=".sessions-", suffix=".json", delete=False
        )
        try:
            with handle:
                json.dump(payload, handle)
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def _expire(self) -> None:
        if not self.ttl_seconds:
            return
        cutoff = time.time() - self.ttl_seconds
        stale = [key for key, session in self._sessions.items() if session.updated_at < cutoff]
        for key in stale:
            # Keep the chosen target across an idle gap, drop the transcript.
            self._sessions[key].turns.clear()
            self._sessions[key].updated_at = time.time()

    def get(self, key: str) -> Session:
        with self._lock:
            self._expire()
            session = self._sessions.get(key)
            if session is None:
                session = Session(key=key, target=self.default_target)
                self._sessions[key] = session
            return session

    def append(self, key: str, role: str, content: str) -> None:
        with self._lock:
            session = self.get(key)
            session.turns.append(Turn(role=role, content=content))
            if self.max_turns > 0:
                del session.turns[: max(0, len(session.turns) - self.max_turns)]
            session.updated_at = time.time()
            if role == "assistant":
                session.last_reply = content
            self._save()

    def set_target(self, key: str, target: str) -> None:
        with self._lock:
            session = self.get(key)
            session.target = target
            session.updated_at = time.time()
            self._save()

    def reset(self, key: str) -> None:
        with self._lock:
            session = self.get(key)
            session.turns.clear()
            session.last_reply = ""
            session.updated_at = time.time()
            self._save()

    def summary(self) -> dict[str, dict[str, object]]:
        with self._lock:
            return {
                key: {"target": s.target, "turns": len(s.turns), "updated_at": s.updated_at}
                for key, s in self._sessions.items()
            }
