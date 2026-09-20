from __future__ import annotations

import time

from voicebridge.sessions import SessionStore


def _store(tmp_path, **kwargs):
    defaults = {"default_target": "claude", "ttl_minutes": 60, "max_turns": 4}
    defaults.update(kwargs)
    return SessionStore(tmp_path / "state" / "sessions.json", **defaults)


def test_new_session_uses_the_default_target(tmp_path):
    store = _store(tmp_path)
    assert store.get("whatsapp:41790000000").target == "claude"


def test_history_is_capped(tmp_path):
    store = _store(tmp_path, max_turns=4)
    for index in range(6):
        store.append("mic:local", "user", f"line {index}")
    turns = store.get("mic:local").turns
    assert len(turns) == 4
    assert turns[0].content == "line 2"


def test_state_survives_a_restart(tmp_path):
    store = _store(tmp_path)
    store.set_target("http:simon", "perplexity")
    store.append("http:simon", "user", "what's new")
    store.append("http:simon", "assistant", "plenty")

    reopened = _store(tmp_path)
    session = reopened.get("http:simon")
    assert session.target == "perplexity"
    assert [turn.content for turn in session.turns] == ["what's new", "plenty"]
    assert session.last_reply == "plenty"


def test_reset_keeps_the_target_but_drops_the_transcript(tmp_path):
    store = _store(tmp_path)
    store.set_target("mic:local", "cursor")
    store.append("mic:local", "user", "hello")
    store.reset("mic:local")
    session = store.get("mic:local")
    assert session.turns == []
    assert session.last_reply == ""
    assert session.target == "cursor"


def test_expiry_drops_stale_transcripts_only(tmp_path):
    store = _store(tmp_path, ttl_minutes=30)
    store.set_target("mic:local", "chatgpt")
    store.append("mic:local", "user", "old question")
    store._sessions["mic:local"].updated_at = time.time() - 3600

    session = store.get("mic:local")
    assert session.turns == []
    assert session.target == "chatgpt"


def test_corrupt_state_file_is_tolerated(tmp_path):
    path = tmp_path / "state" / "sessions.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    store = SessionStore(path, default_target="claude", ttl_minutes=60, max_turns=4)
    assert store.get("mic:local").target == "claude"


def test_history_window(tmp_path):
    store = _store(tmp_path, max_turns=10)
    for index in range(6):
        store.append("mic:local", "user", f"line {index}")
    assert len(store.get("mic:local").history(2)) == 2
    assert store.get("mic:local").history(0) == []
