from __future__ import annotations

import pytest

from voicebridge.backends.base import Backend, BackendError, Request, Result
from voicebridge.bridge import Bridge
from voicebridge.config import Config
from voicebridge.sessions import SessionStore


class FakeBackend(Backend):
    """Records what it was asked and returns a canned answer."""

    def __init__(self, name: str, *, reply: str = "", ready: bool = True, reason: str = "", fail: str = "") -> None:
        self.name = name
        self.reply = reply or f"{name} says hello"
        self.ready = ready
        self.reason = reason
        self.fail = fail
        self.calls: list[Request] = []

    def available(self) -> tuple[bool, str]:
        return self.ready, self.reason

    def run(self, request: Request) -> Result:
        self.calls.append(request)
        if self.fail:
            raise BackendError(self.fail)
        return Result(text=self.reply)


class FakeTranscriber:
    def __init__(self, text: str = "claude what is the time") -> None:
        self.text = text
        self.calls: list[tuple[bytes, str]] = []

    def available(self) -> tuple[bool, str]:
        return True, ""

    def transcribe(self, audio: bytes, *, filename: str = "audio.ogg") -> str:
        self.calls.append((audio, filename))
        return self.text


class FakeSynthesizer:
    extension = ".ogg"
    mime = "audio/ogg"

    def __init__(self, *, ready: bool = True) -> None:
        self.ready = ready
        self.calls: list[str] = []

    def available(self) -> tuple[bool, str]:
        return self.ready, "" if self.ready else "disabled in tests"

    def synthesize(self, text: str) -> bytes:
        self.calls.append(text)
        return b"OggS-fake-audio"


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        default_target="claude",
        state_dir=str(tmp_path / "state"),
        ingest_token="test-token",
        spoken_word_limit=40,
    )


@pytest.fixture
def backends() -> dict[str, Backend]:
    return {
        "claude": FakeBackend("claude", reply="Claude's answer."),
        "chatgpt": FakeBackend("chatgpt", reply="ChatGPT's answer."),
        "perplexity": FakeBackend("perplexity", reply="Perplexity's answer."),
        "cursor": FakeBackend("cursor", reply="Cursor agent started."),
    }


@pytest.fixture
def sessions(config) -> SessionStore:
    return SessionStore(
        config.state_path / "sessions.json",
        default_target=config.default_target,
        ttl_minutes=config.session_ttl_minutes,
        max_turns=8,
    )


@pytest.fixture
def bridge(config, backends, sessions) -> Bridge:
    return Bridge(
        config,
        sessions=sessions,
        backends=backends,
        transcriber=FakeTranscriber(),
        synthesizer=FakeSynthesizer(),
    )
