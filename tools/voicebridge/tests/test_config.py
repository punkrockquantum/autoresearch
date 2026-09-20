from __future__ import annotations

import json

import pytest

from voicebridge.config import Config


def test_env_populates_keys_and_targets(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-abc")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-abc")
    monkeypatch.setenv("VOICEBRIDGE_DEFAULT_TARGET", "perplexity")
    monkeypatch.setenv("VOICEBRIDGE_ALLOWED_WHATSAPP_SENDERS", "+41790000000, +41791111111")
    config = Config.from_env()

    assert config.default_target == "perplexity"
    assert config.claude.api_key == "sk-ant-abc"
    assert config.perplexity.api_key == "pplx-abc"
    assert config.whatsapp.allowed_senders == ["+41790000000", "+41791111111"]


def test_unknown_default_target_falls_back(monkeypatch):
    monkeypatch.setenv("VOICEBRIDGE_DEFAULT_TARGET", "gemini")
    assert Config.from_env().default_target == "claude"


def test_whatsapp_enables_itself_when_credentials_exist(monkeypatch):
    assert Config.from_env().whatsapp.enabled is False
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "graph-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "12345")
    assert Config.from_env().whatsapp.enabled is True


def test_groq_transcriber_defaults(monkeypatch):
    monkeypatch.setenv("VOICEBRIDGE_STT_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-abc")
    stt = Config.from_env().stt
    assert stt.api_key == "gsk-abc"
    assert "groq.com" in stt.base_url
    assert stt.model.startswith("whisper-large")


def test_json_overlay_wins_over_defaults(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"default_target": "cursor", "cursor": {"mode": "queue", "queue_file": "q.md"}}))
    config = Config.load(path)
    assert config.default_target == "cursor"
    assert config.cursor.mode == "queue"
    assert config.cursor.queue_file == "q.md"


def test_overlay_rejects_unknown_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"nope": 1}))
    with pytest.raises(ValueError):
        Config.load(path)


def test_redacted_config_hides_secrets():
    config = Config()
    config.claude.api_key = "sk-ant-secret-value"
    config.whatsapp.access_token = "graph-token"
    redacted = config.redacted()

    assert "sk-ant" not in json.dumps(redacted)
    assert redacted["claude"]["api_key"].startswith("set (")
    assert redacted["telegram"]["bot_token"] == "unset"
    assert redacted["default_target"] == "claude"
