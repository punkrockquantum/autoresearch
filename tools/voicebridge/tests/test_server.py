from __future__ import annotations

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from voicebridge.ingress import whatsapp as wa
from voicebridge.server import create_app

AUTH = {"Authorization": "Bearer test-token"}


@pytest.fixture
def client(config, bridge):
    config.whatsapp.enabled = True
    config.whatsapp.app_secret = "app-secret"
    config.whatsapp.verify_token = "verify-me"
    config.whatsapp.access_token = "graph-token"
    config.whatsapp.phone_number_id = "1234567890"
    config.whatsapp.allowed_senders = ["+41 79 000 00 00"]
    config.telegram.enabled = True
    config.telegram.bot_token = "bot-token"
    config.telegram.secret_token = "tg-secret"
    config.telegram.allowed_chat_ids = ["999"]
    return TestClient(create_app(config, bridge))


def _signed(body: dict, secret: str = "app-secret") -> tuple[bytes, dict[str, str]]:
    raw = json.dumps(body).encode()
    digest = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return raw, {"X-Hub-Signature-256": f"sha256={digest}", "Content-Type": "application/json"}


def _whatsapp_text(body: str, sender: str = "41790000000") -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "messages": [
                                {"from": sender, "id": "wamid.1", "type": "text", "text": {"body": body}}
                            ],
                        }
                    }
                ]
            }
        ],
    }


def test_healthz_needs_no_auth(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_ingest_rejects_a_missing_token(client):
    assert client.post("/v1/ingest", json={"text": "claude hi"}).status_code == 401
    assert client.post(
        "/v1/ingest", json={"text": "claude hi"}, headers={"Authorization": "Bearer wrong"}
    ).status_code == 401


def test_ingest_is_disabled_without_a_configured_token(config, bridge):
    config.ingest_token = ""
    disabled = TestClient(create_app(config, bridge))
    assert disabled.post("/v1/ingest", json={"text": "hi"}, headers=AUTH).status_code == 503


def test_ingest_answers_text_synchronously(client, backends):
    response = client.post(
        "/v1/ingest", json={"text": "perplexity what's new", "speaker": "simon"}, headers=AUTH
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["target"] == "perplexity"
    assert payload["reply"] == "Perplexity's answer."
    assert payload["speech"]
    assert "audio_base64" not in payload
    assert backends["perplexity"].calls[0].prompt == "what's new"


def test_ingest_can_return_spoken_audio(client):
    response = client.post("/v1/ingest", json={"text": "claude hi", "speak": True}, headers=AUTH)
    payload = response.json()
    assert base64.b64decode(payload["audio_base64"]) == b"OggS-fake-audio"
    assert payload["audio_mime"] == "audio/ogg"


def test_ingest_accepts_base64_audio(client, bridge):
    bridge.transcriber.text = "chatgpt summarise this"
    audio = base64.b64encode(b"fake-audio-bytes").decode()
    response = client.post(
        "/v1/ingest", json={"audio_base64": audio, "filename": "clip.m4a"}, headers=AUTH
    )
    payload = response.json()
    assert payload["transcript"] == "chatgpt summarise this"
    assert payload["target"] == "chatgpt"
    assert bridge.transcriber.calls[0] == (b"fake-audio-bytes", "clip.m4a")


def test_ingest_rejects_bad_base64(client):
    response = client.post("/v1/ingest", json={"audio_base64": "not base64!!"}, headers=AUTH)
    assert response.status_code == 400


def test_ingest_target_parameter_forces_the_assistant(client, backends):
    response = client.post(
        "/v1/ingest", json={"text": "summarise the readme", "target": "chatgpt"}, headers=AUTH
    )
    assert response.json()["target"] == "chatgpt"
    assert backends["chatgpt"].calls


def test_ingest_audio_upload(client, bridge):
    bridge.transcriber.text = "claude hello there"
    response = client.post(
        "/v1/ingest/audio",
        files={"file": ("clip.wav", b"RIFFfake", "audio/wav")},
        data={"speaker": "simon"},
        headers=AUTH,
    )
    assert response.status_code == 200
    assert response.json()["transcript"] == "claude hello there"


def test_whatsapp_verification_handshake(client):
    ok = client.get(
        "/webhook/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "42"},
    )
    assert ok.status_code == 200
    assert ok.text == "42"

    bad = client.get(
        "/webhook/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "42"},
    )
    assert bad.status_code == 403


def test_whatsapp_rejects_an_unsigned_payload(client):
    response = client.post("/webhook/whatsapp", json=_whatsapp_text("claude hi"))
    assert response.status_code == 401


def test_whatsapp_rejects_a_forged_signature(client):
    raw, headers = _signed(_whatsapp_text("claude hi"), secret="wrong-secret")
    response = client.post("/webhook/whatsapp", content=raw, headers=headers)
    assert response.status_code == 401


def test_whatsapp_ignores_senders_off_the_allowlist(client, backends):
    raw, headers = _signed(_whatsapp_text("claude hi", sender="19995550000"))
    response = client.post("/webhook/whatsapp", content=raw, headers=headers)
    assert response.status_code == 200
    assert response.json()["accepted"] == 0
    assert not backends["claude"].calls


def test_whatsapp_answers_an_allowed_sender(client, backends, monkeypatch):
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "voicebridge.server.wa.send_text", lambda config, to, text: sent.append((to, text))
    )
    raw, headers = _signed(_whatsapp_text("claude what is bits per byte"))
    response = client.post("/webhook/whatsapp", content=raw, headers=headers)

    assert response.json()["accepted"] == 1
    assert backends["claude"].calls[0].prompt == "what is bits per byte"
    to, text = sent[0]
    assert to == "41790000000"
    # The reply echoes the transcript so a mishearing is obvious.
    assert "what is bits per byte" in text
    assert "Claude's answer." in text


def test_whatsapp_voice_note_is_transcribed(client, backends, bridge, monkeypatch):
    bridge.transcriber.text = "perplexity who won the race"
    monkeypatch.setattr("voicebridge.server.wa.send_text", lambda config, to, text: None)
    monkeypatch.setattr(
        "voicebridge.server.wa.fetch_media", lambda config, media_id: (b"voice-bytes", "note.ogg")
    )
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "from": "41790000000",
                                    "id": "wamid.2",
                                    "type": "audio",
                                    "audio": {"id": "media-99", "mime_type": "audio/ogg; codecs=opus"},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
    raw, headers = _signed(payload)
    assert client.post("/webhook/whatsapp", content=raw, headers=headers).json()["accepted"] == 1
    assert backends["perplexity"].calls[0].prompt == "who won the race"
    assert bridge.transcriber.calls[0] == (b"voice-bytes", "note.ogg")


def test_whatsapp_status_callbacks_are_ignored(client):
    raw, headers = _signed({"entry": [{"changes": [{"value": {"statuses": [{"status": "read"}]}}]}]})
    assert client.post("/webhook/whatsapp", content=raw, headers=headers).json()["accepted"] == 0


def test_telegram_requires_the_secret_header(client):
    payload = {"message": {"chat": {"id": 999}, "message_id": 1, "text": "claude hi"}}
    assert client.post("/webhook/telegram", json=payload).status_code == 401
    response = client.post(
        "/webhook/telegram", json=payload, headers={"X-Telegram-Bot-Api-Secret-Token": "nope"}
    )
    assert response.status_code == 401


def test_telegram_answers_an_allowed_chat(client, backends, monkeypatch):
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "voicebridge.server.tg.send_text", lambda config, chat, text: sent.append((chat, text))
    )
    monkeypatch.setattr("voicebridge.server.tg.send_voice", lambda *a, **k: None)
    payload = {"message": {"chat": {"id": 999}, "message_id": 7, "text": "cursor add a test"}}
    response = client.post(
        "/webhook/telegram", json=payload, headers={"X-Telegram-Bot-Api-Secret-Token": "tg-secret"}
    )
    assert response.json()["accepted"] == 1
    assert backends["cursor"].calls[0].prompt == "add a test"
    assert sent[0][0] == "999"


def test_telegram_ignores_other_chats(client, backends):
    payload = {"message": {"chat": {"id": 12345}, "message_id": 7, "text": "claude hi"}}
    response = client.post(
        "/webhook/telegram", json=payload, headers={"X-Telegram-Bot-Api-Secret-Token": "tg-secret"}
    )
    assert response.json()["accepted"] == 0
    assert not backends["claude"].calls


def test_status_endpoint_reports_readiness(client):
    payload = client.get("/v1/status", headers=AUTH).json()
    assert payload["diagnostics"]["default_target"] == "claude"
    assert payload["diagnostics"]["backends"]["claude"]["ready"] is True


def test_signature_helper_rejects_empty_secret():
    assert wa.verify_signature("", b"{}", "sha256=whatever") is False
    assert wa.verify_signature("secret", b"{}", None) is False
