"""WhatsApp Cloud API ingress — the fully hands-free route.

"Hey Meta, send a message to <bridge contact> on WhatsApp" lets the glasses
dictate and send without touching the phone; the reply is read back by the
glasses' notification readout. A voice note lands here as audio instead and gets
transcribed.
"""

from __future__ import annotations

import hashlib
import hmac
import logging

from ..config import WhatsAppConfig
from ..http import HttpError, post_multipart, request_bytes, request_json
from .base import Inbound

log = logging.getLogger(__name__)

GRAPH_HOST = "https://graph.facebook.com"


def verify_signature(app_secret: str, raw_body: bytes, header: str | None) -> bool:
    """Check Meta's ``X-Hub-Signature-256`` header against the raw request body."""
    if not app_secret:
        return False
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header[len("sha256=") :].strip())


def sender_allowed(config: WhatsAppConfig, sender: str) -> bool:
    """Only numbers on the allowlist may spend your API credits."""
    if not config.allowed_senders:
        return False
    digits = _digits(sender)
    return any(_digits(allowed) == digits for allowed in config.allowed_senders)


def parse_webhook(payload: dict) -> list[Inbound]:
    """Pull the user-authored messages out of a Cloud API webhook payload.

    Status callbacks (delivered/read receipts) and unsupported message types are
    skipped rather than answered.
    """
    messages: list[Inbound] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            for message in value.get("messages") or []:
                sender = str(message.get("from") or "")
                if not sender:
                    continue
                kind = message.get("type")
                if kind == "text":
                    body = (message.get("text") or {}).get("body", "")
                    if body.strip():
                        messages.append(
                            Inbound(
                                channel="whatsapp",
                                sender=sender,
                                text=body.strip(),
                                message_id=str(message.get("id", "")),
                            )
                        )
                elif kind in {"audio", "voice"}:
                    media = message.get(kind) or {}
                    media_id = str(media.get("id", ""))
                    if media_id:
                        messages.append(
                            Inbound(
                                channel="whatsapp",
                                sender=sender,
                                audio_filename=f"{media_id}{_extension(media.get('mime_type', ''))}",
                                media_id=media_id,
                                message_id=str(message.get("id", "")),
                            )
                        )
                else:
                    log.info("ignoring whatsapp message of type %s", kind)
    return messages


def fetch_media(config: WhatsAppConfig, media_id: str) -> tuple[bytes, str]:
    """Resolve a media id to bytes. Returns ``(content, filename)``."""
    meta = request_json(
        "GET",
        f"{GRAPH_HOST}/{config.graph_version}/{media_id}",
        headers={"Authorization": f"Bearer {config.access_token}"},
    )
    url = meta.get("url")
    if not url:
        raise RuntimeError(f"no download url for media {media_id}")
    content, _ = request_bytes(
        "GET", url, headers={"Authorization": f"Bearer {config.access_token}"}
    )
    return content, f"{media_id}{_extension(meta.get('mime_type', ''))}"


def send_text(config: WhatsAppConfig, to: str, text: str) -> None:
    request_json(
        "POST",
        f"{GRAPH_HOST}/{config.graph_version}/{config.phone_number_id}/messages",
        headers={"Authorization": f"Bearer {config.access_token}"},
        json_body={
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {"preview_url": False, "body": text[:4000]},
        },
    )


def send_voice(config: WhatsAppConfig, to: str, audio: bytes, *, filename: str, mime: str) -> None:
    """Upload audio then send it as a voice message."""
    upload = post_multipart(
        f"{GRAPH_HOST}/{config.graph_version}/{config.phone_number_id}/media",
        headers={"Authorization": f"Bearer {config.access_token}"},
        fields={"messaging_product": "whatsapp", "type": mime},
        files={"file": (filename, audio, mime)},
    )
    media_id = upload.get("id")
    if not media_id:
        raise HttpError(500, "media upload", str(upload))
    request_json(
        "POST",
        f"{GRAPH_HOST}/{config.graph_version}/{config.phone_number_id}/messages",
        headers={"Authorization": f"Bearer {config.access_token}"},
        json_body={
            "messaging_product": "whatsapp",
            "to": to,
            "type": "audio",
            "audio": {"id": media_id},
        },
    )


def _extension(mime: str) -> str:
    base = (mime or "").split(";")[0].strip()
    return {
        "audio/ogg": ".ogg",
        "audio/opus": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/mp4": ".m4a",
        "audio/aac": ".aac",
        "audio/amr": ".amr",
        "audio/wav": ".wav",
    }.get(base, ".ogg")


def _digits(value: str) -> str:
    return "".join(char for char in value if char.isdigit())
