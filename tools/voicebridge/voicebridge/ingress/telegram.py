"""Telegram ingress.

The glasses can't dictate into Telegram directly ("Hey Meta, send a message"
covers WhatsApp, Messenger and Instagram), but Telegram is the easiest channel
to stand up — a bot token and nothing else — and it is a good way to test the
whole pipeline, to send voice notes from the phone, and to receive answers as
voice notes that play through the glasses.
"""

from __future__ import annotations

from ..config import TelegramConfig
from ..http import post_multipart, request_bytes, request_json
from .base import Inbound

API_HOST = "https://api.telegram.org"


def chat_allowed(config: TelegramConfig, chat_id: str) -> bool:
    if not config.allowed_chat_ids:
        return False
    return str(chat_id) in {str(allowed) for allowed in config.allowed_chat_ids}


def parse_update(update: dict) -> list[Inbound]:
    message = update.get("message") or update.get("edited_message")
    if not isinstance(message, dict):
        return []
    chat_id = str((message.get("chat") or {}).get("id") or "")
    if not chat_id:
        return []
    message_id = str(message.get("message_id", ""))

    text = (message.get("text") or message.get("caption") or "").strip()
    if text:
        return [Inbound(channel="telegram", sender=chat_id, text=text, message_id=message_id)]

    media = message.get("voice") or message.get("audio") or message.get("video_note")
    if isinstance(media, dict) and media.get("file_id"):
        return [
            Inbound(
                channel="telegram",
                sender=chat_id,
                media_id=str(media["file_id"]),
                audio_filename="voice.ogg",
                message_id=message_id,
            )
        ]
    return []


def fetch_media(config: TelegramConfig, file_id: str) -> tuple[bytes, str]:
    meta = request_json("GET", f"{API_HOST}/bot{config.bot_token}/getFile?file_id={file_id}")
    file_path = ((meta.get("result") or {}).get("file_path")) if isinstance(meta, dict) else None
    if not file_path:
        raise RuntimeError(f"telegram gave no file_path for {file_id}")
    content, _ = request_bytes("GET", f"{API_HOST}/file/bot{config.bot_token}/{file_path}")
    suffix = file_path.rsplit(".", 1)[-1] if "." in file_path else "ogg"
    return content, f"voice.{suffix}"


def send_text(config: TelegramConfig, chat_id: str, text: str) -> None:
    request_json(
        "POST",
        f"{API_HOST}/bot{config.bot_token}/sendMessage",
        json_body={"chat_id": chat_id, "text": text[:4000], "disable_web_page_preview": True},
    )


def send_voice(config: TelegramConfig, chat_id: str, audio: bytes, *, filename: str, mime: str) -> None:
    post_multipart(
        f"{API_HOST}/bot{config.bot_token}/sendVoice",
        fields={"chat_id": str(chat_id)},
        files={"voice": (filename, audio, mime)},
    )
