"""Shared shape for anything that can deliver a spoken line to the gateway."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Inbound:
    """One inbound message, already reduced to text or to raw audio."""

    channel: str
    sender: str
    text: str = ""
    audio: bytes | None = None
    audio_filename: str = "audio.ogg"
    #: Provider-side handle for audio not yet downloaded (webhooks stay fast by
    #: acknowledging first and fetching the media afterwards).
    media_id: str = ""
    message_id: str = ""
    #: Where a reply should go (chat id, phone number). Defaults to ``sender``.
    reply_to: str = ""

    def __post_init__(self) -> None:
        if not self.reply_to:
            self.reply_to = self.sender

    @property
    def channel_key(self) -> str:
        return f"{self.channel}:{self.sender}"
