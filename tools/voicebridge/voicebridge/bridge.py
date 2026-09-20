"""The gateway core: audio or text in, an answer out.

Every ingress path (WhatsApp, Telegram, HTTP, local microphone) funnels through
one ``Bridge`` so the routing rules, conversation memory and spoken-reply
shaping behave identically no matter how you talked to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .backends import Backend, build_backends
from .config import TARGETS, Config
from .router import Reply, Router
from .sessions import SessionStore
from .speech import SpeechError, Synthesizer, Transcriber

log = logging.getLogger(__name__)


@dataclass
class Exchange:
    """What one round trip produced."""

    transcript: str
    reply: Reply

    @property
    def target(self) -> str:
        return self.reply.target


class Bridge:
    def __init__(
        self,
        config: Config,
        *,
        sessions: SessionStore | None = None,
        backends: dict[str, Backend] | None = None,
        transcriber: Transcriber | None = None,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.config = config
        self.sessions = sessions or SessionStore(
            Path(config.state_path) / "sessions.json",
            default_target=config.default_target,
            ttl_minutes=config.session_ttl_minutes,
            max_turns=config.history_turns * 2,
        )
        self.backends = backends if backends is not None else build_backends(config)
        self.transcriber = transcriber or Transcriber(config.stt)
        self.synthesizer = synthesizer or Synthesizer(config.tts)
        self.router = Router(config, self.sessions, self.backends)

    def handle_text(
        self,
        text: str,
        channel_key: str,
        *,
        spoken: bool = True,
        target_override: str | None = None,
    ) -> Exchange:
        reply = self.router.handle(text, channel_key, spoken=spoken, target_override=target_override)
        log.info("[%s] %r -> %s (%s)", channel_key, text[:80], reply.target, reply.kind)
        return Exchange(transcript=text, reply=reply)

    def handle_audio(
        self,
        audio: bytes,
        channel_key: str,
        *,
        filename: str = "audio.ogg",
        spoken: bool = True,
        target_override: str | None = None,
    ) -> Exchange:
        try:
            transcript = self.transcriber.transcribe(audio, filename=filename)
        except SpeechError as exc:
            message = f"I couldn't transcribe that: {exc}"
            return Exchange(
                transcript="",
                reply=Reply(text=message, speech=message, target=self.sessions.get(channel_key).target, kind="error"),
            )
        return self.handle_text(
            transcript, channel_key, spoken=spoken, target_override=target_override
        )

    def speech_audio(self, reply: Reply) -> tuple[bytes, str, str] | None:
        """Render the spoken form of a reply. Returns ``(audio, filename, mime)``."""
        ready, reason = self.synthesizer.available()
        if not ready:
            log.info("not synthesizing speech: %s", reason)
            return None
        text = reply.speech or reply.text
        if not text.strip():
            return None
        try:
            audio = self.synthesizer.synthesize(text)
        except SpeechError as exc:
            log.warning("speech synthesis failed: %s", exc)
            return None
        return audio, f"reply{self.synthesizer.extension}", self.synthesizer.mime

    def diagnostics(self) -> dict[str, object]:
        """Everything ``voicebridge doctor`` and ``GET /v1/status`` report."""
        backends: dict[str, object] = {}
        for name in TARGETS:
            backend = self.backends.get(name)
            if backend is None:
                backends[name] = {"ready": False, "reason": "not built"}
                continue
            ready, reason = backend.available()
            backends[name] = {"ready": ready, "reason": reason or "ok"}
        stt_ready, stt_reason = self.transcriber.available()
        tts_ready, tts_reason = self.synthesizer.available()
        return {
            "default_target": self.config.default_target,
            "backends": backends,
            "speech_to_text": {
                "provider": self.config.stt.provider,
                "model": self.config.stt.model,
                "ready": stt_ready,
                "reason": stt_reason or "ok",
            },
            "text_to_speech": {
                "provider": self.config.tts.provider,
                "voice": self.config.tts.voice,
                "ready": tts_ready,
                "reason": tts_reason or "ok",
            },
            "ingress": {
                "http_ingest": bool(self.config.ingest_token),
                "whatsapp": self.config.whatsapp.enabled
                and bool(self.config.whatsapp.access_token)
                and bool(self.config.whatsapp.allowed_senders),
                "telegram": self.config.telegram.enabled
                and bool(self.config.telegram.bot_token)
                and bool(self.config.telegram.allowed_chat_ids),
            },
            "state_dir": str(self.config.state_path),
        }
