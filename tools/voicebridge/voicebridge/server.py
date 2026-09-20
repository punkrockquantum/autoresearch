"""FastAPI app exposing every remote ingress path.

Routes
------
``GET  /healthz``            liveness, no auth
``GET  /v1/status``          readiness of each backend and channel (bearer auth)
``POST /v1/ingest``          JSON text or base64 audio, answers synchronously
``POST /v1/ingest/audio``    multipart audio upload, answers synchronously
``GET  /webhook/whatsapp``   Meta webhook verification handshake
``POST /webhook/whatsapp``   inbound WhatsApp messages (HMAC-verified)
``POST /webhook/telegram``   inbound Telegram updates (secret-token header)

Messaging webhooks answer 200 immediately and do the model call in a background
task: Meta and Telegram both retry a slow webhook, and a retried prompt would be
answered twice.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import logging

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .bridge import Bridge, Exchange
from .config import TARGETS, Config
from .ingress import telegram as tg
from .ingress import whatsapp as wa
from .ingress.base import Inbound
from .router import Reply

log = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def create_app(config: Config | None = None, bridge: Bridge | None = None) -> FastAPI:
    config = config or (bridge.config if bridge else Config.load())
    bridge = bridge or Bridge(config)

    app = FastAPI(title="voicebridge", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.config = config
    app.state.bridge = bridge

    def require_ingest_token(authorization: str | None = Header(default=None)) -> None:
        expected = config.ingest_token
        if not expected:
            raise HTTPException(503, "VOICEBRIDGE_INGEST_TOKEN is not set; /v1 endpoints are disabled")
        supplied = ""
        if authorization and authorization.lower().startswith("bearer "):
            supplied = authorization[7:].strip()
        if not supplied or not hmac.compare_digest(supplied, expected):
            raise HTTPException(401, "bad or missing bearer token")

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"ok": True, "service": "voicebridge", "version": app.version}

    @app.get("/v1/status", dependencies=[Depends(require_ingest_token)])
    def status() -> dict[str, object]:
        return {"diagnostics": bridge.diagnostics(), "sessions": bridge.sessions.summary()}

    @app.post("/v1/ingest", dependencies=[Depends(require_ingest_token)])
    async def ingest(request: Request) -> JSONResponse:
        """Text or base64 audio in, the answer (and optionally its audio) out."""
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001 - Shortcuts sometimes posts empty bodies
            raise HTTPException(400, "expected a JSON body") from None
        if not isinstance(payload, dict):
            raise HTTPException(400, "expected a JSON object")

        channel = str(payload.get("channel") or "http")
        speaker = str(payload.get("speaker") or "default")
        target = _clean_target(payload.get("target"))
        want_audio = bool(payload.get("speak"))
        channel_key = f"{channel}:{speaker}"

        text = str(payload.get("text") or "").strip()
        audio_b64 = payload.get("audio_base64")
        if audio_b64:
            try:
                audio = base64.b64decode(str(audio_b64), validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(400, "audio_base64 is not valid base64") from None
            if len(audio) > MAX_AUDIO_BYTES:
                raise HTTPException(413, "audio too large")
            exchange = bridge.handle_audio(
                audio,
                channel_key,
                filename=str(payload.get("filename") or "audio.m4a"),
                target_override=target,
            )
        elif text:
            exchange = bridge.handle_text(text, channel_key, target_override=target)
        else:
            raise HTTPException(400, "send either text or audio_base64")

        return JSONResponse(_exchange_payload(bridge, exchange, want_audio))

    @app.post("/v1/ingest/audio", dependencies=[Depends(require_ingest_token)])
    async def ingest_audio(
        file: UploadFile = File(...),
        channel: str = Form("http"),
        speaker: str = Form("default"),
        target: str | None = Form(None),
        speak: bool = Form(False),
    ) -> JSONResponse:
        audio = await file.read()
        if not audio:
            raise HTTPException(400, "empty upload")
        if len(audio) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio too large")
        exchange = bridge.handle_audio(
            audio,
            f"{channel}:{speaker}",
            filename=file.filename or "audio.m4a",
            target_override=_clean_target(target),
        )
        return JSONResponse(_exchange_payload(bridge, exchange, speak))

    @app.get("/webhook/whatsapp")
    def whatsapp_verify(request: Request) -> PlainTextResponse:
        params = request.query_params
        expected = config.whatsapp.verify_token
        if not expected:
            raise HTTPException(503, "WHATSAPP_VERIFY_TOKEN is not set")
        if params.get("hub.mode") == "subscribe" and hmac.compare_digest(
            params.get("hub.verify_token", ""), expected
        ):
            return PlainTextResponse(params.get("hub.challenge", ""))
        raise HTTPException(403, "verification failed")

    @app.post("/webhook/whatsapp")
    async def whatsapp_inbound(
        request: Request,
        background: BackgroundTasks,
        x_hub_signature_256: str | None = Header(default=None),
    ) -> dict[str, object]:
        if not config.whatsapp.enabled:
            raise HTTPException(503, "the WhatsApp channel is disabled")
        raw = await request.body()
        if not wa.verify_signature(config.whatsapp.app_secret, raw, x_hub_signature_256):
            raise HTTPException(401, "bad signature")
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "malformed JSON") from None

        accepted = 0
        for message in wa.parse_webhook(payload):
            if not wa.sender_allowed(config.whatsapp, message.sender):
                log.warning("ignoring whatsapp message from unlisted sender %s", message.sender)
                continue
            background.add_task(_process_whatsapp, bridge, config, message)
            accepted += 1
        return {"accepted": accepted}

    @app.post("/webhook/telegram")
    async def telegram_inbound(
        request: Request,
        background: BackgroundTasks,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ) -> dict[str, object]:
        if not config.telegram.enabled:
            raise HTTPException(503, "the Telegram channel is disabled")
        expected = config.telegram.secret_token
        if not expected:
            raise HTTPException(503, "TELEGRAM_SECRET_TOKEN is not set")
        if not x_telegram_bot_api_secret_token or not hmac.compare_digest(
            x_telegram_bot_api_secret_token, expected
        ):
            raise HTTPException(401, "bad secret token")
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "malformed JSON") from None

        accepted = 0
        for message in tg.parse_update(payload):
            if not tg.chat_allowed(config.telegram, message.sender):
                log.warning("ignoring telegram message from unlisted chat %s", message.sender)
                continue
            background.add_task(_process_telegram, bridge, config, message)
            accepted += 1
        return {"accepted": accepted}

    return app


def _clean_target(value: object) -> str | None:
    if not value:
        return None
    candidate = str(value).strip().lower()
    return candidate if candidate in TARGETS else None


def _exchange_payload(bridge: Bridge, exchange: Exchange, want_audio: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "transcript": exchange.transcript,
        "target": exchange.reply.target,
        "kind": exchange.reply.kind,
        "reply": exchange.reply.text,
        "speech": exchange.reply.speech,
    }
    if want_audio:
        rendered = bridge.speech_audio(exchange.reply)
        if rendered:
            audio, _, mime = rendered
            payload["audio_base64"] = base64.b64encode(audio).decode("ascii")
            payload["audio_mime"] = mime
    return payload


def _process_whatsapp(bridge: Bridge, config: Config, message: Inbound) -> None:
    try:
        exchange = _resolve(bridge, config, message, fetch=lambda mid: wa.fetch_media(config.whatsapp, mid))
        wa.send_text(config.whatsapp, message.reply_to, _with_transcript(exchange))
        if config.whatsapp.reply_with_voice:
            rendered = bridge.speech_audio(exchange.reply)
            if rendered:
                audio, filename, mime = rendered
                wa.send_voice(config.whatsapp, message.reply_to, audio, filename=filename, mime=mime)
    except Exception:  # noqa: BLE001 - a background task must not die silently
        log.exception("failed to handle whatsapp message %s", message.message_id)


def _process_telegram(bridge: Bridge, config: Config, message: Inbound) -> None:
    try:
        exchange = _resolve(bridge, config, message, fetch=lambda fid: tg.fetch_media(config.telegram, fid))
        tg.send_text(config.telegram, message.reply_to, _with_transcript(exchange))
        if config.telegram.reply_with_voice:
            rendered = bridge.speech_audio(exchange.reply)
            if rendered:
                audio, filename, mime = rendered
                tg.send_voice(config.telegram, message.reply_to, audio, filename=filename, mime=mime)
    except Exception:  # noqa: BLE001
        log.exception("failed to handle telegram message %s", message.message_id)


def _resolve(bridge: Bridge, config: Config, message: Inbound, *, fetch) -> Exchange:
    """Download audio if needed, then answer."""
    if message.text:
        return bridge.handle_text(message.text, message.channel_key)
    audio = message.audio
    filename = message.audio_filename
    if audio is None and message.media_id:
        audio, filename = fetch(message.media_id)
    if audio is None:
        reply = Reply(text="Nothing to answer there.", speech="Nothing to answer there.", target="", kind="empty")
        return Exchange(transcript="", reply=reply)
    return bridge.handle_audio(audio, message.channel_key, filename=filename)


def _with_transcript(exchange: Exchange) -> str:
    """Echo what was heard — mis-transcriptions are otherwise baffling."""
    if exchange.transcript and exchange.reply.kind == "prompt":
        return f"🎙 {exchange.transcript}\n\n{exchange.reply.text}"
    return exchange.reply.text
