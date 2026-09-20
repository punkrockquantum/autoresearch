"""Configuration, loaded from the environment with an optional JSON overlay.

Everything is env-first so the gateway can run from a systemd unit, a Docker
container or a laptop shell with nothing but exported secrets. A JSON file at
``$VOICEBRIDGE_CONFIG`` (default ``~/.config/voicebridge/config.json``) can
override any field; the file is handy for non-secret preferences like
``default_target`` while keys stay in the environment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

TARGETS = ("claude", "chatgpt", "perplexity", "cursor")

DEFAULT_CONFIG_PATH = Path("~/.config/voicebridge/config.json")
DEFAULT_STATE_DIR = Path("~/.local/state/voicebridge")


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [part.strip() for part in raw.split(",") if part.strip()]


@dataclass
class ClaudeConfig:
    #: ``api`` calls the Messages API; ``cli`` shells out to the Claude Code CLI
    #: so a voice prompt can act on a real checkout.
    mode: str = "api"
    api_key: str = ""
    model: str = "claude-opus-5"
    max_tokens: int = 1024
    base_url: str = "https://api.anthropic.com"
    cli_path: str = "claude"
    cli_workspace: str = ""
    cli_timeout: int = 900

    @classmethod
    def from_env(cls) -> "ClaudeConfig":
        return cls(
            mode=_env("VOICEBRIDGE_CLAUDE_MODE", default="api").lower(),
            api_key=_env("ANTHROPIC_API_KEY"),
            model=_env("VOICEBRIDGE_CLAUDE_MODEL", "ANTHROPIC_MODEL", default="claude-opus-5"),
            max_tokens=_env_int("VOICEBRIDGE_CLAUDE_MAX_TOKENS", 1024),
            base_url=_env("ANTHROPIC_BASE_URL", default="https://api.anthropic.com").rstrip("/"),
            cli_path=_env("VOICEBRIDGE_CLAUDE_CLI", default="claude"),
            cli_workspace=_env("VOICEBRIDGE_CLAUDE_WORKSPACE"),
            cli_timeout=_env_int("VOICEBRIDGE_CLAUDE_CLI_TIMEOUT", 900),
        )


@dataclass
class ChatGPTConfig:
    api_key: str = ""
    model: str = "gpt-5"
    max_output_tokens: int = 1024
    base_url: str = "https://api.openai.com"

    @classmethod
    def from_env(cls) -> "ChatGPTConfig":
        return cls(
            api_key=_env("OPENAI_API_KEY"),
            model=_env("VOICEBRIDGE_OPENAI_MODEL", "OPENAI_MODEL", default="gpt-5"),
            max_output_tokens=_env_int("VOICEBRIDGE_OPENAI_MAX_TOKENS", 1024),
            base_url=_env("OPENAI_BASE_URL", default="https://api.openai.com").rstrip("/"),
        )


@dataclass
class PerplexityConfig:
    api_key: str = ""
    #: Agent API slugs: perplexity/sonar, perplexity/sonar-reasoning, ...
    model: str = "perplexity/sonar"
    max_output_tokens: int = 1024
    base_url: str = "https://api.perplexity.ai"
    web_search: bool = True

    @classmethod
    def from_env(cls) -> "PerplexityConfig":
        return cls(
            api_key=_env("PERPLEXITY_API_KEY"),
            model=_env("VOICEBRIDGE_PERPLEXITY_MODEL", default="perplexity/sonar"),
            max_output_tokens=_env_int("VOICEBRIDGE_PERPLEXITY_MAX_TOKENS", 1024),
            base_url=_env("PERPLEXITY_BASE_URL", default="https://api.perplexity.ai").rstrip("/"),
            web_search=_env_bool("VOICEBRIDGE_PERPLEXITY_WEB_SEARCH", True),
        )


@dataclass
class CursorConfig:
    #: ``api`` launches a cloud agent, ``cli`` runs cursor-agent locally,
    #: ``queue`` just appends the prompt to a file you open in Cursor later.
    mode: str = "api"
    api_key: str = ""
    base_url: str = "https://api.cursor.com"
    repository: str = ""
    ref: str = "main"
    cli_path: str = "cursor-agent"
    cli_workspace: str = ""
    cli_timeout: int = 900
    queue_file: str = ""

    @classmethod
    def from_env(cls) -> "CursorConfig":
        return cls(
            mode=_env("VOICEBRIDGE_CURSOR_MODE", default="api").lower(),
            api_key=_env("CURSOR_API_KEY"),
            base_url=_env("CURSOR_BASE_URL", default="https://api.cursor.com").rstrip("/"),
            repository=_env("VOICEBRIDGE_CURSOR_REPO"),
            ref=_env("VOICEBRIDGE_CURSOR_REF", default="main"),
            cli_path=_env("VOICEBRIDGE_CURSOR_CLI", default="cursor-agent"),
            cli_workspace=_env("VOICEBRIDGE_CURSOR_WORKSPACE"),
            cli_timeout=_env_int("VOICEBRIDGE_CURSOR_CLI_TIMEOUT", 900),
            queue_file=_env("VOICEBRIDGE_CURSOR_QUEUE_FILE"),
        )


@dataclass
class SpeechToTextConfig:
    #: openai | groq | local | none
    provider: str = "openai"
    model: str = "whisper-1"
    language: str = ""
    api_key: str = ""
    base_url: str = ""
    #: faster-whisper size for provider=local
    local_model: str = "base.en"
    local_compute_type: str = "int8"

    @classmethod
    def from_env(cls) -> "SpeechToTextConfig":
        provider = _env("VOICEBRIDGE_STT_PROVIDER", default="openai").lower()
        if provider == "groq":
            default_key, default_url, default_model = (
                _env("GROQ_API_KEY"),
                "https://api.groq.com/openai",
                "whisper-large-v3-turbo",
            )
        else:
            default_key, default_url, default_model = (
                _env("OPENAI_API_KEY"),
                _env("OPENAI_BASE_URL", default="https://api.openai.com"),
                "whisper-1",
            )
        return cls(
            provider=provider,
            model=_env("VOICEBRIDGE_STT_MODEL", default=default_model),
            language=_env("VOICEBRIDGE_STT_LANGUAGE"),
            api_key=_env("VOICEBRIDGE_STT_API_KEY", default=default_key),
            base_url=_env("VOICEBRIDGE_STT_BASE_URL", default=default_url).rstrip("/"),
            local_model=_env("VOICEBRIDGE_STT_LOCAL_MODEL", default="base.en"),
            local_compute_type=_env("VOICEBRIDGE_STT_LOCAL_COMPUTE", default="int8"),
        )


@dataclass
class TextToSpeechConfig:
    #: openai | elevenlabs | piper | none
    provider: str = "openai"
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    api_key: str = ""
    base_url: str = ""
    #: ogg/opus keeps WhatsApp and Telegram voice notes happy
    audio_format: str = "opus"
    piper_path: str = "piper"
    piper_model: str = ""

    @classmethod
    def from_env(cls) -> "TextToSpeechConfig":
        provider = _env("VOICEBRIDGE_TTS_PROVIDER", default="openai").lower()
        if provider == "elevenlabs":
            default_key, default_url, default_model, default_voice = (
                _env("ELEVENLABS_API_KEY"),
                "https://api.elevenlabs.io",
                "eleven_turbo_v2_5",
                "21m00Tcm4TlvDq8ikWAM",
            )
        else:
            default_key, default_url, default_model, default_voice = (
                _env("OPENAI_API_KEY"),
                _env("OPENAI_BASE_URL", default="https://api.openai.com"),
                "gpt-4o-mini-tts",
                "alloy",
            )
        return cls(
            provider=provider,
            model=_env("VOICEBRIDGE_TTS_MODEL", default=default_model),
            voice=_env("VOICEBRIDGE_TTS_VOICE", default=default_voice),
            api_key=_env("VOICEBRIDGE_TTS_API_KEY", default=default_key),
            base_url=_env("VOICEBRIDGE_TTS_BASE_URL", default=default_url).rstrip("/"),
            audio_format=_env("VOICEBRIDGE_TTS_FORMAT", default="opus"),
            piper_path=_env("VOICEBRIDGE_PIPER_PATH", default="piper"),
            piper_model=_env("VOICEBRIDGE_PIPER_MODEL"),
        )


@dataclass
class WhatsAppConfig:
    """Meta WhatsApp Cloud API — the hands-free route through the glasses."""

    enabled: bool = False
    verify_token: str = ""
    app_secret: str = ""
    access_token: str = ""
    phone_number_id: str = ""
    graph_version: str = "v22.0"
    #: E.164 numbers allowed to drive the bridge; empty means "nobody", since an
    #: open webhook spends your API credits for strangers.
    allowed_senders: list[str] = field(default_factory=list)
    reply_with_voice: bool = False

    @classmethod
    def from_env(cls) -> "WhatsAppConfig":
        token = _env("WHATSAPP_ACCESS_TOKEN")
        phone_id = _env("WHATSAPP_PHONE_NUMBER_ID")
        return cls(
            enabled=_env_bool("VOICEBRIDGE_WHATSAPP_ENABLED", bool(token and phone_id)),
            verify_token=_env("WHATSAPP_VERIFY_TOKEN"),
            app_secret=_env("WHATSAPP_APP_SECRET"),
            access_token=token,
            phone_number_id=phone_id,
            graph_version=_env("WHATSAPP_GRAPH_VERSION", default="v22.0"),
            allowed_senders=_env_list("VOICEBRIDGE_ALLOWED_WHATSAPP_SENDERS"),
            reply_with_voice=_env_bool("VOICEBRIDGE_WHATSAPP_VOICE_REPLY", False),
        )


@dataclass
class TelegramConfig:
    enabled: bool = False
    bot_token: str = ""
    #: Value Telegram echoes in X-Telegram-Bot-Api-Secret-Token.
    secret_token: str = ""
    allowed_chat_ids: list[str] = field(default_factory=list)
    reply_with_voice: bool = True

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        token = _env("TELEGRAM_BOT_TOKEN")
        return cls(
            enabled=_env_bool("VOICEBRIDGE_TELEGRAM_ENABLED", bool(token)),
            bot_token=token,
            secret_token=_env("TELEGRAM_SECRET_TOKEN"),
            allowed_chat_ids=_env_list("VOICEBRIDGE_ALLOWED_TELEGRAM_CHATS"),
            reply_with_voice=_env_bool("VOICEBRIDGE_TELEGRAM_VOICE_REPLY", True),
        )


@dataclass
class Config:
    default_target: str = "claude"
    #: Spoken answers get a hard word budget; the full text still goes to chat.
    spoken_word_limit: int = 90
    long_word_limit: int = 320
    history_turns: int = 12
    session_ttl_minutes: int = 180
    state_dir: str = str(DEFAULT_STATE_DIR)
    host: str = "0.0.0.0"
    port: int = 8808
    #: Bearer token for /v1/* (iOS Shortcuts, Tasker, curl).
    ingest_token: str = ""
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    chatgpt: ChatGPTConfig = field(default_factory=ChatGPTConfig)
    perplexity: PerplexityConfig = field(default_factory=PerplexityConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    stt: SpeechToTextConfig = field(default_factory=SpeechToTextConfig)
    tts: TextToSpeechConfig = field(default_factory=TextToSpeechConfig)
    whatsapp: WhatsAppConfig = field(default_factory=WhatsAppConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    @classmethod
    def from_env(cls) -> "Config":
        target = _env("VOICEBRIDGE_DEFAULT_TARGET", default="claude").lower()
        if target not in TARGETS:
            target = "claude"
        return cls(
            default_target=target,
            spoken_word_limit=_env_int("VOICEBRIDGE_SPOKEN_WORD_LIMIT", 90),
            long_word_limit=_env_int("VOICEBRIDGE_LONG_WORD_LIMIT", 320),
            history_turns=_env_int("VOICEBRIDGE_HISTORY_TURNS", 12),
            session_ttl_minutes=_env_int("VOICEBRIDGE_SESSION_TTL_MINUTES", 180),
            state_dir=_env("VOICEBRIDGE_STATE_DIR", default=str(DEFAULT_STATE_DIR)),
            host=_env("VOICEBRIDGE_HOST", default="0.0.0.0"),
            port=_env_int("VOICEBRIDGE_PORT", 8808),
            ingest_token=_env("VOICEBRIDGE_INGEST_TOKEN"),
            claude=ClaudeConfig.from_env(),
            chatgpt=ChatGPTConfig.from_env(),
            perplexity=PerplexityConfig.from_env(),
            cursor=CursorConfig.from_env(),
            stt=SpeechToTextConfig.from_env(),
            tts=TextToSpeechConfig.from_env(),
            whatsapp=WhatsAppConfig.from_env(),
            telegram=TelegramConfig.from_env(),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "Config":
        config = cls.from_env()
        candidate = Path(path) if path else Path(os.environ.get("VOICEBRIDGE_CONFIG", DEFAULT_CONFIG_PATH))
        candidate = candidate.expanduser()
        if candidate.is_file():
            overlay = json.loads(candidate.read_text("utf-8"))
            _apply_overlay(config, overlay)
        return config

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir).expanduser()

    def redacted(self) -> dict[str, Any]:
        """Config as a dict with every secret masked, for /v1/status and doctor."""
        return _redact(self)


_SECRET_HINTS = ("api_key", "token", "secret")


def _apply_overlay(target: Any, overlay: dict[str, Any]) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in overlay.items():
        spec = known.get(key)
        if spec is None:
            raise ValueError(f"unknown config key: {key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_overlay(current, value)
        else:
            setattr(target, key, value)


def _redact(value: Any, key: str = "") -> Any:
    if is_dataclass(value):
        return {f.name: _redact(getattr(value, f.name), f.name) for f in fields(value)}
    if isinstance(value, str) and any(hint in key for hint in _SECRET_HINTS):
        return f"set ({len(value)} chars)" if value else "unset"
    return value
