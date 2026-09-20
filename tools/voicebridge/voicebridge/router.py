"""Understanding a spoken line: which assistant, and what to ask it.

Speech recognition mangles product names in predictable ways ("cloud" for
Claude, "chat gbt", "curser"), and dictation through the glasses adds filler
and stray punctuation. Parsing is therefore fuzzy on the target word and strict
about nothing else.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field

from .backends.base import Backend, BackendError, Request, Result
from .config import TARGETS, Config
from .sessions import SessionStore
from .textutil import for_speech

log = logging.getLogger(__name__)

#: Spellings a transcriber realistically produces for each target.
TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "claude": ("claude", "clawed", "cloud", "claud", "clode", "chlode", "klaus", "anthropic"),
    "chatgpt": ("chatgpt", "chat gpt", "chat gbt", "gpt", "gbt", "openai", "open ai", "chat"),
    "perplexity": ("perplexity", "perplexity ai", "complexity", "perplexed", "search", "look up", "lookup"),
    "cursor": ("cursor", "curser", "courser", "cursor agent", "agent", "coder"),
}

_FILLERS = (
    "hey", "hi", "hello", "ok", "okay", "yo", "um", "uh", "er", "so", "please",
    "could you", "can you", "would you", "i want to", "i'd like to",
)
_ASK_VERBS = ("ask", "tell", "prompt", "query", "send to", "talk to", "message", "ping")

_RESET_PHRASES = (
    "new chat", "new conversation", "new thread", "reset", "start over", "start fresh",
    "clear context", "clear history", "forget that", "forget all that", "wipe context",
)
_REPEAT_PHRASES = ("repeat that", "say that again", "read that back", "repeat", "again", "what was that")
_STATUS_PHRASES = ("status", "bridge status", "who am i talking to", "which assistant", "what's the default")
_HELP_PHRASES = ("help", "what can you do", "how do i use this", "commands")
_SWITCH_PATTERNS = (
    re.compile(r"^(?:switch|change|set)(?: the)?(?: default)?(?: to| target to)?\s+(?P<target>.+)$"),
    re.compile(r"^(?:use|default to|talk to)\s+(?P<target>.+)\s*(?:from now on|by default)?$"),
)
_LONG_PHRASES = ("in detail", "in full", "long answer", "full answer", "be thorough", "go deep", "verbose")
_SHORT_PHRASES = ("briefly", "in brief", "short answer", "one line", "one sentence", "quickly", "tldr", "keep it short")

_PUNCT_EDGES = re.compile(r"^[\s,.;:!?\-–—]+|[\s,;:\-–—]+$")
_NORMALIZE = re.compile(r"[^a-z0-9'\s]")


@dataclass
class Parsed:
    """The structured reading of one spoken line."""

    kind: str  # "prompt" | "reset" | "switch" | "status" | "repeat" | "help" | "empty"
    prompt: str = ""
    target: str | None = None
    style: str = "normal"  # "short" | "normal" | "long"
    explicit_target: bool = False
    raw: str = ""


@dataclass
class Reply:
    """What to send back: full text for the screen, short text for the ear."""

    text: str
    speech: str
    target: str
    kind: str = "prompt"
    meta: dict[str, object] = field(default_factory=dict)


def _normalize(text: str) -> str:
    return _NORMALIZE.sub(" ", text.lower()).strip()


def _strip_fillers(text: str) -> str:
    changed = True
    while changed:
        changed = False
        lowered = text.lower()
        for filler in _FILLERS:
            if lowered.startswith(filler + " "):
                text = text[len(filler) + 1 :].lstrip(" ,.")
                changed = True
                break
    return text


def match_target(phrase: str, *, cutoff: float = 0.8) -> str | None:
    """Map a possibly-misheard phrase to a target, or ``None``.

    Multi-word phrases must match an alias exactly: "perplexity ai" and
    "perplexity what" are similar enough to fool a ratio test, and swallowing the
    first word of the question is worse than missing the alias. Single words are
    matched loosely, which is where the mishearings actually happen.
    """
    phrase = _normalize(phrase)
    if not phrase:
        return None
    flat = {alias: target for target, aliases in TARGET_ALIASES.items() for alias in aliases}
    if phrase in flat:
        return flat[phrase]
    if len(phrase.split()) > 1:
        return None
    single = [alias for alias in flat if " " not in alias]
    close = difflib.get_close_matches(phrase, single, n=1, cutoff=cutoff)
    return flat[close[0]] if close else None


def _peel_target(text: str) -> tuple[str | None, str]:
    """Strip a leading address ("claude, …", "ask chat gpt to …") off the text."""
    working = text
    lowered = working.lower()
    for verb in _ASK_VERBS:
        if lowered.startswith(verb + " "):
            working = working[len(verb) + 1 :]
            break

    words = working.split()
    # Longest match first so "chat gpt" wins over "chat".
    for span in (3, 2, 1):
        if len(words) < span:
            continue
        candidate = " ".join(words[:span])
        target = match_target(candidate)
        if target is None:
            continue
        rest = " ".join(words[span:])
        rest = _PUNCT_EDGES.sub("", rest)
        # "claude to fix the test" / "perplexity about the weather"
        for lead in ("to ", "about ", "that ", "if ", "whether "):
            if rest.lower().startswith(lead) and lead in ("to ", "about "):
                rest = rest[len(lead) :]
                break
        return target, rest.strip()
    return None, text


def _extract_style(text: str) -> tuple[str, str]:
    lowered = text.lower()
    for phrase in _LONG_PHRASES:
        if lowered.startswith(phrase + " ") or lowered.endswith(" " + phrase) or lowered == phrase:
            return "long", _strip_phrase(text, phrase)
    for phrase in _SHORT_PHRASES:
        if lowered.startswith(phrase + " ") or lowered.endswith(" " + phrase) or lowered == phrase:
            return "short", _strip_phrase(text, phrase)
    return "normal", text


def _strip_phrase(text: str, phrase: str) -> str:
    lowered = text.lower()
    if lowered.startswith(phrase):
        text = text[len(phrase) :]
    elif lowered.endswith(phrase):
        text = text[: -len(phrase)]
    return _PUNCT_EDGES.sub("", text).strip()


def parse(text: str) -> Parsed:
    """Parse one transcript line into a command or a prompt."""
    raw = (text or "").strip()
    if not raw:
        return Parsed(kind="empty", raw=raw)

    stripped = _PUNCT_EDGES.sub("", _strip_fillers(raw)).strip()
    normalized = _normalize(stripped)
    if not normalized:
        return Parsed(kind="empty", raw=raw)

    if normalized in _RESET_PHRASES:
        return Parsed(kind="reset", raw=raw)
    if normalized in _REPEAT_PHRASES:
        return Parsed(kind="repeat", raw=raw)
    if normalized in _STATUS_PHRASES:
        return Parsed(kind="status", raw=raw)
    if normalized in _HELP_PHRASES:
        return Parsed(kind="help", raw=raw)

    for pattern in _SWITCH_PATTERNS:
        found = pattern.match(normalized)
        if not found:
            continue
        target = match_target(found.group("target"))
        if target:
            return Parsed(kind="switch", target=target, explicit_target=True, raw=raw)

    # "claude, new chat" — an address plus a command, not a question about one.
    target, remainder = _peel_target(stripped)
    remainder_norm = _normalize(remainder)
    if target:
        for phrases, kind in (
            (_RESET_PHRASES, "reset"),
            (_REPEAT_PHRASES, "repeat"),
            (_STATUS_PHRASES, "status"),
            (_HELP_PHRASES, "help"),
        ):
            if remainder_norm in phrases:
                return Parsed(kind=kind, target=target, explicit_target=True, raw=raw)
    if target and not remainder_norm:
        return Parsed(kind="switch", target=target, explicit_target=True, raw=raw)

    style, prompt = _extract_style(remainder if target else stripped)
    if not prompt.strip():
        return Parsed(kind="empty", target=target, explicit_target=target is not None, raw=raw)
    return Parsed(
        kind="prompt",
        prompt=prompt.strip(),
        target=target,
        style=style,
        explicit_target=target is not None,
        raw=raw,
    )


SYSTEM_SPOKEN = (
    "You are answering someone who is wearing smart glasses and will hear your reply read aloud. "
    "Answer in at most {limit} words of plain spoken English. No markdown, no bullet lists, no code blocks, "
    "no URLs. Lead with the answer, then at most one sentence of why. "
    "If the question genuinely needs code or a long list, say so in one sentence and give the gist."
)
SYSTEM_TEXT = (
    "You are answering a short dictated question. Be direct and concise; "
    "skip preamble and restating the question."
)

HELP_TEXT = (
    "Say the assistant's name first, then your question. "
    "Claude for general questions and code, ChatGPT for a second opinion, Perplexity for anything needing "
    "live web sources, Cursor to start a coding agent on your repo. "
    "Say 'new chat' to clear context, 'switch to Perplexity' to change the default, "
    "'in detail' for a longer answer, or 'repeat that'."
)


class Router:
    """Parses a line, picks a backend, keeps the conversation going."""

    def __init__(self, config: Config, sessions: SessionStore, backends: dict[str, Backend]) -> None:
        self.config = config
        self.sessions = sessions
        self.backends = backends

    def handle(
        self,
        text: str,
        channel_key: str,
        *,
        spoken: bool = True,
        target_override: str | None = None,
    ) -> Reply:
        """Answer one spoken line.

        ``target_override`` is for callers that already know the assistant (an
        iOS Shortcut wired to one button, ``voicebridge ask --target``); a target
        spoken in the line still wins.
        """
        parsed = parse(text)
        if parsed.target is None and target_override in TARGETS:
            parsed.target = target_override
        session = self.sessions.get(channel_key)

        if parsed.kind == "empty":
            message = "I didn't catch a question there."
            return Reply(text=message, speech=message, target=session.target, kind="empty")

        if parsed.kind == "help":
            return Reply(text=HELP_TEXT, speech=HELP_TEXT, target=session.target, kind="help")

        if parsed.kind == "reset":
            if parsed.target:
                self.sessions.set_target(channel_key, parsed.target)
            self.sessions.reset(channel_key)
            target = parsed.target or session.target
            message = f"Fresh start with {_spoken_name(target)}."
            return Reply(text=message, speech=message, target=target, kind="reset")

        if parsed.kind == "switch":
            target = parsed.target or session.target
            self.sessions.set_target(channel_key, target)
            message = f"Talking to {_spoken_name(target)} now."
            return Reply(text=message, speech=message, target=target, kind="switch")

        if parsed.kind == "status":
            message = (
                f"Default is {_spoken_name(session.target)}, "
                f"{len(session.turns)} turns in this conversation."
            )
            return Reply(text=message, speech=message, target=session.target, kind="status")

        if parsed.kind == "repeat":
            last = session.last_reply or "Nothing to repeat yet."
            return Reply(
                text=last,
                speech=for_speech(last, self.config.spoken_word_limit),
                target=session.target,
                kind="repeat",
            )

        target = parsed.target or session.target
        if target not in TARGETS:
            target = self.config.default_target
        backend = self.backends.get(target)
        if backend is None:
            message = f"{_spoken_name(target)} isn't wired up in this build."
            return Reply(text=message, speech=message, target=target, kind="error")

        ready, reason = backend.available()
        if not ready:
            message = f"{_spoken_name(target)} isn't set up: {reason}"
            return Reply(text=message, speech=message, target=target, kind="error")

        if parsed.explicit_target:
            self.sessions.set_target(channel_key, target)

        limit = self.config.long_word_limit if parsed.style == "long" else self.config.spoken_word_limit
        if parsed.style == "short":
            limit = max(25, self.config.spoken_word_limit // 3)
        system = SYSTEM_SPOKEN.format(limit=limit) if spoken else SYSTEM_TEXT

        request = Request(
            prompt=parsed.prompt,
            history=list(session.history(self.config.history_turns)),
            system=system,
            style=parsed.style,
            spoken=spoken,
            channel_key=channel_key,
        )
        try:
            result: Result = backend.run(request)
        except BackendError as exc:
            log.warning("%s backend failed: %s", target, exc)
            message = f"{_spoken_name(target)} failed: {exc}"
            return Reply(text=message, speech=message, target=target, kind="error")
        except Exception as exc:  # noqa: BLE001 - never crash the webhook
            log.exception("%s backend crashed", target)
            message = f"{_spoken_name(target)} errored: {type(exc).__name__}: {exc}"
            return Reply(text=message, speech=message, target=target, kind="error")

        self.sessions.append(channel_key, "user", parsed.prompt)
        self.sessions.append(channel_key, "assistant", result.text)

        speech = result.speech or for_speech(result.text, limit)
        text = result.text if not result.footer else f"{result.text}\n\n{result.footer}"
        return Reply(text=text, speech=speech, target=target, kind="prompt", meta=result.meta)


def _spoken_name(target: str) -> str:
    return {
        "claude": "Claude",
        "chatgpt": "ChatGPT",
        "perplexity": "Perplexity",
        "cursor": "Cursor",
    }.get(target, target)
