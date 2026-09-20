"""Backend contract shared by the four assistants."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..sessions import Turn


class BackendError(RuntimeError):
    """A backend could not answer; the message is read aloud, so keep it short."""


@dataclass
class Request:
    prompt: str
    history: list[Turn] = field(default_factory=list)
    system: str = ""
    style: str = "normal"
    spoken: bool = True
    channel_key: str = ""

    def messages(self) -> list[dict[str, str]]:
        """History plus the new prompt, in the OpenAI/Anthropic message shape."""
        out = [{"role": turn.role, "content": turn.content} for turn in self.history]
        out.append({"role": "user", "content": self.prompt})
        return out


@dataclass
class Result:
    text: str
    #: Set when the backend wants a different wording read aloud than shown.
    speech: str | None = None
    #: Appended to the text reply only (citations, agent links).
    footer: str = ""
    meta: dict[str, object] = field(default_factory=dict)


class Backend:
    name = "backend"

    def available(self) -> tuple[bool, str]:
        """Return ``(ready, reason)``; the reason is spoken when not ready."""
        raise NotImplementedError

    def run(self, request: Request) -> Result:
        raise NotImplementedError
