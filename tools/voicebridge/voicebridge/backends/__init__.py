"""Backend registry."""

from __future__ import annotations

from ..config import Config
from .base import Backend, BackendError, Request, Result
from .chatgpt import ChatGPTBackend
from .claude import ClaudeBackend
from .cursor import CursorBackend
from .perplexity import PerplexityBackend

__all__ = [
    "Backend",
    "BackendError",
    "Request",
    "Result",
    "ChatGPTBackend",
    "ClaudeBackend",
    "CursorBackend",
    "PerplexityBackend",
    "build_backends",
]


def build_backends(config: Config) -> dict[str, Backend]:
    return {
        "claude": ClaudeBackend(config.claude),
        "chatgpt": ChatGPTBackend(config.chatgpt),
        "perplexity": PerplexityBackend(config.perplexity),
        "cursor": CursorBackend(config.cursor),
    }
