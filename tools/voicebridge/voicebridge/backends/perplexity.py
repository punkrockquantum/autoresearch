"""Perplexity via the Agent API (``POST /v1/agent``).

The older Sonar ``/chat/completions`` endpoint is sunset on 2026-09-27, so this
targets the Agent API shape: ``input`` instead of ``messages``, ``instructions``
for the system prompt, web search requested explicitly as a tool, and citations
carried in a ``search_results`` item inside ``output``.
"""

from __future__ import annotations

from typing import Any

from ..config import PerplexityConfig
from ..http import HttpError, describe_http_error, request_json
from ..textutil import format_citations
from .base import Backend, BackendError, Request, Result


class PerplexityBackend(Backend):
    name = "perplexity"

    def __init__(self, config: PerplexityConfig) -> None:
        self.config = config

    def available(self) -> tuple[bool, str]:
        if not self.config.api_key:
            return False, "PERPLEXITY_API_KEY isn't set"
        return True, ""

    def run(self, request: Request) -> Result:
        body: dict[str, Any] = {
            "model": self.config.model,
            "input": request.messages(),
            "max_output_tokens": self.config.max_output_tokens,
        }
        if request.system:
            body["instructions"] = request.system
        if self.config.web_search:
            body["tools"] = [{"type": "web_search"}]

        try:
            payload = request_json(
                "POST",
                f"{self.config.base_url}/v1/agent",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json_body=body,
            )
        except HttpError as exc:
            raise BackendError(describe_http_error(exc)) from None

        status = payload.get("status")
        text = _output_text(payload)
        if status in {"failed", "cancelled"} and not text:
            reason = (payload.get("error") or {}).get("message") if isinstance(payload.get("error"), dict) else ""
            raise BackendError(f"Perplexity {status}. {reason}".strip())
        if not text:
            raise BackendError(f"Perplexity returned no text (status {status or 'unknown'})")

        citations = _citations(payload)
        return Result(
            text=text,
            footer=format_citations(citations),
            meta={"model": payload.get("model", self.config.model), "status": status, "citations": citations},
        )


def _output_text(payload: dict[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    if isinstance(direct, list):
        joined = " ".join(str(part) for part in direct if part).strip()
        if joined:
            return joined

    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {None, "message", "output_text"}:
            continue
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        for block in content or []:
            if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                chunks.append(str(block.get("text", "")))
    return "\n".join(chunk for chunk in chunks if chunk).strip()


def _citations(payload: dict[str, Any]) -> list[str]:
    urls: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value.startswith("http") and value not in urls:
            urls.append(value)
        elif isinstance(value, dict):
            add(value.get("url"))

    for item in payload.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "search_results":
            for result in item.get("results") or item.get("search_results") or []:
                add(result)
    # Tolerate the legacy top-level fields if the API still sends them.
    for key in ("citations", "search_results"):
        for value in payload.get(key) or []:
            add(value)
    return urls
