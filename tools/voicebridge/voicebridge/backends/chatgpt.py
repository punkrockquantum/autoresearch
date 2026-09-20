"""ChatGPT via the OpenAI chat completions API."""

from __future__ import annotations

from ..config import ChatGPTConfig
from ..http import HttpError, describe_http_error, request_json
from .base import Backend, BackendError, Request, Result


class ChatGPTBackend(Backend):
    name = "chatgpt"

    def __init__(self, config: ChatGPTConfig) -> None:
        self.config = config

    def available(self) -> tuple[bool, str]:
        if not self.config.api_key:
            return False, "OPENAI_API_KEY isn't set"
        return True, ""

    def run(self, request: Request) -> Result:
        messages: list[dict[str, str]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(request.messages())

        body: dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": self.config.max_output_tokens,
        }
        payload = self._post(body)
        choices = payload.get("choices") or []
        if not choices:
            raise BackendError("ChatGPT returned no choices")
        text = (choices[0].get("message", {}).get("content") or "").strip()
        if not text:
            # Reasoning models can spend the whole budget before emitting text.
            raise BackendError("ChatGPT returned an empty answer; try raising the token limit")
        return Result(
            text=text,
            meta={"model": payload.get("model", self.config.model), "usage": payload.get("usage", {})},
        )

    def _post(self, body: dict[str, object]) -> dict:
        url = f"{self.config.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        try:
            return request_json("POST", url, headers=headers, json_body=body)
        except HttpError as exc:
            # Older models and some gateways only accept the legacy token field.
            if exc.status == 400 and "max_completion_tokens" in exc.body and "max_completion_tokens" in body:
                legacy = dict(body)
                legacy["max_tokens"] = legacy.pop("max_completion_tokens")
                try:
                    return request_json("POST", url, headers=headers, json_body=legacy)
                except HttpError as retry_exc:
                    raise BackendError(describe_http_error(retry_exc)) from None
            raise BackendError(describe_http_error(exc)) from None
