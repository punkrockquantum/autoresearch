"""Claude, either through the Messages API or the Claude Code CLI.

CLI mode is what makes "Claude, why is the training loss spiking?" useful while
walking around: the prompt runs inside a real checkout with file access.
"""

from __future__ import annotations

import json
import shutil
import subprocess

from ..config import ClaudeConfig
from ..http import HttpError, describe_http_error, request_json
from .base import Backend, BackendError, Request, Result

ANTHROPIC_VERSION = "2023-06-01"


class ClaudeBackend(Backend):
    name = "claude"

    def __init__(self, config: ClaudeConfig) -> None:
        self.config = config

    def available(self) -> tuple[bool, str]:
        if self.config.mode == "cli":
            if shutil.which(self.config.cli_path) is None:
                return False, f"the {self.config.cli_path} CLI isn't on PATH"
            return True, ""
        if not self.config.api_key:
            return False, "ANTHROPIC_API_KEY isn't set"
        return True, ""

    def run(self, request: Request) -> Result:
        if self.config.mode == "cli":
            return self._run_cli(request)
        return self._run_api(request)

    def _run_api(self, request: Request) -> Result:
        body = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": request.messages(),
        }
        if request.system:
            body["system"] = request.system
        try:
            payload = request_json(
                "POST",
                f"{self.config.base_url}/v1/messages",
                headers={
                    "x-api-key": self.config.api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                json_body=body,
            )
        except HttpError as exc:
            raise BackendError(describe_http_error(exc)) from None

        text = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if block.get("type") == "text"
        ).strip()
        if not text:
            raise BackendError("Claude returned an empty answer")
        usage = payload.get("usage", {})
        return Result(text=text, meta={"model": payload.get("model", self.config.model), "usage": usage})

    def _run_cli(self, request: Request) -> Result:
        prompt = request.prompt
        if request.system:
            prompt = f"{request.system}\n\nQuestion: {request.prompt}"
        command = [self.config.cli_path, "-p", prompt, "--output-format", "json"]
        try:
            completed = subprocess.run(
                command,
                cwd=self.config.cli_workspace or None,
                capture_output=True,
                text=True,
                timeout=self.config.cli_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise BackendError("the Claude CLI timed out") from None
        except OSError as exc:
            raise BackendError(f"could not start the Claude CLI: {exc}") from None

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            raise BackendError(detail[-1] if detail else f"CLI exited {completed.returncode}")

        raw = (completed.stdout or "").strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return Result(text=raw or "The Claude CLI returned nothing.")
        text = payload.get("result") or payload.get("text") or raw
        return Result(text=str(text).strip(), meta={"mode": "cli", "session_id": payload.get("session_id", "")})
