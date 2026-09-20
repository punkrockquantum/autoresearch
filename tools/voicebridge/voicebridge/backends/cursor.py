"""Cursor: launch a coding agent from a spoken instruction.

Three modes, because Cursor is an editor rather than a chat endpoint:

``api``    POST /v0/agents on api.cursor.com — a cloud agent picks up the task
           against a GitHub repo and opens a branch/PR. Nothing has to be
           running on your laptop, which is the point when you're out walking.
``cli``    ``cursor-agent -p "…"`` in a local checkout.
``queue``  Append the instruction to a markdown file in the repo. No API key,
           no agent; you open the file in Cursor when you sit down.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import CursorConfig
from ..http import HttpError, describe_http_error, request_json
from .base import Backend, BackendError, Request, Result


class CursorBackend(Backend):
    name = "cursor"

    def __init__(self, config: CursorConfig) -> None:
        self.config = config

    def available(self) -> tuple[bool, str]:
        mode = self.config.mode
        if mode == "api":
            if not self.config.api_key:
                return False, "CURSOR_API_KEY isn't set"
            if not self.config.repository:
                return False, "no target repository configured"
            return True, ""
        if mode == "cli":
            if shutil.which(self.config.cli_path) is None:
                return False, f"the {self.config.cli_path} CLI isn't on PATH"
            return True, ""
        if mode == "queue":
            if not self.config.queue_file:
                return False, "no queue file configured"
            return True, ""
        return False, f"unknown cursor mode {mode}"

    def run(self, request: Request) -> Result:
        if self.config.mode == "cli":
            return self._run_cli(request)
        if self.config.mode == "queue":
            return self._run_queue(request)
        return self._run_api(request)

    def _run_api(self, request: Request) -> Result:
        body: dict[str, Any] = {
            "prompt": {"text": request.prompt},
            "source": {"repository": self.config.repository, "ref": self.config.ref},
        }
        try:
            payload = request_json(
                "POST",
                f"{self.config.base_url}/v0/agents",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                json_body=body,
            )
        except HttpError as exc:
            raise BackendError(describe_http_error(exc)) from None

        agent_id = str(payload.get("id") or payload.get("agentId") or "")
        status = str(payload.get("status") or (payload.get("run") or {}).get("status") or "queued")
        url = _agent_url(payload)
        repo_name = self.config.repository.rstrip("/").split("/")[-1] or self.config.repository
        spoken = f"Cursor agent started on {repo_name}, status {status}. I'll leave the link in the text reply."
        text_parts = [f"Cursor agent {agent_id or '(no id)'} started on {self.config.repository} @ {self.config.ref}."]
        text_parts.append(f"Status: {status}")
        if url:
            text_parts.append(url)
        return Result(
            text="\n".join(text_parts),
            speech=spoken,
            meta={"agent_id": agent_id, "status": status, "url": url},
        )

    def status(self, agent_id: str) -> dict[str, Any]:
        """Poll a previously launched cloud agent (used by ``voicebridge cursor-status``)."""
        if not self.config.api_key:
            raise BackendError("CURSOR_API_KEY isn't set")
        try:
            return request_json(
                "GET",
                f"{self.config.base_url}/v0/agents/{agent_id}",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
            )
        except HttpError as exc:
            raise BackendError(describe_http_error(exc)) from None

    def _run_cli(self, request: Request) -> Result:
        command = [self.config.cli_path, "-p", request.prompt, "--output-format", "text"]
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
            raise BackendError("cursor-agent timed out") from None
        except OSError as exc:
            raise BackendError(f"could not start cursor-agent: {exc}") from None
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            raise BackendError(detail[-1] if detail else f"cursor-agent exited {completed.returncode}")
        text = (completed.stdout or "").strip() or "cursor-agent finished with no output."
        return Result(
            text=text,
            speech="Cursor agent finished. Details are in the text reply.",
            meta={"mode": "cli"},
        )

    def _run_queue(self, request: Request) -> Result:
        path = Path(self.config.queue_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        entry = f"- [ ] ({stamp}) {request.prompt}\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        return Result(
            text=f"Queued for Cursor in {path}:\n{entry.strip()}",
            speech="Queued that for Cursor.",
            meta={"mode": "queue", "queue_file": str(path)},
        )


def _agent_url(payload: dict[str, Any]) -> str:
    target = payload.get("target")
    if isinstance(target, dict):
        for key in ("url", "prUrl", "branchUrl"):
            value = target.get(key)
            if isinstance(value, str) and value:
                return value
    for key in ("url", "webUrl"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""
