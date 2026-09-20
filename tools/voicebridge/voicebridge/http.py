"""Minimal HTTP helpers.

The gateway talks to four different vendor APIs and two messaging APIs. Rather
than pull in a client library, everything goes through these three functions so
that error reporting (which ends up spoken back into your ear) stays uniform.
"""

from __future__ import annotations

import json
import mimetypes
import os
import ssl
import urllib.error
import urllib.request
import uuid
from typing import Any

DEFAULT_TIMEOUT = 180.0


class HttpError(RuntimeError):
    """An HTTP call came back with a non-2xx status."""

    def __init__(self, status: int, url: str, body: str) -> None:
        self.status = status
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")


def describe_http_error(exc: "HttpError") -> str:
    """One short line about a failed call — this gets read aloud."""
    message = ""
    try:
        payload = json.loads(exc.body)
    except (json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("type") or "")
        elif isinstance(error, str):
            message = error
        if not message:
            message = str(payload.get("message") or payload.get("detail") or "")
    if not message:
        message = " ".join(exc.body.split())[:160]
    return f"HTTP {exc.status}. {message}".strip() if message else f"HTTP {exc.status}"


def _ssl_context() -> ssl.SSLContext:
    ca = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()


def request_bytes(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[bytes, str]:
    """Perform a request and return ``(body, content_type)``."""
    req = urllib.request.Request(url, data=body, method=method.upper())
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        detail = exc.read().decode("utf-8", "replace")
        raise HttpError(exc.code, url, detail) from None
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"could not reach {url}: {exc.reason}") from None


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    json_body: Any | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Perform a JSON request and decode the JSON response."""
    merged = {"Accept": "application/json", **(headers or {})}
    payload: bytes | None = None
    if json_body is not None:
        payload = json.dumps(json_body).encode("utf-8")
        merged.setdefault("Content-Type", "application/json")
    raw, _ = request_bytes(method, url, headers=merged, body=payload, timeout=timeout)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def post_multipart(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    fields: dict[str, str] | None = None,
    files: dict[str, tuple[str, bytes, str]] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """POST ``multipart/form-data`` and decode the JSON response.

    ``files`` maps a form field name to ``(filename, content, content_type)``.
    """
    boundary = f"----voicebridge{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in (fields or {}).items():
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for name, (filename, content, content_type) in (files or {}).items():
        content_type = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n".encode()
        )
        chunks.append(content)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    merged = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        **(headers or {}),
    }
    raw, _ = request_bytes("POST", url, headers=merged, body=body, timeout=timeout)
    return json.loads(raw.decode("utf-8")) if raw else {}
