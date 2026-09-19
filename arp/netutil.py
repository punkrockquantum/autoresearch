"""Tiny JSON-over-HTTP helper (urllib only, so the platform stays dependency-free).

Every caller treats the network as optional: a failure returns `None` and the
platform continues on its offline path rather than crashing a long research run
because a search API was rate-limited.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from arp.config import HTTP_TIMEOUT


class HttpError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status = status
        self.body = body


def post_json(
    url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None,
    timeout: float = HTTP_TIMEOUT,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HttpError(exc.code, exc.read().decode("utf-8", "replace")) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise HttpError(0, str(exc)) from exc


def get_json(
    url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None,
    timeout: float = HTTP_TIMEOUT,
) -> Dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, method="GET")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise HttpError(exc.code, exc.read().decode("utf-8", "replace")) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise HttpError(0, str(exc)) from exc
