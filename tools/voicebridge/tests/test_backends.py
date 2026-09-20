from __future__ import annotations

import json

import pytest

from voicebridge import http as vb_http
from voicebridge.backends.base import BackendError, Request
from voicebridge.backends.chatgpt import ChatGPTBackend
from voicebridge.backends.claude import ClaudeBackend
from voicebridge.backends.cursor import CursorBackend
from voicebridge.backends.perplexity import PerplexityBackend
from voicebridge.config import ChatGPTConfig, ClaudeConfig, CursorConfig, PerplexityConfig
from voicebridge.sessions import Turn


@pytest.fixture
def capture(monkeypatch):
    """Intercept outbound JSON calls and serve canned responses."""
    recorded: list[dict] = []
    responses: list[object] = []

    def fake_request_json(method, url, *, headers=None, json_body=None, timeout=None):
        recorded.append({"method": method, "url": url, "headers": headers or {}, "body": json_body})
        if not responses:
            return {}
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    for module in ("claude", "chatgpt", "perplexity", "cursor"):
        monkeypatch.setattr(f"voicebridge.backends.{module}.request_json", fake_request_json)
    return recorded, responses


def test_claude_api_call_shape(capture):
    recorded, responses = capture
    responses.append(
        {"model": "claude-opus-5", "content": [{"type": "text", "text": "42"}], "usage": {"input_tokens": 5}}
    )
    backend = ClaudeBackend(ClaudeConfig(api_key="sk-ant-test", model="claude-opus-5"))
    history = [Turn("user", "hi"), Turn("assistant", "hello")]
    request = Request(prompt="what is six times seven", history=history, system="be brief")
    result = backend.run(request)

    assert result.text == "42"
    call = recorded[0]
    assert call["url"].endswith("/v1/messages")
    assert call["headers"]["x-api-key"] == "sk-ant-test"
    assert call["headers"]["anthropic-version"]
    assert call["body"]["system"] == "be brief"
    assert call["body"]["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what is six times seven"},
    ]


def test_claude_reports_http_errors_briefly(capture):
    _, responses = capture
    responses.append(
        vb_http.HttpError(
            401,
            "https://api.anthropic.com/v1/messages",
            json.dumps({"error": {"message": "invalid x-api-key"}}),
        )
    )
    backend = ClaudeBackend(ClaudeConfig(api_key="nope"))
    with pytest.raises(BackendError) as excinfo:
        backend.run(Request(prompt="hello"))
    assert "401" in str(excinfo.value)
    assert "invalid x-api-key" in str(excinfo.value)


def test_claude_requires_a_key():
    ready, reason = ClaudeBackend(ClaudeConfig(api_key="")).available()
    assert not ready
    assert "ANTHROPIC_API_KEY" in reason


def test_chatgpt_sends_system_prompt_first(capture):
    recorded, responses = capture
    responses.append({"model": "gpt-5", "choices": [{"message": {"content": "sure"}}]})
    backend = ChatGPTBackend(ChatGPTConfig(api_key="sk-test", model="gpt-5"))
    backend.run(Request(prompt="hi", system="be brief"))

    body = recorded[0]["body"]
    assert recorded[0]["url"].endswith("/v1/chat/completions")
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["max_completion_tokens"] > 0


def test_chatgpt_retries_with_legacy_token_field(capture):
    recorded, responses = capture
    responses.append(
        vb_http.HttpError(
            400,
            "https://api.openai.com/v1/chat/completions",
            json.dumps({"error": {"message": "Unsupported parameter: 'max_completion_tokens'"}}),
        )
    )
    responses.append({"choices": [{"message": {"content": "ok"}}]})
    backend = ChatGPTBackend(ChatGPTConfig(api_key="sk-test"))
    result = backend.run(Request(prompt="hi"))

    assert result.text == "ok"
    assert "max_completion_tokens" in recorded[0]["body"]
    assert "max_tokens" in recorded[1]["body"]


def test_perplexity_uses_the_agent_api(capture):
    recorded, responses = capture
    responses.append(
        {
            "status": "succeeded",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Rates held steady."}]},
                {"type": "search_results", "results": [{"url": "https://example.com/fed"}]},
            ],
        }
    )
    backend = PerplexityBackend(PerplexityConfig(api_key="pplx-test"))
    result = backend.run(Request(prompt="what did the fed do", system="be brief"))

    call = recorded[0]
    assert call["url"].endswith("/v1/agent")
    assert call["body"]["input"][-1] == {"role": "user", "content": "what did the fed do"}
    assert call["body"]["instructions"] == "be brief"
    assert call["body"]["tools"] == [{"type": "web_search"}]
    assert "messages" not in call["body"]
    assert result.text == "Rates held steady."
    assert "https://example.com/fed" in result.footer


def test_perplexity_handles_output_text_shortcut(capture):
    _, responses = capture
    responses.append({"status": "succeeded", "output_text": "Short answer."})
    result = PerplexityBackend(PerplexityConfig(api_key="pplx-test")).run(Request(prompt="hi"))
    assert result.text == "Short answer."
    assert result.footer == ""


def test_perplexity_surfaces_a_failed_run(capture):
    _, responses = capture
    responses.append({"status": "failed", "error": {"message": "search unavailable"}, "output": []})
    with pytest.raises(BackendError) as excinfo:
        PerplexityBackend(PerplexityConfig(api_key="pplx-test")).run(Request(prompt="hi"))
    assert "search unavailable" in str(excinfo.value)


def test_cursor_launches_a_cloud_agent(capture):
    recorded, responses = capture
    responses.append(
        {
            "id": "bc_abc123",
            "status": "RUNNING",
            "target": {"url": "https://cursor.com/agents/bc_abc123"},
        }
    )
    backend = CursorBackend(
        CursorConfig(api_key="key_test", repository="https://github.com/me/autoresearch", ref="main")
    )
    result = backend.run(Request(prompt="add a test for the dataloader"))

    call = recorded[0]
    assert call["url"].endswith("/v0/agents")
    assert call["headers"]["Authorization"] == "Bearer key_test"
    assert call["body"] == {
        "prompt": {"text": "add a test for the dataloader"},
        "source": {"repository": "https://github.com/me/autoresearch", "ref": "main"},
    }
    assert "bc_abc123" in result.text
    assert "cursor.com/agents/bc_abc123" in result.text
    # The spoken form must not read a URL out loud.
    assert "http" not in result.speech


def test_cursor_api_mode_needs_a_repository():
    ready, reason = CursorBackend(CursorConfig(api_key="key_test", repository="")).available()
    assert not ready
    assert "repository" in reason


def test_cursor_queue_mode_appends_to_a_file(tmp_path):
    queue = tmp_path / "PROMPT_QUEUE.md"
    backend = CursorBackend(CursorConfig(mode="queue", queue_file=str(queue)))
    assert backend.available() == (True, "")
    backend.run(Request(prompt="refactor the tokenizer"))
    backend.run(Request(prompt="add type hints"))

    lines = queue.read_text().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("- [ ] (")
    assert "refactor the tokenizer" in lines[0]
    assert "add type hints" in lines[1]
