from __future__ import annotations

import pytest

from voicebridge.router import HELP_TEXT, Router, match_target, parse


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("claude", "claude"),
        ("cloud", "claude"),          # the mishearing you get most often
        ("clawed", "claude"),
        ("chat gpt", "chatgpt"),
        ("chat gbt", "chatgpt"),
        ("gpt", "chatgpt"),
        ("open ai", "chatgpt"),
        ("perplexity", "perplexity"),
        ("complexity", "perplexity"),
        ("curser", "cursor"),
        ("cursor", "cursor"),
        ("bananas", None),
    ],
)
def test_match_target_tolerates_mishearings(phrase, expected):
    assert match_target(phrase) == expected


@pytest.mark.parametrize(
    ("line", "target", "prompt"),
    [
        ("Claude, why is my loss spiking?", "claude", "why is my loss spiking?"),
        ("hey claude what's the capital of Peru", "claude", "what's the capital of Peru"),
        ("ask chat gpt to summarise the Muon optimizer", "chatgpt", "summarise the Muon optimizer"),
        ("perplexity what did the Fed do today", "perplexity", "what did the Fed do today"),
        ("Cursor, add a unit test for the dataloader", "cursor", "add a unit test for the dataloader"),
        ("okay cloud explain bits per byte", "claude", "explain bits per byte"),
    ],
)
def test_parse_peels_the_addressed_assistant(line, target, prompt):
    parsed = parse(line)
    assert parsed.kind == "prompt"
    assert parsed.target == target
    assert parsed.prompt == prompt
    assert parsed.explicit_target is True


def test_parse_without_a_name_leaves_target_open():
    parsed = parse("what's the weather in Zurich")
    assert parsed.kind == "prompt"
    assert parsed.target is None
    assert parsed.explicit_target is False
    assert parsed.prompt == "what's the weather in Zurich"


@pytest.mark.parametrize(
    ("line", "kind"),
    [
        ("new chat", "reset"),
        ("Reset.", "reset"),
        ("start over", "reset"),
        ("repeat that", "repeat"),
        ("say that again", "repeat"),
        ("status", "status"),
        ("help", "help"),
        ("", "empty"),
        ("   ", "empty"),
    ],
)
def test_parse_recognises_control_phrases(line, kind):
    assert parse(line).kind == kind


@pytest.mark.parametrize("line", ["switch to perplexity", "use complexity", "set default to perplexity"])
def test_parse_switches_target(line):
    parsed = parse(line)
    assert parsed.kind == "switch"
    assert parsed.target == "perplexity"


def test_parse_extracts_style_hints():
    assert parse("claude explain Muon in detail").style == "long"
    assert parse("claude explain Muon in detail").prompt == "explain Muon"
    assert parse("claude briefly explain Muon").style == "short"
    assert parse("claude explain Muon").style == "normal"


@pytest.mark.parametrize(
    ("line", "kind"),
    [("claude, new chat", "reset"), ("cursor status", "status"), ("claude repeat that", "repeat")],
)
def test_an_address_plus_a_command_is_still_a_command(line, kind):
    parsed = parse(line)
    assert parsed.kind == kind
    assert parsed.explicit_target is True


def test_bare_name_switches_rather_than_prompting():
    parsed = parse("cursor")
    assert parsed.kind == "switch"
    assert parsed.target == "cursor"


def test_router_routes_to_the_named_backend(config, sessions, backends):
    router = Router(config, sessions, backends)
    reply = router.handle("perplexity what's new in fusion", "test:1")
    assert reply.target == "perplexity"
    assert reply.text == "Perplexity's answer."
    assert backends["perplexity"].calls[0].prompt == "what's new in fusion"
    # An explicitly named assistant becomes the default for the next line.
    assert sessions.get("test:1").target == "perplexity"
    follow_up = router.handle("and in fission", "test:1")
    assert follow_up.target == "perplexity"


def test_router_falls_back_to_the_session_default(config, sessions, backends):
    router = Router(config, sessions, backends)
    reply = router.handle("what is bits per byte", "test:2")
    assert reply.target == "claude"


def test_router_target_override_is_not_sticky(config, sessions, backends):
    router = Router(config, sessions, backends)
    reply = router.handle("summarise this", "test:3", target_override="chatgpt")
    assert reply.target == "chatgpt"
    assert sessions.get("test:3").target == "claude"


def test_router_keeps_history_per_channel(config, sessions, backends):
    router = Router(config, sessions, backends)
    router.handle("claude first question", "a:1")
    router.handle("claude second question", "a:1")
    router.handle("claude other channel", "b:1")
    assert [turn.content for turn in backends["claude"].calls[1].history] == [
        "first question",
        "Claude's answer.",
    ]
    assert backends["claude"].calls[2].history == []


def test_router_reset_clears_history(config, sessions, backends):
    router = Router(config, sessions, backends)
    router.handle("claude remember this", "c:1")
    reply = router.handle("new chat", "c:1")
    assert reply.kind == "reset"
    assert sessions.get("c:1").turns == []


def test_router_repeat_returns_the_last_answer(config, sessions, backends):
    router = Router(config, sessions, backends)
    router.handle("claude tell me something", "d:1")
    reply = router.handle("repeat that", "d:1")
    assert reply.kind == "repeat"
    assert reply.text == "Claude's answer."


def test_router_help_is_spoken_verbatim(config, sessions, backends):
    router = Router(config, sessions, backends)
    reply = router.handle("what can you do", "e:1")
    assert reply.speech == HELP_TEXT


def test_router_reports_an_unconfigured_backend(config, sessions, backends):
    backends["cursor"].ready = False
    backends["cursor"].reason = "CURSOR_API_KEY isn't set"
    router = Router(config, sessions, backends)
    reply = router.handle("cursor fix the dataloader", "f:1")
    assert reply.kind == "error"
    assert "CURSOR_API_KEY" in reply.text
    assert not backends["cursor"].calls


def test_router_turns_backend_failure_into_a_spoken_message(config, sessions, backends):
    backends["claude"].fail = "HTTP 429. rate limited"
    router = Router(config, sessions, backends)
    reply = router.handle("claude hello", "g:1")
    assert reply.kind == "error"
    assert "rate limited" in reply.speech
    # A failed exchange must not poison the conversation history.
    assert sessions.get("g:1").turns == []


def test_spoken_requests_ask_for_a_short_answer(config, sessions, backends):
    router = Router(config, sessions, backends)
    router.handle("claude hi", "h:1", spoken=True)
    router.handle("claude hi", "h:2", spoken=False)
    spoken_system = backends["claude"].calls[0].system
    text_system = backends["claude"].calls[1].system
    assert "read aloud" in spoken_system and str(config.spoken_word_limit) in spoken_system
    assert "read aloud" not in text_system
