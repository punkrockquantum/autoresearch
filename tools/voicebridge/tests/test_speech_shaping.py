from __future__ import annotations

from voicebridge.textutil import for_speech, format_citations, truncate_words


def test_code_blocks_are_not_read_aloud():
    spoken = for_speech("Use this:\n\n```python\nprint('hi')\n```\n\nThat's it.", 60)
    assert "print" not in spoken
    assert "code omitted" in spoken


def test_urls_and_markdown_are_flattened():
    spoken = for_speech(
        "**Bold** and _italic_ and `code` and a [link](https://example.com/page) and https://raw.example.com/x",
        60,
    )
    assert "**" not in spoken and "_italic_" not in spoken and "`" not in spoken
    assert "https://" not in spoken
    assert "link" in spoken


def test_bullets_and_headings_become_plain_sentences():
    spoken = for_speech("## Findings\n\n- first point\n- second point\n1. third point", 60)
    assert spoken.startswith("Findings")
    assert "-" not in spoken
    assert "first point" in spoken


def test_word_budget_backs_up_to_a_sentence_boundary():
    text = "One two three four five. Six seven eight nine ten. Eleven twelve."
    spoken = for_speech(text, 8)
    assert spoken == "One two three four five. That's the short version."


def test_a_clip_landing_on_a_sentence_end_is_left_alone():
    text = "One two three four five. Six seven eight nine ten. Eleven twelve."
    assert for_speech(text, 5) == "One two three four five."


def test_short_answers_pass_through_untouched():
    assert for_speech("Forty two.", 60) == "Forty two."


def test_truncate_words_handles_a_single_long_sentence():
    text = " ".join(f"word{index}" for index in range(50))
    clipped = truncate_words(text, 10)
    assert clipped.startswith("word0 ")
    assert "word20" not in clipped


def test_citations_are_numbered_and_capped():
    footer = format_citations([f"https://example.com/{index}" for index in range(9)])
    assert footer.startswith("Sources:")
    assert footer.count("http") == 5
    assert format_citations([]) == ""
