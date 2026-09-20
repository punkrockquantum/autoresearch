"""Turning model output into something that survives being read aloud."""

from __future__ import annotations

import re

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_URL = re.compile(r"https?://\S+")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(\S.*?\S|\S)\1", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


def for_speech(text: str, word_limit: int) -> str:
    """Flatten markdown, drop things that sound like noise, respect a word budget."""
    spoken = _CODE_FENCE.sub(" (code omitted, see the text reply) ", text)
    spoken = _MD_LINK.sub(r"\1", spoken)
    spoken = _URL.sub("a link in the text reply", spoken)
    spoken = _MD_HEADING.sub("", spoken)
    spoken = _MD_BULLET.sub("", spoken)
    spoken = _MD_EMPHASIS.sub(r"\2", spoken)
    spoken = _INLINE_CODE.sub(r"\1", spoken)
    spoken = _WHITESPACE.sub(" ", spoken).strip()
    return truncate_words(spoken, word_limit)


def truncate_words(text: str, word_limit: int) -> str:
    """Cut to ``word_limit`` words, preferring the last sentence boundary."""
    if word_limit <= 0:
        return text
    words = text.split()
    if len(words) <= word_limit:
        return text
    clipped = " ".join(words[:word_limit])
    if clipped.rstrip().endswith((".", "!", "?")):
        return clipped
    sentences = _SENTENCE_END.split(clipped)
    if len(sentences) > 1:
        kept = " ".join(sentences[:-1]).strip()
        if len(kept.split()) >= word_limit // 2:
            return f"{kept} That's the short version."
    return f"{clipped}… that's the short version."


def format_citations(citations: list[str], limit: int = 5) -> str:
    if not citations:
        return ""
    lines = [f"[{index}] {url}" for index, url in enumerate(citations[:limit], start=1)]
    return "Sources:\n" + "\n".join(lines)
