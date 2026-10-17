"""Sentence splitting returning half-open code-point spans."""

import re

_BOUNDARY = re.compile(r"[.!?]+[\"'\u201d\u2019)\]]*(?=\s|\Z)|[。！？]+|\n")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split text into sentence spans [start, end) over the original text.

    Spans exclude leading/trailing whitespace and cover all other content.
    """
    spans: list[tuple[int, int]] = []

    def add(start: int, end: int) -> None:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))

    pos = 0
    for m in _BOUNDARY.finditer(text):
        end = m.start() if m.group() == "\n" else m.end()
        add(pos, end)
        pos = m.end()
    add(pos, len(text))
    return spans
