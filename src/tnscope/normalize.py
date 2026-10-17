"""Text normalizer that tracks a map back to original code-point offsets."""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class Normalized:
    """Normalized text plus, per output character, its original half-open range.

    ``origin[i]`` / ``origin_end[i]`` are the start / end (exclusive) code-point
    offsets in the original text that output character ``i`` came from. Both
    lists are non-decreasing (no rule reorders characters).
    """

    text: str
    origin: list[int]
    origin_end: list[int]

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        return (self.origin[start], self.origin_end[end - 1])

    def index_range(self, orig_start: int, orig_end: int) -> tuple[int, int]:
        """Normalized index range of characters whose origin lies in [orig_start, orig_end)."""
        lo = bisect_left(self.origin, orig_start)
        hi = bisect_left(self.origin, orig_end, lo)
        return lo, hi

    def slice_for(self, orig_start: int, orig_end: int) -> str:
        lo, hi = self.index_range(orig_start, orig_end)
        return self.text[lo:hi]


def _class_body(between: str) -> str:
    # Body of a character class (literals and a-b ranges); escape only specials.
    return re.sub(r"([\\\]\[^])", r"\\\1", between)


_JAMO_V_T = ((0x1160, 0x11FF), (0xD7B0, 0xD7FF))


@lru_cache(maxsize=65536)
def _starts_segment(prev: str, c: str) -> bool:
    """True when NFKC never combines ``c`` with the text before it."""
    first = unicodedata.normalize("NFKD", c)[:1] or c
    if unicodedata.combining(first):
        return False
    cp = ord(first)
    if any(lo <= cp <= hi for lo, hi in _JAMO_V_T):
        return False
    # Starter + starter compositions (e.g. Hangul LV + T, some Indic vowel signs).
    return unicodedata.normalize("NFKC", prev + c) == (
        unicodedata.normalize("NFKC", prev) + unicodedata.normalize("NFKC", c)
    )


def _seg_start(chars: list[str], j: int) -> bool:
    c = chars[j]
    return c < "\x80" or _starts_segment(chars[j - 1], c)


_BLOCK = 256


def _nfkc(chars: list[str], origin: list[int], origin_end: list[int]):
    """NFKC over composition segments; each output char maps to its segment's range.

    A segment is a starter plus everything NFKC may combine with it (combining
    marks, half-width voiced sound marks, Hangul V/T jamo, ...). Blocks of
    segments that are already NFKC-stable are copied unchanged.
    """
    n = len(chars)
    text = "".join(chars)
    if unicodedata.is_normalized("NFKC", text):
        return chars, origin, origin_end
    nc: list[str] = []
    no: list[int] = []
    ne: list[int] = []
    a = 0
    while a < n:
        b = min(a + _BLOCK, n)
        while b < n and not _seg_start(chars, b):
            b += 1
        if unicodedata.is_normalized("NFKC", text[a:b]):
            nc.extend(chars[a:b])
            no.extend(origin[a:b])
            ne.extend(origin_end[a:b])
            a = b
            continue
        i = a
        while i < b:
            j = i + 1
            while j < b and not _seg_start(chars, j):
                j += 1
            out = unicodedata.normalize("NFKC", text[i:j])
            start, end = origin[i], origin_end[j - 1]
            nc.extend(out)
            no.extend([start] * len(out))
            ne.extend([end] * len(out))
            i = j
        a = b
    return nc, no, ne


def _apply(chars: list[str], origin: list[int], origin_end: list[int], rule: dict):
    kind = rule.get("type")
    if kind == "strip":
        drop = set(rule["chars"])
        keep = [i for i, c in enumerate(chars) if c not in drop]
    elif kind == "map":
        mapping = rule["mapping"]
        if any(len(k) != 1 or len(v) != 1 for k, v in mapping.items()):
            raise ValueError(
                f"map rule {rule.get('id', '<no id>')!r}: keys and values must be single characters"
            )
        return [mapping.get(c, c) for c in chars], origin, origin_end
    elif kind == "nfkc":
        return _nfkc(chars, origin, origin_end)
    elif kind == "strip_between":
        b = _class_body(rule["between"])
        c = "".join(re.escape(x) for x in rule["chars"])
        pat = re.compile(f"(?<=[{b}])[{c}]+(?=[{b}])")
        s = "".join(chars)
        drop_idx: set[int] = set()
        for m in pat.finditer(s):
            drop_idx.update(range(m.start(), m.end()))
        keep = [i for i in range(len(chars)) if i not in drop_idx]
    elif kind == "collapse_repeats":
        max_run = rule["max_run"]
        keep = []
        run = 0
        for i, c in enumerate(chars):
            run = run + 1 if i and chars[i - 1] == c else 1
            if run <= max_run:
                keep.append(i)
    else:
        raise ValueError(f"unknown normalize rule type: {kind!r}")
    return (
        [chars[i] for i in keep],
        [origin[i] for i in keep],
        [origin_end[i] for i in keep],
    )


def normalize(text: str, rules: list[dict]) -> Normalized:
    chars = list(text)
    origin = list(range(len(chars)))
    origin_end = list(range(1, len(chars) + 1))
    for rule in rules:
        chars, origin, origin_end = _apply(chars, origin, origin_end, rule)
    return Normalized("".join(chars), origin, origin_end)
