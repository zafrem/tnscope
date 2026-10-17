"""Aho-Corasick prefilter + RE2 regex matcher over normalized text."""
from __future__ import annotations

from dataclasses import dataclass

import ahocorasick
import re2

from tnscope.data import Entry


@dataclass
class Match:
    entry: Entry
    start: int
    end: int


def fold(s: str) -> str:
    """Per-character lower(), keeping the original char if its length would change."""
    out = []
    for ch in s:
        lo = ch.lower()
        out.append(lo if len(lo) == 1 else ch)
    return "".join(out)


def _is_alnum_ascii(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z") or ("0" <= ch <= "9")


def _boundary_ok(text: str, start: int, end: int) -> bool:
    if start > 0 and _is_alnum_ascii(text[start - 1]):
        return False
    if end < len(text) and _is_alnum_ascii(text[end]):
        return False
    return True


class Matcher:
    def __init__(self, entries: list[Entry]):
        self._literals: list[Entry] = []
        self._regexes: list[tuple[Entry, object]] = []
        # automaton value: list of ("lit", idx) / ("kw", idx)
        payload: dict[str, list[tuple[str, int]]] = {}

        for entry in entries:
            if entry.type in ("literal", "phrase"):
                idx = len(self._literals)
                self._literals.append(entry)
                if entry.term:
                    payload.setdefault(fold(entry.term), []).append(("lit", idx))
            elif entry.type == "regex":
                pattern = entry.term
                if "IGNORECASE" in entry.flags:
                    pattern = "(?i)" + pattern
                try:
                    compiled = re2.compile(pattern)
                except Exception as exc:
                    raise ValueError(
                        f"entry {entry.id}: invalid regex ({type(exc).__name__})"
                    ) from None
                idx = len(self._regexes)
                self._regexes.append((entry, compiled))
                if not entry.always_run:
                    for kw in entry.hit_keywords:
                        if kw:
                            payload.setdefault(fold(kw), []).append(("kw", idx))
            else:
                raise ValueError(f"entry {entry.id}: unknown type {entry.type!r}")

        self._automaton = None
        if payload:
            auto = ahocorasick.Automaton()
            for key, val in payload.items():
                auto.add_word(key, val)
            auto.make_automaton()
            self._automaton = auto

    def find(self, text: str) -> list[Match]:
        results: list[Match] = []
        kw_hit: set[int] = set()
        if self._automaton is not None and text:
            folded = fold(text)
            for end_incl, vals in self._automaton.iter(folded):
                end = end_incl + 1
                for kind, idx in vals:
                    if kind == "kw":
                        kw_hit.add(idx)
                        continue
                    entry = self._literals[idx]
                    start = end - len(entry.term)
                    if start < 0:
                        continue
                    if "IGNORECASE" not in entry.flags and text[start:end] != entry.term:
                        continue
                    if "WORD_BOUNDARY" in entry.flags and not _boundary_ok(text, start, end):
                        continue
                    results.append(Match(entry, start, end))

        for idx, (entry, rx) in enumerate(self._regexes):
            if entry.always_run or not entry.hit_keywords or idx in kw_hit:
                wb = "WORD_BOUNDARY" in entry.flags
                for m in rx.finditer(text):
                    s, e = m.span()
                    if e == s:
                        continue
                    if wb and not _boundary_ok(text, s, e):
                        continue
                    results.append(Match(entry, s, e))

        results.sort(key=lambda m: (m.start, m.end))
        return results
