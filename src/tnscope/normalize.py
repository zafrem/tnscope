"""Text normalizer that tracks a map back to original code-point offsets."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class Normalized:
    text: str
    origin: list[int]

    def to_original_span(self, start: int, end: int) -> tuple[int, int]:
        return (self.origin[start], self.origin[end - 1] + 1)

    def slice_for(self, orig_start: int, orig_end: int) -> str:
        return "".join(
            c for c, o in zip(self.text, self.origin) if orig_start <= o < orig_end
        )


def _class_body(between: str) -> str:
    # Body of a character class (literals and a-b ranges); escape only specials.
    return re.sub(r"([\\\]\[^])", r"\\\1", between)


def _apply(chars: list[str], origin: list[int], rule: dict):
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
        return [mapping.get(c, c) for c in chars], origin
    elif kind == "nfkc":
        nc: list[str] = []
        no: list[int] = []
        for c, o in zip(chars, origin):
            for d in unicodedata.normalize("NFKC", c):
                nc.append(d)
                no.append(o)
        return nc, no
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
    return [chars[i] for i in keep], [origin[i] for i in keep]


def normalize(text: str, rules: list[dict]) -> Normalized:
    chars = list(text)
    origin = list(range(len(chars)))
    for rule in rules:
        chars, origin = _apply(chars, origin, rule)
    return Normalized("".join(chars), origin)
