"""Span masking."""


def mask_text(text: str, spans: list[tuple[int, int]], mask_char: str = "*") -> str:
    """Replace each code point covered by half-open spans with ``mask_char``.

    Spans are merged when overlapping or adjacent; empty spans are ignored and
    spans past the end are clipped. Output length equals input length.
    """
    if len(mask_char) != 1:
        raise ValueError("mask_char must be exactly one character")
    n = len(text)
    clean: list[tuple[int, int]] = []
    for start, end in spans:
        if start < 0:
            raise ValueError("span start must be non-negative")
        end = min(end, n)
        if start >= end:
            continue
        clean.append((start, end))
    clean.sort()
    merged: list[list[int]] = []
    for start, end in clean:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    out: list[str] = []
    pos = 0
    for start, end in merged:
        out.append(text[pos:start])
        out.append(mask_char * (end - start))
        pos = end
    out.append(text[pos:])
    return "".join(out)
