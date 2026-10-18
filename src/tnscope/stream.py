"""Scan text that arrives in pieces (chat messages split by a transport, a model's output, a log tail).

``Engine.scan`` needs the whole text. :class:`StreamScanner` takes chunks of any size, holds back only the
sentence that is still being written, and settles each complete sentence exactly once. What it returns is the
same as scanning the whole text in one call:

* ``update.text`` pieces concatenated equal ``Engine.scan(text).masked_text``;
* detections carry the same scores and actions, with spans in code points of the whole stream.

This works because scoring is per sentence and a sentence boundary is final once another sentence has begun
(the splitter only ends a sentence at punctuation that is followed by more text). Each sentence is scanned once,
so a stage 2 add-on, and a stage 3 judge if you enabled it, run once per sentence and not once per chunk.

A single sentence with no boundary can grow without limit (a log line, text without punctuation). When the
pending text exceeds ``max_pending`` characters it is settled anyway, in front of the last ``overlap``
characters, at a word boundary, and never inside a stage 1 match. That path has less context for the cut
sentence (``update.forced`` is ``True``) and cannot see a match that starts before the cut and is longer than
``overlap`` once normalized (for example symbols inserted between every letter of a very long run). The longest
lexicon terms are 36 characters, so the default overlap of 128 leaves room for ordinary evasions.

Not thread-safe: use one scanner per stream. The engine itself can be shared. Chunks are ``str``; decode bytes
incrementally first. Nothing here logs or keeps text beyond the pending buffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tnscope.models import Detection, SentenceResult
from tnscope.sentences import split_sentences

if TYPE_CHECKING:
    from tnscope.engine import Engine

# Characters that can end a sentence or follow its final punctuation (see tnscope.sentences).
_CLOSERS = frozenset(".!?\u3002\uff01\uff1f\"'\u201d\u2019)]")
DEFAULT_OVERLAP = 128
DEFAULT_MAX_PENDING = 4096


@dataclass
class StreamUpdate:
    """What became final with one call.

    ``text`` is the masked text of the settled part, ready to forward. ``start``/``end`` are its offsets in the
    whole stream. ``sentences`` and their detections use stream offsets too. ``warnings`` holds only codes not
    reported before.
    """

    text: str = ""
    start: int = 0
    end: int = 0
    sentences: list[SentenceResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    forced: bool = False

    @property
    def detections(self) -> list[Detection]:
        return [d for s in self.sentences for d in s.detections]


def _shift_span(span: tuple[int, int], by: int) -> tuple[int, int]:
    return (span[0] + by, span[1] + by)


def _shifted(sentences: list[SentenceResult], by: int) -> list[SentenceResult]:
    out = []
    for s in sentences:
        dets = [d.model_copy(update={"span": _shift_span(d.span, by)}) for d in s.detections]
        out.append(s.model_copy(update={"span": _shift_span(s.span, by), "detections": dets}))
    return out


class StreamScanner:
    def __init__(
        self,
        engine: "Engine",
        lang: str,
        *,
        overlap: int = DEFAULT_OVERLAP,
        max_pending: int = DEFAULT_MAX_PENDING,
        **scan_kwargs,
    ) -> None:
        """``scan_kwargs`` go to :meth:`Engine.scan` (``thresholds``, ``mask_min_score``, ``mask_char``,
        ``max_stage``, ``verify_definite``)."""
        for name, v in (("overlap", overlap), ("max_pending", max_pending)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_pending < 2 * overlap:
            raise ValueError("max_pending must be at least twice overlap")
        engine.scan(" ", lang, **scan_kwargs)   # rejects an unsupported language or bad options now, not later
        self._engine = engine
        self._lang = lang
        self._kw = scan_kwargs
        self._overlap = overlap
        self._max_pending = max_pending
        self._buf = ""
        self._base = 0          # stream offset of self._buf[0]
        self._seen: set[str] = set()
        self._done = False

    @property
    def pending(self) -> int:
        """Characters held back because their sentence is not finished."""
        return len(self._buf)

    def feed(self, chunk: str) -> StreamUpdate:
        """Add a chunk; return whatever sentences are now complete (possibly nothing)."""
        if self._done:
            raise ValueError("stream already finished")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be str")
        old_len = len(self._buf)
        self._buf += chunk
        cut = self._sentence_cut(old_len)
        forced = False
        if cut == 0 and len(self._buf) > self._max_pending:
            cut = self._forced_cut()
            forced = cut > 0
        return self._settle(cut, forced)

    def finish(self) -> StreamUpdate:
        """The stream ended: settle everything that is pending. The scanner cannot be used afterwards."""
        if self._done:
            raise ValueError("stream already finished")
        self._done = True
        return self._settle(len(self._buf), False)

    # -- internals -----------------------------------------------------------------------------------------

    def _sentence_cut(self, old_len: int) -> int:
        """Start of the last sentence: everything before it is complete and its boundaries are final.

        The text before ``old_len`` held no confirmed boundary (otherwise it would have been settled), except
        possibly punctuation and spaces at its very end, which the new chunk can confirm. So only the text
        from there on needs splitting, which keeps the cost of a call independent of how long the pending
        sentence has grown.
        """
        i = old_len
        while i > 0 and (self._buf[i - 1].isspace() or self._buf[i - 1] in _CLOSERS):
            i -= 1
        start = max(0, i - 1)
        spans = split_sentences(self._buf[start:])
        return start + spans[-1][0] if len(spans) > 1 else 0

    def _forced_cut(self) -> int:
        """A cut inside one over-long sentence: at a word boundary at least ``overlap`` from the end and not
        inside a stage 1 match. Returns 0 when no safe cut exists yet."""
        limit = len(self._buf) - self._overlap
        cut = max((i + 1 for i in range(limit - 1, 0, -1) if self._buf[i].isspace()), default=0)
        if cut <= 0:
            return 0
        probe = self._engine.scan(self._buf, self._lang, max_stage=1)
        spans = [d.span for sent in probe.sentences for d in sent.detections]
        moved = True
        while moved and cut > 0:   # a match start may itself sit inside an earlier, overlapping match
            moved = False
            for a, b in spans:
                if a < cut < b:
                    cut, moved = a, True
        return cut

    def _settle(self, cut: int, forced: bool) -> StreamUpdate:
        if cut <= 0:
            return StreamUpdate(start=self._base, end=self._base)
        head, self._buf = self._buf[:cut], self._buf[cut:]
        result = self._engine.scan(head, self._lang, **self._kw)
        base, self._base = self._base, self._base + cut
        new = [w for w in dict.fromkeys(result.warnings) if w not in self._seen]
        self._seen.update(new)
        return StreamUpdate(
            text=result.masked_text, start=base, end=base + cut,
            sentences=_shifted(result.sentences, base), warnings=new, forced=forced,
        )
