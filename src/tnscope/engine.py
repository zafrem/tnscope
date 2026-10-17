"""Engine: ties normalize, match, sentences, scoring and masking together."""
from __future__ import annotations

import math
from bisect import bisect_right
from pathlib import Path

from tnscope.data import load_data
from tnscope.mask import mask_text
from tnscope.match import Matcher
from tnscope.models import Detection, ScanResult, SentenceResult, Thresholds
from tnscope.normalize import normalize
from tnscope.score import context_adjust, prepare_context, score_concept
from tnscope.sentences import split_sentences


class Engine:
    def __init__(self, data_dir: Path | None = None):
        self.data = load_data(data_dir)
        self._matchers = {
            lang: Matcher(self.data.lexicon.get(lang, [])) for lang in self.data.languages
        }
        self.default_thresholds = Thresholds.from_scoring(self.data.scoring)
        self._context = {
            lang: prepare_context(self.data.context_rules.get(lang, []))
            for lang in self.data.languages
        }

    def scan(
        self,
        text: str,
        lang: str,
        *,
        thresholds: Thresholds | None = None,
        mask_min_score: float | None = None,
        mask_char: str = "*",
    ) -> ScanResult:
        """Scan ``text`` (language ``lang``).

        ``thresholds`` defaults to the data's scoring.yml ladder
        (:attr:`default_thresholds`); ``mask_min_score`` defaults to the
        effective thresholds' ``flag`` value.
        """
        if lang not in self._matchers:
            raise ValueError(
                f"unsupported language; supported languages: {', '.join(sorted(self._matchers))}"
            )
        thresholds = thresholds or self.default_thresholds
        if mask_min_score is None:
            mask_min_score = thresholds.flag
        elif isinstance(mask_min_score, bool) or not isinstance(mask_min_score, (int, float)):
            raise ValueError("mask_min_score must be a finite number")
        elif not math.isfinite(mask_min_score):
            raise ValueError("mask_min_score must be a finite number")
        version = self.data.data_version
        if not text.strip():
            return ScanResult(sentences=[], masked_text=text, data_version=version)

        norm = normalize(text, self.data.normalize_rules.get(lang, []))
        matches = self._matchers[lang].find(norm.text)
        sentences = split_sentences(text)

        per_sentence: list[list[tuple]] = [[] for _ in sentences]
        starts = [s for s, _ in sentences]
        last = len(sentences) - 1
        for m in matches:
            o_start, o_end = norm.to_original_span(m.start, m.end)
            # Sentence containing o_start; a start in a gap between sentences goes
            # to the next sentence (or the last one if none follows). Never dropped.
            i = bisect_right(starts, o_start) - 1
            if i < 0:
                i = 0
            elif o_start >= sentences[i][1] and i < last:
                i += 1
            per_sentence[i].append((m, (o_start, o_end)))

        context_rules = self.data.context_rules.get(lang, [])
        prepared = self._context[lang]
        results: list[SentenceResult] = []
        mask_spans: list[tuple[int, int]] = []
        for (s, e), items in zip(sentences, per_sentence):
            groups: dict[int, list] = {}
            for m, _ in items:
                groups.setdefault(m.entry.concept, []).append(m)
            if groups:
                sent_norm = norm.slice_for(s, e)
                adjust = context_adjust(sent_norm, prepared)
            scores = {
                c: score_concept(ms, sent_norm, context_rules, self.data.scoring, adjust=adjust)
                for c, ms in groups.items()
            }
            ids = {c: sorted({m.entry.id for m in ms}) for c, ms in groups.items()}
            detections: list[Detection] = []
            for m, span in items:
                score = scores[m.entry.concept]
                label, action = thresholds.action_for(score, m.entry.category)
                detections.append(
                    Detection(
                        span=span,
                        concept=m.entry.concept,
                        category=m.entry.category,
                        score=score,
                        label=label,
                        action=action,
                        stage_reached=1,
                        matched_entry_ids=ids[m.entry.concept],
                    )
                )
                if score >= mask_min_score:
                    mask_spans.append(span)
            results.append(
                SentenceResult(
                    span=(s, e),
                    rank=max((m.entry.rank for m, _ in items), default=0),
                    top_score=max((d.score for d in detections), default=0.0),
                    detections=detections,
                )
            )
        return ScanResult(
            sentences=results,
            masked_text=mask_text(text, mask_spans, mask_char),
            data_version=version,
        )


_default: Engine | None = None


def default_engine() -> Engine:
    """The lazily created module-level Engine over the bundled data."""
    global _default
    if _default is None:
        _default = Engine()
    return _default


def scan(text: str, lang: str, **kwargs) -> ScanResult:
    """Scan with the default engine; keyword arguments go to :meth:`Engine.scan`."""
    return default_engine().scan(text, lang, **kwargs)
