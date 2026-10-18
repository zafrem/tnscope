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
from tnscope.router import refine
from tnscope.sentences import split_sentences
from tnscope.stages import discover_stage2, discover_stage3


class Engine:
    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        stage2: list | None = None,
        stage3: list | None = None,
    ):
        """``stage2``/``stage3`` are scorer objects (see :mod:`tnscope.stages`);
        when ``None`` they are discovered from installed entry points."""
        self.data = load_data(data_dir)
        self.plugin_warnings: list[str] = []
        self.stage2 = discover_stage2(self.plugin_warnings) if stage2 is None else list(stage2)
        self.stage3 = discover_stage3(self.plugin_warnings) if stage3 is None else list(stage3)
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
        max_stage: int = 2,
        verify_definite: bool = False,
    ) -> ScanResult:
        """Scan ``text`` (language ``lang``).

        ``thresholds`` defaults to the data's scoring.yml ladder
        (:attr:`default_thresholds`); ``mask_min_score`` defaults to the
        effective thresholds' ``flag`` value.

        ``max_stage`` caps the pipeline: 1 = lexicon only, 2 (default) = also
        run an installed stage 2 add-on for undecided hits, 3 = additionally
        allow the LLM judge (opt-in; it may send text to a model). A failing
        add-on never raises: the previous stage's score stands and a code is
        appended to ``ScanResult.warnings``.

        ``verify_definite`` also sends definite stage-1 hits to the add-ons (off by
        default, which follows the data's router); it costs one add-on call per
        such hit and lets quoted or negated uses of strong terms be re-scored.
        """
        if isinstance(max_stage, bool) or max_stage not in (1, 2, 3):
            raise ValueError("max_stage must be 1, 2 or 3")
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
        warnings = list(self.plugin_warnings)
        if not text.strip():
            return ScanResult(sentences=[], masked_text=text, data_version=version, warnings=warnings)

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
        ambiguous_min = float(self.data.scoring.get("router", {}).get("ambiguous_min", 0.5))
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
            stages: dict[int, int] = {}
            extra: dict[int, dict] = {}
            if groups:
                spans_by_concept: dict[int, list[tuple[int, int]]] = {}
                for m, span in items:
                    lst = spans_by_concept.setdefault(m.entry.concept, [])
                    if span not in lst:
                        lst.append(span)
                for c, ms in groups.items():
                    ref = refine(
                        score1=scores[c],
                        signals={
                            "ambiguous": any(m.entry.ambiguity >= ambiguous_min for m in ms),
                            "requires_context": any(m.entry.requires_context for m in ms),
                        },
                        has_floor=any(m.entry.rank >= self.data.scoring["floor"]["rank"] for m in ms),
                        lang=lang,
                        category=ms[0].entry.category,
                        sentence=text[s:e],
                        spans=[(a - s, min(b, e) - s) for a, b in spans_by_concept[c]],
                        concept=c,
                        scoring=self.data.scoring,
                        stage2=self.stage2,
                        stage3=self.stage3,
                        max_stage=max_stage,
                        warnings=warnings,
                        verify_definite=verify_definite,
                    )
                    scores[c], stages[c], extra[c] = ref.score, ref.stage, ref.extra_signals
            ids = {c: sorted({m.entry.id for m in ms}) for c, ms in groups.items()}
            signals = {
                c: {
                    "ambiguous": any(m.entry.ambiguity >= ambiguous_min for m in ms),
                    "requires_context": any(m.entry.requires_context for m in ms),
                }
                for c, ms in groups.items()
            }
            # One detection per (original span, concept): the highest-rank entry.
            unique: dict[tuple, tuple] = {}
            for m, span in items:
                key = (span, m.entry.concept)
                prev = unique.get(key)
                if prev is None or m.entry.rank > prev[0].entry.rank:
                    unique[key] = (m, span)
            detections: list[Detection] = []
            for m, span in unique.values():
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
                        stage_reached=stages[m.entry.concept],
                        matched_entry_ids=ids[m.entry.concept],
                        signals={**signals[m.entry.concept], **extra[m.entry.concept]},
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
            warnings=warnings,
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
