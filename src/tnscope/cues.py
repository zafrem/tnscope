"""Shared base for rule-based stage 2 (context) scorers.

A language add-on supplies :class:`CuePatterns` (regular expressions for its
language) and inherits the scoring policy, so every add-on caps scores the same
way. The request carries no stage-1 score, so the score is absolute: ``BASE``,
raised by ``TARGET_ADJUST`` when a person is targeted, then capped when the
sentence shows the term is not used abusively. Caps sit below tnscope's default
flag threshold (35), so a capped sentence cannot be pushed over it by this stage.

The returned signals follow the NLP result schema: ``quotation``, ``negation``,
``target_person`` and an ``intent`` of ``report``, ``media`` or ``discussion``
(see :data:`tnscope.stages.NON_ABUSIVE_INTENTS`), which the ``veto`` fusion
method reads.

Patterns are the add-on's responsibility. Negation patterns in particular must
be anchored to the term or limited to one clause: a negation of some other verb
in the sentence must not count as negating the term.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from tnscope.stages import API_VERSION, NlpRequest, NlpResult, NlpSignals

BASE = 60.0
TARGET_ADJUST = 15.0
# Upper bounds applied when a non-abusive frame is detected (lowest cap wins).
QUOTE_CAP = 20.0
NEGATION_CAP = 20.0
REPORT_CAP = 25.0
MEDIA_CAP = 25.0
DISCUSS_CAP = 30.0


@dataclass(frozen=True)
class CuePatterns:
    """One language's cue patterns.

    ``quote_pairs``: two-character strings (open, close); a pair with equal
    characters is toggled by counting.
    ``after_quote`` / ``after_negation`` are matched at the start of the text
    after the term (use ``^``); ``before_negation`` is searched in the text
    before the term (use ``$``).
    ``target`` is searched in the sentence outside the term.
    """

    quote_pairs: tuple[str, ...]
    after_quote: re.Pattern
    after_negation: re.Pattern
    before_negation: re.Pattern
    report: re.Pattern
    media: re.Pattern
    discussion: re.Pattern
    target: re.Pattern


def is_quoted(text: str, a: int, b: int, patterns: CuePatterns) -> bool:
    before, after = text[:a], text[b:]
    for pair in patterns.quote_pairs:
        lo, hi = pair[0], pair[1]
        if lo != hi and before.rfind(lo) > before.rfind(hi) and hi in after:
            return True
        if lo == hi and before.count(lo) % 2 == 1 and lo in after:
            return True
    return bool(patterns.after_quote.match(after))


class CueScorer:
    """Stage 2 scorer driven by :class:`CuePatterns`; subclasses set ``langs`` and ``patterns``."""

    api_version = API_VERSION
    langs: frozenset[str]
    patterns: CuePatterns

    def score(self, request: NlpRequest) -> NlpResult:
        p, text = self.patterns, request.text
        quotation = negation = target = False
        for ref in request.spans:
            a, b = ref.span
            quotation = quotation or is_quoted(text, a, b, p)
            negation = negation or bool(
                p.after_negation.match(text[b:]) or p.before_negation.search(text[:a])
            )
            # Target cues anywhere in the sentence outside the flagged span.
            target = target or bool(p.target.search(text[:a] + " " + text[b:]))
        caps: list[tuple[float, str | None]] = []
        if quotation:
            caps.append((QUOTE_CAP, None))
        if negation:
            caps.append((NEGATION_CAP, None))
        if p.report.search(text):
            caps.append((REPORT_CAP, "report"))
        if p.media.search(text):
            caps.append((MEDIA_CAP, "media"))
        if p.discussion.search(text):
            caps.append((DISCUSS_CAP, "discussion"))
        score = BASE + (TARGET_ADJUST if target else 0.0)
        intent = None
        if caps:
            cap, intent = min(caps, key=lambda c: c[0])
            score = min(score, cap)
            intent = intent or next((i for _, i in caps if i), None)
        return NlpResult(
            score=max(0.0, min(100.0, score)),
            signals=NlpSignals(
                quotation=quotation, negation=negation, target_person=target, intent=intent
            ),
        )
