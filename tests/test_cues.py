import re

import pytest

from tnscope import Engine
from tnscope.cues import (
    BASE,
    DISCUSS_CAP,
    MEDIA_CAP,
    NEGATION_CAP,
    QUOTE_CAP,
    REPORT_CAP,
    TARGET_ADJUST,
    CuePatterns,
    CueScorer,
    is_quoted,
)
from tnscope.stages import NlpRequest, SpanRef, Stage2Scorer


class ToyScorer(CueScorer):
    """A made-up language: quotes with [], 'said' after the term quotes it, etc."""

    langs = frozenset({"en"})
    patterns = CuePatterns(
        quote_pairs=("[]", '""'),
        after_quote=re.compile(r"^\s*(?:,? he said|,? she said)"),
        after_negation=re.compile(r"^\s*is not\b"),
        before_negation=re.compile(r"never say\s*$"),
        report=re.compile(r"police said|reported"),
        media=re.compile(r"\bmovie\b"),
        discussion=re.compile(r"\bresearch\b"),
        target=re.compile(r"\byou\b"),
    )


def sig(text, term):
    a = text.index(term)
    return ToyScorer().score(NlpRequest(
        text=text, lang="en", spans=[SpanRef(span=(a, a + len(term)), concept=1001)]))


def test_satisfies_the_stage2_protocol():
    assert isinstance(ToyScorer(), Stage2Scorer)
    assert ToyScorer.api_version == 1


def test_plain_and_target():
    assert sig("what a blorp", "blorp").score == BASE
    r = sig("you blorp", "blorp")
    assert r.score == BASE + TARGET_ADJUST and r.signals.target_person is True


@pytest.mark.parametrize("text,cap,signal", [
    ("he is a [blorp] today", QUOTE_CAP, "quotation"),
    ('"blorp", he said', QUOTE_CAP, "quotation"),
    ("blorp, she said", QUOTE_CAP, "quotation"),
    ("blorp is not true", NEGATION_CAP, "negation"),
    ("never say blorp", NEGATION_CAP, "negation"),
])
def test_quote_and_negation(text, cap, signal):
    r = sig(text, "blorp")
    assert r.score == cap and getattr(r.signals, signal) is True


def test_negation_must_follow_the_term_directly():
    # "is not" later in the sentence negates something else, not the term.
    assert sig("blorp was here and it is not late", "blorp").signals.negation is False


@pytest.mark.parametrize("text,cap,intent", [
    ("police said a blorp was held", REPORT_CAP, "report"),
    ("the movie had a blorp", MEDIA_CAP, "media"),
    ("research on blorp", DISCUSS_CAP, "discussion"),
])
def test_frames(text, cap, intent):
    r = sig(text, "blorp")
    assert r.score == cap < 35 and r.signals.intent == intent


def test_lowest_cap_wins_and_target_does_not_lift():
    r = sig("you said blorp is not true, police said", "blorp")
    assert r.score == NEGATION_CAP and r.signals.target_person is True


def test_is_quoted_same_char_pair_toggles_by_count():
    p = ToyScorer.patterns
    assert is_quoted('say "blorp" now', 5, 10, p) is True
    assert is_quoted('say blorp now', 4, 9, p) is False


def test_works_as_a_registered_stage2_scorer():
    # An add-on built on CueScorer is accepted by the engine and used for routing.
    e = Engine(stage2=[ToyScorer()], stage3=[])
    d = e.scan("Never say wetback.", "en").sentences[0].detections[0]
    assert d.stage_reached == 2 and d.signals["nlp"]["negation"] is False
