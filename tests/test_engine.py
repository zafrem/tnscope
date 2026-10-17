import json
from importlib import resources

import jsonschema
import pytest

from tnscope import Engine, Thresholds, scan


def test_wetback_example():
    r = scan("Go back, wetback.", "en")
    assert len(r.sentences) == 1
    s = r.sentences[0]
    assert s.rank == 5
    assert len(s.detections) == 1
    d = s.detections[0]
    assert d.span == (9, 16)
    assert d.score == 90.0
    assert d.action == "block"
    assert r.masked_text == "Go back, *******."


def test_context_lowers_action():
    d = scan("Never say wetback.", "en").sentences[0].detections[0]
    assert d.action == "review"
    assert d.score == 70.0


def test_threshold_override():
    r = scan("Go back, wetback.", "en", thresholds=Thresholds(block=95))
    assert r.sentences[0].detections[0].action == "review"


def test_evasion_offsets():
    r = scan("w.e.t.b.a.c.k", "en")
    assert r.masked_text == "*" * 13


def test_upper_case():
    assert scan("WETBACK", "en").sentences[0].detections


def test_astral_before_match():
    r = scan("😀 wetback", "en")
    assert r.sentences[0].detections[0].span == (2, 9)
    assert r.masked_text == "😀 *******"


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_empty_and_whitespace(text):
    r = scan(text, "en")
    assert r.sentences == []
    assert r.masked_text == text


def test_unsupported_lang_message():
    with pytest.raises(ValueError) as ei:
        scan("secret input text", "xx")
    msg = str(ei.value)
    assert "en" in msg
    assert "secret input text" not in msg


def test_clean_sentence_included():
    r = scan("Hello there. Go back, wetback.", "en")
    assert len(r.sentences) == 2
    assert r.sentences[0].rank == 0 and r.sentences[0].detections == []
    assert r.sentences[0].top_score == 0.0


def test_mask_min_score():
    r = scan("Go back, wetback.", "en", mask_min_score=95)
    assert r.masked_text == "Go back, wetback."
    assert r.sentences[0].detections


def test_detections_validate_against_schema():
    path = resources.files("tnscope") / "_data" / "schemas" / "detection.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    r = scan("Go back, wetback. Never say wetback.", "en")
    dets = [d for s in r.sentences for d in s.detections]
    assert dets
    for d in dets:
        jsonschema.validate(d.model_dump(mode="json"), schema)


def test_each_language_smoke():
    eng = Engine()
    for lang in eng.data.languages:
        if lang == "en":
            continue
        entries = eng.data.lexicon.get(lang) or []
        hate = [e for e in entries if e.category == "hate" and e.type == "literal"]
        if not hate:
            continue
        term = hate[0].term
        r = eng.scan(term, lang)
        assert sum(len(s.detections) for s in r.sentences) >= 1, (lang, term)


def test_halfwidth_dakuten_end_to_end():
    src = "在日ｺﾞｷﾌﾞﾘ"
    r = scan(src, "ja")
    dets = [d for s in r.sentences for d in s.detections]
    assert dets and dets[0].span == (0, len(src))
    assert r.masked_text == "*" * len(src)
    assert len(r.masked_text) == len(src)


def test_nfd_japanese_end_to_end():
    import unicodedata

    src = "あれは" + unicodedata.normalize("NFD", "在日ゴキブリ") + "だ"
    r = scan(src, "ja")
    dets = [d for s in r.sentences for d in s.detections]
    assert dets
    assert len(r.masked_text) == len(src)
    a, b = dets[0].span
    assert r.masked_text[a:b] == "*" * (b - a)
    assert src[:a] == "あれは" and src[b:] == "だ"


def test_match_starting_in_sentence_gap_not_dropped(custom_data):
    root = custom_data("hate", "en", [{
        "id": "hate_en_9001", "concept": 2001, "term": "\\s+slurx", "type": "regex",
        "rank": 5, "ambiguity": 0.0, "always_run": True,
    }])
    r = Engine(data_dir=root).scan("Hello.\n slurx here", "en")
    dets = [d for s in r.sentences for d in s.detections]
    assert len(dets) == 1
    assert "slurx" not in r.masked_text


def test_long_input_is_linear():
    import time

    base = (
        "The weather was mild and the market opened early today. "
        "Children walked to school while shop owners swept the pavement! "
        "Did anyone notice the new bakery near the station? "
    )
    hit = "Go back, wetback. "
    small = base * 3 + hit + base
    big_parts = []
    while sum(len(p) for p in big_parts) < 1_000_000:
        big_parts.append(small)
    big = "".join(big_parts)
    eng = Engine()
    small_r = eng.scan(small, "en")
    t0 = time.perf_counter()
    big_r = eng.scan(big, "en")
    elapsed = time.perf_counter() - t0
    assert elapsed < 10, elapsed
    reps = len(big_parts)
    assert len(big_r.sentences) == reps * len(small_r.sentences)
    assert big_r.masked_text == small_r.masked_text * reps
    n = len(small)
    for i, s in enumerate(big_r.sentences):
        ref = small_r.sentences[i % len(small_r.sentences)]
        off = (i // len(small_r.sentences)) * n
        assert s.span == (ref.span[0] + off, ref.span[1] + off)
        assert [(d.span[0] - off, d.span[1] - off, d.score, d.action) for d in s.detections] == [
            (*d.span, d.score, d.action) for d in ref.detections
        ]
