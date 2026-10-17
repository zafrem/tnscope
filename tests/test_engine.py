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
        dumped = d.model_dump(mode="json")
        assert set(dumped["signals"]) == {"ambiguous", "requires_context"}
        jsonschema.validate(dumped, schema)


def test_each_language_smoke():
    eng = Engine()
    for lang in eng.data.languages:
        if lang == "en":
            continue
        entries = eng.data.lexicon.get(lang) or []
        hate = [e for e in entries if e.category == "hate" and e.type == "literal"]
        assert hate, lang
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


def _custom_scoring(root, **ladder):
    import yaml

    path = root / "scoring" / "scoring.yml"
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    for row in doc["thresholds"]:
        if row["action"] in ladder:
            row["min"] = ladder[row["action"]]
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return root


def test_default_thresholds_come_from_data(custom_data):
    root = _custom_scoring(custom_data.root, block=95, review=70)
    eng = Engine(data_dir=root)
    assert eng.default_thresholds.block == 95
    d = eng.scan("Go back, wetback.", "en").sentences[0].detections[0]
    assert d.score == 90.0 and d.action == "review"


def _low_rank_root(custom_data):
    return custom_data("hate", "en", [{
        "id": "hate_en_9002", "concept": 2001, "term": "slurq", "type": "literal",
        "rank": 1, "ambiguity": 0.0, "flags": ["IGNORECASE", "WORD_BOUNDARY"],
    }])


def test_mask_default_follows_flag(custom_data):
    eng = Engine(data_dir=_low_rank_root(custom_data))
    r = eng.scan("a slurq b", "en")
    assert r.sentences[0].detections[0].score == 20.0
    assert r.masked_text == "a slurq b"
    r = eng.scan("a slurq b", "en", thresholds=Thresholds(flag=15))
    assert r.masked_text == "a ***** b"
    assert r.sentences[0].detections[0].action == "flag"


def test_mask_default_follows_data_flag(custom_data):
    root = _custom_scoring(_low_rank_root(custom_data), flag=10)
    assert Engine(data_dir=root).scan("a slurq b", "en").masked_text == "a ***** b"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_mask_min_score_non_finite_rejected(bad):
    with pytest.raises(ValueError) as ei:
        scan("secret wetback text", "en", mask_min_score=bad)
    assert "secret" not in str(ei.value)


def _dets(r):
    return [d for s in r.sentences for d in s.detections]


def test_signals_default_false():
    d = _dets(scan("You are disgusting scum.", "en"))[0]
    assert d.signals == {"ambiguous": False, "requires_context": False}


def test_signals_ambiguous_and_requires_context():
    d = _dets(scan("He is a fraudster.", "en"))[0]
    assert d.signals == {"ambiguous": True, "requires_context": True}


def test_signals_do_not_cap_actions():
    d = _dets(scan("Go back, wetback.", "en", thresholds=Thresholds(block=40, review=30, flag=20)))[0]
    assert d.action == "block"
    d = _dets(scan("He is a fraudster.", "en", thresholds=Thresholds(block=1, review=1, flag=1, category_block_min={})))[0]
    assert d.signals["ambiguous"] and d.action == "block"


def test_duplicate_span_concept_deduped(custom_data):
    root = custom_data("hate", "en", [
        {"id": "hate_en_9003", "concept": 2001, "term": "slurz", "type": "literal",
         "rank": 3, "ambiguity": 0.0, "flags": ["IGNORECASE"]},
        {"id": "hate_en_9004", "concept": 2001, "term": "slurz", "type": "literal",
         "rank": 5, "ambiguity": 0.6, "requires_context": True, "flags": ["IGNORECASE"]},
    ])
    r = Engine(data_dir=root).scan("a slurz b", "en")
    dets = _dets(r)
    assert len(dets) == 1
    assert dets[0].span == (2, 7)
    assert dets[0].matched_entry_ids == ["hate_en_9003", "hate_en_9004"]
    assert dets[0].signals == {"ambiguous": True, "requires_context": True}
    assert r.sentences[0].rank == 5


def test_hate_and_defamation_in_one_sentence():
    r = scan("You wetback, disgusting scum.", "en")
    assert len(r.sentences) == 1
    dets = r.sentences[0].detections
    assert sorted(d.category for d in dets) == ["defamation", "hate"]
    hate = next(d for d in dets if d.category == "hate")
    dfm = next(d for d in dets if d.category == "defamation")
    assert hate.matched_entry_ids == ["hate_en_0005"]
    assert dfm.matched_entry_ids == ["defamation_en_0004"]
    assert hate.score == 90.0 and dfm.score == 42.0
    assert r.sentences[0].top_score == 90.0


def test_defamation_87_is_review(custom_data):
    root = custom_data("defamation", "en", [{
        "id": "defamation_en_9005", "concept": 1001, "term": "slurd", "type": "literal",
        "rank": 5, "ambiguity": 0.13, "flags": ["IGNORECASE"],
    }])
    d = _dets(Engine(data_dir=root).scan("a slurd b", "en"))[0]
    assert d.score == 87.0
    assert (d.label, d.action) == ("probable", "review")


def test_mask_min_score_equal_to_score_masks():
    assert scan("Go back, wetback.", "en", mask_min_score=90).masked_text == "Go back, *******."
    assert scan("Go back, wetback.", "en", mask_min_score=90.01).masked_text == "Go back, wetback."


def test_quoted_sentence_context_not_leaking():
    r = scan('He said "you are a wetback." Never mind.', "en")
    assert len(r.sentences) == 2
    d = r.sentences[0].detections[0]
    assert d.score == 100.0  # 90 + target_person 15, capped; "never" is in sentence 2
    assert r.sentences[1].detections == []
