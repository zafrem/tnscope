import pytest

from tnscope.data import Entry, load_data
from tnscope.match import Match, Matcher, fold


def E(term, type="literal", flags=(), hit_keywords=(), always_run=False, id="t1"):
    return Entry(
        id=id, concept=1, term=term, type=type, rank=3, ambiguity=0.0,
        requires_context=False, flags=tuple(flags), hit_keywords=tuple(hit_keywords),
        always_run=always_run, category="hate",
    )


def spans(ms):
    return [(m.start, m.end) for m in ms]


def test_literal_word_boundary():
    m = Matcher([E("wetback", flags=["IGNORECASE", "WORD_BOUNDARY"])])
    assert spans(m.find("a WETBACK.")) == [(2, 9)]
    assert m.find("xwetbacky") == []
    assert m.find("wetbacks") == []


def test_literal_case_sensitive_without_flag():
    m = Matcher([E("Foo")])
    assert spans(m.find("a Foo foo")) == [(2, 5)]


def test_phrase_ignorecase():
    m = Matcher([E("go back home", type="phrase", flags=["IGNORECASE"])])
    assert spans(m.find("Please GO Back Home now")) == [(7, 19)]


def test_regex_gated_by_hit_keywords():
    e = E(r"t\.me/\w+", type="regex", hit_keywords=["telegram"])
    m = Matcher([e])
    assert m.find("see t.me/abc") == []
    assert spans(m.find("telegram: t.me/abc")) == [(10, 18)]


def test_always_run_regex_without_keywords():
    assert spans(Matcher([E(r"\d{3}", type="regex", always_run=True)]).find("ab 123")) == [(3, 6)]
    assert spans(Matcher([E(r"\d{3}", type="regex")]).find("ab 123")) == [(3, 6)]


def test_regex_ignorecase_and_word_boundary():
    m = Matcher([E(r"bad\w*", type="regex", flags=["IGNORECASE", "WORD_BOUNDARY"], always_run=True)])
    assert spans(m.find("so BADLY")) == [(3, 8)]
    assert m.find("xbad") == []


def test_cjk_literal():
    m = Matcher([E("짱깨")])
    assert spans(m.find("그 짱깨 놈")) == [(2, 4)]


def test_overlapping_matches_all_reported():
    m = Matcher([E("ab", id="a"), E("bc", id="b")])
    assert sorted(spans(m.find("abc"))) == [(0, 2), (1, 3)]


def test_fold_keeps_length():
    assert fold("ABC") == "abc"
    s = "İ"  # lower() expands to 2 chars
    assert fold(s) == s
    assert len(fold("aİB")) == 3


def test_bundled_lexicons_load():
    d = load_data()
    assert len(d.languages) == 5
    for lang, entries in d.lexicon.items():
        assert entries, lang
        Matcher(entries)
    cats = {e.category for v in d.lexicon.values() for e in v}
    assert cats == {"hate", "defamation"}


def test_invalid_regex_error_has_no_term():
    with pytest.raises(ValueError) as ei:
        Matcher([E("(unclosed", type="regex", id="bad_1")])
    assert "bad_1" in str(ei.value)
    assert "unclosed" not in str(ei.value)
