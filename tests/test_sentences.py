from tnscope.sentences import split_sentences


def texts(s):
    return [s[a:b] for a, b in split_sentences(s)]


def test_basic_two():
    assert texts("One. Two!") == ["One.", "Two!"]


def test_no_terminator():
    assert texts("no end") == ["no end"]


def test_korean():
    assert len(split_sentences("안녕. 잘가")) == 2


def test_cjk_terminator_no_space():
    assert texts("こんにちは。さようなら") == ["こんにちは。", "さようなら"]


def test_newline():
    assert texts("line1\nline2") == ["line1", "line2"]


def test_empty_and_blank():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_code_point_offsets():
    s = "😀 hi. yo"
    assert split_sentences(s) == [(0, 5), (6, 8)]


def test_decimal_not_split():
    assert texts("3.14 is pi. ok") == ["3.14 is pi.", "ok"]


def test_repeated_terminators():
    assert texts("Wait... what?!") == ["Wait...", "what?!"]


def test_trims_whitespace():
    s = "  a.  b  "
    assert texts(s) == ["a.", "b"]
