import pytest

from tnscope.mask import mask_text


def test_single_span():
    assert mask_text("hello world", [(0, 5)]) == "***** world"


def test_overlapping_spans():
    assert mask_text("abcdefghij", [(0, 5), (3, 8)]) == "********ij"


def test_adjacent_spans():
    assert mask_text("abcdefgh", [(0, 3), (3, 6)]) == "******gh"


def test_empty_list_unchanged():
    assert mask_text("abc", []) == "abc"


def test_astral_text():
    assert mask_text("\U0001F600 abc", [(2, 5)]) == "\U0001F600 ***"


def test_custom_mask_char():
    assert mask_text("abc", [(0, 2)], "#") == "##c"


@pytest.mark.parametrize("bad", ["", "**"])
def test_mask_char_must_be_one_char(bad):
    with pytest.raises(ValueError):
        mask_text("abc", [(0, 1)], bad)


def test_unsorted_spans():
    assert mask_text("abcdefghij", [(6, 8), (0, 2), (1, 3)]) == "***def**ij"


def test_zero_width_span_ignored():
    assert mask_text("abc", [(1, 1), (2, 1)]) == "abc"


def test_span_clipped():
    assert mask_text("abc", [(1, 99)]) == "a**"


def test_negative_start_raises_without_text():
    with pytest.raises(ValueError) as e:
        mask_text("secret", [(-1, 2)])
    assert "secret" not in str(e.value)


def test_input_not_mutated():
    spans = [(6, 8), (0, 2)]
    mask_text("abcdefghij", spans)
    assert spans == [(6, 8), (0, 2)]
