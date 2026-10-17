import json
import math

import pytest

from tnscope.models import Detection, Thresholds


def test_action_ladder():
    t = Thresholds()
    assert t.action_for(90, "hate") == ("definite", "block")
    assert t.action_for(70, "hate") == ("probable", "review")
    assert t.action_for(40, "hate") == ("possible", "flag")
    assert t.action_for(10, "hate") == ("clean", "pass")


def test_defamation_block_min():
    t = Thresholds()
    assert t.action_for(87, "defamation") == ("probable", "review")
    assert t.action_for(90, "defamation") == ("definite", "block")


def test_custom_thresholds():
    assert Thresholds(block=50, review=40, flag=30).action_for(55, "hate")[1] == "block"


def test_detection_dump_span_list():
    d = Detection(span=(0, 3), concept=1001, category="hate", score=90.0,
                  label="definite", action="block", matched_entry_ids=["a"])
    out = json.loads(d.model_dump_json())
    assert out["span"] == [0, 3] and out["stage_reached"] == 1


def test_stricter_caller_block_applies_to_defamation():
    t = Thresholds(block=95)
    assert t.action_for(93, "defamation") == ("probable", "review")
    assert t.action_for(95, "defamation") == ("definite", "block")


def test_lower_caller_block_does_not_lower_defamation():
    t = Thresholds(block=50, review=40, flag=30)
    assert t.action_for(89, "defamation")[1] == "review"
    assert t.action_for(90, "defamation")[1] == "block"


def test_laxer_category_override_keeps_meaning():
    t = Thresholds(block=95)
    assert t.action_for(75, "illegal")[1] == "block"
    assert t.action_for(69.99, "illegal")[1] == "review"


@pytest.mark.parametrize("score,action", [
    (100, "block"), (85, "block"), (84.99, "review"), (60, "review"),
    (59.99, "flag"), (35, "flag"), (34.99, "pass"), (0, "pass"),
])
def test_ladder_boundaries(score, action):
    assert Thresholds().action_for(score, "hate")[1] == action


@pytest.mark.parametrize("category,score,action", [
    ("defamation", 89.99, "review"), ("defamation", 90, "block"),
    ("illegal", 69.99, "review"), ("illegal", 70, "block"),
])
def test_category_min_boundaries(category, score, action):
    assert Thresholds().action_for(score, category)[1] == action


@pytest.mark.parametrize("field", ["block", "review", "flag"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_rejects_non_finite(field, value):
    with pytest.raises(ValueError):
        Thresholds(**{field: value})


def test_rejects_non_finite_category_min():
    with pytest.raises(ValueError):
        Thresholds(category_block_min={"defamation": math.nan})


@pytest.mark.parametrize("kw", [{"flag": 70}, {"review": 90}, {"block": 50, "review": 60, "flag": 55}])
def test_rejects_inconsistent_order(kw):
    with pytest.raises(ValueError) as ei:
        Thresholds(**kw)
    msg = str(ei.value)
    assert "flag <= review <= block" in msg
    assert "input_value" not in msg


def test_equal_thresholds_allowed():
    t = Thresholds(block=50, review=50, flag=50)
    assert t.action_for(50, "hate")[1] == "block"


def test_from_scoring_reads_data():
    scoring = {
        "thresholds": [
            {"min": 95, "label": "definite", "action": "block"},
            {"min": 70, "label": "probable", "action": "review"},
            {"min": 20, "label": "possible", "action": "flag"},
            {"min": 0, "label": "clean", "action": "pass"},
        ],
        "category_overrides": {"defamation": {"block_min": 99}},
    }
    t = Thresholds.from_scoring(scoring)
    assert (t.block, t.review, t.flag) == (95, 70, 20)
    assert t.category_block_min == {"defamation": 99}


def test_from_scoring_bundled_matches_defaults():
    from tnscope.data import load_data

    assert Thresholds.from_scoring(load_data().scoring) == Thresholds()
