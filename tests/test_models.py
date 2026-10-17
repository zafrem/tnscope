import json
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
    assert Thresholds(block=50).action_for(55, "hate")[1] == "block"


def test_detection_dump_span_list():
    d = Detection(span=(0, 3), concept=1001, category="hate", score=90.0,
                  label="definite", action="block", matched_entry_ids=["a"])
    out = json.loads(d.model_dump_json())
    assert out["span"] == [0, 3] and out["stage_reached"] == 1
