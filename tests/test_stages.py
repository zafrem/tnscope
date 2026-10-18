import jsonschema
import pytest
from importlib import resources
import json

from tnscope import Engine, LlmVerdict, NlpResult, NlpSignals

UNDECIDED = "Never say wetback."  # stage-1 score 70: below definite_min
DEFINITE = "Go back, wetback."  # stage-1 score 90, no ambiguous hit


class FakeNlp:
    api_version = 1
    langs = frozenset({"en"})

    def __init__(self, score=95.0, fail=False):
        self._score, self._fail, self.calls = score, fail, []

    def score(self, request):
        self.calls.append(request)
        if self._fail:
            raise RuntimeError("secret input text must not leak")
        return NlpResult(score=self._score, signals=NlpSignals(quotation=True))


class FakeJudge:
    api_version = 1
    langs = frozenset({"*"})

    def __init__(self, score=80.0, fail=False):
        self._score, self._fail, self.calls = score, fail, []

    def judge(self, request):
        self.calls.append(request)
        if self._fail:
            raise RuntimeError("boom")
        return LlmVerdict(scores={"hate": self._score}, label="probable",
                          rationale="quoted text here", model_id="m", prompt_version=1)


def engine(**kw):
    kw.setdefault("stage2", [])
    kw.setdefault("stage3", [])
    return Engine(**kw)


def det(result):
    return result.sentences[0].detections[0]


def test_no_plugins_stays_stage1():
    d = det(engine().scan(UNDECIDED, "en"))
    assert d.stage_reached == 1 and d.score == 70.0


def test_definite_hit_skips_stage2():
    nlp = FakeNlp()
    d = det(engine(stage2=[nlp]).scan(DEFINITE, "en"))
    assert nlp.calls == [] and d.stage_reached == 1


def test_stage2_runs_and_fuses_with_max():
    nlp = FakeNlp(score=95)
    r = engine(stage2=[nlp]).scan(UNDECIDED, "en")
    d = det(r)
    assert d.stage_reached == 2 and d.score == 95.0 and d.action == "block"
    assert d.signals["nlp"] == {"quotation": True}
    req = nlp.calls[0]
    assert req.text == UNDECIDED
    assert [sp.span for sp in req.spans] == [(10, 17)]
    assert UNDECIDED[10:17] == "wetback"


def test_stage2_can_lower_nothing_under_max_fuse():
    d = det(engine(stage2=[FakeNlp(score=5)]).scan(UNDECIDED, "en"))
    assert d.score == 70.0 and d.stage_reached == 2


def test_wrong_language_not_routed():
    nlp = FakeNlp()
    nlp.langs = frozenset({"ko"})
    d = det(engine(stage2=[nlp]).scan(UNDECIDED, "en"))
    assert nlp.calls == [] and d.stage_reached == 1


def test_max_stage_1_disables_plugins():
    nlp = FakeNlp()
    d = det(engine(stage2=[nlp]).scan(UNDECIDED, "en", max_stage=1))
    assert nlp.calls == [] and d.stage_reached == 1


def test_failing_stage2_falls_back_without_leaking_text():
    r = engine(stage2=[FakeNlp(fail=True)]).scan(UNDECIDED, "en")
    assert det(r).stage_reached == 1 and det(r).score == 70.0
    assert r.warnings == ["stage2_failed lang=en error=RuntimeError"]
    assert "secret" not in " ".join(r.warnings)


def test_invalid_stage2_result_falls_back():
    class Bad(FakeNlp):
        def score(self, request):
            return {"score": 500, "signals": {}}

    r = engine(stage2=[Bad()]).scan(UNDECIDED, "en")
    assert det(r).stage_reached == 1
    assert r.warnings and r.warnings[0].startswith("stage2_failed")


def test_judge_is_opt_in():
    judge = FakeJudge()
    nlp = FakeNlp(score=50)
    e = engine(stage2=[nlp], stage3=[judge])
    assert det(e.scan(UNDECIDED, "en")).stage_reached == 2
    assert judge.calls == []
    d = det(e.scan(UNDECIDED, "en", max_stage=3))
    assert d.stage_reached == 3 and d.score == 80.0
    assert judge.calls[0].prompt_template_id == "hate.en"
    assert d.signals["llm"] == {"label": "probable", "model_id": "m", "prompt_version": 1}
    assert "rationale" not in d.signals["llm"]


def test_confident_nlp_skips_judge():
    judge = FakeJudge()
    e = engine(stage2=[FakeNlp(score=95)], stage3=[judge])
    assert det(e.scan(UNDECIDED, "en", max_stage=3)).stage_reached == 2
    assert judge.calls == []


def test_judge_without_nlp_addon():
    d = det(engine(stage3=[FakeJudge(score=88)]).scan(UNDECIDED, "en", max_stage=3))
    assert d.stage_reached == 3 and d.score == 88.0


def test_failing_judge_keeps_stage2():
    r = engine(stage2=[FakeNlp(score=50)], stage3=[FakeJudge(fail=True)]).scan(
        UNDECIDED, "en", max_stage=3)
    assert det(r).stage_reached == 2
    assert r.warnings == ["stage3_failed lang=en error=RuntimeError"]


def test_floor_applies_to_fused_score():
    from tnscope.router import _fuse
    scoring = {"floor": {"rank": 5, "min_score": 60}, "fuse": {"method": "max"}}
    assert _fuse({"stage1": 10.0, "nlp": 0.0}, scoring, has_floor=True) == 60.0
    assert _fuse({"stage1": 10.0, "nlp": 0.0}, scoring, has_floor=False) == 10.0


def test_weighted_fuse(custom_data):
    import yaml
    data = custom_data("hate", "en", [])
    path = data / "scoring" / "scoring.yml"
    doc = yaml.safe_load(path.read_text())
    doc["fuse"]["method"] = "weighted"
    path.write_text(yaml.safe_dump(doc))
    d = det(Engine(data, stage2=[FakeNlp(score=100)], stage3=[]).scan(UNDECIDED, "en"))
    assert d.score == pytest.approx((0.4 * 70 + 0.3 * 100) / 0.7)


def test_stage_records_validate_against_schema():
    schema = json.loads((resources.files("tnscope") / "_data" / "schemas" / "detection.schema.json").read_text())
    r = engine(stage2=[FakeNlp(score=50)], stage3=[FakeJudge()]).scan(UNDECIDED, "en", max_stage=3)
    jsonschema.validate(det(r).model_dump(mode="json"), schema)


def test_invalid_max_stage():
    with pytest.raises(ValueError):
        engine().scan("x", "en", max_stage=4)


def test_discovery_skips_incompatible_and_broken(monkeypatch):
    from tnscope import stages

    class EP:
        def __init__(self, name, obj):
            self.name, self._obj = name, obj
        def load(self):
            if isinstance(self._obj, Exception):
                raise self._obj
            return self._obj

    class Old(FakeNlp):
        api_version = 99

    monkeypatch.setattr(stages, "entry_points", lambda group: [
        EP("good", FakeNlp()), EP("old", Old()), EP("broken", ImportError("x"))])
    warnings = []
    found = stages.discover_stage2(warnings)
    assert len(found) == 1
    assert warnings == [
        "plugin_incompatible group=tnscope.stage2 name=old",
        "plugin_load_failed group=tnscope.stage2 name=broken error=ImportError",
    ]


def test_definite_hit_verified_only_on_request():
    nlp = FakeNlp(score=5)
    e = engine(stage2=[nlp])
    e.scan(DEFINITE, "en")
    assert nlp.calls == []
    d = det(e.scan(DEFINITE, "en", verify_definite=True))
    assert len(nlp.calls) == 1 and d.stage_reached == 2


def test_verify_definite_cannot_drop_below_floor_under_weighted(custom_data):
    import yaml
    data = custom_data("hate", "en", [])
    path = data / "scoring" / "scoring.yml"
    doc = yaml.safe_load(path.read_text())
    doc["fuse"]["method"] = "weighted"
    path.write_text(yaml.safe_dump(doc))
    d = det(Engine(data, stage2=[FakeNlp(score=0)], stage3=[]).scan(
        DEFINITE, "en", verify_definite=True))
    assert d.score == 60.0 and d.action == "review"


def _veto_engine(custom_data, nlp):
    import yaml
    data = custom_data("hate", "en", [])
    path = data / "scoring" / "scoring.yml"
    doc = yaml.safe_load(path.read_text())
    doc["fuse"]["method"] = "veto"
    path.write_text(yaml.safe_dump(doc))
    return Engine(data, stage2=[nlp], stage3=[])


class FrameNlp(FakeNlp):
    def __init__(self, score, **signals):
        super().__init__(score)
        self._signals = signals

    def score(self, request):
        self.calls.append(request)
        return NlpResult(score=self._score, signals=NlpSignals(**self._signals))


def test_veto_without_frame_behaves_like_max(custom_data):
    d = det(_veto_engine(custom_data, FrameNlp(95)).scan(UNDECIDED, "en"))
    assert d.score == 95.0
    d = det(_veto_engine(custom_data, FrameNlp(5)).scan(UNDECIDED, "en"))
    assert d.score == 70.0


FRAMES = [
    {"quotation": True}, {"negation": True},
    {"intent": "report"}, {"intent": "media"}, {"intent": "discussion"},
]


@pytest.mark.parametrize("signals", FRAMES)
def test_veto_frame_caps_to_floor_for_rank5_hit(custom_data, signals):
    # "wetback" is rank 5: the data's rank floor keeps it at 60 (review).
    d = det(_veto_engine(custom_data, FrameNlp(20, **signals)).scan(UNDECIDED, "en"))
    assert d.score == 60.0 and d.action == "review" and d.stage_reached == 2


@pytest.mark.parametrize("signals", FRAMES)
def test_veto_frame_clears_hit_below_floor_rank(custom_data, signals):
    import yaml
    data = custom_data("hate", "en", [
        {"id": "hate_en_t001", "concept": 2001, "term": "blorp", "type": "literal",
         "rank": 3, "ambiguity": 0.0}])
    path = data / "scoring" / "scoring.yml"
    doc = yaml.safe_load(path.read_text())
    doc["fuse"]["method"] = "veto"
    path.write_text(yaml.safe_dump(doc))
    e = Engine(data, stage2=[FrameNlp(20, **signals)], stage3=[])
    d = det(e.scan("You blorp", "en"))
    assert d.score == 20.0 and d.action == "pass"
    assert det(Engine(data, stage2=[], stage3=[]).scan("You blorp", "en")).score == 60.0


def test_veto_other_intent_is_not_a_frame(custom_data):
    d = det(_veto_engine(custom_data, FrameNlp(20, intent="threat")).scan(UNDECIDED, "en"))
    assert d.score == 70.0 and d.action == "review"


def test_veto_rank_floor_still_applies_to_definite_hit(custom_data):
    e = _veto_engine(custom_data, FrameNlp(10, quotation=True))
    d = det(e.scan(DEFINITE, "en", verify_definite=True))
    assert d.score == 60.0 and d.action == "review"


def test_veto_never_raises_above_stage1(custom_data):
    d = det(_veto_engine(custom_data, FrameNlp(95, quotation=True)).scan(UNDECIDED, "en"))
    assert d.score == 70.0


def test_judge_can_override_the_frame(custom_data):
    import yaml
    e = _veto_engine(custom_data, FrameNlp(30, quotation=True))
    e.stage3 = [FakeJudge(score=80)]
    d = det(e.scan(UNDECIDED, "en", max_stage=3))
    assert d.score == 70.0 and d.stage_reached == 3  # ceiling max(30, 80) > stage 1
