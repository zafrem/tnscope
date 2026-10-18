import pytest

from tnscope import Engine, StreamScanner
from tnscope.stages import NlpResult, NlpSignals

EN = (
    "Go back, wetback. He is not a fraudster, so stop spreading rumors.\n"
    'The teacher said "wetback" is a slur. Mike is a fraudster and a scumbag! '
    "Nothing to see here. Call me at 555-0100. They plan to burn down their mosques."
)
EDGE = ('He said "no."   Then wetback!!!  \n\n  Really?!  (see 3.14) Fine."\n'
        "A fraudster.\r\n\tB scumbag...   and wetback.")
KO = "그 땡중 정말 싫다. 오늘 날씨가 좋네요. 이 사기꾼 같은 놈아! 회의는 내일입니다."
ZH = "他就是个骗子。今天天气很好。你们这些小日本鬼子!明天开会。"


def chunks(text, n):
    return [text[i:i + n] for i in range(0, len(text), n)]


def run(engine, lang, text, n, **kw):
    sc = engine.stream(lang, **kw)
    updates = [sc.feed(c) for c in chunks(text, n)] + [sc.finish()]
    return updates


def detections(result_sentences):
    return [(d.span, d.concept, d.score, d.action, d.stage_reached) for s in result_sentences for d in s.detections]


@pytest.fixture(scope="module")
def engine():
    return Engine(stage2=[], stage3=[])


@pytest.mark.parametrize("lang,text", [("en", EN), ("en", EDGE), ("ko", KO), ("zh-hans", ZH)])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 25, 10_000])
def test_streaming_equals_one_full_scan(engine, lang, text, n):
    full = engine.scan(text, lang)
    updates = run(engine, lang, text, n)
    assert "".join(u.text for u in updates) == full.masked_text
    got = detections([s for u in updates for s in u.sentences])
    assert got == detections(full.sentences)
    # offsets: contiguous, covering the whole stream, and each detection points at its own text
    assert updates[0].start == 0 and updates[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(updates, updates[1:]))
    assert not any(u.forced for u in updates)


def test_offsets_are_stream_offsets(engine):
    updates = run(engine, "en", EN, 5)
    for u in updates:
        for d in u.detections:
            a, b = d.span
            assert EN[a:b].lower() in {"wetback", "fraudster", "scumbag", "burn down their mosques"}


def test_holds_back_only_the_unfinished_sentence(engine):
    sc = engine.stream("en")
    u = sc.feed("Go back, wetback. He is")
    assert u.text == "Go back, *******. " and (u.start, u.end) == (0, 18)
    assert sc.pending == len("He is")
    u = sc.feed(" a fraudster.")
    assert u.text == "" and sc.pending == len("He is a fraudster.")
    u = sc.finish()
    assert u.text.startswith("He is a ") and u.end == len("Go back, wetback. He is a fraudster.")


def test_a_term_split_across_chunks_is_still_found(engine):
    sc = engine.stream("en")
    out = sc.feed("Go back, wet").text + sc.feed("back. Fine.").text + sc.finish().text
    assert out == "Go back, *******. Fine."


def test_decimal_point_is_not_a_boundary_across_chunks(engine):
    sc = engine.stream("en")
    assert sc.feed("Pi is 3.").text == "" and sc.pending == len("Pi is 3.")
    assert sc.feed("14 and wetback. ").text == ""   # still the same sentence, which is not followed by another yet
    assert sc.finish().text == "Pi is 3.14 and *******. "


def test_empty_and_whitespace_chunks(engine):
    sc = engine.stream("en")
    assert sc.feed("").text == "" and sc.feed("   ").text == ""
    assert sc.finish().text == "   "


def test_finish_twice_or_feed_after_finish_is_an_error(engine):
    sc = engine.stream("en")
    sc.finish()
    with pytest.raises(ValueError):
        sc.finish()
    with pytest.raises(ValueError):
        sc.feed("x")


def test_bad_arguments(engine):
    with pytest.raises(ValueError):
        engine.stream("xx")
    with pytest.raises(ValueError):
        engine.stream("en", max_stage=9)
    with pytest.raises(ValueError):
        engine.stream("en", overlap=0)
    with pytest.raises(ValueError):
        engine.stream("en", overlap=100, max_pending=150)
    with pytest.raises(TypeError):
        engine.stream("en").feed(b"bytes")


def test_scan_options_are_passed_through(engine):
    sc = engine.stream("en", mask_char="#")
    assert sc.feed("Go back, wetback. Next.").text == "Go back, #######. "


# ── very long sentences: forced cuts ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pad", range(0, 60, 3))
@pytest.mark.parametrize("n", [1, 5, 37])
def test_forced_cut_never_splits_a_match(engine, pad, n):
    text = ("lorem ipsum " * 12) + "x" * pad + " go back to your country and then wetback and more words " * 3 + "end"
    full = engine.scan(text, "en")
    updates = run(engine, "en", text, n, overlap=40, max_pending=100)
    assert "".join(u.text for u in updates) == full.masked_text
    assert any(u.forced for u in updates)
    assert sorted(span for span, *_ in detections([s for u in updates for s in u.sentences])) == sorted(
        span for span, *_ in detections(full.sentences))


def test_forced_cut_with_no_whitespace_waits_instead_of_cutting_blindly(engine):
    text = "a" * 500 + " wetback"
    updates = run(engine, "en", text, 10, overlap=40, max_pending=100)
    assert "".join(u.text for u in updates) == engine.scan(text, "en").masked_text


def test_memory_stays_bounded_on_an_endless_sentence(engine):
    sc = engine.stream("en", overlap=40, max_pending=200)
    for _ in range(2000):
        sc.feed("word ")
        assert sc.pending <= 200 + 5


# ── add-ons and judges run once per sentence ───────────────────────────────────────────────────────────────

class CountingStage2:
    api_version = 1
    langs = frozenset({"en"})

    def __init__(self):
        self.calls = []

    def score(self, request):
        self.calls.append(request.text)
        return NlpResult(score=60.0, signals=NlpSignals())


def test_each_sentence_reaches_the_addon_once_whatever_the_chunking():
    text = "Mike is a fraudster. Ann is a fraudster too. Bob is fine."
    counts = {}
    for n in (1, 4, 1000):
        addon = CountingStage2()
        eng = Engine(stage2=[addon], stage3=[])
        updates = run(eng, "en", text, n)
        counts[n] = len(addon.calls)
        assert detections([s for u in updates for s in u.sentences]) == detections(eng.scan(text, "en").sentences)
    assert counts == {1: 2, 4: 2, 1000: 2}


def test_warnings_are_reported_once():
    class Boom:
        api_version = 1
        langs = frozenset({"en"})

        def score(self, request):
            raise RuntimeError("secret detail")

    eng = Engine(stage2=[Boom()], stage3=[])
    updates = run(eng, "en", "A fraudster. Another fraudster. Third fraudster.", 6)
    warnings = [w for u in updates for w in u.warnings]
    assert warnings == ["stage2_failed lang=en error=RuntimeError"]
    assert not any("secret" in w or "fraudster" in w for w in warnings)
