from importlib import resources
from pathlib import Path

import pytest
import yaml

from tnscope.data import load_data
from tnscope.normalize import normalize

_DIR = Path(str(resources.files("tnscope") / "_data" / "normalize"))


def _cases():
    out = []
    for p in sorted(_DIR.glob("*.yml")):
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
        for i, fx in enumerate(doc.get("fixtures") or []):
            out.append(pytest.param(doc["rules"], fx["input"], fx["expected"], id=f"{p.stem}-{i}"))
    return out


def test_fixture_discovery():
    langs = {p.stem for p in _DIR.glob("*.yml")}
    assert len(langs) >= 5
    assert len(_cases()) > 0


@pytest.mark.parametrize("rules,inp,expected", _cases())
def test_data_fixtures(rules, inp, expected):
    assert normalize(inp, rules).text == expected


def test_offsets_through_strip():
    rules = load_data().normalize_rules["en"]
    n = normalize("w.e.t.b.a.c.k", rules)
    assert n.text == "wetback"
    assert n.to_original_span(0, 7) == (0, 13)


def test_astral_offsets():
    rules = load_data().normalize_rules["en"]
    n = normalize("\U0001F600 w3tback", rules)
    assert n.text == "\U0001F600 wetback"
    i = n.text.index("wetback")
    assert n.to_original_span(i, i + 7) == (2, 9)
    assert n.slice_for(2, 9) == "wetback"


def test_unknown_rule_type():
    with pytest.raises(ValueError) as e:
        normalize("secret", [{"type": "bogus"}])
    assert "secret" not in str(e.value)


def test_map_rejects_multichar():
    with pytest.raises(ValueError) as e:
        normalize("straße", [{"id": "r1", "type": "map", "mapping": {"ß": "ss"}}])
    assert "r1" in str(e.value) and "straße" not in str(e.value)
    with pytest.raises(ValueError):
        normalize("x", [{"id": "r2", "type": "map", "mapping": {"ab": "c"}}])


def test_collapse_keeps_first_origins():
    n = normalize("aaaab", [{"type": "collapse_repeats", "max_run": 2}])
    assert n.text == "aab"
    assert n.origin == [0, 1, 4]


def test_nfkc_expansion_shares_origin():
    rules = load_data().normalize_rules["en"]
    n = normalize("x\ufb01", rules)
    assert n.text == "xfi"
    assert n.origin == [0, 1, 1]


def test_strip_between_edges_not_stripped():
    rule = {"type": "strip_between", "chars": ["."], "between": "a-z"}
    assert normalize(".a.b.", [rule]).text == ".ab."
