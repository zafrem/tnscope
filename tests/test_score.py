from dataclasses import replace

import pytest

from tnscope.data import load_data
from tnscope.match import Match
from tnscope.score import score_concept

D = load_data()
ENTRIES = {e.id: e for e in D.lexicon["en"]}
RULES = D.context_rules["en"]


def M(eid, entry=None):
    return Match(entry or ENTRIES[eid], 0, 1)


def score(matches, text="wetback"):
    return score_concept(matches, text, RULES, D.scoring)


def test_single_rank5():
    assert score([M("hate_en_0005")]) == 90.0


def test_corroboration():
    assert score([M("hate_en_0005"), M("hate_en_0006")]) == 100.0


def test_negation_context():
    assert score([M("hate_en_0005")], "never say wetback") == 70.0


def test_floor():
    assert score([M("hate_en_0005")], "never reported: wetback") == 60.0


def test_empty_matches():
    assert score([]) == 0.0


def test_clamped_at_100():
    ms = [M("hate_en_0005"), M("hate_en_0006")]
    assert score(ms, "you are a wetback, act now") == 100.0


def test_clamped_at_0():
    e = replace(ENTRIES["hate_en_0005"], rank=1, ambiguity=0.0)
    assert score([M(None, e)], "never reported: wetback") == 0.0


def test_rule_fires_once():
    # 'never' and 'not a' both belong to the negation rule: -20 only once.
    assert score([M("hate_en_0005")], "never not a wetback") == 70.0
