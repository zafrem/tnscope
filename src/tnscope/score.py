"""Sentence-level stage-1 scoring for one concept."""
from __future__ import annotations

from tnscope.match import Match, fold


def score_concept(
    matches: list[Match],
    sentence_norm_text: str,
    context_rules: list[dict],
    scoring: dict,
) -> float:
    """Score matches sharing one concept within one sentence (0-100)."""
    if not matches:
        return 0.0
    weights = scoring["rank_weight"]
    lexicon = max(weights[m.entry.rank - 1] * (1 - m.entry.ambiguity) for m in matches)

    corr = scoring["corroboration"]
    distinct = len({m.entry.id for m in matches})
    corroboration = min((distinct - 1) * corr["step"], corr["cap"])

    folded = fold(sentence_norm_text)
    adjust = 0.0
    for rule in context_rules:
        if any(fold(t) in folded for t in rule["terms"]):
            adjust += rule["adjust"]

    result = max(0.0, min(100.0, lexicon + corroboration + adjust))

    floor = scoring["floor"]
    if any(m.entry.rank >= floor["rank"] for m in matches):
        result = max(result, float(floor["min_score"]))
    return float(result)
