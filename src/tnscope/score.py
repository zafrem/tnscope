"""Sentence-level stage-1 scoring for one concept."""
from __future__ import annotations

from tnscope.match import Match, fold


def prepare_context(context_rules: list[dict]) -> list[tuple[tuple[str, ...], float]]:
    """Pre-fold each context rule's terms: [(folded_terms, adjust), ...]."""
    return [(tuple(fold(t) for t in rule["terms"]), rule["adjust"]) for rule in context_rules]


def context_adjust(sentence_norm_text: str, prepared: list[tuple[tuple[str, ...], float]]) -> float:
    """Sum of the adjusts of context rules with any term in the sentence."""
    folded = fold(sentence_norm_text)
    return float(sum(adj for terms, adj in prepared if any(t in folded for t in terms)))


def score_concept(
    matches: list[Match],
    sentence_norm_text: str,
    context_rules: list[dict],
    scoring: dict,
    *,
    adjust: float | None = None,
) -> float:
    """Score matches sharing one concept within one sentence (0-100).

    ``adjust`` may carry a precomputed :func:`context_adjust` for the sentence;
    otherwise it is computed from ``context_rules``.
    """
    if not matches:
        return 0.0
    weights = scoring["rank_weight"]
    lexicon = max(weights[m.entry.rank - 1] * (1 - m.entry.ambiguity) for m in matches)

    corr = scoring["corroboration"]
    distinct = len({m.entry.id for m in matches})
    corroboration = min((distinct - 1) * corr["step"], corr["cap"])

    if adjust is None:
        adjust = context_adjust(sentence_norm_text, prepare_context(context_rules))

    result = max(0.0, min(100.0, lexicon + corroboration + adjust))

    floor = scoring["floor"]
    if any(m.entry.rank >= floor["rank"] for m in matches):
        result = max(result, float(floor["min_score"]))
    return float(result)
