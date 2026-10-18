"""Stage routing and score fusion (INTEGRATION.md pipeline, scoring.yml router/fuse)."""
from __future__ import annotations

from dataclasses import dataclass, field

from tnscope.stages import (
    LlmRequest,
    LlmVerdict,
    NlpRequest,
    NlpResult,
    SpanRef,
    covers,
)


@dataclass
class Refined:
    score: float
    stage: int = 1
    extra_signals: dict = field(default_factory=dict)


def _fuse(
    scores: dict[str, float], scoring: dict, has_floor: bool, frame: bool = False
) -> float:
    fuse = scoring.get("fuse") or {}
    method = fuse.get("method")
    if method == "veto" and frame and len(scores) > 1:
        # Stage 2/3 saw a non-abusive frame: their (higher) score caps stage 1.
        ceiling = max(v for k, v in scores.items() if k != "stage1")
        value = min(scores["stage1"], ceiling)
    elif method == "weighted":
        weights = fuse["weights"]
        total = sum(weights[k] for k in scores)
        value = sum(weights[k] * v for k, v in scores.items()) / total
    else:
        value = max(scores.values())
    value = max(0.0, min(100.0, value))
    if has_floor:
        value = max(value, float(scoring["floor"]["min_score"]))
    return float(value)


def refine(
    *,
    score1: float,
    signals: dict,
    has_floor: bool,
    lang: str,
    category: str,
    sentence: str,
    spans: list[tuple[int, int]],
    concept: int,
    scoring: dict,
    stage2: list,
    stage3: list,
    max_stage: int,
    warnings: list[str],
    verify_definite: bool = False,
) -> Refined:
    """Run stages 2 and 3 for one (sentence, concept) when the router says so.

    ``verify_definite`` also sends definite hits (score at or above
    ``router.definite_min``, not ambiguous) to the add-ons, so quoted, negated or
    reported uses of a strong term can be re-scored. The data's rank floor still
    applies afterwards.
    """
    router = scoring.get("router", {})
    undecided = signals.get("ambiguous") or signals.get("requires_context")
    definite = score1 >= router["definite_min"] and not undecided
    if max_stage < 2 or (definite and not verify_definite):
        return Refined(score1)

    refs = [SpanRef(span=sp, concept=concept) for sp in spans]
    scores = {"stage1": score1}
    out = Refined(score1)
    frame = False

    scorer = next((s for s in stage2 if covers(s, lang)), None)
    if scorer is not None:
        try:
            raw = scorer.score(NlpRequest(text=sentence, lang=lang, spans=refs))
            res = raw if isinstance(raw, NlpResult) else NlpResult.model_validate(raw)
            scores["nlp"] = res.score
            frame = res.non_abusive_frame
            out.stage = 2
            out.extra_signals["nlp"] = res.signals.model_dump(exclude_none=True)
        except Exception as exc:
            warnings.append(f"stage2_failed lang={lang} error={type(exc).__name__}")

    if max_stage >= 3 and (out.stage == 2 or scorer is None):
        hi, lo = router["nlp_confident_hi"], router["nlp_confident_lo"]
        if "nlp" not in scores or lo < scores["nlp"] < hi:
            judge = next((s for s in stage3 if covers(s, lang)), None)
            if judge is not None:
                try:
                    req = LlmRequest(
                        text=sentence, lang=lang, spans=refs,
                        prompt_template_id=f"{category}.{lang}",
                    )
                    raw = judge.judge(req)
                    v = raw if isinstance(raw, LlmVerdict) else LlmVerdict.model_validate(raw)
                    if category not in v.scores:
                        raise KeyError(category)
                    scores["llm"] = v.scores[category]
                    out.stage = 3
                    out.extra_signals["llm"] = {
                        "label": v.label, "model_id": v.model_id,
                        "prompt_version": v.prompt_version,
                    }
                except Exception as exc:
                    warnings.append(f"stage3_failed lang={lang} error={type(exc).__name__}")

    if out.stage > 1:
        out.score = _fuse(scores, scoring, has_floor, frame)
    return out
