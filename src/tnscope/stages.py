"""Stage 2 (NLP) and stage 3 (LLM judge) plugin contract.

The wire formats mirror ``tns-context-engine/schemas/{nlp,llm}_*.schema.json``.
Add-on packages register a scorer object under the ``tnscope.stage2`` or
``tnscope.stage3`` entry-point group; the core never imports one by name.

Request ``text`` is the sentence under review and request ``spans`` are offsets
into that text (original characters, code points, half-open), not into the
whole input.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

API_VERSION = 1
STAGE2_GROUP = "tnscope.stage2"
STAGE3_GROUP = "tnscope.stage3"
ANY_LANG = "*"


class SpanRef(BaseModel):
    span: tuple[int, int]
    concept: int


class NlpRequest(BaseModel):
    text: str
    lang: str
    spans: list[SpanRef]


class NlpSignals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    negation: bool | None = None
    quotation: bool | None = None
    target_person: bool | None = None
    intent: str | None = Field(default=None, min_length=1)
    sentiment: float | None = Field(default=None, ge=-1, le=1)


class NlpResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=100)
    signals: NlpSignals


class LlmRequest(BaseModel):
    text: str
    lang: str
    spans: list[SpanRef]
    prompt_template_id: str


class LlmVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: dict[str, float] = Field(min_length=1)
    label: str
    rationale: str | None = None
    model_id: str = Field(min_length=1)
    prompt_version: int = Field(ge=1)


@runtime_checkable
class Stage2Scorer(Protocol):
    api_version: int
    langs: frozenset[str]

    def score(self, request: NlpRequest) -> NlpResult | dict: ...


@runtime_checkable
class Stage3Scorer(Protocol):
    api_version: int
    langs: frozenset[str]

    def judge(self, request: LlmRequest) -> LlmVerdict | dict: ...


def covers(scorer, lang: str) -> bool:
    return lang in scorer.langs or ANY_LANG in scorer.langs


def _discover(group: str, method: str, warnings: list[str]) -> list:
    found = []
    for ep in entry_points(group=group):
        try:
            obj = ep.load()
            scorer = obj() if isinstance(obj, type) else obj
            if not callable(getattr(scorer, method, None)):
                raise TypeError("missing " + method)
            if getattr(scorer, "api_version", None) != API_VERSION:
                warnings.append(f"plugin_incompatible group={group} name={ep.name}")
                continue
            found.append(scorer)
        except Exception as exc:  # a broken add-on must not break the core
            warnings.append(f"plugin_load_failed group={group} name={ep.name} error={type(exc).__name__}")
    return found


def discover_stage2(warnings: list[str]) -> list:
    return _discover(STAGE2_GROUP, "score", warnings)


def discover_stage3(warnings: list[str]) -> list:
    return _discover(STAGE3_GROUP, "judge", warnings)
