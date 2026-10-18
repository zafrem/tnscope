from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["hate", "defamation"]
Label = Literal["definite", "probable", "possible", "clean"]
Action = Literal["block", "review", "flag", "pass"]


class Detection(BaseModel):
    span: tuple[int, int]
    concept: int
    category: Category
    score: float
    label: Label
    action: Action
    stage_reached: Literal[1, 2, 3] = 1
    matched_entry_ids: list[str]
    signals: dict = Field(default_factory=dict)


class SentenceResult(BaseModel):
    span: tuple[int, int]
    rank: int
    top_score: float
    detections: list[Detection]


class ScanResult(BaseModel):
    sentences: list[SentenceResult]
    masked_text: str
    data_version: str
    warnings: list[str] = Field(default_factory=list)


DEFAULT_BLOCK = 85.0


class Thresholds(BaseModel):
    """Score -> (label, action) ladder.

    ``category_block_min`` overrides ``block`` per category. A stricter override
    (higher than the default block, 85) never lets a caller's even stricter
    ``block`` be undercut: the effective block is ``max(block, block_min)``. A
    laxer override (e.g. illegal=70) applies as-is.
    """

    model_config = ConfigDict(allow_inf_nan=False, hide_input_in_errors=True)

    block: float = DEFAULT_BLOCK
    review: float = 60
    flag: float = 35
    category_block_min: dict[str, float] = Field(
        default_factory=lambda: {"illegal": 70, "defamation": 90}
    )

    @model_validator(mode="after")
    def _check_order(self):
        if not (self.flag <= self.review <= self.block):
            raise ValueError("thresholds must satisfy flag <= review <= block")
        return self

    @classmethod
    def from_scoring(cls, scoring: dict) -> "Thresholds":
        """Build thresholds from a scoring.yml mapping (thresholds ladder + overrides)."""
        kwargs: dict = {}
        for row in scoring.get("thresholds") or []:
            action = row.get("action")
            if action in ("block", "review", "flag") and action not in kwargs:
                kwargs[action] = float(row["min"])
        overrides = scoring.get("category_overrides")
        if overrides is not None:
            kwargs["category_block_min"] = {
                cat: float(v["block_min"])
                for cat, v in overrides.items()
                if isinstance(v, dict) and "block_min" in v
            }
        return cls(**kwargs)

    def block_for(self, category: str) -> float:
        block_min = self.category_block_min.get(category)
        if block_min is None:
            return self.block
        if block_min > DEFAULT_BLOCK:
            return max(self.block, block_min)
        return block_min

    def action_for(self, score: float, category: str) -> tuple[str, str]:
        block = self.block_for(category)
        if score >= block:
            return ("definite", "block")
        if score >= self.review:
            return ("probable", "review")
        if score >= self.flag:
            return ("possible", "flag")
        return ("clean", "pass")
