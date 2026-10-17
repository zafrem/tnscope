from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
    stage_reached: Literal[1] = 1
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


class Thresholds(BaseModel):
    block: float = 85
    review: float = 60
    flag: float = 35
    category_block_min: dict[str, float] = Field(
        default_factory=lambda: {"illegal": 70, "defamation": 90}
    )

    def action_for(self, score: float, category: str) -> tuple[str, str]:
        block = self.category_block_min.get(category, self.block)
        if score >= block:
            return ("definite", "block")
        if score >= self.review:
            return ("probable", "review")
        if score >= self.flag:
            return ("possible", "flag")
        return ("clean", "pass")
