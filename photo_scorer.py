from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ScoreResult:
    filename: str
    technical: float   # 1–10
    aesthetic: float   # 1–10
    content: float     # 1–10
    total: float       # weighted average, filled by compute_total
    reason: str        # lý do ngắn gọn bằng tiếng Việt


def compute_total(result: ScoreResult, weights: dict[str, float]) -> float:
    return (
        result.technical * weights["technical"]
        + result.aesthetic * weights["aesthetic"]
        + result.content * weights["content"]
    )


class VisionProvider(ABC):
    @abstractmethod
    def score(self, image_path: str) -> ScoreResult:
        """Score one image. Returns ScoreResult with total=0.0 (caller fills it)."""
