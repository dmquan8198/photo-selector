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


import json
import os

SCORING_PROMPT = """Evaluate this photo and return ONLY a JSON object with these exact keys:
- "technical": score 1-10 for sharpness, exposure, focus, no blur/noise
- "aesthetic": score 1-10 for composition, color harmony, visual balance
- "content": score 1-10 for subject clarity and appeal (faces/expressions if people present, otherwise main subject quality)
- "reason": one short sentence in Vietnamese explaining the top strength or weakness

Return ONLY valid JSON, no extra text. Example:
{"technical": 8.5, "aesthetic": 7.0, "content": 9.0, "reason": "Ánh sáng tốt, khuôn mặt rõ nét"}"""


try:
    import google.generativeai as genai
    import PIL.Image
except ImportError:
    genai = None
    PIL = None


class GeminiProvider(VisionProvider):
    def __init__(self, api_key: str) -> None:
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")

    def score(self, image_path: str) -> ScoreResult:
        import PIL.Image
        filename = os.path.basename(image_path)
        image = PIL.Image.open(image_path)
        response = self._model.generate_content([SCORING_PROMPT, image])
        return _parse_response(filename, response.text)


def _parse_response(filename: str, text: str) -> ScoreResult:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        return ScoreResult(
            filename=filename,
            technical=float(data["technical"]),
            aesthetic=float(data["aesthetic"]),
            content=float(data["content"]),
            total=0.0,
            reason=data.get("reason", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return ScoreResult(
            filename=filename,
            technical=5.0,
            aesthetic=5.0,
            content=5.0,
            total=0.0,
            reason="Không thể phân tích ảnh này",
        )
