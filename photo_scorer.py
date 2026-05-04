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
        self._model = genai.GenerativeModel("gemini-2.0-flash")

    def score(self, image_path: str) -> ScoreResult:
        import PIL.Image
        filename = os.path.basename(image_path)
        image = PIL.Image.open(image_path)
        response = self._model.generate_content([SCORING_PROMPT, image])
        return _parse_response(filename, response.text)


def _extract_float(value) -> float:
    """Extract float from a value that might be int, float, str, or dict."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    if isinstance(value, dict):
        # llava sometimes returns {"score": 8.5} or {"value": 8.5}
        for key in ("score", "value", "rating", "points"):
            if key in value:
                return float(value[key])
        # fallback: grab first numeric value in dict
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
    return 5.0


def _parse_response(filename: str, text: str) -> ScoreResult:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        data = json.loads(text[start:end])
        return ScoreResult(
            filename=filename,
            technical=_extract_float(data["technical"]),
            aesthetic=_extract_float(data["aesthetic"]),
            content=_extract_float(data["content"]),
            total=0.0,
            reason=str(data.get("reason", "")),
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


try:
    import anthropic
except ImportError:
    anthropic = None
import base64


class ClaudeProvider(VisionProvider):
    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def score(self, image_path: str) -> ScoreResult:
        filename = os.path.basename(image_path)
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        response = self._client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": SCORING_PROMPT},
                ],
            }],
        )
        return _parse_response(filename, response.content[0].text)


class OllamaProvider(VisionProvider):
    def __init__(self, model: str = "llava") -> None:
        self._model = model
        self._base_url = "http://localhost:11434"

    def score(self, image_path: str) -> ScoreResult:
        import requests
        import base64
        filename = os.path.basename(image_path)
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": SCORING_PROMPT,
                "images": [image_data],
                "stream": False,
            },
            timeout=60,
        )
        response.raise_for_status()
        return _parse_response(filename, response.json()["response"])


def get_provider(vision_config: dict) -> VisionProvider:
    provider_name = vision_config["provider"]
    if provider_name == "gemini":
        return GeminiProvider(api_key=vision_config["gemini_api_key"])
    elif provider_name == "claude":
        return ClaudeProvider(api_key=vision_config["anthropic_api_key"])
    elif provider_name == "ollama":
        return OllamaProvider(model=vision_config.get("ollama_model", "llava"))
    raise ValueError(f"Unknown provider: {provider_name}. Must be 'gemini', 'claude', or 'ollama'.")


def rank_photos(
    results: list[ScoreResult],
    weights: dict[str, float],
    top_n: int,
) -> list[ScoreResult]:
    for r in results:
        r.total = compute_total(r, weights)
    sorted_results = sorted(results, key=lambda r: r.total, reverse=True)
    return sorted_results[:top_n]
