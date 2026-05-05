from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ScoreResult:
    filename: str
    technical: float        # 2–10, bội số 0.5
    aesthetic: float        # 2–10, bội số 0.5
    content: float          # 2–10, bội số 0.5
    total: float            # weighted average, filled by compute_total
    reason: str             # điểm mạnh/yếu chính bằng tiếng Việt
    photo_type: str = "unknown"   # portrait|landscape|event_group|food_object|street_candid
    direction: str = "balanced"   # technical_leaning|emotional_leaning|balanced


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

SCORING_PROMPT = """You are a professional photo analyst. Evaluate this photo for social media posting quality.
Follow all 4 steps in order. Do NOT skip any step.

═══ STEP 1 — CLASSIFY photo type (pick exactly one):
• portrait        → person/people as main subject (headshot, selfie, individual)
• landscape       → scenery, nature, architecture, places
• event_group     → events, celebrations, 3+ people together
• food_object     → food, drinks, products, still life
• street_candid   → street photography, candid moments, action shots

═══ STEP 2 — SCORE each dimension using ANCHOR POINTS below.
Rules: pick the nearest anchor (2 / 4 / 6 / 8 / 10), then add or subtract 0.5 only if truly borderline.
Valid scores: 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0

── TECHNICAL (sharpness · exposure · noise):
  10 → Subject perfectly sharp, exposure ideal (no blown highlights, no crushed blacks), noise-free
   8 → Subject sharp, minor exposure issue OR slight shadow noise — overall clean
   6 → Subject slightly soft OR exposure off ~1 stop, still clearly usable
   4 → Noticeable motion blur OR significantly over/underexposed
   2 → Unrecoverable: extreme blur, black/white frame, completely unusable

  Type-specific adjustments:
  portrait     → eyes MUST be sharp for 8+; blown skin = max 6
  landscape    → horizon line + sky exposure are primary; overall sharpness across frame
  event_group  → faces must be recognizable for 8+; motion blur heavily penalised
  food_object  → hero subject must be sharp; lighting revealing texture critical
  street_candid→ intentional motion blur acceptable; decisive moment outweighs slight softness

── AESTHETIC (composition · color · social media visual appeal):
  10 → Strong composition (rule of thirds / symmetry / leading lines), harmonious palette, thumb-stopping
   8 → Visually appealing; one minor weakness in composition OR color
   6 → Neutral — nothing jarring but nothing draws the eye
   4 → Cluttered OR clashing colors OR awkward crop cutting subject badly
   2 → No clear visual hierarchy; disorienting or repellent

  Type-specific adjustments:
  portrait     → subject placement, background separation, expression framing
  landscape    → foreground interest, golden-hour light = +0.5 bonus, horizon must be level for 8+
  event_group  → natural vs forced arrangement; energy and emotion in frame
  food_object  → styling, angle choice, color contrast with background
  street_candid→ decisive moment, geometry, human element as anchor

── CONTENT (subject · emotion · story · social media potential):
  10 → Compelling subject with strong emotion/story, immediately engaging, high post potential
   8 → Good subject, clear intent, would perform well on social media
   6 → Adequate subject; generic or lacks a hook
   4 → Unclear main subject OR subject is uninteresting for any audience
   2 → No discernible subject or story

  Type-specific adjustments:
  portrait     → expression authenticity, eye contact, emotional connection
  landscape    → sense of place, mood, makes viewer want to be there
  event_group  → peak moment captured, candid emotion > posed smiles
  food_object  → appetite/desire appeal, lifestyle context
  street_candid→ story told in single frame, unexpected or human angle

═══ STEP 3 — DETERMINE direction (the photo's PRIMARY strength):
  technical_leaning  → strongest quality is sharpness / exposure / technical execution
  emotional_leaning  → strongest quality is emotion / moment / story
  balanced           → both technical and emotional qualities are strong

═══ STEP 4 — Return ONLY valid JSON, no extra text, no markdown fences:
{
  "photo_type": "<portrait|landscape|event_group|food_object|street_candid>",
  "technical": <number>,
  "aesthetic": <number>,
  "content": <number>,
  "direction": "<technical_leaning|emotional_leaning|balanced>",
  "reason": "<one sentence in Vietnamese: name the single biggest strength OR weakness>"
}"""


try:
    import google.generativeai as genai
    import PIL.Image
except ImportError:
    genai = None
    PIL = None


class GeminiProvider(VisionProvider):
    def __init__(self, api_key: str) -> None:
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-05-20",
            generation_config=genai.GenerationConfig(temperature=0),
        )

    def score(self, image_path: str) -> ScoreResult:
        import PIL.Image
        filename = os.path.basename(image_path)
        image = PIL.Image.open(image_path)
        response = self._model.generate_content([SCORING_PROMPT, image])
        return _parse_response(filename, response.text)


def _extract_float(value) -> float:
    """Extract float from value that might be int, float, str, or dict."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "value", "rating", "points"):
            if key in value:
                return float(value[key])
        for v in value.values():
            if isinstance(v, (int, float)):
                return float(v)
    return 5.0


def _parse_response(filename: str, text: str) -> ScoreResult:
    try:
        # Strip markdown fences if model wraps in ```json ... ```
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        data = json.loads(cleaned[start:end])
        return ScoreResult(
            filename=filename,
            technical=_extract_float(data["technical"]),
            aesthetic=_extract_float(data["aesthetic"]),
            content=_extract_float(data["content"]),
            total=0.0,
            reason=str(data.get("reason", "")),
            photo_type=str(data.get("photo_type", "unknown")),
            direction=str(data.get("direction", "balanced")),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return ScoreResult(
            filename=filename,
            technical=5.0,
            aesthetic=5.0,
            content=5.0,
            total=0.0,
            reason="Không thể phân tích ảnh này",
            photo_type="unknown",
            direction="balanced",
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
            max_tokens=1024,
            temperature=0,
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
                "options": {"temperature": 0},
            },
            timeout=120,
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
