from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import os
import base64


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    filename: str

    # Sub-scores (2 decimal places, 2–10)
    sharpness: float = 5.0       # Technical
    exposure: float = 5.0        # Technical
    noise: float = 5.0           # Technical
    composition: float = 5.0     # Aesthetic
    color_harmony: float = 5.0   # Aesthetic
    visual_impact: float = 5.0   # Aesthetic
    subject_clarity: float = 5.0 # Content
    emotion_story: float = 5.0   # Content
    social_potential: float = 5.0# Content

    # Main scores (avg of sub-scores)
    technical: float = 5.0
    aesthetic: float = 5.0
    content: float = 5.0

    # Weighted total (filled by rank_photos)
    total: float = 0.0

    reason: str = ""
    photo_type: str = "unknown"
    direction: str = "balanced"


def _avg(*vals: float) -> float:
    return round(sum(vals) / len(vals), 2)


def compute_total(result: ScoreResult, weights: dict[str, float]) -> float:
    return round(
        result.technical * weights["technical"]
        + result.aesthetic * weights["aesthetic"]
        + result.content * weights["content"],
        2,
    )


# ── Abstract provider ─────────────────────────────────────────────────────────

class VisionProvider(ABC):
    @abstractmethod
    def score(self, image_path: str) -> ScoreResult:
        """Score one image. Returns ScoreResult (total=0.0, caller fills it)."""


# ── Prompt ────────────────────────────────────────────────────────────────────

# Prompt đầy đủ cho cloud models (Gemini, Claude) — có rubric chi tiết
SCORING_PROMPT = """You are a professional photo analyst. Evaluate this photo for social media posting quality.
Follow all steps in order.

═══ STEP 1 — CLASSIFY photo type (pick exactly one):
• portrait        → person/people as main subject
• landscape       → scenery, nature, architecture, places
• event_group     → events, celebrations, 3+ people
• food_object     → food, drinks, products, still life
• street_candid   → street, candid moments, action

═══ STEP 2 — SCORE 9 sub-criteria independently.
Use anchor scale: 2=failed · 4=poor · 6=average · 8=good · 10=excellent
Half-points (e.g. 7.5) allowed. Each sub-criterion is judged independently.

TECHNICAL sub-criteria:
  sharpness   : Is the main subject tack-sharp? (eyes for portrait, overall for landscape)
  exposure    : Is brightness ideal — no blown highlights, no crushed blacks?
  noise       : Is image clean — no grain, no digital noise in shadows?

AESTHETIC sub-criteria:
  composition : Rule of thirds / symmetry / leading lines / subject placement
  color_harmony: Colors pleasing and harmonious — no clashing palette
  visual_impact: Would this stop a scroll? Thumb-stopping quality for social media?

CONTENT sub-criteria:
  subject_clarity  : Is the main subject obvious and well-presented?
  emotion_story    : Does the image convey feeling, narrative, or a memorable moment?
  social_potential : Would this perform well on social media for its category?

Type-specific notes (apply to all sub-criteria):
  portrait     → sharpness judged primarily on eyes; emotion judged on expression authenticity
  landscape    → sharpness across frame; visual_impact boosted for golden hour light
  event_group  → motion blur heavily penalises sharpness; candid emotion > posed smiles
  food_object  → sharpness on hero item; social_potential = appetite appeal
  street_candid→ intentional blur may be acceptable for sharpness; story > perfection

═══ STEP 3 — DIRECTION (primary strength):
  technical_leaning  → strongest quality is technical execution
  emotional_leaning  → strongest quality is emotion / moment / story
  balanced           → both technical and emotional are strong

═══ STEP 4 — Return ONLY valid JSON, no markdown fences, no extra text:
{
  "photo_type": "<portrait|landscape|event_group|food_object|street_candid>",
  "sharpness": <2–10>,
  "exposure": <2–10>,
  "noise": <2–10>,
  "composition": <2–10>,
  "color_harmony": <2–10>,
  "visual_impact": <2–10>,
  "subject_clarity": <2–10>,
  "emotion_story": <2–10>,
  "social_potential": <2–10>,
  "direction": "<technical_leaning|emotional_leaning|balanced>",
  "reason": "<one sentence in Vietnamese: name the single biggest strength OR weakness>"
}"""


# Prompt rút gọn cho local models (Ollama) — ngắn gọn, dễ follow hơn
SCORING_PROMPT_LITE = """Evaluate this photo. Return ONLY a JSON object with these exact keys and numeric scores from 2 to 10.

Score each item INDEPENDENTLY — do NOT give the same score to everything.
Use the full range: 2=terrible, 4=poor, 6=average, 8=good, 10=perfect. Half-points OK (e.g. 7.5).

{
  "photo_type": "<portrait|landscape|event_group|food_object|street_candid>",
  "sharpness": <is the subject in focus?>,
  "exposure": <is brightness correct, no blown or crushed areas?>,
  "noise": <is the image clean, no grain?>,
  "composition": <is the framing/layout well-balanced?>,
  "color_harmony": <are the colors pleasing?>,
  "visual_impact": <would this stop someone scrolling social media?>,
  "subject_clarity": <is the main subject obvious?>,
  "emotion_story": <does it convey emotion or a story?>,
  "social_potential": <how well would this perform on social media?>,
  "direction": "<technical_leaning|emotional_leaning|balanced>",
  "reason": "<one sentence in Vietnamese about the biggest strength or weakness>"
}

Return ONLY the JSON. No explanation. No markdown."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ef(value, default: float = 5.0) -> float:
    """Extract float, robust to int/float/str/dict."""
    try:
        if isinstance(value, (int, float)):
            return round(float(value), 2)
        if isinstance(value, str):
            return round(float(value), 2)
        if isinstance(value, dict):
            for k in ("score", "value", "rating", "points"):
                if k in value:
                    return round(float(value[k]), 2)
            for v in value.values():
                if isinstance(v, (int, float)):
                    return round(float(v), 2)
    except (ValueError, TypeError):
        pass
    return default


def _parse_response(filename: str, text: str) -> ScoreResult:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        data  = json.loads(cleaned[start:end])

        sharpness        = _ef(data.get("sharpness", 5))
        exposure         = _ef(data.get("exposure", 5))
        noise            = _ef(data.get("noise", 5))
        composition      = _ef(data.get("composition", 5))
        color_harmony    = _ef(data.get("color_harmony", 5))
        visual_impact    = _ef(data.get("visual_impact", 5))
        subject_clarity  = _ef(data.get("subject_clarity", 5))
        emotion_story    = _ef(data.get("emotion_story", 5))
        social_potential = _ef(data.get("social_potential", 5))

        return ScoreResult(
            filename         = filename,
            sharpness        = sharpness,
            exposure         = exposure,
            noise            = noise,
            composition      = composition,
            color_harmony    = color_harmony,
            visual_impact    = visual_impact,
            subject_clarity  = subject_clarity,
            emotion_story    = emotion_story,
            social_potential = social_potential,
            technical        = _avg(sharpness, exposure, noise),
            aesthetic        = _avg(composition, color_harmony, visual_impact),
            content          = _avg(subject_clarity, emotion_story, social_potential),
            total            = 0.0,
            reason           = str(data.get("reason", "")),
            photo_type       = str(data.get("photo_type", "unknown")),
            direction        = str(data.get("direction", "balanced")),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return ScoreResult(filename=filename, reason="Không thể phân tích ảnh này")


# ── Providers ─────────────────────────────────────────────────────────────────

try:
    import google.generativeai as genai
    import PIL.Image
except ImportError:
    genai = None
    PIL   = None


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
        image    = PIL.Image.open(image_path)
        response = self._model.generate_content([SCORING_PROMPT, image])
        return _parse_response(filename, response.text)


try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None


class ClaudeProvider(VisionProvider):
    def __init__(self, api_key: str) -> None:
        self._client = _anthropic.Anthropic(api_key=api_key)

    def score(self, image_path: str) -> ScoreResult:
        filename = os.path.basename(image_path)
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")
        response = self._client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 1024,
            temperature= 0,
            messages   = [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                    {"type": "text",  "text": SCORING_PROMPT},
                ],
            }],
        )
        return _parse_response(filename, response.content[0].text)


class OllamaProvider(VisionProvider):
    def __init__(self, model: str = "llama3.2-vision") -> None:
        self._model    = model
        self._base_url = "http://localhost:11434"

    def score(self, image_path: str) -> ScoreResult:
        import requests
        filename = os.path.basename(image_path)
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        response = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model":   self._model,
                "prompt":  SCORING_PROMPT_LITE,
                "images":  [image_data],
                "stream":  False,
                "options": {"temperature": 0},
            },
            timeout=120,
        )
        response.raise_for_status()
        return _parse_response(filename, response.json()["response"])


# ── Factory & ranking ─────────────────────────────────────────────────────────

def get_provider(vision_config: dict) -> VisionProvider:
    name = vision_config["provider"]
    if name == "gemini":
        return GeminiProvider(api_key=vision_config["gemini_api_key"])
    if name == "claude":
        return ClaudeProvider(api_key=vision_config["anthropic_api_key"])
    if name == "ollama":
        return OllamaProvider(model=vision_config.get("ollama_model", "llama3.2-vision"))
    raise ValueError(f"Unknown provider: {name}")


def rank_photos(
    results: list[ScoreResult],
    weights: dict[str, float],
    top_n: int,
) -> list[ScoreResult]:
    for r in results:
        r.total = compute_total(r, weights)
    return sorted(results, key=lambda r: r.total, reverse=True)[:top_n]
