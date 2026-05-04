# Photo Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI script that reads photos from macOS Photos.app, scores them with Gemini Vision AI, and prints the Top 3–5 best photos with explanations.

**Architecture:** `photo_loader.py` reads the Photos.app library via osxphotos and exports thumbnails to `/tmp/photo_selector/`. `photo_scorer.py` defines a swappable `VisionProvider` interface implemented by `GeminiProvider` (default) and `ClaudeProvider`. `select_photos.py` is the CLI entry point that wires everything together and prints ranked results.

**Tech Stack:** Python 3.11+, osxphotos, google-generativeai, anthropic, Pillow, tqdm, PyYAML, pytest

---

## File Map

```
Analyst_Image/
├── config.yaml                  # user-facing config (provider, weights, top_n)
├── requirements.txt             # all dependencies
├── select_photos.py             # CLI entry point
├── photo_loader.py              # osxphotos → thumbnail export → PhotoInfo list
├── photo_scorer.py              # ScoreResult, VisionProvider, GeminiProvider, ClaudeProvider, rank_photos
└── tests/
    ├── test_photo_scorer.py     # unit tests for scoring logic and providers (mocked API)
    ├── test_photo_loader.py     # unit tests for loader (mocked osxphotos)
    └── test_select_photos.py    # integration test for CLI (mocked everything)
```

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `config.yaml`

- [ ] **Step 1: Create `requirements.txt`**

```
osxphotos
google-generativeai
anthropic
Pillow
tqdm
PyYAML
pytest
pytest-mock
```

- [ ] **Step 2: Create `config.yaml`**

```yaml
vision:
  provider: "gemini"
  gemini_api_key: ""        # lấy miễn phí tại aistudio.google.com
  anthropic_api_key: ""     # dùng khi chuyển sang provider "claude"

scoring:
  weights:
    technical: 0.30
    aesthetic: 0.40
    content: 0.30
  top_n: 5

processing:
  thumbnail_size: 1200
```

- [ ] **Step 3: Install dependencies**

Run trên MacBook (dự án chạy trên Mac):
```bash
pip install -r requirements.txt
```

Expected: tất cả packages cài thành công. Nếu lỗi `osxphotos` không tìm thấy Photos.app, đảm bảo chạy trên macOS (không phải VM/Windows).

- [ ] **Step 4: Commit**

```bash
git init
git add requirements.txt config.yaml
git commit -m "chore: project setup"
```

---

## Task 2: ScoreResult Dataclass + VisionProvider Interface

**Files:**
- Create: `photo_scorer.py`
- Create: `tests/test_photo_scorer.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_photo_scorer.py
from photo_scorer import ScoreResult, compute_total

def test_compute_total_weighted_average():
    result = ScoreResult(
        filename="IMG_001.jpg",
        technical=8.0,
        aesthetic=9.0,
        content=7.0,
        total=0.0,  # will be computed
        reason="test"
    )
    weights = {"technical": 0.30, "aesthetic": 0.40, "content": 0.30}
    total = compute_total(result, weights)
    # 8.0*0.30 + 9.0*0.40 + 7.0*0.30 = 2.4 + 3.6 + 2.1 = 8.1
    assert round(total, 2) == 8.1

def test_compute_total_equal_weights():
    result = ScoreResult(
        filename="IMG_002.jpg",
        technical=6.0,
        aesthetic=6.0,
        content=6.0,
        total=0.0,
        reason="test"
    )
    weights = {"technical": 0.33, "aesthetic": 0.34, "content": 0.33}
    total = compute_total(result, weights)
    assert round(total, 1) == 6.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_photo_scorer.py -v
```

Expected: `ImportError: cannot import name 'ScoreResult' from 'photo_scorer'`

- [ ] **Step 3: Implement `ScoreResult` and `compute_total` in `photo_scorer.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_photo_scorer.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add photo_scorer.py tests/test_photo_scorer.py
git commit -m "feat: ScoreResult dataclass and VisionProvider interface"
```

---

## Task 3: GeminiProvider

**Files:**
- Modify: `photo_scorer.py`
- Modify: `tests/test_photo_scorer.py`

- [ ] **Step 1: Write failing test**

Thêm vào `tests/test_photo_scorer.py`:

```python
from unittest.mock import MagicMock, patch
from photo_scorer import GeminiProvider

MOCK_GEMINI_RESPONSE = '{"technical": 8.5, "aesthetic": 7.0, "content": 9.0, "reason": "Ánh sáng tốt, khuôn mặt rõ"}'

def test_gemini_provider_parses_response(tmp_path):
    # tạo ảnh JPEG giả để test
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(img_path)

    with patch("photo_scorer.genai") as mock_genai:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_response = MagicMock()
        mock_response.text = MOCK_GEMINI_RESPONSE
        mock_model.generate_content.return_value = mock_response

        provider = GeminiProvider(api_key="fake-key")
        result = provider.score(str(img_path))

    assert result.filename == "test.jpg"
    assert result.technical == 8.5
    assert result.aesthetic == 7.0
    assert result.content == 9.0
    assert result.reason == "Ánh sáng tốt, khuôn mặt rõ"
    assert result.total == 0.0  # caller fills this

def test_gemini_provider_handles_invalid_json(tmp_path):
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100)).save(img_path)

    with patch("photo_scorer.genai") as mock_genai:
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_response = MagicMock()
        mock_response.text = "Xin lỗi, tôi không thể đánh giá ảnh này."
        mock_model.generate_content.return_value = mock_response

        provider = GeminiProvider(api_key="fake-key")
        result = provider.score(str(img_path))

    # fallback khi JSON parse lỗi: điểm trung bình 5.0
    assert result.technical == 5.0
    assert result.aesthetic == 5.0
    assert result.content == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_photo_scorer.py::test_gemini_provider_parses_response -v
```

Expected: `ImportError: cannot import name 'GeminiProvider'`

- [ ] **Step 3: Implement `GeminiProvider` trong `photo_scorer.py`**

Thêm vào cuối file (sau class `VisionProvider`):

```python
import json
import os
import google.generativeai as genai
import PIL.Image

SCORING_PROMPT = """Evaluate this photo and return ONLY a JSON object with these exact keys:
- "technical": score 1-10 for sharpness, exposure, focus, no blur/noise
- "aesthetic": score 1-10 for composition, color harmony, visual balance
- "content": score 1-10 for subject clarity and appeal (faces/expressions if people present, otherwise main subject quality)
- "reason": one short sentence in Vietnamese explaining the top strength or weakness

Return ONLY valid JSON, no extra text. Example:
{"technical": 8.5, "aesthetic": 7.0, "content": 9.0, "reason": "Ánh sáng tốt, khuôn mặt rõ nét"}"""


class GeminiProvider(VisionProvider):
    def __init__(self, api_key: str) -> None:
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel("gemini-1.5-flash")

    def score(self, image_path: str) -> ScoreResult:
        filename = os.path.basename(image_path)
        image = PIL.Image.open(image_path)
        response = self._model.generate_content([SCORING_PROMPT, image])
        return _parse_response(filename, response.text)


def _parse_response(filename: str, text: str) -> ScoreResult:
    try:
        # extract JSON even if model adds surrounding text
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_photo_scorer.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add photo_scorer.py tests/test_photo_scorer.py
git commit -m "feat: GeminiProvider with JSON response parsing"
```

---

## Task 4: ClaudeProvider

**Files:**
- Modify: `photo_scorer.py`
- Modify: `tests/test_photo_scorer.py`

- [ ] **Step 1: Write failing test**

Thêm vào `tests/test_photo_scorer.py`:

```python
from photo_scorer import ClaudeProvider

def test_claude_provider_parses_response(tmp_path):
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 100), color=(200, 150, 100)).save(img_path)

    with patch("photo_scorer.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text='{"technical": 7.0, "aesthetic": 8.5, "content": 6.0, "reason": "Màu sắc đẹp nhưng hơi mờ"}')]
        mock_client.messages.create.return_value = mock_message

        provider = ClaudeProvider(api_key="fake-key")
        result = provider.score(str(img_path))

    assert result.filename == "test.jpg"
    assert result.technical == 7.0
    assert result.aesthetic == 8.5
    assert result.content == 6.0
    assert result.reason == "Màu sắc đẹp nhưng hơi mờ"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_photo_scorer.py::test_claude_provider_parses_response -v
```

Expected: `ImportError: cannot import name 'ClaudeProvider'`

- [ ] **Step 3: Implement `ClaudeProvider` trong `photo_scorer.py`**

Thêm vào cuối file, sau `GeminiProvider`:

```python
import anthropic
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
```

- [ ] **Step 4: Run all tests to verify they pass**

```bash
pytest tests/test_photo_scorer.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add photo_scorer.py tests/test_photo_scorer.py
git commit -m "feat: ClaudeProvider for swappable vision backend"
```

---

## Task 5: `get_provider` Factory + `rank_photos`

**Files:**
- Modify: `photo_scorer.py`
- Modify: `tests/test_photo_scorer.py`

- [ ] **Step 1: Write failing tests**

Thêm vào `tests/test_photo_scorer.py`:

```python
from photo_scorer import get_provider, rank_photos

def test_get_provider_returns_gemini():
    with patch("photo_scorer.genai"):
        provider = get_provider({"provider": "gemini", "gemini_api_key": "key", "anthropic_api_key": ""})
    assert isinstance(provider, GeminiProvider)

def test_get_provider_returns_claude():
    with patch("photo_scorer.anthropic"):
        provider = get_provider({"provider": "claude", "gemini_api_key": "", "anthropic_api_key": "key"})
    assert isinstance(provider, ClaudeProvider)

def test_get_provider_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider({"provider": "openai", "gemini_api_key": "", "anthropic_api_key": ""})

def test_rank_photos_returns_top_n_sorted():
    results = [
        ScoreResult("a.jpg", 7.0, 6.0, 8.0, 0.0, "ok"),
        ScoreResult("b.jpg", 9.0, 9.0, 9.0, 0.0, "great"),
        ScoreResult("c.jpg", 5.0, 5.0, 5.0, 0.0, "poor"),
        ScoreResult("d.jpg", 8.0, 8.0, 7.0, 0.0, "good"),
    ]
    weights = {"technical": 0.30, "aesthetic": 0.40, "content": 0.30}
    ranked = rank_photos(results, weights, top_n=2)
    assert len(ranked) == 2
    assert ranked[0].filename == "b.jpg"   # highest score
    assert ranked[1].filename == "d.jpg"   # second highest
    # totals are filled in
    assert ranked[0].total > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_photo_scorer.py::test_rank_photos_returns_top_n_sorted -v
```

Expected: `ImportError: cannot import name 'rank_photos'`

- [ ] **Step 3: Implement `get_provider` và `rank_photos` trong `photo_scorer.py`**

Thêm vào cuối file:

```python
def get_provider(vision_config: dict) -> VisionProvider:
    provider_name = vision_config["provider"]
    if provider_name == "gemini":
        return GeminiProvider(api_key=vision_config["gemini_api_key"])
    elif provider_name == "claude":
        return ClaudeProvider(api_key=vision_config["anthropic_api_key"])
    raise ValueError(f"Unknown provider: {provider_name}. Must be 'gemini' or 'claude'.")


def rank_photos(
    results: list[ScoreResult],
    weights: dict[str, float],
    top_n: int,
) -> list[ScoreResult]:
    for r in results:
        r.total = compute_total(r, weights)
    results.sort(key=lambda r: r.total, reverse=True)
    return results[:top_n]
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_photo_scorer.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add photo_scorer.py tests/test_photo_scorer.py
git commit -m "feat: get_provider factory and rank_photos aggregation"
```

---

## Task 6: `photo_loader.py`

**Files:**
- Create: `photo_loader.py`
- Create: `tests/test_photo_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_photo_loader.py
from unittest.mock import MagicMock, patch, call
from datetime import datetime
from photo_loader import load_photos_by_album, load_photos_by_days, PhotoInfo

def _make_mock_photo(uuid="abc123", filename="IMG_001.JPG", date=None):
    photo = MagicMock()
    photo.uuid = uuid
    photo.original_filename = filename
    photo.date = date or datetime(2026, 5, 1, 12, 0, 0)
    photo.export.return_value = [f"/tmp/photo_selector/{filename}"]
    return photo

def test_load_by_album_returns_photo_infos(tmp_path):
    mock_photo = _make_mock_photo()
    mock_album = MagicMock()
    mock_album.title = "Đà Lạt"
    mock_album.photos = [mock_photo]

    with patch("photo_loader.osxphotos.PhotosDB") as mock_db_class, \
         patch("photo_loader.PIL.Image.open") as mock_open:
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.album_info = [mock_album]

        mock_img = MagicMock()
        mock_open.return_value.__enter__ = lambda s: mock_img
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_img.size = (3000, 4000)

        results = load_photos_by_album("Đà Lạt", thumbnail_size=1200, tmp_dir=str(tmp_path))

    assert len(results) == 1
    assert results[0].original_filename == "IMG_001.JPG"
    assert isinstance(results[0].thumbnail_path, str)

def test_load_by_album_raises_if_not_found():
    with patch("photo_loader.osxphotos.PhotosDB") as mock_db_class:
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.album_info = []
        import pytest
        with pytest.raises(ValueError, match="không tìm thấy"):
            load_photos_by_album("Không tồn tại", thumbnail_size=1200)

def test_load_by_days_filters_by_date(tmp_path):
    from datetime import timedelta
    recent = _make_mock_photo(uuid="r1", filename="recent.JPG", date=datetime.now() - timedelta(days=1))
    old = _make_mock_photo(uuid="o1", filename="old.JPG", date=datetime.now() - timedelta(days=30))

    with patch("photo_loader.osxphotos.PhotosDB") as mock_db_class, \
         patch("photo_loader.PIL.Image.open") as mock_open:
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.photos.return_value = [recent, old]

        mock_img = MagicMock()
        mock_open.return_value.__enter__ = lambda s: mock_img
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_img.size = (3000, 4000)

        results = load_photos_by_days(days=3, thumbnail_size=1200, tmp_dir=str(tmp_path))

    # only recent photo (within 3 days of 2026-05-03) should be returned
    filenames = [r.original_filename for r in results]
    assert "recent.JPG" in filenames
    assert "old.JPG" not in filenames
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_photo_loader.py -v
```

Expected: `ModuleNotFoundError: No module named 'photo_loader'`

- [ ] **Step 3: Implement `photo_loader.py`**

```python
from __future__ import annotations
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta

import osxphotos
import PIL.Image

TMP_DIR = "/tmp/photo_selector"


@dataclass
class PhotoInfo:
    original_filename: str
    thumbnail_path: str
    date_taken: datetime


def load_photos_by_album(
    album_name: str,
    thumbnail_size: int,
    tmp_dir: str = TMP_DIR,
) -> list[PhotoInfo]:
    db = osxphotos.PhotosDB()
    album = next((a for a in db.album_info if a.title == album_name), None)
    if album is None:
        raise ValueError(f"Album '{album_name}' không tìm thấy trong Photos.app")
    return _export_thumbnails(album.photos, thumbnail_size, tmp_dir)


def load_photos_by_days(
    days: int,
    thumbnail_size: int,
    tmp_dir: str = TMP_DIR,
) -> list[PhotoInfo]:
    db = osxphotos.PhotosDB()
    cutoff = datetime.now() - timedelta(days=days)
    photos = [p for p in db.photos() if p.date >= cutoff]
    return _export_thumbnails(photos, thumbnail_size, tmp_dir)


def _export_thumbnails(
    photos: list,
    thumbnail_size: int,
    tmp_dir: str,
) -> list[PhotoInfo]:
    os.makedirs(tmp_dir, exist_ok=True)
    results = []
    for photo in photos:
        exported = photo.export(tmp_dir, overwrite=True)
        if not exported:
            continue
        src_path = exported[0]
        thumb_path = os.path.join(tmp_dir, f"thumb_{photo.uuid}.jpg")
        _resize_to_thumbnail(src_path, thumb_path, thumbnail_size)
        if src_path != thumb_path:
            os.remove(src_path)
        results.append(PhotoInfo(
            original_filename=photo.original_filename,
            thumbnail_path=thumb_path,
            date_taken=photo.date,
        ))
    return results


def _resize_to_thumbnail(src: str, dst: str, max_size: int) -> None:
    with PIL.Image.open(src) as img:
        img.thumbnail((max_size, max_size), PIL.Image.LANCZOS)
        img.convert("RGB").save(dst, "JPEG", quality=85)


def cleanup(tmp_dir: str = TMP_DIR) -> None:
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_photo_loader.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add photo_loader.py tests/test_photo_loader.py
git commit -m "feat: photo_loader with osxphotos and thumbnail export"
```

---

## Task 7: `select_photos.py` CLI Entry Point

**Files:**
- Create: `select_photos.py`
- Create: `tests/test_select_photos.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_select_photos.py
from unittest.mock import MagicMock, patch
from datetime import datetime

from photo_loader import PhotoInfo
from photo_scorer import ScoreResult


def _make_photo_info(filename="IMG_001.JPG"):
    return PhotoInfo(
        original_filename=filename,
        thumbnail_path=f"/tmp/photo_selector/thumb_{filename}",
        date_taken=datetime(2026, 5, 1),
    )


def _make_score_result(filename="IMG_001.JPG", total=8.0):
    return ScoreResult(
        filename=filename,
        technical=8.0,
        aesthetic=8.0,
        content=8.0,
        total=total,
        reason="Ảnh đẹp",
    )


def test_cli_album_mode_prints_results(capsys):
    photo_infos = [_make_photo_info("IMG_001.JPG"), _make_photo_info("IMG_002.JPG")]
    ranked = [
        _make_score_result("IMG_001.JPG", total=9.0),
        _make_score_result("IMG_002.JPG", total=7.5),
    ]

    with patch("select_photos.load_photos_by_album", return_value=photo_infos), \
         patch("select_photos.get_provider") as mock_provider_factory, \
         patch("select_photos.rank_photos", return_value=ranked), \
         patch("select_photos.cleanup"):
        mock_provider = MagicMock()
        mock_provider_factory.return_value = mock_provider
        mock_provider.score.side_effect = [
            _make_score_result("IMG_001.JPG"),
            _make_score_result("IMG_002.JPG"),
        ]

        import select_photos
        select_photos.run(["--album", "Đà Lạt"])

    out = capsys.readouterr().out
    assert "IMG_001.JPG" in out
    assert "9.0" in out
    assert "#1" in out


def test_cli_weights_override(capsys):
    photo_infos = [_make_photo_info()]
    ranked = [_make_score_result(total=8.0)]

    with patch("select_photos.load_photos_by_album", return_value=photo_infos), \
         patch("select_photos.get_provider") as mock_provider_factory, \
         patch("select_photos.rank_photos", return_value=ranked) as mock_rank, \
         patch("select_photos.cleanup"):
        mock_provider_factory.return_value = MagicMock()
        mock_provider_factory.return_value.score.return_value = _make_score_result()

        import select_photos
        select_photos.run(["--album", "Đà Lạt", "--weights", "50,30,20"])

    called_weights = mock_rank.call_args[0][1]
    assert called_weights["technical"] == 0.50
    assert called_weights["aesthetic"] == 0.30
    assert called_weights["content"] == 0.20


def test_cli_invalid_weights_raises(capsys):
    import pytest
    import select_photos
    with pytest.raises(SystemExit):
        select_photos.run(["--album", "Đà Lạt", "--weights", "60,30,20"])  # sums to 110
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_select_photos.py -v
```

Expected: `ModuleNotFoundError: No module named 'select_photos'`

- [ ] **Step 3: Implement `select_photos.py`**

```python
from __future__ import annotations
import argparse
import sys
import time

import yaml
from tqdm import tqdm

from photo_loader import PhotoInfo, load_photos_by_album, load_photos_by_days, cleanup
from photo_scorer import ScoreResult, VisionProvider, get_provider, rank_photos


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config("config.yaml")

    weights = _resolve_weights(args, config)
    top_n = args.top or config["scoring"]["top_n"]
    thumbnail_size = config["processing"]["thumbnail_size"]

    photos = _load_photos(args, thumbnail_size)
    if not photos:
        print("Không tìm thấy ảnh nào.")
        return

    provider = get_provider(config["vision"])
    print(f"\nĐang phân tích {len(photos)} ảnh...")

    start = time.time()
    raw_results: list[ScoreResult] = []
    for photo in tqdm(photos, ncols=60):
        result = provider.score(photo.thumbnail_path)
        raw_results.append(result)

    ranked = rank_photos(raw_results, weights, top_n)
    elapsed = time.time() - start

    _print_results(ranked, elapsed)
    cleanup()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chọn ảnh đẹp nhất từ Photos.app")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--album", type=str, help="Tên album trong Photos.app")
    group.add_argument("--days", type=int, help="Ảnh trong N ngày gần nhất")
    parser.add_argument("--top", type=int, help="Số ảnh muốn lấy (mặc định theo config)")
    parser.add_argument(
        "--weights",
        type=str,
        help="Trọng số technical/aesthetic/content, tổng=100. Ví dụ: 40,40,20",
    )
    return parser.parse_args(argv)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_weights(args: argparse.Namespace, config: dict) -> dict[str, float]:
    if not args.weights:
        return config["scoring"]["weights"]
    parts = args.weights.split(",")
    if len(parts) != 3:
        print("Lỗi: --weights phải có đúng 3 giá trị. Ví dụ: 40,40,20", file=sys.stderr)
        sys.exit(1)
    values = [int(p) for p in parts]
    if sum(values) != 100:
        print(f"Lỗi: --weights phải tổng bằng 100 (hiện tại: {sum(values)})", file=sys.stderr)
        sys.exit(1)
    return {
        "technical": values[0] / 100,
        "aesthetic": values[1] / 100,
        "content": values[2] / 100,
    }


def _load_photos(args: argparse.Namespace, thumbnail_size: int) -> list[PhotoInfo]:
    if args.album:
        return load_photos_by_album(args.album, thumbnail_size)
    return load_photos_by_days(args.days, thumbnail_size)


def _print_results(ranked: list[ScoreResult], elapsed: float) -> None:
    print(f"\n🏆 Top {len(ranked)} ảnh được chọn:\n")
    for i, r in enumerate(ranked, 1):
        print(f"#{i}  {r.filename}  —  {r.total:.1f}/10")
        print(f"    Kỹ thuật: {r.technical:.1f}  |  Thẩm mỹ: {r.aesthetic:.1f}  |  Nội dung: {r.content:.1f}")
        print(f"    → {r.reason}\n")
    print(f"Thời gian chạy: {elapsed:.0f} giây")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```

Expected: tất cả passed (12 tests)

- [ ] **Step 5: Commit**

```bash
git add select_photos.py tests/test_select_photos.py
git commit -m "feat: CLI entry point with album/days mode and weight override"
```

---

## Task 8: Smoke Test End-to-End

**Files:** không có file mới — test thủ công trên MacBook

- [ ] **Step 1: Điền Gemini API key vào `config.yaml`**

Truy cập aistudio.google.com → Get API Key → Copy key → paste vào `config.yaml`:
```yaml
vision:
  gemini_api_key: "AIza..."
```

- [ ] **Step 2: Chạy thử với album thực**

```bash
python select_photos.py --album "Tên album thực trong Photos.app" --top 3
```

Expected output:
```
Đang phân tích X ảnh...  ████  100%

🏆 Top 3 ảnh được chọn:

#1  IMG_XXXX.jpg  —  8.x/10
    Kỹ thuật: ...  |  Thẩm mỹ: ...  |  Nội dung: ...
    → [lý do bằng tiếng Việt]
...
Thời gian chạy: XX giây
```

- [ ] **Step 3: Verify ảnh gốc không bị thay đổi**

Mở Photos.app → album vừa chạy → kiểm tra ảnh vẫn nguyên vẹn.

- [ ] **Step 4: Verify cleanup**

```bash
ls /tmp/photo_selector
```

Expected: `No such file or directory` (đã bị xóa sau khi chạy)

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "chore: project complete - photo selector v1"
```

---

## Provider Swap Verification (Optional)

Khi muốn đổi sang Claude API:

```bash
# 1. Lấy API key tại console.anthropic.com
# 2. Cập nhật config.yaml:
#    provider: "claude"
#    anthropic_api_key: "sk-ant-..."
# 3. Chạy lại — không cần thay đổi gì khác
python select_photos.py --album "Test" --top 3
```
