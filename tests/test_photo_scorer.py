import pytest
from photo_scorer import ScoreResult, compute_total


def test_compute_total_weighted_average():
    result = ScoreResult(
        filename="IMG_001.jpg",
        technical=8.0,
        aesthetic=9.0,
        content=7.0,
        total=0.0,
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


from unittest.mock import MagicMock, patch
from photo_scorer import GeminiProvider

MOCK_GEMINI_RESPONSE = '{"technical": 8.5, "aesthetic": 7.0, "content": 9.0, "reason": "Ánh sáng tốt, khuôn mặt rõ"}'


def test_gemini_provider_parses_response(tmp_path):
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
    assert result.total == 0.0


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

    assert result.technical == 5.0
    assert result.aesthetic == 5.0
    assert result.content == 5.0
