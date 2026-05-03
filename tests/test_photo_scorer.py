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
    assert ranked[0].filename == "b.jpg"
    assert ranked[1].filename == "d.jpg"
    assert ranked[0].total > 0
