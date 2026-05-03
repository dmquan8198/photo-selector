import pytest
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
        reason="Anh dep",
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
        select_photos.run(["--album", "Da Lat"])

    out = capsys.readouterr().out
    assert "IMG_001.JPG" in out
    assert "9.0" in out
    assert "#1" in out
    assert "Kỹ thuật" in out


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
        select_photos.run(["--album", "Da Lat", "--weights", "50,30,20"])

    called_weights = mock_rank.call_args[0][1]
    assert called_weights["technical"] == 0.50
    assert called_weights["aesthetic"] == 0.30
    assert called_weights["content"] == 0.20


def test_cli_invalid_weights_raises():
    import select_photos
    with pytest.raises(SystemExit):
        select_photos.run(["--album", "Da Lat", "--weights", "60,30,20"])  # sums to 110
