import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
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
    mock_album.title = "Da Lat"
    mock_album.photos = [mock_photo]

    with patch("photo_loader.osxphotos.PhotosDB") as mock_db_class, \
         patch("photo_loader._resize_to_thumbnail") as mock_resize:
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.album_info = [mock_album]
        mock_photo.export.return_value = [str(tmp_path / "IMG_001.JPG")]

        results = load_photos_by_album("Da Lat", thumbnail_size=1200, tmp_dir=str(tmp_path))

    assert len(results) == 1
    assert results[0].original_filename == "IMG_001.JPG"
    assert isinstance(results[0].thumbnail_path, str)
    assert isinstance(results[0].date_taken, datetime)


def test_load_by_album_raises_if_not_found():
    with patch("photo_loader.osxphotos.PhotosDB") as mock_db_class:
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.album_info = []

        with pytest.raises(ValueError, match="không tìm thấy"):
            load_photos_by_album("Does Not Exist", thumbnail_size=1200)


def test_load_by_days_filters_by_date(tmp_path):
    recent = _make_mock_photo(uuid="r1", filename="recent.JPG", date=datetime.now() - timedelta(days=1))
    old = _make_mock_photo(uuid="o1", filename="old.JPG", date=datetime.now() - timedelta(days=30))

    with patch("photo_loader.osxphotos.PhotosDB") as mock_db_class, \
         patch("photo_loader._resize_to_thumbnail") as mock_resize:
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db
        mock_db.photos.return_value = [recent, old]
        recent.export.return_value = [str(tmp_path / "recent.JPG")]
        old.export.return_value = [str(tmp_path / "old.JPG")]

        results = load_photos_by_days(days=3, thumbnail_size=1200, tmp_dir=str(tmp_path))

    filenames = [r.original_filename for r in results]
    assert "recent.JPG" in filenames
    assert "old.JPG" not in filenames
