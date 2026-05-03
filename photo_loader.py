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
        if src_path != thumb_path and os.path.exists(src_path):
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
