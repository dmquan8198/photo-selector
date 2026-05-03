from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

from photo_loader import PhotoInfo, load_photos_by_album, load_photos_by_days, cleanup
from photo_scorer import ScoreResult, VisionProvider, get_provider, rank_photos


def run(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(Path(__file__).parent / "config.yaml")

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


def _load_config(path) -> dict:
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
