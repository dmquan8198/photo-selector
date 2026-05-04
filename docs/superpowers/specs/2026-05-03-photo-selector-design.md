# Photo Selector — Design Spec
*Date: 2026-05-03 | Status: Approved*

## Problem

Sau mỗi chuyến đi, người dùng chụp 30–100 ảnh bằng iPhone và mất nhiều thời gian để chọn ảnh đẹp nhất để đăng mạng xã hội. Workflow này tự động hóa bước chọn lọc ảnh bằng AI Vision.

## Scope (Step 1 only)

Spec này chỉ cover **Step 1: chọn lọc ảnh**. Step 2 (chọn nhạc + caption tự động) sẽ là spec riêng.

## Architecture

```
Photos.app (iCloud synced on MacBook)
        ↓
  osxphotos — đọc Photos library, lọc theo album hoặc date range
        ↓
  photo_loader.py — export thumbnail 1200px vào /tmp/photo_selector/
        ↓
  photo_scorer.py — gửi từng thumbnail lên Vision API, nhận điểm 3 tiêu chí
        ↓
  select_photos.py — tổng hợp điểm, xếp hạng, in Top 3–5 ra terminal
        ↓
  Cleanup — xóa /tmp/photo_selector/ sau khi chạy xong
```

## Components

### `config.yaml`
File cấu hình duy nhất — người dùng chỉnh tại đây, không cần sửa code.

```yaml
vision:
  provider: "gemini"          # "gemini" hoặc "claude"
  gemini_api_key: ""          # lấy miễn phí tại aistudio.google.com
  anthropic_api_key: ""       # dùng khi chuyển sang provider "claude"

scoring:
  weights:
    technical: 0.30           # Độ nét, ánh sáng, không blur
    aesthetic: 0.40           # Bố cục, màu sắc, tổng thể
    content: 0.30             # Khuôn mặt, biểu cảm (nếu có người)
  top_n: 5                    # Số ảnh muốn lấy ra

processing:
  thumbnail_size: 1200        # px (cạnh dài), đủ để đánh giá, nhỏ hơn ảnh gốc ~10x
```

### `photo_loader.py`
- Dùng `osxphotos` để truy cập Photos.app library (không cần API key, không cần authentication)
- Filter ảnh theo `--album <tên>` hoặc `--days <số ngày>`
- Export thumbnail ra `/tmp/photo_selector/` — ảnh gốc không bị thay đổi
- Trả về list các file path thumbnail kèm metadata (tên file gốc, thời gian chụp)

### `photo_scorer.py`
- Định nghĩa `VisionProvider` interface với method `score(image_path) -> ScoreResult`
- `GeminiProvider`: gọi Gemini 1.5 Flash API (free tier)
- `ClaudeProvider`: gọi Claude Vision API (Anthropic API key)
- Để đổi provider: chỉ cần thay `provider` trong `config.yaml`
- Prompt yêu cầu model trả về JSON với điểm 1–10 cho từng tiêu chí + lý do ngắn
- Nếu ảnh không có người: tiêu chí "Nội dung" chấm dựa trên chủ thể chính (đồ vật, cảnh vật, động vật) — biểu cảm/khuôn mặt không áp dụng, thay bằng đánh giá sự rõ ràng và hấp dẫn của chủ thể

`ScoreResult` schema:
```python
@dataclass
class ScoreResult:
    filename: str
    technical: float    # 1–10
    aesthetic: float    # 1–10
    content: float      # 1–10
    total: float        # weighted average
    reason: str         # lý do ngắn gọn bằng tiếng Việt
```

### `select_photos.py`
Script chính — entry point duy nhất.

```bash
# Chọn theo album
python select_photos.py --album "Đà Lạt tháng 5"

# Chọn ảnh trong N ngày gần nhất
python select_photos.py --days 3

# Override số ảnh muốn lấy
python select_photos.py --album "Đà Lạt" --top 3

# Override trọng số (technical/aesthetic/content) — tổng phải bằng 100
python select_photos.py --album "Đà Lạt" --weights 40,40,20
```

## Output Format

```
Đang phân tích 47 ảnh...  ████████████████████  100%

🏆 Top 5 ảnh được chọn:

#1  IMG_4821.jpg  —  8.7/10
    Kỹ thuật: 9.0  |  Thẩm mỹ: 9.0  |  Nội dung: 8.0
    → Ánh sáng vàng buổi chiều, bố cục rule-of-thirds, nụ cười tự nhiên

#2  IMG_4835.jpg  —  8.2/10
    Kỹ thuật: 8.0  |  Thẩm mỹ: 9.0  |  Nội dung: 7.0
    → Màu sắc hài hòa, nền đẹp, hơi mất nét ở mắt

#3  IMG_4809.jpg  —  7.9/10
    ...

Thư mục ảnh gốc: ~/Pictures/Photos Library.photoslibrary
Thời gian chạy: 43 giây
```

## Provider Swap Guide

Để chuyển từ Gemini sang Claude API:
1. Tạo API key tại console.anthropic.com
2. Mở `config.yaml`, đổi `provider: "gemini"` → `provider: "claude"`
3. Điền `anthropic_api_key`
4. Chạy lại script — không cần thay đổi gì khác

## Dependencies

```
osxphotos       # đọc Photos.app library (Mac only)
google-generativeai  # Gemini Vision (default provider)
anthropic       # Claude Vision (optional provider)
Pillow          # resize thumbnail
tqdm            # progress bar
PyYAML          # đọc config.yaml
```

## Cost Estimate

| Provider | Free tier | Cost/lần chạy (50 ảnh) |
|---|---|---|
| Gemini 1.5 Flash | 1,500 req/ngày | $0 |
| Claude Vision (Haiku) | Không có | ~$0.10–0.20 |

## Out of Scope (Step 1)

- Chọn nhạc tự động
- Tạo caption tự động
- Đăng lên mạng xã hội
- Giao diện web/GUI
- Tự động chạy khi có ảnh mới
