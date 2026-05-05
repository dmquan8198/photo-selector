from __future__ import annotations
import subprocess
import time
from pathlib import Path

import streamlit as st
import yaml

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Photo Selector", page_icon="📸", layout="centered")

CONFIG_PATH = Path(__file__).parent / "config.yaml"

PROVIDER_OPTIONS = {
    "🦙 llama3.2-vision · Local (khuyên dùng)": {"provider": "ollama", "ollama_model": "llama3.2-vision"},
    "🦙 llava · Local":                          {"provider": "ollama", "ollama_model": "llava"},
    "✨ Gemini 2.5 Flash · Cloud":               {"provider": "gemini"},
    "🤖 Claude Haiku · Cloud":                   {"provider": "claude"},
}

PHOTO_TYPE_LABELS = {
    "portrait":      "👤 Portrait",
    "landscape":     "🏔️ Phong cảnh",
    "event_group":   "🎉 Sự kiện / Nhóm",
    "food_object":   "🍜 Đồ ăn / Vật thể",
    "street_candid": "📸 Street / Candid",
    "unknown":       "❓ Không xác định",
}
DIRECTION_LABELS = {
    "technical_leaning": "🔧 Thiên kỹ thuật",
    "emotional_leaning": "❤️ Thiên cảm xúc",
    "balanced":          "⚖️ Cân bằng",
}


@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner="Đang tải danh sách album...")
def list_albums() -> list[str]:
    try:
        import osxphotos
        db = osxphotos.PhotosDB()
        return sorted(a.title for a in db.album_info if a.title)
    except Exception as e:
        st.error(f"Không thể đọc Photos.app: {e}")
        return []


def open_in_photos(uuid: str) -> None:
    """Dùng AppleScript để highlight ảnh trong Photos.app."""
    script = f"""
tell application "Photos"
    activate
    spotlight media item id "{uuid}"
end tell
"""
    subprocess.Popen(["osascript", "-e", script])


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cài đặt")
    config = load_config()

    st.subheader("🤖 AI Model")
    default_provider = config["vision"].get("provider", "ollama")
    default_model    = config["vision"].get("ollama_model", "llama3.2-vision")

    def _default_index() -> int:
        for i, overrides in enumerate(PROVIDER_OPTIONS.values()):
            if overrides["provider"] == default_provider:
                if overrides["provider"] != "ollama":
                    return i
                if overrides.get("ollama_model") == default_model:
                    return i
        return 0

    selected_label = st.selectbox(
        "Chọn model AI",
        list(PROVIDER_OPTIONS.keys()),
        index=_default_index(),
    )
    provider_overrides = PROVIDER_OPTIONS[selected_label]

    if provider_overrides["provider"] == "gemini":
        st.caption("Cần `gemini_api_key` trong config.yaml")
    elif provider_overrides["provider"] == "claude":
        st.caption("Cần `anthropic_api_key` trong config.yaml")

    st.divider()

    st.subheader("Trọng số chấm điểm")
    st.caption("Ba giá trị phải tổng bằng 100")
    w_tech = st.slider("🔧 Kỹ thuật (nét, sáng)", 0, 100, int(config["scoring"]["weights"]["technical"] * 100), step=5)
    w_aest = st.slider("🎨 Thẩm mỹ (bố cục, màu)", 0, 100, int(config["scoring"]["weights"]["aesthetic"] * 100), step=5)
    w_cont = st.slider("👤 Nội dung (chủ thể, mặt)", 0, 100, int(config["scoring"]["weights"]["content"] * 100), step=5)

    total_w = w_tech + w_aest + w_cont
    if total_w != 100:
        st.warning(f"Tổng hiện tại: {total_w}/100 — cần điều chỉnh để tổng = 100")

    top_n = st.number_input("📊 Số ảnh muốn chọn", min_value=1, max_value=10, value=config["scoring"]["top_n"])


# ── main ──────────────────────────────────────────────────────────────────────
st.title("📸 Photo Selector")
st.caption("Chọn album, bấm nút — AI sẽ tìm ra ảnh đẹp nhất cho bạn.")

albums = list_albums()
if not albums:
    st.info("Không tìm thấy album nào trong Photos.app. Hãy chạy app trên MacBook.")
    st.stop()

album = st.selectbox("Chọn album", albums, index=0)
run_btn = st.button("✨ Chọn ảnh đẹp nhất", disabled=(total_w != 100), use_container_width=True)

if total_w != 100:
    st.error(f"Tổng trọng số = {total_w}. Hãy điều chỉnh sidebar để tổng = 100 trước khi chạy.")

# ── scoring ───────────────────────────────────────────────────────────────────
if run_btn and total_w == 100:
    from photo_loader import load_photos_by_album, cleanup
    from photo_scorer import get_provider, rank_photos

    cleanup()

    weights = {
        "technical": w_tech / 100,
        "aesthetic": w_aest / 100,
        "content":   w_cont / 100,
    }

    with st.spinner(f'Đang tải ảnh từ album "{album}"...'):
        try:
            photos, skipped, total_in_album = load_photos_by_album(
                album, thumbnail_size=config["processing"]["thumbnail_size"]
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if not photos:
        st.warning("Album này không có ảnh nào export được.")
        st.stop()

    if skipped > 0:
        st.warning(
            f"⚠️ {skipped}/{total_in_album} ảnh bị bỏ qua (có thể chưa tải về từ iCloud). "
            f"Đang phân tích {len(photos)} ảnh khả dụng."
        )

    vision_config = dict(config["vision"])
    vision_config.update(provider_overrides)
    provider = get_provider(vision_config)

    progress = st.progress(0, text=f"Đang phân tích 0/{len(photos)} ảnh...")
    raw_results = []

    start = time.time()
    for i, photo in enumerate(photos):
        result = provider.score(photo.thumbnail_path)
        raw_results.append(result)
        progress.progress((i + 1) / len(photos), text=f"Đang phân tích {i+1}/{len(photos)} ảnh...")

    ranked = rank_photos(raw_results, weights, top_n=int(top_n))
    elapsed = time.time() - start
    progress.empty()

    # Lưu kết quả vào session_state để giữ khi click nút phụ
    st.session_state["ranked"]         = ranked
    st.session_state["photos"]         = photos
    st.session_state["elapsed"]        = elapsed
    st.session_state["model_label"]    = selected_label.split("·")[0].strip()

# ── hiển thị kết quả ──────────────────────────────────────────────────────────
if "ranked" in st.session_state:
    ranked      = st.session_state["ranked"]
    photos      = st.session_state["photos"]
    elapsed     = st.session_state["elapsed"]
    model_label = st.session_state["model_label"]

    # Xử lý yêu cầu mở ảnh trong Photos (từ lần click trước)
    if "open_uuid" in st.session_state:
        open_in_photos(st.session_state.pop("open_uuid"))

    st.success(f"Phân tích xong {len(photos)} ảnh trong {elapsed:.0f} giây · model: {model_label}")
    st.divider()
    st.subheader(f"🏆 Top {len(ranked)} ảnh đẹp nhất")

    for i, r in enumerate(ranked):
        photo_info = next((p for p in photos if Path(p.thumbnail_path).name == r.filename), None)

        col_img, col_info = st.columns([1, 2])
        with col_img:
            if photo_info and Path(photo_info.thumbnail_path).exists():
                st.image(photo_info.thumbnail_path, use_container_width=True)
            else:
                st.caption("(không tìm thấy ảnh)")

        with col_info:
            type_label = PHOTO_TYPE_LABELS.get(getattr(r, "photo_type", "unknown"), "❓")
            dir_label  = DIRECTION_LABELS.get(getattr(r, "direction", "balanced"), "⚖️ Cân bằng")
            st.markdown(f"**#{i+1} — {r.filename}**")
            st.caption(f"{type_label} · {dir_label}")
            st.markdown(f"### ⭐ {r.total:.2f} / 10")

            # Main scores
            c1, c2, c3 = st.columns(3)
            c1.metric("🔧 Kỹ thuật", f"{r.technical:.2f}")
            c2.metric("🎨 Thẩm mỹ", f"{r.aesthetic:.2f}")
            c3.metric("👤 Nội dung", f"{r.content:.2f}")

            # Sub-scores chi tiết
            with st.expander("📊 Chi tiết điểm"):
                st.markdown("**🔧 Kỹ thuật** *(avg → điểm trên)*")
                s1, s2, s3 = st.columns(3)
                s1.metric("Độ nét", f"{getattr(r, 'sharpness', 0):.2f}")
                s2.metric("Ánh sáng", f"{getattr(r, 'exposure', 0):.2f}")
                s3.metric("Nhiễu", f"{getattr(r, 'noise', 0):.2f}")

                st.markdown("**🎨 Thẩm mỹ** *(avg → điểm trên)*")
                a1, a2, a3 = st.columns(3)
                a1.metric("Bố cục", f"{getattr(r, 'composition', 0):.2f}")
                a2.metric("Màu sắc", f"{getattr(r, 'color_harmony', 0):.2f}")
                a3.metric("Thu hút", f"{getattr(r, 'visual_impact', 0):.2f}")

                st.markdown("**👤 Nội dung** *(avg → điểm trên)*")
                n1, n2, n3 = st.columns(3)
                n1.metric("Chủ thể", f"{getattr(r, 'subject_clarity', 0):.2f}")
                n2.metric("Cảm xúc", f"{getattr(r, 'emotion_story', 0):.2f}")
                n3.metric("Social", f"{getattr(r, 'social_potential', 0):.2f}")

            st.info(f"💬 {r.reason}")

            if photo_info and photo_info.uuid:
                if st.button("📱 Mở trong Photos", key=f"open_{i}", use_container_width=True):
                    st.session_state["open_uuid"] = photo_info.uuid
                    st.rerun()

        st.divider()
