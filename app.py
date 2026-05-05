from __future__ import annotations
import subprocess
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Photo Selector", page_icon="📸", layout="wide")

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
    "event_group":   "🎉 Sự kiện",
    "food_object":   "🍜 Đồ ăn",
    "street_candid": "📸 Candid",
    "unknown":       "❓",
}
DIRECTION_LABELS = {
    "technical_leaning": "🔧 Kỹ thuật",
    "emotional_leaning": "❤️ Cảm xúc",
    "balanced":          "⚖️ Cân bằng",
}
DIM_COLORS = {
    "🔧 Kỹ thuật": "#4C9BE8",
    "🎨 Thẩm mỹ":  "#F4845F",
    "👤 Nội dung": "#56C596",
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

    selected_label = st.selectbox("Chọn model AI", list(PROVIDER_OPTIONS.keys()), index=_default_index())
    provider_overrides = PROVIDER_OPTIONS[selected_label]
    if provider_overrides["provider"] == "gemini":
        st.caption("Cần `gemini_api_key` trong config.yaml")
    elif provider_overrides["provider"] == "claude":
        st.caption("Cần `anthropic_api_key` trong config.yaml")

    st.divider()

    st.subheader("Trọng số chấm điểm")
    st.caption("Ba giá trị phải tổng bằng 100")
    w_tech = st.slider("🔧 Kỹ thuật", 0, 100, int(config["scoring"]["weights"]["technical"] * 100), step=5)
    w_aest = st.slider("🎨 Thẩm mỹ",  0, 100, int(config["scoring"]["weights"]["aesthetic"] * 100), step=5)
    w_cont = st.slider("👤 Nội dung",  0, 100, int(config["scoring"]["weights"]["content"] * 100),   step=5)

    total_w = w_tech + w_aest + w_cont
    if total_w != 100:
        st.warning(f"Tổng: {total_w}/100")

    top_n = st.number_input("📊 Số ảnh muốn chọn", min_value=1, max_value=10, value=config["scoring"]["top_n"])


# ── main ──────────────────────────────────────────────────────────────────────
st.title("📸 Photo Selector")

albums = list_albums()
if not albums:
    st.info("Không tìm thấy album nào trong Photos.app.")
    st.stop()

col_sel, col_btn = st.columns([3, 1])
with col_sel:
    album = st.selectbox("Chọn album", albums, index=0, label_visibility="collapsed")
with col_btn:
    run_btn = st.button("✨ Phân tích", disabled=(total_w != 100), use_container_width=True)

if total_w != 100:
    st.error(f"Tổng trọng số = {total_w}. Cần = 100.")

# ── scoring ───────────────────────────────────────────────────────────────────
if run_btn and total_w == 100:
    from photo_loader import load_photos_by_album, cleanup
    from photo_scorer import get_provider, rank_photos

    cleanup()
    weights = {"technical": w_tech / 100, "aesthetic": w_aest / 100, "content": w_cont / 100}

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
        st.warning(f"⚠️ {skipped}/{total_in_album} ảnh bị bỏ qua (chưa tải từ iCloud). Phân tích {len(photos)} ảnh.")

    vision_config = dict(config["vision"])
    vision_config.update(provider_overrides)
    provider = get_provider(vision_config)

    progress = st.progress(0, text=f"Đang phân tích 0/{len(photos)} ảnh...")
    raw_results = []
    start = time.time()
    for i, photo in enumerate(photos):
        raw_results.append(provider.score(photo.thumbnail_path))
        progress.progress((i + 1) / len(photos), text=f"Đang phân tích {i+1}/{len(photos)} ảnh...")

    ranked  = rank_photos(raw_results, weights, top_n=int(top_n))
    elapsed = time.time() - start
    progress.empty()

    st.session_state["ranked"]      = ranked
    st.session_state["photos"]      = photos
    st.session_state["elapsed"]     = elapsed
    st.session_state["model_label"] = selected_label.split("·")[0].strip()

# ── kết quả ───────────────────────────────────────────────────────────────────
if "ranked" in st.session_state:
    ranked      = st.session_state["ranked"]
    photos      = st.session_state["photos"]
    elapsed     = st.session_state["elapsed"]
    model_label = st.session_state["model_label"]

    if "open_uuid" in st.session_state:
        open_in_photos(st.session_state.pop("open_uuid"))

    st.success(f"✅ {len(photos)} ảnh · {elapsed:.0f}s · {model_label}")
    st.divider()

    # ── 1. Grid thumbnail + tổng điểm ────────────────────────────────────────
    st.subheader("🏆 Xếp hạng")
    thumb_cols = st.columns(len(ranked))
    for i, (col, r) in enumerate(zip(thumb_cols, ranked)):
        photo_info = next((p for p in photos if Path(p.thumbnail_path).name == r.filename), None)
        with col:
            if photo_info and Path(photo_info.thumbnail_path).exists():
                st.image(photo_info.thumbnail_path, use_container_width=True)
            rank_color = "#FFD700" if i == 0 else "#C0C0C0" if i == 1 else "#CD7F32" if i == 2 else "#888"
            st.markdown(
                f"<div style='text-align:center'>"
                f"<span style='font-size:1.1em;font-weight:bold;color:{rank_color}'>#{i+1}</span> "
                f"<span style='font-size:1.3em;font-weight:bold'>{r.total:.2f}</span><br>"
                f"<small>{PHOTO_TYPE_LABELS.get(getattr(r,'photo_type','unknown'),'❓')} · "
                f"{DIRECTION_LABELS.get(getattr(r,'direction','balanced'),'')}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if photo_info and photo_info.uuid:
                if st.button("📱 Photos", key=f"open_{i}", use_container_width=True):
                    st.session_state["open_uuid"] = photo_info.uuid
                    st.rerun()

    st.divider()

    # ── 2. Bar chart so sánh 3 dimension chính ───────────────────────────────
    st.subheader("📊 So sánh tổng quan")
    rows = []
    for i, r in enumerate(ranked):
        label = f"#{i+1}"
        rows += [
            {"Ảnh": label, "Tiêu chí": "🔧 Kỹ thuật", "Điểm": r.technical},
            {"Ảnh": label, "Tiêu chí": "🎨 Thẩm mỹ",  "Điểm": r.aesthetic},
            {"Ảnh": label, "Tiêu chí": "👤 Nội dung",  "Điểm": r.content},
        ]
    df_main = pd.DataFrame(rows)

    chart_main = (
        alt.Chart(df_main)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Ảnh:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Điểm:Q", scale=alt.Scale(domain=[0, 10]), title="Điểm"),
            color=alt.Color(
                "Tiêu chí:N",
                scale=alt.Scale(domain=list(DIM_COLORS.keys()), range=list(DIM_COLORS.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            xOffset="Tiêu chí:N",
            tooltip=["Ảnh", "Tiêu chí", alt.Tooltip("Điểm:Q", format=".2f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart_main, use_container_width=True)

    # ── 3. Sub-scores chi tiết (expandable) ──────────────────────────────────
    with st.expander("🔍 Chi tiết 9 tiêu chí"):
        SUB_LABELS = {
            "sharpness":        ("🔧", "Độ nét"),
            "exposure":         ("🔧", "Ánh sáng"),
            "noise":            ("🔧", "Nhiễu"),
            "composition":      ("🎨", "Bố cục"),
            "color_harmony":    ("🎨", "Màu sắc"),
            "visual_impact":    ("🎨", "Thu hút"),
            "subject_clarity":  ("👤", "Chủ thể"),
            "emotion_story":    ("👤", "Cảm xúc"),
            "social_potential": ("👤", "Social"),
        }
        sub_rows = []
        for i, r in enumerate(ranked):
            for key, (_, label) in SUB_LABELS.items():
                sub_rows.append({
                    "Ảnh":    f"#{i+1}",
                    "Tiêu chí": label,
                    "Điểm":   getattr(r, key, 5.0),
                })
        df_sub = pd.DataFrame(sub_rows)

        chart_sub = (
            alt.Chart(df_sub)
            .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
            .encode(
                x=alt.X("Ảnh:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Điểm:Q", scale=alt.Scale(domain=[0, 10])),
                color=alt.Color("Ảnh:N", legend=None),
                column=alt.Column(
                    "Tiêu chí:N",
                    title=None,
                    header=alt.Header(labelFontSize=12, labelOrient="bottom"),
                ),
                tooltip=["Ảnh", "Tiêu chí", alt.Tooltip("Điểm:Q", format=".2f")],
            )
            .properties(width=60, height=180)
        )
        st.altair_chart(chart_sub)

    # ── 4. Nhận xét từng ảnh ─────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Nhận xét")
    for i, r in enumerate(ranked):
        rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
        st.markdown(f"**{rank_icon}** — {r.reason}")
