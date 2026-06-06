import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.pipeline import DatePipeline


@st.cache_resource
def load_pipeline():
    return DatePipeline()


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 0.5rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #48C6EF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #6B7280;
    font-size: 1.05rem;
}

/* ── Feature cards ── */
.features {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.8rem 0;
}
.feat {
    background: #1A1D26;
    border: 1px solid #2A2D3A;
    border-radius: 14px;
    padding: 1.3rem;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.feat:hover { border-color: #6C63FF55; transform: translateY(-2px); }
.feat-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.feat-title { font-weight: 600; color: #E5E7EB; font-size: 0.95rem; margin-bottom: 0.25rem; }
.feat-desc { color: #6B7280; font-size: 0.8rem; line-height: 1.4; }

/* ── Pipeline ── */
.pipeline {
    display: flex;
    justify-content: center;
    gap: 0.3rem;
    align-items: center;
    padding: 0.6rem 0;
    margin-bottom: 0.5rem;
}
.pipe-step {
    background: #1A1D26;
    border: 1px solid #2A2D3A;
    border-radius: 8px;
    padding: 0.35rem 0.8rem;
    font-size: 0.75rem;
    color: #6B7280;
    font-weight: 500;
}
.pipe-step.active { border-color: #6C63FF; color: #6C63FF; background: #6C63FF11; }
.pipe-step.done { border-color: #10B981; color: #10B981; background: #10B98111; }
.pipe-arrow { color: #374151; font-size: 0.85rem; }

/* ── Result panel ── */
.result-card {
    background: #1A1D26;
    border: 1px solid #2A2D3A;
    border-radius: 16px;
    padding: 1.8rem;
}
.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}
.result-date {
    font-size: 2.2rem;
    font-weight: 700;
    color: #F9FAFB;
    letter-spacing: 0.5px;
}
.status-pill {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 0.85rem;
    margin-top: 0.8rem;
}
.s-valid   { background: #065F4622; color: #34D399; border: 1px solid #34D39933; }
.s-warning { background: #78350F22; color: #FBBF24; border: 1px solid #FBBF2433; }
.s-expired { background: #7F1D1D22; color: #F87171; border: 1px solid #F8717133; }

/* ── Detection table ── */
.det-card {
    background: #1A1D26;
    border: 1px solid #2A2D3A;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.det-table { width: 100%; border-collapse: collapse; }
.det-table th {
    color: #6B7280; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.8px;
    padding: 0.5rem 0.6rem; text-align: left;
    border-bottom: 1px solid #2A2D3A;
}
.det-table td {
    padding: 0.55rem 0.6rem;
    border-bottom: 1px solid #1E2130;
    color: #D1D5DB; font-size: 0.85rem;
}
.det-table tr:last-child td { border-bottom: none; }
.conf-bar { height: 5px; border-radius: 3px; background: #2A2D3A; margin-top: 3px; }
.conf-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #6C63FF, #48C6EF); }

/* ── Upload styling ── */
.upload-wrapper {
    max-width: 500px;
    margin: 0 auto 1.5rem;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #374151 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6C63FF66 !important;
}

/* ── Hide defaults ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
"""


def draw_boxes(image, detections):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")

    scale = max(1, image.width // 800)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, 18 * scale))
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        x, y, x1, y1 = det.bbox
        conf = det.confidence

        draw.rectangle([x, y, x1, y1], fill=(108, 99, 255, 25))
        draw.rectangle([x, y, x1, y1], outline=(108, 99, 255, 240), width=max(2, 2 * scale))

        label = f" {conf:.0%} "
        bb = font.getbbox(label)
        lw, lh = bb[2] - bb[0], bb[3] - bb[1]
        ly = y - lh - 8 if y > lh + 8 else y1 + 2
        draw.rectangle([x, ly, x + lw + 6, ly + lh + 6], fill=(108, 99, 255, 210))
        draw.text((x + 3, ly + 2), label, fill="white", font=font)

    return annotated


def render_pipeline_html(step=0):
    names = ["Upload", "Detect", "OCR", "Parse", "Done"]
    parts = []
    for i, n in enumerate(names):
        if i < step:
            cls = "pipe-step done"
        elif i == step:
            cls = "pipe-step active"
        else:
            cls = "pipe-step"
        parts.append(f'<span class="{cls}">{n}</span>')
        if i < len(names) - 1:
            parts.append('<span class="pipe-arrow">→</span>')
    return f'<div class="pipeline">{"".join(parts)}</div>'


def main():
    st.set_page_config(
        page_title="DateReg — Expiry Date Recognition",
        page_icon="📦",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    pipeline = load_pipeline()

    # ── State ──
    if "result" not in st.session_state:
        st.session_state.result = None

    # ── Header (always) ──
    st.markdown("""
    <div class="hero">
        <h1>DateReg</h1>
        <p>Nhận diện hạn sử dụng sản phẩm bằng AI</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Upload ──
    st.markdown('<div class="upload-wrapper">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════
    # STATE 1: No image → show landing page
    # ══════════════════════════════════════════
    if uploaded_file is None:
        st.session_state.result = None
        st.markdown("""
        <div class="features">
            <div class="feat">
                <div class="feat-icon">🔍</div>
                <div class="feat-title">YOLOv8 Detection</div>
                <div class="feat-desc">Tự động phát hiện vùng chứa ngày trên bao bì sản phẩm</div>
            </div>
            <div class="feat">
                <div class="feat-icon">🔤</div>
                <div class="feat-title">CTC-OCR</div>
                <div class="feat-desc">Nhận diện ký tự từ vùng date đã crop bằng mô hình CTC</div>
            </div>
            <div class="feat">
                <div class="feat-icon">📊</div>
                <div class="feat-title">Smart Analysis</div>
                <div class="feat-desc">Phân tích, so sánh và đánh giá trạng thái hạn sử dụng</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(render_pipeline_html(0), unsafe_allow_html=True)
        return

    # ══════════════════════════════════════════
    # STATE 2: Image uploaded
    # ══════════════════════════════════════════
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_result = st.columns([3, 2], gap="large")

    with col_img:
        btn = st.button("🚀 Nhận diện", type="primary", use_container_width=True)

        if btn:
            # ── Run pipeline ──
            pipeline_ph = st.empty()
            progress = st.progress(0, text="Đang xử lý ảnh...")

            pipeline_ph.markdown(render_pipeline_html(2), unsafe_allow_html=True)
            result = pipeline.run(image)
            progress.progress(100, text="Hoàn tất!")

            pipeline_ph.markdown(render_pipeline_html(4), unsafe_allow_html=True)

            st.session_state.result = result

            annotated = draw_boxes(image, result.detections)
            st.image(annotated, use_container_width=True)
        else:
            r = st.session_state.result
            if r and r.detections:
                st.markdown(render_pipeline_html(4), unsafe_allow_html=True)
                annotated = draw_boxes(image, r.detections)
                st.image(annotated, use_container_width=True)
            else:
                st.markdown(render_pipeline_html(0), unsafe_allow_html=True)
                st.image(image, use_container_width=True)

    with col_result:
        r = st.session_state.result
        if r and r.date:
            result_date = r.date
            detections = r.detections
            status, delta = r.status, r.days_remaining
            if status == "valid":
                pill = f'<div class="status-pill s-valid">✓ Còn hạn — còn {delta} ngày</div>'
            elif status == "warning":
                pill = f'<div class="status-pill s-warning">⚠ Sắp hết hạn — còn {delta} ngày</div>'
            elif status == "expired":
                pill = f'<div class="status-pill s-expired">✕ Đã hết hạn — quá {abs(delta)} ngày</div>'
            else:
                pill = '<div class="status-pill" style="color:#6B7280;border:1px solid #374151;">Không xác định</div>'

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Ngày hết hạn</div>
                <div class="result-date">{result_date}</div>
                {pill}
            </div>
            """, unsafe_allow_html=True)

            if detections:
                rows = ""
                for i, det in enumerate(detections):
                    c = det.confidence
                    txt = det.text or "—"
                    rows += f"""<tr>
                        <td style="color:#6B7280;">#{i+1}</td>
                        <td><code style="background:#6C63FF18;padding:2px 6px;border-radius:4px;color:#A5B4FC;">{txt}</code></td>
                        <td>{c:.1%}<div class="conf-bar"><div class="conf-fill" style="width:{c*100:.0f}%"></div></div></td>
                    </tr>"""

                st.markdown(f"""
                <div class="det-card">
                    <div class="result-label">Chi tiết phát hiện</div>
                    <table class="det-table">
                        <thead><tr><th>#</th><th>OCR</th><th>Confidence</th></tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card" style="text-align:center; padding:2.5rem 1.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.6rem;">📷</div>
                <div style="color:#6B7280; font-size:0.95rem; line-height:1.6;">
                    Nhấn <strong style="color:#6C63FF;">Nhận diện</strong> để bắt đầu<br>
                    phân tích hạn sử dụng
                </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
