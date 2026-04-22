import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
from ultralytics import YOLO

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Detection – Drone-X",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark gradient background */
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .metric-card h2 { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-card p  { font-size: 0.85rem; color: #aaa; margin: 0; }

    /* Upload area */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04);
        border: 2px dashed rgba(130,100,255,0.5);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Result image container */
    .result-img {
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        overflow: hidden;
    }

    /* Confidence badge */
    .conf-badge {
        display: inline-block;
        background: linear-gradient(90deg, #7f5af0, #2cb67d);
        color: white;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    h1, h2, h3 { color: #e2e8f0 !important; }
    p, label, .stMarkdown { color: #cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO model…")
def load_model(path: str = "best.pt") -> YOLO:
    return YOLO(path)


# ── Inference helper ──────────────────────────────────────────────────────────
BOX_COLOR  = (127, 90, 240)   # purple
TEXT_COLOR = (255, 255, 255)
FONT       = cv2.FONT_HERSHEY_SIMPLEX


def run_inference(model: YOLO, image: Image.Image, conf_thresh: float, iou_thresh: float):
    """Run YOLO on a PIL image. Returns annotated image (BGR numpy) + result list."""
    img_np = np.array(image.convert("RGB"))

    results = model.predict(
        source=img_np,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
    )

    result   = results[0]
    img_draw = img_np.copy()
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names.get(cls, str(cls))

            detections.append({"label": label, "conf": conf, "bbox": (x1, y1, x2, y2)})

            # Draw bounding box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), BOX_COLOR, 2)

            # Badge background
            tag   = f"{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, FONT, 0.55, 1)
            cv2.rectangle(img_draw, (x1, y1 - th - 10), (x1 + tw + 8, y1), BOX_COLOR, -1)
            cv2.putText(img_draw, tag, (x1 + 4, y1 - 4), FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

    return img_draw, detections


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Detection Parameters")
    st.markdown("---")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.01, max_value=1.0,
        value=0.25, step=0.01,
        help="Only detections above this confidence score are shown.",
    )

    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.01, max_value=1.0,
        value=0.45, step=0.01,
        help="Non-Maximum Suppression overlap threshold.",
    )

    st.markdown("---")
    st.markdown(
        """
        **How to use**
        1. Upload an image on the right  
        2. Adjust thresholds above  
        3. Detections appear instantly  

        **Model:** YOLOv8 custom — `best.pt`
        """
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("# 🚗 Car Movement Detection")
st.markdown("Upload an image and the model will locate every vehicle with bounding boxes and confidence scores.")

uploaded_file = st.file_uploader(
    "Drop an image here or click to browse",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    model = load_model()

    with st.spinner("Running inference…"):
        annotated, detections = run_inference(model, image, conf_threshold, iou_threshold)

    # ── Compute average confidence ───────────────────────────────────────────
    avg_conf = np.mean([d["conf"] for d in detections]) if detections else 0.0

    # Update page title dynamically via JS trick
    st.markdown(
        f"<script>document.title = 'Avg Conf: {avg_conf:.0%} | Car Detection';</script>",
        unsafe_allow_html=True,
    )

    # ── Metrics row ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                  <p>Vehicles Detected</p>
                  <h2>🚗 {len(detections)}</h2>
                </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div class="metric-card">
                  <p>Average Confidence</p>
                  <h2>🎯 {avg_conf:.1%}</h2>
                </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        max_conf = max((d["conf"] for d in detections), default=0.0)
        st.markdown(
            f"""<div class="metric-card">
                  <p>Peak Confidence</p>
                  <h2>⚡ {max_conf:.1%}</h2>
                </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side-by-side images ───────────────────────────────────────────────────
    col_orig, col_det = st.columns(2)

    with col_orig:
        st.markdown("**Original Image**")
        st.image(image, use_container_width=True)

    with col_det:
        st.markdown(
            f"**Detections** &nbsp; "
            f"<span class='conf-badge'>Avg {avg_conf:.1%}</span>",
            unsafe_allow_html=True,
        )
        # Convert BGR → RGB for Streamlit
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

    # ── Detection table ───────────────────────────────────────────────────────
    if detections:
        st.markdown("### 📋 Detection Details")
        table_data = [
            {
                "#": i + 1,
                "Label": d["label"],
                "Confidence": f"{d['conf']:.2%}",
                "BBox (x1,y1,x2,y2)": str(d["bbox"]),
            }
            for i, d in enumerate(detections)
        ]
        st.dataframe(table_data, use_container_width=True, hide_index=True)

    # ── Download button ───────────────────────────────────────────────────────
    buf = io.BytesIO()
    Image.fromarray(annotated_rgb).save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download Annotated Image",
        data=buf.getvalue(),
        file_name="detection_result.png",
        mime="image/png",
    )

else:
    # Placeholder state
    st.markdown(
        """
        <div style="
            border: 2px dashed rgba(127,90,240,0.4);
            border-radius: 16px;
            padding: 4rem 2rem;
            text-align: center;
            color: #888;
            margin-top: 2rem;
        ">
            <div style="font-size:4rem;">🚗</div>
            <h3 style="color:#aaa !important;">No image uploaded yet</h3>
            <p>Use the uploader above to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
