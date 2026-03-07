"""Streamlit app — Deepfake Detection + Grad-CAM explanation.

Workflow:
  1. Set model checkpoint in the sidebar (auto-filled from config).
  2. Upload any JPG / PNG / WEBP image.
  3. Click **Analyse** → see class prediction + confidence bar chart.
  4. Explore the Grad-CAM panel: pick target class, colormap, opacity.
  5. Download the overlay if needed.

Run from the project root:
    streamlit run frontend/app.py
"""

import io
import os
import sys

import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

# Ensure project root is on sys.path regardless of how Streamlit launches this.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from frontend import config
from frontend import inference
from frontend import gradcam as gradcam_module

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]
CLASS_COLORS = {"Real": "🟢", "AI Generated": "🔴", "AI Edited": "🟠"}


# ── Cached model loader ────────────────────────────────────────────────────────
# st.cache_resource keeps the model in memory across interactions so it is only
# loaded once per session — avoids reloading on every widget change.
@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(checkpoint_path: str, use_gpu: bool):
    device = inference.get_device(use_gpu)
    model = inference.load_model(checkpoint_path, device=device)
    return model, device


# ── Helpers ────────────────────────────────────────────────────────────────────
def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def validate_upload(uploaded) -> tuple:
    try:
        img = Image.open(uploaded)
        img.verify()
        uploaded.seek(0)
        img = Image.open(uploaded).convert("RGB")
        if img.width * img.height > 50_000_000:
            return None, "Image too large (> 50 MP)"
        return img, None
    except Exception as e:
        return None, f"Invalid image: {e}"


# ── Page layout ────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Deepfake Detector",
        page_icon="🔍",
        layout="wide",
    )
    st.title("🔍 Deepfake Detection")
    st.caption("Upload an image → get a classification → inspect what the model is looking at with Grad-CAM.")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        checkpoint = st.text_input(
            "Model checkpoint",
            value=config.MODEL_CHECKPOINT,
            help="Path to .pth file, relative to the project root"
        )
        use_gpu = st.checkbox("Use GPU if available", value=True)

        st.divider()
        st.subheader("Grad-CAM options")
        target_mode = st.radio(
            "Target class for heatmap",
            ["Predicted class", "Choose manually"],
            index=0,
        )
        manual_class = st.selectbox(
            "Class",
            CLASS_NAMES,
            disabled=(target_mode == "Predicted class"),
        )
        colormap = st.selectbox("Colormap", ["jet", "viridis", "hot", "plasma"], index=0)
        alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.5, 0.05)

        st.divider()
        st.subheader("✂️ Crop")
        enable_crop = st.checkbox("Enable crop before analysis", value=False)
        crop_aspect = st.selectbox(
            "Aspect ratio",
            ["Free", "1:1", "4:3", "16:9", "3:4"],
            disabled=not enable_crop,
        )
        _aspect_map = {"Free": None, "1:1": (1, 1), "4:3": (4, 3), "16:9": (16, 9), "3:4": (3, 4)}

    # ── Upload ─────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload image (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        st.info("⬆️  Upload an image above to get started.")
        return

    pil_img, err = validate_upload(uploaded)
    if err:
        st.error(err)
        return

    # ── Crop (optional) ────────────────────────────────────────────────────────
    if enable_crop:
        st.subheader("✂️ Crop your image")
        st.caption("Drag the handles to select the region you want to analyse, then click **Analyse**.")
        box_color = "#FF4B4B"  # Streamlit red
        aspect_ratio = _aspect_map[crop_aspect]
        # st_cropper returns a cropped PIL Image in real time
        pil_img = st_cropper(
            pil_img,
            realtime_update=True,
            box_color=box_color,
            aspect_ratio=aspect_ratio,
            return_type="image",
        )
        st.caption(f"Crop preview — {pil_img.width}×{pil_img.height}px")
    else:
        col_img, _ = st.columns([4, 1])
        with col_img:
            st.image(pil_img, caption=f"Uploaded — {pil_img.width}×{pil_img.height}px", use_container_width=True)

    col_btn_row = st.columns([4, 1])
    with col_btn_row[1]:
        run = st.button("🔎 Analyse", type="primary", use_container_width=True)

    if not run:
        return

    # ── Load model (cached) ────────────────────────────────────────────────────
    if not checkpoint or not os.path.isfile(checkpoint):
        st.error(f"Checkpoint not found: `{checkpoint}`")
        return

    try:
        model, device = load_model_cached(checkpoint, use_gpu)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # ── Inference ──────────────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        try:
            top_label, probs = inference.predict(model, pil_img, device=device)
        except Exception as e:
            st.error(f"Inference error: {e}")
            return

    pred_idx = CLASS_NAMES.index(top_label)
    conf = probs[top_label]

    st.divider()

    # ── Results ────────────────────────────────────────────────────────────────
    st.subheader("Classification result")

    # Large verdict badge
    color = CLASS_COLORS[top_label]
    st.markdown(
        f"<h2 style='text-align:center'>{color} {top_label}</h2>"
        f"<p style='text-align:center; font-size:1.3rem; color:grey'>Confidence: <b>{conf*100:.1f}%</b></p>",
        unsafe_allow_html=True,
    )

    # Per-class probability bars
    st.subheader("Confidence per class")
    bar_cols = st.columns(3)
    for i, name in enumerate(CLASS_NAMES):
        p = probs[name]
        bar_cols[i].metric(name, f"{p*100:.1f}%")
        bar_cols[i].progress(float(p))

    # ── Grad-CAM ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Grad-CAM explanation")

    target_idx = pred_idx if target_mode == "Predicted class" else CLASS_NAMES.index(manual_class)
    target_name = CLASS_NAMES[target_idx]
    st.caption(
        f"Heatmap computed for class **{target_name}** — "
        "red/hot areas are the pixels that pushed the model toward that prediction."
    )

    with st.spinner(f"Generating Grad-CAM for '{target_name}'…"):
        try:
            cam = gradcam_module.GradCAM(model, verbose=False)
            tensor = inference.preprocess_image(pil_img).to(device)  # [1,C,H,W]
            heatmap = cam(tensor, class_idx=target_idx)
            cam.cleanup()

            overlay    = gradcam_module.overlay_heatmap(pil_img, heatmap, alpha=alpha, colormap=colormap)
            comparison = gradcam_module.create_gradcam_comparison(pil_img, heatmap, alpha=alpha)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")
            return

    tab_overlay, tab_compare, tab_raw = st.tabs(
        ["🌡️ Overlay", "📊 Side-by-side comparison", "🗺️ Raw heatmap"]
    )

    with tab_overlay:
        st.image(overlay, use_container_width=True)
        st.download_button(
            "⬇️  Download overlay",
            data=pil_to_bytes(overlay),
            file_name=f"gradcam_{target_name.replace(' ', '_').lower()}.png",
            mime="image/png",
        )

    with tab_compare:
        st.image(comparison, caption="Original  |  Raw heatmap  |  Overlay", use_container_width=True)
        st.download_button(
            "⬇️  Download comparison",
            data=pil_to_bytes(comparison),
            file_name=f"gradcam_comparison_{target_name.replace(' ', '_').lower()}.png",
            mime="image/png",
        )

    with tab_raw:
        import numpy as np
        from PIL import Image as PilImage
        raw_pil = PilImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
            pil_img.size, resample=Image.BILINEAR
        )
        st.image(raw_pil, caption="Raw activation map (grayscale)", use_container_width=True)

    # All-class comparison (expandable)
    with st.expander("Compare Grad-CAM across all three classes"):
        with st.spinner("Generating heatmaps for all classes…"):
            try:
                all_cols = st.columns(3)
                cam_all = gradcam_module.GradCAM(model, verbose=False)
                for i, name in enumerate(CLASS_NAMES):
                    h = cam_all(inference.preprocess_image(pil_img).to(device), class_idx=i)
                    ov = gradcam_module.overlay_heatmap(pil_img, h, alpha=alpha, colormap=colormap)
                    all_cols[i].image(ov, caption=f"{CLASS_COLORS[name]} {name}", use_container_width=True)
                cam_all.cleanup()
            except Exception as e:
                st.warning(f"All-class Grad-CAM failed: {e}")


if __name__ == "__main__":
    main()

