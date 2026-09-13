"""Streamlit app — Deepfake Detection with a predicted edit mask and Grad-CAM.

Workflow:
  1. Set model checkpoint in the sidebar (auto-filled from config).
  2. Upload any JPG / PNG / WEBP image.
  3. Click **Analyse** → see class prediction + confidence bar chart.
  4. Explanation panel. For the mask-head models this is the supervised edit
     mask first (trained against ground-truth masks) and a Grad-CAM on the
     encoder's token grid second. Legacy ConvNeXt checkpoints get conv Grad-CAM.
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
from frontend import mask_head as mask_head_module

CLASS_NAMES = ["Real", "AI Generated", "AI Edited"]
CLASS_COLORS = {"Real": "🟢", "AI Generated": "🔴", "AI Edited": "🟠",
                mask_head_module.ABSTAIN: "⚪"}


# ── Cached model loader ────────────────────────────────────────────────────────
# st.cache_resource keeps the model in memory across interactions so it is only
# loaded once per session — avoids reloading on every widget change.
@st.cache_resource(show_spinner="Loading model…")
def load_model_cached(checkpoint_path: str, use_gpu: bool):
    device = inference.get_device(use_gpu)
    if mask_head_module.is_mask_head_checkpoint(checkpoint_path):
        return mask_head_module.MaskHeadModel(checkpoint_path, device=device), device
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
    st.caption("Upload an image → get a classification → see the predicted edit mask and a Grad-CAM of what moved the score.")

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        checkpoint = st.text_input(
            "Model checkpoint",
            value=config.MODEL_CHECKPOINT,
            help="Path to .pth file, relative to the project root"
        )
        use_gpu = st.checkbox("Use GPU if available", value=True)
        rule = mask_head_module.load_decision_rule(checkpoint) if checkpoint else {}
        abstain_below = st.slider(
            "Answer only when top probability is at least", 0.5, 0.99,
            float(rule.get("abstain_below", mask_head_module.DEFAULT_ABSTAIN_BELOW)), 0.01,
            help="Below this the verdict is 'Cannot tell'. At 0.90 the frozen CLIP "
                 "model answers 56% of in-distribution images and is right 94% of "
                 "the time when it does; real photos get a confident wrong answer "
                 "0.5% of the time instead of 6.8%. Lower it to answer more and be "
                 "wrong more. Mask-head checkpoints only.")
        match_encoding = st.checkbox(
            "Re-encode like the training data (512px JPEG q90)", value=True,
            help="Every training image was squashed to 512x512 and saved at JPEG "
                 "q90. Off = feed the upload as-is, which is a distribution shift "
                 "worth seeing but not the condition the model was trained for. "
                 "Mask-head checkpoints only.")

        st.divider()
        st.subheader("Explanation options")
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

    is_mask_head = isinstance(model, mask_head_module.MaskHeadModel)

    # ── Inference ──────────────────────────────────────────────────────────────
    with st.spinner("Running inference…"):
        try:
            if is_mask_head:
                model_input = (model.match_training_encoding(pil_img)
                               if match_encoding else pil_img)
                top_label, probs, edit_mask = model.predict(model_input)
            else:
                model_input, edit_mask = pil_img, None
                top_label, probs = inference.predict(model, pil_img, device=device)
        except Exception as e:
            st.error(f"Inference error: {e}")
            return
    if is_mask_head:
        st.caption(f"Model: {model.describe()}  ·  input "
                   f"{'re-encoded to 512px JPEG q90' if match_encoding else 'as uploaded'}")

    pred_idx = CLASS_NAMES.index(top_label)
    conf = probs[top_label]
    verdict = (mask_head_module.apply_rule(probs, abstain_below) if is_mask_head
               else top_label)

    st.divider()

    # ── Results ────────────────────────────────────────────────────────────────
    st.subheader("Classification result")

    # Large verdict badge
    color = CLASS_COLORS[verdict]
    if verdict == mask_head_module.ABSTAIN:
        st.markdown(
            f"<h2 style='text-align:center'>{color} {verdict}</h2>"
            f"<p style='text-align:center; font-size:1.1rem; color:grey'>"
            f"Leaning <b>{top_label}</b> at {conf*100:.1f}%, below the {abstain_below:.2f} line. "
            f"The model is not sure enough to say, and a confident wrong answer is worse than none.</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<h2 style='text-align:center'>{color} {verdict}</h2>"
            f"<p style='text-align:center; font-size:1.3rem; color:grey'>Confidence: <b>{conf*100:.1f}%</b></p>",
            unsafe_allow_html=True,
        )
    if is_mask_head:
        st.caption("Probabilities are softmax outputs, not calibrated: a 0.99 on a wrong "
                   "answer is not rarer than a 0.6 on one. The threshold was set by "
                   "measuring coverage against accuracy, see scripts/evaluation/abstain_sweep.py.")

    # Per-class probability bars
    st.subheader("Confidence per class")
    bar_cols = st.columns(3)
    for i, name in enumerate(CLASS_NAMES):
        p = probs[name]
        bar_cols[i].metric(name, f"{p*100:.1f}%")
        bar_cols[i].progress(float(p))

    # ── Explanation ────────────────────────────────────────────────────────────
    st.divider()
    target_idx = pred_idx if target_mode == "Predicted class" else CLASS_NAMES.index(manual_class)
    target_name = CLASS_NAMES[target_idx]

    if is_mask_head:
        explain_mask_head(model, model_input, edit_mask, target_idx, target_name, alpha, colormap)
    else:
        explain_legacy(model, device, pil_img, target_idx, target_name, alpha, colormap)


def explain_mask_head(model, img, edit_mask, target_idx, target_name, alpha, colormap):
    """Supervised mask first, Grad-CAM on the token grid second."""
    import numpy as np

    area = float((edit_mask > 0.5).mean())
    st.subheader("Predicted edit mask")
    st.caption(
        f"The decoder's estimate of which pixels were altered, trained against "
        f"OpenSDI's ground-truth masks. **{area:.1%}** of the frame is flagged. "
        "For a fully generated image there is no meaningful 'edited region', and "
        "the model was never asked to produce one; a real photo should be near blank."
    )
    mask_overlay = gradcam_module.overlay_heatmap(img, edit_mask, alpha=alpha, colormap=colormap)
    tab_m, tab_mc, tab_mraw = st.tabs(["🎭 Mask overlay", "📊 Side-by-side", "🗺️ Raw mask"])
    with tab_m:
        st.image(mask_overlay, use_container_width=True)
        st.download_button("⬇️  Download mask overlay", data=pil_to_bytes(mask_overlay),
                           file_name="edit_mask_overlay.png", mime="image/png")
    with tab_mc:
        st.image(gradcam_module.create_gradcam_comparison(img, edit_mask, alpha=alpha),
                 caption="Input  |  Mask probability  |  Overlay", use_container_width=True)
    with tab_mraw:
        st.image(Image.fromarray((edit_mask * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR),
                 caption="Mask probability, white = altered", use_container_width=True)

    st.subheader("Grad-CAM on the encoder's token grid")
    st.caption(
        f"Gradient of the **{target_name}** logit with respect to the final "
        f"{model.grid}x{model.grid} patch tokens, the ViT counterpart of conv-layer "
        "Grad-CAM. Post-hoc and unsupervised, so read it as 'which patches moved the "
        "score', not as a manipulation map; the mask above is the trained answer to that. "
        "Expect some hot patches in flat background (sky, walls): ViTs park high-norm "
        "tokens there as working memory, and Grad-CAM picks them up."
    )
    with st.spinner(f"Computing Grad-CAM for '{target_name}'…"):
        try:
            cam = model.gradcam(img, target_idx)
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")
            return
    cam_overlay = gradcam_module.overlay_heatmap(img, cam, alpha=alpha, colormap=colormap)
    tab_o, tab_c = st.tabs(["🌡️ Overlay", "📊 Side-by-side"])
    with tab_o:
        st.image(cam_overlay, use_container_width=True)
        st.download_button("⬇️  Download Grad-CAM", data=pil_to_bytes(cam_overlay),
                           file_name=f"gradcam_{target_name.replace(' ', '_').lower()}.png",
                           mime="image/png")
    with tab_c:
        st.image(gradcam_module.create_gradcam_comparison(img, cam, alpha=alpha),
                 caption="Input  |  Grad-CAM  |  Overlay", use_container_width=True)

    with st.expander("Compare Grad-CAM across all three classes"):
        cols = st.columns(3)
        for i, name in enumerate(CLASS_NAMES):
            try:
                ov = gradcam_module.overlay_heatmap(img, model.gradcam(img, i), alpha=alpha, colormap=colormap)
                cols[i].image(ov, caption=f"{CLASS_COLORS[name]} {name}", use_container_width=True)
            except Exception as e:
                cols[i].warning(f"{name}: {e}")


def explain_legacy(model, device, pil_img, target_idx, target_name, alpha, colormap):
    """Conv-layer Grad-CAM for the original ConvNeXt checkpoints."""
    st.subheader("Grad-CAM explanation")
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
        raw_pil = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
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

