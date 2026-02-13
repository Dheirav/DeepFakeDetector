"""Streamlit frontend app for multi-class deepfake detection.

This UI is a single-page app that accepts image uploads, runs inference
through the project's PyTorch model (if available), and displays prediction,
confidence, and a Grad-CAM heatmap overlay. The app is modular and imports
inference and gradcam utilities from the frontend package.
"""

from typing import Optional
import io
import os
import sys

import streamlit as st
from PIL import Image

# When Streamlit runs a script (e.g. `streamlit run frontend/app.py`), the
# process's import path may be the `frontend` directory itself rather than the
# project root. Ensure the project root is on `sys.path` so absolute imports
# like `from frontend import ...` work reliably.
if __package__ is None:  # running as a script
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from frontend import config
from frontend import inference
from frontend import gradcam as gradcam_module


def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def validate_image(uploaded) -> (Optional[Image.Image], Optional[str]):
    try:
        img = Image.open(uploaded)
        img.verify()
        uploaded.seek(0)
        img = Image.open(uploaded).convert("RGB")
        if img.width * img.height > 50_000_000:
            return None, "Image too large (width*height > 50M pixels)"
        return img, None
    except Exception as e:
        return None, f"Invalid or corrupted image: {e}"


def main():
    st.set_page_config(page_title="Deepfake Detection", layout="wide")
    st.title("Deepfake Detection — Upload & Explain")

    cols = st.columns([3, 1])
    with cols[1]:
        checkpoint_path = st.text_input("Model checkpoint path", value=config.MODEL_CHECKPOINT)
        use_gpu = st.checkbox("Use GPU if available", value=True)
        analyze = st.button("Analyze")

    uploaded_file = st.file_uploader("Upload image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is None:
        st.info("Upload an image (JPG, PNG, WEBP) to begin.")
        return

    pil_img, err = validate_image(uploaded_file)
    if err:
        st.error(err)
        return

    left, right = st.columns([2, 1])
    with left:
        st.image(pil_img, caption="Uploaded image", use_column_width=True)

    # Try to load model (best-effort). If fails, fallback to dummy predictor.
    model = None
    cam = None
    try:
        device = inference.get_device(use_gpu)
        model = inference.load_model(checkpoint_path, device=device) if checkpoint_path else None
        if model is not None:
            cam = gradcam_module.GradCAM(model)
    except Exception as e:
        st.warning(f"Model/Grad-CAM unavailable, using dummy predictor: {e}")

    if analyze:
        status = st.empty()
        status.info("Processing...")
        try:
            top, probs = inference.predict(model, pil_img, device=device if 'device' in locals() else None)
            conf = probs[top]

            st.subheader("Prediction")
            st.success(f"{top} — {conf * 100:.1f}%")

            st.subheader("Class probabilities")
            # show as bar chart of percentages
            percent = {k: [v * 100] for k, v in probs.items()}
            st.bar_chart(percent)

            st.subheader("Explainability")
            view = st.radio("View", ["Original", "Heatmap overlay", "Side-by-side"], index=1)

            if model is not None and cam is not None:
                try:
                    import numpy as np
                    # compute tensor and heatmap
                    tensor = inference.preprocess_image(pil_img)
                    heatmap = cam(tensor, class_idx=int(list(probs.keys()).index(top)))
                    overlay = gradcam_module.overlay_heatmap(pil_img, heatmap, alpha=0.5)
                except Exception as e:
                    st.warning(f"Grad-CAM generation failed, showing original: {e}")
                    overlay = pil_img
            else:
                # fallback dummy overlay
                overlay = pil_img

            if view == "Original":
                st.image(pil_img, use_column_width=True)
            elif view == "Heatmap overlay":
                st.image(overlay, use_column_width=True)
            else:
                c1, c2 = st.columns(2)
                c1.image(pil_img, caption="Original")
                c2.image(overlay, caption="Heatmap overlay")

            st.download_button("Download Heatmap", data=image_to_bytes(overlay), file_name="heatmap.png", mime="image/png")
            status.success("Prediction complete")

        except Exception as e:
            st.error(f"Inference failed: {e}")
        finally:
            status.empty()


if __name__ == "__main__":
    main()

