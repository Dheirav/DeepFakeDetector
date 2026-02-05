import streamlit as st
from PIL import Image


# -----------------------------
# Page configuration and layout
# -----------------------------
st.set_page_config(
    page_title="Deepfake Image Detection",
    layout="wide",
)


def generate_dummy_gradcam(image: Image.Image) -> Image.Image:
    """
    Create a simple dummy Grad-CAM style heatmap overlay.

    NOTE: This does NOT use any real model.
    It just overlays a semi-transparent red layer on top of the image
    to mimic a Grad-CAM visualization.
    """
    # Ensure image has an alpha channel
    image_rgba = image.convert("RGBA")

    # Create a red semi-transparent overlay
    heatmap = Image.new("RGBA", image_rgba.size, (255, 0, 0, 120))

    # Blend original image with the red overlay
    gradcam = Image.alpha_composite(image_rgba, heatmap)
    return gradcam


def main() -> None:
    # -----------------------------
    # Page title and description
    # -----------------------------
    st.title("Deepfake Image Detection (Demo UI)")
    st.markdown(
        """
        This Streamlit app demonstrates a **front-end UI** for a deepfake detection project.
        
        - No real model or backend is used here.
        - Predictions and Grad-CAM are **dummy placeholders** for UI prototyping.
        """
    )

    st.markdown("---")

    # -----------------------------
    # File upload section
    # -----------------------------
    st.subheader("1. Upload Input Image")
    uploaded_file = st.file_uploader(
        "Upload a face image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    # Analyze button (disabled until an image is uploaded)
    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        disabled=uploaded_file is None,
    )

    # -----------------------------
    # Main content area
    # -----------------------------
    if uploaded_file is not None:
        # Load the uploaded image using PIL
        input_image = Image.open(uploaded_file)

        # Create two columns: left for input image, right for Grad-CAM
        col1, col2 = st.columns(2)

        with col1:
            # -----------------------------
            # Display original image
            # -----------------------------
            st.subheader("2. Input Image")
            st.image(
                input_image,
                caption="Uploaded image",
                use_container_width=True,
            )

        with col2:
            # -----------------------------
            # Display dummy Grad-CAM
            # -----------------------------
            st.subheader("3. Grad-CAM Visualization (Dummy)")
            gradcam_image = generate_dummy_gradcam(input_image)
            st.image(
                gradcam_image,
                caption="Dummy Grad-CAM style heatmap overlay",
                use_container_width=True,
            )

        st.markdown("---")

        # -----------------------------
        # Prediction results section
        # -----------------------------
        st.subheader("4. Prediction Result")

        if analyze_clicked:
            # Dummy prediction label and confidence
            prediction_label = "Deepfake (Fake)"
            confidence = 0.87  # Dummy confidence score

            # Display prediction in a highlighted box
            st.success(f"Prediction: **{prediction_label}**")
            st.write(f"Confidence: **{confidence * 100:.1f}%**")

            # Academic-style note clarifying that this is a mock result
            st.caption(
                "Note: This is a mock prediction for UI demonstration only. "
                "No real deepfake detection model is running."
            )
        else:
            st.info("Click **Analyze** to see dummy prediction and confidence.")

    else:
        # If no file is uploaded, guide the user
        st.info("Please upload an image to begin the deepfake analysis demo.")


if __name__ == "__main__":
    # -----------------------------
    # Entry point
    # -----------------------------
    main()

