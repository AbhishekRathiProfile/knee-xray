import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import io

# ========== Page & Style Config ==========
st.set_page_config(page_title="Knee X-ray Classifier", layout="centered")

# White background, black text
st.markdown(
    """
    <style>
        body, .stApp {
            background-color: black !important;
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6, p, span, div {
            color: white !important;
        }
        header, footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ========== Model Load ==========
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mymodel.h5")

model = load_model()

# ========== Classes ==========
class_names = ["Healthy Knee", "Moderate Knee", "Severe Knee"]

# ========== Grad-CAM Function ==========
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(original_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlayed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed_img

# ========== Title & Upload ==========
st.markdown("#### Knee X-ray Classification (EfficientNet)")
st.write("Upload a 224x224 knee X-ray image to classify severity and view Grad-CAM visualization.")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.write("#### Prediction Results")

    # Show original image and grad-cam side-by-side
    col1, col2 = st.columns(2)

    # Preprocess
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized) / 255.0
    image_input = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(image_input)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    prediction_label = class_names[class_index]

    # Grad-CAM
    heatmap = make_gradcam_heatmap(image_input, model)
    overlay_img = overlay_gradcam(np.array(image_resized), heatmap)
    overlay_pil = Image.fromarray(overlay_img)

    with col1:
        st.image(image_resized, caption="Uploaded X-ray", width=224)

    with col2:
        st.image(overlay_pil, caption="Grad-CAM Visualization", width=224)

    st.markdown(f"**Prediction:** `{prediction_label}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")
