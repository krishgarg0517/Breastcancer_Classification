import streamlit as st
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==============================
# 🌸 Streamlit Page Configuration
# ==============================
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")
st.title("🎗️ Breast Cancer Tumor Detection & Segmentation")
st.caption("Detects tumors from breast ultrasound images and highlights tumor regions")

# ==============================
# 🌈 Apply Local CSS (Optional)
# ==============================
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Style file not found: {file_name}")

local_css("style.css")

# ==============================
# 🧠 Load Trained Model
# ==============================
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) /
                (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

try:
    model = load_model("tumor_unet_optimized.h5",
                       custom_objects={"bce_dice_loss": bce_dice_loss, "dice_loss": dice_loss})
    st.sidebar.success("✅ U-Net Model Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"⚠️ Could not load model: {e}")
    model = None

# ==============================
# 📤 File Upload Section
# ==============================
st.header("📸 Upload Breast Ultrasound Image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_eq = cv2.equalizeHist(img_gray)
    img_eq = cv2.bilateralFilter(img_eq, 7, 75, 75)
    img_resized = cv2.resize(img_eq, (256, 256)) / 255.0

    st.image(img_color[:, :, ::-1], caption="Uploaded Image", use_container_width=True)

    if model is not None:
        # Predict mask
        pred_mask = model.predict(np.expand_dims(np.expand_dims(img_resized, 0), -1))[0, :, :, 0]
        pred_mask = cv2.GaussianBlur(pred_mask, (3, 3), 0)
        mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_img = cv2.cvtColor((img_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        if contours:
            best_cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(best_cnt)

            if area > 50:
                x, y, w, h = cv2.boundingRect(best_cnt)
                overlay = result_img.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), -1)
                result_img = cv2.addWeighted(overlay, 0.3, result_img, 0.7, 0)
                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result_img, "Tumor", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                st.success("🩺 Result: **Breast Cancer Detected**")
                st.image(result_img, caption="Detected Tumor Region", use_container_width=True)
            else:
                st.info("✅ Result: **CLEAR / NULL** (No significant tumor region found)")
        else:
            st.info("✅ Result: **CLEAR / NULL** (No tumor contours detected)")
    else:
        st.error("⚠️ Model not loaded — check your .h5 file path")

# ==============================
# 🔚 Footer
# ==============================
st.markdown("---")
st.caption("Developed by Mudrik | Deep Learning-based Breast Cancer Detection System 🎗️")
                   